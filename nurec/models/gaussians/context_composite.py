# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite aliasing primitives.

Two render contexts must share one read-only copy of the gaussian weights without N× VRAM:

- ``make_context_composite(shared)`` returns a ``GaussiansComposite`` **shell** whose parameters and
  buffers **alias** ``shared``'s tensor storage (zero-copy), but whose module tree has its **own**
  ``_parameters``/``_buffers``/``_modules`` dicts so that a structural edit on one context (which
  *rebinds* an ``nn.Parameter``) does not touch any other context or the shared model (copy-on-write).
- ``build_composite_skeleton(reference)`` returns the same structural shell but with **storage-less
  meta** tensors — same ``state_dict()`` keys/shapes, no real allocation, no datasource/initializer
  run. Used to reason about / reconstruct structure without owning weights.

Why a structural clone of a *reference* (rather than a config-only constructor): the composite's node
set and per-node tensor shapes are determined by the **datasource** at ``__init__`` time
(`gaussians_composite.py` per-layer `BaseGaussianModel.factory`), not by config alone — so the
cheapest correct way to get the exact structure is to mirror an already-built composite (in-process)
or, in the multi-process worker, the owner's exported tensors + manifest.

Hard invariant (Read-only gate): never mutate a shared tensor **in place**. Rebinding
(`self.x = nn.Parameter(...)`) is isolated per shell; `self.x.copy_(...)` / `self.x.mul_(...)` on an
aliased tensor would corrupt every context.
"""

from __future__ import annotations

import weakref

from collections import OrderedDict
from collections.abc import Mapping
from typing import Callable, Optional

import torch

from torch import nn

from nurec.models.gaussians.fused_parameter_gather import _PayloadIntegerRangeValidationState
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import RigidGaussianModel


# (tensor, is_parameter) -> replacement tensor used in the cloned module tree.
TensorMap = Callable[[torch.Tensor, bool], torch.Tensor]

_MODULE_HOOK_REGISTRIES = (
    "_backward_hooks",
    "_backward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_always_called",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_load_state_dict_post_hooks",
    "_load_state_dict_pre_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
)


def _alias_tensor(tensor: torch.Tensor, _is_parameter: bool) -> torch.Tensor:
    """Reuse the same tensor object: the clone aliases ``shared``'s storage (zero-copy)."""
    return tensor


def _meta_tensor(tensor: torch.Tensor, is_parameter: bool) -> torch.Tensor:
    """A storage-less ``meta`` tensor with the same shape/dtype (for the skeleton)."""
    meta = torch.empty_like(tensor, device="meta")
    if is_parameter:
        return nn.Parameter(meta, requires_grad=tensor.requires_grad)
    return meta


def _clone_module_tree(
    module: nn.Module,
    tensor_map: TensorMap,
    tensor_memo: dict[int, torch.Tensor],
    module_memo: dict[int, nn.Module],
) -> nn.Module:
    """Recursively clone an ``nn.Module`` tree, giving each module its **own**
    ``_parameters``/``_buffers``/``_modules`` dicts while routing every leaf tensor through
    ``tensor_map`` (deduplicated by object id so a tensor shared across nodes stays shared).

    Both a **tensor** memo and a **module** memo are threaded through the recursion: a leaf tensor
    shared across nodes maps to one cloned tensor (``tensor_memo``), and a submodule that appears under
    several names — e.g. the composite binds one renderer under ``_camera_renderer`` /
    ``gaussians_renderer`` / ``gaussians_renderer_camera`` — is cloned **once** and the same clone is
    reused (``module_memo``). Without the module memo such a shared submodule would fan out into N
    independent clones, silently breaking the cross-node identity the source tree relied on.

    Non-module, non-tensor attributes (config, renderer handles, flags, bookkeeping) are shared **by
    reference** via a shallow ``__dict__`` copy; per-context mutable state is reset afterwards by the
    callers below.
    """
    cached = module_memo.get(id(module))
    if cached is not None:
        return cached

    clone = module.__class__.__new__(module.__class__)
    # Register the clone before recursing so a submodule reached again (under another name) reuses this
    # same clone instead of being cloned a second time.
    module_memo[id(module)] = clone
    # Shallow-share every attribute (incl. plain renderer/config/bookkeeping refs); the three module
    # dicts below are then replaced with fresh ones so rebinds stay local to this shell.
    clone.__dict__.update(module.__dict__)
    for name in _MODULE_HOOK_REGISTRIES:
        registry = getattr(module, name, None)
        if registry is not None:
            object.__setattr__(clone, name, registry.copy())

    new_parameters: OrderedDict[str, Optional[nn.Parameter]] = OrderedDict()
    for name, param in module._parameters.items():
        if param is None:
            new_parameters[name] = None
            continue
        if id(param) not in tensor_memo:
            tensor_memo[id(param)] = tensor_map(param, True)
        new_parameters[name] = tensor_memo[id(param)]  # type: ignore[assignment]

    new_buffers: OrderedDict[str, Optional[torch.Tensor]] = OrderedDict()
    for name, buffer in module._buffers.items():
        if buffer is None:
            new_buffers[name] = None
            continue
        if id(buffer) not in tensor_memo:
            tensor_memo[id(buffer)] = tensor_map(buffer, False)
        new_buffers[name] = tensor_memo[id(buffer)]

    new_modules: OrderedDict[str, Optional[nn.Module]] = OrderedDict()
    for name, child in module._modules.items():
        new_modules[name] = None if child is None else _clone_module_tree(child, tensor_map, tensor_memo, module_memo)

    object.__setattr__(clone, "_parameters", new_parameters)
    object.__setattr__(clone, "_buffers", new_buffers)
    object.__setattr__(clone, "_modules", new_modules)
    return clone


def _reset_per_context_state(composite: GaussiansComposite) -> None:
    """Reset the composite's per-request mutable attributes on a freshly cloned shell.

    These are plain attributes shared by reference through the shallow ``__dict__`` copy; each context
    must own its own so concurrent/independent renders don't collide. Renderer handles and structural
    bookkeeping (`cuboid_tracks`/`obj_track_ids`) are intentionally left shared: the gsplat renderer is
    stateless w.r.t. weights at inference, and structural state is read-only on the render path (it is
    copied-on-write only by edits).

    Each gaussian node also carries its own copy-on-write override snapshots
    (``overriden_parameters``/``overriden_buffers``/``overriden_state``) and may carry mutable render
    caches. The shallow ``__dict__`` copy shares those objects across every shell and the shared model.
    Give each cloned node fresh state so sibling contexts cannot invalidate or serialize one another —
    the isolation is structural, not merely policy.

    Invariant (documented, not enforced here): a shell must never run gaussian *initialization* or
    *edit* mutators. In particular ``maybe_initialize_gaussians`` / ``_maybe_initialize_gaussians`` is a
    closure that captured the *original* node's ``self`` at ``__init__`` and is copied by reference onto
    the shell; invoking it on a shell would mutate the shared model, not the shell. Shells are render-only
    (built via ``__new__`` without ``__init__``); initialization happens once on the shared model before
    cloning.
    """
    composite._lidar_stream = None
    composite._visibility_local_ranges = None
    # A shell shares its parameters with the model it was cloned from, so it must not run
    # a mutator. _ensure_tracked_rows_grouped is the one a render can reach; it asserts
    # on this rather than permuting rows out from under the shared model.
    composite._is_context_shell = True
    composite.clear_collect_caches()
    # Not covered by clear_collect_caches: the staging buffer holds no scene state, but
    # two contexts uploading into one device buffer would overwrite each other.
    composite._fused_payload_integer_range_validation_state = _PayloadIntegerRangeValidationState()
    # Generation plans and refresh tokens are training-only runtime state.
    # The shallow clone must not alias either object: in particular, a pending
    # refresh owns one-shot NCCL work that remains owned by the source model.
    # Do not drain it here; cloning is not a lifecycle boundary for the source.
    composite._fused_gather_generation_plan = None
    composite._pending_fused_gather_generation_plan_refresh = None
    composite.overriden_state = {}
    for node in composite.gaussians_nodes.values():
        node.overriden_parameters = {}
        node.overriden_buffers = {}
        node.overriden_state = {}
        if hasattr(node, "_frame_slot_cache"):
            setattr(node, "_frame_slot_cache", None)


def _remap_motion_sources(module_memo: dict[int, nn.Module]) -> None:
    """Redirect copied motion references after every sibling has been cloned."""
    for module in module_memo.values():
        if not isinstance(module, RigidGaussianModel) or module._motion_source_ref is None:
            continue
        source = module.get_motion_source()
        cloned_source = module_memo.get(id(source))
        if not isinstance(cloned_source, RigidGaussianModel):
            raise RuntimeError("Cannot clone a rigid layer whose motion owner is outside the module tree")
        # Preserve the existing relationship: owners can have appended tracks
        # since binding, while dependent appearance tables retain their old rows.
        module._motion_source_ref = weakref.ref(cloned_source)


def make_context_composite(shared: GaussiansComposite) -> GaussiansComposite:
    """A ``GaussiansComposite`` shell whose params/buffers alias ``shared``'s storage (zero-copy).

    The shell has its own module-dict structure (so structural edits copy-on-write per context) and
    fresh per-request state, but adds **no** weight VRAM beyond ``shared``.
    """
    module_memo: dict[int, nn.Module] = {}
    context = _clone_module_tree(shared, _alias_tensor, {}, module_memo)
    assert isinstance(context, GaussiansComposite)
    _remap_motion_sources(module_memo)
    # Context shells are render-only and alias live training Parameters. Do not
    # inherit state-load hooks that reinstall gradient hooks or mutate
    # requires_grad on those aliased objects. Skeleton clones retain their
    # independently copied registries because they are explicitly loadable.
    for module in context.modules():
        module._load_state_dict_pre_hooks = OrderedDict()
        module._load_state_dict_post_hooks = OrderedDict()
    _reset_per_context_state(context)
    return context


def _set_qualified_tensor(root: nn.Module, qualified_name: str, value: torch.Tensor, is_parameter: bool) -> None:
    *parent_path, leaf = qualified_name.split(".")
    submodule = root.get_submodule(".".join(parent_path)) if parent_path else root
    if is_parameter:
        existing = submodule._parameters[leaf]
        requires_grad = existing.requires_grad if existing is not None else False
        submodule._parameters[leaf] = nn.Parameter(value, requires_grad=requires_grad)
    else:
        submodule._buffers[leaf] = value


def _assert_aliasable(name: str, local: torch.Tensor, external: torch.Tensor) -> None:
    """Fail loudly when the external (owner IPC) tensor does not match the local parameter/buffer it
    replaces. Owner and worker load the same artifact, so they must agree; but this is exactly an
    "assert shape and dtype at the public boundary" case — a silent structural divergence (e.g. a buffer
    materialized differently on the worker) would otherwise alias a mismatched tensor and render garbage
    instead of erroring."""
    if tuple(external.shape) != tuple(local.shape):
        raise ValueError(
            f"alias_external_weights: shape mismatch for {name!r}: local {tuple(local.shape)} vs "
            f"external {tuple(external.shape)}"
        )
    if external.dtype != local.dtype:
        raise ValueError(
            f"alias_external_weights: dtype mismatch for {name!r}: local {local.dtype} vs external {external.dtype}"
        )


def alias_external_weights(module: nn.Module, named_tensors: Mapping[str, torch.Tensor]) -> None:
    """In place, rebind every parameter/buffer of ``module`` to the externally-provided tensor of the
    same fully-qualified name (zero-copy alias; the previous local storage is dropped and freed).

    Used by the multi-process **worker**: it loads the composite structure locally, then
    points its weights at the **owner's IPC-mapped** tensors so it renders from shared memory instead
    of its own copy. ``named_tensors`` keys are the composite's ``state_dict()`` names (the same names
    the owner's `model_owner` manifest exports). Each external tensor's shape/dtype is asserted against the
    local param/buffer it replaces before rebinding, so a structural divergence fails loudly. Like the
    clone primitives, this rebinds (does not `copy_` in place), so it never mutates the shared storage —
    the Read-only invariant holds.

    Both walks use ``remove_duplicate=False`` so a tensor registered under several qualified names is
    rebound under **every** one of them. The default (``True``) would rebind only the first, leaving the
    remaining names on their original (local/meta) storage — silent VRAM retention and a broken alias,
    exactly the fail-loud contract this module exists to uphold. Because the owner's manifest is
    ``remove_duplicate``-deduplicated (one canonical name per shared tensor), a duplicate name may be
    absent from ``named_tensors``; when its local tensor was already aliased under the canonical name we
    rebind it to that *same* external tensor (idempotent), rather than failing. Finally, every external
    key must be consumed by some local param/buffer: a surplus key (owner has a tensor the worker's
    structure lacks) raises, closing the other direction of the fail-loud check.
    """
    consumed: set[str] = set()
    external_by_local_id: dict[int, torch.Tensor] = {}

    def _rebind(name: str, local: torch.Tensor, is_parameter: bool) -> None:
        external = named_tensors.get(name)
        if external is not None:
            _assert_aliasable(name, local, external)
            _set_qualified_tensor(module, name, external, is_parameter=is_parameter)
            consumed.add(name)
            external_by_local_id[id(local)] = external
            return
        # No external tensor under this exact name. If this local tensor object was already aliased under
        # another (owner-canonical) name, rebind to that same external tensor: one shared tensor under N
        # names is idempotent. Otherwise it is a genuine missing external -> fail loud.
        already = external_by_local_id.get(id(local))
        if already is None:
            kind = "parameter" if is_parameter else "buffer"
            raise KeyError(f"alias_external_weights: missing external tensor for {kind} {name!r}")
        _assert_aliasable(name, local, already)
        _set_qualified_tensor(module, name, already, is_parameter=is_parameter)

    for name, param in list(module.named_parameters(recurse=True, remove_duplicate=False)):
        _rebind(name, param, is_parameter=True)
    for name, buffer in list(module.named_buffers(recurse=True, remove_duplicate=False)):
        _rebind(name, buffer, is_parameter=False)

    surplus = set(named_tensors) - consumed
    if surplus:
        raise KeyError(
            f"alias_external_weights: {len(surplus)} external tensor(s) had no matching local "
            f"parameter/buffer: {sorted(surplus)}"
        )


def build_composite_skeleton(reference: GaussiansComposite) -> GaussiansComposite:
    """A structural shell of ``reference`` with storage-less ``meta`` params/buffers.

    Same ``state_dict()`` keys/shapes as ``reference``; allocates no real weight storage and runs no
    datasource/initializer. Populate it with real tensors via ``load_state_dict(..., assign=True)``.
    """
    module_memo: dict[int, nn.Module] = {}
    skeleton = _clone_module_tree(reference, _meta_tensor, {}, module_memo)
    assert isinstance(skeleton, GaussiansComposite)
    _remap_motion_sources(module_memo)
    _reset_per_context_state(skeleton)
    return skeleton
