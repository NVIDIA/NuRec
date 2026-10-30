# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the composite aliasing primitives (CPU; no datasource/artifact)."""

import unittest

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import torch

from torch import nn

from nurec.models.gaussians.context_composite import (
    _alias_tensor,
    _clone_module_tree,
    alias_external_weights,
    build_composite_skeleton,
    make_context_composite,
)
from nurec.models.gaussians.fused_parameter_gather import _PayloadIntegerRangeValidationState
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import BaseGaussianModel, RigidGaussianModel
from nurec.models.nn_extensions import TypedModuleDict


class _TinyNode(nn.Module):
    """Stand-in for a gaussian node: a couple of weight Parameters + a buffer."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.positions = nn.Parameter(torch.randn(n, 3))
        self.features = nn.Parameter(torch.randn(n, 8))
        self.register_buffer("activation_scale", torch.ones(n))
        # Mirror BaseGaussianModel's per-node copy-on-write override snapshots so context isolation of
        # these dicts can be exercised (see TestMakeContextComposite.test_node_override_dicts_are_isolated).
        self.overriden_parameters: dict[str, nn.Parameter] = {}
        self.overriden_buffers: dict[str, torch.Tensor] = {}
        self.overriden_state: dict[str, object] = {}


def _make_composite() -> GaussiansComposite:
    """A minimal `GaussiansComposite` with real weight tensors but no datasource/renderers."""
    composite = GaussiansComposite.__new__(GaussiansComposite)
    nn.Module.__init__(composite)
    nodes = {
        "background": cast(BaseGaussianModel, _TinyNode(64)),
        "actor": cast(BaseGaussianModel, _TinyNode(8)),
    }
    composite.gaussians_nodes = TypedModuleDict(nodes)
    # `obj_track_ids` is read by `get_extra_state()` during `state_dict()`; the real composite always
    # sets it in `__init__`, so the fixture mirrors that with an empty mapping.
    composite.obj_track_ids = {}
    # Per-request mutable state that must be reset per context.
    composite._lidar_stream = None
    composite._visibility_local_ranges = None
    composite._segment_tables = {"node": object()}  # type: ignore[dict-item]
    composite._static_collect_cache = object()  # type: ignore[assignment]
    composite._static_collect_cache_stream = object()  # type: ignore[assignment]
    composite._static_collect_cache_layout = (64, 8)
    composite._fourier_fold = {"sentinel": 1}
    composite._edit_overrides = object()  # type: ignore[assignment]
    composite._fused_payload_integer_range_validation_state = _PayloadIntegerRangeValidationState()
    composite._fused_gather_generation_plan = None
    composite._pending_fused_gather_generation_plan_refresh = None
    composite.overriden_state = {"sentinel": 1}
    return composite


def _make_motion_composite(*, appended: bool, dependent_first: bool) -> GaussiansComposite:
    """Build a bound rigid pair, optionally after an owner-only track insertion."""
    nodes: dict[str, BaseGaussianModel] = {}
    for name in ("body", "lights"):
        node = RigidGaussianModel.__new__(RigidGaussianModel)
        nn.Module.__init__(node)
        node.config = cast(Any, SimpleNamespace(motion_source_layer=None if name == "body" else "body"))
        node._motion_source_ref = None
        node.is_static = False
        node.nearest_neighbor_track_for_lidar = False
        node.cuboid_tracks = cast(Any, SimpleNamespace(tracks_id=["a", "b"]))
        node.tracks_calib = cast(Any, nn.Linear(1, 1))
        node.tracks_timestamp_estimation_ray_stride = 4 if name == "body" else 1
        nodes[name] = node
    owner = cast(RigidGaussianModel, nodes["body"])
    dependent = cast(RigidGaussianModel, nodes["lights"])
    dependent.bind_motion_source(owner)
    if appended:
        # Insertion extends the motion owner's table while appearance rows stay local.
        owner.cuboid_tracks = cast(Any, SimpleNamespace(tracks_id=["a", "b", "inserted"]))
    composite = _make_composite()
    composite.gaussians_nodes = TypedModuleDict(dict(reversed(nodes.items())) if dependent_first else nodes)
    return composite


class TestClonedMotionOwnership(unittest.TestCase):
    def test_motion_references_stay_with_cloned_owners(self) -> None:
        for clone_factory in (make_context_composite, build_composite_skeleton):
            for appended in (False, True):
                for dependent_first in (False, True):
                    with self.subTest(
                        factory=clone_factory.__name__, appended=appended, dependent_first=dependent_first
                    ):
                        shared = _make_motion_composite(appended=appended, dependent_first=dependent_first)
                        original_owner = cast(RigidGaussianModel, shared.gaussians_nodes["body"])
                        original_dependent = cast(RigidGaussianModel, shared.gaussians_nodes["lights"])

                        context = clone_factory(shared)
                        owner = cast(RigidGaussianModel, context.gaussians_nodes["body"])
                        dependent = cast(RigidGaussianModel, context.gaussians_nodes["lights"])
                        self.assertIs(owner.get_motion_source(), owner)
                        self.assertIs(dependent.get_motion_source(), owner)
                        self.assertIsNot(owner, original_owner)
                        self.assertIs(original_dependent.get_motion_source(), original_owner)
                        self.assertEqual(dependent.cuboid_tracks.tracks_id, ["a", "b"])
                        self.assertEqual(
                            owner.cuboid_tracks.tracks_id, ["a", "b", "inserted"] if appended else ["a", "b"]
                        )
                        self.assertIs(dependent.get_motion_tracks_calib(), owner.tracks_calib)
                        self.assertIsNot(owner.tracks_calib, original_owner.tracks_calib)

                        context.eval()
                        owner.tracks_timestamp_estimation_ray_stride = 16
                        self.assertFalse(dependent.get_motion_source().training)
                        self.assertEqual(dependent.get_motion_source().tracks_timestamp_estimation_ray_stride, 16)
                        self.assertTrue(original_owner.training)
                        self.assertEqual(original_owner.tracks_timestamp_estimation_ray_stride, 4)


class TestMakeContextComposite(unittest.TestCase):
    def test_params_and_buffers_alias_shared_storage(self) -> None:
        shared = _make_composite()
        context = make_context_composite(shared)

        shared_params = dict(shared.named_parameters())
        context_params = dict(context.named_parameters())
        self.assertEqual(shared_params.keys(), context_params.keys())
        for name, shared_param in shared_params.items():
            self.assertEqual(
                shared_param.data_ptr(),
                context_params[name].data_ptr(),
                msg=f"param {name} is not aliased (data_ptr differs)",
            )

        shared_buffers = dict(shared.named_buffers())
        context_buffers = dict(context.named_buffers())
        self.assertEqual(shared_buffers.keys(), context_buffers.keys())
        for name, shared_buffer in shared_buffers.items():
            self.assertEqual(shared_buffer.data_ptr(), context_buffers[name].data_ptr())

    def test_module_dicts_are_independent(self) -> None:
        shared = _make_composite()
        context = make_context_composite(shared)
        # Distinct dict objects at every level so a rebind on one shell stays local.
        self.assertIsNot(context._modules, shared._modules)
        self.assertIsNot(context.gaussians_nodes._modules, shared.gaussians_nodes._modules)
        self.assertIsNot(
            context.gaussians_nodes["actor"]._parameters,
            shared.gaussians_nodes["actor"]._parameters,
        )

    def test_render_shell_does_not_inherit_state_load_hooks(self) -> None:
        shared = _make_composite()

        def freeze_after_load(module: nn.Module, incompatible_keys: object) -> None:
            del incompatible_keys
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        shared.register_load_state_dict_post_hook(freeze_after_load)
        context = make_context_composite(shared)

        self.assertIsNot(context._load_state_dict_post_hooks, shared._load_state_dict_post_hooks)
        self.assertEqual(len(context._load_state_dict_post_hooks), 0)
        nn.Module.load_state_dict(context, context.state_dict())
        self.assertTrue(all(parameter.requires_grad for parameter in shared.parameters()))

    def test_per_context_state_is_reset_and_not_shared(self) -> None:
        shared = _make_composite()
        plan = object()
        work = Mock()
        count_gather = SimpleNamespace(work=work, state="pending")
        pending = SimpleNamespace(count_gather=count_gather)
        frame_slot_cache = object()
        setattr(shared.gaussians_nodes["actor"], "_frame_slot_cache", frame_slot_cache)
        shared._fused_gather_generation_plan = plan  # type: ignore[assignment]
        shared._pending_fused_gather_generation_plan_refresh = pending  # type: ignore[assignment]
        context = make_context_composite(shared)
        sibling = make_context_composite(shared)
        self.assertEqual(context.overriden_state, {})
        self.assertIsNot(context.overriden_state, shared.overriden_state)
        self.assertIsNone(context._lidar_stream)
        self.assertIsNone(context._visibility_local_ranges)
        self.assertTrue(context._is_context_shell)
        self.assertFalse(shared._is_context_shell, "the shared model stays writable")
        self.assertEqual(context._segment_tables, {})
        self.assertIsNot(context._segment_tables, shared._segment_tables)
        self.assertIsNone(context._static_collect_cache)
        self.assertIsNone(context._static_collect_cache_stream)
        self.assertIsNone(context._static_collect_cache_layout)
        self.assertIsNone(context._fourier_fold)
        self.assertIs(context._edit_overrides, shared._edit_overrides)
        self.assertIsNot(
            context._fused_payload_integer_range_validation_state,
            shared._fused_payload_integer_range_validation_state,
        )
        self.assertIsNot(
            context._fused_payload_integer_range_validation_state,
            sibling._fused_payload_integer_range_validation_state,
        )
        self.assertIsNone(context._fused_gather_generation_plan)
        self.assertIsNone(context._pending_fused_gather_generation_plan_refresh)
        self.assertIs(shared._fused_gather_generation_plan, plan)
        self.assertIs(shared._pending_fused_gather_generation_plan_refresh, pending)
        self.assertIsNone(getattr(context.gaussians_nodes["actor"], "_frame_slot_cache"))
        self.assertIsNone(getattr(sibling.gaussians_nodes["actor"], "_frame_slot_cache"))
        self.assertIs(getattr(shared.gaussians_nodes["actor"], "_frame_slot_cache"), frame_slot_cache)
        self.assertEqual(count_gather.state, "pending")
        work.wait.assert_not_called()

    def test_structural_rebind_is_copy_on_write(self) -> None:
        shared = _make_composite()
        context = make_context_composite(shared)
        original_ptr = shared.gaussians_nodes["actor"].positions.data_ptr()

        # A structural edit rebinds the Parameter (the COW path); it must touch only this context.
        context.gaussians_nodes["actor"].positions = nn.Parameter(torch.randn(5, 3))

        self.assertEqual(shared.gaussians_nodes["actor"].positions.data_ptr(), original_ptr)
        self.assertNotEqual(context.gaussians_nodes["actor"].positions.data_ptr(), original_ptr)

    def test_in_place_mutation_is_visible_across_aliases(self) -> None:
        # Documents the Read-only invariant: an in-place write to a shared tensor hits every alias.
        shared = _make_composite()
        context = make_context_composite(shared)
        with torch.no_grad():
            shared.gaussians_nodes["background"].positions.add_(1.0)
        self.assertTrue(
            torch.equal(
                context.gaussians_nodes["background"].positions,
                shared.gaussians_nodes["background"].positions,
            )
        )

    def test_many_shells_share_one_copy(self) -> None:
        shared = _make_composite()
        shells = [make_context_composite(shared) for _ in range(4)]
        base_ptr = shared.gaussians_nodes["background"].positions.data_ptr()
        for shell in shells:
            self.assertEqual(shell.gaussians_nodes["background"].positions.data_ptr(), base_ptr)

    def test_node_override_dicts_are_isolated(self) -> None:
        # Each node's copy-on-write override snapshots must be per-context. Before the fix the shallow
        # __dict__ copy shared the *dict objects* with the shared model, so a save/restore/remove on one
        # shell wrote into every context. Assert a shell's dicts are fresh, empty, and independent.
        shared = _make_composite()
        shared.gaussians_nodes["actor"].overriden_parameters["sentinel"] = nn.Parameter(torch.zeros(1))
        context = make_context_composite(shared)

        actor_shell = context.gaussians_nodes["actor"]
        actor_shared = shared.gaussians_nodes["actor"]
        self.assertEqual(actor_shell.overriden_parameters, {})
        self.assertIsNot(actor_shell.overriden_parameters, actor_shared.overriden_parameters)
        self.assertIsNot(actor_shell.overriden_buffers, actor_shared.overriden_buffers)
        self.assertIsNot(actor_shell.overriden_state, actor_shared.overriden_state)

        # A write on the shell node must not leak into the shared model (or any sibling shell).
        actor_shell.overriden_parameters["edited"] = nn.Parameter(torch.zeros(1))
        self.assertNotIn("edited", actor_shared.overriden_parameters)
        self.assertEqual(list(actor_shared.overriden_parameters), ["sentinel"])

    def test_shared_submodule_identity_survives_cloning(self) -> None:
        # A submodule bound under two names (mirrors the composite's shared renderer handles) must be
        # cloned once so both names still point at the *same* clone — the module memo in
        # _clone_module_tree. Without it each name would fan out to a distinct clone.
        class _Parent(nn.Module):
            def __init__(self, child: nn.Module) -> None:
                super().__init__()
                self.primary = child
                self.alias = child  # same object under a second submodule name

        child = _TinyNode(4)
        parent = _Parent(child)
        self.assertIs(parent.primary, parent.alias)  # sanity: shared in the source tree

        clone = _clone_module_tree(parent, _alias_tensor, {}, {})
        self.assertIs(clone.primary, clone.alias)  # identity preserved across the clone
        self.assertIsNot(clone.primary, child)  # but it is a fresh clone, not the original


class TestAliasExternalWeights(unittest.TestCase):
    def test_rebinds_params_and_buffers_to_external_tensors(self) -> None:
        composite = _make_composite()
        # External tensors with the composite's state_dict names but *different* storage.
        external: dict[str, torch.Tensor] = {
            name: torch.empty_like(tensor).copy_(tensor) for name, tensor in composite.named_parameters()
        }
        external.update({name: torch.empty_like(tensor).copy_(tensor) for name, tensor in composite.named_buffers()})
        original_ptrs = {name: tensor.data_ptr() for name, tensor in composite.named_parameters()}

        alias_external_weights(composite, external)

        for name, tensor in composite.named_parameters():
            self.assertEqual(tensor.data_ptr(), external[name].data_ptr(), msg=f"param {name} not aliased to external")
            self.assertNotEqual(tensor.data_ptr(), original_ptrs[name], msg=f"param {name} still on local storage")
        for name, buffer in composite.named_buffers():
            self.assertEqual(buffer.data_ptr(), external[name].data_ptr())

    def test_missing_external_tensor_raises(self) -> None:
        composite = _make_composite()
        partial = dict(composite.named_parameters())  # buffers omitted
        with self.assertRaises(KeyError):
            alias_external_weights(composite, partial)

    def test_shared_parameter_under_two_names_is_rebound_idempotently(self) -> None:
        # A single Parameter registered under two qualified names (two sibling submodules sharing one
        # Parameter object) must be rebound under BOTH names. `named_parameters` default dedup would
        # visit only the first, leaving the second on local storage. One external tensor supplied under
        # both names must alias cleanly (idempotent) rather than double-bind or error.
        class _Leaf(nn.Module):
            def __init__(self, weight: nn.Parameter) -> None:
                super().__init__()
                self.weight = weight

        class _Root(nn.Module):
            def __init__(self, weight: nn.Parameter) -> None:
                super().__init__()
                self.a = _Leaf(weight)
                self.b = _Leaf(weight)  # same Parameter object under a.weight and b.weight

        shared_weight = nn.Parameter(torch.randn(4, 3))
        root = _Root(shared_weight)
        # Sanity: default dedup hides the second name; remove_duplicate=False surfaces both.
        self.assertEqual({n for n, _ in root.named_parameters()}, {"a.weight"})
        self.assertEqual({n for n, _ in root.named_parameters(remove_duplicate=False)}, {"a.weight", "b.weight"})

        external = torch.empty_like(shared_weight).copy_(shared_weight)
        # The SAME external tensor under BOTH names: the idempotence trap (owner-deduplicated manifests
        # supply one canonical tensor for a shared weight).
        alias_external_weights(root, {"a.weight": external, "b.weight": external})

        self.assertEqual(root.a.weight.data_ptr(), external.data_ptr())
        self.assertEqual(root.b.weight.data_ptr(), external.data_ptr())
        self.assertNotEqual(root.a.weight.data_ptr(), shared_weight.data_ptr())

    def test_shared_parameter_with_owner_deduplicated_manifest_rebinds_both(self) -> None:
        # The real owner manifest is remove_duplicate=True: a shared weight is exported under only its
        # canonical name. The worker (remove_duplicate=False) sees both names; the duplicate whose name
        # is absent from the manifest must still be aliased to the canonical external tensor, not error.
        class _Leaf(nn.Module):
            def __init__(self, weight: nn.Parameter) -> None:
                super().__init__()
                self.weight = weight

        class _Root(nn.Module):
            def __init__(self, weight: nn.Parameter) -> None:
                super().__init__()
                self.a = _Leaf(weight)
                self.b = _Leaf(weight)

        shared_weight = nn.Parameter(torch.randn(4, 3))
        root = _Root(shared_weight)
        external = torch.empty_like(shared_weight).copy_(shared_weight)

        alias_external_weights(root, {"a.weight": external})  # only the canonical name, like the owner

        self.assertEqual(root.a.weight.data_ptr(), external.data_ptr())
        self.assertEqual(root.b.weight.data_ptr(), external.data_ptr())

    def test_surplus_external_key_raises(self) -> None:
        # Fail-loud is bidirectional: a surplus external tensor (owner exports a name the worker's
        # structure lacks -- exactly the divergence _assert_aliasable guards) must raise, not pass.
        composite = _make_composite()
        external: dict[str, torch.Tensor] = {
            name: torch.empty_like(tensor).copy_(tensor) for name, tensor in composite.named_parameters()
        }
        external.update({name: torch.empty_like(tensor).copy_(tensor) for name, tensor in composite.named_buffers()})
        external["gaussians_nodes.actor.nonexistent"] = torch.zeros(1)

        with self.assertRaisesRegex(KeyError, "no matching local"):
            alias_external_weights(composite, external)

    def test_shape_mismatch_raises(self) -> None:
        # A structural divergence (an external tensor whose shape differs from the local param it replaces)
        # must fail loudly rather than silently alias a mismatched tensor and render garbage.
        composite = _make_composite()
        external: dict[str, torch.Tensor] = {
            name: torch.empty_like(tensor).copy_(tensor) for name, tensor in composite.named_parameters()
        }
        external.update({name: torch.empty_like(tensor).copy_(tensor) for name, tensor in composite.named_buffers()})
        bad_name, local = next(iter(composite.named_parameters()))
        external[bad_name] = local.unsqueeze(0).clone()  # same dtype, deliberately wrong shape

        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            alias_external_weights(composite, external)

    def test_dtype_mismatch_raises(self) -> None:
        composite = _make_composite()
        external: dict[str, torch.Tensor] = {
            name: torch.empty_like(tensor).copy_(tensor) for name, tensor in composite.named_parameters()
        }
        external.update({name: torch.empty_like(tensor).copy_(tensor) for name, tensor in composite.named_buffers()})
        bad_name, local = next(iter(composite.named_parameters()))
        alt_dtype = torch.float64 if local.dtype != torch.float64 else torch.float32
        external[bad_name] = torch.zeros_like(local, dtype=alt_dtype)  # right shape, wrong dtype

        with self.assertRaisesRegex(ValueError, "dtype mismatch"):
            alias_external_weights(composite, external)


class TestBuildCompositeSkeleton(unittest.TestCase):
    def test_skeleton_matches_structure_with_meta_storage(self) -> None:
        reference = _make_composite()
        skeleton = build_composite_skeleton(reference)

        reference_params = dict(reference.named_parameters())
        skeleton_params = dict(skeleton.named_parameters())
        self.assertEqual(reference_params.keys(), skeleton_params.keys())
        for name, reference_param in reference_params.items():
            self.assertEqual(skeleton_params[name].shape, reference_param.shape)
            self.assertEqual(skeleton_params[name].device.type, "meta")

        self.assertEqual(set(skeleton.state_dict().keys()), set(reference.state_dict().keys()))

    def test_skeleton_allocates_no_real_storage(self) -> None:
        reference = _make_composite()
        skeleton = build_composite_skeleton(reference)
        self.assertTrue(all(p.device.type == "meta" for p in skeleton.parameters()))
        self.assertTrue(all(b.device.type == "meta" for b in skeleton.buffers()))

    def test_skeleton_does_not_alias_training_generation_state(self) -> None:
        reference = _make_composite()
        plan = object()
        work = Mock()
        count_gather = SimpleNamespace(work=work, state="pending")
        pending = SimpleNamespace(count_gather=count_gather)
        reference._fused_gather_generation_plan = plan  # type: ignore[assignment]
        reference._pending_fused_gather_generation_plan_refresh = pending  # type: ignore[assignment]

        skeleton = build_composite_skeleton(reference)

        self.assertIsInstance(
            skeleton._fused_payload_integer_range_validation_state,
            _PayloadIntegerRangeValidationState,
        )
        self.assertIsNot(
            skeleton._fused_payload_integer_range_validation_state,
            reference._fused_payload_integer_range_validation_state,
        )
        self.assertIsNone(skeleton._fused_gather_generation_plan)
        self.assertIsNone(skeleton._pending_fused_gather_generation_plan_refresh)
        self.assertIs(reference._fused_gather_generation_plan, plan)
        self.assertIs(reference._pending_fused_gather_generation_plan_refresh, pending)
        self.assertEqual(count_gather.state, "pending")
        work.wait.assert_not_called()

    # Note: materializing real (aliased) weights into a context is done by
    # `make_context_composite` (tested above), which clones+aliases directly and therefore does NOT
    # rely on `GaussiansComposite.load_state_dict` (an override that reaches into `self.calib` and
    # other real submodules). The skeleton + `load_state_dict(assign=True)` path used by the worker
    # reconstruction is exercised separately against a real (GPU) composite.


if __name__ == "__main__":
    unittest.main()
