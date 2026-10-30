# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render callables for the PPISP SH-match optimization, driven by the production camera renderer.

The :func:`nurec.utils.io.export.ppisp_export.sh_match.optimization.optimize_sh_to_match_ppisp`
loop is renderer-agnostic. This module wires it to the system's
parameter-driven camera renderer
(``model.gaussians_renderer.render(rendering_data, gaussian_parameters,
...)``) -- the same entry point the trainer used to produce the
checkpoint, and the only Gaussian-render path in the codebase that
exposes a differentiable forward. The concrete renderer backend is
selected by the Hydra training config and is not referenced here; the
protocol :class:`ParameterDrivenRenderer` captures the only call shape
needed.

Two callables are constructed against the live system:

* ``reference_render(view) -> RGB`` renders with a frozen SH snapshot
  applied via a per-Gaussian-node ``param.data`` swap. The optimization
  loop wraps this call in ``torch.no_grad()``.
* ``live_render(view) -> RGB`` renders the live model. Gradients flow
  from RGB through ``collect_gaussian_parameters`` and the differentiable
  renderer back into the SH-fit parameters.

The renderer is stateless w.r.t. SH (``render`` consumes
``gaussian_parameters`` per call), so we never need to ``deepcopy`` the
model -- a ``param.data`` swap is enough to redirect the renderer at the
snapshot tensors. Other (frozen) parameters -- ``positions``,
``rotations``, ``scales``, ``extra_signal``, etc. -- are identical in
both the reference and live collections by construction.

Historical context (what motivated this module) lives in
``docs/ppisp_sh_optimization_renderer_plan.md``.

Determinism for deformable models
---------------------------------

``DeformableGaussianModel.get_layer_data`` may consume RNG when
``is_training_batch=True`` (per-iteration ``timestamp_delta``). To prevent
the loss from measuring deformation noise instead of PPISP/SH discrepancy,
the two callables share a small mutable RNG-state cell:

* ``reference_render`` snapshots the CPU/CUDA RNG state on entry.
* ``live_render`` restores that snapshot on entry, so both views see the
  same RNG.

Contract: within one optimization iteration, callers must invoke
``reference_render`` before ``live_render`` on the same view. The
production loop does exactly that
(:func:`nurec.utils.io.export.ppisp_export.sh_match.optimization.optimize_sh_to_match_ppisp`).
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    cast,
    runtime_checkable,
)

import torch
import torch.nn as nn

from nurec.utils.io.export.ppisp_export.sh_match.node_iteration import (
    SINGLE_NODE_FALLBACK_KEY,
    iter_gaussian_nodes,
)
from nurec.utils.io.export.ppisp_export.sh_match.optimization import TrainingViewContext


if TYPE_CHECKING:
    from nurec.utils.batch import DataAndRenderingBatch


log = logging.getLogger(__name__)


# =============================================================================
# Public constants
# =============================================================================


SH_FIT_PARAMETER_NAMES: Tuple[str, ...] = ("features_albedo", "features_specular", "densities")
"""Per-node SH-fit parameters carried in :class:`SHParameterSnapshot`.

Mirrors :data:`nurec.utils.io.export.ppisp_export.sh_match.optimization._TRAINABLE_PARAMETER_NAMES`
exposed publicly so callers can iterate without reaching into the loop's
private namespace.
"""


# =============================================================================
# Node iteration (re-exported from node_iteration to keep the public surface
# of this module stable; the definitions live in the leaf module to break the
# static import cycle with sh_match.optimization).
# =============================================================================


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class SHFitGaussianNode(Protocol):
    """Per-gaussian-node SH-fit parameter surface.

    All three attributes are :class:`torch.nn.Parameter`. Snapshotting
    relies on ``param.data`` being swappable; the production
    ``SHGaussianModel`` and its subclasses satisfy this.
    """

    features_albedo: nn.Parameter
    features_specular: nn.Parameter
    densities: nn.Parameter


class ParameterDrivenRenderer(Protocol):
    """Surface of :class:`nurec.models.gaussians.renderers.BaseGaussianRenderer`.

    Only the differentiable ``render`` entry point is needed -- not
    ``render_with_deferred_bp`` (deferred-bp is a training-only
    memory-optimization hook that the SH-match loop does not exercise).
    """

    def render(
        self,
        rendering_data: object,
        gaussian_parameters: Dict[str, torch.Tensor],
        n_active_features: int,
        extra_ray_signal_infos: Tuple[List[str], List[int], List[Callable[..., object]]],
        frame_meta: Optional[List[object]] = ...,
    ) -> object: ...


class SHMatchRenderableModel(Protocol):
    """Composite surface the render callables consume.

    Production: satisfied by :class:`nurec.models.gaussians.gaussians_composite.GaussiansComposite`.
    Tests pass a duck-typed fake exposing the same five members.

    Attributes:
        gaussians_nodes: Per-node mapping. Each value satisfies
            :class:`SHFitGaussianNode`.
        gaussians_renderer: Differentiable camera renderer.
        camera_extra_ray_signal_infos: Renderer plumbing for camera-
            extra ray signals (forwarded unchanged).
    """

    gaussians_nodes: Mapping[str, SHFitGaussianNode]
    gaussians_renderer: ParameterDrivenRenderer
    camera_extra_ray_signal_infos: Tuple[List[str], List[int], List[Callable[..., object]]]

    def collect_gaussian_parameters(
        self,
        rendering_data: object,
        is_training_batch: bool = ...,
    ) -> Dict[str, torch.Tensor]: ...

    def get_n_active_features(self) -> int: ...


# =============================================================================
# View payload
# =============================================================================


@dataclass(frozen=True)
class TrainingViewBatch:
    """Per-iteration training-view payload carried on ``TrainingViewContext.extra``.

    Mirrors the dataclass of the same name in the deprecated
    :mod:`nurec.utils.io.export.ppisp_export.sh_match.renderer_callable`. Re-exported
    here so consumers depend on the renderer-driven module rather than
    the about-to-be-removed one.

    Attributes:
        batch_idx: Sequence number assigned by the sampler.
        rendering_data: ``RenderingData`` for the camera view.
        frame_meta: ``data.camera.meta[0]`` for the batch.
    """

    batch_idx: int
    rendering_data: object
    frame_meta: object


# =============================================================================
# Snapshot
# =============================================================================


@dataclass(frozen=True)
class SHParameterSnapshot:
    """Frozen per-node copy of the three SH-fit parameters.

    Each per-node entry stores a ``detach().clone()`` of the live
    parameter's ``.data`` at construction time. The snapshot is
    consumed by :func:`make_sh_match_render_callables` to redirect the
    renderer at frozen SH coefficients without ``copy.deepcopy(model)``.

    Attributes:
        per_node: ``{node_id: {parameter_name: tensor}}``. Required
            parameter names are exactly :data:`SH_FIT_PARAMETER_NAMES`.
    """

    per_node: Mapping[str, Mapping[str, torch.Tensor]]


def snapshot_sh_fit_parameters(model: SHMatchRenderableModel) -> SHParameterSnapshot:
    """Take a frozen copy of every SH-fit parameter on every gaussian node.

    Args:
        model: Object satisfying :class:`SHMatchRenderableModel`. Each
            entry of ``model.gaussians_nodes`` must expose all three
            :data:`SH_FIT_PARAMETER_NAMES` as :class:`torch.nn.Parameter`.

    Returns:
        Immutable :class:`SHParameterSnapshot`.

    Raises:
        TypeError: ``model.gaussians_nodes`` is missing or not a mapping.
        AttributeError: a node is missing a required SH-fit parameter.
        TypeError: a SH-fit attribute is not a :class:`torch.nn.Parameter`.
    """
    nodes = getattr(model, "gaussians_nodes", None)
    if nodes is None or not hasattr(nodes, "items"):
        raise TypeError(
            f"model must expose a mapping 'gaussians_nodes', got {type(model).__name__} (attr={type(nodes).__name__})"
        )

    per_node: Dict[str, Dict[str, torch.Tensor]] = {}
    for node_id, node in nodes.items():
        node_snapshot: Dict[str, torch.Tensor] = {}
        for name in SH_FIT_PARAMETER_NAMES:
            if not hasattr(node, name):
                raise AttributeError(
                    f"gaussian node {node_id!r} is missing SH-fit parameter {name!r}; "
                    "snapshot_sh_fit_parameters requires features_albedo, features_specular, and densities."
                )
            param = getattr(node, name)
            if not isinstance(param, nn.Parameter):
                raise TypeError(
                    f"gaussian node {node_id!r} attribute {name!r} must be a torch.nn.Parameter, got {type(param).__name__}"
                )
            node_snapshot[name] = param.data.detach().clone()
        per_node[node_id] = node_snapshot
    return SHParameterSnapshot(per_node=per_node)


# =============================================================================
# Render callables factory
# =============================================================================


@dataclass
class _RNGSyncCell:
    """Mutable cell sharing CPU/CUDA RNG state between the two render callables.

    Held in a dataclass so :func:`make_sh_match_render_callables`'s two
    closures agree on identity and mypy can track ``Optional`` narrowing.
    """

    cpu_state: Optional[torch.Tensor] = None
    cuda_state_per_device: Dict[int, torch.Tensor] = field(default_factory=dict)


def make_sh_match_render_callables(
    model: SHMatchRenderableModel,
    snapshot: SHParameterSnapshot,
    *,
    is_training_batch: bool = True,
) -> Tuple[
    Callable[[TrainingViewContext], torch.Tensor],
    Callable[[TrainingViewContext], torch.Tensor],
]:
    """Build the two render callables consumed by the SH-match optimization loop.

    Both callables consume a :class:`TrainingViewContext` whose
    ``extra`` is a :class:`TrainingViewBatch`, and return the rendered
    RGB tensor produced by ``model.gaussians_renderer.render(...)``.

    The reference closure swaps each gaussian node's SH-fit ``.data``
    with the snapshot tensors for the duration of one
    ``collect_gaussian_parameters`` call, then restores the live
    ``.data`` (in a ``finally`` block) before any other code observes
    the model. The optimization loop is responsible for wrapping the
    reference call in :func:`torch.no_grad`.

    The live closure renders without any swap. Gradients flow from the
    rendered RGB through ``collect_gaussian_parameters`` and the
    differentiable renderer into the live SH-fit parameters.

    For deformable models, ``collect_gaussian_parameters`` may consume
    RNG when ``is_training_batch=True``. The two closures share an
    RNG-state cell so the live render restores the state captured at
    the start of the reference render -- both collections see identical
    ``timestamp_delta`` draws and the loss measures only PPISP/SH
    discrepancy. Caller contract: invoke ``reference_render`` before
    ``live_render`` for the same view within one iteration.

    Args:
        model: Object satisfying :class:`SHMatchRenderableModel`.
        snapshot: :class:`SHParameterSnapshot` matching ``model``'s
            gaussian node ids and parameter names. Validated lazily
            by the snapshot-swap step at first call (a missing
            ``(node_id, name)`` raises ``KeyError``).
        is_training_batch: Forwarded to ``collect_gaussian_parameters``.
            Default ``True`` matches the production training path the
            checkpoint was produced with.

    Returns:
        ``(reference_render, live_render)`` -- two callables of
        signature ``(TrainingViewContext) -> torch.Tensor``. Both
        return the renderer's RGB output (shape ``[N_pixels, 3]``).

    Raises:
        TypeError: ``model``, ``snapshot``, or ``is_training_batch``
            has the wrong type.
    """
    if not isinstance(snapshot, SHParameterSnapshot):
        raise TypeError(f"snapshot must be a SHParameterSnapshot, got {type(snapshot).__name__}")
    if isinstance(is_training_batch, int) and not isinstance(is_training_batch, bool):
        # bool is a subclass of int -- accept True/False, reject raw ints.
        raise TypeError(f"is_training_batch must be bool, got {type(is_training_batch).__name__}")
    if not isinstance(is_training_batch, bool):
        raise TypeError(f"is_training_batch must be bool, got {type(is_training_batch).__name__}")
    for required in ("gaussians_nodes", "gaussians_renderer", "camera_extra_ray_signal_infos"):
        if not hasattr(model, required):
            raise TypeError(f"model is missing required attribute {required!r} (got {type(model).__name__})")
    if not callable(getattr(model, "collect_gaussian_parameters", None)):
        raise TypeError(f"model.collect_gaussian_parameters must be callable, got {type(model).__name__}")
    if not callable(getattr(model, "get_n_active_features", None)):
        raise TypeError(f"model.get_n_active_features must be callable, got {type(model).__name__}")

    rng_cell = _RNGSyncCell()

    def reference_render(view: TrainingViewContext) -> torch.Tensor:
        batch = _unpack_view(view)
        _capture_rng_state(rng_cell)
        # Defense in depth: even though the optimization loop wraps this
        # call in torch.no_grad(), running the render with the swapped
        # snapshot tensors must never build an autograd graph that could
        # back-propagate into the live Parameter when caller forgets the
        # outer wrap.
        with torch.no_grad():
            with _swap_sh_data(model, snapshot):
                params = model.collect_gaussian_parameters(
                    batch.rendering_data,
                    is_training_batch=is_training_batch,
                )
            return _render_and_extract_rgb(model, batch, params)

    def live_render(view: TrainingViewContext) -> torch.Tensor:
        batch = _unpack_view(view)
        _restore_rng_state(rng_cell)
        params = model.collect_gaussian_parameters(
            batch.rendering_data,
            is_training_batch=is_training_batch,
        )
        return _render_and_extract_rgb(model, batch, params)

    return reference_render, live_render


# =============================================================================
# Internals
# =============================================================================


def _unpack_view(view: TrainingViewContext) -> TrainingViewBatch:
    """Extract the :class:`TrainingViewBatch` from a :class:`TrainingViewContext`.

    Raises:
        TypeError: ``view`` is not a :class:`TrainingViewContext` or
            ``view.extra`` is not a :class:`TrainingViewBatch`.
    """
    if not isinstance(view, TrainingViewContext):
        raise TypeError(f"view must be a TrainingViewContext, got {type(view).__name__}")
    extra = view.extra
    if not isinstance(extra, TrainingViewBatch):
        raise TypeError(
            f"view.extra must be a TrainingViewBatch, got {type(extra).__name__}; "
            "construct the view with make_training_view_sampler from the renderer-callable module."
        )
    return extra


def _render_and_extract_rgb(
    model: SHMatchRenderableModel,
    batch: TrainingViewBatch,
    gaussian_parameters: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Drive ``gaussians_renderer.render`` and return ``out.rgb``.

    Frame metadata is wrapped in a single-element list to match
    :class:`nurec.models.gaussians.renderers.BaseGaussianRenderer.render`'s
    ``Optional[List[FrameMeta]]`` signature.
    """
    frame_meta_list: Optional[List[object]]
    if batch.frame_meta is None:
        frame_meta_list = None
    elif isinstance(batch.frame_meta, list):
        frame_meta_list = list(batch.frame_meta)
    else:
        frame_meta_list = [batch.frame_meta]

    out = model.gaussians_renderer.render(
        batch.rendering_data,
        gaussian_parameters,
        model.get_n_active_features(),
        model.camera_extra_ray_signal_infos,
        frame_meta_list,
    )
    rgb = getattr(out, "rgb", None)
    if rgb is None or not isinstance(rgb, torch.Tensor):
        raise RuntimeError("gaussians_renderer.render returned no rgb tensor; SH-match optimization cannot proceed.")
    return rgb


class _SHDataSwap:
    """Context manager that swaps SH-fit ``.data`` with snapshot tensors.

    Saves each ``(parameter, original_data)`` on entry, swaps in the
    snapshot tensor, and restores the original ``.data`` on exit (always,
    even on exception). Uses :class:`torch.nn.Parameter`'s
    ``.data`` reassignment, which preserves Parameter object identity --
    optimizer state (keyed by Parameter id) is unaffected.
    """

    def __init__(self, model: SHMatchRenderableModel, snapshot: SHParameterSnapshot) -> None:
        self._model = model
        self._snapshot = snapshot
        self._restore: List[Tuple[nn.Parameter, torch.Tensor]] = []

    def __enter__(self) -> "_SHDataSwap":
        for node_id, node_snapshot in self._snapshot.per_node.items():
            try:
                node = self._model.gaussians_nodes[node_id]
            except KeyError as exc:
                self._restore_all()
                raise KeyError(
                    f"snapshot references gaussian node {node_id!r} that is not present on the live model."
                ) from exc
            for name, snap_tensor in node_snapshot.items():
                if not hasattr(node, name):
                    self._restore_all()
                    raise AttributeError(
                        f"gaussian node {node_id!r} is missing SH-fit attribute {name!r} required by the snapshot."
                    )
                param = getattr(node, name)
                if not isinstance(param, nn.Parameter):
                    self._restore_all()
                    raise TypeError(
                        f"gaussian node {node_id!r} attribute {name!r} is not a torch.nn.Parameter "
                        f"(got {type(param).__name__}); cannot swap .data."
                    )
                self._restore.append((param, param.data))
                param.data = snap_tensor
        return self

    def __exit__(self, *_exc: object) -> None:
        self._restore_all()

    def _restore_all(self) -> None:
        for param, original_data in self._restore:
            param.data = original_data
        self._restore.clear()


def _swap_sh_data(model: SHMatchRenderableModel, snapshot: SHParameterSnapshot) -> _SHDataSwap:
    """Factory helper -- one-line construction of :class:`_SHDataSwap`."""
    return _SHDataSwap(model, snapshot)


def _capture_rng_state(cell: _RNGSyncCell) -> None:
    """Snapshot CPU and per-device CUDA RNG state into ``cell``.

    Captures CUDA state for every visible device so multi-GPU systems
    do not accidentally desync between reference and live renders. CPU
    state is always captured.
    """
    cell.cpu_state = torch.get_rng_state()
    cell.cuda_state_per_device.clear()
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            cell.cuda_state_per_device[device_index] = torch.cuda.get_rng_state(device_index)


def _restore_rng_state(cell: _RNGSyncCell) -> None:
    """Restore CPU and per-device CUDA RNG state from ``cell``.

    Silently no-ops when ``cell.cpu_state`` is unset -- the live closure
    is then exposed to whatever RNG state the caller had, which is the
    correct behavior when the optimization loop has not yet run a
    reference render (e.g. an isolated unit test that exercises only
    :func:`live_render`).
    """
    if cell.cpu_state is None:
        return
    torch.set_rng_state(cell.cpu_state)
    if torch.cuda.is_available():
        for device_index, state in cell.cuda_state_per_device.items():
            torch.cuda.set_rng_state(state, device_index)


# =============================================================================
# Training-view sampler -- view selection from a calibrated dataloader
# =============================================================================


def make_training_view_sampler(
    train_dataloader: Iterable["DataAndRenderingBatch"],
) -> Callable[[torch.Generator], TrainingViewContext]:
    """Build a ``sample_view`` callable backed by a calibrated training dataloader.

    The dataloader's native shuffling drives per-iteration randomness;
    the ``torch.Generator`` argument from the optimization loop is
    deliberately ignored. Mirrors the lavoisier-validated algorithmic
    choice of re-rendering the reference on whichever training view
    the dataloader serves next, rather than caching reference frames.

    The dataloader is iterated lazily. When the inner iterator is
    exhausted, a fresh iterator is created so the optimization loop can
    run for an arbitrary number of iterations independent of the
    dataloader's epoch length.

    Args:
        train_dataloader: Any iterable yielding ``DataAndRenderingBatch``
            instances with a populated ``data.camera`` and
            ``rendering.camera``. Use :func:`iterate_calibrated_batches`
            to wrap a raw training-split ``DataLoader`` so the sampler
            can rely on the rendering field being present.

    Returns:
        A ``Callable[[torch.Generator], TrainingViewContext]`` whose
        ``extra`` field is a :class:`TrainingViewBatch`.

    Raises:
        ValueError: ``train_dataloader`` is empty.
    """
    state: _SamplerState = _SamplerState(dataloader=train_dataloader, batch_idx=0)
    state.iterator = iter(train_dataloader)

    def sample_view(generator: torch.Generator) -> TrainingViewContext:
        del generator  # dataloader handles shuffling
        return _next_training_view(state)

    return sample_view


@dataclass
class _SamplerState:
    """Mutable state carried inside the sampler closure."""

    dataloader: Iterable["DataAndRenderingBatch"]
    batch_idx: int
    iterator: Optional[Iterator["DataAndRenderingBatch"]] = None


def _next_training_view(state: _SamplerState) -> TrainingViewContext:
    """Pull the next training batch, restarting the iterator on exhaustion."""
    iterator = state.iterator
    if iterator is None:
        iterator = iter(state.dataloader)
        state.iterator = iterator
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(state.dataloader)
        state.iterator = iterator
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise ValueError(
                "training dataloader is empty; SH-match optimization needs at least one training view"
            ) from exc

    view = _build_training_view_context(batch, state.batch_idx)
    state.batch_idx += 1
    return view


def _build_training_view_context(batch: "DataAndRenderingBatch", batch_idx: int) -> TrainingViewContext:
    """Convert a calibrated ``DataAndRenderingBatch`` into a :class:`TrainingViewContext`.

    Reads four fields from ``batch``: ``rendering.camera``,
    ``data.camera.meta[0]``, and the rendering data's resolution.
    Everything else (LiDAR rays, GT pixels, validation masks) is left
    on the dataloader side -- the optimization only needs the rendered
    RGB to drive its loss.
    """
    if batch.data.camera is None:
        raise ValueError(
            f"training batch idx={batch_idx} has no data.camera; SH-match optimization needs camera batches only"
        )
    if batch.rendering is None or batch.rendering.camera is None:
        raise ValueError(
            f"training batch idx={batch_idx} has no rendering.camera; SH-match optimization needs camera "
            "rendering data. Wrap the raw dataloader with iterate_calibrated_batches() to populate it."
        )
    rendering_data = batch.rendering.camera
    frame_meta = batch.data.camera.meta[0]
    width = int(rendering_data.w)
    height = int(rendering_data.h)
    pixel_coords = _full_grid_pixel_coords(width=width, height=height, device=rendering_data.rays.device)

    extra = TrainingViewBatch(
        batch_idx=batch_idx,
        rendering_data=rendering_data,
        frame_meta=frame_meta,
    )
    return TrainingViewContext(
        pixel_coords=pixel_coords,
        resolution=(width, height),
        extra=extra,
    )


def _full_grid_pixel_coords(*, width: int, height: int, device: torch.device | str) -> torch.Tensor:
    """Return ``[width*height, 2]`` ``int16`` pixel coords in row-major (xy) order.

    Mirrors :func:`nurec.utils.batch.generate_grid_2d_indices` without
    importing it; the dependency would pull a much larger surface than
    this adapter needs and would prevent hermetic CPU-only testing on
    hosts that do not have the full system stack available.
    """
    xs, ys = torch.meshgrid(
        torch.arange(width, dtype=torch.int16, device=device),
        torch.arange(height, dtype=torch.int16, device=device),
        indexing="xy",
    )
    return torch.stack([xs.flatten(), ys.flatten()], dim=1)


# =============================================================================
# Calibrated dataloader wrapper
# =============================================================================


CalibFn = Callable[..., object]
"""Type alias for ``model.calib`` -- ``(data_batch, *, skip_calib=...) -> RenderingBatch``.

Modeled as ``Callable[..., object]`` because the production
:class:`nurec.models.calib.BaseCalib` signature carries optional kwargs
the SH-match path does not need to surface in static typing.
"""


class CalibratedBatchLoader:
    """Re-iterable wrapper that materialises ``batch.rendering`` for SH-match.

    Production raw training batches do not carry a populated
    ``batch.rendering``; the system's training step calls
    ``model.calib(batch.data, skip_calib=True)`` to fill it just-in-time.
    The export-time SH-match path reuses the training dataset but
    bypasses ``GaussiansSystem.training_step``, so this wrapper applies
    the same calibration step.

    The wrapper is re-iterable: every call to ``__iter__`` builds a
    fresh underlying iterator so :func:`make_training_view_sampler`'s
    epoch-restart logic continues to work.

    Args:
        raw_loader: An iterable yielding raw ``DataAndRenderingBatch``
            instances. Typically a ``torch.utils.data.DataLoader`` over
            the train-split dataset.
        calib_fn: Bound method (or callable) that, given a
            ``DataBatch``, returns a ``RenderingBatch`` to assign on
            ``batch.rendering``. Production: ``system.model.calib``.
        device: Optional torch device. When set, every batch is moved
            via ``batch.to(device)`` before calibration. ``None`` skips
            the move (the raw loader is assumed to already place
            tensors on the desired device).
    """

    def __init__(
        self,
        raw_loader: Iterable["DataAndRenderingBatch"],
        calib_fn: CalibFn,
        device: Optional[torch.device] = None,
    ) -> None:
        if not callable(calib_fn):
            raise TypeError(f"calib_fn must be callable, got {type(calib_fn).__name__}")
        if device is not None and not isinstance(device, torch.device):
            raise TypeError(f"device must be a torch.device or None, got {type(device).__name__}")
        self._raw_loader = raw_loader
        self._calib_fn = calib_fn
        self._device = device

    def __iter__(self) -> Iterator["DataAndRenderingBatch"]:
        for raw_batch in self._raw_loader:
            batch = raw_batch.to(self._device) if self._device is not None else raw_batch
            if batch.rendering is None:
                # ``calib_fn`` is duck-typed (``Callable[..., object]``) so this
                # module stays CPU-testable without importing the whole
                # ``nurec.models.calib`` stack. Production callable returns a
                # ``RenderingBatch``; the cast bridges the gap for mypy.
                from nurec.utils.batch import RenderingBatch  # local import: avoid cycles

                batch.rendering = cast(RenderingBatch, self._calib_fn(batch.data, skip_calib=True))
            yield batch


def iterate_calibrated_batches(
    raw_loader: Iterable["DataAndRenderingBatch"],
    calib_fn: CalibFn,
    device: Optional[torch.device] = None,
) -> CalibratedBatchLoader:
    """Function-style alias for :class:`CalibratedBatchLoader`.

    Provided for symmetry with :func:`make_training_view_sampler` and
    :func:`make_sh_match_render_callables`. The orchestrator imports
    this name; tests can import either form.
    """
    return CalibratedBatchLoader(raw_loader=raw_loader, calib_fn=calib_fn, device=device)


# =============================================================================
# Re-exports kept for downstream import stability
# =============================================================================


__all__ = [
    "SH_FIT_PARAMETER_NAMES",
    "SINGLE_NODE_FALLBACK_KEY",
    "CalibFn",
    "CalibratedBatchLoader",
    "SHFitGaussianNode",
    "SHMatchRenderableModel",
    "SHParameterSnapshot",
    "TrainingViewBatch",
    "iter_gaussian_nodes",
    "iterate_calibrated_batches",
    "make_sh_match_render_callables",
    "make_training_view_sampler",
    "snapshot_sh_fit_parameters",
]
