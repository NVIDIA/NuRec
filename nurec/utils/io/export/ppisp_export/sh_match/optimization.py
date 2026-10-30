# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SH Optimization to match a fixed PPISP transform.

Hosts the configuration, the adapter object, the optimizer-setup
helper, and the multi-view Adam refinement loop that folds a fixed
PPISP transform into the spherical harmonic coefficients of a
Gaussian model. The loop is renderer-agnostic: it consumes
caller-provided ``sample_view`` and ``render_view`` callables plus a
frozen ``reference_model`` instance, so the actual borel
renderer/dataset integration lives at the call site (the export
entry point) and never leaks into this module's import surface.

The reference render is recomputed on every iteration (matching the
3dgrut training loop and the validated lavoisier SH-bake algorithm)
rather than cached. This keeps memory bounded by the working-set of
two model instances regardless of how many training views the
caller's dataset has, at the cost of one extra ``no_grad`` reference
render per step. For a typical multi-camera export
(e.g. 8 cameras x 300 frames at 4K) caching all reference targets
would require tens of GB of extra memory; per-iter rendering avoids
that entirely.

Workflow
--------
1. *DC warm-start* (see
   :func:`nurec.utils.io.export.ppisp_export.sh_match.dc_warmstart.warmstart_sh_dc_from_ppisp`):
   seed ``features_albedo`` by passing each Gaussian's diffuse RGB
   through PPISP at a single ``(camera_id, frame_id)``.
2. *Multi-view Adam refinement*: co-optimize ``features_albedo``,
   ``features_specular``, and ``densities`` against PPISP-processed
   reference renders across multiple training views. The reference
   rendering uses :class:`FixedCameraFramePPISP` with vignetting
   disabled to stay color-space-consistent with the warm-start
   (display-referred RGB).
3. The refined Gaussians no longer need a runtime PPISP shader; the
   USD asset can be consumed by any renderer that supports SH eval.

Naming choices
--------------
* ``SHMatchingOptimizationConfig`` -- the config drives the SH
  coefficient optimization that *matches* PPISP output, not the
  optimization of PPISP parameters. The verbose name avoids the
  ambiguity of "ppisp optimization".
* ``PPISPSHOptimizationAdapter`` -- mediates between the trained
  PPISP module and the SH refinement: warm-start in, identity
  fit-transform out (PPISP is rendered into the reference, not
  applied during the model's forward pass).
* ``FixedCameraFramePPISP`` -- the PPISP forward used as the
  reference target is pinned to one ``(camera_id, frame_id)``, with
  vignetting disabled and the controller forced off.
"""

from __future__ import annotations

import copy
import logging
import os

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Protocol, Tuple, cast, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from nurec.utils.io.export.gaussian_radiance_scaling import SHCoefficientsBearer
from nurec.utils.io.export.ppisp_export.sh_match.dc_warmstart import warmstart_sh_dc_from_ppisp
from nurec.utils.io.export.ppisp_export.sh_match.node_iteration import iter_gaussian_nodes


if TYPE_CHECKING:
    from ppisp import PPISP  # type: ignore[import-not-found]


@runtime_checkable
class SHFitTrainable(Protocol):
    """Structural type for any object that exposes the full SH-fit parameter set.

    The multi-view SH refinement touches all three of
    ``features_albedo``, ``features_specular``, and ``densities`` plus
    the ``parameters()`` iterator (``nn.Module`` interface) needed to
    freeze everything else first.

    Production: satisfied by ``SHGaussianModel`` and its subclasses.
    """

    features_albedo: nn.Parameter
    features_specular: nn.Parameter
    densities: nn.Parameter

    def parameters(self) -> Iterable[nn.Parameter]: ...


log = logging.getLogger(__name__)

DEFAULT_NUM_ITERATIONS = 3000
DEFAULT_LR_ALBEDO = 2.5e-3
DEFAULT_LR_DENSITY = 5.0e-2
SPECULAR_LR_RATIO = 1.0 / 20.0  # 3DGS-default specular-to-albedo LR ratio
HOLD_OUT_SAMPLE_FOR_VALIDATION_ENV = "NUREC_SH_MATCH_HOLD_OUT_SAMPLE_FOR_VALIDATION"
VALIDATION_IMAGE_DIR_ENV = "NUREC_SH_MATCH_VALIDATION_IMAGE_DIR"
ZERO_SPECULAR_AFTER_WARMSTART_ENV = "NUREC_SH_MATCH_ZERO_SPECULAR_AFTER_WARMSTART"


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class SHMatchingOptimizationConfig:
    """Hyperparameters for SH coefficient optimization against fixed PPISP.

    Compute is bounded by a fixed number of optimization steps with
    random view sampling rather than a number of epochs over the
    training dataloader, so the budget does not scale with dataset
    size. The default 3000 iterations brings parity with the
    ``threedgrut/export/usd/post_processing_sh_bake.py`` baseline
    (~7 epochs * a few hundred training views) on representative
    scenes; the previous 700 default produced visible color/detail
    regressions on valiant-scale datasets. Per-parameter LRs inherit
    3dgrut's defaults: albedo LR 2.5e-3, specular LR = albedo / 20,
    density LR 5e-2. Letting density breathe alongside SH absorbs
    spatial frequencies the SH alone cannot represent without
    aliasing (worth several dB on harder scenes).

    Attributes:
        num_iterations: Number of optimizer steps to run. At each
            step a single training view is drawn uniformly at random
            (with replacement) from the training set, so the budget
            is independent of dataset size. Must be >= 1.
        learning_rate_albedo: Adam LR for ``features_albedo``. Must be
            strictly positive and finite.
        learning_rate_specular: Adam LR for ``features_specular``.
            ``None`` (default) yields ``learning_rate_albedo / 20``,
            matching the 3DGS training ratio. Must be strictly positive
            and finite when set explicitly.
        learning_rate_density: Adam LR for ``densities``. Must be
            strictly positive and finite.
        device: Torch device used for the optimization, e.g. ``"cuda"``
            or ``"cuda:0"``. Must be a string parseable by
            :class:`torch.device`.
        hold_out_sample_for_validation: Optional number of sampled views
            to evaluate after optimization. ``0`` disables the PSNR/SSIM
            diagnostic pass.

    Raises:
        ValueError: any numeric field outside its valid range.
        TypeError: any field with the wrong type.
    """

    num_iterations: int = DEFAULT_NUM_ITERATIONS
    learning_rate_albedo: float = DEFAULT_LR_ALBEDO
    learning_rate_specular: Optional[float] = None
    learning_rate_density: float = DEFAULT_LR_DENSITY
    device: str = "cuda"
    hold_out_sample_for_validation: int = 0

    def __post_init__(self) -> None:
        _validate_optimization_config(self)

    @property
    def resolved_learning_rate_specular(self) -> float:
        """Effective specular LR after applying the default ratio."""
        if self.learning_rate_specular is not None:
            return float(self.learning_rate_specular)
        return float(self.learning_rate_albedo) * SPECULAR_LR_RATIO


def _validate_optimization_config(config: SHMatchingOptimizationConfig) -> None:
    """Fail-loud validation of every :class:`SHMatchingOptimizationConfig` field.

    Validation runs in :meth:`__post_init__` so an invalid config never
    leaves the constructor.
    """
    if isinstance(config.num_iterations, bool) or not isinstance(config.num_iterations, int):
        raise TypeError(f"num_iterations must be int, got {type(config.num_iterations).__name__}")
    if config.num_iterations < 1:
        raise ValueError(f"num_iterations must be >= 1, got {config.num_iterations}")
    if isinstance(config.hold_out_sample_for_validation, bool) or not isinstance(
        config.hold_out_sample_for_validation, int
    ):
        raise TypeError(
            f"hold_out_sample_for_validation must be int, got {type(config.hold_out_sample_for_validation).__name__}"
        )
    if config.hold_out_sample_for_validation < 0:
        raise ValueError(f"hold_out_sample_for_validation must be >= 0, got {config.hold_out_sample_for_validation}")

    for name, value in (
        ("learning_rate_albedo", config.learning_rate_albedo),
        ("learning_rate_density", config.learning_rate_density),
    ):
        _validate_positive_finite_float(name, value)

    if config.learning_rate_specular is not None:
        _validate_positive_finite_float("learning_rate_specular", config.learning_rate_specular)

    if not isinstance(config.device, str) or not config.device:
        raise TypeError(f"device must be a non-empty str, got {config.device!r}")
    try:
        torch.device(config.device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"device {config.device!r} is not a valid torch device: {exc}") from exc


def _validate_positive_finite_float(name: str, value: object) -> None:
    """Reject non-numeric, ``bool``, non-finite, or non-positive values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not torch.isfinite(torch.tensor(float(value))).item():
        raise ValueError(f"{name} must be finite, got {value!r}")
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {value!r}")


# =============================================================================
# Fixed (camera, frame) PPISP wrapper
# =============================================================================


class FixedCameraFramePPISP(nn.Module):
    """PPISP forward pinned to a single ``(camera_id, frame_id)``.

    Wraps a ``copy.deepcopy`` of the trained PPISP module so the source
    state stays untouched. Vignetting is zeroed and the controller is
    forced off:

    * Vignetting is disabled to match the DC warm-start, which has no
      spatial location for per-Gaussian colors.
    * The controller is disabled because the multi-view SH refinement
      needs a deterministic, view-independent PPISP transform;
      per-frame controller dispatch would re-introduce the very
      view-dependence the SH optimization is meant to fold into the
      coefficients.

    The forward signature accepts (and ignores) the ``camera_idx``,
    ``frame_idx``, ``exposure_prior`` keyword arguments that the
    trained PPISP's ``__call__`` exposes, so this wrapper is a
    drop-in replacement at the call site.
    """

    def __init__(
        self,
        ppisp_module: "PPISP",
        camera_id: int,
        frame_id: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.camera_id = int(camera_id)
        self.frame_id = int(frame_id)
        self.ppisp = copy.deepcopy(ppisp_module).to(device).eval()

        if hasattr(self.ppisp, "config") and hasattr(self.ppisp.config, "use_controller"):
            self.ppisp.config.use_controller = False
        if hasattr(self.ppisp, "vignetting_params"):
            with torch.no_grad():
                self.ppisp.vignetting_params.zero_()

    def forward(
        self,
        rgb: torch.Tensor,
        pixel_coords: torch.Tensor,
        resolution: Tuple[int, int],
        **_: object,
    ) -> torch.Tensor:
        """Apply the fixed PPISP transform to a rendered RGB tile.

        ``camera_idx``, ``frame_idx``, and ``exposure_prior`` are
        accepted (via ``**_``) for call-site compatibility but
        ignored: the wrapper always evaluates at the configured fixed
        indices.
        """
        return self.ppisp(
            rgb,
            pixel_coords,
            resolution=resolution,
            camera_idx=self.camera_id,
            frame_idx=self.frame_id,
            exposure_prior=None,
        )


# =============================================================================
# Adapter
# =============================================================================


class PPISPSHOptimizationAdapter:
    """Glue object passed to the multi-view SH optimization loop.

    Owns the fixed PPISP reference target, drives the warm-start, and
    exposes the identity fit-transform that the optimizer applies to
    the rendered RGB. Decoupling these into one object keeps the
    eventual ``optimize_sh_to_match_ppisp()`` loop trivial to read.

    The adapter deliberately does *not* construct a dataloader and
    does *not* render the reference itself: those concerns belong to
    the optimization loop. This keeps the adapter testable without
    the renderer/dataset pipeline.
    """

    def __init__(
        self,
        config: SHMatchingOptimizationConfig,
        ppisp_module: "PPISP",
        camera_id: int,
        frame_id: int,
    ) -> None:
        if not isinstance(config, SHMatchingOptimizationConfig):
            raise TypeError(f"config must be a SHMatchingOptimizationConfig, got {type(config).__name__}")
        _validate_camera_frame_ids(ppisp_module, camera_id, frame_id)

        self._config = config
        self._ppisp_module = ppisp_module
        self._camera_id = int(camera_id)
        self._frame_id = int(frame_id)
        self._device = torch.device(config.device)
        self._fixed_ppisp = FixedCameraFramePPISP(
            ppisp_module=ppisp_module,
            camera_id=self._camera_id,
            frame_id=self._frame_id,
            device=self._device,
        )

    @property
    def config(self) -> SHMatchingOptimizationConfig:
        return self._config

    @property
    def camera_id(self) -> int:
        return self._camera_id

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def fixed_ppisp(self) -> FixedCameraFramePPISP:
        return self._fixed_ppisp

    def warmstart(self, model: SHCoefficientsBearer) -> Tuple[float, torch.Tensor]:
        """Initialize ``features_albedo`` by folding PPISP into the SH DC band.

        Walks the per-node SH surface via :func:`iter_gaussian_nodes`
        and applies :func:`warmstart_sh_dc_from_ppisp` to every node.
        For a :class:`GaussiansComposite` this bakes the PPISP DC into
        each member SH-bearing node; for a single
        ``SHGaussianModel`` (legacy / tests) the helper degenerates to a
        single call.

        Returns the ``(exposure, color)`` actually applied -- identical
        across nodes by construction (depends only on
        ``camera_id`` / ``frame_id`` / ``ppisp_module``), so the
        per-node return value of the last iteration is representative.
        """
        result: Optional[Tuple[float, torch.Tensor]] = None
        for node_id, node in iter_gaussian_nodes(model):
            result = warmstart_sh_dc_from_ppisp(
                model=cast("SHCoefficientsBearer", node),
                ppisp_module=self._ppisp_module,
                camera_id=self._camera_id,
                frame_id=self._frame_id,
                device=self._device,
            )
            if _env_flag(ZERO_SPECULAR_AFTER_WARMSTART_ENV):
                with torch.no_grad():
                    getattr(node, "features_specular").zero_()
                log.info("zeroed features_specular after warm-start for gaussian node %r", node_id)
            log.debug("warmstart applied to gaussian node %r", node_id)
        if result is None:
            raise ValueError(
                "iter_gaussian_nodes returned no nodes for the warm-start input; "
                "expected at least one SH-bearing node (composite or single)."
            )
        return result

    def apply_fit_transform(self, rgb: torch.Tensor) -> torch.Tensor:
        """Identity-with-clamp transform on the rendered RGB.

        The model's forward is what actually evolves: PPISP is *not*
        re-applied during the loop's forward pass because
        ``features_albedo`` already lives in display-referred (gamma)
        space after the DC warm-start. Clamping mirrors the
        ``saturate(rgb)`` at the end of the SPG shader and keeps the
        loss from diverging into negatives.
        """
        return torch.clamp(rgb, 0.0, 1.0)

    def setup_optimizer(self, model: SHFitTrainable) -> torch.optim.Optimizer:
        """Build the 3-group Adam optimizer used by the multi-view SH refinement.

        Delegates to :func:`_setup_optimization_parameters` so the
        parameter-freezing policy stays in one place.
        """
        return _setup_optimization_parameters(model, self._config)


# =============================================================================
# Optimizer setup
# =============================================================================


_TRAINABLE_PARAMETER_NAMES = ("features_albedo", "features_specular", "densities")


def _setup_optimization_parameters(
    model: SHFitTrainable,
    config: SHMatchingOptimizationConfig,
) -> torch.optim.Optimizer:
    """Freeze every parameter, unfreeze the SH-fit set, and return Adam.

    The fit set is exactly ``features_albedo``, ``features_specular``,
    and ``densities``. Letting density breathe alongside SH absorbs
    spatial frequencies the SH alone cannot represent without
    aliasing.

    Composite-aware: walks :func:`iter_gaussian_nodes` so a
    :class:`GaussiansComposite` contributes one SH-fit triple per
    member node and the optimizer registers them all in a single
    per-group list. The historical single-node ``SHGaussianModel``
    path collapses to one triple via the helper's fallback.

    Args:
        model: Either a :class:`GaussiansComposite` whose
            ``gaussians_nodes`` each expose the three SH-fit
            ``torch.nn.Parameter`` attributes, or a single SH gaussian
            node exposing them directly. Must also expose a
            ``parameters()`` iterator (``nn.Module`` interface) so the
            global freeze pass covers everything outside the SH-fit
            set.
        config: Optimization config providing the per-group LRs.

    Returns:
        ``torch.optim.Adam`` with three parameter groups:
        ``features_albedo`` at ``learning_rate_albedo``,
        ``features_specular`` at the resolved specular LR,
        ``densities`` at ``learning_rate_density``. Each group's
        ``params`` list contains one entry per gaussian node.

    Raises:
        AttributeError: a gaussian node is missing a required SH-fit
            parameter.
        TypeError: a required attribute is not a ``torch.nn.Parameter``
            or ``model.parameters()`` is missing.
        ValueError: ``iter_gaussian_nodes`` returns no nodes.
    """
    if not callable(getattr(model, "parameters", None)):
        raise TypeError(f"model must expose a parameters() iterator (nn.Module interface), got {type(model).__name__}")

    per_group: dict[str, List[nn.Parameter]] = {name: [] for name in _TRAINABLE_PARAMETER_NAMES}
    nodes = iter_gaussian_nodes(model)
    if not nodes:
        raise ValueError(
            "iter_gaussian_nodes returned no nodes for optimizer setup; "
            "expected at least one SH-bearing node (composite or single)."
        )
    for node_id, node in nodes:
        for name in _TRAINABLE_PARAMETER_NAMES:
            if not hasattr(node, name):
                raise AttributeError(
                    f"gaussian node {node_id!r} is missing required SH-fit parameter {name!r}. "
                    "The SH refinement requires features_albedo, features_specular, and densities."
                )
            param = getattr(node, name)
            if not isinstance(param, nn.Parameter):
                raise TypeError(
                    f"gaussian node {node_id!r} attribute {name!r} must be a torch.nn.Parameter, "
                    f"got {type(param).__name__}"
                )
            per_group[name].append(param)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for group in per_group.values():
        for parameter in group:
            parameter.requires_grad_(True)

    return torch.optim.Adam(
        [
            {"params": per_group["features_albedo"], "lr": float(config.learning_rate_albedo)},
            {"params": per_group["features_specular"], "lr": float(config.resolved_learning_rate_specular)},
            {"params": per_group["densities"], "lr": float(config.learning_rate_density)},
        ]
    )


# =============================================================================
# Internals: shared validators
# =============================================================================


def _validate_camera_frame_ids(ppisp_module: object, camera_id: object, frame_id: object) -> None:
    """Range-check ``camera_id`` and ``frame_id`` against the trained PPISP module.

    Mirrors :func:`dc_warmstart._validate_warmstart_inputs`'s
    index-checking logic so the adapter rejects bad indices at
    construction time, before any deep-copy of the PPISP module.
    """
    if isinstance(camera_id, bool) or not isinstance(camera_id, int):
        raise TypeError(f"camera_id must be int, got {type(camera_id).__name__}")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int):
        raise TypeError(f"frame_id must be int, got {type(frame_id).__name__}")

    for attr in ("num_cameras", "exposure_params"):
        if not hasattr(ppisp_module, attr):
            raise AttributeError(
                f"ppisp_module is missing required attribute '{attr}'. "
                "PPISPSHOptimizationAdapter requires a trained PPISP module."
            )

    num_cameras = int(ppisp_module.num_cameras)  # type: ignore[attr-defined]
    if camera_id < 0 or camera_id >= num_cameras:
        raise ValueError(f"camera_id must be in [0, {num_cameras - 1}], got {camera_id}")

    num_frames = int(ppisp_module.exposure_params.shape[0])  # type: ignore[attr-defined]
    if frame_id < 0 or frame_id >= num_frames:
        raise ValueError(f"frame_id must be in [0, {num_frames - 1}], got {frame_id}")


# =============================================================================
# Multi-view Adam refinement loop
# =============================================================================


DEFAULT_PROGRESS_LOG_EVERY = 50


@dataclass(frozen=True)
class TrainingViewContext:
    """Caller-defined view metadata the SH refinement loop needs.

    The loop never inspects ``extra``; only ``pixel_coords`` and
    ``resolution`` are passed into ``adapter.fixed_ppisp(...)`` to
    apply the fixed PPISP transform to the reference render. The
    caller's ``render_view`` callable is free to read whatever it
    needs out of ``extra`` (e.g. rays, pose, frame index, dataset
    row id) when actually rendering.

    Attributes:
        pixel_coords: ``[N_pixels, 2]`` pixel coordinates for the
            view. Forwarded to the fixed PPISP transform; PPISP
            samples vignetting and CRF based on these coords.
        resolution: ``(width, height)`` of the rendered tile.
            Forwarded to the fixed PPISP transform.
        extra: Caller-defined view payload (rays, pose, indices, ...).
            Opaque to the loop; surfaced in log lines and error
            messages via ``repr(extra)`` only.
    """

    pixel_coords: torch.Tensor
    resolution: Tuple[int, int]
    extra: object = None


def optimize_sh_to_match_ppisp(
    model: SHFitTrainable,
    adapter: PPISPSHOptimizationAdapter,
    sample_view: Callable[[torch.Generator], TrainingViewContext],
    reference_render: Callable[[TrainingViewContext], torch.Tensor],
    live_render: Callable[[TrainingViewContext], torch.Tensor],
    seed: Optional[int] = None,
    progress_log_every: int = DEFAULT_PROGRESS_LOG_EVERY,
) -> None:
    """Refine SH coefficients (and densities) to match a fixed PPISP transform.

    Implements the lavoisier-validated multi-view SH bake against a
    parameter-driven differentiable renderer. At each iteration the
    same training view is rendered twice -- once through a frozen SH
    snapshot (the reference) and once through the live SH (the
    refinement). The reference is pushed through the fixed PPISP
    transform and used as the target for an MSE loss against the live
    render.

    Mutates ``model`` in place. Runs ``adapter.config.num_iterations``
    Adam steps; at each step:

    1. Sample one :class:`TrainingViewContext` from ``sample_view``.
    2. ``reference_render(view)`` -- the closure handles its own
       ``torch.no_grad()`` wrap and SH ``param.data`` swap. The loop
       additionally wraps the reference call in ``torch.no_grad()``
       (defense in depth). The result is pushed through
       ``adapter.fixed_ppisp(...)`` -> ``target_rgb``.
    3. ``live_render(view)`` -- autograd wired through the SH-fit
       parameters. The result is passed through
       ``adapter.apply_fit_transform(...)`` -> ``fitted_rgb``.
    4. ``optimizer.step()`` on the 3-group Adam built by
       :meth:`PPISPSHOptimizationAdapter.setup_optimizer`.

    The reference render is recomputed every iteration. Caching all
    training views' reference targets would require
    O(num_cameras x num_frames x H x W) memory, which for a typical
    8-camera 300-frame 4K dataset is several tens of GB. Per-iter
    rendering keeps the working set bounded.

    No ``copy.deepcopy(model)``: the reference closure handles SH
    isolation via a Parameter-data swap against a previously taken
    snapshot. See
    :func:`nurec.utils.io.export.ppisp_export.sh_match.renderer_callable.make_sh_match_render_callables`
    for the production factory.

    The DC warm-start is *not* run here: callers that want it should
    invoke :meth:`PPISPSHOptimizationAdapter.warmstart` on ``model``
    before this loop.

    Args:
        model: Object satisfying :class:`SHFitTrainable`. Mutated in
            place. The optimizer freezes everything outside the SH-fit
            parameter set.
        adapter: :class:`PPISPSHOptimizationAdapter` carrying the
            optimization config, the fixed PPISP target, and the
            identity-with-clamp fit-transform.
        sample_view: Callable mapping a seeded :class:`torch.Generator`
            to one :class:`TrainingViewContext`. The
            ``torch.Generator`` is provided so the sampler can be
            seeded; sampler implementations are free to ignore it
            when their underlying source (e.g. a ``DataLoader``) has
            its own shuffling.
        reference_render: Closure that takes one
            :class:`TrainingViewContext` and returns the linear RGB of
            the SH-snapshot render. Must be safe to call without an
            outer ``no_grad`` (the loop wraps it for defense in depth).
        live_render: Closure that takes the same
            :class:`TrainingViewContext` and returns the linear RGB of
            the live SH render with autograd wired through the SH-fit
            parameters. The loop calls ``reference_render`` *before*
            ``live_render`` per iteration so any RNG-state
            synchronisation between the two closures (deformable
            models with stochastic ``timestamp_delta`` draws) sees a
            consistent state.
        seed: Optional integer seed for the view sampler's
            :class:`torch.Generator`. ``None`` uses
            :func:`torch.Generator.seed` (non-reproducible).
        progress_log_every: Emit a progress log line every N
            iterations. Set to ``0`` to disable progress logging
            (the final-loss line is always emitted). Must be >= 0.

    Raises:
        TypeError: any input has the wrong type, or ``reference_render``
            / ``live_render`` returns a non-tensor.
        ValueError: ``progress_log_every`` is negative; ``sample_view``
            returns a non-:class:`TrainingViewContext`.
        AttributeError: ``model`` is missing a required SH-fit parameter.
    """
    _validate_optimization_loop_inputs(
        model=model,
        adapter=adapter,
        sample_view=sample_view,
        reference_render=reference_render,
        live_render=live_render,
        seed=seed,
        progress_log_every=progress_log_every,
    )

    optimizer = adapter.setup_optimizer(model)

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    else:
        generator.seed()

    num_iterations = int(adapter.config.num_iterations)
    log.info(
        "ppisp.sh_match.optimization: starting %d iterations (seed=%s)",
        num_iterations,
        "auto" if seed is None else str(seed),
    )

    last_loss: Optional[float] = None
    for iteration in range(num_iterations):
        view = sample_view(generator)
        if not isinstance(view, TrainingViewContext):
            raise TypeError(
                f"sample_view must return a TrainingViewContext, got {type(view).__name__} at iteration {iteration}"
            )

        with torch.no_grad():
            reference_rgb = reference_render(view)
            if not isinstance(reference_rgb, torch.Tensor):
                raise TypeError(
                    f"reference_render must return a torch.Tensor, got "
                    f"{type(reference_rgb).__name__} at iteration {iteration}"
                )
            target_rgb = adapter.fixed_ppisp(
                reference_rgb,
                view.pixel_coords,
                view.resolution,
            ).detach()

        rendered_rgb = live_render(view)
        if not isinstance(rendered_rgb, torch.Tensor):
            raise TypeError(
                f"live_render must return a torch.Tensor, got {type(rendered_rgb).__name__} at iteration {iteration}"
            )

        fitted_rgb = adapter.apply_fit_transform(rendered_rgb)
        loss = F.mse_loss(fitted_rgb, target_rgb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.detach().item())
        if progress_log_every > 0 and (iteration + 1) % progress_log_every == 0:
            log.info(
                "ppisp.sh_match.optimization: iter %d/%d  mse=%.6f",
                iteration + 1,
                num_iterations,
                last_loss,
            )

    log.info(
        "ppisp.sh_match.optimization: done after %d iterations, final mse=%s",
        num_iterations,
        "n/a" if last_loss is None else f"{last_loss:.6f}",
    )
    _report_hold_out_metrics(
        adapter=adapter,
        sample_view=sample_view,
        reference_render=reference_render,
        live_render=live_render,
        generator=generator,
    )


def _report_hold_out_metrics(
    *,
    adapter: PPISPSHOptimizationAdapter,
    sample_view: Callable[[torch.Generator], TrainingViewContext],
    reference_render: Callable[[TrainingViewContext], torch.Tensor],
    live_render: Callable[[TrainingViewContext], torch.Tensor],
    generator: torch.Generator,
) -> None:
    """Optionally report PSNR/SSIM on sampled fixed-PPISP targets."""
    num_samples = _resolve_hold_out_sample_for_validation(adapter.config)
    if num_samples == 0:
        return

    validation_image_dir = _resolve_validation_image_dir()
    psnr_values: List[float] = []
    ssim_values: List[float] = []
    for sample_idx in range(num_samples):
        view = sample_view(generator)
        if not isinstance(view, TrainingViewContext):
            raise TypeError(
                f"sample_view must return a TrainingViewContext, got "
                f"{type(view).__name__} during validation sample {sample_idx}"
            )

        with torch.no_grad():
            reference_rgb = reference_render(view)
            if not isinstance(reference_rgb, torch.Tensor):
                raise TypeError(
                    f"reference_render must return a torch.Tensor, got "
                    f"{type(reference_rgb).__name__} during validation sample {sample_idx}"
                )
            target_rgb = adapter.fixed_ppisp(
                reference_rgb,
                view.pixel_coords,
                view.resolution,
            ).detach()

            rendered_rgb = live_render(view)
            if not isinstance(rendered_rgb, torch.Tensor):
                raise TypeError(
                    f"live_render must return a torch.Tensor, got "
                    f"{type(rendered_rgb).__name__} during validation sample {sample_idx}"
                )
            fitted_rgb = adapter.apply_fit_transform(rendered_rgb)

            psnr_values.append(_compute_psnr(fitted_rgb, target_rgb))
            ssim_value = _compute_ssim(fitted_rgb, target_rgb, view.resolution)
            if ssim_value is not None:
                ssim_values.append(ssim_value)
            if validation_image_dir is not None:
                _write_validation_image_pair(
                    output_dir=validation_image_dir,
                    sample_idx=sample_idx,
                    output_rgb=fitted_rgb,
                    target_rgb=target_rgb,
                    resolution=view.resolution,
                )

    log.info(
        "ppisp.sh_match.optimization: validation samples=%d psnr_mean=%.4f "
        "psnr_min=%.4f psnr_max=%.4f ssim_mean=%s ssim_min=%s ssim_max=%s",
        num_samples,
        _mean(psnr_values),
        min(psnr_values),
        max(psnr_values),
        _format_optional_metric(_mean(ssim_values) if ssim_values else None),
        _format_optional_metric(min(ssim_values) if ssim_values else None),
        _format_optional_metric(max(ssim_values) if ssim_values else None),
    )


def _compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute scalar PSNR for RGB tensors in [0, 1]."""
    mse = F.mse_loss(pred, target)
    psnr = 10.0 * torch.log10(torch.tensor(1.0, device=mse.device, dtype=mse.dtype) / torch.clamp(mse, min=1e-10))
    return float(psnr.detach().item())


def _compute_ssim(pred: torch.Tensor, target: torch.Tensor, resolution: Tuple[int, int]) -> Optional[float]:
    """Compute SSIM when the flat RGB tensors can be reshaped to an image."""
    width, height = resolution
    if width * height != pred.shape[0] or pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 3:
        log.warning(
            "ppisp.sh_match.optimization: skipping SSIM for shape pred=%s target=%s resolution=%s",
            tuple(pred.shape),
            tuple(target.shape),
            resolution,
        )
        return None
    min_resolution = min(width, height)
    if min_resolution < 3:
        log.warning(
            "ppisp.sh_match.optimization: skipping SSIM for small resolution=%s",
            resolution,
        )
        return None

    from torchmetrics.image import StructuralSimilarityIndexMeasure

    kernel_size = min(11, min_resolution if min_resolution % 2 == 1 else min_resolution - 1)
    pred_image = _flat_rgb_to_bchw(pred, width=width, height=height)
    target_image = _flat_rgb_to_bchw(target, width=width, height=height)
    metric = StructuralSimilarityIndexMeasure(
        data_range=1.0,
        kernel_size=kernel_size,
        gaussian_kernel=False,
    ).to(pred_image.device)
    value = metric(pred_image, target_image)
    metric.reset()
    return float(value.detach().item())


def _flat_rgb_to_bchw(rgb: torch.Tensor, *, width: int, height: int) -> torch.Tensor:
    """Convert flat row-major RGB ``[H*W, 3]`` to ``[1, 3, H, W]``."""
    return rgb.reshape(height, width, 3).permute(2, 0, 1).unsqueeze(0).contiguous()


def _write_validation_image_pair(
    *,
    output_dir: Path,
    sample_idx: int,
    output_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    resolution: Tuple[int, int],
) -> None:
    width, height = resolution
    _write_flat_rgb_png(
        target_rgb,
        output_dir / f"sample_{sample_idx:04d}_target.png",
        width=width,
        height=height,
    )
    _write_flat_rgb_png(
        output_rgb,
        output_dir / f"sample_{sample_idx:04d}_output.png",
        width=width,
        height=height,
    )


def _write_flat_rgb_png(rgb: torch.Tensor, path: Path, *, width: int, height: int) -> None:
    if width * height != rgb.shape[0] or rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(
            f"cannot write validation image {path}: rgb shape={tuple(rgb.shape)} is incompatible with "
            f"resolution={(width, height)}"
        )

    from PIL import Image

    image = rgb.detach().clamp(0.0, 1.0).reshape(height, width, 3).mul(255.0).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(image, mode="RGB").save(path)


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values))


def _format_optional_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _resolve_hold_out_sample_for_validation(config: SHMatchingOptimizationConfig) -> int:
    env_value = os.environ.get(HOLD_OUT_SAMPLE_FOR_VALIDATION_ENV)
    if env_value is None:
        return int(config.hold_out_sample_for_validation)
    try:
        parsed = int(env_value)
    except ValueError as exc:
        raise ValueError(f"{HOLD_OUT_SAMPLE_FOR_VALIDATION_ENV} must be an int, got {env_value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{HOLD_OUT_SAMPLE_FOR_VALIDATION_ENV} must be >= 0, got {parsed}")
    if parsed != config.hold_out_sample_for_validation:
        log.info(
            "ppisp.sh_match.optimization: overriding hold_out_sample_for_validation=%d via %s=%d",
            config.hold_out_sample_for_validation,
            HOLD_OUT_SAMPLE_FOR_VALIDATION_ENV,
            parsed,
        )
    return parsed


def _resolve_validation_image_dir() -> Optional[Path]:
    env_value = os.environ.get(VALIDATION_IMAGE_DIR_ENV)
    if env_value is None:
        return None
    output_dir = Path(env_value)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("ppisp.sh_match.optimization: writing validation image pairs to %s", output_dir)
    return output_dir


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _validate_optimization_loop_inputs(
    model: object,
    adapter: object,
    sample_view: object,
    reference_render: object,
    live_render: object,
    seed: object,
    progress_log_every: object,
) -> None:
    """Fail-loud validation of every :func:`optimize_sh_to_match_ppisp` argument.

    Run before the optimizer is built so a bad call surfaces immediately,
    not several iterations into the loop.
    """
    if not isinstance(adapter, PPISPSHOptimizationAdapter):
        raise TypeError(f"adapter must be a PPISPSHOptimizationAdapter, got {type(adapter).__name__}")

    if not callable(getattr(model, "parameters", None)):
        raise TypeError(f"model must expose a parameters() iterator (nn.Module interface), got {type(model).__name__}")

    nodes = iter_gaussian_nodes(model)
    if not nodes:
        raise ValueError(
            "iter_gaussian_nodes returned no nodes for the optimization model; "
            "expected at least one SH-bearing node (composite or single)."
        )
    for node_id, node in nodes:
        for name in _TRAINABLE_PARAMETER_NAMES:
            if not hasattr(node, name):
                raise AttributeError(
                    f"gaussian node {node_id!r} is missing required SH-fit parameter {name!r}. "
                    "The SH refinement requires features_albedo, features_specular, and densities."
                )

    if not callable(sample_view):
        raise TypeError(f"sample_view must be callable, got {type(sample_view).__name__}")
    if not callable(reference_render):
        raise TypeError(f"reference_render must be callable, got {type(reference_render).__name__}")
    if not callable(live_render):
        raise TypeError(f"live_render must be callable, got {type(live_render).__name__}")
    if reference_render is live_render:
        raise ValueError(
            "reference_render and live_render must be distinct callables. "
            "Build them via make_sh_match_render_callables(model, snapshot)."
        )

    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError(f"seed must be int or None, got {type(seed).__name__}")

    if isinstance(progress_log_every, bool) or not isinstance(progress_log_every, int):
        raise TypeError(f"progress_log_every must be int, got {type(progress_log_every).__name__}")
    if progress_log_every < 0:
        raise ValueError(f"progress_log_every must be >= 0, got {progress_log_every}")
