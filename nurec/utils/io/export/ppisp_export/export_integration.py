# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PPISP integration helpers for the Gaussian USD asset exporter.

Hosts the cross-cutting decisions that the export orchestrator
(``gaussian_usd_asset``) and any CLI on top of it need to make
*before* they touch the heavyweight Click / system-loading code:

* Mode constants and validation: which PPISP integration mode the
  user requested and whether the surrounding flag combination is
  internally consistent.
* SH-match config resolution: which
  :class:`SHMatchingOptimizationConfig` to use for the bake -- the
  caller-supplied one if any, otherwise the dataclass defaults
  tuned for the algorithm.
* SH-match orchestration: a thin wrapper that wires the validated
  GaussiansSystem, PPISP module, and dataloader into the renderer-
  agnostic :func:`optimize_sh_to_match_ppisp` loop.
* SPG sidecar selection: which CUDA/Lua/USDA files to package
  alongside the USDZ archive depending on whether the exporter
  authored the static or the automatic-parameter (controller) PPISP
  prim on each RenderProduct.

These helpers are deliberately *public* (no underscore prefix) so
hermetic tests and any future CLI front-end can import them by name
without reaching into the orchestrator's private surface. The
orchestrator itself imports them and remains the single Click entry
point.
"""

from __future__ import annotations

import dataclasses
import logging
import math

from typing import Any, Optional, cast

from torch.utils.data import DataLoader

from nurec.utils.batch import batch_collate_fn
from nurec.utils.io.export.ppisp_export.sh_match.optimization import (
    PPISPSHOptimizationAdapter,
    SHFitTrainable,
    SHMatchingOptimizationConfig,
    optimize_sh_to_match_ppisp,
)
from nurec.utils.io.export.ppisp_export.sh_match.renderer_callable import (
    iterate_calibrated_batches,
    make_sh_match_render_callables,
    make_training_view_sampler,
    snapshot_sh_fit_parameters,
)
from nurec.utils.io.export.ppisp_export.spg_files import (
    PPISP_INTEGRATION_MODE_SPG_RUNTIME,
    ppisp_has_controller,  # noqa: F401  # re-exported for downstream importers
    resolve_ppisp_controller_export_enabled,  # noqa: F401  # re-exported for downstream importers
    select_spg_files_for_export,  # noqa: F401  # re-exported for downstream importers
)


log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Mode constants
# -----------------------------------------------------------------------------


PPISP_INTEGRATION_MODE_SH_OPTIMIZED = "sh-optimized"
VALID_PPISP_INTEGRATION_MODES = (
    PPISP_INTEGRATION_MODE_SPG_RUNTIME,
    PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def validate_export_integration_args(
    *,
    ppisp_integration_mode: str,
    scene_radiance_scale: float,
    ppisp_responsivity: float,
    sh_optimization_config: Optional[SHMatchingOptimizationConfig],
    enable_ppisp_controller_export: Optional[bool],
) -> None:
    """Fail-loud validation of the export integration arguments.

    The presence checks for ``ppisp_module`` (required by ``sh-optimized``
    and controller export) are deferred until the orchestrator has loaded
    the system, because PPISP availability is decided after checkpoint
    load.

    Raises:
        ValueError: ``ppisp_integration_mode`` is not in
            ``VALID_PPISP_INTEGRATION_MODES``, ``scene_radiance_scale`` /
            ``ppisp_responsivity`` are not strictly positive finite,
            ``ppisp_integration_mode`` is ``sh-optimized`` without an
            ``sh_optimization_config``, or explicit
            ``enable_ppisp_controller_export=True`` is set together
            with ``ppisp_integration_mode='sh-optimized'`` (incompatible:
            SH-match folds PPISP into Gaussian SH so the runtime
            controller has nothing to do).
    """
    if ppisp_integration_mode not in VALID_PPISP_INTEGRATION_MODES:
        raise ValueError(
            f"ppisp_integration_mode must be one of {VALID_PPISP_INTEGRATION_MODES!r}, got {ppisp_integration_mode!r}"
        )
    _require_positive_finite("scene_radiance_scale", scene_radiance_scale)
    _require_positive_finite("ppisp_responsivity", ppisp_responsivity)

    if ppisp_integration_mode == PPISP_INTEGRATION_MODE_SH_OPTIMIZED and sh_optimization_config is None:
        raise ValueError(
            "ppisp_integration_mode='sh-optimized' requires sh_optimization_config "
            "(SHMatchingOptimizationConfig instance). Pass a config to control the "
            "number of iterations, learning rates, and device."
        )
    if enable_ppisp_controller_export and ppisp_integration_mode == PPISP_INTEGRATION_MODE_SH_OPTIMIZED:
        raise ValueError(
            "enable_ppisp_controller_export is incompatible with "
            "ppisp_integration_mode='sh-optimized': SH-match folds PPISP into "
            "Gaussian SH coefficients, leaving no per-frame work for the runtime "
            "controller. Use spg-runtime mode for controller export."
        )


def _require_positive_finite(name: str, value: float) -> None:
    """Reject non-finite or non-positive numeric values.

    Internal helper kept private to this module: every call site is
    a field-level check inside :func:`validate_export_integration_args`,
    and the standalone signature is not part of the integration
    surface.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    # ``value > 0.0`` is False for NaN, -inf, 0.0 and negatives, so the only
    # non-finite case left is +inf; the second branch catches just that and
    # keeps the failure-mode-specific error messages distinct.
    if not (value > 0.0):
        raise ValueError(f"{name} must be strictly positive, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_positive_int(name: str, value: int) -> None:
    """Reject non-int or non-positive iteration counts.

    Bool is rejected explicitly because ``bool`` is an ``int``
    subclass in Python and would otherwise sneak through as ``1`` /
    ``0``, which is never what the caller meant.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive, got {value!r}")


# -----------------------------------------------------------------------------
# SH-match config resolution
# -----------------------------------------------------------------------------


def resolve_sh_optimization_config(
    *,
    explicit_config: Optional[SHMatchingOptimizationConfig],
    ppisp_integration_mode: str,
    num_iterations_override: Optional[int] = None,
) -> Optional[SHMatchingOptimizationConfig]:
    """Pick the effective SH-match config for this export run.

    The SH-match learning rates and device are tuned and not exposed
    on the CLI; only the iteration count is, because it is the
    knob users actually reach for when trading export time against
    SH-match quality. The resolver therefore picks between four
    states:

    1. Modes other than ``sh-optimized`` always return ``None``: SH
       optimization is a no-op there, and the orchestrator's
       fail-loud validation deliberately rejects an explicit config
       in those modes (avoids silent dead-code).
    2. If a Python caller provided an explicit config and no
       ``num_iterations_override``, use the explicit config as-is.
    3. If a ``num_iterations_override`` is provided, apply it on top
       of either the explicit config or a freshly-built default.
    4. Otherwise build a default
       :class:`SHMatchingOptimizationConfig` and let it pick the
       tuned algorithm defaults (3000 iterations, 3DGS-default LRs,
       device ``"cuda"``).

    Args:
        explicit_config: Python escape hatch carrying a fully
            specified :class:`SHMatchingOptimizationConfig`.
        ppisp_integration_mode: Selected PPISP integration mode.
        num_iterations_override: When set, replaces the
            ``num_iterations`` field of the resolved config. Must be
            a strictly positive ``int``. Setting this in a non
            ``sh-optimized`` mode is rejected so a CLI typo does not
            silently become a no-op.

    Raises:
        TypeError: ``num_iterations_override`` is not an ``int``.
        ValueError: ``num_iterations_override`` is not strictly
            positive, or is set while
            ``ppisp_integration_mode != 'sh-optimized'``.
    """
    if num_iterations_override is not None:
        _require_positive_int("sh_optimization_num_iterations", num_iterations_override)
        if ppisp_integration_mode != PPISP_INTEGRATION_MODE_SH_OPTIMIZED:
            raise ValueError(
                "sh_optimization_num_iterations is only meaningful when "
                "ppisp_integration_mode='sh-optimized'; got "
                f"ppisp_integration_mode={ppisp_integration_mode!r}."
            )

    if ppisp_integration_mode != PPISP_INTEGRATION_MODE_SH_OPTIMIZED:
        return None

    base = explicit_config if explicit_config is not None else SHMatchingOptimizationConfig()
    if num_iterations_override is None:
        return base
    return dataclasses.replace(base, num_iterations=num_iterations_override)


# -----------------------------------------------------------------------------
# SH-match orchestration
# -----------------------------------------------------------------------------


def run_sh_match_optimization(
    *,
    system: Any,
    ppisp_module: Any,
    train_dataset: Any,
    config: SHMatchingOptimizationConfig,
    reference_camera_id: int,
    reference_frame_id: int,
) -> None:
    """Run the PPISP SH-match optimization in place on ``system.model``.

    Wires the renderer-agnostic :func:`optimize_sh_to_match_ppisp`
    loop to the Gaussian system's *differentiable* camera renderer
    (``system.model.gaussians_renderer``) and the train-split dataset.

    1. Snapshot the current SH-fit parameters
       (``features_albedo``, ``features_specular``, ``densities``) per
       gaussian node. The snapshot becomes the frozen reference target
       the live SH is driven toward.
    2. Build a :class:`PPISPSHOptimizationAdapter` pinned to the
       resolved ``(camera_id, frame_id)`` pair and apply the SH DC
       warm-start so ``features_albedo`` lives in
       PPISP-display-referred space before the loop starts.
    3. Construct a calibrated training-view sampler: a small
       ``DataLoader`` over ``train_dataset`` wrapped by
       :func:`iterate_calibrated_batches` (which calls
       ``system.model.calib`` per batch to populate
       ``batch.rendering`` -- the export pipeline composes ``mode=val``
       before typed config parsing, so the system's training dataloader
       infrastructure is not available here).
    4. Build the ``(reference_render, live_render)`` callables off the
       same model: reference uses the snapshot via a ``param.data``
       swap (no ``deepcopy``); live uses the live SH and propagates
       gradients into it via the differentiable renderer.
    5. Run the multi-view Adam loop. The reference render is
       re-computed every iteration; no full reference cache is held
       in memory.

    Autograd context: the export CLI
    (:func:`nurec.utils.io.export.gaussian_usd_asset.export_gaussian_usd_asset`)
    must NOT be decorated with ``@torch.inference_mode()``. Inference
    mode permanently brands any tensor it creates as an *inference
    tensor*, and inference tensors cannot be in-place-mutated outside
    inference mode (``RuntimeError: Inplace update to inference tensor
    outside InferenceMode is not allowed``) nor participate in autograd
    (``RuntimeError: element 0 of tensors does not require grad``). Both
    properties would block the SH-match warm-start and optimization loop
    respectively. The CLI was historically decorated for inference-time
    perf; the decorator was removed when SH-match landed.

    Mutates ``system.model`` in place. Callers must run this *before*
    applying any radiance scaling so the scale layers on top of
    PPISP-folded coefficients, not the other way around.
    """
    log.info(
        "Running SH-match optimization (camera_id=%d, frame_id=%d, num_iterations=%d)",
        reference_camera_id,
        reference_frame_id,
        config.num_iterations,
    )

    adapter = PPISPSHOptimizationAdapter(
        config=config,
        ppisp_module=ppisp_module,
        camera_id=reference_camera_id,
        frame_id=reference_frame_id,
    )

    # Snapshot BEFORE the warm-start: the reference render must see the
    # original-checkpoint SH state, NOT the warm-started state. The
    # warm-start is an approximate DC-only fold of PPISP into the
    # albedo band; using it as the reference target degrades the
    # optimization signal (the live SH starts equal to the reference
    # and the loss is driven by the warm-start residual, not by the
    # difference between the live render and the full-fidelity
    # PPISP-applied original render). Matches the two-model pattern in
    # ``threedgrut/export/usd/post_processing_sh_bake.py`` where
    # ``reference_model = model`` (original) is rendered without
    # warm-start and ``baked_model = model.clone()`` is the warm-
    # started target of the optimizer.
    snapshot = snapshot_sh_fit_parameters(cast(Any, system.model))
    adapter.warmstart(cast(SHFitTrainable, system.model))
    reference_render, live_render = make_sh_match_render_callables(
        cast(Any, system.model),
        snapshot,
        is_training_batch=True,
    )

    raw_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        collate_fn=batch_collate_fn,
        pin_memory=False,
    )
    calibrated_loader = iterate_calibrated_batches(
        raw_loader,
        calib_fn=system.model.calib,
        device=adapter.device,
    )
    sample_view = make_training_view_sampler(calibrated_loader)

    optimize_sh_to_match_ppisp(
        model=cast(SHFitTrainable, system.model),
        adapter=adapter,
        sample_view=sample_view,
        reference_render=reference_render,
        live_render=live_render,
    )
