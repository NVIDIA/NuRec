# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PPISP -> SH DC Warm-Start

Initialize the DC band of a Gaussian model's spherical harmonic
coefficients by folding a single fixed PPISP transform into them, so
the asset can be rendered by runtimes that do not support an SPG-capable
PPISP shader.

This module *initializes* the DC band of ``features_albedo`` by passing
each Gaussian's diffuse RGB through PPISP at a fixed
``(camera_id, frame_id)`` with vignetting disabled. It deliberately
does NOT touch ``features_specular``: rotating the higher-order SH
bands by the per-Gaussian PPISP Jacobian (the ``higher_order=True``
path in the 3dgrut reference implementation) introduces
ill-conditioned outliers at PPISP saturation, which destabilize a
subsequent multi-view Adam refinement and empirically make the fit
converge worse than starting from the trained specular bands.

This warm-start by itself is *not* a complete export path: used alone
it sits ~7 dB PSNR below the multi-view-refined result on
representative scenes. It is intended as the starting point for an
Adam optimization that co-fits all SH bands and density (see
:mod:`nurec.utils.io.export.ppisp_export.sh_match.optimization`).

Vignetting compensation is intentionally omitted from the warm-start
target: per-Gaussian colors have no spatial coordinate at this stage,
and the reference rendering used by the downstream multi-view fit
also runs PPISP without vignetting to stay color-space-consistent
with this warm-start.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch

from ppisp import ppisp_apply  # type: ignore[import-not-found]

from nurec.models.gaussians.utils import RGB2SH, SH2RGB
from nurec.utils.io.export.gaussian_radiance_scaling import SHCoefficientsBearer


if TYPE_CHECKING:
    from ppisp import PPISP  # type: ignore[import-not-found]


log = logging.getLogger(__name__)

DeviceLike = Union[torch.device, str]


def warmstart_sh_dc_from_ppisp(
    model: SHCoefficientsBearer,
    ppisp_module: "PPISP",
    camera_id: int,
    frame_id: int,
    device: Optional[DeviceLike] = None,
) -> Tuple[float, torch.Tensor]:
    """In-place initialize ``model.features_albedo`` by baking PPISP into the SH DC band.

    Each Gaussian's DC RGB color is decoded from SH, fed through PPISP at
    the requested ``(camera_id, frame_id)`` (vignetting suppressed,
    pixel coordinates set to the optical center), then re-encoded with
    ``RGB2SH`` and copied back into ``features_albedo``. The specular
    bands are left untouched.

    Args:
        model: Object exposing ``features_albedo`` (shape ``[N, 3]``)
            and ``features_specular`` (shape ``[N, num_specular_dims]``)
            as ``torch.Tensor`` attributes. In production this is an
            ``SHGaussianModel`` or one of its subclasses; the
            ``SHCoefficientsBearer`` protocol documents the minimal
            contract.
        ppisp_module: Trained PPISP module providing
            ``num_cameras``, ``exposure_params``, ``color_params``,
            ``vignetting_params``, and ``crf_params``.
        camera_id: PPISP camera index whose CRF will be applied. Must
            satisfy ``0 <= camera_id < ppisp_module.num_cameras``.
        frame_id: PPISP frame index whose exposure/color params will be
            applied. Must satisfy
            ``0 <= frame_id < ppisp_module.exposure_params.shape[0]``.
        device: Optional device on which to run PPISP. Defaults to the
            device of ``model.features_albedo``. The albedo tensor is
            written back on its original device.

    Returns:
        ``(exposure_offset, color_params)`` actually used for the bake.
        ``exposure_offset`` is a Python ``float``; ``color_params`` is a
        detached ``torch.Tensor`` of shape ``[8]`` on the bake device.
        Returning these makes downstream logging and round-trip
        assertions explicit.

    Raises:
        TypeError: ``features_albedo`` / ``features_specular`` not a
            ``torch.Tensor``.
        ValueError: ``camera_id`` or ``frame_id`` out of range, or
            ``features_albedo`` not shape ``[N, 3]``.
        AttributeError: ``model`` missing required SH attributes, or
            ``ppisp_module`` missing required parameter tensors.
    """
    _validate_warmstart_inputs(model, ppisp_module, camera_id, frame_id)

    bake_device: torch.device = torch.device(device) if device is not None else model.features_albedo.device
    exposure, color = _get_fixed_frame_params(ppisp_module, frame_id)

    with torch.no_grad():
        dc_rgb_linear = SH2RGB(model.features_albedo.detach().to(bake_device))
        dc_rgb_baked = _apply_ppisp_to_dc_colors(
            dc_rgb_linear=dc_rgb_linear,
            ppisp_module=ppisp_module,
            camera_id=camera_id,
            exposure=exposure,
            color=color,
            device=bake_device,
        )
        model.features_albedo.copy_(RGB2SH(dc_rgb_baked).to(model.features_albedo.device))

    log.info(
        "DC warm-start applied: camera_id=%d, frame_id=%d, exposure=%.4f, num_gaussians=%d (specular bands preserved).",
        camera_id,
        frame_id,
        exposure,
        model.features_albedo.shape[0],
    )
    return exposure, color


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


def _validate_warmstart_inputs(
    model: object,
    ppisp_module: object,
    camera_id: object,
    frame_id: object,
) -> None:
    """Fail-loud validation for :func:`warmstart_sh_dc_from_ppisp` inputs.

    Validation order is fixed (model attributes -> ppisp attributes ->
    indices) so error messages are stable across releases.
    """
    if isinstance(camera_id, bool) or not isinstance(camera_id, int):
        raise TypeError(f"camera_id must be int, got {type(camera_id).__name__}")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int):
        raise TypeError(f"frame_id must be int, got {type(frame_id).__name__}")

    for attr_name in ("features_albedo", "features_specular"):
        if not hasattr(model, attr_name):
            raise AttributeError(
                f"model is missing required SH parameter '{attr_name}'. "
                "warmstart_sh_dc_from_ppisp requires an SHGaussianModel-like "
                "model exposing features_albedo and features_specular."
            )
        param = getattr(model, attr_name)
        if not isinstance(param, torch.Tensor):
            raise TypeError(f"model.{attr_name} must be a torch.Tensor, got {type(param).__name__}")

    features_albedo = model.features_albedo  # type: ignore[attr-defined]
    if features_albedo.ndim != 2 or features_albedo.shape[1] != 3:
        raise ValueError(
            f"model.features_albedo must have shape [N, 3] (DC RGB band), got {tuple(features_albedo.shape)}"
        )

    for attr_name in ("num_cameras", "exposure_params", "color_params", "vignetting_params", "crf_params"):
        if not hasattr(ppisp_module, attr_name):
            raise AttributeError(
                f"ppisp_module is missing required attribute '{attr_name}'. "
                "warmstart_sh_dc_from_ppisp requires a trained PPISP module."
            )

    num_cameras = int(ppisp_module.num_cameras)  # type: ignore[attr-defined]
    if camera_id < 0 or camera_id >= num_cameras:
        raise ValueError(f"camera_id must be in [0, {num_cameras - 1}], got {camera_id}")

    exposure_params = ppisp_module.exposure_params  # type: ignore[attr-defined]
    if not isinstance(exposure_params, torch.Tensor):
        raise TypeError(f"ppisp_module.exposure_params must be a torch.Tensor, got {type(exposure_params).__name__}")
    num_frames = int(exposure_params.shape[0])
    if frame_id < 0 or frame_id >= num_frames:
        raise ValueError(f"frame_id must be in [0, {num_frames - 1}], got {frame_id}")


def _get_fixed_frame_params(ppisp_module: "PPISP", frame_id: int) -> Tuple[float, torch.Tensor]:
    """Return the ``(exposure, color)`` pair for a single PPISP frame.

    Reads ``ppisp_module.exposure_params[frame_id]`` and
    ``ppisp_module.color_params[frame_id]`` and detaches the color
    tensor so it cannot be backpropagated through.
    """
    exposure = float(ppisp_module.exposure_params[frame_id].item())
    color = ppisp_module.color_params[frame_id].detach()
    return exposure, color


def _apply_ppisp_to_dc_colors(
    dc_rgb_linear: torch.Tensor,
    ppisp_module: "PPISP",
    camera_id: int,
    exposure: float,
    color: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Apply PPISP to per-Gaussian DC RGB colors, with vignetting disabled.

    Builds single-frame parameter tensors (exposure of shape ``[1]``,
    color of shape ``[1, 8]``) so ``ppisp_apply`` can be called with
    ``frame_idx=0`` regardless of the source ``frame_id`` in the
    trained module. Vignetting parameters are zeroed and pixel
    coordinates are pinned to the optical center ``(0, 0)`` because
    per-Gaussian colors have no spatial location at warm-start time.
    """
    dtype = dc_rgb_linear.dtype
    num_gaussians = dc_rgb_linear.shape[0]

    exposure_params = torch.tensor([exposure], device=device, dtype=dtype)
    color_params = color.to(device=device, dtype=dtype).unsqueeze(0)
    vignetting_params = torch.zeros_like(ppisp_module.vignetting_params, device=device, dtype=dtype)
    crf_params = ppisp_module.crf_params.to(device=device, dtype=dtype)
    pixel_coords = torch.zeros(num_gaussians, 2, device=device, dtype=dtype)

    return ppisp_apply(
        exposure_params=exposure_params,
        vignetting_params=vignetting_params,
        color_params=color_params,
        crf_params=crf_params,
        rgb_in=dc_rgb_linear.contiguous(),
        pixel_coords=pixel_coords,
        resolution_w=1,
        resolution_h=1,
        camera_idx=camera_id,
        frame_idx=0,
    )
