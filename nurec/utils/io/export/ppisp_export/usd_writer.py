# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PPISP USD Writer

Export PPISP (Physically Plausible Image Signal Processing) shader to USD RenderProducts.

This module provides functions to add PPISP post-processing shaders to USD RenderProducts
for each camera, enabling ISP pipeline rendering in USD-compatible renderers.

PPISP pipeline stages:
1. Exposure compensation (per-frame)
2. Vignetting correction (per-camera, per-channel)
3. Color correction via ZCA-based homography (per-frame)
4. Camera Response Function (per-camera, per-channel, 4-param toe-shoulder)
"""

from __future__ import annotations

import logging
import math

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from nurec.utils.io.export.ppisp_export.controller.weights import select_camera_controller
from nurec.utils.io.export.ppisp_export.controller.writer import (
    add_ppisp_auto_shader_to_render_product,
)
from nurec.utils.io.export.ppisp_export.reference_selection import (
    ResolvedPPISPReferenceIds,
    resolve_ppisp_reference_ids,
)
from nurec.utils.types import NamedSerialized


if TYPE_CHECKING:
    from ppisp import PPISP  # type: ignore[import-not-found]

    from nurec.utils.types import RigTrajectories


log = logging.getLogger(__name__)

NUM_CHANNELS = 3
COLOR_PARAMS_PER_FRAME = 8
CRF_PARAMS_PER_CHANNEL = 4

CHANNEL_SUFFIXES = ["R", "G", "B"]

DEFAULT_PPISP_RESPONSIVITY = 1.0
PPISP_RESPONSIVITY_INPUT_NAME = "responsivity"

# Suffix for the per-RenderProduct PPISP camera (formerly ``_no_isp``).
# The camera now carries every PPISP parameter as a namespaced
# ``ppisp:*`` attribute (the source of truth) in addition to the
# neutralized ``exposure:*`` values, so the name reflects its broader role.
PPISP_CAMERA_SUFFIX = "_ppisp"
# Namespace prefix for the source-of-truth PPISP attributes authored on
# the ``<cam>_ppisp`` camera. Shader inputs connect to ``<camera>.ppisp:<input>``.
PPISP_ATTR_NAMESPACE = "ppisp:"

# Tiled-RenderProduct grid. Tiling is a RenderProduct property (Kit derives
# the live grid at runtime from the RenderProduct ``camera`` target count and
# the ``omni:rtx:viewTile:*`` render settings), so it is authored as constant
# ``inputs:tileCount{X,Y}`` directly on the PPISP shader nodes -- not on the
# camera. The exporter authors the untiled default (1x1); a downstream
# consumer assembling a multi-camera tiled product overrides these shader
# inputs to match ``viewTilesPerDimension`` (the controller buffers are sized
# for the full ``tileCountX * tileCountY`` grid).
PPISP_TILE_COUNT_INPUT_NAMES = ("tileCountX", "tileCountY")
DEFAULT_PPISP_TILE_COUNT = 1

# Static-path PPISP SPG sidecars live in the ppisp_export/spg_sidecars/
# subpackage (carved out of the flat export/ directory in the
# Task 4 refactor commit). Basenames are preserved with the
# ``ppisp_`` prefix because they ship inside the published USDZ
# archive and must stay self-documenting to downstream consumers.
PPISP_SPG_DIR = Path(__file__).parent / "spg_sidecars"
PPISP_SPG_CU_FILE = "ppisp_usd_spg.cu"
PPISP_SPG_LUA_FILE = "ppisp_usd_spg.cu.lua"
PPISP_SPG_USDA_FILE = "ppisp_usd_spg.usda"

# Shared RenderVar names for SPG pipeline
PPISP_INPUT_RENDER_VAR = "HdrColor"  # Input AOV name
PPISP_OUTPUT_RENDER_VAR = "PPISPColor"  # Output AOV name
LDR_COLOR_RENDER_VAR = "LdrColor"  # Display AOV wired to PPISP result


@dataclass
class CameraRenderProductInfo:
    """Camera information needed to create a RenderProduct."""

    camera_name: str
    rig_name: str
    width: int
    height: int


def extract_camera_render_products(rig_trajectories: RigTrajectories) -> List[CameraRenderProductInfo]:
    """Extract camera RenderProduct information from rig trajectories."""
    camera_render_products: List[CameraRenderProductInfo] = []

    for i, rig_trajectory in enumerate(rig_trajectories.rig_trajectories):
        rig_name = f"sensor_rig_{i}"
        for camera_unique_name in rig_trajectory.cameras_frame_timestamps_us:
            camera_data = rig_trajectories.camera_calibrations[camera_unique_name]
            camera_name = camera_data.logical_sensor_name
            camera_model = camera_data.camera_model_parameters

            resolution = getattr(camera_model, "resolution", None)
            if resolution is not None:
                resolution_list = resolution.tolist()
                width, height = int(resolution_list[0]), int(resolution_list[1])
            else:
                width, height = 1920, 1080
                log.warning(
                    "Camera %r has no resolution in camera_model_parameters; defaulting to %dx%d. "
                    "This usually means the source calibration is missing display-size metadata; "
                    "the RenderProduct will be sized accordingly.",
                    camera_unique_name,
                    width,
                    height,
                )

            camera_render_products.append(
                CameraRenderProductInfo(
                    camera_name=camera_name,
                    rig_name=rig_name,
                    width=width,
                    height=height,
                )
            )

    return camera_render_products


def add_camera_render_products(
    *,
    stage: Usd.Stage,
    camera_render_products: List[CameraRenderProductInfo],
    has_runtime_ppisp: bool,
    render_scope_path: str = "/Render",
    camera_root_path: str = "/World/rig_trajectories",
) -> None:
    """Create camera RenderProducts using the AOV expected by the PPISP path."""
    if not camera_render_products:
        return

    stage.DefinePrim(render_scope_path, "Scope")
    for cam_info in camera_render_products:
        camera_path = f"{camera_root_path}/{cam_info.rig_name}/{cam_info.camera_name}"
        render_product_path = f"{render_scope_path}/{cam_info.camera_name}"
        render_product = stage.DefinePrim(render_product_path, "RenderProduct")
        render_product.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(
            Gf.Vec2i(cam_info.width, cam_info.height)
        )
        render_product.CreateRelationship("camera").SetTargets([Sdf.Path(camera_path)])

        render_var_name = PPISP_INPUT_RENDER_VAR if has_runtime_ppisp else LDR_COLOR_RENDER_VAR
        render_var_path = f"{render_product_path}/{render_var_name}"
        render_var = stage.DefinePrim(render_var_path, "RenderVar")
        render_var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(render_var_name)
        render_product.CreateRelationship("orderedVars").SetTargets([Sdf.Path(render_var_path)])


def _add_ldr_color_render_var(stage: Usd.Stage, render_product_path: str, ppisp_output_path: Sdf.Path) -> str:
    """Create a LdrColor RenderVar wired to the PPISP output (omni:rtx:aov connects to PPISP.outputs:PPISPColor).
    Returns the path to the created RenderVar for appending to orderedVars.

    The ``omni:rtx:aov`` attribute MUST be authored as a non-custom (schema)
    attribute (``custom=False``); Kit's SPG reflection layer looks up the
    binding through the schema-defined ``omni:rtx:aov`` and silently skips
    custom attributes, which manifests as
    ``Failed to find binding index for output 'outputs:PPISPColor'``
    at runtime.
    """
    ldr_var_path = f"{render_product_path}/{LDR_COLOR_RENDER_VAR}"
    ldr_var = stage.DefinePrim(ldr_var_path, "RenderVar")
    ldr_var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(LDR_COLOR_RENDER_VAR)
    aov_attr = ldr_var.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)
    aov_attr.SetConnections([ppisp_output_path])
    return ldr_var_path


def _create_shader_prim(stage: Usd.Stage, render_product_path: str) -> UsdShade.Shader:
    """Create the PPISP Shader prim on a RenderProduct with input/output wiring and LdrColor RenderVar.

    Returns the UsdShade.Shader for parameter setting.
    """
    render_product = stage.GetPrimAtPath(render_product_path)
    if not render_product.IsValid():
        raise ValueError(f"RenderProduct not found at path: {render_product_path}")

    # Add opaque omni:rtx:aov to input RenderVar (no connection, just declared).
    # Must be non-custom (``custom=False``); Kit's SPG reflection looks up
    # the binding via the schema-defined ``omni:rtx:aov`` attribute, and
    # silently misses custom attributes, surfacing as
    # ``Failed to find binding index for input 'inputs:HdrColor'``.
    input_var_path = f"{render_product_path}/{PPISP_INPUT_RENDER_VAR}"
    input_var_prim = stage.GetPrimAtPath(input_var_path)
    if input_var_prim.IsValid():
        input_var_prim.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)

    # Create PPISP Shader via UsdShade and reference SPG definition
    ppisp_shader_path = f"{render_product_path}/PPISP"
    shader = UsdShade.Shader.Define(stage, ppisp_shader_path)
    shader.GetPrim().GetReferences().AddReference(PPISP_SPG_USDA_FILE)
    # Duplicate the source metadata on the instance. Some Kit SPG/Fabric paths
    # do not resolve referenced shader metadata when opening packaged USDZ files.
    shader.GetPrim().CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token, custom=False).Set(
        "sourceAsset"
    )
    shader.GetPrim().CreateAttribute("info:spg:sourceAsset", Sdf.ValueTypeNames.Asset, custom=False).Set(
        Sdf.AssetPath(PPISP_SPG_CU_FILE)
    )
    shader.GetPrim().CreateAttribute("info:spg:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token, custom=False).Set(
        "ppispProcess"
    )

    # Connect HdrColor input to input RenderVar (Omniverse-specific; no UsdShade on RenderVar)
    hdr_input = shader.CreateInput(PPISP_INPUT_RENDER_VAR, Sdf.ValueTypeNames.Opaque)
    hdr_input.GetAttr().SetConnections([Sdf.Path(f"../{PPISP_INPUT_RENDER_VAR}.omni:rtx:aov")])

    # Declare output for LdrColor to connect to
    shader.CreateOutput(PPISP_OUTPUT_RENDER_VAR, Sdf.ValueTypeNames.Opaque)

    # LdrColor RenderVar wired to PPISP result (display AOV)
    ppisp_output_path = shader.GetPath().AppendProperty(f"outputs:{PPISP_OUTPUT_RENDER_VAR}")
    ldr_var_path = _add_ldr_color_render_var(stage, render_product_path, ppisp_output_path)

    # Update orderedVars to include LdrColor
    ordered_vars_rel = render_product.GetRelationship("orderedVars")
    if ordered_vars_rel:
        existing_targets = list(ordered_vars_rel.GetTargets())
        existing_targets.append(Sdf.Path(ldr_var_path))
        ordered_vars_rel.SetTargets(existing_targets)

    return shader


def add_ppisp_shader_to_render_product(
    stage: Usd.Stage,
    render_product_path: str,
    camera_index: int,
    ppisp: PPISP,
    frame_indices: List[int],
    responsivity: float = DEFAULT_PPISP_RESPONSIVITY,
    *,
    resolved_reference_ids: Optional[ResolvedPPISPReferenceIds] = None,
    ppisp_camera_path: Optional[Sdf.Path] = None,
) -> Usd.Prim:
    """Add a PPISP shader to a RenderProduct for a specific camera.

    Per-camera parameters (vignetting, CRF) are always static (per-camera).
    Per-frame parameters (exposure, color) are dispatched based on
    ``resolved_reference_ids``:

    * ``None`` (default): time-sampled authoring -- one USD time sample
      per training frame plus a per-camera mean as the default value.
      Preserves backward-compatible behavior.
    * ``camera_was_user_set=True``, ``frame_was_user_set=False``: static
      single-frame authoring with neutral exposure (``0.0``) and identity
      color latents (zeros). Vignetting and CRF still come from the
      configured ``camera_index``.
    * ``camera_was_user_set=True``, ``frame_was_user_set=True``: static
      single-frame authoring with exposure / color latents pulled from
      ``ppisp.exposure_params[frame_id]`` / ``ppisp.color_params[frame_id]``.

    The fourth row of the resolution table (``frame_only`` without
    ``camera_id``) is rejected upstream by
    :func:`resolve_ppisp_reference_ids`; this function therefore never
    has to handle it.

    Args:
        stage: USD stage containing the RenderProduct.
        render_product_path: Path to the RenderProduct prim.
        camera_index: Index of the camera in the PPISP model. When
            ``resolved_reference_ids`` is set, callers must pass
            ``resolved_reference_ids.camera_id`` here so vignetting /
            CRF and exposure / color all source from the same camera.
        ppisp: PPISP module containing the ISP parameters.
        frame_indices: Training frame indices belonging to this camera.
            Used only by the time-sampled path; ignored when
            ``resolved_reference_ids`` is set (the resolved frame_id
            takes over).
        responsivity: Achromatic HDR multiplier authored on the shader's
            ``inputs:responsivity`` attribute. Applied inside the SPG shader
            before exposure; orthogonal to the asset-baked radiance scale.
            Must be a strictly positive finite number.
        resolved_reference_ids: Resolved PPISP reference (camera, frame)
            selection. ``None`` selects the time-sampled path.
        ppisp_camera_path: Optional path to the ``<cam>_ppisp`` camera that
            holds the PPISP parameters as ``ppisp:*`` attributes. When set,
            the parameter values are authored on the camera (the source of
            truth required by other workflows). The shader inputs themselves
            are currently hardcoded as literal ``inputs:*`` (the same as the
            ``None`` path); connecting them to the camera ``ppisp:*`` attrs
            is disabled pending Kit support for resolving dynamic params from
            the bound RenderProduct camera (see the ``TODO(kit)`` in-body).

    Returns:
        The created PPISP Shader prim.
    """
    _validate_ppisp_responsivity(responsivity)
    assert camera_index < ppisp.num_cameras, f"camera_index {camera_index} out of bounds"
    if resolved_reference_ids is None:
        assert frame_indices, "frame_indices must not be empty; no training frames available for PPISP export"

    shader = _create_shader_prim(stage, render_product_path)

    # Tiling is a RenderProduct property: author it as a constant shader
    # input regardless of the camera-source-of-truth path below.
    _set_tile_count_params(shader)

    if ppisp_camera_path is not None:
        # Author the ppisp:* attributes on the <cam>_ppisp camera. They are
        # the source of truth required by other workflows. Value dispatch
        # (time-sampled / static / neutral) lives in
        # _author_ppisp_camera_attributes.
        _author_ppisp_camera_attributes(
            stage,
            ppisp_camera_path,
            ppisp,
            camera_index,
            responsivity=responsivity,
            resolved_reference_ids=resolved_reference_ids,
            frame_indices=frame_indices,
        )
        # TODO(kit): connect the shader inputs to the camera ppisp:* attrs
        # (via _connect_ppisp_shader_inputs) instead of hardcoding the
        # literals below, once SPG can resolve dynamic params from the bound
        # RenderProduct camera. Disabled for now: SPG bakes the connection's
        # literal source path and reads it from Fabric each frame
        # (FabricToSpgTranslator), so a connection does not follow a change
        # of the RenderProduct's bound camera. The camera attrs above are
        # still authored because other workflows rely on them.
        # _connect_ppisp_shader_inputs(shader, ppisp_camera_path)

    # Hardcode the PPISP params as literal shader inputs. This runs for both
    # the camera path (see TODO above) and the back-compat no-camera path.
    _set_responsivity_params(shader, responsivity)
    _set_vignetting_params(shader, ppisp, camera_index)
    _set_crf_params(shader, ppisp, camera_index)

    if resolved_reference_ids is None:
        _set_animated_exposure_params(shader, ppisp, frame_indices)
        _set_animated_color_params(shader, ppisp, frame_indices)
        log.info(
            "Added PPISP shader to %s for camera %d (time-sampled, %d frames)",
            render_product_path,
            camera_index,
            len(frame_indices),
        )
    else:
        # Single-frame authoring; both branches ensure deterministic
        # static attributes regardless of caller's frame_indices list.
        if resolved_reference_ids.frame_was_user_set:
            _set_static_exposure_params(shader, ppisp, resolved_reference_ids.frame_id)
            _set_static_color_params(shader, ppisp, resolved_reference_ids.frame_id)
            log.info(
                "Added PPISP shader to %s for camera %d (static frame=%d)",
                render_product_path,
                camera_index,
                resolved_reference_ids.frame_id,
            )
        else:
            _set_neutral_exposure_params(shader)
            _set_identity_color_params(shader)
            log.info(
                "Added PPISP shader to %s for camera %d (camera-only, neutral exposure / identity color)",
                render_product_path,
                camera_index,
            )

    return shader.GetPrim()


# -----------------------------------------------------------------------------
# Static Parameter Setters
# -----------------------------------------------------------------------------


def _validate_ppisp_responsivity(responsivity: object) -> None:
    """Fail-loud validation of the ``responsivity`` argument.

    Rejects non-numeric, ``bool``, non-finite, and non-positive values with
    explicit ``TypeError`` / ``ValueError``. Validation is centralized so
    that both public entrypoints surface identical messages.
    """
    if isinstance(responsivity, bool) or not isinstance(responsivity, (int, float)):
        raise TypeError(f"responsivity must be a real number (int or float), got {type(responsivity).__name__}")
    if not math.isfinite(float(responsivity)):
        raise ValueError(f"responsivity must be finite, got {responsivity!r}")
    if float(responsivity) <= 0.0:
        raise ValueError(f"responsivity must be strictly positive, got {responsivity!r}")


def _set_responsivity_params(shader: UsdShade.Shader, responsivity: float) -> None:
    """Author the per-shader ``inputs:responsivity`` attribute.

    The responsivity is an achromatic HDR multiplier consumed inside the
    PPISP SPG shader. It is the runtime, USD-overridable counterpart to the
    asset-baked radiance scaling on the Gaussian SH coefficients (see
    :mod:`nurec.utils.io.export.gaussian_radiance_scaling`).

    Args:
        shader: PPISP shader prim wrapper.
        responsivity: Strictly positive finite multiplier. Already validated
            by :func:`_validate_ppisp_responsivity` at the public entrypoint.
    """
    shader.CreateInput(PPISP_RESPONSIVITY_INPUT_NAME, Sdf.ValueTypeNames.Float).Set(float(responsivity))


def _set_vignetting_params(shader: UsdShade.Shader, ppisp: PPISP, camera_index: int) -> None:
    """Set per-camera vignetting parameters (static).

    Accesses ppisp.vignetting_params[camera_index] with shape [3, 5] = [cx, cy, alpha1, alpha2, alpha3] per channel.
    """
    vig = ppisp.vignetting_params[camera_index].detach().cpu().numpy()  # [3, 5]

    for ch in range(NUM_CHANNELS):
        s = CHANNEL_SUFFIXES[ch]
        shader.CreateInput(f"vignettingCenter{s}", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(vig[ch, 0]), float(vig[ch, 1]))
        )
        shader.CreateInput(f"vignettingAlpha1{s}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 2]))
        shader.CreateInput(f"vignettingAlpha2{s}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 3]))
        shader.CreateInput(f"vignettingAlpha3{s}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 4]))


def _set_crf_params(shader: UsdShade.Shader, ppisp: PPISP, camera_index: int) -> None:
    """Set per-camera CRF raw parameters (static).

    Accesses ppisp.crf_params[camera_index] with shape [3, 4] = [toe, shoulder, gamma, center] per channel.
    """
    crf = ppisp.crf_params[camera_index].detach().cpu().numpy()  # [3, 4]

    for ch in range(NUM_CHANNELS):
        s = CHANNEL_SUFFIXES[ch]
        shader.CreateInput(f"crfToe{s}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 0]))
        shader.CreateInput(f"crfShoulder{s}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 1]))
        shader.CreateInput(f"crfGamma{s}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 2]))
        shader.CreateInput(f"crfCenter{s}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 3]))


# -----------------------------------------------------------------------------
# Animated Parameter Setters (Time-Sampled with Mean Defaults)
# -----------------------------------------------------------------------------


def _set_animated_exposure_params(
    shader: UsdShade.Shader,
    ppisp: PPISP,
    frame_indices: List[int],
) -> None:
    """Write time-sampled exposure offset with per-camera mean as default.

    The default value (no time code) is set to the mean exposure across this camera's
    training frames, so USD consumers that don't evaluate at a specific time code
    get a representative value.

    Time code for each sample is ``float(training_frame_idx)``. Stage
    ``timeCodesPerSecond`` therefore controls real-time playback speed.

    Args:
        shader: PPISP shader (UsdShade) to write attributes to
        ppisp: PPISP module containing exposure parameters
        frame_indices: Training frame indices belonging to this camera.
    """
    exposure = ppisp.exposure_params.detach().cpu().numpy()  # [num_frames]
    valid = [i for i in frame_indices if i < len(exposure)]

    mean_exposure = float(np.mean(exposure[valid])) if valid else 0.0
    exposure_input = shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float)
    exposure_attr = exposure_input.GetAttr()
    exposure_attr.Set(mean_exposure)

    for frame_idx in frame_indices:
        if frame_idx >= len(exposure):
            log.warning(
                f"PPISP frame index {frame_idx} out of bounds for exposure params (length {len(exposure)}), skipping"
            )
            continue
        exposure_attr.Set(float(exposure[frame_idx]), float(frame_idx))


def _set_animated_color_params(
    shader: UsdShade.Shader,
    ppisp: PPISP,
    frame_indices: List[int],
) -> None:
    """Write time-sampled color latent offsets with per-camera mean as default.

    Color params are [num_frames, 8] = [db_r, db_g, dr_r, dr_g, dg_r, dg_g, dgray_r, dgray_g].
    Written as 4 float2 attributes: colorLatentBlue, colorLatentRed, colorLatentGreen, colorLatentNeutral.

    Time code for each sample is ``float(training_frame_idx)``.

    Args:
        shader: PPISP shader (UsdShade) to write attributes to
        ppisp: PPISP module containing color parameters
        frame_indices: Training frame indices belonging to this camera.
    """
    color = ppisp.color_params.detach().cpu().numpy()  # [num_frames, 8]
    valid = [i for i in frame_indices if i < len(color)]

    mean_color = np.mean(color[valid], axis=0) if valid else np.zeros(COLOR_PARAMS_PER_FRAME)

    control_point_names = ["colorLatentBlue", "colorLatentRed", "colorLatentGreen", "colorLatentNeutral"]
    attrs: list[Usd.Attribute] = []
    for i, name in enumerate(control_point_names):
        color_input = shader.CreateInput(name, Sdf.ValueTypeNames.Float2)
        attr = color_input.GetAttr()
        attr.Set(Gf.Vec2f(float(mean_color[i * 2]), float(mean_color[i * 2 + 1])))
        attrs.append(attr)

    for frame_idx in frame_indices:
        if frame_idx >= len(color):
            log.warning(f"PPISP frame index {frame_idx} out of bounds for color params (length {len(color)}), skipping")
            continue
        frame_color = color[frame_idx]
        for i, attr in enumerate(attrs):
            attr.Set(Gf.Vec2f(float(frame_color[i * 2]), float(frame_color[i * 2 + 1])), float(frame_idx))


# -----------------------------------------------------------------------------
# Static (Single-Frame) Parameter Setters
# -----------------------------------------------------------------------------


_COLOR_LATENT_INPUT_NAMES = ("colorLatentBlue", "colorLatentRed", "colorLatentGreen", "colorLatentNeutral")


def _set_static_exposure_params(shader: UsdShade.Shader, ppisp: PPISP, frame_id: int) -> None:
    """Author a single non-time-sampled exposure offset attribute.

    Used when a caller fixes the SPG export to the PPISP configuration
    of a specific ``(camera_id, frame_id)`` pair: the shader runs with
    a constant exposure regardless of USD playhead time.
    """
    exposure = ppisp.exposure_params.detach().cpu().numpy()
    if frame_id >= len(exposure):
        raise ValueError(f"frame_id {frame_id} out of bounds for exposure_params of length {len(exposure)}")
    shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).Set(float(exposure[frame_id]))


def _set_static_color_params(shader: UsdShade.Shader, ppisp: PPISP, frame_id: int) -> None:
    """Author the four color-latent attributes from a single frame.

    The 8-D color parameter vector at ``frame_id`` is split into four
    ``float2`` attributes whose names match the SPG shader's
    ``inputs:colorLatent*`` declarations.
    """
    color = ppisp.color_params.detach().cpu().numpy()
    if frame_id >= len(color):
        raise ValueError(f"frame_id {frame_id} out of bounds for color_params of length {len(color)}")
    frame_color = color[frame_id]
    for i, name in enumerate(_COLOR_LATENT_INPUT_NAMES):
        shader.CreateInput(name, Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(frame_color[i * 2]), float(frame_color[i * 2 + 1]))
        )


def _set_neutral_exposure_params(shader: UsdShade.Shader) -> None:
    """Author exposure offset = 0.0 (neutral) for camera-only fixing.

    Used when the caller fixes ``ppisp_reference_camera_id`` but leaves
    ``ppisp_reference_frame_id`` unset: vignetting and CRF still come
    from the camera, but exposure is neutralized so the choice of
    training frame does not implicitly leak into the SPG export.
    """
    shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def _set_identity_color_params(shader: UsdShade.Shader) -> None:
    """Author identity color-latent attributes (all zeros).

    Companion to :func:`_set_neutral_exposure_params`: zero color
    latents map to the identity ZCA homography inside the SPG shader,
    so the camera-only fixing leaves color correction off.
    """
    for name in _COLOR_LATENT_INPUT_NAMES:
        shader.CreateInput(name, Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.0, 0.0))


# -----------------------------------------------------------------------------
# Camera Source-of-Truth Authoring (``<cam>_ppisp.ppisp:*`` attributes)
# -----------------------------------------------------------------------------


def _ppisp_camera_attr(camera_prim: Usd.Prim, input_name: str, value_type: Sdf.ValueTypeName) -> Usd.Attribute:
    """Create (or fetch) a namespaced ``ppisp:<input_name>`` custom attribute."""
    return camera_prim.CreateAttribute(f"{PPISP_ATTR_NAMESPACE}{input_name}", value_type, custom=True)


def _author_camera_vignetting(camera_prim: Usd.Prim, ppisp: PPISP, camera_index: int) -> None:
    """Author per-camera vignetting parameters as ``ppisp:*`` attributes."""
    vig = ppisp.vignetting_params[camera_index].detach().cpu().numpy()  # [3, 5]
    for ch in range(NUM_CHANNELS):
        s = CHANNEL_SUFFIXES[ch]
        _ppisp_camera_attr(camera_prim, f"vignettingCenter{s}", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(vig[ch, 0]), float(vig[ch, 1]))
        )
        _ppisp_camera_attr(camera_prim, f"vignettingAlpha1{s}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 2]))
        _ppisp_camera_attr(camera_prim, f"vignettingAlpha2{s}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 3]))
        _ppisp_camera_attr(camera_prim, f"vignettingAlpha3{s}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 4]))


def _author_camera_crf(camera_prim: Usd.Prim, ppisp: PPISP, camera_index: int) -> None:
    """Author per-camera CRF parameters as ``ppisp:*`` attributes."""
    crf = ppisp.crf_params[camera_index].detach().cpu().numpy()  # [3, 4]
    for ch in range(NUM_CHANNELS):
        s = CHANNEL_SUFFIXES[ch]
        _ppisp_camera_attr(camera_prim, f"crfToe{s}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 0]))
        _ppisp_camera_attr(camera_prim, f"crfShoulder{s}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 1]))
        _ppisp_camera_attr(camera_prim, f"crfGamma{s}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 2]))
        _ppisp_camera_attr(camera_prim, f"crfCenter{s}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 3]))


def _author_camera_animated_exposure(camera_prim: Usd.Prim, ppisp: PPISP, frame_indices: List[int]) -> None:
    """Author time-sampled ``ppisp:exposureOffset`` with per-camera mean as default."""
    exposure = ppisp.exposure_params.detach().cpu().numpy()
    valid = [i for i in frame_indices if i < len(exposure)]
    mean_exposure = float(np.mean(exposure[valid])) if valid else 0.0
    attr = _ppisp_camera_attr(camera_prim, "exposureOffset", Sdf.ValueTypeNames.Float)
    attr.Set(mean_exposure)
    for frame_idx in frame_indices:
        if frame_idx >= len(exposure):
            log.warning(
                f"PPISP frame index {frame_idx} out of bounds for exposure params (length {len(exposure)}), skipping"
            )
            continue
        attr.Set(float(exposure[frame_idx]), float(frame_idx))


def _author_camera_animated_color(camera_prim: Usd.Prim, ppisp: PPISP, frame_indices: List[int]) -> None:
    """Author the four time-sampled ``ppisp:colorLatent*`` attributes with mean defaults."""
    color = ppisp.color_params.detach().cpu().numpy()  # [num_frames, 8]
    valid = [i for i in frame_indices if i < len(color)]
    mean_color = np.mean(color[valid], axis=0) if valid else np.zeros(COLOR_PARAMS_PER_FRAME)

    attrs: list[Usd.Attribute] = []
    for i, name in enumerate(_COLOR_LATENT_INPUT_NAMES):
        attr = _ppisp_camera_attr(camera_prim, name, Sdf.ValueTypeNames.Float2)
        attr.Set(Gf.Vec2f(float(mean_color[i * 2]), float(mean_color[i * 2 + 1])))
        attrs.append(attr)

    for frame_idx in frame_indices:
        if frame_idx >= len(color):
            log.warning(f"PPISP frame index {frame_idx} out of bounds for color params (length {len(color)}), skipping")
            continue
        frame_color = color[frame_idx]
        for i, attr in enumerate(attrs):
            attr.Set(Gf.Vec2f(float(frame_color[i * 2]), float(frame_color[i * 2 + 1])), float(frame_idx))


def _author_camera_static_exposure(camera_prim: Usd.Prim, ppisp: PPISP, frame_id: int) -> None:
    """Author a single non-time-sampled ``ppisp:exposureOffset`` from one frame."""
    exposure = ppisp.exposure_params.detach().cpu().numpy()
    if frame_id >= len(exposure):
        raise ValueError(f"frame_id {frame_id} out of bounds for exposure_params of length {len(exposure)}")
    _ppisp_camera_attr(camera_prim, "exposureOffset", Sdf.ValueTypeNames.Float).Set(float(exposure[frame_id]))


def _author_camera_static_color(camera_prim: Usd.Prim, ppisp: PPISP, frame_id: int) -> None:
    """Author the four ``ppisp:colorLatent*`` attributes from a single frame."""
    color = ppisp.color_params.detach().cpu().numpy()
    if frame_id >= len(color):
        raise ValueError(f"frame_id {frame_id} out of bounds for color_params of length {len(color)}")
    frame_color = color[frame_id]
    for i, name in enumerate(_COLOR_LATENT_INPUT_NAMES):
        _ppisp_camera_attr(camera_prim, name, Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(frame_color[i * 2]), float(frame_color[i * 2 + 1]))
        )


def _author_camera_neutral_exposure(camera_prim: Usd.Prim) -> None:
    """Author ``ppisp:exposureOffset`` = 0.0 (neutral) for camera-only fixing."""
    _ppisp_camera_attr(camera_prim, "exposureOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def _author_camera_identity_color(camera_prim: Usd.Prim) -> None:
    """Author identity (all-zero) ``ppisp:colorLatent*`` attributes."""
    for name in _COLOR_LATENT_INPUT_NAMES:
        _ppisp_camera_attr(camera_prim, name, Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.0, 0.0))


def _set_tile_count_params(shader: UsdShade.Shader) -> None:
    """Author the untiled-default ``inputs:tileCount{X,Y}`` = 1 on the shader.

    Tiling is a RenderProduct property, so the grid is a constant shader
    input rather than a camera attribute. The tiling math reduces exactly
    to the pre-tiling behavior at 1x1; a downstream consumer that assembles
    a multi-camera tiled RenderProduct overrides these inputs to match the
    renderer's ``viewTilesPerDimension``.
    """
    for name in PPISP_TILE_COUNT_INPUT_NAMES:
        shader.CreateInput(name, Sdf.ValueTypeNames.Int).Set(DEFAULT_PPISP_TILE_COUNT)


def _author_ppisp_camera_attributes(
    stage: Usd.Stage,
    camera_path: Sdf.Path,
    ppisp: PPISP,
    camera_index: int,
    *,
    responsivity: float,
    resolved_reference_ids: Optional[ResolvedPPISPReferenceIds],
    frame_indices: List[int],
) -> None:
    """Author every PPISP parameter as a ``ppisp:*`` attribute on ``<cam>_ppisp``.

    This is the single source of truth for PPISP parameter values in the
    camera-connection design. Per-camera params (responsivity, vignetting,
    CRF) are always static; per-frame params (exposure, color) follow the
    same dispatch table as :func:`add_ppisp_shader_to_render_product`:

    * ``resolved_reference_ids is None`` -- time-sampled exposure/color.
    * ``frame_was_user_set`` -- static exposure/color from the resolved frame.
    * camera-only -- neutral exposure / identity color.
    """
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid():
        raise ValueError(f"PPISP camera prim not found at path: {camera_path}")

    _ppisp_camera_attr(camera_prim, PPISP_RESPONSIVITY_INPUT_NAME, Sdf.ValueTypeNames.Float).Set(float(responsivity))
    _author_camera_vignetting(camera_prim, ppisp, camera_index)
    _author_camera_crf(camera_prim, ppisp, camera_index)

    if resolved_reference_ids is None:
        _author_camera_animated_exposure(camera_prim, ppisp, frame_indices)
        _author_camera_animated_color(camera_prim, ppisp, frame_indices)
    elif resolved_reference_ids.frame_was_user_set:
        _author_camera_static_exposure(camera_prim, ppisp, resolved_reference_ids.frame_id)
        _author_camera_static_color(camera_prim, ppisp, resolved_reference_ids.frame_id)
    else:
        _author_camera_neutral_exposure(camera_prim)
        _author_camera_identity_color(camera_prim)


def _ppisp_shader_input_specs() -> List[tuple[str, Sdf.ValueTypeName]]:
    """Return the (input_name, value_type) specs for every connectable PPISP input.

    Mirrors the literal-authoring setters above; the order is irrelevant
    because each input connects independently to its ``ppisp:*`` source.
    """
    specs: List[tuple[str, Sdf.ValueTypeName]] = [
        (PPISP_RESPONSIVITY_INPUT_NAME, Sdf.ValueTypeNames.Float),
        ("exposureOffset", Sdf.ValueTypeNames.Float),
    ]
    for s in CHANNEL_SUFFIXES:
        specs.append((f"vignettingCenter{s}", Sdf.ValueTypeNames.Float2))
        specs.append((f"vignettingAlpha1{s}", Sdf.ValueTypeNames.Float))
        specs.append((f"vignettingAlpha2{s}", Sdf.ValueTypeNames.Float))
        specs.append((f"vignettingAlpha3{s}", Sdf.ValueTypeNames.Float))
        specs.append((f"crfToe{s}", Sdf.ValueTypeNames.Float))
        specs.append((f"crfShoulder{s}", Sdf.ValueTypeNames.Float))
        specs.append((f"crfGamma{s}", Sdf.ValueTypeNames.Float))
        specs.append((f"crfCenter{s}", Sdf.ValueTypeNames.Float))
    for name in _COLOR_LATENT_INPUT_NAMES:
        specs.append((name, Sdf.ValueTypeNames.Float2))
    return specs


def _connect_input_to_camera(
    shader: UsdShade.Shader,
    input_name: str,
    value_type: Sdf.ValueTypeName,
    camera_path: Sdf.Path,
) -> None:
    """Declare a shader input and connect it to ``<camera>.ppisp:<input_name>``.

    The SPG/Fabric translator resolves the ``.connect`` to the camera
    attribute at runtime (the dynamic-parameter path), and the SPG
    populator auto-prefetches the connected attribute into Fabric.
    """
    shader_input = shader.CreateInput(input_name, value_type)
    source = camera_path.AppendProperty(f"{PPISP_ATTR_NAMESPACE}{input_name}")
    shader_input.GetAttr().SetConnections([source])


def _connect_ppisp_shader_inputs(shader: UsdShade.Shader, camera_path: Sdf.Path) -> None:
    """Connect all PPISP shader inputs to the ``<cam>_ppisp.ppisp:*`` attributes."""
    for input_name, value_type in _ppisp_shader_input_specs():
        _connect_input_to_camera(shader, input_name, value_type, camera_path)


# -----------------------------------------------------------------------------
# Batch Export (All RenderProducts)
# -----------------------------------------------------------------------------


PPISP_CAMERA_EXPOSURE = 0.0
PPISP_CAMERA_EXPOSURE_FSTOP = 1.0
PPISP_CAMERA_EXPOSURE_ISO = 0.0
PPISP_CAMERA_EXPOSURE_RESPONSIVITY = 1.0
PPISP_CAMERA_EXPOSURE_TIME = 1.0
PPISP_CAMERA_AUTO_EXPOSURE_ENABLED = False
PPISP_CAMERA_API_SCHEMAS = [
    "OmniRtxCameraAutoExposureAPI_1",
    "OmniRtxCameraExposureAPI_1",
]


def _create_ppisp_camera(stage: Usd.Stage, render_product: Usd.Prim) -> Optional[Sdf.Path]:
    """Create a hidden ``<cam>_ppisp`` camera as a sibling of the source camera.

    Authors a Camera prim alongside the user-facing rig camera, inheriting
    the source camera's intrinsics via a path-based ``inherits`` arc and
    overriding the ``exposure*`` axis to neutral values (so Kit's automatic
    exposure compensation is suppressed downstream of the PPISP SPG node).
    The RenderProduct's ``camera`` relationship is re-pointed at this camera.
    The user-facing rig camera is left untouched.

    The PPISP parameters themselves are authored separately as ``ppisp:*``
    attributes on this camera (see :func:`_author_ppisp_camera_attributes`);
    this function only handles the prim creation, exposure neutralization,
    and rel retargeting.

    Returns the ``Sdf.Path`` of the ``<cam>_ppisp`` camera, or ``None`` when
    the RenderProduct has no camera target (nothing authored).

    Shim placement -- sibling-of-source, not child-of-RenderProduct.

    USD's ``inherits`` arc is a *specialization* arc: it brings only the
    target prim's own opinions into the inheritor, not the opinions of
    the target's ancestors. NCORE rigs author the time-sampled pose on
    the rig-parent Xform (``sensor_rig_0``) and keep only intrinsics on
    the camera prim itself, so a shim placed under ``/Render/<rp>/<leaf>``
    inherits the intrinsics but is detached from the rig pose -- the
    shim's world transform is then identity, not the rig pose at any
    time sample. Placing the shim as a sibling of the source camera
    instead lets it inherit the rig pose naturally via standard
    parent-transform composition.

    The source camera need not be resolvable on the *authoring* stage:
    in production the rig-camera payload is brought in via a reference
    added moments earlier in ``compose_default_usd_stage`` and composed
    at load-time by downstream consumers. All authoring here is path-
    based (``inherits`` target, ``DefinePrim`` at the shim path, and
    ``OverridePrim`` on each ancestor), so none of it depends on the
    source prim being resolvable yet.

    Idempotence: re-invocation for the same RenderProduct is a no-op
    (``DefinePrim`` / ``OverridePrim`` / ``AddInherit`` / ``Set`` are
    all idempotent on identical inputs).
    """
    camera_rel = render_product.GetRelationship("camera")
    camera_targets = camera_rel.GetTargets() if camera_rel else []
    if not camera_targets:
        log.warning(
            "RenderProduct %s has no camera target; skipping PPISP camera override",
            render_product.GetPath(),
        )
        return None

    source_camera_path: Sdf.Path = camera_targets[0]
    if source_camera_path.name.endswith(PPISP_CAMERA_SUFFIX):
        # Idempotence: the rel was already retargeted to a PPISP camera on
        # a prior invocation. Re-deriving the path from this would double
        # the suffix and author a redundant prim. Return the existing path
        # so callers can (re-)author the ``ppisp:*`` attributes onto it.
        return source_camera_path

    shim_parent_path: Sdf.Path = source_camera_path.GetParentPath()
    ppisp_camera_path: Sdf.Path = shim_parent_path.AppendChild(f"{source_camera_path.name}{PPISP_CAMERA_SUFFIX}")

    stage.OverridePrim(shim_parent_path)

    ppisp_camera_prim = stage.DefinePrim(ppisp_camera_path, "Camera")
    ppisp_camera_prim.SetHidden(True)
    ppisp_camera_prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=PPISP_CAMERA_API_SCHEMAS))
    UsdGeom.Imageable(ppisp_camera_prim).CreateVisibilityAttr().Set("invisible")
    ppisp_camera_prim.GetInherits().AddInherit(source_camera_path)
    ppisp_camera_prim.CreateAttribute("exposure", Sdf.ValueTypeNames.Float).Set(PPISP_CAMERA_EXPOSURE)
    ppisp_camera_prim.CreateAttribute("exposure:fStop", Sdf.ValueTypeNames.Float).Set(PPISP_CAMERA_EXPOSURE_FSTOP)
    ppisp_camera_prim.CreateAttribute("exposure:iso", Sdf.ValueTypeNames.Float).Set(PPISP_CAMERA_EXPOSURE_ISO)
    ppisp_camera_prim.CreateAttribute("exposure:responsivity", Sdf.ValueTypeNames.Float).Set(
        PPISP_CAMERA_EXPOSURE_RESPONSIVITY
    )
    ppisp_camera_prim.CreateAttribute("exposure:time", Sdf.ValueTypeNames.Float).Set(PPISP_CAMERA_EXPOSURE_TIME)
    ppisp_camera_prim.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool, custom=False).Set(
        PPISP_CAMERA_AUTO_EXPOSURE_ENABLED
    )
    camera_rel.SetTargets([ppisp_camera_path])
    return ppisp_camera_path


def add_ppisp_to_all_render_products(
    stage: Usd.Stage,
    ppisp: PPISP,
    camera_name_to_index: dict[str, int],
    camera_frame_mappings: Dict[str, List[int]],
    render_scope_path: str = "/Render",
    responsivity: float = DEFAULT_PPISP_RESPONSIVITY,
    *,
    ppisp_reference_camera_id: Optional[int] = None,
    ppisp_reference_frame_id: Optional[int] = None,
    ppisp_disable_controller: bool = True,
) -> list[Usd.Prim]:
    """Add PPISP shaders to all RenderProducts in the Render scope.

    For each RenderProduct, creates a hidden per-RenderProduct neutral-
    exposure camera shim (see :func:`_create_ppisp_camera`) so Kit's
    automatic exposure compensation does not double up on PPISP's
    exposure parameter, then authors the appropriate PPISP shader graph:

    * ``ppisp_disable_controller=True`` (default) -- static-parameter
      PPISP shader. Per-frame attributes follow the dispatch table
      documented on :func:`add_ppisp_shader_to_render_product`,
      driven by ``ppisp_reference_camera_id`` /
      ``ppisp_reference_frame_id``.
    * ``ppisp_disable_controller=False`` -- automatic-parameter
      PPISP graph (controller + auto-PPISP shader, see
      :func:`add_ppisp_auto_shader_to_render_product`). Per-camera
      controller weights are embedded in generated CUDA sidecars;
      ``ppisp_reference_*`` IDs are not consumed here because the
      controller produces exposure / color latents at runtime.
      Omniverse SPG CUDA controller support is runtime-specific.

    Args:
        camera_frame_mappings: per-camera ``[training_frame_idx, ...]``
            lists, keyed by camera logical name. USD time codes are
            written as ``float(training_frame_idx)``. Used only by the
            time-sampled static branch.
        responsivity: Achromatic HDR multiplier propagated to every
            PPISP shader's ``inputs:responsivity`` attribute. Default
            ``1.0`` is a no-op; consumers in target USD scenes can
            override it to retune asset brightness without re-baking
            the Gaussian SH coefficients. Must be a strictly positive
            finite number.
        ppisp_reference_camera_id: Optional PPISP camera index to fix
            the SPG export to. ``None`` (default) preserves the
            time-sampled per-camera authoring. Only consumed when
            ``ppisp_disable_controller=True``.
        ppisp_reference_frame_id: Optional PPISP frame index to fix the
            SPG export to. ``None`` (default) means "no static frame";
            see :func:`add_ppisp_shader_to_render_product` for the
            full dispatch table.
        ppisp_disable_controller: ``True`` (default) selects the
            static-parameter authoring path; ``False`` selects the
            automatic-parameter (controller) path.

    Raises:
        TypeError: If ``responsivity`` is not numeric, or any reference
            ID has the wrong type.
        ValueError: If ``responsivity`` is non-finite or non-positive,
            or if a reference ID is out of range, or if
            ``ppisp_reference_frame_id`` is set without
            ``ppisp_reference_camera_id``.
    """
    _validate_ppisp_responsivity(responsivity)

    resolved_ids: Optional[ResolvedPPISPReferenceIds] = None
    if ppisp_disable_controller and (ppisp_reference_camera_id is not None or ppisp_reference_frame_id is not None):
        # Resolve once; the resolver rejects frame-only fixing in
        # spg-runtime mode and validates ID types and ranges.
        resolved_ids = resolve_ppisp_reference_ids(
            ppisp_reference_camera_id,
            ppisp_reference_frame_id,
            ppisp,
            mode="spg-runtime",
        )

    render_scope = stage.GetPrimAtPath(render_scope_path)
    if not render_scope.IsValid():
        log.warning(f"Render scope not found at {render_scope_path}")
        return []

    created_shaders: list[Usd.Prim] = []
    for child in render_scope.GetChildren():
        if child.GetTypeName() != "RenderProduct":
            continue
        camera_name = child.GetName()
        if camera_name not in camera_name_to_index:
            log.warning(f"Camera '{camera_name}' not found in camera_name_to_index mapping")
            continue
        if camera_name not in camera_frame_mappings:
            log.warning(f"Camera '{camera_name}' not found in camera_frame_mappings")
            continue

        ppisp_camera_path = _create_ppisp_camera(stage, child)

        camera_index = camera_name_to_index[camera_name]
        if ppisp_disable_controller:
            shader_prim = _author_static_ppisp(
                stage=stage,
                render_product_path=str(child.GetPath()),
                camera_index=camera_index,
                ppisp=ppisp,
                frame_indices=camera_frame_mappings[camera_name],
                responsivity=responsivity,
                resolved_reference_ids=resolved_ids,
                ppisp_camera_path=ppisp_camera_path,
            )
        else:
            shader_prim = _author_auto_ppisp(
                stage=stage,
                render_product_path=str(child.GetPath()),
                camera_index=camera_index,
                ppisp=ppisp,
                responsivity=responsivity,
                ppisp_camera_path=ppisp_camera_path,
            )
        created_shaders.append(shader_prim)

    log.info(
        "Added PPISP shaders to %d RenderProducts (responsivity=%.6f, controller_disabled=%s)",
        len(created_shaders),
        float(responsivity),
        ppisp_disable_controller,
    )
    return created_shaders


def _author_static_ppisp(
    stage: Usd.Stage,
    render_product_path: str,
    camera_index: int,
    ppisp: PPISP,
    frame_indices: List[int],
    responsivity: float,
    resolved_reference_ids: Optional[ResolvedPPISPReferenceIds],
    ppisp_camera_path: Optional[Sdf.Path] = None,
) -> Usd.Prim:
    """Author the static-parameter PPISP shader for a single RenderProduct.

    Wraps :func:`add_ppisp_shader_to_render_product` with the resolved
    reference IDs. Honors the resolver's contract that a fixed
    ``camera_id`` overrides the iteration's ``camera_index``.
    """
    effective_camera = resolved_reference_ids.camera_id if resolved_reference_ids is not None else camera_index
    return add_ppisp_shader_to_render_product(
        stage=stage,
        render_product_path=render_product_path,
        camera_index=effective_camera,
        ppisp=ppisp,
        frame_indices=frame_indices,
        responsivity=responsivity,
        resolved_reference_ids=resolved_reference_ids,
        ppisp_camera_path=ppisp_camera_path,
    )


def _author_auto_ppisp(
    stage: Usd.Stage,
    render_product_path: str,
    camera_index: int,
    ppisp: PPISP,
    responsivity: float,
    ppisp_camera_path: Optional[Sdf.Path] = None,
) -> Usd.Prim:
    """Author the automatic-parameter PPISP graph.

    Resolves the per-camera controller via :func:`select_camera_controller`
    and delegates to :func:`add_ppisp_auto_shader_to_render_product`.
    This wrapper only adds the per-camera controller resolution step.
    """
    controller = select_camera_controller(ppisp, camera_index)
    return add_ppisp_auto_shader_to_render_product(
        stage=stage,
        render_product_path=render_product_path,
        camera_index=camera_index,
        ppisp=ppisp,
        controller=controller,
        responsivity=responsivity,
        ppisp_camera_path=ppisp_camera_path,
    )


def build_camera_name_to_index_mapping(rig_trajectories: RigTrajectories) -> Dict[str, int]:
    """Build a mapping from camera logical sensor name to PPISP camera index.

    The camera index corresponds to the order cameras appear in the rig trajectories,
    which matches how PPISP parameters are indexed.

    Args:
        rig_trajectories: RigTrajectories object from datasource

    Returns:
        Dictionary mapping camera name to index
    """
    camera_name_to_index: Dict[str, int] = {}
    camera_index = 0

    for rig_trajectory in rig_trajectories.rig_trajectories:
        for camera_unique_name in rig_trajectory.cameras_frame_timestamps_us:
            camera_data = rig_trajectories.camera_calibrations[camera_unique_name]
            logical_name = camera_data.logical_sensor_name
            if logical_name not in camera_name_to_index:
                camera_name_to_index[logical_name] = camera_index
                camera_index += 1

    return camera_name_to_index


def build_camera_frame_mappings(
    rig_trajectories: RigTrajectories,
    global_to_training_idx: np.ndarray,
) -> Dict[str, List[int]]:
    """Build per-camera lists of training-frame indices for SPG export.

    Returns ``{camera_name: [training_frame_idx, ...]}`` for each camera.
    Only training frames (where ``global_to_training_idx >= 0``) are
    included. The returned lists are used as USD time codes
    (``float(training_frame_idx)``) and as direct indices into
    ``ppisp.exposure_params`` / ``ppisp.color_params``.

    Args:
        rig_trajectories: Complete rig trajectory (all frames).
        global_to_training_idx: 1-D array of shape ``(total_all_frames,)``
            mapping each all-frame global camera index to a contiguous
            training-only index, or ``-1`` for excluded (e.g. holdout
            validation) frames. Produced by
            ``NCOREDataSource.setup_training_frame_mapping()``.
    """
    camera_mappings: Dict[str, List[int]] = {}

    for rig_trajectory in rig_trajectories.rig_trajectories:
        linear_start_indices = rig_trajectory.cameras_linear_start_frame_indices
        assert linear_start_indices is not None, "cameras_linear_start_frame_indices required for PPISP frame mapping"

        for camera_unique_id in rig_trajectory.cameras_frame_timestamps_us:
            camera_data = rig_trajectories.camera_calibrations[camera_unique_id]
            camera_name = camera_data.logical_sensor_name

            if camera_name in camera_mappings:
                continue

            linear_start_idx = linear_start_indices[camera_unique_id]
            timestamps_tensor = rig_trajectory.cameras_frame_timestamps_us[camera_unique_id]
            num_frames = timestamps_tensor.shape[0]

            frame_indices: List[int] = []
            for local_idx in range(num_frames):
                global_idx = linear_start_idx + local_idx
                training_idx = int(global_to_training_idx[global_idx])
                if training_idx >= 0:
                    frame_indices.append(training_idx)

            camera_mappings[camera_name] = frame_indices

            log.debug(f"Camera '{camera_name}': total_frames={num_frames}, training_frames={len(frame_indices)}")

    return camera_mappings


# -----------------------------------------------------------------------------
# PPISP SPG File Handling
# -----------------------------------------------------------------------------


def get_ppisp_spg_files() -> List[NamedSerialized]:
    """Load all PPISP SPG files (CUDA, Lua, USDA) as serialized data.

    Returns:
        List of NamedSerialized objects for each SPG file
    """
    spg_files: List[NamedSerialized] = []

    for filename in [PPISP_SPG_CU_FILE, PPISP_SPG_LUA_FILE, PPISP_SPG_USDA_FILE]:
        file_path = PPISP_SPG_DIR / filename
        if file_path.exists():
            with open(file_path, "rb") as f:
                file_data = f.read()
            spg_files.append(NamedSerialized(filename=filename, serialized=file_data))
            log.info(f"Loaded PPISP SPG file: {filename}")
        else:
            log.warning(f"PPISP SPG file not found: {file_path}")

    return spg_files
