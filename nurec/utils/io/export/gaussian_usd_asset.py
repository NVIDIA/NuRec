# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gaussian Splatting USD Export

Export Gaussian models to USD format as UsdGeom.Points primitives or ParticleField schema.

This module provides:
- USDGaussianExportCache: Manages USD stages and HDR files for USDZ packaging
- export_gaussians_as_usd_asset: Core export function for Gaussian data to USD
- CLI tool for checkpoint-based USD export

Coordinate System:
- USD uses Y-up by default; transforms applied via flip_*_axis parameters
- Timestamps in microseconds, converted to USD time codes via frame rate
"""

from __future__ import annotations

import io
import logging
import zipfile

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import click
import imageio
import numpy as np
import torch

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
from tqdm import tqdm

from nurec.config.parse import parse_typed_config
from nurec.datasets import make as make_dataset
from nurec.datasets.base import RigTrajectoriesProvider
from nurec.datasets.ncore import NCOREDataSource
from nurec.models.background import EnvMapType, SkyEnvMapBackground
from nurec.models.background_sampling import sample_cubemap_seamless
from nurec.models.post_processing import get_ppisp_from_model
from nurec.systems import make as make_system
from nurec.systems.gaussians import GaussiansSystem
from nurec.utils.io.export.gaussian_export_accessor import (
    GaussianAttributes,
    GaussianExportAccessor,
    ModelCapabilities,
)
from nurec.utils.io.export.gaussian_radiance_scaling import apply_radiance_scale_to_sh_nodes
from nurec.utils.io.export.gaussian_usd_writer import GaussianUSDSchemaType, create_gaussian_writer
from nurec.utils.io.export.ppisp_export.export_integration import (
    PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
    PPISP_INTEGRATION_MODE_SPG_RUNTIME,
    VALID_PPISP_INTEGRATION_MODES,
    ppisp_has_controller,
    resolve_ppisp_controller_export_enabled,
    resolve_sh_optimization_config,
    run_sh_match_optimization,
    select_spg_files_for_export,
    validate_export_integration_args,
)
from nurec.utils.io.export.ppisp_export.reference_selection import resolve_ppisp_reference_ids
from nurec.utils.io.export.ppisp_export.sh_match.node_iteration import iter_gaussian_nodes
from nurec.utils.io.export.ppisp_export.sh_match.optimization import SHMatchingOptimizationConfig
from nurec.utils.io.export.ppisp_export.usd_writer import (
    CameraRenderProductInfo,
    add_camera_render_products,
    add_ppisp_to_all_render_products,
    build_camera_frame_mappings,
    build_camera_name_to_index_mapping,
    extract_camera_render_products,
)
from nurec.utils.io.rig_trajectories import rig_trajectories_time_range, serialize_rig_trajectories
from nurec.utils.io.usd_nrend import serialize_nrend_usd
from nurec.utils.io.utils import initialize_usd_stage
from nurec.utils.types import NamedSerialized, NamedUSDStage


log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_FRAME_RATE = 24.0
USD_DEFAULT_TIME_CODE = 0.0
USD_MICROSECONDS_TO_SECONDS = 1e-06

# USD path constants
USD_WORLD_PATH = "/World"
USD_LOOKS_PATH = USD_WORLD_PATH + "/Looks"
USD_GAUSSIAN_ROOT_PATH = USD_WORLD_PATH + "/NuRec"
SKY_DOME_LIGHT_PATH = USD_WORLD_PATH + "/SkyBackground"

# Dome light defaults
DOME_LIGHT_DEFAULT_INTENSITY = 500.0
DOME_LIGHT_DEFAULT_EXPOSURE = 0.0

# Material constants (used by create_gaussian_material)
USD_GAUSSIAN_MATERIAL_PATH = USD_LOOKS_PATH + "/GaussianEmissive"
USD_GAUSSIAN_SHADER_PATH = USD_GAUSSIAN_MATERIAL_PATH + "/Shader"
GAUSSIAN_MATERIAL_MDL_FILE = "GaussianEmissive.mdl"
GAUSSIAN_MATERIAL_NAME = "GaussianEmissive"


# -----------------------------------------------------------------------------
# USDZ Export Cache
# -----------------------------------------------------------------------------


@dataclass
class USDGaussianExportCache:
    """
    Manages USD stages and HDR files for packaging into a USDZ archive.

    Collects multiple USD stages and HDR texture files, then composes them
    into a single default stage with references and packages as USDZ.
    """

    usd_stages: List[NamedUSDStage]
    hdr_files: List[NamedSerialized]
    serialized_files: List[NamedSerialized]
    # SPG sidecars packaged alongside the USDZ archive (e.g. slang/lua/usda
    # files produced by the PPISP export path).
    spg_files: List[NamedSerialized]
    camera_render_products: List[CameraRenderProductInfo]
    ppisp_module: Any  # ppisp.PPISP module if available
    camera_name_to_index: Dict[str, int]  # Camera name to PPISP index mapping
    camera_frame_mappings: Dict[str, List[int]]  # {camera_name: [training_frame_idx, ...]}
    ppisp_responsivity: float
    ppisp_reference_camera_id: Optional[int]
    ppisp_reference_frame_id: Optional[int]
    ppisp_disable_controller: bool

    def __init__(self) -> None:
        self.usd_stages = []
        self.hdr_files = []
        self.serialized_files = []
        self.spg_files = []
        self.camera_render_products = []
        self.ppisp_module = None
        self.camera_name_to_index = {}
        self.camera_frame_mappings = {}
        self.ppisp_responsivity = 1.0
        self.ppisp_reference_camera_id = None
        self.ppisp_reference_frame_id = None
        self.ppisp_disable_controller = True

    def add_usd_stage(self, stage: NamedUSDStage) -> None:
        """Add a USD stage to the cache for later packaging."""
        self.usd_stages.append(stage)

    def add_hdr_file(self, hdr_path: Path) -> None:
        """Load and cache an HDR file from disk if it exists."""
        if hdr_path and hdr_path.exists():
            with open(hdr_path, "rb") as f:
                hdr_data = f.read()
            self.hdr_files.append(NamedSerialized(filename=hdr_path.name, serialized=hdr_data))

    def add_spg_files(self, spg_files: List[NamedSerialized]) -> None:
        """Add SPG files to the cache."""
        self.spg_files.extend(spg_files)

    def add_serialized_files(self, serialized_files: List[NamedSerialized]) -> None:
        """Add non-USD sidecar files to the cache."""
        self.serialized_files.extend(serialized_files)

    def set_camera_render_products(self, camera_render_products: List[CameraRenderProductInfo]) -> None:
        """Set the camera render products info for creating RenderProducts in default.usda."""
        self.camera_render_products = camera_render_products

    def set_ppisp(
        self,
        ppisp_module: Any,
        camera_name_to_index: Dict[str, int],
        camera_frame_mappings: Dict[str, List[int]],
        *,
        responsivity: float = 1.0,
        ppisp_reference_camera_id: Optional[int] = None,
        ppisp_reference_frame_id: Optional[int] = None,
        ppisp_disable_controller: bool = True,
    ) -> None:
        """Set PPISP module and mappings for RenderProduct post-processing.

        The reference IDs and ``ppisp_disable_controller`` flag are
        forwarded as-is to :func:`add_ppisp_to_all_render_products`
        when the cache composes the default stage. See that function's
        docstring for the static / single-frame / automatic-parameter
        dispatch semantics.
        """
        self.ppisp_module = ppisp_module
        self.camera_name_to_index = camera_name_to_index
        self.camera_frame_mappings = camera_frame_mappings
        self.ppisp_responsivity = responsivity
        self.ppisp_reference_camera_id = ppisp_reference_camera_id
        self.ppisp_reference_frame_id = ppisp_reference_frame_id
        self.ppisp_disable_controller = ppisp_disable_controller

    def compose_default_usd_stage(self) -> Usd.Stage:
        """
        Create a composition stage that references all cached USD stages.

        The default stage sets up render settings and creates references to
        all component stages, handling sky dome lights specially.
        """
        stage = initialize_usd_stage()

        has_runtime_ppisp = (
            self.ppisp_module is not None and bool(self.camera_name_to_index) and bool(self.camera_frame_mappings)
        )
        has_nrend_volume = any(serialized_file.filename == "volume.nurec" for serialized_file in self.serialized_files)

        # Configure render settings for tone mapping.
        # Runtime PPISP consumes the Gaussian HDR output and produces LDR
        # color itself, so Kit's Gaussian skip-tonemapping path must be
        # disabled. SH-optimized exports do not call set_ppisp(), so baked
        # PPISP assets keep the vanilla renderer setting.
        render_settings: Dict[str, Any] = {"rtx:post:tonemap:op": 2}
        if has_runtime_ppisp:
            render_settings["rtx:rtpt:gaussian:skipTonemapping:enabled"] = False
        if has_runtime_ppisp and has_nrend_volume:
            render_settings["rtx:post:registeredCompositing:invertToneMap"] = False
            render_settings["rtx:post:registeredCompositing:invertColorCorrection"] = False
        stage.SetMetadataByDictKey("customLayerData", "renderSettings", render_settings)

        for named_stage in self.usd_stages:
            filename_stem = Path(named_stage.filename).stem
            sky_prim = self._try_get_sky_prim(named_stage)

            if sky_prim is not None and sky_prim.IsValid():
                # Reference sky dome light directly at the sky path
                # Intensity is set in the referenced stage (based on PPISP presence)
                prim = stage.OverridePrim(SKY_DOME_LIGHT_PATH)
                prim.GetReferences().AddReference(named_stage.filename, SKY_DOME_LIGHT_PATH)
            else:
                # Reference full stage under World
                prim_path = f"{USD_WORLD_PATH}/{filename_stem}"
                prim = stage.OverridePrim(prim_path)
                prim.GetReferences().AddReference(named_stage.filename)

            # Propagate animation settings from referenced layer
            referenced_layer = named_stage.stage.GetRootLayer()
            update_animation_settings(stage, referenced_layer)

        # Create RenderProducts for each camera in /Render scope
        add_camera_render_products(
            stage=stage,
            camera_render_products=self.camera_render_products,
            has_runtime_ppisp=has_runtime_ppisp,
        )

        # Apply PPISP shaders to RenderProducts if available
        if has_runtime_ppisp:
            add_ppisp_to_all_render_products(
                stage=stage,
                ppisp=self.ppisp_module,
                camera_name_to_index=self.camera_name_to_index,
                camera_frame_mappings=self.camera_frame_mappings,
                responsivity=self.ppisp_responsivity,
                ppisp_reference_camera_id=self.ppisp_reference_camera_id,
                ppisp_reference_frame_id=self.ppisp_reference_frame_id,
                ppisp_disable_controller=self.ppisp_disable_controller,
            )

        return stage

    def write_to_usdz(self, output_path: Path) -> None:
        """
        Package all cached stages and HDR files into a USDZ archive.

        Args:
            output_path: Destination path for the .usdz file

        Raises:
            ValueError: If no USD stages are cached
        """
        if not self.usd_stages:
            raise ValueError("No USD stages to package - nothing to export to USDZ")

        usd_default_layer = NamedUSDStage(
            filename="default.usda",  # Text format - composition only, no gaussian data
            stage=self.compose_default_usd_stage(),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
            usd_default_layer.save_to_zip(zip_file)
            for usd_stage in self.usd_stages:
                usd_stage.save_to_zip(zip_file)
            for hdr_file in self.hdr_files:
                hdr_file.save_to_zip(zip_file)
            for serialized_file in self.serialized_files:
                serialized_file.save_to_zip(zip_file)
            for spg_file in self.spg_files:
                spg_file.save_to_zip(zip_file)

        log.info(f"USDZ package created: {output_path}")

    def _try_get_sky_prim(self, named_stage: NamedUSDStage) -> Optional[Usd.Prim]:
        """Attempt to retrieve sky dome light prim from a stage."""
        try:
            referenced_stage = named_stage.stage
            if referenced_stage:
                return referenced_stage.GetPrimAtPath(SKY_DOME_LIGHT_PATH)
        except Exception as e:
            log.warning(f"Failed to get sky prim from stage {named_stage.filename}: {e}")
        return None


# -----------------------------------------------------------------------------
# Timestamp Utilities
# -----------------------------------------------------------------------------


def resample_timestamps(
    timestamps_us: List[int],
    reference_timestamps_us: List[int],
) -> List[int]:
    """
    Find the consecutive subset of reference timestamps that covers the input range.

    Used to align track timestamps with rig trajectory timestamps for consistent
    animation timing across the export.

    Args:
        timestamps_us: Source timestamps to cover
        reference_timestamps_us: Reference timestamps to sample from

    Returns:
        Subset of reference_timestamps_us spanning the input range
    """
    if not timestamps_us or not reference_timestamps_us:
        return timestamps_us

    start_us = timestamps_us[0]
    end_us = timestamps_us[-1]

    # Initialize to full range as safe defaults
    start_idx: int = 0
    end_idx: int = len(reference_timestamps_us) - 1

    # Find last reference timestamp before or at start_us (searching backwards)
    for i in range(len(reference_timestamps_us) - 1, -1, -1):
        if reference_timestamps_us[i] <= start_us:
            start_idx = i
            break

    # Find first reference timestamp after or at end_us (searching forwards)
    for i in range(len(reference_timestamps_us)):
        if reference_timestamps_us[i] >= end_us:
            end_idx = i
            break

    # Validate indices - if no valid overlap found, return full reference range
    if start_idx > end_idx:
        log.warning(
            f"No overlap between timestamps [{start_us}, {end_us}] and reference range "
            f"[{reference_timestamps_us[0]}, {reference_timestamps_us[-1]}]. Using full reference range."
        )
        return reference_timestamps_us

    return reference_timestamps_us[start_idx : end_idx + 1]


def _compute_time_code(tps: float, timestamp_us: int, offset_us: int) -> float:
    """Convert microsecond timestamp to USD time code."""
    return float(tps) * USD_MICROSECONDS_TO_SECONDS * float(timestamp_us - offset_us)


# -----------------------------------------------------------------------------
# Rig Trajectories
# -----------------------------------------------------------------------------


def create_rig_trajectories_stage(
    datasource: Any,
    frame_rate: float,
) -> Tuple[Optional[NamedUSDStage], int, float, List[int]]:
    """
    Create a USD stage containing rig trajectory data from the datasource.

    Args:
        datasource: Data source that may provide rig trajectories
        frame_rate: Target frame rate for time code calculation

    Returns:
        Tuple of (stage, offset_us, scale, all_timestamps):
        - stage: USD stage with serialized rig trajectories (or None)
        - offset_us: Timestamp offset in microseconds
        - scale: Time code scale factor
        - all_timestamps: List of all trajectory timestamps
    """
    rig_stage: Optional[NamedUSDStage] = None
    offset_us: int = 0
    all_ts: List[int] = []
    scale = float(frame_rate) * USD_MICROSECONDS_TO_SECONDS

    try:
        if isinstance(datasource, RigTrajectoriesProvider):
            rig_trajectories = datasource.get_rig_trajectories()
            timerange = rig_trajectories_time_range(rig_trajectories)
            offset_us = int(timerange.start)

            # Collect all unique timestamps across rigs
            all_set: set[int] = set()
            for rig in rig_trajectories.rig_trajectories:
                ts = rig.T_rig_world_timestamps_us
                try:
                    ts_list = ts.numpy().astype(int).tolist()
                except Exception as e:
                    log.debug(f"Falling back to iterative timestamp conversion: {e}")
                    ts_list = [int(t) for t in ts]
                all_set.update(ts_list)

            if all_set:
                all_ts = sorted(all_set)

            # Serialize to USD. Exposure neutralization for PPISP-augmented
            # exports is handled by per-RenderProduct camera shims (see
            # _create_ppisp_camera in ppisp_export.usd_writer), not by mutating the
            # rig cameras here.
            rig_usd = serialize_rig_trajectories(
                rig_trajectories,
                usd_timestamp_offset=offset_us,
                formats=["usda"],
                add_default_cameras=False,
            )
            for item in rig_usd:
                if isinstance(item, NamedUSDStage):
                    rig_stage = item
                    break
    except Exception as e:
        log.warning(f"Failed to create rig trajectories stage: {e}")

    return rig_stage, offset_us, scale, all_ts


# -----------------------------------------------------------------------------
# USD Stage Setup
# -----------------------------------------------------------------------------


def create_gaussian_model_root(
    stage: Usd.Stage,
    flip_x_axis: bool,
    flip_y_axis: bool,
    flip_z_axis: bool,
    dataset_offset: Optional[np.ndarray],
    root_path: str = USD_GAUSSIAN_ROOT_PATH,
) -> str:
    """
    Create the root Xform for Gaussian content with optional coordinate transforms.

    Args:
        stage: USD stage to create the root on
        flip_x_axis: Negate X coordinates
        flip_y_axis: Negate Y coordinates
        flip_z_axis: Negate Z coordinates
        dataset_offset: Optional translation offset to apply
        root_path: USD path for the root prim

    Returns:
        The root path string
    """
    root_xform = UsdGeom.Xform.Define(stage, root_path)
    transform_op = root_xform.AddTransformOp()

    # Build scale matrix for axis flipping
    scale_x = -1.0 if flip_x_axis else 1.0
    scale_y = -1.0 if flip_y_axis else 1.0
    scale_z = -1.0 if flip_z_axis else 1.0
    scale_mat = Gf.Matrix4d().SetScale(Gf.Vec3d(scale_x, scale_y, scale_z))

    # Apply translation for dataset offset
    if dataset_offset is not None:
        t = Gf.Vec3d(-float(dataset_offset[0]), -float(dataset_offset[1]), -float(dataset_offset[2]))
        translate_mat = Gf.Matrix4d().SetTranslate(t)
        final_mat = translate_mat * scale_mat
    else:
        final_mat = scale_mat

    transform_op.Set(final_mat)
    return root_path


def update_animation_settings(stage: Usd.Stage, referenced_layer: Sdf.Layer) -> None:
    """
    Propagate animation timing metadata from a referenced layer to the root stage.

    Updates start/end time codes, time codes per second, and absolute time offset
    while checking for conflicts.

    Args:
        stage: Target stage to update
        referenced_layer: Source layer with animation settings

    Raises:
        ValueError: If time codes per second or absolute offset conflict
    """
    USD_DEFAULT_TPS = DEFAULT_FRAME_RATE

    # Update start time code (use minimum)
    if referenced_layer.startTimeCode != USD_DEFAULT_TIME_CODE:
        current_start = stage.GetStartTimeCode()
        new_start = referenced_layer.startTimeCode
        if current_start == USD_DEFAULT_TIME_CODE:
            current_start = new_start
        stage.SetStartTimeCode(min(current_start, new_start))

    # Update end time code (use maximum)
    if referenced_layer.endTimeCode != USD_DEFAULT_TIME_CODE:
        current_end = stage.GetEndTimeCode()
        new_end = referenced_layer.endTimeCode
        if current_end == USD_DEFAULT_TIME_CODE:
            current_end = new_end
        stage.SetEndTimeCode(max(current_end, new_end))

    # Set time codes per second (must match if already set)
    if referenced_layer.timeCodesPerSecond != USD_DEFAULT_TPS:
        current_tps = stage.GetTimeCodesPerSecond()
        new_tps = referenced_layer.timeCodesPerSecond
        if current_tps == USD_DEFAULT_TPS:
            stage.SetTimeCodesPerSecond(new_tps)
        elif current_tps != new_tps:
            raise ValueError(f"TimeCodesPerSecond mismatch: existing value {current_tps} and new value {new_tps}")

    # Propagate absolute time offset
    if "absoluteTimeOffsetMicroSec" in referenced_layer.customLayerData:
        new_abs_off = referenced_layer.customLayerData["absoluteTimeOffsetMicroSec"]
        current_abs_off = stage.GetMetadataByDictKey("customLayerData", "absoluteTimeOffsetMicroSec")
        if not current_abs_off:
            stage.SetMetadataByDictKey("customLayerData", "absoluteTimeOffsetMicroSec", new_abs_off)
        elif new_abs_off != current_abs_off:
            raise ValueError(
                f"absoluteTimeOffsetMicroSec mismatch: existing value {current_abs_off} and new value {new_abs_off}"
            )


# -----------------------------------------------------------------------------
# Background / Environment
# -----------------------------------------------------------------------------


def create_background_domelight(
    background: Any,
    output_path: Path,
    stage: Optional[Usd.Stage] = None,
    light_path: str = SKY_DOME_LIGHT_PATH,
    hdr_filename: str = "sky_envmap",
    intensity: float = DOME_LIGHT_DEFAULT_INTENSITY,
) -> Optional[Tuple[NamedUSDStage, NamedSerialized]]:
    """Create USD DomeLight and HDR texture from a SkyEnvMapBackground model.

    Exports environment map textures (equirectangular or cubemap) as HDR images
    (serialized in memory) and creates a DomeLight prim that references the HDR texture.

    Note: HDR data is only serialized in memory, not written to disk. The caller
    is responsible for writing to disk if needed (for non-USDZ formats).

    Args:
        background: Background model to export (must be SkyEnvMapBackground)
        output_path: Directory path (used only for creating directory structure)
        stage: Optional existing USD stage (creates new if None)
        light_path: USD path for the DomeLight prim
        hdr_filename: Base filename for the HDR texture (without extension)
        intensity: DomeLight intensity value

    Returns:
        Tuple of (DomeLight USD stage, HDR texture serialized data) if successful,
        None if background is not SkyEnvMapBackground or export fails
    """
    if not isinstance(background, SkyEnvMapBackground):
        return None

    # Prepare HDR data
    textures = background.flattened_textures()
    inpaint_result = background.maybe_inpaint()
    if inpaint_result is not None:
        textures, _ = inpaint_result
    textures_np = textures.detach().cpu().numpy()

    if background.envmap_type == EnvMapType.EQUIRECTANGULAR:
        # textures_np is either (H, W, 3) or (1, H, W, 3)
        hdr_data = textures_np.squeeze(0) if textures_np.ndim == 4 else textures_np
    elif background.envmap_type == EnvMapType.CUBEMAP:
        log.info("Converting cubemap to equirectangular for HDR export...")
        faces = background.textures  # (1,6,H,W,3) # type: ignore
        device = faces.device
        Hf, Wf = faces.shape[2], faces.shape[3]
        eq_h, eq_w = Hf * 2, Wf * 3
        theta = torch.linspace(0.0, np.pi, eq_h, device=device)
        phi = torch.linspace(-np.pi, np.pi, eq_w, device=device)
        theta_grid = theta[:, None].expand(eq_h, eq_w)
        phi_grid = phi[None, :].expand(eq_h, eq_w) + (np.pi / 2.0)  # center +Z
        dx = torch.sin(theta_grid) * torch.cos(phi_grid)
        dy = torch.cos(theta_grid)
        dz = torch.sin(theta_grid) * torch.sin(phi_grid)
        dirs = torch.stack([dx, dy, dz], dim=-1).reshape(-1, 3)
        sampled = sample_cubemap_seamless(faces.squeeze(0), dirs).reshape(eq_h, eq_w, 3)
        hdr_data = sampled.detach().cpu().numpy()
    else:
        log.error(f"Unsupported environment map type: {background.envmap_type}")
        return None

    hdr_data = hdr_data.astype(np.float32)
    saturate_radiance = getattr(background.config, "saturate_radiance", True)
    if saturate_radiance:
        hdr_data = np.clip(hdr_data, 0.0, 1.0)
    else:
        hdr_data = np.maximum(hdr_data, 0.0)

    # Serialize HDR to memory for USDZ packaging
    hdr_buffer = io.BytesIO()
    imageio.imwrite(uri=hdr_buffer, im=hdr_data, format="HDR")  # type: ignore[call-overload]
    hdr_filename_with_ext = f"{hdr_filename}.hdr"
    named_hdr = NamedSerialized(filename=hdr_filename_with_ext, serialized=hdr_buffer.getvalue())

    # Build DomeLight stage
    if stage is None:
        stage_filename = f"{hdr_filename}_domelight.usda"
        stage = initialize_usd_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    else:
        stage_filename = "domelight.usda"

    dome_light = UsdLux.DomeLight.Define(stage, light_path)
    dome_light.CreateIntensityAttr(intensity)
    dome_light.CreateExposureAttr(DOME_LIGHT_DEFAULT_EXPOSURE)
    dome_light.CreateTextureFileAttr(hdr_filename_with_ext)
    dome_light.CreateTextureFormatAttr("latlong")

    named_stage = NamedUSDStage(stage=stage, filename=stage_filename)
    return named_stage, named_hdr


def compute_runtime_post_processing(
    model_has_postprocessing: bool,
    ppisp_module: Optional[Any],
    ppisp_integration_mode: str,
) -> bool:
    """Decide whether the exported USD has a runtime PPISP stage downstream of MDL.

    Returns ``True`` iff a PPISP SPG shader will be authored on each
    RenderProduct, i.e. ``ppisp_integration_mode == 'spg-runtime'``
    *and* the model exposes a PPISP module.

    Why this is *not* the same as ``model_has_postprocessing``: the
    ``sh-optimized`` mode folds PPISP into the Gaussian SH coefficients
    during export, so the runtime pipeline is just ``SH -> MDL`` even
    though the source model still carries the PPISP collection (used
    to obtain the bake target). Coupling the MDL color-correction
    switches to "model carries PPISP" therefore lies to MDL about
    whether a downstream PPISP stage will undo the corrections.

    Consumed by the orchestrator (``gaussian_usd_asset_from_system``)
    to drive both:

    * ``create_gaussian_material(has_post_processing=...)`` -- toggles
      ``inputs:apply_srgb_linear`` / ``inputs:apply_inverse_tonemap``
      on the MDL shader.
    * ``gaussian_node_to_usd(linear_srgb=..., has_post_processing=...)``
      -- SH packing color space (linear vs sRGB) and the per-node MDL
      material switches.

    Args:
        model_has_postprocessing: ``len(model.post_processings) > 0``.
        ppisp_module: The PPISP module obtained from the model when
            ``model_has_postprocessing`` is true; ``None`` otherwise.
            Passed in for defense-in-depth so this helper is robust
            to a future caller that decouples the two.
        ppisp_integration_mode: One of the values in
            :data:`VALID_PPISP_INTEGRATION_MODES`.

    Returns:
        ``True`` iff a runtime PPISP SPG stage will be authored.
    """
    return (
        model_has_postprocessing
        and ppisp_module is not None
        and ppisp_integration_mode == PPISP_INTEGRATION_MODE_SPG_RUNTIME
    )


def add_nrend_volume_assets_to_cache(
    *,
    cache: USDGaussianExportCache,
    model: Any,
    datasource: Any,
    has_runtime_ppisp: bool = False,
) -> None:
    """Serialize the model as a NuRec/NRend volume and add it to the USD asset cache."""
    nrend_contents = serialize_nrend_usd(
        model,
        datasource.aabb,
        datasource.get_offset(),
        proxy_mesh_paths=None,
        sequence_track_paths=None,
        serialize_to_legacy_nrend_dict=False,
        invert_registered_compositing=not has_runtime_ppisp,
        accept_unsupported_ppisp=has_runtime_ppisp,
        disable_runtime_post_processing=has_runtime_ppisp,
    )
    for item in nrend_contents:
        if isinstance(item, NamedUSDStage):
            cache.add_usd_stage(item)
        elif isinstance(item, NamedSerialized):
            cache.add_serialized_files([item])
        else:
            raise TypeError(f"Unsupported NRend export artifact: {type(item).__name__}")


def create_gaussian_material(stage: Usd.Stage, has_post_processing: bool = False) -> Usd.Prim:
    """
    Create a Gaussian emissive material on the given stage.

    This creates a Material prim with an MDL shader configured for
    Gaussian splatting rendering.

    Args:
        stage: USD stage to create the material on
        has_post_processing: If True, disables sRGB-to-linear and inverse
            tonemap in the MDL shader because a runtime PPISP stage
            downstream of MDL will handle color correction itself. The
            orchestrator feeds this flag from
            :func:`compute_runtime_post_processing`, not from
            ``len(model.post_processings) > 0``: in ``sh-optimized``
            mode PPISP is baked into the SH coefficients and there is
            no runtime PPISP stage, so MDL must apply its own color
            correction as in a vanilla export.

    Returns:
        The created Material prim
    """
    from pxr import UsdShade

    # Ensure Looks scope exists
    looks_prim = stage.GetPrimAtPath(USD_LOOKS_PATH)
    if not looks_prim.IsValid():
        stage.DefinePrim(USD_LOOKS_PATH, "Scope")

    # Create material and shader prims
    material_prim = stage.DefinePrim(USD_GAUSSIAN_MATERIAL_PATH, "Material")
    shader_prim = stage.DefinePrim(USD_GAUSSIAN_SHADER_PATH, "Shader")

    # Configure shader implementation
    shader_prim.CreateAttribute(
        "info:implementationSource", Sdf.ValueTypeNames.Token, custom=False, variability=Sdf.VariabilityUniform
    ).Set("sourceAsset")
    shader_prim.CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset, custom=False, variability=Sdf.VariabilityUniform
    ).Set(Sdf.AssetPath(GAUSSIAN_MATERIAL_MDL_FILE))
    shader_prim.CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token, custom=False, variability=Sdf.VariabilityUniform
    ).Set(GAUSSIAN_MATERIAL_NAME)

    # Configure MDL shader parameters when post-processing is enabled
    # Post-processing handles color correction, so disable MDL's built-in correction
    if has_post_processing:
        shader_prim.CreateAttribute("inputs:apply_srgb_linear", Sdf.ValueTypeNames.Bool).Set(False)
        shader_prim.CreateAttribute("inputs:apply_inverse_tonemap", Sdf.ValueTypeNames.Bool).Set(False)

    # Create shader output
    outputs_out = shader_prim.CreateAttribute("outputs:out", Sdf.ValueTypeNames.Token)
    outputs_out.SetMetadata("renderType", "material")

    # Connect material outputs to shader
    material = UsdShade.Material(material_prim)
    shader = UsdShade.Shader(shader_prim)
    for output_name in ["mdl:displacement", "mdl:surface", "mdl:volume"]:
        output = material.CreateOutput(output_name, Sdf.ValueTypeNames.Token)
        output.ConnectToSource(shader.GetOutput("out"))

    return material_prim


# -----------------------------------------------------------------------------
# Gaussian Subsampling
# -----------------------------------------------------------------------------


def apply_gaussian_subsampling(gaussian_indices: torch.Tensor, percentage_gaussians: float) -> torch.Tensor:
    """
    Randomly subsample Gaussian indices to the specified percentage.

    Args:
        gaussian_indices: Input tensor of Gaussian indices
        percentage_gaussians: Percentage to keep (0-100]

    Returns:
        Subsampled indices tensor (sorted)
    """
    if percentage_gaussians >= 100.0:
        return gaussian_indices

    n_gaussians_original = len(gaussian_indices)
    n_gaussians_to_keep = max(1, int(n_gaussians_original * percentage_gaussians / 100))

    if n_gaussians_to_keep >= n_gaussians_original:
        return gaussian_indices

    random_indices = torch.randperm(n_gaussians_original, device=gaussian_indices.device)[:n_gaussians_to_keep]
    random_indices = torch.sort(random_indices)[0]
    return gaussian_indices[random_indices]


# -----------------------------------------------------------------------------
# Track Export Context
# -----------------------------------------------------------------------------


@dataclass
class TrackExportContext:
    """Holds all metadata needed to export a single track."""

    track_id: str
    track_enum_index: int
    gaussian_indices: torch.Tensor
    timestamps_us: List[int]
    timestamps_set: Set[int]
    writer: Any
    all_positions: List[np.ndarray] = field(default_factory=list)
    first_frame_written: bool = False


# -----------------------------------------------------------------------------
# Export Loop Helpers
# -----------------------------------------------------------------------------


def _prepare_track_contexts(
    accessor: GaussianExportAccessor,
    stage: Usd.Stage,
    schema_type: GaussianUSDSchemaType,
    capabilities: ModelCapabilities,
    content_root_path: str,
    node_layer_name: Optional[str],
    percentage_gaussians: float,
    reference_timestamps_us: Optional[List[int]],
    sorting_mode_hint: str,
    half_precision: bool,
    linear_srgb: bool,
    has_post_processing: bool,
    valid_gaussian_mask: Optional[np.ndarray] = None,
) -> Tuple[List[TrackExportContext], List[int]]:
    """
    Prepare track contexts and collect all timestamps for frame-first export.

    Track transforms are not computed here - they are extracted from
    get_attributes_at_timestamp during the frame loop.

    Args:
        valid_gaussian_mask: Optional boolean mask [N] to filter out invalid Gaussians (NaN/Inf).
            If provided, only Gaussians where mask is True will be exported.

    Returns:
        Tuple of (track_contexts, all_timestamps_sorted)
    """
    track_contexts: List[TrackExportContext] = []
    all_timestamps_set: Set[int] = set()

    tracks = accessor.get_cuboid_tracks()
    track_mapping = accessor.get_track_gaussian_mapping()

    skipped_no_mapping: List[str] = []
    skipped_no_gaussians: List[str] = []
    skipped_no_timestamps_in_range: List[str] = []

    for track_enum_index, track_id in tqdm(
        enumerate(tracks.tracks_id), desc="Preparing tracks", total=len(tracks.tracks_id), unit="track"
    ):
        if track_id not in track_mapping:
            skipped_no_mapping.append(track_id)
            continue

        gaussian_indices = track_mapping[track_id]

        # Filter out invalid Gaussians (NaN/Inf) before subsampling
        if valid_gaussian_mask is not None:
            valid_mask_tensor = torch.from_numpy(valid_gaussian_mask).to(gaussian_indices.device)
            gaussian_indices = gaussian_indices[valid_mask_tensor[gaussian_indices]]

        gaussian_indices = apply_gaussian_subsampling(gaussian_indices, percentage_gaussians)

        # Skip tracks with no valid Gaussians
        if len(gaussian_indices) == 0:
            skipped_no_gaussians.append(track_id)
            continue

        # Get timestamps for this track (native pose keyframe times from model)
        native_timestamps = accessor.get_track_timestamps(track_enum_index)
        valid_time_samples = list(native_timestamps) if native_timestamps else []
        if reference_timestamps_us:
            valid_time_samples = resample_timestamps(valid_time_samples, reference_timestamps_us)

        # Restrict to track's native time range so we never write extrapolated poses
        # (poses outside [track_min_us, track_max_us] can be wrong and make tracks render at wrong position)
        time_range = accessor.get_track_time_range_safe(track_enum_index)
        if time_range is not None and valid_time_samples:
            track_min_us, track_max_us = time_range
            original_count = len(valid_time_samples)
            valid_time_samples = [t for t in valid_time_samples if track_min_us <= t <= track_max_us]
            dropped = original_count - len(valid_time_samples)
            if dropped > 0:
                log.debug(
                    "USD export: track '%s' clipped %d frame(s) outside native range [%d, %d] us",
                    track_id,
                    dropped,
                    track_min_us,
                    track_max_us,
                )

        # Skip tracks with no valid timestamps (no pose data in range); do not export with a single frame
        # at offset, which would use an extrapolated/wrong pose.
        if not valid_time_samples:
            skipped_no_timestamps_in_range.append(track_id)
            continue

        # Create writer for this track (transforms written per-frame from accessor)
        writer = create_gaussian_writer(
            schema_type=schema_type,
            stage=stage,
            capabilities=capabilities,
            track_id=track_id,
            track_index=track_enum_index,
            node_layer_name=node_layer_name,
            content_root_path=content_root_path,
            sorting_mode_hint=sorting_mode_hint,
            half_precision=half_precision,
            linear_srgb=linear_srgb,
            has_post_processing=has_post_processing,
        )

        # Create prim upfront (no transforms - they come from get_attributes_at_timestamp)
        writer.create_prim(num_gaussians=len(gaussian_indices), track_transforms=None)

        track_contexts.append(
            TrackExportContext(
                track_id=track_id,
                track_enum_index=track_enum_index,
                gaussian_indices=gaussian_indices,
                timestamps_us=valid_time_samples,
                timestamps_set=set(valid_time_samples),
                writer=writer,
            )
        )

        all_timestamps_set.update(valid_time_samples)

    if skipped_no_mapping:
        log.warning(
            "USD export: %s track(s) skipped (no gaussians assigned in model): %s",
            len(skipped_no_mapping),
            skipped_no_mapping[:20] if len(skipped_no_mapping) > 20 else skipped_no_mapping,
        )
    if skipped_no_gaussians:
        log.warning(
            "USD export: %s track(s) skipped (zero gaussians after NaN/Inf filter and subsampling): %s",
            len(skipped_no_gaussians),
            skipped_no_gaussians[:20] if len(skipped_no_gaussians) > 20 else skipped_no_gaussians,
        )
    if skipped_no_timestamps_in_range:
        log.warning(
            "USD export: %s track(s) skipped (no timestamps inside track native pose range): %s",
            len(skipped_no_timestamps_in_range),
            skipped_no_timestamps_in_range[:20]
            if len(skipped_no_timestamps_in_range) > 20
            else skipped_no_timestamps_in_range,
        )

    all_timestamps_sorted = sorted(all_timestamps_set)
    return track_contexts, all_timestamps_sorted


def _write_track_frame(
    ctx: TrackExportContext,
    attributes: GaussianAttributes,
    interpolated_track_poses: Optional[np.ndarray],
    timestamp_us: int,
    capabilities: ModelCapabilities,
    computed_offset_us: int,
    tps: float,
    skip_gaussian_deformation: bool,
    force_sh_0: bool,
    do_not_cast_shadows: bool,
    animate_albedo: bool,
) -> None:
    """Write data for a single track at a single frame.

    Uses albedo_coefficients and specular_coefficients from attributes directly
    (returned by get_attributes_at_timestamp).
    Caller must only invoke when the track is visible at this timestamp (so we do not write
    extrapolated poses); visibility is authored separately via write_track_visibility.
    """
    np_indices = ctx.gaussian_indices.cpu().numpy()

    # Write track transform (caller ensures we are only called when track is visible)
    if interpolated_track_poses is not None and capabilities.has_rigid_tracks:
        if ctx.track_enum_index >= len(interpolated_track_poses):
            log.warning(
                "Track index %s out of bounds for interpolated_track_poses (length %s). Skipping transform for track '%s'.",
                ctx.track_enum_index,
                len(interpolated_track_poses),
                ctx.track_id,
            )
        else:
            # Get pose for this track (indexed by track_enum_index, not gaussian index)
            track_pose = interpolated_track_poses[ctx.track_enum_index]  # [7] = [tx, ty, tz, qx, qy, qz, qw]

            # Convert pose to SE3 matrix in USD format
            # USD uses row-vector convention: point * matrix
            # Gf.Matrix4d.SetRotate/SetTranslateOnly already produce USD-format matrices
            pose_np = track_pose.astype(np.float64)
            translation = pose_np[:3]
            quat = pose_np[3:]  # [qx, qy, qz, qw]

            # Build rotation matrix from quaternion (xyzw -> USD's wxyz constructor)
            qx, qy, qz, qw = quat
            rot = Gf.Rotation(Gf.Quatd(qw, qx, qy, qz))
            usd_matrix = Gf.Matrix4d()
            usd_matrix.SetRotate(rot)
            usd_matrix.SetTranslateOnly(Gf.Vec3d(translation[0], translation[1], translation[2]))

            time_code = _compute_time_code(tps, timestamp_us, computed_offset_us)
            ctx.writer.write_track_transform(time_code, usd_matrix)

    # Collect positions for extent computation
    ctx.all_positions.append(attributes.positions[np_indices])

    if not ctx.first_frame_written:
        # First frame: write static and initial animated attributes
        ctx.writer.write_static_attributes(
            attributes=attributes,
            gaussian_indices=ctx.gaussian_indices,
            do_not_cast_shadows=do_not_cast_shadows,
            force_sh_0=force_sh_0,
        )
        ctx.writer.write_first_frame_attributes(
            attributes=attributes,
            gaussian_indices=ctx.gaussian_indices,
        )
        # USD defaults do not participate in animation once explicit time
        # samples exist. Author the first temporal albedo at its real time code
        # so the animation does not clamp backwards from the second frame.
        if animate_albedo:
            time_code = _compute_time_code(tps, timestamp_us, computed_offset_us)
            ctx.writer.write_animated_attributes(
                attributes=attributes,
                gaussian_indices=ctx.gaussian_indices,
                time_code=time_code,
                animate_positions=False,
                animate_rotations=False,
                animate_scales=False,
                animate_albedo=True,
            )
        ctx.first_frame_written = True
    else:
        # Subsequent frames: write animated attributes
        should_animate_gaussians = (not skip_gaussian_deformation) and capabilities.has_deformation
        animate_rotations = (not skip_gaussian_deformation) and capabilities.can_deform_rotations
        animate_scales = (not skip_gaussian_deformation) and capabilities.can_deform_scales

        if should_animate_gaussians or animate_rotations or animate_scales or animate_albedo:
            time_code = _compute_time_code(tps, timestamp_us, computed_offset_us)
            ctx.writer.write_animated_attributes(
                attributes=attributes,
                gaussian_indices=ctx.gaussian_indices,
                time_code=time_code,
                animate_positions=should_animate_gaussians,
                animate_rotations=animate_rotations,
                animate_scales=animate_scales,
                animate_albedo=animate_albedo,
            )


# -----------------------------------------------------------------------------
# Main Export Function
# -----------------------------------------------------------------------------


def export_gaussians_as_usd_asset(
    accessor: GaussianExportAccessor,
    stage: Usd.Stage,
    content_root_path: str,
    schema_type: GaussianUSDSchemaType,
    percentage_gaussians: float = 100.0,
    node_layer_name: Optional[str] = None,
    force_sh_0: bool = False,
    preactivation: bool = False,
    do_not_cast_shadows: bool = False,
    skip_gaussian_deformation: bool = False,
    usd_timestamp_offset_us: Optional[int] = None,
    reference_timestamps_us: Optional[List[int]] = None,
    sorting_mode_hint: Optional[str] = None,
    half_precision: bool = False,
    linear_srgb: bool = False,
    has_post_processing: bool = False,
) -> List[PrimExportInfo]:
    """
    Export Gaussians to USD stage as points primitives or particle fields.

    Uses frame-first loop ordering: iterates over timestamps in outer loop,
    tracks in inner loop. This avoids recomputing attributes for each track
    since get_attributes_at_timestamp returns data for all gaussians at once.

    Modifies the provided stage in-place. Supports both track-based (rigid body)
    and non-track-based export paths. Caller must configure stage upAxis and
    timeCodesPerSecond before calling.

    Args:
        accessor: Export accessor wrapping the Gaussian model
        stage: USD stage to export to (must have upAxis and timeCodesPerSecond set)
        content_root_path: Root path for content prims
        schema_type: USD schema type (geompoints or lightfield)
        percentage_gaussians: Percentage of Gaussians to export (subsampling)
        node_layer_name: Optional layer name for composite models
        force_sh_0: Force SH degree to 0 (skip f_rest coefficients)
        preactivation: Export pre-activation parameter values
        do_not_cast_shadows: Set shadow casting disabled flag
        skip_gaussian_deformation: Skip Gaussian deformation animation
        usd_timestamp_offset_us: Timestamp offset for time codes
        reference_timestamps_us: Reference timestamps for resampling tracks and temporal appearance
        sorting_mode_hint: LightField sorting mode hint (if None, auto-detected from renderer config)
        half_precision: Use half-precision (float16) attributes for LightField schema
        linear_srgb: Whether to use linear sRGB color space. True iff a runtime
            PPISP stage downstream of MDL will undo the linearization (i.e.
            ``spg-runtime`` mode with a PPISP module). The orchestrator
            computes this via :func:`compute_runtime_post_processing`.
        has_post_processing: Whether a runtime PPISP stage downstream of MDL
            will handle color correction (disables MDL's built-in sRGB-to-linear
            and inverse-tonemap). Same gate as ``linear_srgb``; ``sh-optimized``
            mode bakes PPISP into the SH and therefore passes ``False`` here so
            MDL applies its own color correction.

    Returns:
        List of PrimExportInfo objects containing export statistics for each prim
    """
    num_gaussians = accessor.get_num_gaussians()
    log.info(f"Exporting {node_layer_name if node_layer_name else 'model'} with {num_gaussians} Gaussians to USD stage")

    computed_offset_us: int = usd_timestamp_offset_us if usd_timestamp_offset_us is not None else 0
    capabilities = accessor.get_capabilities()
    tps = stage.GetTimeCodesPerSecond()
    animate_albedo = capabilities.has_temporal_appearance

    # Use provided sorting mode hint (already determined from renderer config)
    if sorting_mode_hint is None:
        sorting_mode_hint = "cameraDistance"  # Default fallback

    # Compute valid gaussian mask to filter out NaN/Inf values
    # Use first reference timestamp if available, otherwise use computed offset
    nan_check_timestamp = reference_timestamps_us[0] if reference_timestamps_us else computed_offset_us
    valid_gaussian_mask = accessor.get_valid_gaussian_mask(nan_check_timestamp, preactivation=preactivation)
    num_invalid = num_gaussians - np.sum(valid_gaussian_mask)
    if num_invalid > 0:
        log.info(f"Filtering {num_invalid} Gaussians with NaN/Inf values ({100 * num_invalid / num_gaussians:.4f}%)")

    if capabilities.has_rigid_tracks:
        # =====================================================================
        # Track-based export with frame-first loop ordering
        # =====================================================================
        tracks = accessor.get_cuboid_tracks()
        log.info(f"Exporting {len(tracks.tracks_id)} tracks as separate prims")

        # Phase 1: Prepare all track contexts and collect all timestamps
        track_contexts, all_timestamps = _prepare_track_contexts(
            accessor=accessor,
            stage=stage,
            schema_type=schema_type,
            capabilities=capabilities,
            content_root_path=content_root_path,
            node_layer_name=node_layer_name,
            percentage_gaussians=percentage_gaussians,
            reference_timestamps_us=reference_timestamps_us,
            sorting_mode_hint=sorting_mode_hint,
            half_precision=half_precision,
            linear_srgb=linear_srgb,
            has_post_processing=has_post_processing,
            valid_gaussian_mask=valid_gaussian_mask,
        )

        if not track_contexts:
            log.warning("No tracks to export")
            return []

        # Log subsampling and track coverage
        total_tracks_in_model = len(tracks.tracks_id)
        total_gaussians_after = sum(len(ctx.gaussian_indices) for ctx in track_contexts)
        log.info(
            f"Tracks: {len(track_contexts)} exported (of {total_tracks_in_model} in model). "
            f"Gaussians: {num_gaussians} total -> {total_gaussians_after} after NaN/Inf filtering and {percentage_gaussians:.1f}% subsampling"
        )

        # Loop only over frames where at least one track has valid data (all_timestamps).
        # Using the full reference timeline would visit frames where no track has poses and we'd
        # set all tracks invisible there, making them appear "removed" when opening at default time.
        frames_to_export = all_timestamps
        # Phase 2: Loop over frames (outer), then tracks (inner)
        log.debug(
            "Exporting %s frames for %s tracks",
            len(frames_to_export),
            len(track_contexts),
        )
        _pose_count_checked = False
        for timestamp_us in tqdm(frames_to_export, desc="Exporting frames", unit="frame"):
            # Get attributes once for this frame (all gaussians + track poses + per-track visibility)
            attributes, interpolated_track_poses, track_visibility_mask = accessor.get_attributes_at_timestamp(
                timestamp_us, preactivation=preactivation
            )

            # One-time sanity check: pose array length must match model track count
            if not _pose_count_checked and interpolated_track_poses is not None:
                _pose_count_checked = True
                n_poses = len(interpolated_track_poses)
                if n_poses != total_tracks_in_model:
                    log.warning(
                        "USD export: interpolated_track_poses length (%d) != model track count (%d); "
                        "track transforms may be misaligned for some tracks",
                        n_poses,
                        total_tracks_in_model,
                    )

            time_code = _compute_time_code(tps, timestamp_us, computed_offset_us)
            for ctx in track_contexts:
                is_visible = timestamp_us in ctx.timestamps_set and (
                    track_visibility_mask is None
                    or (
                        ctx.track_enum_index < len(track_visibility_mask)
                        and track_visibility_mask[ctx.track_enum_index]
                    )
                )
                if is_visible:
                    _write_track_frame(
                        ctx=ctx,
                        attributes=attributes,
                        interpolated_track_poses=interpolated_track_poses,
                        timestamp_us=timestamp_us,
                        capabilities=capabilities,
                        computed_offset_us=computed_offset_us,
                        tps=tps,
                        skip_gaussian_deformation=skip_gaussian_deformation,
                        force_sh_0=force_sh_0,
                        do_not_cast_shadows=do_not_cast_shadows,
                        animate_albedo=animate_albedo,
                    )
                    ctx.writer.write_track_visibility(time_code, True)
                # Do not author "invisible" when out of range: it makes tracks disappear at default
                # playhead time. We only write "visible" when in range; otherwise no sample (default visible).

        # Phase 3: Finalize all writers and collect export info
        prim_export_info: List[PrimExportInfo] = []
        for ctx in tqdm(track_contexts, desc="Finalizing tracks", unit="track"):
            ctx.writer.finalize(ctx.all_positions)
            # Collect export info for summary
            prim_path = ctx.writer.prim.GetPath().pathString if ctx.writer.prim else "unknown"
            prim_export_info.append(
                PrimExportInfo(
                    prim_name=prim_path,
                    track_id=ctx.track_id,
                    track_index=ctx.track_enum_index,
                    num_gaussians=len(ctx.gaussian_indices),
                    num_timesamples=len(ctx.timestamps_us),
                    timestamps_us=ctx.timestamps_us,
                )
            )
        return prim_export_info

    else:
        # =====================================================================
        # Non-track export: single prim for all Gaussians
        # =====================================================================
        gaussian_indices = torch.arange(num_gaussians)

        # Filter out invalid Gaussians (NaN/Inf) before subsampling
        valid_mask_tensor = torch.from_numpy(valid_gaussian_mask)
        gaussian_indices = gaussian_indices[valid_mask_tensor]

        gaussian_indices = apply_gaussian_subsampling(gaussian_indices, percentage_gaussians)

        log.info(
            f"Gaussians: {num_gaussians} total -> {len(gaussian_indices)} after NaN/Inf filtering and {percentage_gaussians:.1f}% subsampling"
        )

        # Determine time samples for non-track case
        non_track_time_samples: List[int]
        if capabilities.has_temporal_appearance and reference_timestamps_us is not None:
            non_track_time_samples = reference_timestamps_us
        else:
            non_track_time_samples = [computed_offset_us]

        # Create writer and prim
        writer = create_gaussian_writer(
            schema_type=schema_type,
            stage=stage,
            capabilities=capabilities,
            track_id="background",
            track_index=0,
            node_layer_name=node_layer_name,
            content_root_path=content_root_path,
            sorting_mode_hint=sorting_mode_hint,
            half_precision=half_precision,
            linear_srgb=linear_srgb,
            has_post_processing=has_post_processing,
        )
        writer.create_prim(num_gaussians=len(gaussian_indices), track_transforms=None)

        # Create a single track context for the non-track case
        ctx = TrackExportContext(
            track_id="background",
            track_enum_index=0,
            gaussian_indices=gaussian_indices,
            timestamps_us=non_track_time_samples,
            timestamps_set=set(non_track_time_samples),
            writer=writer,
        )

        # Loop over frames
        for timestamp_us in tqdm(non_track_time_samples, desc="Exporting frames", unit="frame"):
            attributes, interpolated_track_poses, _ = accessor.get_attributes_at_timestamp(
                timestamp_us, preactivation=preactivation
            )

            _write_track_frame(
                ctx=ctx,
                attributes=attributes,
                interpolated_track_poses=interpolated_track_poses,
                timestamp_us=timestamp_us,
                capabilities=capabilities,
                computed_offset_us=computed_offset_us,
                tps=tps,
                skip_gaussian_deformation=skip_gaussian_deformation,
                force_sh_0=force_sh_0,
                do_not_cast_shadows=do_not_cast_shadows,
                animate_albedo=animate_albedo,
            )

        writer.finalize(ctx.all_positions)

        # Collect export info for summary
        prim_path = ctx.writer.prim.GetPath().pathString if ctx.writer.prim else "unknown"
        prim_export_info = [
            PrimExportInfo(
                prim_name=prim_path,
                track_id=ctx.track_id,
                track_index=ctx.track_enum_index,
                num_gaussians=len(ctx.gaussian_indices),
                num_timesamples=len(ctx.timestamps_us),
                timestamps_us=ctx.timestamps_us,
            )
        ]
        return prim_export_info


# -----------------------------------------------------------------------------
# Export Summary
# -----------------------------------------------------------------------------


@dataclass
class PrimExportInfo:
    """Information about an exported prim."""

    prim_name: str
    track_id: str
    track_index: int  # Index in the layer's cuboid_tracks (for matching to poses / GT)
    num_gaussians: int
    num_timesamples: int
    timestamps_us: List[int]


@dataclass
class ExportSummary:
    """Summary of USD export statistics."""

    schema_type: str
    total_prims: int
    total_gaussians: int
    total_unique_timesamples: int
    sh_degree: Optional[int]
    has_temporal_appearance: bool
    has_deformation: bool
    has_rigid_tracks: bool
    prims: List[PrimExportInfo]
    output_path: Path
    percentage_gaussians: float

    def to_text(self) -> str:
        """Generate a human-readable text summary.

        Format is designed to be easily comparable across exports.
        """
        lines = []
        lines.append("=" * 100)
        lines.append("GAUSSIAN USD EXPORT SUMMARY")
        lines.append("=" * 100)
        lines.append(f"Output: {self.output_path}")
        lines.append(f"Schema: {self.schema_type}")
        lines.append(f"Subsampling: {self.percentage_gaussians:.1f}%")
        lines.append("")
        lines.append("-" * 100)
        lines.append("OVERALL STATISTICS")
        lines.append("-" * 100)
        lines.append(f"Total prims exported:        {self.total_prims:>10}")
        lines.append(f"Total gaussians:             {self.total_gaussians:>10}")
        lines.append(f"Total unique timesamples:    {self.total_unique_timesamples:>10}")
        lines.append(f"Spherical harmonics degree:  {self.sh_degree if self.sh_degree is not None else 'N/A':>10}")
        lines.append(f"Has temporal appearance:     {str(self.has_temporal_appearance):>10}")
        lines.append(f"Has deformation:             {str(self.has_deformation):>10}")
        lines.append(f"Has rigid tracks:            {str(self.has_rigid_tracks):>10}")
        lines.append("")
        lines.append("-" * 100)
        lines.append("PER-PRIM DETAILS")
        lines.append("-" * 100)
        lines.append(f"{'Prim Name':<50} {'Track ID':<20} {'Idx':>4} {'Gaussians':>12} {'Timesamples':>12}")
        lines.append("-" * 100)
        for prim_info in self.prims:
            # Truncate from left with ".." prefix to show truncation occurred
            if len(prim_info.prim_name) > 50:
                prim_name_short = ".." + prim_info.prim_name[-48:]
            else:
                prim_name_short = prim_info.prim_name
            if len(prim_info.track_id) > 20:
                track_id_short = ".." + prim_info.track_id[-18:]
            else:
                track_id_short = prim_info.track_id
            lines.append(
                f"{prim_name_short:<50} {track_id_short:<20} {prim_info.track_index:>4} "
                f"{prim_info.num_gaussians:>12} {prim_info.num_timesamples:>12}"
            )
        lines.append("=" * 100)
        return "\n".join(lines)


def write_export_summary(summary: ExportSummary, output_path: Path) -> None:
    """Write export summary to a text file.

    Args:
        summary: Export summary data
        output_path: Path to write summary file
    """
    summary_path = output_path / "export_summary.txt"
    summary_text = summary.to_text()
    with open(summary_path, "w") as f:
        f.write(summary_text)
    log.info(f"Export summary written to: {summary_path}")


# -----------------------------------------------------------------------------
# CLI Command
# -----------------------------------------------------------------------------


@click.command("export-gaussian-usd-asset")
@click.option(
    "--config-name", type=str, required=True, help="Hydra config to load - has to contain a dataset specification"
)
@click.option("--output-dir", type=str, required=False, help="Path to the output target directory")
@click.option(
    "--percentage-gaussians",
    type=float,
    default=100.0,
    help="Percentage of Gaussians to export (0, 100].",
)
@click.option(
    "--usd-format",
    type=click.Choice(["usda", "usdc", "usd", "usdz"], case_sensitive=False),
    default="usdz",
    help="USD output format.",
)
@click.option(
    "--usd-schema",
    type=click.Choice([schema.value for schema in GaussianUSDSchemaType], case_sensitive=False),
    default="lightfield",
    help="USD schema type.",
)
@click.option("--force-sh-0", is_flag=True, default=False, help="Force SH degree to 0 (skip f_rest coefficients)")
@click.option(
    "--apply-activation/--no-apply-activation",
    default=True,
    help="Apply activations to parameters (export post-activation)",
)
@click.option(
    "--export-parsed-extra-signals",
    is_flag=True,
    default=False,
    help="Export parsed extra signal components as primvars (not yet implemented)",
)
@click.option("--export-rig-trajectories", is_flag=True, default=True, help="Export rig trajectories")
@click.option("--flip-axis", type=str, default="", help="Axes to flip in 'xyz' form")
@click.option("--do-not-cast-shadows", is_flag=True, default=False, help="Author doNotCastShadows primvar")
@click.option("--skip-gaussian-deformation", is_flag=True, default=False, help="Disable gaussian deformation animation")
@click.option(
    "--resample-animation/--no-resample-animation",
    default=True,
    help="Resample track timestamps to consecutive subset of rig timestamps (default: enabled)",
)
@click.option(
    "--half-precision",
    is_flag=True,
    default=False,
    help="Use half-precision (float16) for LightField schema attributes to reduce file size",
)
@click.option(
    "--export-summary",
    is_flag=True,
    default=False,
    help="Generate export summary with prim counts, gaussian counts, and timesample information",
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="info",
    help="Logging level for export messages (e.g. debug to see track clipping).",
)
@click.option(
    "--ppisp-integration-mode",
    type=click.Choice(list(VALID_PPISP_INTEGRATION_MODES), case_sensitive=False),
    default=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
    show_default=True,
    help=(
        "How PPISP effects appear in the USD asset. "
        "'spg-runtime' (default) authors PPISP as Omniverse SPG shaders applied at render "
        "time -- highest fidelity, requires SPG-capable target. "
        "'sh-optimized' folds PPISP into the Gaussian SH coefficients so PPISP-corrected "
        "output is visible in any USD viewer (lower fidelity, but no runtime PPISP needed)."
    ),
)
@click.option(
    "--scene-radiance-scale",
    type=float,
    default=1.0,
    show_default=True,
    help=(
        "Global multiplicative scale applied to Gaussian SH coefficients before export "
        "(strictly positive, finite). Used to retune the asset's brightness for "
        "integration into a target USD scene without re-training. Layered on top of "
        "the optional SH-match output."
    ),
)
@click.option(
    "--ppisp-responsivity",
    type=float,
    default=1.0,
    show_default=True,
    help=(
        "Achromatic HDR multiplier authored as the inputs:responsivity USD attribute on "
        "every PPISP shader (strictly positive, finite). Default 1.0 is a no-op; "
        "consumers can override at runtime to retune brightness without re-baking. Only "
        "consumed in spg-runtime mode."
    ),
)
@click.option(
    "--ppisp-reference-camera-id",
    type=int,
    default=None,
    help=(
        "PPISP camera index used as reference. In sh-optimized mode this selects the "
        "camera whose vignetting/CRF gets folded into the SH; in spg-runtime mode it "
        "pins the static SPG export to that camera (and to a frame if "
        "--ppisp-reference-frame-id is also set). Defaults to 0 in sh-optimized; unset "
        "(time-sampled authoring) in spg-runtime."
    ),
)
@click.option(
    "--ppisp-reference-frame-id",
    type=int,
    default=None,
    help=(
        "PPISP frame index used as reference. Same dual semantics as "
        "--ppisp-reference-camera-id. spg-runtime rejects setting frame_id without "
        "camera_id because vignetting/CRF live on the camera axis."
    ),
)
@click.option(
    "--enable-ppisp-controller-export/--disable-ppisp-controller-export",
    default=None,
    help=(
        "Control whether to export the per-camera PPISP controller plus the "
        "automatic-parameter PPISP shader on every RenderProduct instead of the "
        "static path. When omitted, spg-runtime exports use the controller whenever "
        "the loaded PPISP module was trained with controllers. Use --disable to "
        "force the static SPG path. Only valid in spg-runtime mode."
    ),
)
@click.option(
    "--sh-optimization-num-iterations",
    type=int,
    default=None,
    help=(
        "Number of optimization steps for the SH-match loop, overriding "
        "SHMatchingOptimizationConfig.num_iterations. Only meaningful in "
        "sh-optimized mode (rejected otherwise). Use this to trade export time "
        "against SH-match quality without touching the tuned learning rates "
        "and device, which remain non-CLI-tunable. Strictly positive int."
    ),
)
@click.argument("hydra-args", nargs=-1)
def export_gaussian_usd_asset(
    config_name: str,
    output_dir: Optional[str],
    percentage_gaussians: float,
    usd_format: str,
    usd_schema: str,
    force_sh_0: bool,
    apply_activation: bool,
    export_parsed_extra_signals: bool,
    export_rig_trajectories: bool,
    flip_axis: str,
    do_not_cast_shadows: bool,
    skip_gaussian_deformation: bool,
    resample_animation: bool,
    half_precision: bool,
    export_summary: bool,
    log_level: str,
    ppisp_integration_mode: str,
    scene_radiance_scale: float,
    ppisp_responsivity: float,
    ppisp_reference_camera_id: Optional[int],
    ppisp_reference_frame_id: Optional[int],
    enable_ppisp_controller_export: Optional[bool],
    sh_optimization_num_iterations: Optional[int],
    hydra_args: list[str],
    *,
    sh_optimization_config: Optional[SHMatchingOptimizationConfig] = None,
) -> None:
    """CLI command to export Gaussian model checkpoints to USD format.

    Loads a checkpoint, extracts Gaussian data, and exports to USD with
    optional rig trajectory data and USDZ packaging.

    PPISP integration parameters
    ----------------------------
    ppisp_integration_mode
        How PPISP effects manifest in the USD asset. ``"spg-runtime"``
        (default) authors PPISP as Omniverse SPG shaders applied at
        runtime; ``"sh-optimized"`` folds PPISP into Gaussian SH
        coefficients so renderers without SPG support still see
        PPISP-corrected output.
    scene_radiance_scale
        Global multiplicative scale applied to Gaussian SH
        coefficients before export. Used to retune the asset's
        brightness for integration into a target USD scene without
        re-training. Strictly positive, finite. Applied after
        SH-match optimization so the scaling layers on top of the
        PPISP-folded coefficients.
    ppisp_responsivity
        Achromatic HDR multiplier authored as the
        ``inputs:responsivity`` USD attribute on every PPISP shader.
        Default ``1.0`` is a no-op; consumers in target USD scenes
        can override it at runtime to retune asset brightness without
        re-baking. Strictly positive, finite. Only consumed in
        ``spg-runtime`` mode.
    ppisp_reference_camera_id, ppisp_reference_frame_id
        PPISP indices used as reference. In ``sh-optimized`` mode
        they pick the PPISP config that gets baked into SH; in
        ``spg-runtime`` mode they pin the static SPG export to one
        camera and (optionally) one frame. ``spg-runtime`` rejects a
        ``frame_id`` set without ``camera_id`` because vignetting and
        CRF live on the camera axis.
    enable_ppisp_controller_export
        Tri-state controller export switch. ``None`` (CLI default)
        means ``spg-runtime`` exports use the controller whenever the
        loaded PPISP module was trained with controllers. ``True``
        forces the controller path; ``False`` forces the static SPG
        path. The generated CUDA controller sidecars require
        runtime-specific Omniverse SPG support.
    sh_optimization_num_iterations
        Optional override for the SH-match loop's iteration count.
        Only meaningful in ``sh-optimized`` mode (the orchestrator
        rejects setting this in any other mode). Strictly positive
        ``int`` when set. Use this to trade export time against
        SH-match quality without touching the tuned learning rates
        and device, which remain non-CLI-tunable.

    SH-match optimization config
    ----------------------------
    The SH-match algorithm's learning rates and device are
    deliberately *not* exposed as CLI flags: they are tuned for the
    algorithm and the user is not expected to override them.
    ``sh-optimized`` mode therefore uses
    :class:`SHMatchingOptimizationConfig`'s built-in defaults
    verbatim, optionally overriding only ``num_iterations`` from
    ``sh_optimization_num_iterations``. Programmatic callers who
    need a fully custom config (e.g. forcing the device to ``"cpu"``
    for tests) can pass one via the keyword-only
    ``sh_optimization_config`` parameter below; if both are
    provided, the iteration override wins on top of the explicit
    config.

    Programmatic escape hatch
    -------------------------
    sh_optimization_config (keyword-only)
        Direct Python callers that build a
        :class:`SHMatchingOptimizationConfig` programmatically can
        pass it here. When omitted in ``sh-optimized`` mode the
        orchestrator falls back to the dataclass defaults.
    """
    sh_optimization_config = resolve_sh_optimization_config(
        explicit_config=sh_optimization_config,
        ppisp_integration_mode=ppisp_integration_mode,
        num_iterations_override=sh_optimization_num_iterations,
    )

    validate_export_integration_args(
        ppisp_integration_mode=ppisp_integration_mode,
        scene_radiance_scale=scene_radiance_scale,
        ppisp_responsivity=ppisp_responsivity,
        sh_optimization_config=sh_optimization_config,
        enable_ppisp_controller_export=enable_ppisp_controller_export,
    )

    level = getattr(logging, log_level.upper())
    log.setLevel(level)
    schema_type = GaussianUSDSchemaType(usd_schema.lower())
    log.info(f"Using USD schema type: {schema_type.value}")

    # Validate half-precision is only used with LightField schema
    if half_precision and schema_type != GaussianUSDSchemaType.LIGHT_FIELD:
        raise click.BadParameter(
            f"--half-precision is only supported with --usd-schema=lightfield. "
            f"The {schema_type.value} schema uses UsdGeomPoints which has schema-locked float32 attributes."
        )

    if export_parsed_extra_signals:
        log.warning("--export-parsed-extra-signals is not yet implemented, flag will be ignored")

    # Parse config and load checkpoint
    config = parse_typed_config(config_name=config_name, hydra_args=[*hydra_args, "mode=val"])
    checkpoint_path: Path = Path(config.ckpt_dir) / "last.ckpt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Instantiate system without checkpoint, then load state dict with strict=False so that
    # older checkpoints (e.g. missing model.gaussians_strategy.invisible_steps.*) still load.
    resume = config.resume
    if resume is None:
        config.resume = str(checkpoint_path)

    system = make_system(config.system.name, config)
    checkpoint = torch.load(str(checkpoint_path), map_location="cuda", weights_only=False)
    system.load_state_dict(checkpoint["state_dict"], strict=False, assign=True)
    # Ensure model is on CUDA so Slang/collector ops (e.g. calib_gradient_mask) see CUDA tensors.
    system.cuda()
    system.resume = resume
    config.resume = resume
    assert isinstance(system, GaussiansSystem)

    # Setup output path
    output_path: Path
    if output_dir is None:
        output_path = Path(config.ckpt_dir).parent / "usd_asset"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract model and dataset info
    model = system.model
    dataset = make_dataset(config.dataset.name, config, split="train")
    datasource = dataset.get_datasource()
    dataset_offset = datasource.get_offset()

    # Parse axis flip flags
    flip_axis_lower = (flip_axis or "").lower()
    flip_x_axis = "x" in flip_axis_lower
    flip_y_axis = "y" in flip_axis_lower
    flip_z_axis = "z" in flip_axis_lower

    # Initialize export cache
    usdz_cache = USDGaussianExportCache()

    # ``model_has_postprocessing`` is a structural property of the source
    # model (does it carry a PPISP collection?) and only used internally
    # to acquire ``ppisp_module``. The orchestrator-level flag that drives
    # MDL color-correction + SH packing -- ``runtime_post_processing`` --
    # is computed below once the integration mode is validated, so the
    # ``sh-optimized`` path (which removes PPISP from the runtime pipeline
    # by folding it into the SH) does not mislead MDL about a downstream
    # PPISP stage that no longer exists.
    model_has_postprocessing = False
    ppisp_module = None
    if hasattr(model, "post_processings"):
        model_has_postprocessing = len(model.post_processings) > 0
        if model_has_postprocessing:
            ppisp_module = get_ppisp_from_model(model)

    # Export rig trajectories if available
    rig_stage_named: Optional[NamedUSDStage] = None
    timestamp_offset_us: int = 0
    rig_timestamps: List[int] = []
    camera_name_to_index: Dict[str, int] = {}

    if export_rig_trajectories:
        rig_stage_named, timestamp_offset_us, _, rig_timestamps = create_rig_trajectories_stage(
            datasource, frame_rate=DEFAULT_FRAME_RATE
        )

        # Build camera name to index mapping and extract camera render products for PPISP
        if isinstance(datasource, RigTrajectoriesProvider):
            rig_trajectories = datasource.get_rig_trajectories()
            camera_name_to_index = build_camera_name_to_index_mapping(rig_trajectories)
            log.info(f"Built camera name to index mapping: {camera_name_to_index}")

            camera_render_products = extract_camera_render_products(rig_trajectories)
            usdz_cache.set_camera_render_products(camera_render_products)
            log.info(f"Extracted {len(camera_render_products)} camera render products")

        if rig_stage_named is not None:
            usdz_cache.add_usd_stage(rig_stage_named)

    # PPISP integration deferred-presence checks. The static validation
    # above does not know whether the loaded model exposes a PPISP module;
    # raising here keeps the failure path right next to the dependent
    # branches.
    if ppisp_integration_mode == PPISP_INTEGRATION_MODE_SH_OPTIMIZED and ppisp_module is None:
        raise ValueError(
            "ppisp_integration_mode='sh-optimized' was requested but the loaded "
            "model has no PPISP module. SH-match needs PPISP to define the bake "
            "target."
        )
    if enable_ppisp_controller_export is True:
        if ppisp_module is None:
            raise ValueError(
                "enable_ppisp_controller_export was requested but the loaded model "
                "has no PPISP module. Controller export needs PPISP."
            )
        if not ppisp_has_controller(ppisp_module):
            raise ValueError(
                "enable_ppisp_controller_export was requested but the loaded PPISP "
                "module has no trained controllers. Controller export needs a PPISP "
                "module trained with PPISPConfig(use_controller=True)."
            )

    resolved_enable_ppisp_controller_export = resolve_ppisp_controller_export_enabled(
        requested=enable_ppisp_controller_export,
        ppisp_module=ppisp_module,
        ppisp_integration_mode=ppisp_integration_mode,
    )
    if resolved_enable_ppisp_controller_export and (
        ppisp_reference_camera_id is not None or ppisp_reference_frame_id is not None
    ):
        log.warning(
            "PPISP reference IDs are ignored when controller export is enabled; "
            "the controller predicts exposure and color latents at runtime."
        )
    # PPISP runtime shaders (spg-runtime mode, including controller export)
    # author per-camera prims keyed off the rig-trajectories camera index.
    # Without exported rig trajectories the SPG block downstream is silently
    # skipped, so fail fast here instead of producing a USD that is missing the
    # requested PPISP integration.
    if (
        ppisp_integration_mode == PPISP_INTEGRATION_MODE_SPG_RUNTIME
        and ppisp_module is not None
        and not (export_rig_trajectories and isinstance(datasource, RigTrajectoriesProvider))
    ):
        raise ValueError(
            "ppisp_integration_mode='spg-runtime' requires exported rig trajectories "
            "(export_rig_trajectories=True with a RigTrajectoriesProvider datasource) "
            "to author per-camera PPISP shaders. Either re-enable rig trajectories or "
            "switch to ppisp_integration_mode='sh-optimized' (bakes PPISP into Gaussian "
            "SH and needs no rig trajectories)."
        )

    # Resolve PPISP reference IDs once. The resolver enforces the dual
    # semantics documented on :class:`PPISPReferenceMode` and validates
    # IDs against the PPISP module bounds.
    if ppisp_module is not None and not (
        ppisp_integration_mode == PPISP_INTEGRATION_MODE_SPG_RUNTIME and resolved_enable_ppisp_controller_export
    ):
        resolved_ppisp_ids = resolve_ppisp_reference_ids(
            ppisp_reference_camera_id,
            ppisp_reference_frame_id,
            ppisp_module,
            mode=ppisp_integration_mode,  # type: ignore[arg-type]
        )
    else:
        resolved_ppisp_ids = None

    # Now that the integration mode is validated, derive the runtime
    # flag MDL/SH-packing actually need (see ``compute_runtime_post_processing``
    # docstring for the sh-optimized vs spg-runtime distinction).
    runtime_post_processing = compute_runtime_post_processing(
        model_has_postprocessing=model_has_postprocessing,
        ppisp_module=ppisp_module,
        ppisp_integration_mode=ppisp_integration_mode,
    )
    log.info(
        "PPISP integration: mode=%s, model_has_postprocessing=%s, runtime_post_processing=%s",
        ppisp_integration_mode,
        model_has_postprocessing,
        runtime_post_processing,
    )

    # SH-match optimization runs *before* the export loop so the USD
    # gaussian points carry already-folded PPISP. Mutates system.model
    # in place.
    if ppisp_integration_mode == PPISP_INTEGRATION_MODE_SH_OPTIMIZED:
        assert sh_optimization_config is not None  # validated above
        assert resolved_ppisp_ids is not None
        run_sh_match_optimization(
            system=system,
            ppisp_module=ppisp_module,
            train_dataset=dataset,
            config=sh_optimization_config,
            reference_camera_id=resolved_ppisp_ids.camera_id,
            reference_frame_id=resolved_ppisp_ids.frame_id,
        )

    with torch.no_grad():
        # Scene radiance scaling layers on top of (optional) SH-match output.
        if scene_radiance_scale != 1.0:
            log.info("Applying scene_radiance_scale=%.6f to Gaussian SH", float(scene_radiance_scale))
            apply_radiance_scale_to_sh_nodes(
                cast(Any, iter_gaussian_nodes(model)),
                float(scene_radiance_scale),
            )

        # Author the runtime PPISP SPG prims + sidecars only in spg-runtime mode.
        # In sh-optimized mode the SH-match loop above has already folded PPISP
        # into the Gaussian SH coefficients, so authoring the SPG shaders would
        # double-apply PPISP on SPG-capable viewers (Omniverse Kit / RTX) and ship
        # unnecessary sidecars in the USDZ for vanilla viewers. The downstream
        # ``spg_camera_id`` / ``spg_frame_id`` mode checks are kept as in-place
        # defense-in-depth so the call site reads as self-contained.
        has_runtime_ppisp = False
        if (
            ppisp_integration_mode == PPISP_INTEGRATION_MODE_SPG_RUNTIME
            and ppisp_module is not None
            and camera_name_to_index
            and isinstance(datasource, RigTrajectoriesProvider)
        ):
            # NCOREDataSource is the only datasource currently exposing
            # ``get_global_to_training_camera_frame_mapping`` (used below), so
            # narrow with an explicit check rather than an ``assert`` which is
            # stripped under ``python -O``.
            if not isinstance(datasource, NCOREDataSource):
                raise TypeError(
                    "PPISP export requires NCOREDataSource (to provide the global -> "
                    "training camera-frame mapping); got "
                    f"{type(datasource).__name__!s}."
                )
            rig_trajectories = datasource.get_rig_trajectories()
            global_to_training = datasource.get_global_to_training_camera_frame_mapping()
            camera_frame_mappings = build_camera_frame_mappings(rig_trajectories, global_to_training)
            if camera_frame_mappings:
                total_frames = sum(len(m) for m in camera_frame_mappings.values())
                log.info(
                    f"PPISP: {len(camera_frame_mappings)} cameras, {total_frames} frames "
                    f"(mode={ppisp_integration_mode}, "
                    f"controller_disabled={not resolved_enable_ppisp_controller_export})"
                )
                # Reference IDs are only forwarded to the SPG writer in
                # spg-runtime mode (their effect on sh-optimized was
                # already absorbed into the SH coefficients above).
                spg_camera_id = (
                    resolved_ppisp_ids.camera_id
                    if (
                        resolved_ppisp_ids is not None
                        and resolved_ppisp_ids.camera_was_user_set
                        and ppisp_integration_mode == PPISP_INTEGRATION_MODE_SPG_RUNTIME
                    )
                    else None
                )
                spg_frame_id = (
                    resolved_ppisp_ids.frame_id
                    if (
                        resolved_ppisp_ids is not None
                        and resolved_ppisp_ids.frame_was_user_set
                        and ppisp_integration_mode == PPISP_INTEGRATION_MODE_SPG_RUNTIME
                    )
                    else None
                )
                usdz_cache.set_ppisp(
                    ppisp_module=ppisp_module,
                    camera_name_to_index=camera_name_to_index,
                    camera_frame_mappings=camera_frame_mappings,
                    responsivity=float(ppisp_responsivity),
                    ppisp_reference_camera_id=spg_camera_id,
                    ppisp_reference_frame_id=spg_frame_id,
                    ppisp_disable_controller=not resolved_enable_ppisp_controller_export,
                )
                has_runtime_ppisp = True
                usdz_cache.add_spg_files(
                    select_spg_files_for_export(
                        enable_ppisp_controller_export=resolved_enable_ppisp_controller_export,
                        ppisp_module=ppisp_module if resolved_enable_ppisp_controller_export else None,
                        camera_indices=(
                            sorted(set(camera_name_to_index.values()))
                            if resolved_enable_ppisp_controller_export
                            else None
                        ),
                    )
                )
            else:
                log.warning("Could not build PPISP frame mappings, skipping PPISP export")

        if schema_type == GaussianUSDSchemaType.NREND_VOLUME:
            add_nrend_volume_assets_to_cache(
                cache=usdz_cache,
                model=model,
                datasource=datasource,
                has_runtime_ppisp=has_runtime_ppisp,
            )
            log.info("Exported NuRec/NRend volume assets")

        # Determine stage file extension - use binary usdc for faster serialization
        stage_ext = "usdc" if usd_format.lower() == "usdz" else usd_format.lower()

        # Determine sorting mode hint from renderer config
        sorting_mode_hint = "cameraDistance"  # Default
        try:
            if hasattr(model, "config") and hasattr(model.config, "renderer"):
                global_z_order = getattr(model.config.renderer, "global_z_order", False)
                sorting_mode_hint = "zDepth" if global_z_order else "cameraDistance"
                log.info(f"Detected global_z_order={global_z_order} from renderer config")
        except Exception as e:
            log.warning(f"Could not determine sorting mode from renderer config: {e}")
        log.info(f"Using sorting mode hint: {sorting_mode_hint}")

        # Build node name -> model mapping and export each node as a separate USD stage.
        # The same per-node dispatch is used by the SH-match optimization path so a
        # single helper keeps both call sites in lock-step.
        nodes: Dict[str, Any] = (
            {} if schema_type == GaussianUSDSchemaType.NREND_VOLUME else dict(iter_gaussian_nodes(model))
        )

        # Collect export statistics for summary
        all_prim_info: List[PrimExportInfo] = []
        all_capabilities: Optional[ModelCapabilities] = None

        for node_name, gaussian_model in nodes.items():
            accessor = GaussianExportAccessor(
                gaussian_model,
                group=model.gaussian_process_group,
            )
            filename = f"{node_name}_gaussians.{stage_ext}" if len(nodes) > 1 else f"gaussians.{stage_ext}"
            node_layer_name = node_name if len(nodes) > 1 else None

            gauss_stage = initialize_usd_stage()
            UsdGeom.SetStageUpAxis(gauss_stage, UsdGeom.Tokens.y)
            gauss_stage.SetTimeCodesPerSecond(DEFAULT_FRAME_RATE)
            root_path = create_gaussian_model_root(
                gauss_stage,
                flip_x_axis=flip_x_axis,
                flip_y_axis=flip_y_axis,
                flip_z_axis=flip_z_axis,
                dataset_offset=dataset_offset,
            )

            node_prim_info = export_gaussians_as_usd_asset(
                accessor=accessor,
                stage=gauss_stage,
                percentage_gaussians=percentage_gaussians,
                node_layer_name=node_layer_name,
                force_sh_0=force_sh_0,
                preactivation=(not apply_activation),
                do_not_cast_shadows=do_not_cast_shadows,
                skip_gaussian_deformation=skip_gaussian_deformation,
                usd_timestamp_offset_us=timestamp_offset_us,
                reference_timestamps_us=rig_timestamps if resample_animation else None,
                content_root_path=root_path,
                schema_type=schema_type,
                sorting_mode_hint=sorting_mode_hint,
                half_precision=half_precision,
                linear_srgb=runtime_post_processing,
                has_post_processing=runtime_post_processing,
            )
            all_prim_info.extend(node_prim_info)
            # Log export manifest (index, track_id) per node for comparison with GT
            if len(node_prim_info) > 1:
                manifest = [(p.track_index, p.track_id) for p in node_prim_info]
                log.debug(
                    "USD export: %s tracks for node '%s' (index, track_id): %s",
                    len(manifest),
                    node_name,
                    manifest,
                )
            if all_capabilities is None:
                all_capabilities = accessor.get_capabilities()

            usdz_cache.add_usd_stage(NamedUSDStage(filename=filename, stage=gauss_stage))

        # Export background as USD DomeLight if applicable.
        # TODO(spg-reintro): when external PPISP SPG export is wired back in, switch
        # this default to neutral 1.0 (so the SPG owns exposure) and align with the
        # per-RenderProduct camera shim (see ideutsch/ppisp-update-spg-snapshot).
        background_model = getattr(system.model, "background", None)
        if schema_type != GaussianUSDSchemaType.NREND_VOLUME and background_model is not None:
            result = create_background_domelight(background_model, output_path, intensity=DOME_LIGHT_DEFAULT_INTENSITY)
            if result is not None:
                named_stage, named_hdr = result
                usdz_cache.add_usd_stage(named_stage)
                usdz_cache.hdr_files.append(named_hdr)
                log.info(f"Exported background as USD DomeLight: {named_stage.filename}")

        # Write final output
        if usd_format.lower() == "usdz":
            dataset_basename = Path(config.dataset.path).stem
            usdz_filename = f"{dataset_basename}.usdz"
            usdz_path = output_path / usdz_filename
            usdz_cache.write_to_usdz(usdz_path)
        else:
            # Save all cached stages and HDR files
            for usd_stage in usdz_cache.usd_stages:
                usd_stage.save(output_path)
            for hdr_file in usdz_cache.hdr_files:
                hdr_file.save(output_path)
            for serialized_file in usdz_cache.serialized_files:
                serialized_file.save(output_path)
            for spg_file in usdz_cache.spg_files:
                spg_file.save(output_path)

            # Create and save composition file (preserve references to other stages)
            if usdz_cache.usd_stages:
                usd_default_layer = NamedUSDStage(
                    filename="default.usda",
                    stage=usdz_cache.compose_default_usd_stage(),
                )
                usd_default_layer.save(output_path, preserve_references=True)
                log.info(f"Export complete: {output_path / 'default.usda'}")

        # Generate export summary if requested
        if export_summary and all_prim_info and all_capabilities is not None:
            # Collect all unique timestamps
            all_timestamps_set: Set[int] = set()
            for prim_info in all_prim_info:
                all_timestamps_set.update(prim_info.timestamps_us)

            summary = ExportSummary(
                schema_type=schema_type.value,
                total_prims=len(all_prim_info),
                total_gaussians=sum(p.num_gaussians for p in all_prim_info),
                total_unique_timesamples=len(all_timestamps_set),
                sh_degree=0 if force_sh_0 else all_capabilities.sh_degree,
                has_temporal_appearance=all_capabilities.has_temporal_appearance,
                has_deformation=all_capabilities.has_deformation,
                has_rigid_tracks=all_capabilities.has_rigid_tracks,
                prims=all_prim_info,
                output_path=output_path,
                percentage_gaussians=percentage_gaussians,
            )
            write_export_summary(summary, output_path)


if __name__ == "__main__":
    export_gaussian_usd_asset()
