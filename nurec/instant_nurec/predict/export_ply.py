# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path

import torch

from nurec.instant_nurec.config.predict import PrimitivePLYExportConfig
from nurec.instant_nurec.primitives.base import BaseInstantNuRecPrimitive
from nurec.instant_nurec.primitives.celsius_primitive import CelsiusInstantNuRecPrimitive
from nurec.instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive, KelvinSemanticClass
from nurec.models.gaussians.utils import RGB2SH, PLYGaussianLoader, write_ply_3dgs
from nurec.models.utils import get_activation
from nurec.utils.types import RigTrajectories


logger = logging.getLogger(__name__)


@dataclass
class PLYExportGaussians:
    positions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    densities: torch.Tensor
    rgb: torch.Tensor
    road_mask: torch.Tensor | None = None
    sky_mask: torch.Tensor | None = None
    normals: torch.Tensor | None = None

    def _encoded_tensors(
        self, config: PrimitivePLYExportConfig
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode activated instant_nurec fields exactly as the PLY writer expects."""
        scales = self.scales.float()
        if config.scale_activation is not None:
            scale_activation_inv = get_activation(config.scale_activation, inverse=True)
            scales = scale_activation_inv(scales)

        densities = self.densities.float()
        if config.density_activation is not None:
            density_activation_inv = get_activation(config.density_activation, inverse=True)
            densities = density_activation_inv(densities)

        rgb = self.rgb.float()
        features_albedo = rgb if config.color_mode == "rgb" else RGB2SH(rgb)
        return scales, densities, rgb, features_albedo

    def _to_loader(
        self,
        config: PrimitivePLYExportConfig,
        *,
        device: str | torch.device | None = None,
    ) -> PLYGaussianLoader:
        """Convert activated instant_nurec outputs to the tensor schema read from an exported PLY.

        The returned loader owns copied tensors without first materializing a file.
        """
        scales, densities, _, features_albedo = self._encoded_tensors(config)
        sky_mask = self.sky_mask.float() if self.sky_mask is not None else None
        if sky_mask is not None and sky_mask.ndim == 1:
            # PLYGaussianLoader presents scalar sky-mask properties as [N, 1].
            sky_mask = sky_mask.unsqueeze(-1)

        return PLYGaussianLoader.from_tensors(
            positions=self.positions.float(),
            rotations=self.rotations.float(),
            densities=densities,
            scales=scales,
            features_albedo=features_albedo,
            road_mask=self.road_mask.bool() if self.road_mask is not None else None,
            sky_mask=sky_mask,
            quaternion_format="wxyz",
            device=device,
        )

    def export(self, config: PrimitivePLYExportConfig, output_path: Path) -> None:
        """
        Export the gaussians to the PLY file used in SO, and values are exported as NuRec expects them.
        Colors are exported in SH, while scale and density are exported preactivated.
        """
        scales, densities, rgb, features_albedo = self._encoded_tensors(config)

        custom_attributes = {}
        if self.road_mask is not None:
            custom_attributes["road_mask"] = self.road_mask
        if self.sky_mask is not None:
            custom_attributes["sky_mask"] = self.sky_mask

        write_ply_3dgs(
            path=output_path,
            positions=self.positions.float(),
            rotations=self.rotations.float(),
            scales=scales,
            densities=densities,
            features_albedo=features_albedo,
            color=rgb,
            normals=self.normals.float() if self.normals is not None else None,
            custom_attributes=custom_attributes,
        )


def get_gaussian_shape_pruning_mask(
    config: PrimitivePLYExportConfig, densities: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """
    Get the gaussian pruning mask for the given primitive, based on the shape of the gaussians.
    Returns [num_gaussians, 1] mask tensor.
    """

    mask = torch.isfinite(densities)
    logger.info(f"Removed {(~mask).sum().item():,} non-finite density gaussians.")
    logger.info(f"Starting to filter {mask.sum().item():,} finite gaussians.")

    if config.minimum_density is not None:
        logger.info(f"Filter (density) removed {torch.sum(mask & (densities < config.minimum_density))} gaussians")
        mask = mask & (densities >= config.minimum_density)

    if config.minimum_scale is not None:
        logger.info(
            f"Filter (scale) removed {torch.sum(mask & (torch.min(scales, dim=1).values < config.minimum_scale).unsqueeze(1))} gaussians"
        )
        mask = mask & (torch.min(scales, dim=1).values >= config.minimum_scale).unsqueeze(1)

    if config.minimum_surface_area is not None:
        logger.info(
            f"Filter (surface area) removed {torch.sum(mask & (scales.square().sum(dim=1) < config.minimum_surface_area).unsqueeze(1))} gaussians"
        )
        mask = mask & (scales.square().sum(dim=1) >= config.minimum_surface_area).unsqueeze(1)

    return mask


def export_celsius_ply(
    config: PrimitivePLYExportConfig, primitives: CelsiusInstantNuRecPrimitive
) -> PLYExportGaussians:
    """
    Notes: This is a work in progress and will change. Currently only static layers are handled (dynamic
    gaussians are filted out via the maximum velocity param). Because we do not yet export the affine matrix,
    we apply this directly to the RGB values when exporting a single camera.
    """

    if config.falloff_sigma_timestamp_us is None:
        densities = primitives.densities
    else:
        min_timestamp = torch.min(primitives.timestamps_us).cpu().item()
        if config.falloff_sigma_timestamp_us == -1:
            timestamp_us = torch.max(primitives.timestamps_us).cpu().item()
        else:
            timestamp_us = min_timestamp + config.falloff_sigma_timestamp_us
        timestamps_us = torch.Tensor([timestamp_us, timestamp_us]).to(torch.uint64).unsqueeze(0)
        gaussians = primitives.get_gaussian_parameters(timestamps_us)
        densities = gaussians["densities"]

    mask = get_gaussian_shape_pruning_mask(config, densities, primitives.scales)

    if primitives.dynamic_bbox_mask is not None:
        logger.info(
            f"Filter (dynamic bbox mask) removed {torch.sum(mask & (primitives.dynamic_bbox_mask > 0.5))} gaussians"
        )
        mask = mask & (primitives.dynamic_bbox_mask <= 0.5)

    if primitives.forward_speed_mps is not None and config.maximum_velocity is not None:
        logger.info(
            f"Filter (forward speed mps :3) removed {torch.sum(mask & (torch.norm(primitives.forward_speed_mps[..., :3], dim=1) > config.maximum_velocity).unsqueeze(1))} gaussians"
        )
        mask = mask & (torch.norm(primitives.forward_speed_mps[..., :3], dim=1) <= config.maximum_velocity).unsqueeze(1)
        logger.info(
            f"Filter (forward speed mps 3:) removed {torch.sum(mask & (torch.norm(primitives.forward_speed_mps[..., 3:], dim=1) > config.maximum_velocity).unsqueeze(1))} gaussians"
        )
        mask = mask & (torch.norm(primitives.forward_speed_mps[..., 3:], dim=1) <= config.maximum_velocity).unsqueeze(1)

    mask = mask.squeeze()
    logger.info(f"After filtering, {torch.sum(mask)} gaussians remaining.")

    rgb = primitives.rgb[mask].float()
    # Temporary hack to apply affine matrix in the single camera case
    if primitives.affine_matrix is not None and primitives.affine_bias is not None:
        if config.apply_affine_mtx:
            if len(primitives.affine_matrix) == 1:
                rgb = torch.einsum("n p, q p -> n q", rgb, primitives.affine_matrix[0])
                rgb = torch.clamp(rgb + primitives.affine_bias[0], min=0.0, max=1.0)
                logger.info("Applying affine matrix to RGB values.")
            else:
                logger.warning("Skipping application affine matrix. This only works with a single camera.")

    return PLYExportGaussians(
        positions=primitives.positions[mask],
        rotations=primitives.rotations[mask],
        scales=primitives.scales[mask],
        densities=densities[mask],
        rgb=rgb,
        road_mask=primitives.road_mask[mask].squeeze(-1).to(dtype=torch.uint8)
        if primitives.road_mask is not None
        else None,
        sky_mask=primitives.sky_mask[mask].squeeze(-1).float() if primitives.sky_mask is not None else None,
    )


def export_kelvin_ply(config: PrimitivePLYExportConfig, primitives: KelvinInstantNuRecPrimitive) -> PLYExportGaussians:
    """Export the ply file, static layer only."""
    static_layer = primitives.static_layer

    mask = get_gaussian_shape_pruning_mask(config, static_layer.densities, static_layer.scales).squeeze(-1)
    if static_layer.semantic_class is not None and not config.keep_movable_gaussians:
        movable_mask = (static_layer.semantic_class == KelvinSemanticClass.MOVABLE).squeeze(-1)
        logger.info(f"Filter (MOVABLE semantic class) removed {torch.sum(mask & movable_mask)} gaussians")
        mask = mask & ~movable_mask
    logger.info(f"After filtering, {torch.sum(mask)} gaussians remaining.")

    static_layer = static_layer.mask(mask)

    # Derive road/sky masks from per-gaussian semantic class
    road_mask: torch.Tensor | None = None
    sky_mask: torch.Tensor | None = None
    if static_layer.semantic_class is not None:
        road_mask = (static_layer.semantic_class == KelvinSemanticClass.ROAD).squeeze(-1).to(dtype=torch.uint8)
        sky_mask = (static_layer.semantic_class == KelvinSemanticClass.SKY).squeeze(-1).float()

    return PLYExportGaussians(
        positions=static_layer.positions,
        rotations=static_layer.rotations,
        scales=static_layer.scales,
        densities=static_layer.densities,
        rgb=static_layer.rgb,
        road_mask=road_mask,
        sky_mask=sky_mask,
        normals=static_layer.normals,
    )


def _build_ply_export_gaussians(
    config: PrimitivePLYExportConfig,
    primitives: BaseInstantNuRecPrimitive,
    rig_trajectories: RigTrajectories,
) -> PLYExportGaussians:
    """Transform and filter instant_nurec primitives into activated export intermediates."""
    primitives = primitives.rigid_transform(rig_trajectories.T_world_base.to(dtype=torch.float32))

    if isinstance(primitives, CelsiusInstantNuRecPrimitive):
        gaussians_ply = export_celsius_ply(config, primitives)
    elif isinstance(primitives, KelvinInstantNuRecPrimitive):
        gaussians_ply = export_kelvin_ply(config, primitives)
    else:
        raise ValueError(f"Unsupported primitive type: {type(primitives)}")
    return gaussians_ply


def build_ply_gaussian_loader(
    config: PrimitivePLYExportConfig,
    primitives: BaseInstantNuRecPrimitive,
    rig_trajectories: RigTrajectories,
    *,
    device: str | torch.device | None = None,
) -> PLYGaussianLoader:
    """Build the exact in-memory payload that reloading the instant_nurec initialization PLY yields.

    Positions and wxyz rotations are transformed into NCore world space before the same
    pruning, inverse activations, float32 conversion, RGB-to-SH conversion, and semantic-mask
    conversion used by disk export.
    """
    return _build_ply_export_gaussians(config, primitives, rig_trajectories)._to_loader(config, device=device)


def export_ply(
    config: PrimitivePLYExportConfig,
    primitives: BaseInstantNuRecPrimitive,
    rig_trajectories: RigTrajectories,
    path: Path,
) -> None:
    """Export the instant_nurec Primitives as a ply file after transforming to world space and applying some filtering.
    This ply export is intended to be used as an initialization for NuRec SO.
    """
    _build_ply_export_gaussians(config, primitives, rig_trajectories).export(config, path)
