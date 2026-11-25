# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer 1: adapt NRE tensor layouts for owned gsplat accumulator modules.

GsplatLossAdapter owns gsplat modules and their configuration caches. Its
ordinary family methods keep the caller's gradient mode and the modules' hooks.
AccumulatorModuleLosses selects model data and scheduled weights; this layer adapts tensors,
semantic modes and bounded/flat inputs. Gsplat owns the loss mathematics, reductions,
tensor-contract validation and contiguity. Disabled members arrive as None;
active empty domains retain their tensors and autograd connections.
"""

import math

from typing import cast

import torch

from gsplat.losses import BG_GRID_NUM_COLS, BG_GRID_NUM_ROWS
from gsplat.losses.functional.accumulator import (
    BgGridLossAccumulator,
    BgTrackNodeSemanticLossAccumulator,
    CameraLossAccumulator,
    GaussianLossAccumulator,
    GroundGaussiansLossAccumulator,
    JointSemantics,
    LidarLossAccumulator,
    PackedSemantics,
    SSIMLossAccumulator,
)


GRID_NUM_CHANNELS = BG_GRID_NUM_ROWS * BG_GRID_NUM_COLS


# Every gsplat group writes ``loss_weight * mean(raw member loss)`` directly
# into its canonical slot. Consumers derive the slot count from this tuple.
ACCUMULATOR_LOSS_NAMES: tuple[str, ...] = (
    # Camera + semantic CE
    "rgb_l1_mean",
    "background_mse_mean",
    "semantic_cross_entropy_mean",
    # SSIM
    "ssim_ssim_mean",
    # LiDAR
    "lidar_l1_mean",
    "intensity_mse_mean",
    "raydrop_mse_mean",
    "background_lidar_mse_mean",
    # Gaussian regularizers + deform
    "gaussian_scale_abs_mean",
    "gaussian_density_abs_mean",
    "gaussian_z_scale_abs_mean",
    "out_of_bound_l1_mean",
    "deform_smoothness_abs_mean",
    # Background texture + bilateral grids
    "sky_env_map_background_total_variation_spatial_mean",
    "bilateral_grid_drift_identity_distance_mean",
    "bilateral_grid_per_camera_tv_total_variation_spatial_mean",
    "bilateral_grid_per_frame_spatial_tv_total_variation_spatial_mean",
    # Background track + node semantic
    "background_in_track_gaussian_bce_with_logits_mean",
    "node_semantic_gaussians_bce_with_logits_mean",
    # Ground Gaussian distortion
    "road_gaussians_abs_mean",
)
N_LOSS_SLOTS = len(ACCUMULATOR_LOSS_NAMES)
LOSS_SLOT_BY_NAME: dict[str, int] = {name: slot for slot, name in enumerate(ACCUMULATOR_LOSS_NAMES)}


CAMERA_LOSS_NAMES = ACCUMULATOR_LOSS_NAMES[:3]
LIDAR_LOSS_NAMES = ACCUMULATOR_LOSS_NAMES[4:8]
GAUSSIAN_LOSS_NAMES = ACCUMULATOR_LOSS_NAMES[8:13]
BG_GRID_LOSS_NAMES = ACCUMULATOR_LOSS_NAMES[13:17]
BG_TRACK_NODE_LOSS_NAMES = ACCUMULATOR_LOSS_NAMES[17:19]

# Fixed dispatch metadata is shared across calls rather than rebuilt per step.
_CAMERA_SLOTS = tuple(LOSS_SLOT_BY_NAME[name] for name in CAMERA_LOSS_NAMES)
_SSIM_SLOTS = (LOSS_SLOT_BY_NAME["ssim_ssim_mean"],)
_LIDAR_SLOTS = tuple(LOSS_SLOT_BY_NAME[name] for name in LIDAR_LOSS_NAMES)
_GAUSSIAN_SLOTS = tuple(LOSS_SLOT_BY_NAME[name] for name in GAUSSIAN_LOSS_NAMES)
_BG_GRID_SLOTS = tuple(LOSS_SLOT_BY_NAME[name] for name in BG_GRID_LOSS_NAMES)
_BG_TRACK_NODE_SLOTS = tuple(LOSS_SLOT_BY_NAME[name] for name in BG_TRACK_NODE_LOSS_NAMES)
_GROUND_SLOTS = (LOSS_SLOT_BY_NAME["road_gaussians_abs_mean"],)


def _validate_image(tensor: torch.Tensor, name: str, channels: int, dtype: torch.dtype) -> None:
    """Check the NRE image layout before reshaping for a gsplat dispatch."""
    assert tensor.ndim == 4 and tensor.shape[-1] == channels and tensor.dtype == dtype, (
        f"{name} must be {dtype} Tensor of shape [B,H,W,{channels}], got shape={tensor.shape}, dtype={tensor.dtype}"
    )


def _lidar_ray_pair(pred: torch.Tensor, gt: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_image(gt, f"{name}_gt", 1, torch.float32)
    assert pred.dtype == torch.float32 and pred.numel() == gt.numel(), (
        f"{name}_pred must be float32 with the same number of elements as its target"
    )
    return pred.reshape(-1), gt.reshape(-1)


class GsplatLossAdapter(torch.nn.Module):
    """Own gsplat dispatches and adapt each selected NRE family independently.

    Family methods are ordinary calls, not an autograd Function or a second loss
    scheduler. This module has no combined forward: AccumulatorModuleLosses calls only the
    active families and rebinds the vector returned by each dispatch.
    """

    camera_loss_accumulator: CameraLossAccumulator
    lidar_loss_accumulator: LidarLossAccumulator
    bg_grid_loss_accumulator: BgGridLossAccumulator
    gaussian_loss_accumulators: torch.nn.ModuleDict
    bg_track_node_semantic_loss_accumulators: torch.nn.ModuleDict
    ssim_loss_accumulators: torch.nn.ModuleDict
    ground_gaussians_loss_accumulators: torch.nn.ModuleDict

    def __init__(self):
        super().__init__()
        self.camera_loss_accumulator = CameraLossAccumulator()
        self.lidar_loss_accumulator = LidarLossAccumulator()
        self.bg_grid_loss_accumulator = BgGridLossAccumulator()
        self.gaussian_loss_accumulators = torch.nn.ModuleDict()
        self.bg_track_node_semantic_loss_accumulators = torch.nn.ModuleDict()
        self.ssim_loss_accumulators = torch.nn.ModuleDict()
        self.ground_gaussians_loss_accumulators = torch.nn.ModuleDict()

    def _get_gaussian_loss_accumulator(
        self, z_scale_threshold: float, *, densities_are_logits: bool = False
    ) -> GaussianLossAccumulator:
        key = f"{float(z_scale_threshold).hex()}:preactivation:{densities_are_logits}".replace(".", "_")
        if key not in self.gaussian_loss_accumulators:
            self.gaussian_loss_accumulators[key] = GaussianLossAccumulator(
                z_scale_threshold=z_scale_threshold,
                preactivation=True,
                densities_are_logits=densities_are_logits,
            )
        return cast(GaussianLossAccumulator, self.gaussian_loss_accumulators[key])

    def _get_bg_track_node_semantic_loss_accumulator(
        self,
        density_logits_min: float,
    ) -> BgTrackNodeSemanticLossAccumulator:
        key = f"density_logits_min:{float(density_logits_min).hex()}".replace(".", "_")
        if key not in self.bg_track_node_semantic_loss_accumulators:
            self.bg_track_node_semantic_loss_accumulators[key] = BgTrackNodeSemanticLossAccumulator(
                density_logits_min=density_logits_min
            )
        return cast(
            BgTrackNodeSemanticLossAccumulator,
            self.bg_track_node_semantic_loss_accumulators[key],
        )

    def _get_ssim_loss_accumulator(
        self,
        window_size: int,
        mask_mode: str,
        constant_mask_value: float,
    ) -> SSIMLossAccumulator:
        key = f"{window_size}:{mask_mode}:{constant_mask_value!r}".replace(".", "_")
        if key not in self.ssim_loss_accumulators:
            self.ssim_loss_accumulators[key] = SSIMLossAccumulator(
                window_size=window_size,
                mask_mode=mask_mode,
                constant_mask_value=constant_mask_value,
            )
        return cast(SSIMLossAccumulator, self.ssim_loss_accumulators[key])

    def _get_ground_gaussians_loss_accumulator(
        self,
        min_bias: float,
        range_bias: float,
        grid_len: float,
        rotation_lambda: float,
    ) -> GroundGaussiansLossAccumulator:
        key = ":".join(float(value).hex() for value in (min_bias, range_bias, grid_len, rotation_lambda)).replace(
            ".", "_"
        )
        if key not in self.ground_gaussians_loss_accumulators:
            self.ground_gaussians_loss_accumulators[key] = GroundGaussiansLossAccumulator(
                min_bias=min_bias,
                range_bias=range_bias,
                grid_len=grid_len,
                rotation_lambda=rotation_lambda,
                rotations_are_raw_wxyz=True,
            )
        return cast(GroundGaussiansLossAccumulator, self.ground_gaussians_loss_accumulators[key])

    @staticmethod
    def can_fuse_density_activation(node_count: int) -> bool:
        """Check capacity before Layer 2 inspects activations or loads densities."""
        return node_count <= GaussianLossAccumulator.MAX_SEGMENTS

    def accumulate_camera(
        self,
        losses_vec: torch.Tensor,
        weights: tuple[float, ...],
        *,
        flags: torch.Tensor,
        rgb_pred: torch.Tensor | None,
        rgb_gt: torch.Tensor | None,
        bg_pred: torch.Tensor | None,
        semantic_logits: torch.Tensor | None,
        semantic_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        """Adapt camera images and semantic rows, then write weighted camera slots."""
        if weights[0] != 0.0:
            if rgb_gt is None or rgb_pred is None:
                raise ValueError("Can't unpack empty optional")
            _validate_image(rgb_gt, "rgb_gt", 3, torch.float32)
            assert rgb_pred.dtype == torch.float32 and rgb_pred.numel() == rgb_gt.numel(), (
                "rgb_pred must be float32 with the same number of elements as rgb_gt"
            )
            rgb_pred = rgb_pred.reshape(-1, 3)
            rgb_gt = rgb_gt.reshape(-1, 3)
        if weights[1] != 0.0:
            if bg_pred is None:
                raise ValueError("Can't unpack empty optional")
            assert bg_pred.ndim == 1 and bg_pred.dtype == torch.float32, "bg_pred must be float32 [N_rays]"
            assert bg_pred.numel() == flags.numel(), "bg_pred and camera flags must have the same ray count"
        if weights[2] != 0.0:
            if semantic_targets is None or semantic_logits is None:
                raise ValueError("Can't unpack empty optional")
            semantic_targets = semantic_targets.flatten()
            semantic_logits = semantic_logits.float()
            flags = flags.to(torch.int32)
            assert semantic_logits.ndim == 2 and semantic_logits.shape[1] > 0, (
                "semantic_logits must have shape [N,C] with a non-empty class dimension"
            )
            assert semantic_targets.dtype in (torch.uint8, torch.int64), "semantic_targets must be uint8 or int64"
            assert semantic_logits.shape[0] == semantic_targets.numel() == flags.numel(), (
                "semantic logits, targets, and camera flags must have the same ray count"
            )
        _validate_image(flags, "rgb_flags", 1, torch.int32)
        return self.camera_loss_accumulator(
            flags=flags.reshape(-1),
            rgb_pred=rgb_pred,
            rgb_gt=rgb_gt,
            bg_pred=bg_pred,
            semantic_logits=semantic_logits,
            semantic_targets=semantic_targets,
            losses_vec=losses_vec,
            loss_slots=_CAMERA_SLOTS,
            loss_weights=weights,
        )

    def accumulate_ssim(
        self,
        losses_vec: torch.Tensor,
        weight: float,
        *,
        flags: torch.Tensor,
        rgb_pred: torch.Tensor,
        rgb_gt: torch.Tensor,
        channel: int,
        window_size: int,
        mask_mode: str,
        constant_mask_value: float,
    ) -> torch.Tensor:
        """Restore the NRE image layout used by the configured SSIM module."""
        _validate_image(flags, "rgb_flags", 1, torch.int32)
        _validate_image(rgb_gt, "rgb_gt", 3, torch.float32)
        assert rgb_pred.dtype == torch.float32 and rgb_pred.numel() == rgb_gt.numel(), (
            "rgb_pred must be float32 with the same number of elements as rgb_gt"
        )
        rgb_pred = rgb_pred.reshape_as(rgb_gt)
        if rgb_pred.shape[-1] != channel:
            raise ValueError(f"channel ({channel}) must match rgb_pred.shape[-1] ({rgb_pred.shape[-1]})")
        accumulator = self._get_ssim_loss_accumulator(window_size, mask_mode, constant_mask_value)
        return accumulator(
            flags=flags,
            pred=rgb_pred,
            target=rgb_gt,
            losses_vec=losses_vec,
            loss_slots=_SSIM_SLOTS,
            loss_weights=(weight,),
        )

    def accumulate_lidar(
        self,
        losses_vec: torch.Tensor,
        weights: tuple[float, ...],
        *,
        flags: torch.Tensor,
        distance_pred: torch.Tensor | None,
        distance_gt: torch.Tensor | None,
        intensity_pred: torch.Tensor | None,
        intensity_gt: torch.Tensor | None,
        raydrop_pred: torch.Tensor | None,
        raydrop_gt: torch.Tensor | None,
        bg_pred: torch.Tensor | None,
    ) -> torch.Tensor:
        """Adapt LiDAR image-shaped targets and predictions to flat ray domains."""
        _validate_image(flags, "lidar_flags", 1, torch.int32)
        if weights[0] != 0.0:
            if distance_pred is None or distance_gt is None:
                raise ValueError("Can't unpack empty optional")
            distance_pred, distance_gt = _lidar_ray_pair(distance_pred, distance_gt, "lidar")
        if weights[1] != 0.0:
            if intensity_pred is None or intensity_gt is None:
                raise ValueError("Can't unpack empty optional")
            intensity_pred, intensity_gt = _lidar_ray_pair(intensity_pred, intensity_gt, "intensity")
        if weights[2] != 0.0:
            if raydrop_pred is None or raydrop_gt is None:
                raise ValueError("Can't unpack empty optional")
            raydrop_pred, raydrop_gt = _lidar_ray_pair(raydrop_pred, raydrop_gt, "raydrop")
        if weights[3] != 0.0:
            if bg_pred is None:
                raise ValueError("Can't unpack empty optional")
            assert bg_pred.ndim == 1 and bg_pred.dtype == torch.float32, "bg_lidar_pred must be float32 [N_rays]"
            assert bg_pred.numel() == flags.numel(), "bg_lidar_pred and flags must have the same ray count"
        return self.lidar_loss_accumulator(
            flags=flags.reshape(-1),
            distance_pred=distance_pred,
            distance_gt=distance_gt,
            intensity_pred=intensity_pred,
            intensity_gt=intensity_gt,
            raydrop_pred=raydrop_pred,
            raydrop_gt=raydrop_gt,
            bg_pred=bg_pred,
            losses_vec=losses_vec,
            loss_slots=_LIDAR_SLOTS,
            loss_weights=weights,
        )

    def accumulate_gaussian(
        self,
        losses_vec: torch.Tensor,
        weights: tuple[float, ...],
        *,
        scales: torch.Tensor | None,
        densities: torch.Tensor | None,
        z_scales: torch.Tensor | None,
        visibility: torch.Tensor | None,
        deformation: torch.Tensor | None,
        deform_mask: torch.Tensor | None,
        scales_segments: tuple[torch.Tensor, ...],
        scale_segment_weights: tuple[float, ...],
        densities_segments: tuple[torch.Tensor, ...],
        density_segment_weights: tuple[float, ...],
        oob_positions_segments: tuple[torch.Tensor, ...],
        oob_cuboid_dims_tables: tuple[torch.Tensor, ...],
        oob_cuboid_ids_segments: tuple[torch.Tensor, ...],
        z_scale_threshold: float,
        densities_are_logits: bool,
    ) -> torch.Tensor:
        """Choose each member's bounded or flat input path and dispatch its weighted loss."""
        if scales_segments:
            if len(scales_segments) <= GaussianLossAccumulator.MAX_SEGMENTS:
                # The empty plain input supplies metadata; gradients use original segments.
                scales = scales_segments[0][:0].detach()
            else:
                # Preserve zero-weight non-finite masking before exp and its derivative.
                scales = torch.cat(
                    [
                        segment + math.log(weight)
                        if weight > 0.0
                        else segment.masked_fill(
                            torch.ones((), dtype=torch.bool, device=segment.device),
                            float("-inf"),
                        )
                        for segment, weight in zip(scales_segments, scale_segment_weights)
                    ],
                    dim=0,
                )
                scales_segments = ()
                scale_segment_weights = ()
        if densities_segments:
            if len(densities_segments) <= GaussianLossAccumulator.MAX_SEGMENTS:
                densities = densities_segments[0][:0].detach()
            else:
                densities = torch.cat(
                    [
                        segment * weight
                        if weight > 0.0
                        else segment.masked_fill(
                            torch.ones((), dtype=torch.bool, device=segment.device),
                            0.0,
                        )
                        for segment, weight in zip(densities_segments, density_segment_weights)
                    ],
                    dim=0,
                )
                densities_segments = ()
                density_segment_weights = ()
        positions = cuboid_dims = None
        if oob_positions_segments:
            if len(oob_positions_segments) <= GaussianLossAccumulator.MAX_SEGMENTS:
                positions = oob_positions_segments[0][:0].detach()
                cuboid_dims = oob_positions_segments[0][:0].detach()
            else:
                positions = torch.cat(oob_positions_segments, dim=0)
                cuboid_dims = torch.cat(
                    [table[ids.reshape(-1)] for table, ids in zip(oob_cuboid_dims_tables, oob_cuboid_ids_segments)],
                    dim=0,
                )
                oob_positions_segments = ()
                oob_cuboid_dims_tables = ()
                oob_cuboid_ids_segments = ()
        if deformation is not None:
            deformation = deformation.float()
            assert deformation.ndim == 2, "deformation must have shape [N,D]"
            if deform_mask is not None:
                deform_mask = deform_mask.float()
                deformation = torch.broadcast_tensors(deformation, deform_mask)[0]
        accumulator = self._get_gaussian_loss_accumulator(z_scale_threshold, densities_are_logits=densities_are_logits)
        return accumulator(
            scales=scales,
            densities=densities,
            z_scales=z_scales,
            positions=positions,
            cuboid_dims=cuboid_dims,
            visibility=visibility,
            deformation=deformation,
            deform_mask=deform_mask,
            scales_segments=scales_segments,
            scale_segment_weights=scale_segment_weights,
            densities_segments=densities_segments,
            density_segment_weights=density_segment_weights,
            oob_positions_segments=oob_positions_segments,
            oob_cuboid_dims_tables=oob_cuboid_dims_tables,
            oob_cuboid_ids_segments=oob_cuboid_ids_segments,
            losses_vec=losses_vec,
            loss_slots=_GAUSSIAN_SLOTS,
            loss_weights=weights,
        )

    def accumulate_bg_grid(
        self,
        losses_vec: torch.Tensor,
        weights: tuple[float, ...],
        *,
        bg_tex: torch.Tensor | None,
        grids_camera: torch.Tensor | None,
        grids_frame: torch.Tensor | None,
    ) -> torch.Tensor:
        """Adapt NRE grids/textures and preserve the combined grid drift normalization."""
        if grids_camera is not None:
            assert (
                grids_camera.ndim == 5
                and grids_camera.shape[1] == GRID_NUM_CHANNELS
                and grids_camera.dtype == torch.float32
            ), f"BilateralGridPerCamera must be float32 [B,{GRID_NUM_CHANNELS},D,H,W]"
            batch, channels, depth_camera, height, width = grids_camera.shape
            grids_camera = grids_camera.reshape(batch * channels, depth_camera, height, width)
        if grids_frame is not None:
            assert (
                grids_frame.ndim == 5
                and grids_frame.shape[1] == GRID_NUM_CHANNELS
                and grids_frame.dtype == torch.float32
            ), f"BilateralGridPerFrame must be float32 [B,{GRID_NUM_CHANNELS},D,H,W]"
            batch, channels, depth_frame, height, width = grids_frame.shape
            grids_frame = grids_frame.reshape(batch * channels, depth_frame, height, width)
        depth = 1
        if bg_tex is not None:
            assert bg_tex.ndim in (4, 5) and bg_tex.dtype == torch.float32, "bg_tex must be a float32 4D or 5D Tensor"
            if bg_tex.ndim == 5:
                batch, depth, height, width, channels = bg_tex.shape
                assert depth == 6, "SkyEnvMapBackground cubemap texture must have depth 6"
                bg_tex = bg_tex.reshape(batch * depth, height, width, channels)
        drift_factor = -1.0
        if weights[1] != 0.0:
            cells = (grids_camera.numel() // GRID_NUM_CHANNELS if grids_camera is not None else 0) + (
                grids_frame.numel() // GRID_NUM_CHANNELS if grids_frame is not None else 0
            )
            drift_factor = 1.0 / cells if cells else 0.0
        return self.bg_grid_loss_accumulator(
            bg_tex=bg_tex,
            grids_camera=grids_camera,
            grids_frame=grids_frame,
            bg_tex_depth=depth,
            losses_vec=losses_vec,
            loss_slots=_BG_GRID_SLOTS,
            loss_weights=weights,
            grid_drift_camera_factor=drift_factor,
            grid_drift_frame_factor=drift_factor,
        )

    def accumulate_bg_track_node(
        self,
        losses_vec: torch.Tensor,
        weights: tuple[float, ...],
        *,
        positions: torch.Tensor | None,
        density_logits: torch.Tensor,
        camera_timestamps_startend_us: torch.Tensor | None,
        tracks_packinfo: torch.Tensor | None,
        tracks_poses: torch.Tensor | None,
        tracks_timestamps_us: torch.Tensor | None,
        cuboids_dims: torch.Tensor | None,
        semantic_logits: torch.Tensor | None,
        n_semantic_points: int | None,
        background_semantic_segments: tuple[tuple[int, tuple[int, ...], torch.Tensor], ...],
        background_allowed_class_ids: tuple[int, ...],
        node_primary_predicate: tuple[tuple[int, ...], bool] | None,
        density_logits_min: float,
        other_density_logits: torch.Tensor | None,
        other_semantic_logits: tuple[torch.Tensor, ...],
        other_segments: tuple[tuple[int, tuple[int, ...], bool], ...],
    ) -> torch.Tensor:
        """Build gsplat's semantic mode from selected model-domain metadata and dispatch."""
        if weights[0] != 0.0 and n_semantic_points is not None:
            semantics: JointSemantics | PackedSemantics = PackedSemantics(
                n_semantic_points=n_semantic_points,
                background_semantic_logits=tuple(
                    semantic.detach().float() for _, _, semantic in background_semantic_segments
                ),
                background_segments=tuple((end, class_ids) for end, class_ids, _ in background_semantic_segments),
            )
        else:
            if weights[1] != 0.0 and n_semantic_points is None:
                if semantic_logits is None:
                    raise ValueError("Can't unpack empty optional")
                semantic_logits = semantic_logits.detach().float()
            else:
                semantic_logits = None
            semantics = JointSemantics(
                semantic_logits=semantic_logits,
                background_allowed_class_ids=background_allowed_class_ids if weights[0] != 0.0 else (),
                node_primary_predicate=node_primary_predicate if weights[1] != 0.0 else None,
            )
        accumulator = self._get_bg_track_node_semantic_loss_accumulator(density_logits_min)
        return accumulator(
            positions=positions,
            density_logits=density_logits.flatten(),
            camera_timestamps_startend_us=camera_timestamps_startend_us,
            tracks_packinfo=tracks_packinfo,
            tracks_poses=tracks_poses,
            tracks_timestamps_us=tracks_timestamps_us,
            cuboids_dims=cuboids_dims,
            semantics=semantics,
            other_density_logits=other_density_logits,
            other_semantic_logits=tuple(tensor.detach().float() for tensor in other_semantic_logits),
            other_segments=other_segments,
            losses_vec=losses_vec,
            loss_slots=_BG_TRACK_NODE_SLOTS,
            loss_weights=weights,
        )

    def accumulate_ground(
        self,
        losses_vec: torch.Tensor,
        weight: float,
        *,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        camera_poses: torch.Tensor,
        random_values: torch.Tensor,
        min_bias: float,
        range_bias: float,
        grid_len: float,
        rotation_lambda: float,
    ) -> torch.Tensor:
        """Select the first pose from camera metadata validated before Ground sampling."""
        accumulator = self._get_ground_gaussians_loss_accumulator(min_bias, range_bias, grid_len, rotation_lambda)
        return accumulator(
            positions=positions,
            rotations=rotations,
            cam_tquat=camera_poses[0, 0].detach(),
            random_values=random_values,
            losses_vec=losses_vec,
            loss_slots=_GROUND_SLOTS,
            loss_weights=(weight,),
        )
