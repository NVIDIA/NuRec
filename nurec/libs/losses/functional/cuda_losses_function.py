# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused-loss dispatch onto gsplat.

Shapes this repository's tensors and factors for gsplat's fused loss modules and
calls them. The maths, the kernels and the gradients all live in gsplat; nothing
here is a `torch.autograd.Function` and no loss kernel is implemented in this
repository.
"""

import functools

import torch

from gsplat import (
    fused_bg_grid_losses,
    fused_bg_track_node_semantic_losses,
    fused_camera_losses,
    fused_gaussian_losses,
    fused_lidar_losses,
)
from gsplat.losses import BG_GRID_NUM_COLS, BG_GRID_NUM_ROWS


GRID_NUM_CHANNELS = BG_GRID_NUM_ROWS * BG_GRID_NUM_COLS


# CudaLossesFunction output layout. The forward return is the
# N_ELEMENTWISE_LOSS_OUTPUTS per-element loss maps (ordered like
# ModuleLosses.cuda_loss_names, i.e. ModuleLosses.available excluding
# module-side losses such as SSIM), followed by one raw scalar per grouped loss
# and then one weighted scalar per grouped loss (same order).
# SCALAR_LOSS_OUTPUT_INDEX maps each grouped scalar loss to its
# (raw, weighted) output positions; keep _GROUPED_SCALAR_LOSSES in sync with
# the return statement in CudaLossesFunction.forward.
N_ELEMENTWISE_LOSS_OUTPUTS = 14
_GROUPED_SCALAR_LOSSES = ("bg_track", "node_semantic", "semantic", "deform")
SCALAR_LOSS_OUTPUT_INDEX: dict[str, tuple[int, int]] = {
    name: (
        N_ELEMENTWISE_LOSS_OUTPUTS + i,
        N_ELEMENTWISE_LOSS_OUTPUTS + len(_GROUPED_SCALAR_LOSSES) + i,
    )
    for i, name in enumerate(_GROUPED_SCALAR_LOSSES)
}

# 0-indexed positions of deform_mask / deform_mask_source in
# CudaLossesFunction.forward's parameter list (after ctx). backward reads
# ctx.needs_input_grad at these offsets; appending arguments is safe, but
# inserting one before deform_mask must update these in lock-step.
_DEFORM_MASK_ARG_IDX = 55
_DEFORM_MASK_SOURCE_ARG_IDX = 56


def _gaussian_dispatch(z_scale_threshold: float):
    """The gaussian dispatch with its call-time configuration bound."""
    return functools.partial(
        fused_gaussian_losses,
        z_scale_threshold=z_scale_threshold,
        preactivation=True,  # scales and z-scales arrive in log space
        # The scale and out-of-bound members are published per gaussian here,
        # not per component.
        component_reduction="sum",
    )


def _bg_track_dispatch(density_logits_min: float):
    """The background-in-track dispatch with its call-time configuration bound."""
    return functools.partial(
        fused_bg_track_node_semantic_losses,
        density_logits_min=density_logits_min,
    )


class CudaLossesFunction:
    """
    Stateless dispatcher for the fused losses.

    Routes each fused loss group to its matching gsplat module. A plain callable,
    not a `torch.autograd.Function`: gsplat's dispatches build their own graph and
    carry their own backward.
    """

    @staticmethod
    def forward(
        rgb_flags: torch.Tensor,
        rgb_pred: torch.Tensor,
        rgb_gt: torch.Tensor,
        rgb_factor: float,
        lidar_flags: torch.Tensor,
        lidar_pred: torch.Tensor,
        lidar_gt: torch.Tensor,
        lidar_factor: float,
        intensity_pred: torch.Tensor,
        intensity_gt: torch.Tensor,
        intensity_factor: float,
        raydrop_pred: torch.Tensor,
        raydrop_gt: torch.Tensor,
        raydrop_factor: float,
        bg_pred: torch.Tensor,
        bg_factor: float,
        bg_lidar_pred: torch.Tensor,
        bg_lidar_factor: float,
        grids_per_camera: torch.Tensor,
        grids_per_frame: torch.Tensor,
        grid_drift_per_camera_factor: float,
        grid_drift_per_frame_factor: float,
        grid_camera_spatial_tv_factor: float,
        grid_frame_spatial_tv_factor: float,
        gaussian_scales: torch.Tensor,
        scale_factor: float,
        bg_tex: torch.Tensor,
        bg_tex_factor: float,
        gaussian_densities: torch.Tensor,
        density_factor: float,
        gaussian_visibility: torch.Tensor,
        out_of_bound_positions: torch.Tensor,
        out_of_bound_cuboid_dims: torch.Tensor,
        out_of_bound_factor: float,
        gaussian_z_scales: torch.Tensor,
        z_scale_threshold: float,
        z_scale_factor: float,
        bg_track_positions: torch.Tensor,
        bg_track_density_logits: torch.Tensor,
        bg_track_semantic_logits: torch.Tensor,
        bg_track_n_semantic_points: int | None,
        bg_track_semantic_segments: tuple[tuple[int, tuple[int, ...], torch.Tensor], ...],
        bg_track_camera_timestamps_startend_us: torch.Tensor,
        bg_track_tracks_packinfo: torch.Tensor,
        bg_track_tracks_poses: torch.Tensor,
        bg_track_tracks_timestamps_us: torch.Tensor,
        bg_track_cuboids_dims: torch.Tensor,
        bg_track_density_logits_min: float,
        bg_track_lambda: float,
        other_node_density_logits: torch.Tensor,
        joint_metadata: tuple[
            tuple[int, ...],
            tuple[tuple[int, ...], bool] | None,
            tuple[tuple[int, tuple[int, ...], bool], ...],
            tuple[torch.Tensor, ...],
            float,
        ]
        | None,
        semantic_logits: torch.Tensor,
        semantic_targets: torch.Tensor,
        semantic_lambda: float,
        deformation: torch.Tensor,
        deform_mask: torch.Tensor | None,
        deform_mask_source: torch.Tensor | None,
        deform_lambda: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run the five fused dispatch groups and assemble the tuple this repository publishes.

        Each group is computed by the matching gsplat submodule, which allocates its own
        outputs and carries its own backward; what is left here is shape reconciliation and
        the dummy sizing a disabled member still needs.
        """
        semantic_enabled = semantic_lambda >= 0
        deform_enabled = deform_lambda >= 0
        has_deform_mask = deform_mask is not None
        # deform_mask must be an autograd broadcast view of deform_mask_source
        # (torch.broadcast_tensors in ModuleLosses): backward splits the mask
        # gradient into a numerator term (returned for deform_mask; autograd's
        # broadcast-backward reduces it into the source) and a denominator term
        # (returned for deform_mask_source directly) — only their sum through
        # the autograd graph is the correct total. A materialized or detached
        # mask silently receives wrong gradients.
        assert has_deform_mask == (deform_mask_source is not None), (
            "deform_mask and deform_mask_source must either both be present or both be None"
        )
        assert deform_enabled or not has_deform_mask, "disabled deform smoothness cannot carry a mask"

        # Assert input shapes and types
        assert rgb_flags.ndim == 4 and rgb_flags.shape[3] == 1 and rgb_flags.dtype == torch.int32, (
            f"rgb_flags must be int32 Tensor of shape [B,H,W,1], got shape={rgb_flags.shape}, dtype={rgb_flags.dtype}"
        )
        assert rgb_pred.ndim == 4 and rgb_pred.shape[3] == 3 and rgb_pred.dtype == torch.float32, (
            f"rgb_pred must be float32 Tensor of shape [B,H,W,3], got shape={rgb_pred.shape}, dtype={rgb_pred.dtype}"
        )
        assert rgb_gt.ndim == 4 and rgb_gt.shape[3] == 3 and rgb_gt.dtype == torch.float32, (
            f"rgb_gt must be float32 Tensor of shape [B,H,W,3], got shape={rgb_gt.shape}, dtype={rgb_gt.dtype}"
        )
        assert lidar_flags.ndim == 4 and lidar_flags.shape[3] == 1 and lidar_flags.dtype == torch.int32, (
            f"lidar_flags must be int32 Tensor of shape [B,H,W,1], got shape={lidar_flags.shape}, dtype={lidar_flags.dtype}"
        )
        assert lidar_pred.ndim == 4 and lidar_pred.shape[3] == 1 and lidar_pred.dtype == torch.float32, (
            f"lidar_pred must be float32 Tensor of shape [B,H,W,1], got shape={lidar_pred.shape}, dtype={lidar_pred.dtype}"
        )
        assert lidar_gt.ndim == 4 and lidar_gt.shape[3] == 1 and lidar_gt.dtype == torch.float32, (
            f"lidar_gt must be float32 Tensor of shape [B,H,W,1], got shape={lidar_gt.shape}, dtype={lidar_gt.dtype}"
        )
        assert intensity_pred.ndim == 4 and intensity_pred.shape[3] == 1 and intensity_pred.dtype == torch.float32, (
            f"intensity_pred must be float32 Tensor of shape [B,H,W,1], got shape={intensity_pred.shape}, dtype={intensity_pred.dtype}"
        )
        assert intensity_gt.ndim == 4 and intensity_gt.shape[3] == 1 and intensity_gt.dtype == torch.float32, (
            f"intensity_gt must be float32 Tensor of shape [B,H,W,1], got shape={intensity_gt.shape}, dtype={intensity_gt.dtype}"
        )
        assert raydrop_pred.ndim == 4 and raydrop_pred.shape[3] == 1 and raydrop_pred.dtype == torch.float32, (
            f"raydrop_pred must be float32 Tensor of shape [B,H,W,1], got shape={raydrop_pred.shape}, dtype={raydrop_pred.dtype}"
        )
        assert raydrop_gt.ndim == 4 and raydrop_gt.shape[3] == 1 and raydrop_gt.dtype == torch.float32, (
            f"raydrop_gt must be float32 Tensor of shape [B,H,W,1], got shape={raydrop_gt.shape}, dtype={raydrop_gt.dtype}"
        )
        assert bg_pred.ndim == 1 and bg_pred.dtype == torch.float32, (
            f"bg_pred must be float32 Tensor of shape [N_rays], got shape={bg_pred.shape}, dtype={bg_pred.dtype}"
        )
        assert bg_lidar_pred.ndim == 1 and bg_lidar_pred.dtype == torch.float32, (
            f"bg_lidar_pred must be float32 Tensor of shape [N_rays_lidar], got shape={bg_lidar_pred.shape}, dtype={bg_lidar_pred.dtype}"
        )
        assert (
            grids_per_camera.ndim == 4
            and grids_per_camera.dtype == torch.float32
            and grids_per_camera.shape[0] % GRID_NUM_CHANNELS == 0
        ), (
            f"grids_per_camera must be float32 Tensor [B*C,D,H,W] with C={GRID_NUM_CHANNELS}, got shape={grids_per_camera.shape}, dtype={grids_per_camera.dtype}"
        )
        assert (
            grids_per_frame.ndim == 4
            and grids_per_frame.dtype == torch.float32
            and grids_per_frame.shape[0] % GRID_NUM_CHANNELS == 0
        ), (
            f"grids_per_frame must be float32 Tensor [B*C,D,H,W] with C={GRID_NUM_CHANNELS}, got shape={grids_per_frame.shape}, dtype={grids_per_frame.dtype}"
        )
        assert gaussian_scales.ndim == 2 and gaussian_scales.shape[1] == 3 and gaussian_scales.dtype == torch.float32, (
            f"gaussian_scales must be float32 Tensor of shape [N, 3], got shape={gaussian_scales.shape}, dtype={gaussian_scales.dtype}"
        )
        assert (
            out_of_bound_positions.ndim == 2
            and out_of_bound_positions.shape[1] == 3
            and out_of_bound_positions.dtype == torch.float32
        ), (
            f"out_of_bound_positions must be float32 Tensor of shape [N,3], got shape={out_of_bound_positions.shape}, dtype={out_of_bound_positions.dtype}"
        )
        assert (
            out_of_bound_cuboid_dims.shape == out_of_bound_positions.shape
            and out_of_bound_cuboid_dims.dtype == torch.float32
        ), (
            f"out_of_bound_cuboid_dims must be float32 Tensor of shape [N,3], got shape={out_of_bound_cuboid_dims.shape}, dtype={out_of_bound_cuboid_dims.dtype}"
        )
        assert (bg_tex.ndim == 4 or bg_tex.ndim == 5) and bg_tex.dtype == torch.float32, (
            f"bg_tex must be float32 4D or 5D Tensor, got shape={bg_tex.shape}, dtype={bg_tex.dtype}"
        )
        assert gaussian_densities.ndim == 1 and gaussian_densities.dtype == torch.float32, (
            f"gaussian_densities must be float32 Tensor of shape [N], got shape={gaussian_densities.shape}, dtype={gaussian_densities.dtype}"
        )
        assert gaussian_visibility.ndim == 1 and gaussian_visibility.dtype == torch.float32, (
            f"gaussian_visibility must be float32 Tensor of shape [N], got shape={gaussian_visibility.shape}, dtype={gaussian_visibility.dtype}"
        )
        assert (
            gaussian_z_scales.ndim == 2 and gaussian_z_scales.shape[1] == 3 and gaussian_z_scales.dtype == torch.float32
        ), (
            f"gaussian_z_scales must be float32 Tensor of shape [N, 3], got shape={gaussian_z_scales.shape}, dtype={gaussian_z_scales.dtype}"
        )
        assert (
            bg_track_positions.ndim == 2
            and bg_track_positions.shape[1] == 3
            and bg_track_positions.dtype == torch.float32
        ), (
            "bg_track_positions must be float32 Tensor of shape [N,3], "
            f"got shape={bg_track_positions.shape}, dtype={bg_track_positions.dtype}"
        )
        assert bg_track_density_logits.numel() == bg_track_positions.shape[0] and (
            bg_track_density_logits.dtype == torch.float32
        ), (
            "bg_track_density_logits must be float32 with N elements matching bg_track_positions, "
            f"got shape={bg_track_density_logits.shape}, dtype={bg_track_density_logits.dtype}"
        )
        assert bg_track_lambda == -1.0 or bg_track_lambda >= 0.0, (
            "bg_track_lambda must be -1 when disabled or non-negative when enabled"
        )
        assert semantic_logits.ndim == 2 and semantic_logits.dtype == torch.float32, (
            "semantic_logits must be a float32 Tensor of shape [N,C], "
            f"got shape={semantic_logits.shape}, dtype={semantic_logits.dtype}"
        )
        assert semantic_logits.shape[1] > 0, "semantic_logits must have a non-empty class dimension"
        assert semantic_targets.ndim == 1 and semantic_targets.dtype in (torch.uint8, torch.int64), (
            "semantic_targets must be a uint8 or int64 Tensor of shape [N], "
            f"got shape={semantic_targets.shape}, dtype={semantic_targets.dtype}"
        )
        assert deformation.ndim == 2 and deformation.dtype == torch.float32, (
            f"deformation must be [N, D] float32, got ndim={deformation.ndim}, {deformation.dtype}"
        )
        if has_deform_mask:
            assert deform_mask is not None and deform_mask_source is not None
            assert deform_mask.shape == deformation.shape, (
                "deform_mask must be broadcast to the deformation shape, "
                f"got deformation={deformation.shape}, mask={deform_mask.shape}"
            )
            assert deform_mask.dtype == torch.float32, f"deform_mask must be float32, got {deform_mask.dtype}"
            assert deform_mask_source.dtype == torch.float32, (
                f"deform_mask_source must be float32, got {deform_mask_source.dtype}"
            )
        if semantic_enabled:
            assert semantic_logits.shape[0] == semantic_targets.numel() == rgb_flags.numel(), (
                "semantic logits, targets, and camera flags must have the same ray count"
            )

        # Get dimensions for each input tensor
        B_rgb, H_rgb, W_rgb, _ = rgb_flags.shape
        B_lidar, H_lidar, W_lidar, _ = lidar_flags.shape
        N_rays_rgb = bg_pred.shape[0]
        N_rays_lidar = bg_lidar_pred.shape[0]
        BC_gc, D_gc, H_gc, W_gc = grids_per_camera.shape
        B_gc = BC_gc // GRID_NUM_CHANNELS
        BC_gf, D_gf, H_gf, W_gf = grids_per_frame.shape
        B_gf = BC_gf // GRID_NUM_CHANNELS
        numel_grids_per_camera = B_gc * D_gc * H_gc * W_gc
        numel_grids_per_frame = B_gf * D_gf * H_gf * W_gf
        N_scales = gaussian_scales.shape[0]
        # Normalize 2D env maps to [B, D, H, W, C] so the CUDA texture path has one shape convention.
        if bg_tex.ndim == 4:
            bg_tex = bg_tex.unsqueeze(1)
        else:
            assert bg_tex.shape[1] == 6, "SkyEnvMapBackground texture (B,D,H,W,C) must have D==6 for cubemap"
        B_tex, D_tex, H_tex, W_tex, C_tex = bg_tex.shape
        numel_bg_tex = B_tex * D_tex * H_tex * W_tex * C_tex
        N_densities = gaussian_densities.shape[0]
        num_out_of_bound_gaussians = out_of_bound_positions.shape[0]
        N_z_scales = gaussian_z_scales.shape[0]

        # Extra assert input shape dependency across tensor inputs
        if bg_factor >= 0:
            N_rays_rgb_flags = B_rgb * H_rgb * W_rgb
            assert N_rays_rgb == N_rays_rgb_flags, (
                f"bg_pred shape must be [{N_rays_rgb_flags}], got shape={bg_pred.shape}"
            )
        if bg_lidar_factor >= 0:
            N_rays_lidar_flags = B_lidar * H_lidar * W_lidar
            assert N_rays_lidar == N_rays_lidar_flags, (
                f"bg_lidar_pred shape must be [{N_rays_lidar_flags}], got shape={bg_lidar_pred.shape}"
            )

        # Ensure inputs are contiguous: CUDA kernels require contiguous tensors for correct memory access
        rgb_flags = rgb_flags.contiguous()
        lidar_flags = lidar_flags.contiguous()
        rgb_pred = rgb_pred.contiguous()
        lidar_pred = lidar_pred.contiguous()
        intensity_pred = intensity_pred.contiguous()
        intensity_gt = intensity_gt.contiguous()
        raydrop_pred = raydrop_pred.contiguous()
        raydrop_gt = raydrop_gt.contiguous()
        bg_pred = bg_pred.contiguous()
        bg_lidar_pred = bg_lidar_pred.contiguous()
        rgb_gt = rgb_gt.contiguous()
        lidar_gt = lidar_gt.contiguous()
        grids_per_camera = grids_per_camera.contiguous()
        grids_per_frame = grids_per_frame.contiguous()
        bg_tex = bg_tex.contiguous()
        gaussian_scales = gaussian_scales.contiguous()
        gaussian_densities = gaussian_densities.contiguous()
        gaussian_visibility = gaussian_visibility.contiguous()
        gaussian_z_scales = gaussian_z_scales.contiguous()
        bg_track_density_logits = bg_track_density_logits.flatten().contiguous()
        other_node_density_logits = other_node_density_logits.flatten().contiguous()
        if semantic_logits.stride(1) != 1 or semantic_logits.stride(0) < semantic_logits.shape[1]:
            semantic_logits = semantic_logits.contiguous()
        semantic_targets = semantic_targets.contiguous()
        deformation = deformation.contiguous()
        if has_deform_mask:
            assert deform_mask is not None and deform_mask_source is not None
            deform_mask = deform_mask.contiguous()
            deform_mask_source = deform_mask_source.contiguous()

        out_of_bound_positions = out_of_bound_positions.contiguous()
        out_of_bound_cuboid_dims = out_of_bound_cuboid_dims.contiguous()

        # Allocate output tensors (pre-allocated, passed to CUDA kernels)
        rgb_loss = (
            torch.empty(
                (B_rgb, H_rgb, W_rgb),
                dtype=torch.float32,
                device=rgb_pred.device,
                memory_format=torch.contiguous_format,
            )
            if rgb_factor >= 0
            else rgb_pred.new_empty(0)
        )
        lidar_loss = torch.empty(
            (B_lidar, H_lidar, W_lidar),
            dtype=torch.float32,
            device=lidar_pred.device,
            memory_format=torch.contiguous_format,
        )
        intensity_loss = torch.empty(
            (B_lidar, H_lidar, W_lidar),
            dtype=torch.float32,
            device=intensity_pred.device,
            memory_format=torch.contiguous_format,
        )
        raydrop_loss = torch.empty(
            (B_lidar, H_lidar, W_lidar),
            dtype=torch.float32,
            device=raydrop_pred.device,
            memory_format=torch.contiguous_format,
        )
        bg_loss = torch.empty(
            N_rays_rgb,
            dtype=torch.float32,
            device=bg_pred.device,
            memory_format=torch.contiguous_format,
        )
        bg_lidar_loss = torch.empty(
            N_rays_lidar,
            dtype=torch.float32,
            device=bg_lidar_pred.device,
            memory_format=torch.contiguous_format,
        )
        grids_drift_loss = torch.empty(
            (numel_grids_per_camera + numel_grids_per_frame),
            dtype=torch.float32,
            device=grids_per_camera.device,
            memory_format=torch.contiguous_format,
        )
        grid_camera_spatial_tv_loss = torch.empty(
            numel_grids_per_camera,
            dtype=torch.float32,
            device=grids_per_camera.device,
            memory_format=torch.contiguous_format,
        )
        grid_frame_spatial_tv_loss = torch.empty(
            numel_grids_per_frame,
            dtype=torch.float32,
            device=grids_per_frame.device,
            memory_format=torch.contiguous_format,
        )
        scale_loss = torch.empty(
            N_scales,
            dtype=torch.float32,
            device=gaussian_scales.device,
        )
        bg_tex_loss = torch.empty(
            numel_bg_tex,
            dtype=torch.float32,
            device=bg_tex.device,
            memory_format=torch.contiguous_format,
        )
        density_loss = torch.empty(
            N_densities,
            dtype=torch.float32,
            device=gaussian_densities.device,
            memory_format=torch.contiguous_format,
        )
        out_of_bound_loss = torch.empty(
            num_out_of_bound_gaussians,
            dtype=torch.float32,
            device=out_of_bound_positions.device,
            memory_format=torch.contiguous_format,
        )
        z_scale_loss = torch.empty(
            N_z_scales,
            dtype=torch.float32,
            device=gaussian_z_scales.device,
            memory_format=torch.contiguous_format,
        )
        if semantic_enabled:
            semantic_loss = torch.empty((), dtype=torch.float32, device=semantic_logits.device)
            semantic_weighted_loss = torch.empty((), dtype=torch.float32, device=semantic_logits.device)
        else:
            semantic_loss = semantic_logits[:0, 0]
            semantic_weighted_loss = semantic_loss

        if deform_enabled:
            deform_loss = torch.zeros((), dtype=torch.float32, device=deformation.device)
            deform_weighted_loss = torch.empty((), dtype=torch.float32, device=deformation.device)
        else:
            deform_loss = deformation[:0, 0]
            deform_weighted_loss = deform_loss

        # ========================================================================
        # Launch 5 CUDA kernel dispatches
        # ========================================================================

        # DISPATCH 1: Camera rendering losses (RGB + background + semantic CE)
        if rgb_factor >= 0 or bg_factor >= 0 or semantic_enabled:
            flat_flags = rgb_flags.reshape(-1)
            rgb_active = rgb_factor >= 0
            bg_active = bg_factor >= 0
            # A disabled member is omitted rather than substituted, and the
            # semantic row mask is derived from these same flags upstream.
            camera_out = fused_camera_losses(
                flat_flags,
                rgb_pred.reshape(-1, 3) if rgb_active else None,
                rgb_gt.reshape(-1, 3) if rgb_active else None,
                bg_pred if bg_active else None,
                rgb_factor,
                bg_factor,
                semantic_logits.contiguous() if semantic_enabled else None,
                semantic_targets if semantic_enabled else None,
            )
            if rgb_active:
                rgb_loss = camera_out[0].view(B_rgb, H_rgb, W_rgb)
            if bg_active:
                bg_loss = camera_out[1]
            if semantic_enabled:
                # gsplat returns the unweighted scalar only.
                semantic_loss = camera_out[2]
                semantic_weighted_loss = semantic_loss * semantic_lambda

        # DISPATCH 2: LiDAR rendering losses
        if lidar_factor >= 0 or bg_lidar_factor >= 0 or intensity_factor >= 0 or raydrop_factor >= 0:
            # gsplat interleaves pred/gt per member and orders both the
            # factors and the outputs distance, intensity, raydrop, background;
            # this repository groups the tensors and orders those two lists
            # distance, background, intensity, raydrop. Every argument is a
            # same-shaped float tensor, so a mismatch here computes silently.
            lidar_flat_flags = lidar_flags.reshape(-1)

            def lidar_ray(tensor: torch.Tensor, member_enabled: bool) -> torch.Tensor | None:
                # A disabled member is omitted; its output comes back empty.
                return tensor.reshape(-1) if member_enabled else None

            distance_on = lidar_factor >= 0
            intensity_on = intensity_factor >= 0
            raydrop_on = raydrop_factor >= 0
            bg_on = bg_lidar_factor >= 0
            distance_out, intensity_out, raydrop_out, bg_lidar_out = fused_lidar_losses(
                lidar_flat_flags,
                lidar_ray(lidar_pred, distance_on),
                lidar_ray(lidar_gt, distance_on),
                lidar_ray(intensity_pred, intensity_on),
                lidar_ray(intensity_gt, intensity_on),
                lidar_ray(raydrop_pred, raydrop_on),
                lidar_ray(raydrop_gt, raydrop_on),
                lidar_ray(bg_lidar_pred, bg_on),
                lidar_factor,
                intensity_factor,
                raydrop_factor,
                bg_lidar_factor,
            )
            if distance_on:
                lidar_loss = distance_out.view(B_lidar, H_lidar, W_lidar)
            if intensity_on:
                intensity_loss = intensity_out.view(B_lidar, H_lidar, W_lidar)
            if raydrop_on:
                raydrop_loss = raydrop_out.view(B_lidar, H_lidar, W_lidar)
            if bg_on:
                bg_lidar_loss = bg_lidar_out

        # DISPATCH 3: Gaussian/model regularization losses
        if (
            scale_factor >= 0
            or density_factor >= 0
            or z_scale_factor >= 0
            or out_of_bound_factor >= 0
            or deform_enabled
        ):
            # Row counts are read off the tensors upstream. The z member takes
            # the full [N, 3] triple and reads its own column, the scale and
            # out-of-bound members come back already reduced onto the [N] this
            # repository publishes, and a disabled member is omitted rather
            # than carrying a substituted threshold. The deform mask is passed
            # unexpanded: upstream walks the broadcast with strides rather than
            # taking a materialized view alongside its source.
            # The threshold is constructor config, validated when the module
            # is built and so before it can know the member will be omitted.
            # A disabled member has no meaningful threshold, and this side
            # spells that -1; upstream requires non-negative, so map it to 0.
            z_scale_threshold_arg = z_scale_threshold if z_scale_factor >= 0 else 0.0
            gaussian_out = _gaussian_dispatch(z_scale_threshold_arg)(
                gaussian_scales if scale_factor >= 0 else None,
                gaussian_densities if density_factor >= 0 else None,
                gaussian_z_scales if z_scale_factor >= 0 else None,
                out_of_bound_positions if out_of_bound_factor >= 0 else None,
                out_of_bound_cuboid_dims if out_of_bound_factor >= 0 else None,
                # Visibility spans the scale and density members; with neither
                # present there is nothing for it to span.
                gaussian_visibility if (scale_factor >= 0 or density_factor >= 0) else None,
                deformation if deform_enabled else None,
                deform_mask_source if deform_enabled and has_deform_mask else None,
                scale_factor,
                density_factor,
                z_scale_factor,
                out_of_bound_factor,
            )
            scale_loss_out, density_loss_out, z_scale_loss_out, oob_loss_out = gaussian_out[:4]
            if scale_factor >= 0:
                scale_loss = scale_loss_out
            if density_factor >= 0:
                density_loss = density_loss_out
            if z_scale_factor >= 0:
                z_scale_loss = z_scale_loss_out
            if out_of_bound_factor >= 0:
                out_of_bound_loss = oob_loss_out
            if deform_enabled:
                # The local kernel emitted the mean and lambda * mean together;
                # upstream returns only the mean, so the weighting lands here.
                deform_loss = gaussian_out[4]
                deform_weighted_loss = deform_loss * deform_lambda

        # DISPATCH 4: Background texture & grid regularization losses
        if (
            bg_tex_factor >= 0
            or grid_drift_per_camera_factor >= 0
            or grid_drift_per_frame_factor >= 0
            or grid_camera_spatial_tv_factor >= 0
            or grid_frame_spatial_tv_factor >= 0
        ):
            # Factors and outputs are already in gsplat's order here. What
            # differs is the texture layout: it is [B, D, H, W, C] on this side
            # and [B * D, H, W, C] with the depth passed alongside upstream.
            # The grids are already [B * channels, D, H, W], which matches.
            #
            (
                bg_tex_loss,
                grids_drift_loss,
                grid_camera_spatial_tv_loss,
                grid_frame_spatial_tv_loss,
            ) = fused_bg_grid_losses(
                bg_tex.reshape(B_tex * D_tex, H_tex, W_tex, C_tex),
                grids_per_camera,
                grids_per_frame,
                bg_tex_factor,
                grid_drift_per_camera_factor,
                grid_drift_per_frame_factor,
                grid_camera_spatial_tv_factor,
                grid_frame_spatial_tv_factor,
                D_tex,
            )

        # DISPATCH 5: Background-in-track and node-semantic BCE. Either member
        # may run alone; when both run, they share classification and reduction work.
        background_track_enabled = bg_track_lambda >= 0
        joint_background_allowed_class_ids: tuple[int, ...] = ()
        joint_node_primary_predicate: tuple[tuple[int, ...], bool] | None = None
        joint_node_other_segments: tuple[tuple[int, tuple[int, ...], bool], ...] = ()
        joint_node_other_semantic_logits: tuple[torch.Tensor, ...] = ()
        joint_node_lambda = -1.0
        dispatch_n_semantic_points = bg_track_n_semantic_points
        dispatch_background_semantic_logits: tuple[torch.Tensor, ...] = ()
        dispatch_semantic_segments: tuple[tuple[int, tuple[int, ...]], ...] = ()
        if joint_metadata is not None:
            (
                joint_background_allowed_class_ids,
                joint_node_primary_predicate,
                joint_node_other_segments,
                joint_node_other_semantic_logits,
                joint_node_lambda,
            ) = joint_metadata
        node_semantic_enabled = joint_node_lambda >= 0

        if background_track_enabled and dispatch_n_semantic_points is not None:
            dispatch_background_semantic_logits = tuple(
                semantic.detach().float().contiguous() for _, _, semantic in bg_track_semantic_segments
            )
            dispatch_semantic_segments = tuple(
                (segment_end, class_ids) for segment_end, class_ids, _ in bg_track_semantic_segments
            )

        if node_semantic_enabled or background_track_enabled:
            bg_track_positions = bg_track_positions.contiguous()
            bg_track_semantic_logits = (
                bg_track_semantic_logits.detach().float().contiguous()
                if node_semantic_enabled
                else bg_track_semantic_logits.float().contiguous()
            )
            if node_semantic_enabled:
                joint_node_other_semantic_logits = tuple(
                    tensor.detach().float().contiguous() for tensor in joint_node_other_semantic_logits
                )
            bg_track_camera_timestamps_startend_us = bg_track_camera_timestamps_startend_us.contiguous()
            bg_track_tracks_packinfo = bg_track_tracks_packinfo.contiguous()
            bg_track_tracks_poses = bg_track_tracks_poses.contiguous()
            bg_track_tracks_timestamps_us = bg_track_tracks_timestamps_us.contiguous()
            bg_track_cuboids_dims = bg_track_cuboids_dims.contiguous()
            (
                bg_track_loss,
                bg_track_weighted_loss,
                node_semantic_loss,
                node_semantic_weighted_loss,
            ) = _bg_track_dispatch(bg_track_density_logits_min)(
                bg_track_positions,
                bg_track_density_logits,
                bg_track_camera_timestamps_startend_us,
                bg_track_tracks_packinfo,
                bg_track_tracks_poses,
                bg_track_tracks_timestamps_us,
                bg_track_cuboids_dims,
                bg_track_semantic_logits,
                list(joint_background_allowed_class_ids),
                joint_node_primary_predicate,
                dispatch_n_semantic_points,
                list(dispatch_background_semantic_logits),
                list(dispatch_semantic_segments),
                other_node_density_logits,
                list(joint_node_other_semantic_logits),
                list(joint_node_other_segments),
                bg_track_lambda,
                joint_node_lambda,
            )
        else:
            bg_track_loss = torch.zeros(1, dtype=torch.float32, device=bg_track_density_logits.device)
            inactive_joint_loss = bg_track_density_logits[:0]
            node_semantic_loss = inactive_joint_loss
            bg_track_weighted_loss = inactive_joint_loss
            node_semantic_weighted_loss = inactive_joint_loss

        # Layout mirrored by SCALAR_LOSS_OUTPUT_INDEX at module scope.
        return (
            rgb_loss,
            lidar_loss,
            bg_loss,
            bg_lidar_loss,
            grids_drift_loss,
            grid_camera_spatial_tv_loss,
            grid_frame_spatial_tv_loss,
            scale_loss,
            bg_tex_loss,
            density_loss,
            out_of_bound_loss,
            z_scale_loss,
            intensity_loss,
            raydrop_loss,
            bg_track_loss,
            node_semantic_loss,
            semantic_loss,
            deform_loss,
            bg_track_weighted_loss,
            node_semantic_weighted_loss,
            semantic_weighted_loss,
            deform_weighted_loss,
        )
