# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import unittest

from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from gsplat.losses import BG_GRID_NUM_COLS, BG_GRID_NUM_ROWS

from nurec.config.model import SkyEnvMapBackgroundConfig
from nurec.config.trainer import TrainerConfig
from nurec.datasets.ncore import NCOREDataset
from nurec.datasets.tracks import CuboidTracks
from nurec.libs.losses.functional.cuda_losses_function import GRID_NUM_CHANNELS, CudaLossesFunction
from nurec.libs.losses.models.loss_fns import total_variation_spatial
from nurec.libs.losses.models.losses_module import ModuleLosses
from nurec.libs.losses.models.registry import make_loss
from nurec.libs.losses.orchestration.config import LossConfig
from nurec.libs.losses.orchestration.loss_aggregator import LossAggregator
from nurec.models.background import BaseBackground, SkyEnvMapBackground
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import BaseGaussianModel, RigidGaussianModel
from nurec.models.nn_extensions import TypedModuleDict, TypedModuleList
from nurec.models.post_processing import BasePostProcessing, BilateralGridPerCamera, BilateralGridPerFrame
from nurec.utils.batch import (
    CameraFrameLabels,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    LidarFrameLabels,
    RenderingBatch,
    RenderingData,
)
from nurec.utils.types import (
    ExtraSignal,
    GaussiansCompositeReturn,
    GaussiansRenderReturn,
    RayFlags,
    TrackFlags,
)


# Minimal constructible config for every loss name in ModuleLosses.available
# (keyed by registry name; composed full name is <key>_<fn>_<reduce>). Keep in
# lock-step with the available list: the coverage test enforces it.
MINIMAL_CUDA_LOSS_CONFIGS: dict[str, dict] = {
    "rgb": {"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "ssim": {
        "fn": "ssim",
        "lambda_": 1.0,
        "reduce": {"name": "mean"},
        "window_size": 11,
        "channel": 3,
        "mask": {"mode": "GT", "value": 0.0},
    },
    "lidar": {"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "background": {"fn": "mse", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "background_lidar": {"fn": "mse", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "bilateral_grid_drift": {"fn": "identity_distance", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "bilateral_grid_per_camera_tv": {"fn": "total_variation_spatial", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "bilateral_grid_per_frame_spatial_tv": {
        "fn": "total_variation_spatial",
        "lambda_": 1.0,
        "reduce": {"name": "mean"},
    },
    "gaussian_scale": {"fn": "abs", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "sky_env_map_background": {"fn": "total_variation_spatial", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "gaussian_density": {"fn": "abs", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "out_of_bound": {"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "gaussian_z_scale": {
        "fn": "abs",
        "lambda_": 1.0,
        "reduce": {"name": "mean"},
        "layer_name": "road",
        "road_z_scale": 0.05,
    },
    "intensity": {"fn": "mse", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "raydrop": {"fn": "mse", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "background_in_track_gaussian": {
        "fn": "bce_with_logits",
        "lambda_": 1.0,
        "reduce": {"name": "mean"},
        "layer_names": ["background"],
    },
    "node_semantic_gaussians": {"fn": "bce_with_logits", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "semantic": {"fn": "cross_entropy", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "deform_smoothness": {"fn": "abs", "lambda_": 1.0, "reduce": {"name": "mean"}},
}


class MockTrainerConfig(TrainerConfig):
    def __init__(self):
        super().__init__(
            max_epochs=1,
            check_val_every_n_epoch=1,
            precision="32",
            log_every_n_steps=1,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )


class TestLossAggregatorRouting(unittest.TestCase):
    def _ssim_config(self, *, semantic_weights: bool = False) -> LossConfig:
        ssim_cfg: dict[str, object] = {
            "fn": "ssim",
            "lambda_": 1.0,
            "reduce": {"name": "mean"},
            "window_size": 11,
            "channel": 3,
            "mask": {"mode": "GT", "value": 0.0},
        }
        if semantic_weights:
            ssim_cfg["mask_semantic_classes"] = ["ROAD"]
            ssim_cfg["semantic_lambdas"] = [2.0]
        return LossConfig.model_validate({"ssim": ssim_cfg})

    def test_plain_ssim_routes_to_module_losses(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        aggregator = LossAggregator(self._ssim_config(), MockTrainerConfig())

        self.assertEqual([loss.name for loss in aggregator.in_cuda.losses], ["ssim_ssim_mean"])
        self.assertEqual(aggregator.losses, [])

    def test_semantic_weighted_ssim_stays_on_legacy_path(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        aggregator = LossAggregator(self._ssim_config(semantic_weights=True), MockTrainerConfig())

        self.assertEqual(aggregator.in_cuda.losses, [])
        self.assertEqual([loss.name for loss in aggregator.losses], ["ssim_ssim_mean"])


class TestCudaLosses(unittest.TestCase):
    def test_semantic_cross_entropy_int64_targets(self):
        """int64 semantic targets (incl. an explicit -100 ignore row) match the
        uint8 arm and torch.nn.functional.cross_entropy.

        uint8 targets cannot carry the -100 ignore index, so only this arm can
        exercise SEMANTIC_IGNORE_INDEX on a flag-valid row.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda")
        torch.manual_seed(3)
        n_rows, n_classes = 64, 33
        logits = torch.randn(n_rows, n_classes, device=device)
        flags = torch.full((1, 8, 8, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32, device=device)
        flags.view(-1)[3] |= RayFlags.SKY_SEMANTIC.value  # flag-ignored row
        targets_u8 = torch.randint(0, n_classes, (n_rows,), dtype=torch.uint8, device=device)

        def run(targets):
            logits_leaf = logits.detach().clone().requires_grad_(True)
            outputs = CudaLossesFunction.forward(
                flags,
                torch.zeros((1, 8, 8, 3), dtype=torch.float32, device=device),
                torch.zeros((1, 8, 8, 3), dtype=torch.float32, device=device),
                -1.0,
                torch.zeros((1, 1, 1, 1), dtype=torch.int32, device=device),
                torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=device),
                torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=device),
                -1.0,
                torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=device),
                torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=device),
                -1.0,
                torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=device),
                torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=device),
                -1.0,
                torch.zeros((n_rows,), dtype=torch.float32, device=device),
                -1.0,
                torch.zeros((1,), dtype=torch.float32, device=device),
                -1.0,
                torch.zeros((12, 1, 1, 1), dtype=torch.float32, device=device),
                torch.zeros((12, 1, 1, 1), dtype=torch.float32, device=device),
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                torch.zeros((1, 3), dtype=torch.float32, device=device),
                -1.0,
                torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=device),
                -1.0,
                torch.zeros((1,), dtype=torch.float32, device=device),
                -1.0,
                torch.ones((1,), dtype=torch.float32, device=device),
                torch.zeros((1, 3), dtype=torch.float32, device=device),
                torch.zeros((1, 3), dtype=torch.float32, device=device),
                -1.0,
                torch.zeros((1, 3), dtype=torch.float32, device=device),
                -1.0,
                -1.0,
                torch.empty((0, 3), dtype=torch.float32, device=device),
                torch.empty((0,), dtype=torch.float32, device=device),
                torch.empty((0, 1), dtype=torch.float32, device=device),
                None,
                (),
                torch.empty((0, 2), dtype=torch.int64, device=device),
                torch.empty((0, 2), dtype=torch.int32, device=device),
                torch.empty((0, 7), dtype=torch.float32, device=device),
                torch.empty(0, dtype=torch.int64, device=device),
                torch.empty((0, 3), dtype=torch.float32, device=device),
                -1.0,
                -1.0,
                torch.empty(0, dtype=torch.float32, device=device),
                None,
                logits_leaf,
                targets,
                2.0,  # semantic_lambda
                torch.empty((0, 3), dtype=torch.float32, device=device),
                None,
                None,
                -1.0,
            )
            raw, weighted = outputs[16], outputs[20]
            weighted.backward()
            assert logits_leaf.grad is not None
            return raw.detach(), weighted.detach(), logits_leaf.grad.clone()

        raw_u8, weighted_u8, grad_u8 = run(targets_u8)
        raw_i64, weighted_i64, grad_i64 = run(targets_u8.long())
        # The uint8/int64 template instantiations may accumulate the workspace
        # partials in a different order (ulp-level float differences), so the
        # parity tolerance is tight but not bitwise.
        torch.testing.assert_close(raw_i64, raw_u8, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(grad_i64, grad_u8, rtol=1e-6, atol=1e-7)

        # -100 on a flag-valid row: only expressible with int64 targets.
        targets_ignore = targets_u8.long()
        targets_ignore[5] = -100
        raw_ig, weighted_ig, grad_ig = run(targets_ignore)
        # Denominator contract (parity with the Python mask-mean): flag-valid
        # rows count in the denominator even when their target is the ignore
        # index; the ignored row contributes zero to the numerator only.
        contributing = torch.ones(n_rows, dtype=torch.bool, device=device)
        contributing[3] = False  # SKY_SEMANTIC-flagged row
        contributing[5] = False  # explicit ignore index (numerator only)
        n_flag_valid = n_rows - 1
        reference = (
            torch.nn.functional.cross_entropy(logits[contributing], targets_ignore[contributing], reduction="sum")
            / n_flag_valid
        )
        torch.testing.assert_close(raw_ig, reference, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(weighted_ig, 2.0 * reference, rtol=1e-5, atol=1e-6)
        self.assertEqual(grad_ig[5].abs().max().item(), 0.0, "ignored row must get zero logits grad")

    def test_cuda_simple_gradient_cubemap(self):
        self._run_cuda_simple_gradient("cubemap")

    def test_cuda_simple_gradient_equirectangular(self):
        self._run_cuda_simple_gradient("equirectangular")

    def _run_cuda_simple_gradient(self, envmap_type):
        """Simple gradient test with minimal complexity to debug gradient flow."""

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda")
        torch.manual_seed(42)

        # Simple 2x2 RGB test
        B, H, W, C = 1, 2, 2, 3
        rgb_pred = torch.randn(B, H, W, C, device=device, requires_grad=True)
        rgb_gt = torch.randn(B, H, W, C, device=device)
        rgb_flags = torch.full((B, H, W, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32, device=device)

        # Simple 2x2 Lidar test
        lidar_pred = torch.randn(B, H, W, 1, device=device, requires_grad=True)
        lidar_gt = torch.randn(B, H, W, 1, device=device)
        lidar_flags = torch.zeros((B, H, W, 1), dtype=torch.int32, device=device)  # all valid

        # Simple 2x2 Background
        n_rays = B * H * W
        rgb_flags_flat = rgb_flags.view(-1)
        rgb_flags_flat[0] |= RayFlags.SKY_SEMANTIC.value
        sky_mask = (rgb_flags_flat & RayFlags.SKY_SEMANTIC.value) != 0
        bg_gt = torch.ones(n_rays, device=device)
        bg_gt[sky_mask] = 0.0  # Ground truth mask: 1 for non-sky (foreground), 0 for sky
        bg_pred = torch.rand(n_rays, device=device, requires_grad=True)

        # Simple 2x2 Background Lidar
        lidar_flags_flat = lidar_flags.view(-1)
        lidar_flags_flat[0] |= RayFlags.SKY_SEMANTIC.value
        sky_mask_lidar = (lidar_flags_flat & RayFlags.SKY_SEMANTIC.value) != 0
        bg_lidar_gt = torch.ones(n_rays, device=device)
        bg_lidar_gt[sky_mask_lidar] = 0.0  # Ground truth mask: 1 for non-sky (foreground), 0 for sky
        bg_lidar_pred = torch.rand(n_rays, device=device, requires_grad=True)

        # Compute valid counts manually
        rgb_valid = ((rgb_flags & RayFlags.RGB_LABEL.value) != 0) & ((rgb_flags & RayFlags.INVALID.value) == 0)
        n_valid_rgb = rgb_valid.sum().item()
        rgb_factor = 1.0 / n_valid_rgb if n_valid_rgb > 0 else 0.0

        lidar_valid = ((lidar_flags & RayFlags.INVALID.value) == 0) & ((lidar_flags & RayFlags.DROPPED.value) == 0)
        n_valid_lidar = lidar_valid.sum().item()
        lidar_factor = 1.0 / n_valid_lidar if n_valid_lidar > 0 else 0.0

        bg_valid = (
            ((rgb_flags_flat & RayFlags.INVALID.value) == 0)
            & ((rgb_flags_flat & RayFlags.HARMONIZED.value) == 0)
            & ((rgb_flags_flat & RayFlags.SYNTHETIC.value) == 0)
        )
        n_valid_bg_count = bg_valid.sum().item()
        bg_factor = 1.0 / n_valid_bg_count if n_valid_bg_count > 0 else 0.0

        bg_lidar_valid = ((lidar_flags_flat & RayFlags.INVALID.value) == 0) & (
            (lidar_flags_flat & RayFlags.DROPPED.value) == 0
        )
        n_valid_bg_lidar = bg_lidar_valid.sum().item()
        bg_lidar_factor = 1.0 / n_valid_bg_lidar if n_valid_bg_lidar > 0 else 0.0

        # Simple bilateral grid
        grid_per_camera = torch.randn(
            (1, GRID_NUM_CHANNELS, 2, 2, 2), dtype=torch.float32, device=device, requires_grad=True
        )
        grid_per_frame = torch.randn(
            (3, GRID_NUM_CHANNELS, 2, 2, 2), dtype=torch.float32, device=device, requires_grad=True
        )
        num_cam = (
            grid_per_camera.shape[0] * grid_per_camera.shape[2] * grid_per_camera.shape[3] * grid_per_camera.shape[4]
        )
        num_frame = (
            grid_per_frame.shape[0] * grid_per_frame.shape[2] * grid_per_frame.shape[3] * grid_per_frame.shape[4]
        )
        grid_per_camera_factor = 1.0 / num_cam if num_cam > 0 else 0.0
        grid_per_frame_factor = 1.0 / num_frame if num_frame > 0 else 0.0

        grid_camera_spatial_tv_factor = 1.0 / num_cam if num_cam > 0 else 0.0
        grid_frame_spatial_tv_factor = 1.0 / num_frame if num_frame > 0 else 0.0

        # Prepare flattened grids for CUDALossesFunction
        grids_cam = grid_per_camera.view(
            grid_per_camera.shape[0] * grid_per_camera.shape[1],
            grid_per_camera.shape[2],
            grid_per_camera.shape[3],
            grid_per_camera.shape[4],
        )
        grids_frame = grid_per_frame.view(
            grid_per_frame.shape[0] * grid_per_frame.shape[1],
            grid_per_frame.shape[2],
            grid_per_frame.shape[3],
            grid_per_frame.shape[4],
        )

        # Simple gaussian scales test (pre-activation values for exp() activation)
        N_scales = 10
        # Use randn() - 3 to keep scales small after exp()
        gaussian_scales = (torch.randn(N_scales, 3, device=device) - 3.0).requires_grad_(True)
        scale_factor = 1.0 / gaussian_scales.numel()

        if envmap_type == "cubemap":
            # Cubemap texture with spatial variation (B=1, faces=6, H=4, W=4, C=3).
            # Faces are regularized independently; TV should not couple face indices.
            bg_tex = torch.randn((1, 6, 4, 4, 3), device=device, dtype=torch.float32, requires_grad=True)
        elif envmap_type == "equirectangular":
            bg_tex = torch.randn((1, 4, 4, 3), device=device, dtype=torch.float32, requires_grad=True)
        else:
            raise ValueError(f"Unsupported envmap_type: {envmap_type}")
        bg_tex_factor = 1.0

        # Simple gaussian densities test
        N_densities = 15
        gaussian_densities = torch.randn(N_densities, device=device, requires_grad=True)
        density_factor = 1.0 / gaussian_densities.numel()

        # Prepare out_of_bound positions and cuboid dimensions
        out_of_bound_positions = torch.tensor([[0.2, -0.1, 0.25], [0.1, 0.2, 0.3]], device=device, requires_grad=True)
        out_of_bound_cuboid_dims = torch.tensor([[0.3, 0.25, 0.4], [0.2, 0.15, 0.3]], device=device)
        out_of_bound_factor = 1.0 / (out_of_bound_positions.shape[0] * out_of_bound_positions.shape[1])

        # Simple gaussian z-scales test (pre-activation values for exp() activation)
        N_z_scales = 8192
        # Use randn() - 3 to keep z-scales small after exp()
        # Pass full [N, 3] tensor, CUDA kernel will extract z-component
        gaussian_z_scales = (torch.randn(N_z_scales, 3, device=device) - 3.0).requires_grad_(True)
        z_scale_threshold = 0.01  # Threshold for ReLU (applied to exp(z_scale))
        z_scale_factor = 1.0 / N_z_scales  # Mean over N elements (not N*3)

        # Simple intensity test (enabled, uses same B,H,W as lidar)
        intensity_pred = torch.randn(B, H, W, 1, device=device, requires_grad=True)
        intensity_gt = torch.randn(B, H, W, 1, device=device)
        intensity_factor = 1.0 / n_valid_lidar if n_valid_lidar > 0 else 0.0

        # Simple raydrop test (enabled, uses same B,H,W as lidar)
        raydrop_pred = torch.randn(B, H, W, 1, device=device, requires_grad=True)
        raydrop_gt = torch.randn(B, H, W, 1, device=device)
        raydrop_factor = 1.0 / n_valid_lidar if n_valid_lidar > 0 else 0.0

        # Forward pass
        (
            rgb_loss,
            lidar_loss,
            bg_loss,
            bg_lidar_loss,
            grid_drift_loss,
            grid_camera_spatial_tv_loss,
            grid_frame_spatial_tv_loss,
            scale_loss,
            bg_tex_loss,
            density_loss,
            out_of_bound_loss,
            z_scale_loss,
            intensity_loss,
            raydrop_loss,
            _bg_track_loss,
            _node_semantic_loss,
            _semantic_loss,
            _deform_loss,
            _bg_track_weighted_loss,
            _node_semantic_weighted_loss,
            _semantic_weighted_loss,
            _deform_weighted_loss,
        ) = CudaLossesFunction.forward(
            rgb_flags,
            rgb_pred,
            rgb_gt,
            rgb_factor,
            lidar_flags,
            lidar_pred,
            lidar_gt,
            lidar_factor,
            intensity_pred,
            intensity_gt,
            intensity_factor,
            raydrop_pred,
            raydrop_gt,
            raydrop_factor,
            bg_pred,
            bg_factor,
            bg_lidar_pred,
            bg_lidar_factor,
            grids_cam,
            grids_frame,
            grid_per_camera_factor,
            grid_per_frame_factor,
            grid_camera_spatial_tv_factor,
            grid_frame_spatial_tv_factor,
            gaussian_scales,
            scale_factor,
            bg_tex,
            bg_tex_factor,
            gaussian_densities,
            density_factor,
            torch.ones(max(N_scales, N_densities), device=device, dtype=torch.float32),  # gaussian_visibility
            out_of_bound_positions,
            out_of_bound_cuboid_dims,
            out_of_bound_factor,
            gaussian_z_scales,
            z_scale_threshold,
            z_scale_factor,
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty((0, 1), dtype=torch.float32, device=device),
            None,
            (),
            torch.empty((0, 2), dtype=torch.int64, device=device),
            torch.empty((0, 2), dtype=torch.int32, device=device),
            torch.empty((0, 7), dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty((0, 3), dtype=torch.float32, device=device),
            -1.0,
            -1.0,
            torch.empty(0, dtype=torch.float32, device=device),
            None,
            torch.empty((0, 1), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.uint8, device=device),
            -1.0,
            torch.empty((0, 3), dtype=torch.float32, device=device),
            None,
            None,
            -1.0,
        )

        # Backward pass - sum all losses to get scalar
        total_loss = (
            rgb_loss.sum()
            + lidar_loss.sum()
            + bg_loss.sum()
            + bg_lidar_loss.sum()
            + grid_drift_loss.sum()
            + grid_camera_spatial_tv_loss.sum()
            + grid_frame_spatial_tv_loss.sum()
            + scale_loss.sum()
            + bg_tex_loss.sum()
            + density_loss.sum()
            + out_of_bound_loss.sum()
            + z_scale_loss.sum()
            + intensity_loss.sum()
            + raydrop_loss.sum()
        )

        self.assertTrue(total_loss.requires_grad, "Total loss should require gradients")
        self.assertGreater(total_loss.item(), 0, "Total loss should be positive")

        total_loss.backward()

        # Check gradients
        self.assertIsNotNone(rgb_pred.grad, "RGB pred should have gradients")
        self.assertIsNotNone(lidar_pred.grad, "Lidar pred should have gradients")
        self.assertIsNotNone(intensity_pred.grad, "Intensity pred should have gradients")
        self.assertIsNotNone(raydrop_pred.grad, "Raydrop pred should have gradients")
        self.assertIsNotNone(bg_pred.grad, "Background pred should have gradients")
        self.assertIsNotNone(bg_lidar_pred.grad, "Background Lidar pred should have gradients")
        self.assertIsNotNone(grid_per_camera.grad, "grid_per_camera should have gradients")
        self.assertIsNotNone(grid_per_frame.grad, "grid_per_frame should have gradients")
        self.assertIsNotNone(gaussian_scales.grad, "gaussian_scales should have gradients")
        self.assertIsNotNone(bg_tex.grad, "bg_tex should have gradients")
        self.assertIsNotNone(gaussian_densities.grad, "gaussian_densities should have gradients")
        self.assertIsNotNone(out_of_bound_positions.grad, "out_of_bound positions should have gradients")
        self.assertIsNotNone(gaussian_z_scales.grad, "gaussian_z_scales should have gradients")

        # Gradients should be non-zero
        self.assertGreater(rgb_pred.grad.abs().sum().item(), 1e-8, "RGB gradients should be non-zero")
        self.assertGreater(lidar_pred.grad.abs().sum().item(), 1e-8, "Lidar gradients should be non-zero")
        self.assertGreater(bg_pred.grad.abs().sum().item(), 1e-8, "Background gradients should be non-zero")
        self.assertGreater(bg_lidar_pred.grad.abs().sum().item(), 1e-8, "Background Lidar gradients should be non-zero")
        self.assertGreater(
            grid_per_camera.grad.abs().sum().item(), 1e-8, "grid_per_camera gradients should be non-zero"
        )
        self.assertGreater(grid_per_frame.grad.abs().sum().item(), 1e-8, "grid_per_frame gradients should be non-zero")
        self.assertGreater(
            gaussian_scales.grad.abs().sum().item(), 1e-8, "gaussian_scales gradients should be non-zero"
        )
        self.assertGreater(bg_tex_loss.sum().item(), 0.0, "bg_tex_loss should be non-zero for non-uniform texture")
        self.assertGreater(
            gaussian_densities.grad.abs().sum().item(), 1e-8, "gaussian_densities gradients should be non-zero"
        )
        self.assertGreater(
            out_of_bound_positions.grad.abs().sum().item(), 1e-8, "out_of_bound gradients should be non-zero"
        )
        self.assertGreater(
            gaussian_z_scales.grad.abs().sum().item(), 1e-8, "gaussian_z_scales gradients should be non-zero"
        )
        self.assertGreater(intensity_pred.grad.abs().sum().item(), 1e-8, "Intensity gradients should be non-zero")
        self.assertGreater(raydrop_pred.grad.abs().sum().item(), 1e-8, "Raydrop gradients should be non-zero")

        # Manual gradient check for L1 loss: d/dx |x - gt| = sign(x - gt) * factor
        # For valid pixels only
        expected_rgb_grad = torch.zeros_like(rgb_pred)
        diff_rgb = rgb_pred - rgb_gt
        expected_rgb_grad[rgb_valid.expand_as(rgb_pred)] = (
            torch.sign(diff_rgb[rgb_valid.expand_as(rgb_pred)]) * rgb_factor
        )

        expected_lidar_grad = torch.zeros_like(lidar_pred)
        diff_lidar = lidar_pred - lidar_gt
        expected_lidar_grad[lidar_valid.expand_as(lidar_pred)] = (
            torch.sign(diff_lidar[lidar_valid.expand_as(lidar_pred)]) * lidar_factor
        )

        # Manual gradient check for MSE background: d/dx (p - g)^2 = 2*(p - g)*bg_factor
        expected_bg_grad = torch.zeros_like(bg_pred)
        diff_bg = bg_pred - bg_gt
        expected_bg_grad[bg_valid] = 2 * diff_bg[bg_valid] * bg_factor

        # Manual gradient check for MSE background Lidar: d/dx (p - g)^2 = 2*(p - g)*bg_lidar_factor
        expected_bg_lidar_grad = torch.zeros_like(bg_lidar_pred)
        diff_bg_lidar = bg_lidar_pred - bg_lidar_gt
        expected_bg_lidar_grad[bg_lidar_valid] = 2 * diff_bg_lidar[bg_lidar_valid] * bg_lidar_factor

        # Manual gradient check for bilateral grid drift loss

        # Camera grid expected gradients
        reshaped_cam = grid_per_camera.view(
            grid_per_camera.shape[0], BG_GRID_NUM_ROWS, BG_GRID_NUM_COLS, *grid_per_camera.shape[2:]
        )
        identity = torch.eye(BG_GRID_NUM_ROWS, BG_GRID_NUM_COLS, device=device).view(
            1, BG_GRID_NUM_ROWS, BG_GRID_NUM_COLS, *([1] * (reshaped_cam.dim() - 3))
        )
        diff_cam = reshaped_cam - identity
        norm_cam = torch.norm(diff_cam, p="fro", dim=(1, 2))
        norm_cam_exp = norm_cam.view(norm_cam.shape[0], 1, 1, norm_cam.shape[1], norm_cam.shape[2], norm_cam.shape[3])
        expected_cam_grad = (diff_cam / norm_cam_exp) * grid_per_camera_factor
        expected_cam_grad = expected_cam_grad.view_as(grid_per_camera)

        # Frame grid expected gradients
        reshaped_frame = grid_per_frame.view(
            grid_per_frame.shape[0], BG_GRID_NUM_ROWS, BG_GRID_NUM_COLS, *grid_per_frame.shape[2:]
        )
        # Reuse identity with frame dims: reshape identity to match reshaped_frame
        identity_frame = torch.eye(BG_GRID_NUM_ROWS, BG_GRID_NUM_COLS, device=device).view(
            1, BG_GRID_NUM_ROWS, BG_GRID_NUM_COLS, *([1] * (reshaped_frame.dim() - 3))
        )
        diff_frame = reshaped_frame - identity_frame
        norm_frame = torch.norm(diff_frame, p="fro", dim=(1, 2))
        norm_frame_exp = norm_frame.view(
            norm_frame.shape[0], 1, 1, norm_frame.shape[1], norm_frame.shape[2], norm_frame.shape[3]
        )
        expected_frame_grad = (diff_frame / norm_frame_exp) * grid_per_frame_factor
        expected_frame_grad = expected_frame_grad.view_as(grid_per_frame)

        # Manual gradient check for grid spatial tv loss
        def grid_tv_spatial_grad(x: torch.Tensor) -> torch.Tensor | None:
            x = x.detach().requires_grad_(True)
            x.grad = None
            tv_loss_python = total_variation_spatial(x)
            tv_loss_python.mean().backward()
            return x.grad

        expected_cam_grad += grid_tv_spatial_grad(grid_per_camera)
        expected_frame_grad += grid_tv_spatial_grad(grid_per_frame)

        # Manual gradient check for gaussian scale loss with exp() activation
        # Loss: mean(exp(x) + exp(y) + exp(z)) where mean is via scale_factor pre-multiplication
        # Since exp() is always positive, abs(exp(x)) = exp(x)
        # Gradient: d/dx_preact = exp(x_preact) * scale_factor
        expected_scale_grad = torch.exp(gaussian_scales) * scale_factor

        # Manual gradient check for out_of_bound loss
        expected_out_of_bound_grad = torch.zeros_like(out_of_bound_positions)
        detach_out_of_bound_positions = out_of_bound_positions.detach()
        abs_pos = detach_out_of_bound_positions.abs()
        losses = abs_pos - out_of_bound_cuboid_dims / 2
        mask = losses > 0  # Coordinates where ReLU(loss) is active
        # ReLU derivative is +/-1 for exceeded coords; scale by out_of_bound_factor
        expected_out_of_bound_grad[mask] = torch.sign(detach_out_of_bound_positions)[mask] * out_of_bound_factor

        if envmap_type == "cubemap":
            # Manual gradient check for sky_env_map: treat cubemap faces as independent batch entries.
            bg_tex_permuted = bg_tex.detach().permute(1, 4, 0, 2, 3).requires_grad_(True)
            tv_loss_python = total_variation_spatial(bg_tex_permuted)
            (tv_loss_python.mean() * bg_tex_factor).backward()
            expected_bg_tex_grad = bg_tex_permuted.grad.permute(2, 0, 3, 4, 1)
        else:
            # Manual gradient check for equirectangular sky_env_map: wrap height and width boundaries.
            bg_tex_py = bg_tex.detach().requires_grad_(True)
            bg_tex_permuted = bg_tex_py.permute(0, 3, 1, 2).unsqueeze(2)
            tv_loss_python = total_variation_spatial(bg_tex_permuted, wrap_around=True)
            (tv_loss_python.mean() * bg_tex_factor).backward()
            expected_bg_tex_grad = bg_tex_py.grad

        # Manual gradient check for gaussian density loss: d/dx |x| * factor = sign(x) * factor
        expected_density_grad = torch.sign(gaussian_densities) * density_factor

        # Compare gradients
        self.assertTrue(
            torch.allclose(rgb_pred.grad, expected_rgb_grad, atol=1e-6, rtol=1e-5),
            f"RGB gradients don't match expected. Max diff: {(rgb_pred.grad - expected_rgb_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(lidar_pred.grad, expected_lidar_grad, atol=1e-6, rtol=1e-5),
            f"Lidar gradients don't match expected. Max diff: {(lidar_pred.grad - expected_lidar_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(bg_pred.grad, expected_bg_grad, atol=1e-6, rtol=1e-5),
            f"Background gradients don't match expected. Max diff: {(bg_pred.grad - expected_bg_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(bg_lidar_pred.grad, expected_bg_lidar_grad, atol=1e-6, rtol=1e-5),
            f"Background Lidar gradients don't match expected. Max diff: {(bg_lidar_pred.grad - expected_bg_lidar_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(grid_per_frame.grad, expected_frame_grad, atol=1e-6, rtol=1e-5),
            f"Grid per frame gradients don't match expected. Max diff: {(grid_per_frame.grad - expected_frame_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(grid_per_camera.grad, expected_cam_grad, atol=1e-6, rtol=1e-5),
            f"Grid per camera gradients don't match expected. Max diff: {(grid_per_camera.grad - expected_cam_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(gaussian_scales.grad, expected_scale_grad, atol=1e-6, rtol=1e-5),
            f"Gaussian scales gradients don't match expected. Max diff: {(gaussian_scales.grad - expected_scale_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(bg_tex.grad, expected_bg_tex_grad, atol=1e-6, rtol=1e-5),
            f"{envmap_type} bg_tex gradients don't match expected. "
            f"Max diff: {(bg_tex.grad - expected_bg_tex_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(gaussian_densities.grad, expected_density_grad, atol=1e-6, rtol=1e-5),
            f"Gaussian densities gradients don't match expected. Max diff: {(gaussian_densities.grad - expected_density_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(out_of_bound_positions.grad, expected_out_of_bound_grad, atol=1e-6, rtol=1e-5),
            f"out_of_bound gradients don't match expected. Max diff: {(out_of_bound_positions.grad - expected_out_of_bound_grad).abs().max().item():.6e}",
        )

    def test_cuda_vs_python_loss_forward_and_backward(self):
        """Compare fused CUDA losses against Python losses for forward and backward parity.

        Selective Adam relies on the same Gaussian visibility mask used by scale/density
        regularizers, so this verifies the fused path applies that mask consistently with the
        Python reference while other losses continue to produce matching gradients.
        """
        # Random RGB and lidar test data with some invalid pixels
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda")
        torch.manual_seed(42)

        # Use different dimensions for RGB and lidar to test flexibility
        B_rgb, H_rgb, W_rgb, C = 2, 120, 160, 3
        B_lidar, H_lidar, W_lidar = 3, 64, 180

        # ===== RGB data =====
        rgb_gt = torch.randn(B_rgb, H_rgb, W_rgb, C, device=device)
        rgb_pred_cuda = torch.randn(B_rgb, H_rgb, W_rgb, C, device=device, requires_grad=True)
        rgb_pred_py = rgb_pred_cuda.clone().detach().requires_grad_(True)

        # RGB flags: start valid, some pixels randomly marked INVALID and SKY_SEMANTIC
        rgb_flags = torch.full((B_rgb, H_rgb, W_rgb, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32, device=device)
        rgb_invalid_mask = torch.rand(B_rgb, H_rgb, W_rgb, 1, device=device) < 0.1
        rgb_sky_mask = torch.rand(B_rgb, H_rgb, W_rgb, 1, device=device) < 0.2
        rgb_flags[rgb_invalid_mask] |= RayFlags.INVALID.value
        rgb_flags[rgb_sky_mask] |= RayFlags.SKY_SEMANTIC.value

        # ===== Semantic data =====
        n_semantic_classes = 33
        semantic_gt = torch.randint(
            0,
            n_semantic_classes,
            (B_rgb, H_rgb, W_rgb, 1),
            dtype=torch.uint8,
            device=device,
        )
        rgb_flags[0, 0, 0, 0] |= RayFlags.SKY_SEMANTIC.value
        semantic_gt[0, 0, 0, 0] = 255
        semantic_storage_cuda = torch.randn(
            (B_rgb * H_rgb * W_rgb, n_semantic_classes + 3),
            device=device,
            requires_grad=True,
        )
        semantic_storage_py = semantic_storage_cuda.detach().clone().requires_grad_(True)
        semantic_logits_cuda = semantic_storage_cuda[:, 1 : n_semantic_classes + 1]
        semantic_logits_py = semantic_storage_py[:, 1 : n_semantic_classes + 1]
        self.assertFalse(semantic_logits_cuda.is_contiguous())

        # ===== Deform-smoothness data =====
        deformation_base = torch.tensor(
            [[-1.4, -0.3, 0.2], [0.7, -0.9, 1.1], [2.0, -1.6, 0.4], [-0.8, 0.6, -2.3]],
            device=device,
        )
        deform_mask_base = torch.tensor([[1.0], [0.25], [0.0], [1.5]], device=device)
        deformation_cuda = deformation_base.clone().requires_grad_(True)
        deformation_py = deformation_base.clone().requires_grad_(True)
        deform_mask_cuda = deform_mask_base.clone().requires_grad_(True)
        deform_mask_py = deform_mask_base.clone().requires_grad_(True)

        # ===== Lidar data =====
        lidar_gt = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device) * 100.0  # Random distances 0-100m
        lidar_pred_cuda = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device) * 100.0
        lidar_pred_cuda.requires_grad_(True)
        lidar_pred_py = lidar_pred_cuda.clone().detach().requires_grad_(True)

        # Lidar flags: start with no flags (valid), some rays randomly marked INVALID or DROPPED or SKY_SEMANTIC
        lidar_flags = torch.zeros((B_lidar, H_lidar, W_lidar, 1), dtype=torch.int32, device=device)
        lidar_invalid_mask = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device) < 0.05
        lidar_dropped_mask = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device) < 0.05
        lidar_sky_mask = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device) < 0.2
        lidar_flags[lidar_invalid_mask] |= RayFlags.INVALID.value
        lidar_flags[lidar_dropped_mask] |= RayFlags.DROPPED.value
        lidar_flags[lidar_sky_mask] |= RayFlags.SKY_SEMANTIC.value

        # ===== Intensity data =====
        intensity_gt = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device)
        intensity_pred_cuda = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device).requires_grad_(True)
        intensity_pred_py = intensity_pred_cuda.clone().detach().requires_grad_(True)

        # ===== Raydrop data =====
        raydrop_gt = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device)
        raydrop_pred_cuda = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device).requires_grad_(True)
        raydrop_pred_py = raydrop_pred_cuda.clone().detach().requires_grad_(True)

        # Define loss config and trainer
        config = LossConfig.model_validate(
            {
                "rgb": {"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "lidar": {"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "background": {"fn": "mse", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "background_lidar": {"fn": "mse", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "sky_env_map_background": {
                    "fn": "total_variation_spatial",
                    "lambda_": 1.0,
                    "reduce": {"name": "mean"},
                },
                "bilateral_grid_drift": {"fn": "identity_distance", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "bilateral_grid_per_camera_tv": {
                    "fn": "total_variation_spatial",
                    "lambda_": 1.0,
                    "reduce": {"name": "mean"},
                },
                "bilateral_grid_per_frame_spatial_tv": {
                    "fn": "total_variation_spatial",
                    "lambda_": 1.0,
                    "reduce": {"name": "mean"},
                },
                "gaussian_scale": {
                    "fn": "abs",
                    "lambda_": 1.0,
                    "reduce": {"name": "mean"},
                    "layer_lambdas": {"road": 1.0, "background": 0.0, "rigid": 0.0},
                    "visibility_filter": True,
                },
                # Reads the "road" node's scales, so its gradient lands in the
                # gaussian_scales tensors already compared below. road_z_scale sits
                # mid-distribution of exp(scales[:, 2]) so the relu is partly active.
                "gaussian_z_scale": {
                    "fn": "abs",
                    "lambda_": 1.0,
                    "reduce": {"name": "mean"},
                    "layer_name": "road",
                    "road_z_scale": 1.0,
                },
                "gaussian_density": {
                    "fn": "abs",
                    "lambda_": 0.02,
                    "reduce": {"name": "mean"},
                    "layer_lambdas": {"background": 0.0, "rigid": 0.0},
                    "visibility_filter": True,
                },
                "out_of_bound": {"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "intensity": {"fn": "mse", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "raydrop": {"fn": "mse", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "semantic": {"fn": "cross_entropy", "lambda_": 0.37, "reduce": {"name": "mean"}},
                "deform_smoothness": {"fn": "abs", "lambda_": 1.7, "reduce": {"name": "mean"}},
                "background_in_track_gaussian": {
                    "fn": "bce_with_logits",
                    "lambda_": 0.125,
                    "reduce": {"name": "mean"},
                    "layer_names": ["background"],
                    "layer_labels_to_use": {"background": ["person"]},
                    "density_logits_min": -20.0,
                },
                "node_semantic_gaussians": {
                    "fn": "bce_with_logits",
                    "lambda_": 1.7,
                    "reduce": {"name": "mean"},
                    "layer_labels_to_use": {"road": ["road"]},
                    "layer_labels_to_exclude": {"background": ["road"]},
                },
            }
        )
        trainer_config = MockTrainerConfig()

        # Mock GaussiansComposite to avoid actual model initialization
        class MockGaussiansComposite(GaussiansComposite):
            def __init__(
                self,
                *,
                gaussians_nodes: dict[str, BaseGaussianModel] | None = None,
                post_processings: list[BasePostProcessing] | None = None,
                **attrs,
            ):
                # intentionally skip super().__init__()
                torch.nn.Module.__init__(self)
                self.gaussians_nodes = TypedModuleDict[BaseGaussianModel](gaussians_nodes or {})
                self.post_processings = TypedModuleList[BasePostProcessing](post_processings or [])
                for key, value in attrs.items():
                    setattr(self, key, value)

            def get_gaussians_node_ids(self, non_empty_only: bool = False) -> list[str]:
                return list(self.gaussians_nodes.keys())

        # Mock BilateralGridPerCamera and BilateralGridPerFrame to avoid actual initialization
        class MockBilateralGridPerCamera(BilateralGridPerCamera):
            def __init__(self, **attrs):
                # Initialize nn.Module to allow setting module attributes
                nn.Module.__init__(self)
                for k, v in attrs.items():
                    setattr(self, k, v)

        class MockBilateralGridPerFrame(BilateralGridPerFrame):
            def __init__(self, **attrs):
                # Initialize nn.Module to allow setting module attributes
                nn.Module.__init__(self)
                for k, v in attrs.items():
                    setattr(self, k, v)

        # Mock Gaussian node with scales
        # Aligns with nurec/models/gaussians/gaussians_model.py where preactivated values are stored
        class MockGaussianNode(BaseGaussianModel):
            def __init__(self, scales, densities, positions, semantic_logits):
                # intentionally skip super().__init__()
                torch.nn.Module.__init__(self)
                # Store pre-activation (log-space) values, aligning with gaussians_model.py
                self.scales = scales
                self.densities = densities
                self.positions = nn.Parameter(positions, requires_grad=False)
                self.extra_signal = nn.Parameter(semantic_logits, requires_grad=False)
                self.extra_signal_param_infos = ({"semantic_logits": 0}, [semantic_logits.shape[-1]])

            def get_scales(self, preactivation=False) -> torch.Tensor:
                if preactivation:
                    # Return log-space values for CUDA (which will apply exp() in kernel)
                    return self.scales
                else:
                    # Return activated values for Python loss
                    return torch.exp(self.scales)

            def get_densities(self, preactivation=False):
                return self.densities

        class MockRigidGaussianModel(RigidGaussianModel):
            def __init__(self, positions: torch.Tensor, tracks: CuboidTracks, cuboid_ids: torch.Tensor):
                # intentionally skip super().__init__()
                torch.nn.Module.__init__(self)
                self.positions = torch.nn.Parameter(positions)
                self.cuboid_tracks = tracks
                self.gaussian_cuboid_ids = nn.Buffer(cuboid_ids)
                self.extra_signal = nn.Parameter(
                    torch.zeros((positions.shape[0], n_semantic_classes), device=positions.device),
                    requires_grad=False,
                )
                self.extra_signal_param_infos = ({"semantic_logits": 0}, [n_semantic_classes])

            def get_positions(self) -> torch.nn.Parameter:
                return cast(torch.nn.Parameter, self.positions)

            def get_num_gaussians(self) -> int:
                return self.positions.shape[0]

            def _get_zeros(self, *extra_dims: int) -> torch.Tensor:
                return torch.zeros(
                    self.positions.shape[0],
                    *extra_dims,
                    device=self.positions.device,
                    dtype=self.positions.dtype,
                )

            def get_scales(self, preactivation=False) -> torch.Tensor:
                return self._get_zeros(3)

            def get_densities(self, preactivation=False):
                return self._get_zeros(1)

        # ===== Rigid Gaussian data for out of bound loss =====
        positions_out_base = torch.tensor(
            [[0.4, -0.2, 0.15], [-0.35, 0.5, -0.25], [0.1, 0.05, -0.45]],
            dtype=torch.float32,
            device=device,
        )
        gaussian_cuboid_ids = torch.tensor([2, 0, 1], dtype=torch.long, device=device)
        identity = np.eye(4, dtype=np.float32)
        track_pose_start = identity.copy()
        track_pose_start[0, 3] = -1.0
        track_pose_end = identity.copy()
        track_pose_end[0, 3] = 1.0
        tracks = CuboidTracks.Factory.from_numpy(
            tracks_id=["track_0", "track_1", "track_2"],
            tracks_poses=[np.stack((track_pose_start, track_pose_end)) for _ in range(3)],
            tracks_timestamps_us=[np.array([0, 10], dtype=np.int64) for _ in range(3)],
            tracks_label_class=["car"] * 3,
            tracks_flags=[TrackFlags.NONE] * 3,
            cuboids_dims=[
                np.array([0.6, 0.5, 0.4], dtype=np.float32),
                np.array([0.7, 0.9, 0.8], dtype=np.float32),
                np.array([0.2, 0.3, 0.9], dtype=np.float32),
            ],
            device=device,
        )

        positions_out_cuda = positions_out_base.clone().detach().requires_grad_(True)
        positions_out_py = positions_out_base.clone().detach().requires_grad_(True)
        node_cuda = MockRigidGaussianModel(positions_out_cuda, tracks, gaussian_cuboid_ids)
        node_py = MockRigidGaussianModel(positions_out_py, tracks, gaussian_cuboid_ids)

        # ===== Test CUDA implementation =====
        N_gaussians = 500
        n_rays_rgb = B_rgb * H_rgb * W_rgb
        n_rays_lidar = B_lidar * H_lidar * W_lidar
        gaussian_vis_mask = (torch.rand(N_gaussians + 7, device=device) > 0.3).float()

        road_positions = torch.zeros((N_gaussians, 3), device=device)
        road_semantic_logits = torch.full((N_gaussians, n_semantic_classes), -7.0, device=device)
        road_semantic_winners = torch.arange(N_gaussians, device=device) % 2
        road_semantic_winners[road_semantic_winners == 0] = 32
        road_semantic_logits[
            torch.arange(N_gaussians, device=device),
            road_semantic_winners,
        ] = 7.0
        background_positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.1, 0.0, 0.0]],
            device=device,
        )
        background_scales = torch.zeros((4, 3), device=device)
        background_density_values = torch.tensor([[-25.0], [0.0], [-1.0], [3.0]], device=device)
        background_densities_cuda = background_density_values.clone().requires_grad_(True)
        background_densities_py = background_density_values.clone().requires_grad_(True)
        background_semantic_logits = torch.full((4, n_semantic_classes), -7.0, device=device)
        background_semantic_logits[
            torch.arange(4, device=device),
            torch.tensor([1, 32, 1, 32], device=device),
        ] = 7.0
        # Make opacity require grad for background loss
        opacity_cuda = torch.ones(n_rays_rgb, device=device, requires_grad=True)
        rendered_cam_cuda = GaussiansRenderReturn(
            rgb=rgb_pred_cuda.reshape(-1, C),
            opacity=opacity_cuda,
            distance=torch.zeros(n_rays_rgb, device=device),
            extra_ray_signals=ExtraSignal(semantic_logits=semantic_logits_cuda),
            visibility=gaussian_vis_mask,
        )
        # Make opacity require grad for background lidar loss
        opacity_lidar_cuda = torch.ones(n_rays_lidar, device=device, requires_grad=True)
        rendered_lidar_cuda = GaussiansRenderReturn(
            rgb=torch.zeros(n_rays_lidar, 3, device=device),
            opacity=opacity_lidar_cuda,
            distance=lidar_pred_cuda.reshape(-1, 1),
            extra_ray_signals=ExtraSignal(
                intensity=intensity_pred_cuda.reshape(-1, 1),
                raydrop=raydrop_pred_cuda.reshape(-1, 1),
            ),
        )
        results_cuda = GaussiansCompositeReturn(
            rendered_cam=rendered_cam_cuda,
            rendered_lidar=rendered_lidar_cuda,
            deform_smoothness=deformation_cuda,
            deform_smoothness_mask=deform_mask_cuda,
        )

        camera_labels_cuda = CameraFrameLabels(flags=rgb_flags, rgb=rgb_gt, semantic=semantic_gt)
        camera_cuda = DataBatch.Camera(
            meta=[FrameMeta(unique_sensor_idx=0, unique_frame_idx=0, training_frame_idx=0)],
            labels=camera_labels_cuda,
        )

        lidar_labels_cuda = LidarFrameLabels(
            flags=lidar_flags, distance=lidar_gt, intensity=intensity_gt, raydrop=raydrop_gt
        )
        lidar_cuda = DataBatch.Lidar(
            meta=[FrameMeta(unique_sensor_idx=1, unique_frame_idx=0, training_frame_idx=0)],
            labels=lidar_labels_cuda,
        )

        data_batch_cuda = DataBatch(idx=0, worker_id=None, sequence_id=["dummy"], camera=camera_cuda, lidar=lidar_cuda)
        timestamps = torch.tensor([[4, 6], [4, 6]], dtype=torch.int64, device=device)
        rendering_camera = RenderingData(
            rays=torch.zeros((B_rgb, H_rgb, W_rgb, 6), device=device),
            sensor_model_parameters=[cast(Any, SimpleNamespace()) for _ in range(B_rgb)],
            poses_tquat_startend=torch.tensor(
                [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2] * B_rgb,
                device=device,
            ),
            timestamps_startend_us=timestamps,
            timestamps_startend_us_cpu=timestamps.cpu(),
        )
        batch_cuda = DataAndRenderingBatch(
            data=data_batch_cuda,
            rendering=RenderingBatch(camera=rendering_camera),
        )

        bilateral_grid_per_camera_cuda = torch.randn(
            (1, GRID_NUM_CHANNELS, 6, 8, 8), dtype=torch.float32, device=device, requires_grad=True
        )
        bilateral_grid_per_frame_cuda = torch.randn(
            (29, GRID_NUM_CHANNELS, 8, 16, 16), dtype=torch.float32, device=device, requires_grad=True
        )

        # Mock Gaussian scales for gaussian_scale loss
        gaussian_scales_cuda = torch.randn(N_gaussians, 3, device=device) * 0.1
        gaussian_scales_cuda = gaussian_scales_cuda.detach().requires_grad_(True)

        # Mock Gaussian densities for gaussian_density loss
        gaussian_densities_cuda = torch.randn(N_gaussians, 1, device=device)
        gaussian_densities_cuda = gaussian_densities_cuda.detach().requires_grad_(True)

        bg_cfg = SkyEnvMapBackgroundConfig.model_validate(
            {
                "name": "sky-env-map",
                "envmap_type": "equirectangular",
                "height": H_rgb,
                "width": W_rgb,
                "saturate_radiance": False,
                "should_inpaint": False,
                "inpaint_threshold": 0.0,
                "inpaint_kernel_size": 5,
                "min_grad_updates": 0,
                "composite_in_linear_space": False,
            }
        )

        model_cuda = MockGaussiansComposite(
            gaussians_nodes={
                "background": MockGaussianNode(
                    background_scales,
                    background_densities_cuda,
                    background_positions,
                    background_semantic_logits,
                ),
                "road": MockGaussianNode(
                    gaussian_scales_cuda,
                    gaussian_densities_cuda,
                    road_positions,
                    road_semantic_logits,
                ),
                "rigid": node_cuda,
            },
            post_processings=[
                MockBilateralGridPerCamera(bilateral_grid=SimpleNamespace(grid=bilateral_grid_per_camera_cuda)),
                MockBilateralGridPerFrame(bilateral_grid=SimpleNamespace(grid=bilateral_grid_per_frame_cuda)),
            ],
            background=SkyEnvMapBackground(bg_cfg, trainer_config),
        )

        model_cuda = model_cuda.to(device)

        # Compute forward with CUDA losses enabled
        assert config is not None
        aggregator_cuda = LossAggregator(config, trainer_config)
        dataset = mock.MagicMock(spec=NCOREDataset)
        datasource = mock.MagicMock()
        datasource.get_semantic_classes_map.return_value = {"person": 1, "road": 32}
        dataset.get_datasource.return_value = datasource
        aggregator_cuda.initialize(dataset)
        self.assertEqual(aggregator_cuda.losses, [])
        agg_ret_cuda = aggregator_cuda(step=0, model=model_cuda, results=results_cuda, target=batch_cuda)
        loss_cuda = agg_ret_cuda.total_value

        # Ensure loss requires grad
        self.assertTrue(loss_cuda.requires_grad, "CUDAloss should require gradients")
        self.assertGreater(loss_cuda.item(), 0, "CUDAloss should be positive")

        # Compute backward for CUDA
        scalar_gradient_coefficients = {
            "background_in_track_gaussian_bce_with_logits_mean": (2.0, 3.0),
            "node_semantic_gaussians_bce_with_logits_mean": (5.0, 7.0),
            "semantic_cross_entropy_mean": (11.0, 13.0),
            "deform_smoothness_abs_mean": (17.0, 19.0),
        }
        scalar_cuda_loss_names = tuple(scalar_gradient_coefficients)
        cuda_raw_values = {
            loss_name: agg_ret_cuda.loss_returns[loss_name].value.sum() for loss_name in scalar_cuda_loss_names
        }
        cuda_backward_losses: list[torch.Tensor] = []
        for loss_name, loss_return in agg_ret_cuda.loss_returns.items():
            if loss_name in scalar_gradient_coefficients:
                raw_coefficient, weighted_coefficient = scalar_gradient_coefficients[loss_name]
                cuda_backward_losses.append(
                    raw_coefficient * cuda_raw_values[loss_name] + weighted_coefficient * loss_return.reduced_value
                )
            else:
                cuda_backward_losses.append(loss_return.reduced_value)
        torch.stack(cuda_backward_losses).sum().backward()
        rgb_grad_cuda = rgb_pred_cuda.grad.clone()
        lidar_grad_cuda = lidar_pred_cuda.grad.clone()
        bg_grad_cuda = opacity_cuda.grad.clone()
        bg_lidar_grad_cuda = opacity_lidar_cuda.grad.clone()
        per_camera_grad_cuda = bilateral_grid_per_camera_cuda.grad.clone()
        per_frame_grad_cuda = bilateral_grid_per_frame_cuda.grad.clone()
        gaussian_scales_grad_cuda = gaussian_scales_cuda.grad.clone()
        gaussian_densities_grad_cuda = gaussian_densities_cuda.grad.clone()
        out_of_bound_grad_cuda = node_cuda.positions.grad.clone()
        intensity_grad_cuda = intensity_pred_cuda.grad.clone()
        raydrop_grad_cuda = raydrop_pred_cuda.grad.clone()
        semantic_grad_cuda = semantic_storage_cuda.grad.clone()
        deformation_grad_cuda = deformation_cuda.grad.clone()
        deform_mask_grad_cuda = deform_mask_cuda.grad.clone()
        background_density_grad_cuda = background_densities_cuda.grad.clone()

        # Capture individual loss returns for forward comparisons
        out_of_bound_key = "out_of_bound_l1_mean"
        self.assertIn(out_of_bound_key, agg_ret_cuda.loss_returns)
        out_of_bound_loss_cuda = agg_ret_cuda.loss_returns[out_of_bound_key].reduced_value.detach()

        intensity_key = "intensity_mse_mean"
        self.assertIn(intensity_key, agg_ret_cuda.loss_returns)
        intensity_loss_cuda = agg_ret_cuda.loss_returns[intensity_key].reduced_value.detach()

        raydrop_key = "raydrop_mse_mean"
        self.assertIn(raydrop_key, agg_ret_cuda.loss_returns)
        raydrop_loss_cuda = agg_ret_cuda.loss_returns[raydrop_key].reduced_value.detach()

        # ===== Test Python implementation =====
        # Make opacity require grad for background loss
        opacity_py = torch.ones(n_rays_rgb, device=device, requires_grad=True)
        rendered_cam_py = GaussiansRenderReturn(
            rgb=rgb_pred_py.reshape(-1, C),
            opacity=opacity_py,
            distance=torch.zeros(n_rays_rgb, device=device),
            extra_ray_signals=ExtraSignal(semantic_logits=semantic_logits_py),
            visibility=gaussian_vis_mask,
        )
        # Make opacity require grad for background lidar loss
        opacity_lidar_py = torch.ones(n_rays_lidar, device=device, requires_grad=True)
        rendered_lidar_py = GaussiansRenderReturn(
            rgb=torch.zeros(n_rays_lidar, 3, device=device),
            opacity=opacity_lidar_py,
            distance=lidar_pred_py.reshape(-1, 1),
            extra_ray_signals=ExtraSignal(
                intensity=intensity_pred_py.reshape(-1, 1),
                raydrop=raydrop_pred_py.reshape(-1, 1),
            ),
        )
        results_py = GaussiansCompositeReturn(
            rendered_cam=rendered_cam_py,
            rendered_lidar=rendered_lidar_py,
            deform_smoothness=deformation_py,
            deform_smoothness_mask=deform_mask_py,
        )

        camera_labels_py = CameraFrameLabels(flags=rgb_flags, rgb=rgb_gt, semantic=semantic_gt)
        camera_py = DataBatch.Camera(
            meta=[FrameMeta(unique_sensor_idx=0, unique_frame_idx=0, training_frame_idx=0)],
            labels=camera_labels_py,
        )

        lidar_labels_py = LidarFrameLabels(
            flags=lidar_flags, distance=lidar_gt, intensity=intensity_gt, raydrop=raydrop_gt
        )
        lidar_py = DataBatch.Lidar(
            meta=[FrameMeta(unique_sensor_idx=1, unique_frame_idx=0, training_frame_idx=0)],
            labels=lidar_labels_py,
        )

        data_batch_py = DataBatch(idx=0, worker_id=None, sequence_id=["dummy"], camera=camera_py, lidar=lidar_py)
        batch_py = DataAndRenderingBatch(
            data=data_batch_py,
            rendering=RenderingBatch(camera=rendering_camera),
        )

        bilateral_grid_per_camera_py = bilateral_grid_per_camera_cuda.clone().detach().requires_grad_(True)
        bilateral_grid_per_frame_py = bilateral_grid_per_frame_cuda.clone().detach().requires_grad_(True)

        # Mock Gaussian scales for Python (clone from CUDA to ensure same input)
        gaussian_scales_py = gaussian_scales_cuda.clone().detach().requires_grad_(True)

        # Mock Gaussian densities for Python (clone from CUDA to ensure same input)
        gaussian_densities_py = gaussian_densities_cuda.clone().detach().requires_grad_(True)

        model_py = MockGaussiansComposite(
            gaussians_nodes={
                "background": MockGaussianNode(
                    background_scales.clone(),
                    background_densities_py,
                    background_positions.clone(),
                    background_semantic_logits.clone(),
                ),
                "road": MockGaussianNode(
                    gaussian_scales_py,
                    gaussian_densities_py,
                    road_positions.clone(),
                    road_semantic_logits.clone(),
                ),
                "rigid": node_py,
            },
            post_processings=[
                MockBilateralGridPerCamera(bilateral_grid=SimpleNamespace(grid=bilateral_grid_per_camera_py)),
                MockBilateralGridPerFrame(bilateral_grid=SimpleNamespace(grid=bilateral_grid_per_frame_py)),
            ],
            background=SkyEnvMapBackground(bg_cfg, trainer_config),
        )
        model_py = model_py.to(device)

        # Instantiate with force_disable_cuda to bypass CUDA fused losses
        assert config is not None
        aggregator_py = LossAggregator(config, trainer_config, force_disable_cuda=True)
        aggregator_py.initialize(dataset)
        self.assertEqual(aggregator_py.in_cuda.losses, [])
        agg_ret_py = aggregator_py(step=0, model=model_py, results=results_py, target=batch_py)
        loss_py = agg_ret_py.total_value

        # Ensure loss requires grad
        self.assertTrue(loss_py.requires_grad, "Python loss should require gradients")
        self.assertGreater(loss_py.item(), 0, "Python loss should be positive")

        # Compute backward for Python
        torch_raw_values = {
            loss_name: agg_ret_py.loss_returns[loss_name].reduce_fn(
                agg_ret_py.loss_returns[loss_name].value,
                reduce_mask=agg_ret_py.loss_returns[loss_name].reduce_mask,
            )
            for loss_name in scalar_cuda_loss_names
        }
        torch_backward_losses: list[torch.Tensor] = []
        for loss_name, loss_return in agg_ret_py.loss_returns.items():
            if loss_name in scalar_gradient_coefficients:
                raw_coefficient, weighted_coefficient = scalar_gradient_coefficients[loss_name]
                torch_backward_losses.append(
                    raw_coefficient * torch_raw_values[loss_name] + weighted_coefficient * loss_return.reduced_value
                )
            else:
                torch_backward_losses.append(loss_return.reduced_value)
        torch.stack(torch_backward_losses).sum().backward()
        rgb_grad_py = rgb_pred_py.grad.clone()
        lidar_grad_py = lidar_pred_py.grad.clone()
        bg_grad_py = opacity_py.grad.clone()
        bg_lidar_grad_py = opacity_lidar_py.grad.clone()
        per_camera_grad_py = bilateral_grid_per_camera_py.grad.clone()
        per_frame_grad_py = bilateral_grid_per_frame_py.grad.clone()
        gaussian_scales_grad_py = gaussian_scales_py.grad.clone()
        gaussian_densities_grad_py = gaussian_densities_py.grad.clone()
        out_of_bound_grad_py = node_py.positions.grad.clone()
        intensity_grad_py = intensity_pred_py.grad.clone()
        raydrop_grad_py = raydrop_pred_py.grad.clone()
        semantic_grad_py = semantic_storage_py.grad.clone()
        deformation_grad_py = deformation_py.grad.clone()
        deform_mask_grad_py = deform_mask_py.grad.clone()
        background_density_grad_py = background_densities_py.grad.clone()

        self.assertIn(out_of_bound_key, agg_ret_py.loss_returns)
        out_of_bound_loss_py = agg_ret_py.loss_returns[out_of_bound_key].reduced_value.detach()

        self.assertIn(intensity_key, agg_ret_py.loss_returns)
        intensity_loss_py = agg_ret_py.loss_returns[intensity_key].reduced_value.detach()

        self.assertIn(raydrop_key, agg_ret_py.loss_returns)
        raydrop_loss_py = agg_ret_py.loss_returns[raydrop_key].reduced_value.detach()

        # ===== Compare forward results =====
        print(f"Forward: CUDA={loss_cuda.item():.6f}, Python={loss_py.item():.6f}")
        forward_rel_diff = abs(loss_cuda.item() - loss_py.item()) / (abs(loss_py.item()) + 1e-8)
        self.assertTrue(
            torch.allclose(loss_cuda, loss_py, atol=1e-4, rtol=5e-3),
            f"Forward pass mismatch: CUDA={loss_cuda.item()}, Python={loss_py.item()}, rel_diff={forward_rel_diff:.6e}",
        )

        for loss_name in scalar_cuda_loss_names:
            torch.testing.assert_close(
                cuda_raw_values[loss_name],
                torch_raw_values[loss_name],
                atol=2e-5,
                rtol=2e-5,
            )
            torch.testing.assert_close(
                agg_ret_cuda.loss_returns[loss_name].reduced_value,
                agg_ret_py.loss_returns[loss_name].reduced_value,
                atol=2e-5,
                rtol=2e-5,
            )

        z_scale_key = "gaussian_z_scale_abs_mean"
        self.assertIn(z_scale_key, agg_ret_cuda.loss_returns)
        self.assertIn(z_scale_key, agg_ret_py.loss_returns)
        self.assertGreater(agg_ret_py.loss_returns[z_scale_key].reduced_value.item(), 0.0)
        torch.testing.assert_close(
            agg_ret_cuda.loss_returns[z_scale_key].reduced_value,
            agg_ret_py.loss_returns[z_scale_key].reduced_value,
            atol=2e-5,
            rtol=2e-5,
        )

        # ===== Compare RGB backward results =====
        rgb_cuda_nonzero = rgb_grad_cuda.abs().gt(1e-8).sum().item()
        rgb_py_nonzero = rgb_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(rgb_cuda_nonzero, 0, "CUDARGB gradients should be non-zero")
        self.assertGreater(rgb_py_nonzero, 0, "Python RGB gradients should be non-zero")

        rgb_grad_max_abs_diff = (rgb_grad_cuda - rgb_grad_py).abs().max().item()
        print(f"Backward RGB: nonzero={rgb_cuda_nonzero}/{rgb_grad_cuda.numel()}, max_diff={rgb_grad_max_abs_diff:.6e}")
        self.assertTrue(
            torch.allclose(rgb_grad_cuda, rgb_grad_py, atol=1e-6, rtol=1e-8),
            f"RGB Backward pass mismatch: max_abs_diff={rgb_grad_max_abs_diff:.6e}",
        )

        # ===== Compare Lidar backward results =====
        lidar_cuda_nonzero = lidar_grad_cuda.abs().gt(1e-8).sum().item()
        lidar_py_nonzero = lidar_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(lidar_cuda_nonzero, 0, "CUDALidar gradients should be non-zero")
        self.assertGreater(lidar_py_nonzero, 0, "Python Lidar gradients should be non-zero")

        lidar_grad_max_abs_diff = (lidar_grad_cuda - lidar_grad_py).abs().max().item()
        print(
            f"Backward Lidar: nonzero={lidar_cuda_nonzero}/{lidar_grad_cuda.numel()}, max_diff={lidar_grad_max_abs_diff:.6e}"
        )
        # Allow larger tolerance for Lidar backward differences
        self.assertTrue(
            torch.allclose(lidar_grad_cuda, lidar_grad_py, atol=1e-4, rtol=1e-2),
            f"Lidar Backward pass mismatch: max_abs_diff={lidar_grad_max_abs_diff:.6e}",
        )

        # ===== Compare Background backward results =====
        bg_cuda_nonzero = bg_grad_cuda.abs().gt(1e-8).sum().item()
        bg_py_nonzero = bg_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(bg_cuda_nonzero, 0, "CUDABackground gradients should be non-zero")
        self.assertGreater(bg_py_nonzero, 0, "Python Background gradients should be non-zero")

        bg_grad_max_abs_diff = (bg_grad_cuda - bg_grad_py).abs().max().item()
        print(
            f"Backward Background: nonzero={bg_cuda_nonzero}/{bg_grad_cuda.numel()}, max_diff={bg_grad_max_abs_diff:.6e}"
        )
        self.assertTrue(
            torch.allclose(bg_grad_cuda, bg_grad_py, atol=1e-6, rtol=1e-8),
            f"Background Backward pass mismatch: max_abs_diff={bg_grad_max_abs_diff:.6e}",
        )

        # ===== Compare Background Lidar backward results =====
        bg_lidar_cuda_nonzero = bg_lidar_grad_cuda.abs().gt(1e-8).sum().item()
        bg_lidar_py_nonzero = bg_lidar_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(bg_lidar_cuda_nonzero, 0, "CUDABackground Lidar gradients should be non-zero")
        self.assertGreater(bg_lidar_py_nonzero, 0, "Python Background Lidar gradients should be non-zero")

        bg_lidar_grad_max_abs_diff = (bg_lidar_grad_cuda - bg_lidar_grad_py).abs().max().item()
        print(
            f"Backward Background Lidar: nonzero={bg_lidar_cuda_nonzero}/{bg_lidar_grad_cuda.numel()}, max_diff={bg_lidar_grad_max_abs_diff:.6e}"
        )
        # Allow larger tolerance for Background Lidar backward differences
        self.assertTrue(
            torch.allclose(bg_lidar_grad_cuda, bg_lidar_grad_py, atol=1e-4, rtol=1e-2),
            f"Background Lidar Backward pass mismatch: max_abs_diff={bg_lidar_grad_max_abs_diff:.6e}",
        )

        # ===== Compare Bilateral Grid backward results (Potentially contributed by Grid Dift & Spatial TV) =====
        per_camera_grad_max_abs_diff = (per_camera_grad_cuda - per_camera_grad_py).abs().max().item()
        per_frame_grad_max_abs_diff = (per_frame_grad_cuda - per_frame_grad_py).abs().max().item()
        # Allow some tolerance for grid drift gradient differences
        self.assertTrue(
            torch.allclose(per_camera_grad_cuda, per_camera_grad_py, atol=1e-4, rtol=1e-2),
            f"Bilateral Grid Drift Per Camera backward pass mismatch: max_abs_diff={per_camera_grad_max_abs_diff:.6e}",
        )
        self.assertTrue(
            torch.allclose(per_frame_grad_cuda, per_frame_grad_py, atol=1e-4, rtol=1e-2),
            f"Bilateral Grid Drift Per Frame backward pass mismatch: max_abs_diff={per_frame_grad_max_abs_diff:.6e}",
        )

        # ===== Compare Gaussian Scale backward results =====
        gaussian_scales_cuda_nonzero = gaussian_scales_grad_cuda.abs().gt(1e-8).sum().item()
        gaussian_scales_py_nonzero = gaussian_scales_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(gaussian_scales_cuda_nonzero, 0, "CUDAGaussian scales gradients should be non-zero")
        self.assertGreater(gaussian_scales_py_nonzero, 0, "Python Gaussian scales gradients should be non-zero")

        gaussian_scales_grad_max_abs_diff = (gaussian_scales_grad_cuda - gaussian_scales_grad_py).abs().max().item()
        self.assertTrue(
            torch.allclose(gaussian_scales_grad_cuda, gaussian_scales_grad_py, atol=1e-6, rtol=1e-8),
            f"Gaussian Scales Backward pass mismatch: max_abs_diff={gaussian_scales_grad_max_abs_diff:.6e}",
        )

        # Compare sky-env-map spatial TV loss between CUDA and Python
        bg_tex_cuda = agg_ret_cuda.loss_returns["sky_env_map_background_total_variation_spatial_mean"].reduced_value
        bg_tex_py = agg_ret_py.loss_returns["sky_env_map_background_total_variation_spatial_mean"].reduced_value
        self.assertTrue(
            torch.allclose(bg_tex_cuda, bg_tex_py, atol=1e-6, rtol=1e-6),
            f"sky_env_map_background TV loss mismatch: CUDA={bg_tex_cuda.item()}, Python={bg_tex_py.item()}",
        )

        # ===== Compare Gaussian Density backward results =====
        gaussian_densities_cuda_nonzero = gaussian_densities_grad_cuda.abs().gt(1e-8).sum().item()
        gaussian_densities_py_nonzero = gaussian_densities_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(gaussian_densities_cuda_nonzero, 0, "CUDAGaussian densities gradients should be non-zero")
        self.assertGreater(gaussian_densities_py_nonzero, 0, "Python Gaussian densities gradients should be non-zero")

        gaussian_densities_grad_max_abs_diff = (
            (gaussian_densities_grad_cuda - gaussian_densities_grad_py).abs().max().item()
        )
        print(
            f"Backward Gaussian Densities: nonzero={gaussian_densities_cuda_nonzero}/{gaussian_densities_grad_cuda.numel()}, max_diff={gaussian_densities_grad_max_abs_diff:.6e}"
        )
        self.assertTrue(
            torch.allclose(gaussian_densities_grad_cuda, gaussian_densities_grad_py, atol=1e-6, rtol=1e-8),
            f"Gaussian Densities Backward pass mismatch: max_abs_diff={gaussian_densities_grad_max_abs_diff:.6e}",
        )

        # ===== Compare out_of_bound forward and backward results =====
        print(
            f"Forward out_of_bound: CUDA={out_of_bound_loss_cuda.item():.6f}, Python={out_of_bound_loss_py.item():.6f}"
        )
        self.assertTrue(
            torch.allclose(out_of_bound_loss_cuda, out_of_bound_loss_py, atol=1e-4, rtol=5e-3),
            f"out_of_bound forward mismatch: CUDA={out_of_bound_loss_cuda.item():.6f}, Python={out_of_bound_loss_py.item():.6f}",
        )

        out_of_bound_cuda_nonzero = out_of_bound_grad_cuda.abs().gt(1e-8).sum().item()
        out_of_bound_py_nonzero = out_of_bound_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(out_of_bound_cuda_nonzero, 0, "CUDAout_of_bound gradients should be non-zero")
        self.assertGreater(out_of_bound_py_nonzero, 0, "Python out_of_bound gradients should be non-zero")

        out_of_bound_grad_max_abs_diff = (out_of_bound_grad_cuda - out_of_bound_grad_py).abs().max().item()
        print(f"Backward out_of_bound: CUDA={out_of_bound_grad_cuda}, Python={out_of_bound_grad_py}")
        print(
            f"Backward out_of_bound: nonzero={out_of_bound_cuda_nonzero}/{out_of_bound_grad_cuda.numel()}, max_diff={out_of_bound_grad_max_abs_diff:.6e}"
        )
        self.assertTrue(
            torch.allclose(out_of_bound_grad_cuda, out_of_bound_grad_py, atol=1e-6, rtol=1e-8),
            f"out_of_bound backward mismatch: max_abs_diff={out_of_bound_grad_max_abs_diff:.6e}",
        )

        # ===== Compare Intensity forward results =====
        print(f"Forward intensity: CUDA={intensity_loss_cuda.item():.6f}, Python={intensity_loss_py.item():.6f}")
        self.assertTrue(
            torch.allclose(intensity_loss_cuda, intensity_loss_py, atol=1e-4, rtol=5e-3),
            f"Intensity forward mismatch: CUDA={intensity_loss_cuda.item():.6f}, Python={intensity_loss_py.item():.6f}",
        )

        # ===== Compare Raydrop forward results =====
        print(f"Forward raydrop: CUDA={raydrop_loss_cuda.item():.6f}, Python={raydrop_loss_py.item():.6f}")
        self.assertTrue(
            torch.allclose(raydrop_loss_cuda, raydrop_loss_py, atol=1e-4, rtol=5e-3),
            f"Raydrop forward mismatch: CUDA={raydrop_loss_cuda.item():.6f}, Python={raydrop_loss_py.item():.6f}",
        )

        # ===== Compare Intensity backward results =====
        intensity_cuda_nonzero = intensity_grad_cuda.abs().gt(1e-8).sum().item()
        intensity_py_nonzero = intensity_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(intensity_cuda_nonzero, 0, "CUDAIntensity gradients should be non-zero")
        self.assertGreater(intensity_py_nonzero, 0, "Python Intensity gradients should be non-zero")

        intensity_grad_max_diff = (intensity_grad_cuda - intensity_grad_py).abs().max().item()
        print(
            f"Backward Intensity: nonzero={intensity_cuda_nonzero}/{intensity_grad_cuda.numel()}, max_diff={intensity_grad_max_diff:.6e}"
        )
        self.assertTrue(
            torch.allclose(intensity_grad_cuda, intensity_grad_py, atol=1e-4, rtol=1e-2),
            f"Intensity Backward mismatch: max_abs_diff={intensity_grad_max_diff:.6e}",
        )

        # ===== Compare Raydrop backward results =====
        raydrop_cuda_nonzero = raydrop_grad_cuda.abs().gt(1e-8).sum().item()
        raydrop_py_nonzero = raydrop_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(raydrop_cuda_nonzero, 0, "CUDARaydrop gradients should be non-zero")
        self.assertGreater(raydrop_py_nonzero, 0, "Python Raydrop gradients should be non-zero")

        raydrop_grad_max_diff = (raydrop_grad_cuda - raydrop_grad_py).abs().max().item()
        print(
            f"Backward Raydrop: nonzero={raydrop_cuda_nonzero}/{raydrop_grad_cuda.numel()}, max_diff={raydrop_grad_max_diff:.6e}"
        )
        self.assertTrue(
            torch.allclose(raydrop_grad_cuda, raydrop_grad_py, atol=1e-4, rtol=1e-2),
            f"Raydrop Backward mismatch: max_abs_diff={raydrop_grad_max_diff:.6e}",
        )

        torch.testing.assert_close(semantic_grad_cuda, semantic_grad_py, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(deformation_grad_cuda, deformation_grad_py, atol=2e-5, rtol=2e-5)
        # Active lanes only: the two paths differ on masked-out lanes by design.
        mask_active = deform_mask_base != 0
        torch.testing.assert_close(
            deform_mask_grad_cuda[mask_active],
            deform_mask_grad_py[mask_active],
            atol=2e-5,
            rtol=2e-5,
        )
        # Masked-out lanes still carry the denominator term, so their gradient
        # is not zero; what the fused path omits there is the numerator term
        # |d| / denom, because it never reads the residual behind a zero mask.
        inactive = ~mask_active
        # A zero residual behind the mask would make both paths agree and the
        # assertions below vacuous, so require the fixture to exercise it.
        self.assertGreater(
            deformation_base[inactive.squeeze(-1)].abs().sum().item(),
            0.0,
            "fixture must carry a nonzero residual behind the mask",
        )
        gap = deform_mask_grad_py[inactive] - deform_mask_grad_cuda[inactive]
        self.assertTrue(
            torch.all(gap > 0),
            "the reference must exceed the fused path by the omitted numerator term",
        )
        # Recover the objective's scalar coefficient from an active lane, where
        # the two paths agree: grad = k * (|d| / denom - loss / denom).
        denom = deform_mask_base.sum()
        residual = deformation_base.abs().sum(dim=-1, keepdim=True)
        loss = (deform_mask_base * residual).sum() / denom
        active_idx = int(mask_active.squeeze(-1).nonzero()[0])
        k = deform_mask_grad_py[active_idx] / (residual[active_idx] / denom - loss / denom)
        torch.testing.assert_close(
            gap,
            (k * residual[inactive.squeeze(-1)] / denom).reshape(gap.shape),
            atol=2e-4,
            rtol=2e-3,
        )
        torch.testing.assert_close(
            background_density_grad_cuda,
            background_density_grad_py,
            atol=2e-5,
            rtol=2e-5,
        )
        self.assertEqual(background_density_grad_cuda[0].item(), 0.0)
        self.assertGreater(background_density_grad_cuda[2].abs().item(), 0.0)

        unmasked_deformation_cuda = deformation_base.clone().requires_grad_(True)
        unmasked_deformation_py = deformation_base.clone().requires_grad_(True)
        # All-zeros mask: mask_sum = 0 drives the sub-unit denominator clamp in
        # the finalize kernel and the suppressed mask-source denominator
        # gradient in backward — the edge case the workspace math preserves.
        zero_masked_deformation_cuda = deformation_base.clone().requires_grad_(True)
        zero_masked_deformation_py = deformation_base.clone().requires_grad_(True)
        zero_deform_mask_cuda = torch.zeros_like(deform_mask_base).requires_grad_(True)
        zero_deform_mask_py = torch.zeros_like(deform_mask_base).requires_grad_(True)
        packed_background_config = config["background_in_track_gaussian"].model_copy(update={"layer_labels_to_use": {}})
        # Select-none pin: bg-track filters 'background' but node-semantic is
        # configured for 'road' only, so the shared primary domain gets the
        # explicit _NODE_SEMANTIC_SELECT_NONE predicate.
        road_only_node_semantic_config = config["node_semantic_gaussians"].model_copy(
            update={"layer_labels_to_exclude": {}}
        )
        # Select-all pin: an explicitly empty layer_labels_to_use list means
        # "use nothing" -> every point of that node is penalized, on both
        # routes (predicate (), select_matches=False).
        select_all_background_config = config["node_semantic_gaussians"].model_copy(
            update={"layer_labels_to_use": {"background": []}, "layer_labels_to_exclude": {}}
        )
        for case_config, loss_names, case_results_cuda, case_results_py, cuda_inputs, python_inputs in (
            (
                LossConfig.model_validate({"background_in_track_gaussian": config["background_in_track_gaussian"]}),
                ("background_in_track_gaussian_bce_with_logits_mean",),
                results_cuda,
                results_py,
                (background_densities_cuda,),
                (background_densities_py,),
            ),
            (
                LossConfig.model_validate({"node_semantic_gaussians": config["node_semantic_gaussians"]}),
                ("node_semantic_gaussians_bce_with_logits_mean",),
                results_cuda,
                results_py,
                (background_densities_cuda, gaussian_densities_cuda),
                (background_densities_py, gaussian_densities_py),
            ),
            (
                LossConfig.model_validate(
                    {
                        "background_in_track_gaussian": packed_background_config,
                        "node_semantic_gaussians": config["node_semantic_gaussians"],
                    }
                ),
                (
                    "background_in_track_gaussian_bce_with_logits_mean",
                    "node_semantic_gaussians_bce_with_logits_mean",
                ),
                results_cuda,
                results_py,
                (background_densities_cuda, gaussian_densities_cuda),
                (background_densities_py, gaussian_densities_py),
            ),
            (
                LossConfig.model_validate(
                    {
                        "background_in_track_gaussian": config["background_in_track_gaussian"],
                        "node_semantic_gaussians": road_only_node_semantic_config,
                    }
                ),
                (
                    "background_in_track_gaussian_bce_with_logits_mean",
                    "node_semantic_gaussians_bce_with_logits_mean",
                ),
                results_cuda,
                results_py,
                (background_densities_cuda, gaussian_densities_cuda),
                (background_densities_py, gaussian_densities_py),
            ),
            (
                LossConfig.model_validate({"node_semantic_gaussians": select_all_background_config}),
                ("node_semantic_gaussians_bce_with_logits_mean",),
                results_cuda,
                results_py,
                (background_densities_cuda,),
                (background_densities_py,),
            ),
            (
                LossConfig.model_validate({"deform_smoothness": config["deform_smoothness"]}),
                ("deform_smoothness_abs_mean",),
                GaussiansCompositeReturn(deform_smoothness=unmasked_deformation_cuda),
                GaussiansCompositeReturn(deform_smoothness=unmasked_deformation_py),
                (unmasked_deformation_cuda,),
                (unmasked_deformation_py,),
            ),
            (
                LossConfig.model_validate({"deform_smoothness": config["deform_smoothness"]}),
                ("deform_smoothness_abs_mean",),
                GaussiansCompositeReturn(
                    deform_smoothness=zero_masked_deformation_cuda,
                    deform_smoothness_mask=zero_deform_mask_cuda,
                ),
                GaussiansCompositeReturn(
                    deform_smoothness=zero_masked_deformation_py,
                    deform_smoothness_mask=zero_deform_mask_py,
                ),
                # Mask omitted: with every lane masked out the fused path
                # reports no mask gradient. Pinned on the mixed mask above.
                (zero_masked_deformation_cuda,),
                (zero_masked_deformation_py,),
            ),
        ):
            case_cuda = LossAggregator(case_config, trainer_config)
            case_python = LossAggregator(case_config, trainer_config, force_disable_cuda=True)
            case_cuda.initialize(dataset)
            case_python.initialize(dataset)
            self.assertEqual(case_cuda.losses, [])
            self.assertEqual(case_python.in_cuda.losses, [])

            cuda_returns = case_cuda(
                step=0,
                model=model_cuda,
                results=case_results_cuda,
                target=batch_cuda,
            ).loss_returns
            python_returns = case_python(step=0, model=model_py, results=case_results_py, target=batch_py).loss_returns
            cuda_case_raw = {loss_name: cuda_returns[loss_name].value.sum() for loss_name in loss_names}
            python_case_raw = {
                loss_name: python_returns[loss_name].reduce_fn(
                    python_returns[loss_name].value,
                    reduce_mask=python_returns[loss_name].reduce_mask,
                )
                for loss_name in loss_names
            }
            for loss_name in loss_names:
                torch.testing.assert_close(cuda_case_raw[loss_name], python_case_raw[loss_name], atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(
                    cuda_returns[loss_name].reduced_value,
                    python_returns[loss_name].reduced_value,
                    atol=2e-5,
                    rtol=2e-5,
                )

            cuda_objective = torch.stack(
                [
                    scalar_gradient_coefficients[loss_name][0] * cuda_case_raw[loss_name]
                    + scalar_gradient_coefficients[loss_name][1] * cuda_returns[loss_name].reduced_value
                    for loss_name in loss_names
                ]
            ).sum()
            python_objective = torch.stack(
                [
                    scalar_gradient_coefficients[loss_name][0] * python_case_raw[loss_name]
                    + scalar_gradient_coefficients[loss_name][1] * python_returns[loss_name].reduced_value
                    for loss_name in loss_names
                ]
            ).sum()

            cuda_grads = torch.autograd.grad(cuda_objective, cuda_inputs)
            python_grads = torch.autograd.grad(python_objective, python_inputs)
            for cuda_grad, python_grad in zip(cuda_grads, python_grads, strict=True):
                torch.testing.assert_close(cuda_grad, python_grad, atol=2e-5, rtol=2e-5)

        # A node configured for node-semantic whose model lacks the
        # 'semantic_logits' extra signal is skipped with a warning on both
        # routes instead of crashing; the loss still runs on the remaining
        # configured nodes.
        missing_signal_config = LossConfig.model_validate(
            {"node_semantic_gaussians": config["node_semantic_gaussians"]}
        )
        missing_signal = mock.Mock(side_effect=ValueError("Invalid signal key: semantic_logits"))
        with (
            mock.patch.object(model_cuda.gaussians_nodes["road"], "get_extra_signal_by_key", missing_signal),
            mock.patch.object(model_py.gaussians_nodes["road"], "get_extra_signal_by_key", missing_signal),
        ):
            skip_cuda = LossAggregator(missing_signal_config, trainer_config)
            skip_python = LossAggregator(missing_signal_config, trainer_config, force_disable_cuda=True)
            skip_cuda.initialize(dataset)
            skip_python.initialize(dataset)
            skip_cuda_returns = skip_cuda(
                step=0, model=model_cuda, results=results_cuda, target=batch_cuda
            ).loss_returns
            skip_python_returns = skip_python(step=0, model=model_py, results=results_py, target=batch_py).loss_returns
        skip_name = "node_semantic_gaussians_bce_with_logits_mean"
        torch.testing.assert_close(
            skip_cuda_returns[skip_name].value.sum(),
            skip_python_returns[skip_name].reduce_fn(
                skip_python_returns[skip_name].value,
                reduce_mask=skip_python_returns[skip_name].reduce_mask,
            ),
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            skip_cuda_returns[skip_name].reduced_value,
            skip_python_returns[skip_name].reduced_value,
            atol=2e-5,
            rtol=2e-5,
        )
        skip_cuda_grad = torch.autograd.grad(skip_cuda_returns[skip_name].reduced_value, background_densities_cuda)[0]
        skip_python_grad = torch.autograd.grad(skip_python_returns[skip_name].reduced_value, background_densities_py)[0]
        torch.testing.assert_close(skip_cuda_grad, skip_python_grad, atol=2e-5, rtol=2e-5)

    def test_joint_backward_nan_logit_on_zero_contribution_point(self):
        """Regression test: the joint two-node backward must not read the
        density logit of a zero-contribution point.

        The fused background_in_track + node_semantic backward kernel used to
        compute ``stable_sigmoid(density_logit) * contribution``
        unconditionally, so a non-finite density logit on a point unselected
        by both members wrote NaN into its own gradient slot
        (sigmoid(NaN) * 0 == NaN). Both logit reads are pinned: the primary
        node's (background point 4, filtered from both members by class) and
        the other node's (road point 0, whose winner class is road and is
        therefore excluded from the penalty set). A non-participating point's
        gradient must be an exact zero and every participating gradient must
        be untouched by the poisoned entries.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda")
        torch.manual_seed(42)

        n_semantic_classes = 33

        class MockGaussiansComposite(GaussiansComposite):
            def __init__(self, *, gaussians_nodes: dict[str, BaseGaussianModel] | None = None):
                # intentionally skip super().__init__()
                torch.nn.Module.__init__(self)
                self.gaussians_nodes = TypedModuleDict[BaseGaussianModel](gaussians_nodes or {})
                self.post_processings = TypedModuleList[BasePostProcessing]([])

            def get_gaussians_node_ids(self, non_empty_only: bool = False) -> list[str]:
                return list(self.gaussians_nodes.keys())

        class MockGaussianNode(BaseGaussianModel):
            def __init__(self, scales, densities, positions, semantic_logits):
                # intentionally skip super().__init__()
                torch.nn.Module.__init__(self)
                self.scales = scales
                self.densities = densities
                self.positions = nn.Parameter(positions, requires_grad=False)
                self.extra_signal = nn.Parameter(semantic_logits, requires_grad=False)
                self.extra_signal_param_infos = ({"semantic_logits": 0}, [semantic_logits.shape[-1]])

            def get_scales(self, preactivation=False) -> torch.Tensor:
                return self.scales if preactivation else torch.exp(self.scales)

            def get_densities(self, preactivation=False):
                return self.densities

        class MockRigidGaussianModel(RigidGaussianModel):
            def __init__(self, positions: torch.Tensor, tracks: CuboidTracks, cuboid_ids: torch.Tensor):
                # intentionally skip super().__init__()
                torch.nn.Module.__init__(self)
                self.positions = torch.nn.Parameter(positions)
                self.cuboid_tracks = tracks
                self.gaussian_cuboid_ids = nn.Buffer(cuboid_ids)

            def get_positions(self) -> torch.nn.Parameter:
                return cast(torch.nn.Parameter, self.positions)

            def get_num_gaussians(self) -> int:
                return self.positions.shape[0]

        # One track whose pose interpolates to the origin at the camera mid
        # timestamp (t=5us): points within +-[0.3, 0.25, 0.2] of the origin
        # are inside the track box.
        identity = np.eye(4, dtype=np.float32)
        track_pose_start = identity.copy()
        track_pose_start[0, 3] = -1.0
        track_pose_end = identity.copy()
        track_pose_end[0, 3] = 1.0
        tracks = CuboidTracks.Factory.from_numpy(
            tracks_id=["track_0"],
            tracks_poses=[np.stack((track_pose_start, track_pose_end))],
            tracks_timestamps_us=[np.array([0, 10], dtype=np.int64)],
            tracks_label_class=["car"],
            tracks_flags=[TrackFlags.NONE],
            cuboids_dims=[np.array([0.6, 0.5, 0.4], dtype=np.float32)],
            device=device,
        )

        # Background node: semantic winners are person(1) / road(32) / an
        # unmapped class(5). With bg-track using ["person"] and node-semantic
        # excluding ["road"] on the background member:
        #   point 0: person, logit below density_logits_min -> no member selects it
        #   point 1: road -> node-semantic
        #   point 2: person, inside the track box -> bg-track
        #   point 3: road -> node-semantic
        #   point 4: class 5 -> filtered out of both members by class alone
        background_positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
            device=device,
        )
        background_density_values = torch.tensor([[-25.0], [0.0], [-1.0], [3.0], [0.5]], device=device)
        background_winners = torch.tensor([1, 32, 1, 32, 5], device=device)
        background_semantic_logits = torch.full((5, n_semantic_classes), -7.0, device=device)
        background_semantic_logits[torch.arange(5, device=device), background_winners] = 7.0

        # Road node: node-semantic uses ["road"], so points whose class is NOT
        # road are penalized -- even indices (class road) contribute nothing.
        n_road = 8
        road_density_values = torch.linspace(-2.0, 2.0, n_road, device=device).unsqueeze(1)
        road_winners = torch.arange(n_road, device=device) % 2
        road_winners[road_winners == 0] = 32
        road_semantic_logits = torch.full((n_road, n_semantic_classes), -7.0, device=device)
        road_semantic_logits[torch.arange(n_road, device=device), road_winners] = 7.0

        config = LossConfig.model_validate(
            {
                "background_in_track_gaussian": {
                    "fn": "bce_with_logits",
                    "lambda_": 0.125,
                    "reduce": {"name": "mean"},
                    "layer_names": ["background"],
                    "layer_labels_to_use": {"background": ["person"]},
                    "density_logits_min": -20.0,
                },
                "node_semantic_gaussians": {
                    "fn": "bce_with_logits",
                    "lambda_": 1.7,
                    "reduce": {"name": "mean"},
                    "layer_labels_to_use": {"road": ["road"]},
                    "layer_labels_to_exclude": {"background": ["road"]},
                },
            }
        )
        trainer_config = MockTrainerConfig()
        aggregator = LossAggregator(config, trainer_config)
        dataset = mock.MagicMock(spec=NCOREDataset)
        datasource = mock.MagicMock()
        datasource.get_semantic_classes_map.return_value = {"person": 1, "road": 32}
        dataset.get_datasource.return_value = datasource
        aggregator.initialize(dataset)
        # Both losses run on the CUDA route only.
        self.assertEqual(aggregator.losses, [])

        timestamps = torch.tensor([[4, 6]], dtype=torch.int64, device=device)
        rendering_camera = RenderingData(
            rays=torch.zeros((1, 1, 1, 6), device=device),
            sensor_model_parameters=[cast(Any, SimpleNamespace())],
            poses_tquat_startend=torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2], device=device),
            timestamps_startend_us=timestamps,
            timestamps_startend_us_cpu=timestamps.cpu(),
        )
        batch = DataAndRenderingBatch(
            data=DataBatch(idx=0, worker_id=None, sequence_id=["dummy"]),
            rendering=RenderingBatch(camera=rendering_camera),
        )
        loss_names = (
            "background_in_track_gaussian_bce_with_logits_mean",
            "node_semantic_gaussians_bce_with_logits_mean",
        )

        def run_joint_backward(
            background_densities: torch.Tensor, road_densities: torch.Tensor
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            model = MockGaussiansComposite(
                gaussians_nodes={
                    "background": MockGaussianNode(
                        torch.zeros((5, 3), device=device),
                        background_densities,
                        background_positions.clone(),
                        background_semantic_logits.clone(),
                    ),
                    "road": MockGaussianNode(
                        torch.zeros((n_road, 3), device=device),
                        road_densities,
                        torch.zeros((n_road, 3), device=device),
                        road_semantic_logits.clone(),
                    ),
                    "rigid": MockRigidGaussianModel(
                        torch.tensor([[0.4, -0.2, 0.15]], device=device),
                        tracks,
                        torch.tensor([0], dtype=torch.long, device=device),
                    ),
                }
            )
            returns = aggregator(step=0, model=model, results=GaussiansCompositeReturn(), target=batch).loss_returns
            for loss_name in loss_names:
                self.assertIn(loss_name, returns)
            objective = torch.stack([returns[loss_name].reduced_value for loss_name in loss_names]).sum()
            reduced = torch.stack([returns[loss_name].reduced_value.detach() for loss_name in loss_names])
            grads = torch.autograd.grad(objective, (background_densities, road_densities))
            return reduced, grads

        background_densities_clean = background_density_values.clone().requires_grad_(True)
        road_densities_clean = road_density_values.clone().requires_grad_(True)
        reduced_clean, (background_grad_clean, road_grad_clean) = run_joint_backward(
            background_densities_clean, road_densities_clean
        )

        # The chosen poison targets contribute to neither member in the clean
        # run, and the joint path is live on both members around them: point 4
        # on the primary (background) node exercises the
        # background_density_logits read, and road point 0 (winner class road,
        # excluded from the penalty set) exercises the
        # other_node_density_logits read.
        poison_idx = 4
        road_poison_idx = 0
        self.assertTrue(torch.isfinite(reduced_clean).all())
        self.assertTrue(torch.isfinite(background_grad_clean).all())
        self.assertTrue(torch.isfinite(road_grad_clean).all())
        self.assertEqual(background_grad_clean[poison_idx].item(), 0.0)
        self.assertEqual(road_grad_clean[road_poison_idx].item(), 0.0)
        self.assertGreater(background_grad_clean[1].abs().item(), 0.0)  # node-semantic member
        self.assertGreater(background_grad_clean[2].abs().item(), 0.0)  # bg-track member
        self.assertGreater(road_grad_clean[1].abs().item(), 0.0)  # other-node segment

        poisoned_density_values = background_density_values.clone()
        poisoned_density_values[poison_idx, 0] = float("nan")
        background_densities_poisoned = poisoned_density_values.requires_grad_(True)
        poisoned_road_values = road_density_values.clone()
        poisoned_road_values[road_poison_idx, 0] = float("nan")
        road_densities_poisoned = poisoned_road_values.requires_grad_(True)
        reduced_poisoned, (background_grad_poisoned, road_grad_poisoned) = run_joint_backward(
            background_densities_poisoned, road_densities_poisoned
        )

        # Before the guarded early-write the kernel computed
        # sigmoid(NaN) * 0 == NaN for the unselected points. The poisoned
        # logits are never read (forward is ternary-guarded, backward
        # early-writes), and per-point gradients are direct writes with a
        # constant upstream, so the gradients must match the clean run
        # bit-for-bit. The reduced scalars go through atomicAdd, whose
        # run-to-run summation order is not guaranteed, so they keep a
        # last-ulp tolerance.
        self.assertTrue(torch.isfinite(background_grad_poisoned).all())
        self.assertTrue(torch.isfinite(road_grad_poisoned).all())
        self.assertEqual(background_grad_poisoned[poison_idx].item(), 0.0)
        self.assertEqual(road_grad_poisoned[road_poison_idx].item(), 0.0)
        torch.testing.assert_close(reduced_poisoned, reduced_clean, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(background_grad_poisoned, background_grad_clean, atol=0.0, rtol=0.0)
        torch.testing.assert_close(road_grad_poisoned, road_grad_clean, atol=0.0, rtol=0.0)

    def test_gaussian_z_scale_skips_disabled_layer(self):
        """Component-only: gaussian_z_scale must no-op when its layer is absent.

        Disabled layers are not built into model.gaussians_nodes. Match the
        non-fused loss path by omitting the unavailable loss instead of
        returning an unwritten fused output.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda")

        class _ZScaleNode(BaseGaussianModel):
            def __init__(self, scales: torch.Tensor):
                """Create a test Gaussian node backed by the supplied scales."""
                torch.nn.Module.__init__(self)
                self._scales = scales

            def get_scales(self, preactivation: bool = False):
                """Return the node's configured scale tensor."""
                return self._scales

        class _ZScaleComposite(GaussiansComposite):
            def __init__(self, gaussians_nodes: dict[str, BaseGaussianModel]):
                """Create a minimal composite from the supplied Gaussian nodes."""
                # intentionally skip super().__init__()
                torch.nn.Module.__init__(self)
                self.gaussians_nodes = TypedModuleDict[BaseGaussianModel](gaussians_nodes)
                self.post_processings = TypedModuleList[BasePostProcessing]([])

            def get_gaussians_node_ids(self, non_empty_only: bool = False) -> list[str]:
                """Return all node IDs configured for the test composite."""
                return list(self.gaussians_nodes.keys())

        config = LossConfig.model_validate(
            {
                "gaussian_z_scale": {
                    "fn": "abs",
                    "lambda_": 1.0,
                    "reduce": {"name": "mean"},
                    "layer_name": "road",
                    "road_z_scale": 0.05,
                },
                # Keep the aggregate non-empty when the unavailable z-scale
                # loss is omitted, as it would be in a real training config.
                "gaussian_scale": {"fn": "abs", "lambda_": 1.0, "reduce": {"name": "mean"}},
            }
        )
        trainer_config = MockTrainerConfig()
        batch = DataAndRenderingBatch(
            data=DataBatch(idx=0, worker_id=None, sequence_id=["dummy"], camera=None, lidar=None),
            rendering=None,
        )
        results = GaussiansCompositeReturn()

        # Layer present: the loss computes on the real scales.
        scales = torch.full((4, 3), 0.5, device=device, requires_grad=True)
        model_with_road = _ZScaleComposite({"road": _ZScaleNode(scales)})
        aggregator = LossAggregator(config, trainer_config)
        ret = aggregator(step=0, model=model_with_road, results=results, target=batch)
        self.assertIn("gaussian_z_scale_abs_mean", ret.loss_returns)
        self.assertGreater(ret.loss_returns["gaussian_z_scale_abs_mean"].reduced_value.item(), 0.0)

        # Layer disabled (component-only): the loss is omitted.
        model_without_road = _ZScaleComposite({"background": _ZScaleNode(scales.detach().clone())})
        aggregator = LossAggregator(config, trainer_config)
        ret = aggregator(step=0, model=model_without_road, results=results, target=batch)
        self.assertNotIn("gaussian_z_scale_abs_mean", ret.loss_returns)

    def test_sky_env_map_loss_skips_disabled_sky_model(self):
        """Component-only: sky_env_map_background must no-op when the sky
        model is disabled (a SkipBackground stand-in replaces it — there
        are no env-map textures to regularize)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        class _ScalesNode(BaseGaussianModel):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self._scales = torch.full((4, 3), 0.5, device=torch.device("cuda"))

            def get_scales(self, preactivation: bool = False):
                return self._scales

        class _SkyComposite(GaussiansComposite):
            def __init__(self, background: torch.nn.Module):
                # intentionally skip super().__init__()
                torch.nn.Module.__init__(self)
                self.gaussians_nodes = TypedModuleDict[BaseGaussianModel]({"background": _ScalesNode()})
                self.post_processings = TypedModuleList[BasePostProcessing]([])
                self.background = cast(BaseBackground, background)

            def get_gaussians_node_ids(self, non_empty_only: bool = False) -> list[str]:
                return list(self.gaussians_nodes.keys())

        # A second, always-on loss keeps the aggregator's return
        # non-empty when the sky loss is skipped.
        config = LossConfig.model_validate(
            {
                "sky_env_map_background": {
                    "fn": "total_variation_spatial",
                    "lambda_": 1.0,
                    "reduce": {"name": "mean"},
                },
                "gaussian_scale": {"fn": "abs", "lambda_": 1.0, "reduce": {"name": "mean"}},
            }
        )
        trainer_config = MockTrainerConfig()
        batch = DataAndRenderingBatch(
            data=DataBatch(idx=0, worker_id=None, sequence_id=["dummy"], camera=None, lidar=None),
            rendering=None,
        )
        results = GaussiansCompositeReturn()

        # Sky model present: the TV loss runs on its textures.
        bg_cfg = SkyEnvMapBackgroundConfig.model_validate(
            {
                "name": "sky-env-map",
                "envmap_type": "equirectangular",
                "height": 8,
                "width": 16,
                "saturate_radiance": False,
                "should_inpaint": False,
                "inpaint_threshold": 0.0,
                "inpaint_kernel_size": 5,
                "min_grad_updates": 0,
                "composite_in_linear_space": False,
            }
        )
        model_with_sky = _SkyComposite(SkyEnvMapBackground(bg_cfg, trainer_config).cuda())
        aggregator = LossAggregator(config, trainer_config)
        ret = aggregator(step=0, model=model_with_sky, results=results, target=batch)
        self.assertIn("sky_env_map_background_total_variation_spatial_mean", ret.loss_returns)

        # Sky model disabled: the loss is not enabled at all — no crash,
        # no loss entry.
        model_without_sky = _SkyComposite(torch.nn.Linear(2, 2).cuda())
        aggregator = LossAggregator(config, trainer_config)
        ret = aggregator(step=0, model=model_without_sky, results=results, target=batch)
        self.assertNotIn("sky_env_map_background_total_variation_spatial_mean", ret.loss_returns)

    def test_gaussian_losses_construct_without_layer_lambdas_cpu(self):
        """Regression test for the layer_lambdas default at the loss-class
        level: make_loss must construct GaussianScaleLoss/GaussianDensityLoss
        when the config omits layer_lambdas (it defaults to None), and this
        needs no GPU, so CPU-only runners cover the fix. The aggregator-level
        test below stays CUDA-gated because LossAggregator always instantiates
        ModuleLosses, whose dummy tensors live on the CUDA device.
        """
        config = LossConfig.model_validate(
            {
                "gaussian_scale": {"fn": "abs", "lambda_": 0.01, "reduce": {"name": "mean"}},
                "gaussian_density": {"fn": "abs", "lambda_": 0.01, "reduce": {"name": "mean"}},
            }
        )
        trainer_config = MockTrainerConfig()
        losses = {name: make_loss(str(name), item, trainer_config) for name, item in config.items()}
        self.assertEqual(losses["gaussian_scale"].name, "gaussian_scale_abs_mean")
        self.assertEqual(losses["gaussian_density"].name, "gaussian_density_abs_mean")
        for loss in losses.values():
            self.assertEqual(loss.layer_lambdas, {})

    def test_gaussian_losses_construct_without_layer_lambdas_cuda(self):
        """Regression test: gaussian_scale/gaussian_density configs without layer_lambdas must construct.

        LossItemConfig.layer_lambdas defaults to None, so once the CUDA-enabled
        aggregator path constructs the real loss classes via make_loss, an
        unpack_optional(config.layer_lambdas) in their __init__ would raise
        ValueError for configs that omit layer_lambdas (e.g.
        configs/experimental/3dgut/3dgut_colmap.yaml only sets lambda_).
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        config = LossConfig.model_validate(
            {
                "gaussian_scale": {"fn": "abs", "lambda_": 0.01, "reduce": {"name": "mean"}},
                "gaussian_density": {"fn": "abs", "lambda_": 0.01, "reduce": {"name": "mean"}},
            }
        )
        trainer_config = MockTrainerConfig()

        # CUDA path: both losses route to the fused CUDA module and are built via make_loss.
        aggregator_cuda = LossAggregator(config, trainer_config)
        self.assertEqual(aggregator_cuda.losses, [])
        self.assertEqual(len(aggregator_cuda.in_cuda.losses), 2)
        self.assertEqual(
            sorted(loss.name for loss in aggregator_cuda.in_cuda.losses),
            ["gaussian_density_abs_mean", "gaussian_scale_abs_mean"],
        )
        for loss in aggregator_cuda.in_cuda.losses:
            self.assertEqual(loss.layer_lambdas, {})

        # Python path: the same real classes are constructed directly.
        aggregator_py = LossAggregator(config, trainer_config, force_disable_cuda=True)
        self.assertEqual(aggregator_py.in_cuda.losses, [])
        self.assertEqual(len(aggregator_py.losses), 2)
        for loss in aggregator_py.losses:
            self.assertEqual(loss.layer_lambdas, {})

    def test_all_cuda_available_losses_construct_via_make_loss(self):
        """Every CUDA-eligible loss constructs through make_loss from a minimal config.

        The CUDA aggregator path runs the real __init__ of every loss in
        ModuleLosses.available, so a strictly-required optional field
        (unpack_optional on a defaulted-None config entry) breaks CUDA-eligible
        configs at construction time; the layer_lambdas regression above is the
        precedent. This needs no GPU. When adding a loss to
        ModuleLosses.available, add its minimal config here (enforced by
        test_minimal_cuda_loss_configs_cover_available).
        """
        config = LossConfig.model_validate(MINIMAL_CUDA_LOSS_CONFIGS)
        trainer_config = MockTrainerConfig()
        for name, item in config.items():
            with self.subTest(loss=str(name)):
                loss = make_loss(str(name), item, trainer_config)
                self.assertEqual(loss.name, f"{name}_{item.fn}_{item.reduce.name}")

    def test_minimal_cuda_loss_configs_cover_available(self):
        """MINIMAL_CUDA_LOSS_CONFIGS stays lock-step with ModuleLosses.available.

        available is a class attribute, so this lock-step guard runs on CPU-only
        runners too (instantiation would allocate CUDA dummy tensors).
        """
        from nurec.libs.losses.models.losses_module import ModuleLosses

        composed = sorted(
            f"{name}_{item['fn']}_{item['reduce']['name']}" for name, item in MINIMAL_CUDA_LOSS_CONFIGS.items()
        )
        self.assertEqual(composed, sorted(ModuleLosses.available))


def test_gaussian_z_scale_empty_layer_uses_dummy_inputs(monkeypatch):
    """Ensure empty Gaussian z-scale layers pass dummy tensors into the fused loss.

    This protects CUDA loss invocation when a configured layer has zero Gaussians: the loss
    should remain present with a zero factor instead of returning an unwritten dummy output.
    """
    if not torch.cuda.is_available():
        pytest.skip("ModuleLosses CUDA dummy tensors require CUDA")

    device = torch.device("cuda")
    losses = ModuleLosses()
    losses.losses = [
        SimpleNamespace(
            name="gaussian_z_scale_abs_mean",
            lambda_=1.0,
            layer_name="road",
            road_z_scale=0.01,
            should_run_fn=lambda step: True,
        )
    ]

    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.post_processings = []
    model.gaussians_nodes = {
        "road": SimpleNamespace(get_scales=lambda preactivation=True: torch.empty((0, 3), device=device))
    }
    model.get_gaussians_node_ids = lambda: ["road"]

    captured: dict[str, object] = {}

    # Bind the intercepted call against CudaLossesFunction.forward's signature and read the
    # captured tensors/scalars by name. The dispatch entry takes no ctx, so the
    # arguments bind directly. This is robust to signature reordering — unlike hardcoded positional indices, which silently
    # captured the wrong arguments if forward's parameter list changed.
    forward_signature = inspect.signature(CudaLossesFunction.forward)

    def fake_apply(*args):
        """Capture z-scale arguments and return deterministic fused outputs."""
        bound = forward_signature.bind(*args)
        captured["gaussian_z_scales"] = bound.arguments["gaussian_z_scales"]
        captured["z_scale_threshold"] = bound.arguments["z_scale_threshold"]
        captured["z_scale_factor"] = bound.arguments["z_scale_factor"]
        return tuple(torch.zeros((), device=device) for _ in losses.available)

    monkeypatch.setattr(CudaLossesFunction, "forward", fake_apply)

    ret = losses(
        step=0,
        model=model,
        results=GaussiansCompositeReturn(),
        target=SimpleNamespace(data=SimpleNamespace(camera=None, lidar=None)),
    )

    assert "gaussian_z_scale_abs_mean" in ret
    assert captured["gaussian_z_scales"] is losses.dummy_z_scales
    assert captured["z_scale_threshold"] == 0.01
    assert captured["z_scale_factor"] == 0.0


@pytest.mark.parametrize("include_empty_node", [False, True])
def test_empty_gaussian_regularizers_use_zero_factors(monkeypatch, include_empty_node):
    """Active scale/density losses deterministically return zero without eligible points."""
    if not torch.cuda.is_available():
        pytest.skip("ModuleLosses CUDA dummy tensors require CUDA")

    device = torch.device("cuda")

    class _EmptyGaussianNode(BaseGaussianModel):
        def __init__(self):
            """Create a Gaussian node with empty scale and density tensors."""
            torch.nn.Module.__init__(self)
            self._scales = torch.empty((0, 3), device=device)
            self._densities = torch.empty((0, 1), device=device)

        def get_scales(self, preactivation: bool = False):
            """Return the node's empty scale tensor."""
            return self._scales

        def get_densities(self, preactivation: bool = False):
            """Return the node's empty density tensor."""
            return self._densities

    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.post_processings = TypedModuleList[BasePostProcessing]([])
    nodes = {"empty": _EmptyGaussianNode()} if include_empty_node else {}
    model.gaussians_nodes = TypedModuleDict[BaseGaussianModel](nodes)
    model.get_gaussians_node_ids = lambda: list(model.gaussians_nodes.keys())

    losses = ModuleLosses()
    losses.losses = [
        SimpleNamespace(
            name="gaussian_scale_abs_mean",
            lambda_=1.0,
            layer_lambdas={},
            visibility_filter=False,
            should_run_fn=lambda step: True,
        ),
        SimpleNamespace(
            name="gaussian_density_abs_mean",
            lambda_=1.0,
            layer_lambdas={},
            visibility_filter=False,
            should_run_fn=lambda step: True,
        ),
    ]

    captured: dict[str, float] = {}
    forward_signature = inspect.signature(CudaLossesFunction.forward)

    def fake_apply(*args):
        """Capture regularizer factors and return deterministic fused outputs."""
        bound = forward_signature.bind(*args)
        captured["scale_factor"] = bound.arguments["scale_factor"]
        captured["density_factor"] = bound.arguments["density_factor"]
        return tuple(torch.zeros((), device=device) for _ in losses.available)

    monkeypatch.setattr(CudaLossesFunction, "forward", fake_apply)

    ret = losses(
        step=0,
        model=model,
        results=GaussiansCompositeReturn(),
        target=SimpleNamespace(data=SimpleNamespace(camera=None, lidar=None)),
    )

    assert captured == {"scale_factor": 0.0, "density_factor": 0.0}
    assert ret["gaussian_scale_abs_mean"].reduced_value.item() == 0.0
    assert ret["gaussian_density_abs_mean"].reduced_value.item() == 0.0


def test_fused_camera_losses_require_rendered_camera():
    """Configured camera losses fail before an unwritten fused output can escape."""
    if not torch.cuda.is_available():
        pytest.skip("ModuleLosses CUDA dummy tensors require CUDA")

    device = torch.device("cuda")
    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.post_processings = TypedModuleList[BasePostProcessing]([])
    model.gaussians_nodes = TypedModuleDict[BaseGaussianModel]({})
    model.get_gaussians_node_ids = lambda: []

    losses = ModuleLosses()
    losses.losses = [
        SimpleNamespace(
            name="background_mse_mean",
            lambda_=1.0,
            should_run_fn=lambda step: True,
        )
    ]
    target = SimpleNamespace(
        data=SimpleNamespace(
            camera=SimpleNamespace(labels=SimpleNamespace(flags=torch.zeros((1, 1, 1, 1), device=device))),
            lidar=None,
        )
    )

    with pytest.raises(ValueError, match="require results.rendered_cam"):
        losses(
            step=0,
            model=model,
            results=GaussiansCompositeReturn(),
            target=target,
        )
