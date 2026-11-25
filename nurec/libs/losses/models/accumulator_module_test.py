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

from gsplat.losses.functional.accumulator import (
    GaussianLossAccumulator,
)
from gsplat.losses.functional.fused import (
    FusedBgGridLosses,
    FusedCameraLosses,
    FusedGaussianLosses,
    FusedLidarLosses,
)

from nurec.config.model import SkyEnvMapBackgroundConfig
from nurec.config.trainer import TrainerConfig
from nurec.datasets.ncore import NCOREDataset
from nurec.datasets.tracks import CuboidTracks
from nurec.libs.losses.functional.cuda_losses_accumulator import (
    ACCUMULATOR_LOSS_NAMES,
    GRID_NUM_CHANNELS,
    LOSS_SLOT_BY_NAME,
    GsplatLossAdapter,
)
from nurec.libs.losses.models.accumulator_module import AccumulatorModuleLosses
from nurec.libs.losses.models.fused_loss_inputs import _prepare_loss_tracks
from nurec.libs.losses.models.registry import make_loss
from nurec.libs.losses.orchestration.config import (
    LossConfig,
    LossItemConfig,
)
from nurec.libs.losses.orchestration.loss_accumulator import LossAccumulator as _ProductionLossAccumulator
from nurec.libs.losses.orchestration.loss_aggregator import LossAggregator
from nurec.models.background import SkyEnvMapBackground
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import BaseGaussianModel, RigidGaussianModel
from nurec.models.nn_extensions import TypedModuleDict, TypedModuleList
from nurec.models.post_processing import (
    BasePostProcessing,
    BilateralGridPerCamera,
    BilateralGridPerFrame,
)
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


def _unpack_weighted_loss_values(vector):
    if vector.shape != (len(ACCUMULATOR_LOSS_NAMES),):
        raise ValueError("weighted loss vector must have shape matching the slots")
    return vector.unbind(0)


class _MapAssertions(dict):
    """Test-only named views of the real new return; no producer is mocked."""

    def __init__(self, vector, weights):
        values = vector.unbind(0)
        super().__init__(
            {
                name: SimpleNamespace(
                    name=name, lambda_=weight, value=None, reduce_fn=None, reduced_value=values[LOSS_SLOT_BY_NAME[name]]
                )
                for name, weight in weights.items()
            }
        )
        self.weighted_loss_vector = vector


class ModuleLosses(AccumulatorModuleLosses):
    def forward(self, *args, **kwargs):
        return _MapAssertions(*self.accumulate(*args, **kwargs))


class LossAccumulator(_ProductionLossAccumulator):
    """Run the new public entry while retaining existing parity assertion helpers."""

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        values = result.weighted_values.unbind(0)
        result.loss_returns = {
            name: SimpleNamespace(
                name=name, lambda_=weight, value=None, reduce_fn=None, reduced_value=values[result._slots_by_name[name]]
            )
            for name, weight in result.weights_by_name.items()
        }
        return result


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
    "bilateral_grid_drift": {
        "fn": "identity_distance",
        "lambda_": 1.0,
        "reduce": {"name": "mean"},
    },
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
    "gaussian_scale": {"fn": "abs", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "sky_env_map_background": {
        "fn": "total_variation_spatial",
        "lambda_": 1.0,
        "reduce": {"name": "mean"},
    },
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
    "node_semantic_gaussians": {
        "fn": "bce_with_logits",
        "lambda_": 1.0,
        "reduce": {"name": "mean"},
    },
    "semantic": {"fn": "cross_entropy", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "deform_smoothness": {"fn": "abs", "lambda_": 1.0, "reduce": {"name": "mean"}},
    "road_gaussians": {
        "fn": "abs",
        "lambda_": 1.0,
        "reduce": {"name": "mean"},
        "layer_name": "road",
        "n_samples": 8,
        "grid_len": 3.0,
        "min": 0.0,
        "range": 20.0,
        "rotation_lambda": 0.1,
    },
}


def test_weighted_vector_readout_preserves_slot_values():
    """The model exposes every accumulator slot in its declared order."""
    source = torch.linspace(-2.0, 3.0, len(ACCUMULATOR_LOSS_NAMES), dtype=torch.float64, requires_grad=True)
    losses_vec = source.square()
    weighted_values = _unpack_weighted_loss_values(losses_vec)

    torch.testing.assert_close(torch.stack(weighted_values), losses_vec)


@pytest.mark.parametrize("shape", [(), (len(ACCUMULATOR_LOSS_NAMES) - 1,), (1, len(ACCUMULATOR_LOSS_NAMES))])
def test_weighted_vector_readout_rejects_wrong_shape(shape):
    with pytest.raises(ValueError, match="weighted loss vector must have shape"):
        _unpack_weighted_loss_values(torch.zeros(shape))


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


class TestLossAccumulatorRouting(unittest.TestCase):
    def _ssim_config(self, *, semantic_weights: bool = False, window_size: int = 11) -> LossConfig:
        ssim_cfg: dict[str, object] = {
            "fn": "ssim",
            "lambda_": 1.0,
            "reduce": {"name": "mean"},
            "window_size": window_size,
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

        aggregator = LossAccumulator(self._ssim_config(), MockTrainerConfig())

        self.assertEqual([loss.name for loss in aggregator.in_accumulator.losses], ["ssim_ssim_mean"])
        self.assertEqual(aggregator.accumulator_losses, [])

    def test_semantic_weighted_ssim_stays_on_legacy_path(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        aggregator = LossAccumulator(self._ssim_config(semantic_weights=True), MockTrainerConfig())

        self.assertEqual(aggregator.in_accumulator.losses, [])
        self.assertEqual([loss.name for loss in aggregator.accumulator_losses], ["ssim_ssim_mean"])

    def test_nonstandard_window_ssim_stays_on_legacy_path(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        aggregator = LossAccumulator(self._ssim_config(window_size=7), MockTrainerConfig())

        self.assertEqual(aggregator.in_accumulator.losses, [])
        self.assertEqual([loss.name for loss in aggregator.accumulator_losses], ["ssim_ssim_mean"])


_SIMPLE_CUDA_LOSS_NAMES = (
    "rgb_l1_mean",
    "lidar_l1_mean",
    "background_mse_mean",
    "background_lidar_mse_mean",
    "bilateral_grid_drift_identity_distance_mean",
    "bilateral_grid_per_camera_tv_total_variation_spatial_mean",
    "bilateral_grid_per_frame_spatial_tv_total_variation_spatial_mean",
    "gaussian_scale_abs_mean",
    "sky_env_map_background_total_variation_spatial_mean",
    "gaussian_density_abs_mean",
    "out_of_bound_l1_mean",
    "gaussian_z_scale_abs_mean",
    "intensity_mse_mean",
    "raydrop_mse_mean",
)
_SIMPLE_CUDA_PREDICTION_NAMES = (
    "rgb_pred",
    "lidar_pred",
    "intensity_pred",
    "raydrop_pred",
    "bg_pred",
    "bg_lidar_pred",
    "grid_per_camera",
    "grid_per_frame",
    "gaussian_scales",
    "bg_tex",
    "gaussian_densities",
    "out_of_bound_positions",
    "gaussian_z_scales",
)


def _simple_cuda_loss_inputs(envmap_type: str, *, shared_visibility: bool = False) -> dict[str, torch.Tensor]:
    """Keep main's seeded 2x2 fixture and heterogeneous Gaussian member domains."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    rgb_pred = torch.randn(1, 2, 2, 3, device=device, requires_grad=True)
    rgb_gt = torch.randn(1, 2, 2, 3, device=device)
    rgb_flags = torch.full((1, 2, 2, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32, device=device)
    lidar_pred = torch.randn(1, 2, 2, 1, device=device, requires_grad=True)
    lidar_gt = torch.randn(1, 2, 2, 1, device=device)
    lidar_flags = torch.zeros((1, 2, 2, 1), dtype=torch.int32, device=device)
    rgb_flags.view(-1)[0] |= RayFlags.SKY_SEMANTIC.value
    bg_pred = torch.rand(4, device=device, requires_grad=True)
    lidar_flags.view(-1)[0] |= RayFlags.SKY_SEMANTIC.value
    bg_lidar_pred = torch.rand(4, device=device, requires_grad=True)
    grid_per_camera = torch.randn(
        (1, GRID_NUM_CHANNELS, 2, 2, 2),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    grid_per_frame = torch.randn(
        (3, GRID_NUM_CHANNELS, 2, 2, 2),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    gaussian_scales = (torch.randn(10, 3, device=device) - 3.0).requires_grad_(True)
    if envmap_type == "cubemap":
        bg_tex = torch.randn((1, 6, 4, 4, 3), device=device, dtype=torch.float32, requires_grad=True)
    elif envmap_type == "equirectangular":
        bg_tex = torch.randn((1, 4, 4, 3), device=device, dtype=torch.float32, requires_grad=True)
    else:
        raise ValueError(f"Unsupported envmap_type: {envmap_type}")
    gaussian_densities = torch.randn(10 if shared_visibility else 15, device=device, requires_grad=True)
    out_of_bound_positions = torch.tensor([[0.2, -0.1, 0.25], [0.1, 0.2, 0.3]], device=device, requires_grad=True)
    out_of_bound_cuboid_dims = torch.tensor([[0.3, 0.25, 0.4], [0.2, 0.15, 0.3]], device=device)
    gaussian_z_scales = (torch.randn(8192, 3, device=device) - 3.0).requires_grad_(True)
    intensity_pred = torch.randn(1, 2, 2, 1, device=device, requires_grad=True)
    intensity_gt = torch.randn(1, 2, 2, 1, device=device)
    raydrop_pred = torch.randn(1, 2, 2, 1, device=device, requires_grad=True)
    raydrop_gt = torch.randn(1, 2, 2, 1, device=device)
    inputs = {
        "rgb_pred": rgb_pred,
        "rgb_gt": rgb_gt,
        "rgb_flags": rgb_flags,
        "lidar_pred": lidar_pred,
        "lidar_gt": lidar_gt,
        "lidar_flags": lidar_flags,
        "bg_pred": bg_pred,
        "bg_lidar_pred": bg_lidar_pred,
        "grid_per_camera": grid_per_camera,
        "grid_per_frame": grid_per_frame,
        "gaussian_scales": gaussian_scales,
        "bg_tex": bg_tex,
        "gaussian_densities": gaussian_densities,
        "out_of_bound_positions": out_of_bound_positions,
        "out_of_bound_cuboid_dims": out_of_bound_cuboid_dims,
        "gaussian_z_scales": gaussian_z_scales,
        "intensity_pred": intensity_pred,
        "intensity_gt": intensity_gt,
        "raydrop_pred": raydrop_pred,
        "raydrop_gt": raydrop_gt,
    }
    if shared_visibility:
        inputs["gaussian_visibility"] = torch.tensor([0.0, 1.0, 0.5, 1.0, 0.0, 1.0, 0.25, 1.0, 0.0, 1.0], device=device)
    return inputs


def _simple_cuda_map_losses(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compare public gsplat maps reduced with the production mean factors."""
    rgb_flags, lidar_flags = inputs["rgb_flags"].reshape(-1), inputs["lidar_flags"].reshape(-1)
    rgb_valid = ((rgb_flags & RayFlags.RGB_LABEL.value) != 0) & ((rgb_flags & RayFlags.INVALID.value) == 0)
    bg_valid = (rgb_flags & (RayFlags.INVALID.value | RayFlags.HARMONIZED.value | RayFlags.SYNTHETIC.value)) == 0
    lidar_valid = (lidar_flags & (RayFlags.INVALID.value | RayFlags.DROPPED.value)) == 0
    rgb_factor = 1.0 / (3 * rgb_valid.sum().item())
    lidar_factor = 1.0 / lidar_valid.sum().item()
    rgb_loss, bg_loss = FusedCameraLosses()(
        rgb_flags,
        inputs["rgb_pred"].reshape(-1, 3),
        inputs["rgb_gt"].reshape(-1, 3),
        inputs["bg_pred"],
        rgb_factor=rgb_factor,
        bg_factor=1.0 / bg_valid.sum().item(),
    )
    lidar_loss, intensity_loss, raydrop_loss, bg_lidar_loss = FusedLidarLosses()(
        lidar_flags,
        inputs["lidar_pred"].reshape(-1),
        inputs["lidar_gt"].reshape(-1),
        inputs["intensity_pred"].reshape(-1),
        inputs["intensity_gt"].reshape(-1),
        inputs["raydrop_pred"].reshape(-1),
        inputs["raydrop_gt"].reshape(-1),
        inputs["bg_lidar_pred"],
        distance_factor=lidar_factor,
        intensity_factor=lidar_factor,
        raydrop_factor=lidar_factor,
        bg_factor=lidar_factor,
    )
    scale_loss, density_loss, z_scale_loss, out_of_bound_loss = FusedGaussianLosses(
        z_scale_threshold=0.01,
        preactivation=True,
    )(
        inputs["gaussian_scales"],
        inputs["gaussian_densities"],
        inputs["gaussian_z_scales"],
        inputs["out_of_bound_positions"],
        inputs["out_of_bound_cuboid_dims"],
        # Main's all-ones mask was neutral. Omitting it preserves the original
        # 10/15 independent domains under the shared-visibility contract.
        visibility=inputs.get("gaussian_visibility"),
        scale_factor=1.0 / inputs["gaussian_scales"].numel(),
        density_factor=1.0 / inputs["gaussian_densities"].numel(),
        z_scale_factor=1.0 / inputs["gaussian_z_scales"].shape[0],
        oob_factor=1.0 / inputs["out_of_bound_positions"].numel(),
    )
    bg_tex = inputs["bg_tex"]
    bg_tex_loss, drift_loss, camera_tv_loss, frame_tv_loss = FusedBgGridLosses(normalize_cells=True)(
        bg_tex.flatten(0, 1) if bg_tex.ndim == 5 else bg_tex,
        inputs["grid_per_camera"].flatten(0, 1),
        inputs["grid_per_frame"].flatten(0, 1),
        grid_drift_camera_factor=1.0,
        grid_drift_frame_factor=1.0,
        grid_camera_tv_factor=1.0,
        grid_frame_tv_factor=1.0,
        bg_tex_depth=6 if bg_tex.ndim == 5 else 1,
    )
    maps = (
        rgb_loss,
        lidar_loss,
        bg_loss,
        bg_lidar_loss,
        drift_loss,
        camera_tv_loss,
        frame_tv_loss,
        scale_loss,
        bg_tex_loss,
        density_loss,
        out_of_bound_loss,
        z_scale_loss,
        intensity_loss,
        raydrop_loss,
    )
    return {name: value.sum() for name, value in zip(_SIMPLE_CUDA_LOSS_NAMES, maps, strict=True)}


def _simple_cuda_accumulator_losses(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run the fourteen-member fixture through the model's owned family modules."""
    losses = ModuleLosses()
    selected = {
        name: (
            SimpleNamespace(
                name=name,
                layer_lambdas={},
                visibility_filter="gaussian_visibility" in inputs,
                occlusion_aware=False,
                layer_name="road",
                road_z_scale=0.01,
            ),
            1.0,
        )
        for name in _SIMPLE_CUDA_LOSS_NAMES
    }
    target: Any = SimpleNamespace(
        data=SimpleNamespace(
            camera=SimpleNamespace(labels=SimpleNamespace(flags=inputs["rgb_flags"], rgb=inputs["rgb_gt"])),
            lidar=SimpleNamespace(
                labels=SimpleNamespace(
                    flags=inputs["lidar_flags"],
                    distance=inputs["lidar_gt"],
                    intensity=inputs["intensity_gt"],
                    raydrop=inputs["raydrop_gt"],
                )
            ),
        )
    )
    results: Any = SimpleNamespace(
        rendered_cam=SimpleNamespace(rgb=inputs["rgb_pred"], opacity=inputs["bg_pred"]),
        rendered_lidar=SimpleNamespace(
            distance=inputs["lidar_pred"],
            opacity=inputs["bg_lidar_pred"],
            extra_ray_signals=SimpleNamespace(intensity=inputs["intensity_pred"], raydrop=inputs["raydrop_pred"]),
        ),
    )
    model: Any = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_buffer("_device_indicator", inputs["rgb_pred"].new_zeros(()), persistent=False)
    model.gaussians_nodes = {
        "regularizer": SimpleNamespace(
            get_scales=lambda preactivation=True: inputs["gaussian_scales"],
            get_densities=lambda preactivation=False: inputs["gaussian_densities"],
        ),
        "road": SimpleNamespace(get_scales=lambda preactivation=True: inputs["gaussian_z_scales"]),
    }
    # Preserve the original independent 10/15-point regularizer domains and
    # the separately selected 8192-point road layer.
    model.get_gaussians_node_ids = lambda: ["regularizer"]
    background: Any = object.__new__(SkyEnvMapBackground)
    torch.nn.Module.__init__(background)
    background.textures = inputs["bg_tex"]
    model.background = background
    oob_node: Any = SimpleNamespace(
        get_positions=lambda: inputs["out_of_bound_positions"],
        cuboid_tracks=SimpleNamespace(cuboids_dims=inputs["out_of_bound_cuboid_dims"]),
        gaussian_cuboid_ids=torch.arange(
            inputs["out_of_bound_positions"].shape[0],
            device=model.device,
            dtype=torch.int32,
        ),
    )
    camera_grids: Any = [SimpleNamespace(bilateral_grid=SimpleNamespace(grid=inputs["grid_per_camera"]))]
    frame_grids: Any = [SimpleNamespace(bilateral_grid=SimpleNamespace(grid=inputs["grid_per_frame"]))]
    vector = inputs["rgb_pred"].new_zeros(len(ACCUMULATOR_LOSS_NAMES))
    vector = losses._accumulate_camera(vector, results, target, selected)
    vector = losses._accumulate_lidar(vector, results, target, selected)
    with (
        mock.patch(
            "nurec.libs.losses.models.accumulator_module.get_rendered_visibility_mask",
            return_value=inputs.get("gaussian_visibility"),
        ),
        mock.patch("nurec.libs.losses.models.accumulator_module._maybe_update_mcmc_visibility_counters"),
    ):
        vector = losses._accumulate_gaussian(vector, model, results, [oob_node], selected)
    return losses._accumulate_bg_grid(vector, model, camera_grids, frame_grids, selected)


class TestCudaLosses(unittest.TestCase):
    def test_ssim_constant_mask_no_valid_pixels_is_finite_zero_with_zero_gradient(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")
        flags = torch.full((1, 4, 4, 1), RayFlags.INVALID.value, dtype=torch.int32, device=device)
        rgb_gt = torch.randn((1, 4, 4, 3), device=device)
        rgb_pred = torch.randn((1, 4, 4, 3), device=device, requires_grad=True)
        labels = CameraFrameLabels(flags=flags, rgb=rgb_gt, _n_valid=0)
        camera = DataBatch.Camera(
            meta=[FrameMeta(unique_sensor_idx=0, unique_frame_idx=0, training_frame_idx=0)],
            labels=labels,
        )
        target = DataAndRenderingBatch(
            data=DataBatch(idx=0, worker_id=None, sequence_id=["dummy"], camera=camera, lidar=None),
            rendering=None,
        )
        results = GaussiansCompositeReturn(
            rendered_cam=GaussiansRenderReturn(
                rgb=rgb_pred.reshape(-1, 3),
                opacity=torch.zeros(16, device=device),
                distance=torch.zeros(16, device=device),
            )
        )
        model = object.__new__(GaussiansComposite)
        torch.nn.Module.__init__(model)
        model.register_buffer("_device_indicator", torch.zeros((), device="cuda"), persistent=False)
        model.post_processings = TypedModuleList[BasePostProcessing]([])
        model.gaussians_nodes = TypedModuleDict[BaseGaussianModel]({})
        model.get_gaussians_node_ids = lambda: []

        losses = ModuleLosses()
        losses.losses = [
            make_loss(
                "ssim",
                LossItemConfig.model_validate(
                    {
                        "fn": "ssim",
                        "lambda_": 2.0,
                        "reduce": {"name": "mean"},
                        "window_size": 11,
                        "channel": 3,
                        "mask": {"mode": "constant", "value": 0.25},
                    }
                ),
                MockTrainerConfig(),
            )
        ]

        with mock.patch.object(
            losses.accumulator,
            "_get_ssim_loss_accumulator",
            wraps=losses.accumulator._get_ssim_loss_accumulator,
        ) as get_ssim_loss_accumulator:
            module_return = losses(0, model, results, target)
            loss_return = module_return["ssim_ssim_mean"]

        get_ssim_loss_accumulator.assert_called_once_with(11, "constant", 0.25)

        self.assertIsNone(loss_return.value)
        self.assertTrue(torch.isfinite(loss_return.reduced_value))
        self.assertEqual(loss_return.reduced_value.item(), 0.0)
        (5.0 * loss_return.reduced_value).backward()
        self.assertIsNotNone(rgb_pred.grad)
        self.assertEqual(rgb_pred.grad.abs().max().item(), 0.0)

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

        def run_accumulator(targets):
            logits_leaf = logits.detach().clone().requires_grad_(True)
            semantic_slot = LOSS_SLOT_BY_NAME["semantic_cross_entropy_mean"]
            module = ModuleLosses()
            results = SimpleNamespace(
                rendered_cam=SimpleNamespace(
                    extra_ray_signals=SimpleNamespace(semantic_logits=logits_leaf),
                )
            )
            target = SimpleNamespace(
                data=SimpleNamespace(
                    camera=SimpleNamespace(
                        labels=SimpleNamespace(flags=flags, semantic=targets),
                    )
                )
            )
            losses_vec = module._accumulate_camera(
                logits_leaf.new_zeros(len(ACCUMULATOR_LOSS_NAMES)),
                results,
                target,
                {"semantic_cross_entropy_mean": (SimpleNamespace(), 2.0)},
            )
            weighted = losses_vec[semantic_slot]
            weighted.backward()
            assert logits_leaf.grad is not None
            return weighted.detach(), logits_leaf.grad.clone()

        def run(targets):
            weighted, gradient = run_accumulator(targets)
            reference_logits = logits.detach().clone().requires_grad_(True)
            valid = (flags.reshape(-1) & RayFlags.SKY_SEMANTIC.value) == 0
            per_row = torch.nn.functional.cross_entropy(reference_logits, targets.long(), reduction="none")
            reference = 2.0 * (per_row * valid).sum() / valid.sum()
            reference.backward()
            torch.testing.assert_close(weighted, reference.detach(), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(gradient, reference_logits.grad, rtol=1e-5, atol=1e-6)
            return weighted / 2.0, weighted, gradient

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

    def test_accumulator_matches_public_maps_cubemap(self):
        self._run_accumulator_map_comparison("cubemap")

    def test_accumulator_matches_public_maps_equirectangular(self):
        self._run_accumulator_map_comparison("equirectangular")

    def _run_accumulator_map_comparison(self, envmap_type):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        for shared_visibility in (False, True):
            with self.subTest(shared_visibility=shared_visibility):
                map_inputs = _simple_cuda_loss_inputs(envmap_type, shared_visibility=shared_visibility)
                accum_inputs = {
                    name: value.detach().clone().requires_grad_(value.requires_grad)
                    for name, value in map_inputs.items()
                }
                # Production RGB means include channels. Grid drift averages
                # the combined camera/frame cells.
                reference = _simple_cuda_map_losses(map_inputs)
                actual = _simple_cuda_accumulator_losses(accum_inputs)
                expected = torch.stack([reference[name] for name in _SIMPLE_CUDA_LOSS_NAMES])
                active_slots = [LOSS_SLOT_BY_NAME[name] for name in _SIMPLE_CUDA_LOSS_NAMES]
                for name in _SIMPLE_CUDA_LOSS_NAMES:
                    torch.testing.assert_close(
                        actual[LOSS_SLOT_BY_NAME[name]],
                        reference[name],
                        rtol=2e-5,
                        atol=2e-6,
                        msg=name,
                    )
                inactive_slots = [slot for slot in range(len(ACCUMULATOR_LOSS_NAMES)) if slot not in active_slots]
                torch.testing.assert_close(
                    actual[inactive_slots],
                    torch.zeros_like(actual[inactive_slots]),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(actual.sum(), expected.sum(), rtol=2e-5, atol=2e-6)
                self.assertEqual(expected.numel(), 14)
                for objective in ("unit", "nonuniform"):
                    with self.subTest(objective=objective):
                        coefficients = (
                            torch.ones_like(expected)
                            if objective == "unit"
                            else torch.linspace(-1.25, 2.0, expected.numel(), device=expected.device)
                        )
                        actual_gradients = torch.autograd.grad(
                            actual[active_slots],
                            [accum_inputs[name] for name in _SIMPLE_CUDA_PREDICTION_NAMES],
                            grad_outputs=coefficients,
                            retain_graph=objective == "unit",
                        )
                        reference_gradients = torch.autograd.grad(
                            expected,
                            [map_inputs[name] for name in _SIMPLE_CUDA_PREDICTION_NAMES],
                            grad_outputs=coefficients,
                            retain_graph=objective == "unit",
                        )
                        for name, actual_gradient, reference_gradient in zip(
                            _SIMPLE_CUDA_PREDICTION_NAMES,
                            actual_gradients,
                            reference_gradients,
                            strict=True,
                        ):
                            torch.testing.assert_close(
                                actual_gradient,
                                reference_gradient,
                                rtol=2e-5,
                                atol=2e-6,
                                msg=name,
                            )

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
        runners too.
        """
        from nurec.libs.losses.models.accumulator_module import (
            AccumulatorModuleLosses as ModuleLosses,
        )

        composed = sorted(
            f"{name}_{item['fn']}_{item['reduce']['name']}" for name, item in MINIMAL_CUDA_LOSS_CONFIGS.items()
        )
        self.assertEqual(composed, sorted(ModuleLosses.available))

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
                "ssim": {
                    "fn": "ssim",
                    "lambda_": 0.43,
                    "reduce": {"name": "mean"},
                    "window_size": 11,
                    "channel": 3,
                    "mask": {"mode": "GT", "value": 0.0},
                },
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
                self.register_buffer("_device_indicator", torch.zeros((), device="cuda"), persistent=False)
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
        aggregator_cuda = LossAccumulator(config, trainer_config)
        dataset = mock.MagicMock(spec=NCOREDataset)
        datasource = mock.MagicMock()
        datasource.get_semantic_classes_map.return_value = {"person": 1, "road": 32}
        dataset.get_datasource.return_value = datasource
        aggregator_cuda.initialize(dataset)
        self.assertEqual(aggregator_cuda.accumulator_losses, [])
        agg_ret_cuda = aggregator_cuda(step=0, model=model_cuda, results=results_cuda, target=batch_cuda)
        loss_cuda = agg_ret_cuda.compute_total()

        # Ensure loss requires grad
        self.assertTrue(loss_cuda.requires_grad, "CUDAloss should require gradients")
        self.assertGreater(loss_cuda.item(), 0, "CUDAloss should be positive")

        # Compute backward for CUDA
        fixture_loss_names = tuple(name for name in ACCUMULATOR_LOSS_NAMES if name != "road_gaussians_abs_mean")
        self.assertTrue(set(fixture_loss_names).issubset(agg_ret_cuda.loss_returns))
        # Every loss must be a view from one shared vector-unbind node. Per-slot
        # indexing would expose distinct SelectBackward nodes here.
        accumulator_returns = [agg_ret_cuda.loss_returns[name] for name in fixture_loss_names]
        self.assertTrue(all(loss_return.value is None for loss_return in accumulator_returns))
        weighted_grad_fns = [loss_return.reduced_value.grad_fn for loss_return in accumulator_returns]
        self.assertTrue(all("UnbindBackward" in type(grad_fn).__name__ for grad_fn in weighted_grad_fns))
        self.assertFalse(any("SelectBackward" in type(grad_fn).__name__ for grad_fn in weighted_grad_fns))
        self.assertTrue(all(grad_fn is weighted_grad_fns[0] for grad_fn in weighted_grad_fns))
        accumulator_gradient_coefficients = {
            loss_name: (
                # Keep the pre-existing end-to-end LiDAR comparison at its
                # original scale. Its Python reference broadcasts [N, 1]
                # against [N], an unrelated issue outside this integration.
                1.0 if loss_name == "lidar_l1_mean" else float(59 + 3 * slot)
            )
            for slot, loss_name in enumerate(fixture_loss_names)
        }
        all_cuda_loss_names = tuple(accumulator_gradient_coefficients)
        cuda_backward_losses: list[torch.Tensor] = []
        for loss_name, loss_return in agg_ret_cuda.loss_returns.items():
            if loss_name in accumulator_gradient_coefficients:
                cuda_backward_losses.append(accumulator_gradient_coefficients[loss_name] * loss_return.reduced_value)
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

        # Compare against independently reduced public producers.
        assert config is not None
        aggregator_py = LossAggregator(config, trainer_config)
        aggregator_py.initialize(dataset)
        agg_ret_py = aggregator_py(step=0, model=model_py, results=results_py, target=batch_py)
        loss_py = agg_ret_py.compute_total()

        # Ensure loss requires grad
        self.assertTrue(loss_py.requires_grad, "Python loss should require gradients")
        self.assertGreater(loss_py.item(), 0, "Python loss should be positive")

        # Compute backward for Python
        torch_backward_losses: list[torch.Tensor] = []
        for loss_name, loss_return in agg_ret_py.loss_returns.items():
            if loss_name in accumulator_gradient_coefficients:
                torch_backward_losses.append(accumulator_gradient_coefficients[loss_name] * loss_return.reduced_value)
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

        # The existing Python LiDAR reference broadcasts [N, 1] against [N]
        # and is intentionally outside this MR. The grouped CUDA member keeps
        # the established native LiDAR semantics, so retain the pre-existing
        # end-to-end tolerance but do not add a new exact comparison for that
        # unrelated reference issue here.
        exactly_comparable_loss_names = tuple(name for name in all_cuda_loss_names if name != "lidar_l1_mean")
        for loss_name in exactly_comparable_loss_names:
            torch.testing.assert_close(
                agg_ret_cuda.loss_returns[loss_name].reduced_value,
                agg_ret_py.loss_returns[loss_name].reduced_value,
                atol=2e-5,
                rtol=2e-5,
                msg=f"weighted accumulator mismatch for {loss_name}",
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
        # The unified accumulator exercises node-semantic through the same
        # deliberately large weighted cotangent as every other slot. Its fp32
        # reduction order can differ by a few ulps from the Python reference.
        self.assertTrue(
            torch.allclose(gaussian_densities_grad_cuda, gaussian_densities_grad_py, atol=5e-6, rtol=1e-8),
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
            case_cuda = LossAccumulator(case_config, trainer_config)
            case_python = LossAggregator(case_config, trainer_config)
            case_cuda.initialize(dataset)
            case_python.initialize(dataset)
            self.assertEqual(case_cuda.accumulator_losses, [])

            cuda_returns = case_cuda(
                step=0,
                model=model_cuda,
                results=case_results_cuda,
                target=batch_cuda,
            ).loss_returns
            python_returns = case_python(step=0, model=model_py, results=case_results_py, target=batch_py).loss_returns
            for loss_name in loss_names:
                self.assertIsNone(cuda_returns[loss_name].value)
                torch.testing.assert_close(
                    cuda_returns[loss_name].reduced_value,
                    python_returns[loss_name].reduced_value,
                    atol=2e-5,
                    rtol=2e-5,
                )

            cuda_objective = torch.stack(
                [
                    accumulator_gradient_coefficients[loss_name] * cuda_returns[loss_name].reduced_value
                    for loss_name in loss_names
                ]
            ).sum()
            python_objective = torch.stack(
                [
                    accumulator_gradient_coefficients[loss_name] * python_returns[loss_name].reduced_value
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
            skip_cuda = LossAccumulator(missing_signal_config, trainer_config)
            skip_python = LossAggregator(missing_signal_config, trainer_config)
            skip_cuda.initialize(dataset)
            skip_python.initialize(dataset)
            skip_cuda_returns = skip_cuda(
                step=0, model=model_cuda, results=results_cuda, target=batch_cuda
            ).loss_returns
            skip_python_returns = skip_python(step=0, model=model_py, results=results_py, target=batch_py).loss_returns
        skip_name = "node_semantic_gaussians_bce_with_logits_mean"
        self.assertIsNone(skip_cuda_returns[skip_name].value)
        torch.testing.assert_close(
            skip_cuda_returns[skip_name].reduced_value,
            skip_python_returns[skip_name].reduced_value,
            atol=2e-5,
            rtol=2e-5,
        )
        skip_cuda_grad = torch.autograd.grad(skip_cuda_returns[skip_name].reduced_value, background_densities_cuda)[0]
        skip_python_grad = torch.autograd.grad(skip_python_returns[skip_name].reduced_value, background_densities_py)[0]
        torch.testing.assert_close(skip_cuda_grad, skip_python_grad, atol=2e-5, rtol=2e-5)


def _capture_owned_forward(monkeypatch, module):
    """Intercept an owned module while preserving its accumulator input contract."""
    captured: dict[str, Any] = {}
    signature = inspect.signature(module.forward)

    def capture_forward(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        captured.update(bound.arguments)
        return bound.arguments["losses_vec"]

    monkeypatch.setattr(module, "forward", capture_forward)
    return captured


def _captured_weight(captured, name):
    return captured["loss_weights"][captured["loss_slots"].index(LOSS_SLOT_BY_NAME[name])]


def test_gaussian_z_scale_empty_layer_preserves_model_view(monkeypatch):
    """Ensure empty Gaussian z-scale layers pass the model view into the fused loss.

    This protects CUDA loss invocation when a configured layer has zero Gaussians: the loss
    remains present and gsplat's MEAN producer writes an exact zero.
    """
    if not torch.cuda.is_available():
        pytest.skip("ModuleLosses integration requires CUDA")

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
    model.register_buffer("_device_indicator", torch.zeros((), device="cuda"), persistent=False)
    model.post_processings = []
    empty_z_scales = torch.empty((1, 3), device=device)[:0]
    model.gaussians_nodes = {"road": SimpleNamespace(get_scales=lambda preactivation=True: empty_z_scales)}
    model.get_gaussians_node_ids = lambda: ["road"]

    accumulator = losses.accumulator._get_gaussian_loss_accumulator(0.01)
    captured = _capture_owned_forward(monkeypatch, accumulator)

    module_return = losses(
        step=0,
        model=model,
        results=GaussiansCompositeReturn(),
        target=SimpleNamespace(data=SimpleNamespace(camera=None, lidar=None)),
    )
    ret = module_return

    assert "gaussian_z_scale_abs_mean" in ret
    assert captured["z_scales"] is empty_z_scales
    assert accumulator.z_scale_threshold == 0.01
    assert _captured_weight(captured, "gaussian_z_scale_abs_mean") == 1.0


@pytest.mark.parametrize("include_empty_node", [False, True])
def test_empty_gaussian_regularizers_preserve_model_views(monkeypatch, include_empty_node):
    """Active scale/density losses deterministically return zero without eligible points."""
    if not torch.cuda.is_available():
        pytest.skip("ModuleLosses integration requires CUDA")

    device = torch.device("cuda")

    class _EmptyGaussianNode(BaseGaussianModel):
        def __init__(self):
            """Create a Gaussian node with empty scale and density tensors."""
            torch.nn.Module.__init__(self)
            # Slice non-empty storage so the test can verify that the active
            # model's empty views, rather than default-device dummies, flow to
            # the dispatcher.
            self._scales = torch.empty((1, 3), device=device)[:0]
            self._densities = torch.empty((1, 1), device=device)[:0]

        def get_scales(self, preactivation: bool = False):
            """Return the node's empty scale tensor."""
            return self._scales

        def get_densities(self, preactivation: bool = False):
            """Return the node's empty density tensor."""
            return self._densities

    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_buffer("_device_indicator", torch.tensor(0, device=device), persistent=False)
    model.post_processings = TypedModuleList[BasePostProcessing]([])
    nodes = {"empty": _EmptyGaussianNode()} if include_empty_node else {}
    model.gaussians_nodes = TypedModuleDict[BaseGaussianModel](nodes)
    model.get_gaussians_node_ids = lambda: list(model.gaussians_nodes.keys())

    losses = ModuleLosses()
    assert not any(name.startswith("dummy_") for name in vars(losses))
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

    accumulator = losses.accumulator._get_gaussian_loss_accumulator(0.0, densities_are_logits=not include_empty_node)
    captured = _capture_owned_forward(monkeypatch, accumulator)

    module_return = losses(
        step=0,
        model=model,
        results=GaussiansCompositeReturn(),
        target=SimpleNamespace(data=SimpleNamespace(camera=None, lidar=None)),
    )
    ret = module_return

    assert _captured_weight(captured, "gaussian_scale_abs_mean") == 1.0
    assert _captured_weight(captured, "gaussian_density_abs_mean") == 1.0
    assert captured["scales"].shape == (0, 3)
    assert captured["densities"].shape == (0,)
    assert captured["scales_segments"] == ()
    assert captured["densities_segments"] == ()
    if include_empty_node:
        empty_node = model.gaussians_nodes["empty"]
        assert captured["scales"] is empty_node._scales
        assert captured["densities"].untyped_storage().data_ptr() == (
            empty_node._densities.untyped_storage().data_ptr()
        )
    else:
        assert captured["scales"].device == model.device
        assert captured["densities"].device == model.device
        assert not captured["scales"].is_meta
        assert not captured["densities"].is_meta
    assert ret["gaussian_scale_abs_mean"].reduced_value.item() == 0.0
    assert ret["gaussian_density_abs_mean"].reduced_value.item() == 0.0


@pytest.mark.parametrize(
    ("present_grid_class", "present_input_name", "absent_input_name"),
    [
        (
            BilateralGridPerCamera,
            "grids_camera",
            "grids_frame",
        ),
        (
            BilateralGridPerFrame,
            "grids_frame",
            "grids_camera",
        ),
    ],
    ids=("camera-only", "frame-only"),
)
def test_grid_drift_only_enables_the_present_grid(
    monkeypatch,
    present_grid_class,
    present_input_name,
    absent_input_name,
):
    """A missing grid family must stay disabled instead of routing a CUDA dummy."""
    if not torch.cuda.is_available():
        pytest.skip("ModuleLosses integration requires CUDA")

    device = torch.device("cuda")
    grid = torch.zeros((2, GRID_NUM_CHANNELS, 3, 4, 5), device=device)
    grid_holder = SimpleNamespace(bilateral_grid=SimpleNamespace(grid=grid))

    def fake_get_bilateral_grids(_model, grid_class):
        return [grid_holder] if grid_class is present_grid_class else []

    monkeypatch.setattr(
        "nurec.libs.losses.models.accumulator_module._get_bilateral_grids",
        fake_get_bilateral_grids,
    )

    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_buffer("_device_indicator", torch.zeros((), device="cuda"), persistent=False)
    model.gaussians_nodes = TypedModuleDict[BaseGaussianModel]({})
    model.get_gaussians_node_ids = lambda: []

    losses = ModuleLosses()
    losses.losses = [
        SimpleNamespace(
            name="bilateral_grid_drift_identity_distance_mean",
            lambda_=1.0,
            should_run_fn=lambda step: True,
        )
    ]
    accumulator = losses.accumulator.bg_grid_loss_accumulator
    captured = _capture_owned_forward(monkeypatch, accumulator)

    module_return = losses(
        0,
        model,
        GaussiansCompositeReturn(),
        SimpleNamespace(data=SimpleNamespace(camera=None, lidar=None)),
    )
    ret = module_return

    assert captured[present_input_name].untyped_storage().data_ptr() == grid.untyped_storage().data_ptr()
    assert captured[absent_input_name] is None
    assert _captured_weight(captured, "bilateral_grid_drift_identity_distance_mean") == 1.0
    assert ret["bilateral_grid_drift_identity_distance_mean"].reduced_value.item() == 0.0


def test_configured_accumulator_modules_are_owned_and_reused():
    """Configured gsplat modules register and cache without allocating CUDA inputs."""

    losses = ModuleLosses()
    accumulator = losses.accumulator
    assert isinstance(accumulator, GsplatLossAdapter)
    assert dict(losses.named_children())["accumulator"] is accumulator
    configured = (
        (accumulator._get_gaussian_loss_accumulator, (0.125,)),
        (accumulator._get_bg_track_node_semantic_loss_accumulator, (-18.0,)),
        (accumulator._get_ssim_loss_accumulator, (11, "constant", 0.25)),
        (accumulator._get_ground_gaussians_loss_accumulator, (0.0, 1.0, 0.1, 1.0)),
    )
    owned = [
        accumulator.camera_loss_accumulator,
        accumulator.lidar_loss_accumulator,
        accumulator.bg_grid_loss_accumulator,
    ]
    for factory, args in configured:
        module = factory(*args)
        assert factory(*args) is module
        owned.append(module)

    registered_paths = {module: name for name, module in losses.named_modules()}
    assert len(set(owned)) == 7
    assert all(registered_paths[module].startswith("accumulator.") for module in owned)
    assert all(module in tuple(accumulator.modules()) for module in owned)


def test_empty_rigid_node_skips_out_of_bound_loss(monkeypatch):
    """Empty rigid nodes preserve the original out-of-bound loss filtering."""
    if not torch.cuda.is_available():
        pytest.skip("ModuleLosses integration requires CUDA")

    device = torch.device("cuda")

    class _EmptyRigidNode(RigidGaussianModel):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.positions = torch.nn.Parameter(torch.empty((0, 3), device=device))
            self.cuboid_tracks = cast(
                Any,
                SimpleNamespace(cuboids_dims=torch.empty((0, 3), device=device)),
            )
            self.gaussian_cuboid_ids = cast(
                Any,
                torch.empty((0,), dtype=torch.int32, device=device),
            )

        def get_positions(self):
            return self.positions

    empty_node = _EmptyRigidNode()
    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_buffer("_device_indicator", torch.zeros((), device="cuda"), persistent=False)
    model.post_processings = TypedModuleList[BasePostProcessing]([])
    model.gaussians_nodes = TypedModuleDict[BaseGaussianModel]({"empty": empty_node})
    model.get_gaussians_node_ids = lambda: ["empty"]

    losses = ModuleLosses()
    losses.losses = [
        SimpleNamespace(
            name="out_of_bound_l1_mean",
            lambda_=2.0,
            should_run_fn=lambda step: True,
        )
    ]

    gaussian_forward = mock.Mock(side_effect=AssertionError("empty rigid nodes must not dispatch"))
    monkeypatch.setattr(GaussianLossAccumulator, "forward", gaussian_forward)
    module_return = losses(
        0,
        model,
        GaussiansCompositeReturn(),
        SimpleNamespace(data=SimpleNamespace(camera=None, lidar=None)),
    )
    ret = module_return

    assert "out_of_bound_l1_mean" not in ret
    gaussian_forward.assert_not_called()
    assert module_return.weighted_loss_vector.count_nonzero().item() == 0
    assert empty_node.positions.grad is None


def test_accumulator_dispatch_reads_scheduler_updated_lambdas_and_zeros_inactive_weights(
    monkeypatch,
):
    """Owned modules receive current scheduler weights and zero inactive members."""
    if not torch.cuda.is_available():
        pytest.skip("ModuleLosses integration requires CUDA")

    device = torch.device("cuda")
    node = SimpleNamespace(
        get_scales=lambda preactivation=True: torch.zeros((1, 3), device=device),
        get_densities=lambda preactivation=False: torch.zeros((1, 1), device=device),
    )
    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_buffer("_device_indicator", torch.zeros((), device="cuda"), persistent=False)
    model.post_processings = TypedModuleList[BasePostProcessing]([])
    model.gaussians_nodes = {"node": node}
    model.get_gaussians_node_ids = lambda: ["node"]

    configured_loss = make_loss(
        "gaussian_scale",
        LossItemConfig.model_validate(
            {
                "fn": "abs",
                "lambda_": 2.0,
                "reduce": {"name": "mean"},
                "lambda_scheduler": {
                    "name": "linear_lambda",
                    "start": 0,
                    "end": 10,
                    "update_interval": "step",
                    "update_frequency": 1,
                    "lambda_end": 5.0,
                },
            }
        ),
        MockTrainerConfig(),
    )
    losses = ModuleLosses()
    losses.losses = [configured_loss]
    scale_slot = LOSS_SLOT_BY_NAME[configured_loss.name]
    accumulator = losses.accumulator._get_gaussian_loss_accumulator(0.0)
    forward_signature = inspect.signature(accumulator.forward)
    dispatch_weights: list[tuple[float, ...]] = []

    def fake_forward(*args, **kwargs):
        bound = forward_signature.bind(*args, **kwargs)
        loss_weights = tuple(bound.arguments["loss_weights"])
        dispatch_weights.append(loss_weights)
        losses_vec = bound.arguments["losses_vec"]
        losses_vec[scale_slot] = 7.0 * loss_weights[0]
        return losses_vec

    monkeypatch.setattr(accumulator, "forward", fake_forward)
    target = SimpleNamespace(data=SimpleNamespace(camera=None, lidar=None))

    first = losses(0, model, GaussiansCompositeReturn(), target)[configured_loss.name]
    configured_loss.update_step_train_batch_start(epoch=0, global_step=10, system=None)
    second = losses(10, model, GaussiansCompositeReturn(), target)[configured_loss.name]

    assert configured_loss.lambda_ == pytest.approx(5.0)
    assert first.value is None and second.value is None
    assert first.reduced_value.item() == 14.0
    assert second.reduced_value.item() == 35.0
    assert len(dispatch_weights) == 2
    for weights in dispatch_weights:
        assert len(weights) == 5
        assert all(weight == 0.0 for slot, weight in enumerate(weights) if slot != 0)
    assert dispatch_weights[0][0] == 2.0
    assert dispatch_weights[1][0] == 5.0


@pytest.mark.parametrize(
    "node_count",
    [GaussianLossAccumulator.MAX_SEGMENTS, GaussianLossAccumulator.MAX_SEGMENTS + 1],
)
def test_gaussian_members_segment_independently_with_flat_overflow(monkeypatch, node_count):
    """Scale/density overflow must not force a two-node OOB member to flatten."""
    if not torch.cuda.is_available():
        pytest.skip("ModuleLosses integration requires CUDA")

    device = torch.device("cuda")

    class _Node(BaseGaussianModel):
        def __init__(self, value: float):
            torch.nn.Module.__init__(self)
            self.scale = torch.nn.Parameter(torch.full((1, 3), value, device=device))
            self.density = torch.nn.Parameter(torch.full((1, 1), value + 1.0, device=device))

        def get_scales(self, preactivation: bool = False):
            return self.scale if preactivation else self.scale.exp()

        def get_densities(self, preactivation: bool = False):
            return self.density

    class _RigidNode(RigidGaussianModel):
        def __init__(self, value: float):
            torch.nn.Module.__init__(self)
            self.scale = torch.nn.Parameter(torch.full((1, 3), value, device=device))
            self.density = torch.nn.Parameter(torch.full((1, 1), value + 1.0, device=device))
            self.position = torch.nn.Parameter(torch.full((1, 3), value + 2.0, device=device))
            self.cuboid_tracks = cast(Any, SimpleNamespace(cuboids_dims=torch.ones((2, 3), device=device)))
            self.gaussian_cuboid_ids = cast(Any, torch.tensor([1], dtype=torch.int32, device=device))

        def get_scales(self, preactivation: bool = False):
            return self.scale if preactivation else self.scale.exp()

        def get_densities(self, preactivation: bool = False):
            return self.density

        def get_positions(self):
            return self.position

    nodes = {f"node_{i}": (_RigidNode(float(i)) if i < 2 else _Node(float(i))) for i in range(node_count)}
    if node_count > GaussianLossAccumulator.MAX_SEGMENTS:
        with torch.no_grad():
            nodes["node_0"].scale.fill_(float("inf"))
            nodes["node_0"].density.fill_(float("inf"))
    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_buffer("_device_indicator", torch.zeros((), device="cuda"), persistent=False)
    model.post_processings = TypedModuleList[BasePostProcessing]([])
    model.gaussians_nodes = nodes
    model.get_gaussians_node_ids = lambda: list(nodes)

    zero_weight_node = "node_0"
    common = {
        "lambda_": 1.0,
        "layer_lambdas": {zero_weight_node: 0.0},
        "visibility_filter": True,
        "occlusion_aware": False,
        "should_run_fn": lambda step: True,
    }
    losses = ModuleLosses()
    losses.losses = [
        SimpleNamespace(name="gaussian_scale_abs_mean", **common),
        SimpleNamespace(name="gaussian_density_abs_mean", **common),
        SimpleNamespace(name="out_of_bound_l1_mean", lambda_=1.0, should_run_fn=lambda step: True),
    ]

    captured: dict[str, object] = {}
    accumulator = losses.accumulator._get_gaussian_loss_accumulator(0.0)
    real_forward = accumulator.forward
    forward_signature = inspect.signature(real_forward)

    def fake_forward(*args, **kwargs):
        bound = forward_signature.bind(*args, **kwargs)
        captured.update(bound.arguments)
        return torch.zeros(len(ACCUMULATOR_LOSS_NAMES), device=device)

    visibility_counts: list[int] = []

    def fake_visibility(**kwargs):
        visibility_counts.append(kwargs["element_count"])
        return torch.ones(kwargs["element_count"], device=kwargs["device"], dtype=kwargs["dtype"])

    monkeypatch.setattr(accumulator, "forward", fake_forward)
    monkeypatch.setattr(
        "nurec.libs.losses.models.accumulator_module.get_rendered_visibility_mask",
        fake_visibility,
    )
    monkeypatch.setattr(
        "nurec.libs.losses.models.accumulator_module._maybe_update_mcmc_visibility_counters",
        lambda **kwargs: None,
    )
    losses(
        0,
        model,
        GaussiansCompositeReturn(),
        SimpleNamespace(data=SimpleNamespace(camera=None, lidar=None)),
    )

    assert visibility_counts == [node_count]
    assert _captured_weight(captured, "gaussian_scale_abs_mean") == 1.0
    assert _captured_weight(captured, "gaussian_density_abs_mean") == 1.0
    assert _captured_weight(captured, "out_of_bound_l1_mean") == 1.0
    assert len(captured["oob_positions_segments"]) == 2
    assert captured["oob_positions_segments"][0] is nodes["node_0"].position
    assert captured["positions"].shape == (0, 3)
    assert not captured["positions"].requires_grad
    routed_oob_ids = captured["oob_cuboid_ids_segments"]
    assert len(routed_oob_ids) == 2
    for index, ids in enumerate(routed_oob_ids):
        assert ids is nodes[f"node_{index}"].gaussian_cuboid_ids
        assert ids.dtype == torch.int32

    if node_count == GaussianLossAccumulator.MAX_SEGMENTS:
        assert len(captured["scales_segments"]) == GaussianLossAccumulator.MAX_SEGMENTS
        assert len(captured["densities_segments"]) == GaussianLossAccumulator.MAX_SEGMENTS
        assert captured["scales_segments"][0] is nodes["node_0"].scale
        assert (
            captured["densities_segments"][0].untyped_storage().data_ptr()
            == nodes["node_0"].density.untyped_storage().data_ptr()
        )
        assert captured["scale_segment_weights"][0] == 0.0
        assert captured["density_segment_weights"][0] == 0.0
        assert captured["scales"].shape == (0, 3)
        assert captured["densities"].shape == (0,)
        assert not captured["scales"].requires_grad
        assert not captured["densities"].requires_grad

        # Exercise the owned native accumulator with the tensors
        # produced by ModuleLosses. Detaching metadata must preserve every
        # segment's value and gradient, including the zero-weight first node.
        native_values = real_forward(**captured)
        scales = torch.cat([node.scale for node in nodes.values()])
        densities = torch.cat([node.density.flatten() for node in nodes.values()])
        layer_weights = scales.new_ones(node_count)
        layer_weights[0] = 0.0
        expected_scale = (scales.exp() * layer_weights[:, None]).mean()
        expected_density = (densities.abs() * layer_weights).mean()
        positions = torch.cat([nodes[f"node_{index}"].position for index in range(2)])
        expected_oob = (positions.abs() - 0.5).clamp_min(0).mean()
        expected_values = torch.zeros_like(native_values)
        expected_values[LOSS_SLOT_BY_NAME["gaussian_scale_abs_mean"]] = expected_scale
        expected_values[LOSS_SLOT_BY_NAME["gaussian_density_abs_mean"]] = expected_density
        expected_values[LOSS_SLOT_BY_NAME["out_of_bound_l1_mean"]] = expected_oob
        torch.testing.assert_close(native_values, expected_values, atol=1e-6, rtol=1e-5)
        parameters = [parameter for node in nodes.values() for parameter in (node.scale, node.density)]
        parameters.extend(nodes[f"node_{index}"].position for index in range(2))
        native_grads = torch.autograd.grad(native_values.sum(), parameters)
        reference_grads = torch.autograd.grad(expected_values.sum(), parameters)
        for native_grad, reference_grad in zip(native_grads, reference_grads):
            torch.testing.assert_close(native_grad, reference_grad, atol=1e-6, rtol=1e-5)
    else:
        assert captured["scales_segments"] == ()
        assert captured["densities_segments"] == ()
        assert captured["scales"].shape == (17, 3)
        assert captured["densities"].shape == (17,)
        assert torch.isneginf(captured["scales"][0]).all()
        assert captured["densities"][0].abs().sum().item() == 0.0

        torch.exp(captured["scales"]).sum().backward(retain_graph=True)
        captured["densities"].abs().sum().backward()
        assert nodes[zero_weight_node].scale.grad.abs().sum().item() == 0.0
        assert nodes[zero_weight_node].density.grad.abs().sum().item() == 0.0
        assert nodes["node_1"].scale.grad.abs().sum().item() > 0.0
        assert nodes["node_1"].density.grad.abs().sum().item() > 0.0


def test_out_of_bound_member_flattens_above_segment_limit(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("ModuleLosses integration requires CUDA")

    device = torch.device("cuda")

    class _RigidNode(RigidGaussianModel):
        def __init__(self, value: float):
            torch.nn.Module.__init__(self)
            self.position = torch.nn.Parameter(torch.full((1, 3), value, device=device))
            self.cuboid_tracks = cast(
                Any,
                SimpleNamespace(
                    cuboids_dims=torch.tensor(
                        [[value + 10.0, value + 11.0, value + 12.0]],
                        device=device,
                    )
                ),
            )
            self.gaussian_cuboid_ids = cast(Any, torch.tensor([0], dtype=torch.int32, device=device))

        def get_positions(self):
            return self.position

    nodes = {f"node_{i}": _RigidNode(float(i)) for i in range(17)}
    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_buffer("_device_indicator", torch.zeros((), device="cuda"), persistent=False)
    model.post_processings = TypedModuleList[BasePostProcessing]([])
    model.gaussians_nodes = nodes
    model.get_gaussians_node_ids = lambda: list(nodes)

    losses = ModuleLosses()
    losses.losses = [SimpleNamespace(name="out_of_bound_l1_mean", lambda_=1.0, should_run_fn=lambda step: True)]
    accumulator = losses.accumulator._get_gaussian_loss_accumulator(0.0)
    captured = _capture_owned_forward(monkeypatch, accumulator)

    losses(
        0,
        model,
        GaussiansCompositeReturn(),
        SimpleNamespace(data=SimpleNamespace(camera=None, lidar=None)),
    )

    assert captured["oob_positions_segments"] == ()
    assert captured["oob_cuboid_dims_tables"] == ()
    assert captured["oob_cuboid_ids_segments"] == ()
    assert captured["positions"].shape == (17, 3)
    assert captured["cuboid_dims"].shape == (17, 3)
    assert _captured_weight(captured, "out_of_bound_l1_mean") == 1.0
    torch.testing.assert_close(
        captured["cuboid_dims"][7],
        torch.tensor([17.0, 18.0, 19.0], device=device),
    )


def test_camera_loss_accumulator_requires_rendered_camera():
    """Configured camera losses fail before an unwritten accumulator slot can escape."""
    if not torch.cuda.is_available():
        pytest.skip("ModuleLosses integration requires CUDA")

    device = torch.device("cuda")
    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_buffer("_device_indicator", torch.zeros((), device="cuda"), persistent=False)
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


@pytest.mark.parametrize("mask_shape", [(2, 1), (2, 3)])
def test_deform_broadcast_preserves_source_storage(monkeypatch, mask_shape):
    if not torch.cuda.is_available():
        pytest.skip("ModuleLosses integration requires CUDA")
    deformation = torch.tensor([[1.0], [-2.0]], device="cuda", requires_grad=True)
    mask = torch.ones(mask_shape, device="cuda")
    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_buffer("_device_indicator", torch.zeros((), device="cuda"), persistent=False)
    model.post_processings = TypedModuleList[BasePostProcessing]([])
    model.gaussians_nodes = {}
    model.get_gaussians_node_ids = lambda: []
    losses = ModuleLosses()
    losses.losses = [
        SimpleNamespace(
            name="deform_smoothness_abs_mean",
            lambda_=1.0,
            should_run_fn=lambda step: True,
        )
    ]
    accumulator = losses.accumulator._get_gaussian_loss_accumulator(0.0)
    captured = _capture_owned_forward(monkeypatch, accumulator)

    losses(
        0,
        model,
        GaussiansCompositeReturn(deform_smoothness=deformation, deform_smoothness_mask=mask),
        SimpleNamespace(data=SimpleNamespace(camera=None, lidar=None)),
    )
    assert captured["deform_mask"] is mask
    assert captured["deformation"].shape == mask_shape
    assert captured["deformation"].untyped_storage().data_ptr() == deformation.untyped_storage().data_ptr()
    torch.testing.assert_close(captured["deformation"], deformation.expand(mask_shape))


@pytest.mark.parametrize("activation", ["sigmoid", "exp"])
@pytest.mark.parametrize("node_count", [2, GaussianLossAccumulator.MAX_SEGMENTS + 1])
def test_gaussian_density_sigmoid_fusion_preserves_activation_and_overflow(monkeypatch, activation, node_count):
    if not torch.cuda.is_available():
        pytest.skip("ModuleLosses requires CUDA")
    device = torch.device("cuda")

    class _DensityNode(BaseGaussianModel):
        def __init__(self, index):
            torch.nn.Module.__init__(self)
            self.config = cast(Any, SimpleNamespace(density_activation=activation))
            self.density = torch.nn.Parameter(torch.tensor([[-1.0 + index * 0.1], [0.5]], device=device))
            self.requests = []

        def get_densities(self, preactivation=False):
            self.requests.append(preactivation)
            if preactivation:
                return self.density
            return self.density.sigmoid() if activation == "sigmoid" else self.density.exp()

    nodes = {f"node_{index}": _DensityNode(index) for index in range(node_count)}
    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_buffer("_device_indicator", torch.zeros((), device=device), persistent=False)
    model.post_processings = TypedModuleList[BasePostProcessing]([])
    model.gaussians_nodes = TypedModuleDict[BaseGaussianModel](nodes)
    model.get_gaussians_node_ids = lambda: list(nodes)
    losses = ModuleLosses()
    losses.losses = [
        SimpleNamespace(
            name="gaussian_density_abs_mean",
            lambda_=0.7,
            layer_lambdas={"node_0": 0.0},
            visibility_filter=False,
            should_run_fn=lambda step: True,
        )
    ]
    monkeypatch.setattr(
        "nurec.libs.losses.models.accumulator_module._maybe_update_mcmc_visibility_counters",
        lambda **kwargs: None,
    )
    result = losses(
        0,
        model,
        GaussiansCompositeReturn(),
        SimpleNamespace(data=SimpleNamespace(camera=None, lidar=None)),
    )
    actual = result["gaussian_density_abs_mean"].reduced_value
    refs = [node.density.detach().clone().requires_grad_() for node in nodes.values()]
    activated = [value.sigmoid() if activation == "sigmoid" else value.exp() for value in refs]
    expected = 0.7 * torch.cat([value * (0.0 if index == 0 else 1.0) for index, value in enumerate(activated)]).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    fused = activation == "sigmoid" and node_count <= GaussianLossAccumulator.MAX_SEGMENTS
    for node, ref in zip(nodes.values(), refs):
        assert node.requests == [fused]
        torch.testing.assert_close(node.density.grad, ref.grad)


@pytest.mark.parametrize("weight", [0.0, 0.7, -0.4])
def test_ground_group_matches_existing_inputs_and_skips_zero_sampling(weight):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from gsplat.losses import FusedGroundGaussiansLosses

    torch.manual_seed(47)
    positions = torch.randn((127, 3), device="cuda", requires_grad=True)
    rotations = torch.randn((127, 4), device="cuda", requires_grad=True)
    pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda", requires_grad=True)
    offsets = torch.rand(8, device="cuda")
    sampler = mock.Mock(return_value=offsets)
    get_rotations = mock.Mock(return_value=rotations)
    get_positions = mock.Mock(return_value=positions)
    road = SimpleNamespace(
        name="road_gaussians_abs_mean",
        lambda_=weight,
        layer_name="road",
        min=-2.0,
        range=3.0,
        grid_len=2.0,
        rotation_lambda=0.1,
        _make_random_values=sampler,
        should_run_fn=lambda step: step >= 2,
    )
    losses = ModuleLosses()
    losses.losses = [road]
    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_buffer("_device_indicator", torch.tensor(0, device="cuda"), persistent=False)
    model.post_processings = []
    model.gaussians_nodes = {"road": SimpleNamespace(get_positions=get_positions, get_rotations=get_rotations)}
    model.get_gaussians_node_ids = lambda: list(model.gaussians_nodes)
    target = SimpleNamespace(
        rendering=SimpleNamespace(
            camera=SimpleNamespace(
                b=1,
                **({"poses_tquat_startend": pose.reshape(1, 1, 7).expand(1, 2, 7)} if weight != 0.0 else {}),
            )
        )
    )
    result = losses(2, model, SimpleNamespace(), target)
    value = result[road.name].reduced_value
    if weight == 0.0:
        sampler.assert_not_called()
        get_positions.assert_not_called()
        get_rotations.assert_not_called()
        assert value == 0
    else:
        sampler.assert_called_once_with(positions.device, positions.dtype)
        get_positions.assert_called_once_with()
        get_rotations.assert_called_once_with(quaternion_format="wxyz", preactivation=True)
        # The original map path normalized and reordered stored parameters in Python.
        xyzw_rotations = torch.nn.functional.normalize(rotations, dim=-1)[:, [1, 2, 3, 0]]
        scalar = FusedGroundGaussiansLosses(
            min_bias=road.min,
            range_bias=road.range,
            grid_len=road.grid_len,
            rotation_lambda=road.rotation_lambda,
        )(positions, xyzw_rotations, pose.detach(), offsets)
        torch.testing.assert_close(value, weight * scalar, atol=2e-5, rtol=2e-5)
        actual = torch.autograd.grad(value, (positions, rotations))
        expected = torch.autograd.grad(weight * scalar, (positions, rotations))
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=3e-5, rtol=2e-4)
        assert pose.grad is None
        assert any(type(module).__name__ == "GroundGaussiansLossAccumulator" for module in losses.modules())
    sampler.reset_mock()
    assert not losses(1, model, SimpleNamespace(), target)
    sampler.assert_not_called()
    target.rendering.camera = None
    assert not losses(2, model, SimpleNamespace(), target)
    target.rendering.camera = SimpleNamespace(b=1)
    model.gaussians_nodes.clear()
    assert not losses(2, model, SimpleNamespace(), target)
    sampler.assert_not_called()


@pytest.mark.parametrize(
    "rotation_lambda,grouped",
    [
        (0.1, True),
        (0.0, True),
        (torch.finfo(torch.float32).eps, True),
        (torch.finfo(torch.float32).max, True),
        (None, False),
        (-0.1, False),
        (1e-32, False),
        (torch.finfo(torch.float32).eps / 2, False),
        (1e40, False),
        (float("inf"), False),
        (-float("inf"), False),
        (float("nan"), False),
    ],
)
def test_ground_aggregator_preserves_abs_contract(rotation_lambda, grouped):
    from nurec.libs.losses.orchestration import loss_accumulator as aggregation
    from nurec.libs.losses.orchestration import loss_producers

    config = LossConfig.model_validate(
        {
            "road_gaussians": {
                **MINIMAL_CUDA_LOSS_CONFIGS["road_gaussians"],
                "rotation_lambda": rotation_lambda,
            }
        }
    )
    modules = SimpleNamespace(available=ModuleLosses.available, losses=[])
    road = object()
    with (
        mock.patch.object(aggregation, "AccumulatorModuleLosses", return_value=modules),
        mock.patch.object(loss_producers, "make_loss", return_value=road),
    ):
        aggregator = LossAccumulator(config, SimpleNamespace())
    assert modules.losses == ([road] if grouped else [])
    assert aggregator.accumulator_losses == ([] if grouped else [road])


@pytest.mark.parametrize("rotation_lambda", [0.0, 0.1])
def test_ground_production_accumulator_matches_abs_value_and_gradients(rotation_lambda):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from gsplat.losses import FusedGroundGaussiansLosses

    torch.manual_seed(47)
    positions = torch.randn((127, 3), device="cuda", requires_grad=True)
    rotations = torch.randn((127, 4), device="cuda", requires_grad=True)
    pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda")
    offsets = torch.rand(8, device="cuda")
    config = LossConfig.model_validate(
        {
            "road_gaussians": {
                **MINIMAL_CUDA_LOSS_CONFIGS["road_gaussians"],
                "lambda_": 0.7,
                "rotation_lambda": rotation_lambda,
                "min": -2.0,
                "range": 3.0,
                "grid_len": 2.0,
            }
        }
    )
    accumulator = _ProductionLossAccumulator(config, MockTrainerConfig())
    assert accumulator.accumulator_losses == []
    road = accumulator.in_accumulator.losses[0]
    road._make_random_values = mock.Mock(return_value=offsets)
    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_buffer("_device_indicator", torch.tensor(0, device="cuda"), persistent=False)
    model.post_processings = []
    model.gaussians_nodes = {
        "road": SimpleNamespace(get_positions=lambda: positions, get_rotations=lambda **_: rotations)
    }
    model.get_gaussians_node_ids = lambda: list(model.gaussians_nodes)
    target = SimpleNamespace(
        rendering=SimpleNamespace(
            camera=SimpleNamespace(b=1, poses_tquat_startend=pose.reshape(1, 1, 7).expand(1, 2, 7))
        )
    )
    result = accumulator(step=0, model=model, results=SimpleNamespace(), target=target)
    scalar = FusedGroundGaussiansLosses(
        min_bias=road.min,
        range_bias=road.range,
        grid_len=road.grid_len,
        rotation_lambda=rotation_lambda,
        rotations_are_raw_wxyz=True,
    )(positions, rotations, pose, offsets)
    expected = 0.7 * scalar.abs().mean()
    torch.testing.assert_close(result.compute_total(), expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(result.weighted_value(road.name), expected, atol=2e-5, rtol=2e-5)
    actual_gradients = torch.autograd.grad(result.compute_total(), (positions, rotations))
    expected_gradients = torch.autograd.grad(expected, (positions, rotations))
    for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference, atol=3e-5, rtol=2e-4)


def test_ppisp_accumulator_entry_skips_the_legacy_return():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from nurec.libs.losses.models.render_losses import PPISPLoss
    from nurec.models.post_processing import PPISPPostProcessing

    raw = torch.tensor(3.0, device="cuda", requires_grad=True)
    producer = object.__new__(PPISPLoss)
    torch.nn.Module.__init__(producer)
    producer.name = "ppisp_ppisp_loss_mean"
    producer.lambda_ = 0.25
    producer.start_step = 0
    producer.loss_fn = mock.Mock(return_value=raw)
    post_processing = object.__new__(PPISPPostProcessing)
    torch.nn.Module.__init__(post_processing)
    post_processing.ppisp = object()
    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_buffer("_device_indicator", raw.new_empty(0), persistent=False)
    model.post_processings = TypedModuleList([post_processing])
    aggregator = object.__new__(_ProductionLossAccumulator)
    aggregator.in_accumulator = SimpleNamespace(losses=[])
    aggregator.accumulator_losses = [producer]
    with mock.patch.object(producer, "forward", side_effect=AssertionError("legacy forward must not run")):
        result = aggregator(step=0, model=model, results=SimpleNamespace(), target=SimpleNamespace())
    torch.testing.assert_close(result.compute_total(), raw.new_tensor(0.75))
    result.log(lambda *_: None)
    result.compute_total().backward()
    torch.testing.assert_close(raw.grad, raw.new_tensor(0.25))
    producer.loss_fn.assert_called_once_with(post_processing.ppisp)


def test_independent_legacy_compatibility_keeps_weighting_and_reduces_once():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from nurec.libs.losses.models.base_losses import BaseRenderLoss
    from nurec.libs.losses.orchestration.config import LossReturn

    raw = torch.tensor([2.0, 6.0], device="cuda", requires_grad=True)
    reducer = mock.Mock(side_effect=lambda value, **_: value.mean())
    producer = mock.Mock(spec=BaseRenderLoss)
    producer.name = "independent"
    producer.should_run_fn.return_value = True
    producer.return_value = LossReturn(name=producer.name, lambda_=2.0, value=raw, reduce_fn=reducer)
    producer.accumulate.side_effect = lambda **kwargs: BaseRenderLoss.accumulate(producer, **kwargs)
    aggregator = object.__new__(_ProductionLossAccumulator)
    aggregator.in_accumulator = SimpleNamespace(losses=[])
    aggregator.accumulator_losses = [producer]
    result = aggregator(
        step=0, model=SimpleNamespace(device=raw.device), results=SimpleNamespace(), target=SimpleNamespace()
    )
    records = {}
    result.log(lambda name, value: records.__setitem__(name, value))
    torch.testing.assert_close(result.compute_total(), raw.new_tensor(8.0))
    torch.testing.assert_close(records["loss/independent"], raw.new_tensor(8.0))
    assert records["train_params/independent_lambda"] == 2.0
    result.compute_total().backward()
    torch.testing.assert_close(raw.grad, torch.ones_like(raw))
    reducer.assert_called_once()


def test_generic_independent_accumulator_producer_needs_no_orchestrator_branch():
    from nurec.libs.losses.models.base_losses import BaseRenderLoss
    from nurec.libs.losses.orchestration.config import AccumulatorLoss

    class NewProducer(BaseRenderLoss):
        def forward(self, **kwargs):
            raise AssertionError("legacy forward must not run")

        def accumulate(self, **kwargs):
            assert kwargs["results"] is results
            return AccumulatorLoss(
                name=self.name, lambda_=self.lambda_, value=raw, reduce_fn=lambda value, **_: value.mean()
            )

    raw = torch.tensor([2.0, 6.0], device="cuda", requires_grad=True)
    results = SimpleNamespace()
    producer = object.__new__(NewProducer)
    torch.nn.Module.__init__(producer)
    producer.name, producer.lambda_, producer.start_step = "new_loss", 2.0, 0
    aggregator = object.__new__(_ProductionLossAccumulator)
    aggregator.in_accumulator = SimpleNamespace(losses=[])
    aggregator.accumulator_losses = [producer]
    result = aggregator(step=0, model=SimpleNamespace(device=raw.device), results=results, target=SimpleNamespace())
    torch.testing.assert_close(result.compute_total(), raw.new_tensor(8.0))
    result.compute_total().backward()
    torch.testing.assert_close(raw.grad, torch.ones_like(raw))


def test_generic_primitive_loss_accumulator_uses_shared_input_contract():
    from nurec.libs.losses.models.base_losses import BasePrimitiveLoss
    from nurec.libs.losses.orchestration.config import LossReturn

    class PrimitiveProducer(BasePrimitiveLoss):
        needs_target = True

        def forward(self, **kwargs):
            assert kwargs["primitive"] is primitive
            assert kwargs["target"] is target
            return LossReturn(name=self.name, lambda_=3.0, value=raw, reduce_fn=lambda value, **_: value.mean())

    raw = torch.tensor([2.0, 6.0], device="cuda", requires_grad=True)
    primitive, target = object(), SimpleNamespace()
    producer = object.__new__(PrimitiveProducer)
    torch.nn.Module.__init__(producer)
    producer.name, producer.lambda_, producer.start_step = "primitive_loss", 3.0, 0
    aggregator = object.__new__(_ProductionLossAccumulator)
    aggregator.in_accumulator = SimpleNamespace(losses=[])
    aggregator.accumulator_losses = [producer]
    result = aggregator(
        step=0,
        model=SimpleNamespace(device=raw.device),
        results=SimpleNamespace(),
        target=target,
        primitive=primitive,
        supervision_pack=object(),
        context=object(),
    )
    torch.testing.assert_close(result.compute_total(), raw.new_tensor(12.0))
    assert result.weights_by_name["primitive_loss"] == 3.0
    result.compute_total().backward()
    torch.testing.assert_close(raw.grad, raw.new_full((2,), 1.5))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBackgroundLossTrackPreparation(unittest.TestCase):
    @staticmethod
    def _tracks(n_poses: int, name: str) -> CuboidTracks:
        return CuboidTracks.Factory.from_numpy(
            tracks_id=[name],
            tracks_poses=[np.tile(np.eye(4, dtype=np.float32), (n_poses, 1, 1))],
            tracks_timestamps_us=[np.arange(n_poses, dtype=np.int64) * 1000],
            tracks_label_class=["car"],
            tracks_flags=[TrackFlags.NONE],
            cuboids_dims=[np.array([2.0, 3.0, 4.0], dtype=np.float32)],
            device=torch.device("cuda"),
        )

    def test_loss_fields_and_pose_gradients_match_full_concatenation(self):
        tracks = [self._tracks(2, "first"), self._tracks(4, "second")]
        poses = [track.tracks_poses.data.requires_grad_() for track in tracks]
        original = CuboidTracks.Ops.concatenate(tracks)
        prepared = _prepare_loss_tracks(tracks)
        reference = (
            original.tracks_packinfo,
            original.tracks_poses.data,
            original.tracks_timestamps_us,
            original.cuboids_dims,
        )
        for actual, expected in zip(prepared, reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_grads = torch.autograd.grad(reference[1].square().sum(), poses)
        actual_grads = torch.autograd.grad(prepared.tracks_poses.square().sum(), poses)
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_single_owner_borrows_original_tensors(self):
        tracks = self._tracks(2, "first")
        prepared = _prepare_loss_tracks([tracks])
        self.assertIs(prepared.tracks_packinfo, tracks.tracks_packinfo)
        self.assertIs(prepared.tracks_poses, tracks.tracks_poses.data)
        self.assertIs(prepared.tracks_timestamps_us, tracks.tracks_timestamps_us)
        self.assertIs(prepared.cuboids_dims, tracks.cuboids_dims)
