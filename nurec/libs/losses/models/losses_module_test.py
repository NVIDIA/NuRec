# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent loss producers remain the reference for accumulator parity."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from nurec.config.trainer import TrainerConfig
from nurec.libs.losses.models.fused_loss_inputs import FUSED_LOSS_NAMES
from nurec.libs.losses.orchestration.config import LossConfig
from nurec.libs.losses.orchestration.loss_aggregator import LossAggregator
from nurec.utils.batch import CameraFrameLabels, DataAndRenderingBatch, DataBatch, FrameMeta, LidarFrameLabels
from nurec.utils.types import ExtraSignal, GaussiansCompositeReturn, GaussiansRenderReturn, RayFlags


MINIMAL_LOSS_CONFIGS: dict[str, dict] = {
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


@pytest.mark.parametrize("semantic_weights", [False, True])
def test_ssim_uses_independent_producer_on_cpu(semantic_weights):
    item = dict(MINIMAL_LOSS_CONFIGS["ssim"])
    if semantic_weights:
        item.update(mask_semantic_classes=["ROAD"], semantic_lambdas=[2.0])
    aggregator = LossAggregator(LossConfig.model_validate({"ssim": item}), MockTrainerConfig())
    assert [loss.name for loss in aggregator.losses] == ["ssim_ssim_mean"]
    assert aggregator.configured_losses is aggregator.losses
    assert not hasattr(aggregator, "in_cuda")


def test_all_configured_losses_construct_as_independent_producers_on_cpu():
    config = LossConfig.model_validate(MINIMAL_LOSS_CONFIGS)
    aggregator = LossAggregator(config, MockTrainerConfig())
    assert [loss.name for loss in aggregator.losses] == [
        f"{name}_{item.fn}_{item.reduce.name}" for name, item in config.items()
    ]
    for loss in aggregator.losses:
        if loss.name in ("gaussian_scale_abs_mean", "gaussian_density_abs_mean"):
            assert loss.layer_lambdas == {}


def test_minimal_configs_cover_accumulator_reference_producers():
    names = {f"{name}_{item['fn']}_{item['reduce']['name']}" for name, item in MINIMAL_LOSS_CONFIGS.items()}
    assert set(FUSED_LOSS_NAMES) <= names


def test_independent_producers_receive_initialization_and_training_updates():
    config = LossConfig.model_validate({name: MINIMAL_LOSS_CONFIGS[name] for name in ("rgb", "ssim")})
    aggregator = LossAggregator(config, MockTrainerConfig())
    dataset, system = object(), object()
    initializers, updates = [], []
    for index, loss in enumerate(aggregator.losses):
        loss.initialize = mock.Mock()
        loss.update_step_train_batch_start = mock.Mock(return_value={loss.name: torch.tensor(float(index))})
        initializers.append(loss.initialize)
        updates.append(loss.update_step_train_batch_start)
    aggregator.initialize(dataset)
    params = aggregator.update_step_train_batch_start(2, 7, system, batch="current")
    assert set(params) == {loss.name for loss in aggregator.losses}
    for initialize, update in zip(initializers, updates, strict=True):
        initialize.assert_called_once_with(dataset)
        update.assert_called_once_with(2, 7, system, batch="current")


def _independent_inputs():
    camera_flags = torch.tensor(
        [
            RayFlags.RGB_LABEL.value,
            RayFlags.RGB_LABEL.value | RayFlags.INVALID.value,
            RayFlags.RGB_LABEL.value | RayFlags.SKY_SEMANTIC.value,
            RayFlags.RGB_LABEL.value | RayFlags.HARMONIZED.value,
            RayFlags.RGB_LABEL.value | RayFlags.SYNTHETIC.value,
            0,
        ],
        dtype=torch.int32,
    ).reshape(1, 2, 3, 1)
    lidar_flags = torch.tensor(
        [0, RayFlags.INVALID.value, RayFlags.DROPPED.value, RayFlags.SKY_SEMANTIC.value, 0, 0],
        dtype=torch.int32,
    ).reshape(1, 2, 3, 1)
    tensors = {
        "rgb": torch.linspace(-0.4, 1.2, 18).reshape(6, 3).requires_grad_(),
        "opacity": torch.tensor([0.2, 0.4, 0.6, 0.8, 0.1, 0.3], requires_grad=True),
        "distance": torch.tensor([1.4, 2.6, 0.8, 4.2, 5.5, 6.3], requires_grad=True),
        "intensity": torch.linspace(0.1, 0.9, 6).reshape(6, 1).requires_grad_(),
        "raydrop": torch.linspace(0.8, 0.2, 6).reshape(6, 1).requires_grad_(),
        "lidar_opacity": torch.tensor([0.7, 0.4, 0.2, 0.9, 0.6, 0.3], requires_grad=True),
        "semantics": torch.linspace(-1.2, 1.7, 18).reshape(6, 3).requires_grad_(),
        "deformation": torch.tensor([[1.2, -0.4, 0.8], [-0.3, 1.1, -1.5]], requires_grad=True),
        "deform_mask": torch.tensor([[1.0], [0.25]], requires_grad=True),
    }
    camera_labels = CameraFrameLabels(
        flags=camera_flags,
        rgb=torch.linspace(0.9, -0.2, 18).reshape(1, 2, 3, 3),
        semantic=torch.tensor([0, 255, 255, 255, 1, 255], dtype=torch.uint8).reshape(1, 2, 3, 1),
    )
    lidar_labels = LidarFrameLabels(
        flags=lidar_flags,
        distance=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(1, 2, 3, 1),
        intensity=torch.linspace(0.8, 0.2, 6).reshape(1, 2, 3, 1),
        raydrop=torch.linspace(0.1, 0.7, 6).reshape(1, 2, 3, 1),
    )
    meta = [FrameMeta(unique_sensor_idx=0, unique_frame_idx=0, training_frame_idx=0)]
    target = DataAndRenderingBatch(
        data=DataBatch(
            idx=0,
            worker_id=None,
            sequence_id=["reference"],
            camera=DataBatch.Camera(meta=meta, labels=camera_labels),
            lidar=DataBatch.Lidar(meta=meta, labels=lidar_labels),
        )
    )
    results = GaussiansCompositeReturn(
        rendered_cam=GaussiansRenderReturn(
            rgb=tensors["rgb"],
            opacity=tensors["opacity"],
            distance=torch.zeros(6),
            extra_ray_signals=ExtraSignal(semantic_logits=tensors["semantics"]),
        ),
        rendered_lidar=GaussiansRenderReturn(
            rgb=torch.zeros(6, 3),
            opacity=tensors["lidar_opacity"],
            distance=tensors["distance"],
            extra_ray_signals=ExtraSignal(intensity=tensors["intensity"], raydrop=tensors["raydrop"]),
        ),
        deform_smoothness=tensors["deformation"],
        deform_smoothness_mask=tensors["deform_mask"],
    )
    return tensors, results, target


def test_independent_masked_losses_match_original_formulas_and_gradients():
    names = ("rgb", "background", "lidar", "background_lidar", "intensity", "raydrop", "semantic", "deform_smoothness")
    config = LossConfig.model_validate(
        {name: dict(MINIMAL_LOSS_CONFIGS[name], lambda_=0.5 + index / 4) for index, name in enumerate(names)}
    )
    tensors, results, target = _independent_inputs()
    aggregator = LossAggregator(config, MockTrainerConfig())
    result = aggregator(step=0, model=SimpleNamespace(device=torch.device("cpu")), results=results, target=target)

    # Explicit masks compare public producers against the original formulas
    # independently of the production mask helpers.
    rgb_rows = torch.tensor([0, 2, 3, 4])
    background_rows = torch.tensor([0, 2, 5])
    lidar_rows = torch.tensor([0, 3, 4, 5])
    raydrop_rows = torch.tensor([0, 2, 3, 4, 5])
    semantic_rows = torch.tensor([0, 4])
    camera_labels, lidar_labels = target.data.camera.labels, target.data.lidar.labels
    expected = {
        "rgb_l1_mean": (tensors["rgb"][rgb_rows] - camera_labels.rgb.reshape(6, 3)[rgb_rows]).abs().mean(),
        "background_mse_mean": (tensors["opacity"][background_rows] - torch.tensor([1.0, 0.0, 1.0])).square().mean(),
        "lidar_l1_mean": (tensors["distance"][lidar_rows] - lidar_labels.distance.reshape(6)[lidar_rows]).abs().mean(),
        "background_lidar_mse_mean": (tensors["lidar_opacity"][lidar_rows] - torch.tensor([1.0, 0.0, 1.0, 1.0]))
        .square()
        .mean(),
        "intensity_mse_mean": (
            tensors["intensity"].reshape(6)[lidar_rows] - lidar_labels.intensity.reshape(6)[lidar_rows]
        )
        .square()
        .mean(),
        "raydrop_mse_mean": (
            tensors["raydrop"].reshape(6)[raydrop_rows] - lidar_labels.raydrop.reshape(6)[raydrop_rows]
        )
        .square()
        .mean(),
        "semantic_cross_entropy_mean": F.cross_entropy(
            tensors["semantics"][semantic_rows], camera_labels.semantic.reshape(6)[semantic_rows].long()
        ),
        "deform_smoothness_abs_mean": (tensors["deformation"].abs() * tensors["deform_mask"]).sum()
        / tensors["deform_mask"].sum(),
    }
    assert set(result.loss_returns) == set(expected)
    weighted = []
    for name, raw in expected.items():
        loss = result.loss_returns[name]
        weighted.append(raw * loss.lambda_)
        torch.testing.assert_close(loss.reduced_value, weighted[-1])
    expected_total = torch.stack(weighted).sum()
    torch.testing.assert_close(result.total_value, expected_total)
    actual_gradients = torch.autograd.grad(result.total_value, tuple(tensors.values()))
    expected_gradients = torch.autograd.grad(expected_total, tuple(tensors.values()))
    for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference)


def test_independent_losses_obey_schedule_and_omit_missing_modalities():
    config = LossConfig.model_validate(
        {
            "rgb": dict(MINIMAL_LOSS_CONFIGS["rgb"], start_step=2),
            "lidar": dict(MINIMAL_LOSS_CONFIGS["lidar"], lambda_=0.0),
            "deform_smoothness": MINIMAL_LOSS_CONFIGS["deform_smoothness"],
        }
    )
    _, results, target = _independent_inputs()
    aggregator = LossAggregator(config, MockTrainerConfig())
    model = SimpleNamespace(device=torch.device("cpu"))
    assert set(aggregator(step=1, model=model, results=results, target=target).loss_returns) == {
        "deform_smoothness_abs_mean"
    }
    assert set(aggregator(step=2, model=model, results=results, target=target).loss_returns) == {
        "rgb_l1_mean",
        "deform_smoothness_abs_mean",
    }
    target.data.camera = None
    assert set(aggregator(step=2, model=model, results=results, target=target).loss_returns) == {
        "deform_smoothness_abs_mean"
    }
