# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from omegaconf import OmegaConf

# Import to trigger loss and reduction function registration.
import nurec.libs.losses.models.loss_fns  # noqa: F401

from nurec.config.trainer import TrainerConfig
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.libs.losses.models.render_losses import UnicyclePoseLoss
from nurec.libs.losses.orchestration.config import LossItemConfig
from nurec.models.base import BaseModel
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import RigidGaussianModel
from nurec.models.tracks_calib import ElevationAwareUnicycleTracksCalib
from nurec.models.tracks_calib_data import TracksCalibResiduals


class _ElevationAwareUnicycleTracksCalibStub(ElevationAwareUnicycleTracksCalib):
    def __init__(
        self,
        consistency: torch.Tensor,
        smoothness: torch.Tensor,
        pose_regularization: torch.Tensor,
    ) -> None:
        torch.nn.Module.__init__(self)
        self._consistency = consistency
        self._smoothness = smoothness
        self._pose_regularization = pose_regularization

    def get_consistency(self) -> torch.Tensor:
        return self._consistency

    def get_smoothness(self) -> torch.Tensor:
        return self._smoothness

    def get_pose_regularization(self) -> torch.Tensor:
        return self._pose_regularization


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        max_epochs=1,
        check_val_every_n_epoch=1,
        precision="32",
        log_every_n_steps=1,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )


def _loss_config(**overrides) -> LossItemConfig:
    values = {
        "fn": "abs",
        "lambda_": 1.0,
        "reduce": {"name": "mean"},
        "lambdas": {
            "consistency": 1.0,
            "smoothness": 0.01,
            "pose": 1.0,
        },
    }
    values.update(overrides)
    return LossItemConfig.model_validate(values)


def _cuda_calib_stub() -> _ElevationAwareUnicycleTracksCalibStub:
    assert torch.cuda.is_available(), "unicycle_pose_loss_test requires its CUDA collector path"
    calib = _ElevationAwareUnicycleTracksCalibStub(
        consistency=torch.tensor([100.0], device="cuda"),
        smoothness=torch.tensor([100.0], device="cuda"),
        pose_regularization=torch.tensor([100.0], device="cuda"),
    )
    calib.register_parameter("device_marker", torch.nn.Parameter(torch.zeros((), device="cuda")))
    return calib


def _cuda_collector_residuals() -> TracksCalibResiduals:
    return TracksCalibResiduals(
        consistency=torch.tensor([1.0], device="cuda"),
        smoothness=torch.tensor([10.0], device="cuda"),
        pose_regularization=torch.tensor([4.0], device="cuda"),
    )


def _rigid_owner(tracks_calib: object) -> RigidGaussianModel:
    owner = Mock(spec=RigidGaussianModel)
    owner.tracks_calib = tracks_calib
    return owner


def _composite_with_nodes(**nodes: BaseModel) -> GaussiansComposite:
    model = Mock(spec=GaussiansComposite)
    model.gaussians_nodes = nodes
    return model


def test_each_loss_family_is_averaged_before_families_are_combined() -> None:
    calib = _ElevationAwareUnicycleTracksCalibStub(
        consistency=torch.tensor([-1.0, 1.0, -1.0, 1.0]),
        smoothness=torch.tensor([10.0]),
        pose_regularization=torch.tensor([4.0, 4.0]),
    )
    loss = UnicyclePoseLoss(_loss_config(), _trainer_config())

    loss_ret = loss(results=None, target=None, model=_rigid_owner(calib))

    assert loss_ret is not None
    # The family values are mean(abs(consistency)), 0.01 * mean(abs(smoothness)),
    # and mean(abs(pose)), independent of the number of values in each family. The
    # three family means are then summed.
    torch.testing.assert_close(loss_ret.value, torch.tensor(5.1))
    torch.testing.assert_close(loss_ret.reduced_value, torch.tensor(5.1))


def test_family_means_are_summed_from_all_unicycle_calibs() -> None:
    first = _ElevationAwareUnicycleTracksCalibStub(
        consistency=torch.tensor([1.0]),
        smoothness=torch.empty(0),
        pose_regularization=torch.tensor([1.0]),
    )
    second = _ElevationAwareUnicycleTracksCalibStub(
        consistency=torch.tensor([3.0, 5.0, 7.0]),
        smoothness=torch.empty(0),
        pose_regularization=torch.tensor([3.0]),
    )
    model = _composite_with_nodes(
        first=_rigid_owner(first),
        second=_rigid_owner(second),
        static=_rigid_owner(object()),
    )
    loss = UnicyclePoseLoss(_loss_config(), _trainer_config())

    loss_ret = loss(results=None, target=None, model=model)

    assert loss_ret is not None
    # Each calibration contributes its family means immediately: consistency contributes
    # 1 + mean(3, 5, 7) = 6, while pose contributes 1 + 3 = 4.
    torch.testing.assert_close(loss_ret.value, torch.tensor(10.0))
    torch.testing.assert_close(loss_ret.reduced_value, torch.tensor(10.0))


def test_cuda_calib_uses_its_owning_models_collector_residuals() -> None:
    calib = _cuda_calib_stub()
    compute_residuals = Mock(return_value=_cuda_collector_residuals())
    model = _rigid_owner(calib)
    model.get_tracks_calib_residuals = compute_residuals
    loss = UnicyclePoseLoss(_loss_config(), _trainer_config())

    loss_ret = loss(results=None, target=None, model=model)

    assert loss_ret is not None
    compute_residuals.assert_called_once_with()
    torch.testing.assert_close(loss_ret.value, torch.tensor(5.1, device="cuda"))


def test_cuda_child_calib_uses_its_nodes_collector_residuals() -> None:
    calib = _cuda_calib_stub()
    compute_residuals = Mock(return_value=_cuda_collector_residuals())
    child = _rigid_owner(calib)
    child.get_tracks_calib_residuals = compute_residuals
    model = _composite_with_nodes(dynamic=child)
    loss = UnicyclePoseLoss(_loss_config(), _trainer_config())

    loss_ret = loss(results=None, target=None, model=model)

    assert loss_ret is not None
    compute_residuals.assert_called_once_with()
    torch.testing.assert_close(loss_ret.value, torch.tensor(5.1, device="cuda"))


def test_non_rigid_model_is_not_treated_as_a_calibration_owner() -> None:
    calib = _cuda_calib_stub()
    loss = UnicyclePoseLoss(_loss_config(), _trainer_config())
    model = Mock(spec=BaseModel)
    model.tracks_calib = calib

    assert loss(results=None, target=None, model=model) is None


def test_outer_lambda_is_applied_after_family_averaging() -> None:
    calib = _ElevationAwareUnicycleTracksCalibStub(
        consistency=torch.tensor([1.0]),
        smoothness=torch.tensor([10.0]),
        pose_regularization=torch.tensor([4.0]),
    )
    loss = UnicyclePoseLoss(_loss_config(lambda_=2.0), _trainer_config())

    loss_ret = loss(results=None, target=None, model=_rigid_owner(calib))

    assert loss_ret is not None
    torch.testing.assert_close(loss_ret.reduced_value, torch.tensor(10.2))


def test_real_unicycle_loss_keeps_fixed_endpoints_frozen() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
    poses[:, 0, 3] = np.arange(4, dtype=np.float32)
    tracks = CuboidTracks.Factory.from_numpy(
        tracks_id=["track"],
        tracks_poses=[poses],
        tracks_timestamps_us=[np.arange(4, dtype=np.int64) * 1_000_000],
        tracks_label_class=["unknown"],
        tracks_flags=[TrackFlags.NONE],
        cuboids_dims=[np.ones(3, dtype=np.float32)],
        device=torch.device("cpu"),
    )
    calib = ElevationAwareUnicycleTracksCalib(
        config=OmegaConf.create(
            {
                "visualize": False,
                "start_global_step": -1,
                "fix_first_pose": True,
                "fix_last_pose": True,
                "pitch_slope_clamp_rad": 1.2,
                "cuboid_tracks": {
                    "forward_axis": [1.0, 0.0, 0.0],
                    "up_axis": [0.0, 0.0, 1.0],
                },
            }
        ),
        trainer_config=_trainer_config(),
        cuboid_tracks=tracks,
    )
    with torch.no_grad():
        calib.xy[0] += torch.tensor([0.2, -0.1])
        calib.xy[-1] += torch.tensor([-0.15, 0.1])
        calib.height[0] += 0.1
        calib.height[-1] -= 0.2
        calib.theta[0] += 0.1
        calib.theta[-1] -= 0.15
        calib.pitch[0] += 0.05
        calib.pitch[-1] -= 0.04

    loss = UnicyclePoseLoss(_loss_config(fn="square"), _trainer_config())
    loss_ret = loss(results=None, target=None, model=_rigid_owner(calib))

    assert loss_ret is not None
    loss_ret.reduced_value.backward()

    for param in (calib.xy, calib.height, calib.theta, calib.pitch):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        endpoint_grad = param.grad[[0, -1]]
        torch.testing.assert_close(endpoint_grad, torch.zeros_like(endpoint_grad), rtol=0, atol=0)
        assert param.grad[1:-1].abs().sum() > 0

    for param in (calib.velocity, calib.theta_velocity, calib.pitch_velocity):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0


def test_loss_function_is_applied_once() -> None:
    calib = _ElevationAwareUnicycleTracksCalibStub(
        consistency=torch.tensor([2.0]),
        smoothness=torch.empty(0),
        pose_regularization=torch.empty(0),
    )
    loss = UnicyclePoseLoss(_loss_config(fn="square"), _trainer_config())

    loss_ret = loss(results=None, target=None, model=_rigid_owner(calib))

    assert loss_ret is not None
    torch.testing.assert_close(loss_ret.value, torch.tensor(4.0))
    torch.testing.assert_close(loss_ret.reduced_value, torch.tensor(4.0))


def test_default_family_weights_are_used_when_lambdas_are_omitted() -> None:
    calib = _ElevationAwareUnicycleTracksCalibStub(
        consistency=torch.tensor([2.0]),
        smoothness=torch.tensor([10.0]),
        pose_regularization=torch.tensor([4.0]),
    )
    loss = UnicyclePoseLoss(_loss_config(lambdas=None), _trainer_config())

    loss_ret = loss(results=None, target=None, model=_rigid_owner(calib))

    assert loss_ret is not None
    torch.testing.assert_close(loss_ret.value, torch.tensor(1.5))
    torch.testing.assert_close(loss_ret.reduced_value, torch.tensor(1.5))


def test_rejects_non_mean_reducer() -> None:
    with pytest.raises(AssertionError, match="Invalid reduce name for UnicyclePoseLoss"):
        UnicyclePoseLoss(_loss_config(reduce={"name": "sum"}), _trainer_config())


def test_returns_none_without_unicycle_calibs() -> None:
    loss = UnicyclePoseLoss(_loss_config(), _trainer_config())

    assert loss(results=None, target=None, model=_composite_with_nodes()) is None
