# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

from collections import OrderedDict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from python.runfiles import runfiles

from nurec.config.model import CalibConfig
from nurec.config.nurec import NuRecConfig
from nurec.config.parse import parse_typed_config
from nurec.config.trainer import TrainerConfig
from nurec.datasets import make as make_dataset
from nurec.datasets.ncore import NCORETrainDataset
from nurec.datasets.summary import DataSourceSummary
from nurec.models.calib import BaseCalib, FreePoseCalib, RigCalib
from nurec.models.rig_spline import CameraExtrinsicResiduals, PackedRigSpline, residual_vectors_to_transforms
from nurec.utils.batch import CameraFreePoseViewGeometry, DataBatch, FrameMeta, RectSubsampled
from nurec.utils.misc import unpack_optional
from nurec.utils.types import FrameConversion, RigTrajectories


RUNFILES = runfiles.Create()


@pytest.fixture
def small_clipgt_dataset_path() -> Path:
    path = Path(
        RUNFILES.Rlocation("test_data_ncore_v4/clipgt-9048443e-c482-4228-8326-5b3dff3be711.json"),
    )
    if not path.exists():
        raise AssertionError(
            f"Test dataset not found. This is an issue with your filesystem/test suite, not the code under test. Missing {path=}"
        )
    return path


_TESTDATA_CONFIG_DIR = Path(__file__).resolve().parent / "testdata"


@pytest.fixture
def small_clipgt_dataset(small_clipgt_dataset_path: Path) -> tuple[NCORETrainDataset, NuRecConfig]:
    config_name = "apps/test_app_config"

    # Ensure the path is a *quoted* string for Hydra compatibility with bazel's `~`-separated paths
    small_clipgt_dataset_path_str = '"{0}"'.format(str(small_clipgt_dataset_path))

    hydra_args = [
        f"dataset.path={small_clipgt_dataset_path_str}",
        "dataset.name=ncore",
        "dataset.n_samples_per_epoch=2",  # Using a low number of iterations speeds the test up
        "dataset.camera_ids=[camera_front_wide_120fov]",
        "dataset.lidar_ids=[lidar_gt_top_p128]",
        "dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2",  # test data is 480x270
        "dataset.show_progress_bars=False",
        "dataset.n_val_image_subsample=2",
        "out_dir=/doesnt/matter",  # required field
    ]

    config = parse_typed_config(str(_TESTDATA_CONFIG_DIR / config_name), hydra_args)
    train_dataset = cast(NCORETrainDataset, make_dataset(config.dataset.name, config, split="train"))
    return train_dataset, config


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        max_epochs=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        precision=32,
    )


def _free_pose_calib_config() -> CalibConfig:
    return CalibConfig.model_validate(
        {
            "name": "free-pose-calib",
            "start_global_step": 0,
            "enable_torch_compile": False,
            "lidar": {"enabled": True},
            "camera": {"enabled": True},
            "optimizer": {
                "name": "fused_adam",
                "args": {
                    "lr": 0.0001,
                    "eps": 1e-15,
                    "betas": [0.9, 0.99],
                    "weight_decay": 0.000001,
                },
            },
        }
    )


def _make_free_pose_calib(
    train_dataset: NCORETrainDataset,
    trainer_config: TrainerConfig,
) -> FreePoseCalib:
    calib_config = _free_pose_calib_config()
    calib = BaseCalib.factory(
        name=calib_config.name,
        config=calib_config,
        trainer_config=trainer_config,
        datasource=DataSourceSummary.from_datasource(train_dataset.datasource),
    )
    assert isinstance(calib, FreePoseCalib)
    return calib


def test_v2_free_pose_calib(small_clipgt_dataset: tuple[NCORETrainDataset, NuRecConfig]) -> None:
    train_dataset, config = small_clipgt_dataset

    # Create a DataBatch from the dataset.
    camera_id = "camera_front_wide_120fov"
    camera_frame_idx = 2
    sampled_pixels = RectSubsampled(
        width=240, height=135, subsample_factor=2.0, original_width=480, original_height=270
    )
    camera_data_batch: DataBatch.Camera = train_dataset.get_camera_data_batch(
        sensor_id=camera_id,
        sensor_frame_idx=camera_frame_idx,
        sampled_pixels=sampled_pixels,
    )
    lidar_id = "lidar_gt_top_p128"
    lidar_frame_idx = 2
    lidar_data_batch: DataBatch.Lidar = train_dataset.get_lidar_data_batch(
        sensor_id=lidar_id,
        sensor_frame_idx=lidar_frame_idx,
    )
    data_batch = DataBatch(camera=camera_data_batch, lidar=lidar_data_batch)

    calib = _make_free_pose_calib(train_dataset, config.trainer)

    # Only CUDA version is supported: put model and data on the same device so every tensor is on GPU
    device = torch.device("cuda")
    calib = calib.to(device)
    data_batch = data_batch.to(device=device)
    rendering_batch = calib(data_batch)
    _ = calib.configure_optimizers()  # Test optimizer setup works

    # Test the autograd graph for the pose outputs
    loss = (
        unpack_optional(rendering_batch.camera).poses_tquat_startend.sum()
        + unpack_optional(rendering_batch.lidar).poses_tquat_startend.sum()
    )
    grads = torch.autograd.grad(loss, calib.params)
    for grad in grads:
        assert (grad is not None) and (grad != 0).any()

    # Re-run forward pass to make sure the graph is not retained
    rendering_batch = calib(data_batch)

    # Test the autograd graph for the ray outputs
    loss = unpack_optional(rendering_batch.camera).rays.sum() + unpack_optional(rendering_batch.lidar).rays.sum()
    grads = torch.autograd.grad(loss, calib.params)
    for grad in grads:
        assert (grad is not None) and (grad != 0).any()


def test_active_tile_prewarm_plan_matches_label_loaded_free_pose_calibration(
    small_clipgt_dataset: tuple[NCORETrainDataset, NuRecConfig],
) -> None:
    train_dataset, config = small_clipgt_dataset
    plan = train_dataset.get_active_tile_mapping_prewarm_plan()
    assert plan.skip_reason is None
    (entry,) = plan.entries
    camera_id = "camera_front_wide_120fov"
    lidar_id = "lidar_gt_top_p128"

    sampled_pixels = RectSubsampled(
        width=240,
        height=135,
        subsample_factor=2.0,
        original_width=480,
        original_height=270,
    )
    reference_batch = DataBatch(
        camera=train_dataset.get_camera_data_batch(
            sensor_id=camera_id,
            sensor_frame_idx=9,
            sampled_pixels=sampled_pixels,
        ),
        lidar=train_dataset.get_lidar_data_batch(
            sensor_id=lidar_id,
            sensor_frame_idx=4,
        ),
    )
    planned_camera = unpack_optional(entry.data_batch.camera)
    planned_lidar = unpack_optional(entry.data_batch.lidar)
    reference_camera = unpack_optional(reference_batch.camera)
    reference_lidar = unpack_optional(reference_batch.lidar)
    for planned_meta, reference_meta in (
        (planned_camera.meta[0], reference_camera.meta[0]),
        (planned_lidar.meta[0], reference_lidar.meta[0]),
    ):
        assert planned_meta.unique_sensor_idx == reference_meta.unique_sensor_idx
        assert planned_meta.unique_frame_idx == reference_meta.unique_frame_idx
        assert planned_meta.training_frame_idx == reference_meta.training_frame_idx
        assert planned_meta.subsample == reference_meta.subsample
    assert planned_camera.labels.rgb is None
    assert planned_lidar.labels.distance is None
    assert reference_camera.labels.rgb is not None
    assert reference_lidar.labels.distance is not None

    sampled_training_batch = train_dataset[0].data
    sampled_camera = unpack_optional(sampled_training_batch.camera)
    assert sampled_camera.meta[0].subsample == planned_camera.meta[0].subsample

    device_idx = torch.cuda.current_device()
    device = torch.device("cuda", device_idx)
    with torch.random.fork_rng(devices=[device_idx]):
        torch.random.set_rng_state(torch.Generator(device="cpu").manual_seed(0).get_state())
        torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(0).get_state(), device=device)
        calib = _make_free_pose_calib(train_dataset, config.trainer).to(device)
        with torch.no_grad():
            planned_rendering = calib(entry.data_batch.to(device), skip_calib=False)
            reference_rendering = calib(reference_batch.to(device), skip_calib=False)
            sampled_rendering = calib(sampled_training_batch.to(device), skip_calib=False)

    planned_rendering_camera = unpack_optional(planned_rendering.camera)
    planned_rendering_lidar = unpack_optional(planned_rendering.lidar)
    reference_rendering_camera = unpack_optional(reference_rendering.camera)
    reference_rendering_lidar = unpack_optional(reference_rendering.lidar)
    sampled_rendering_camera = unpack_optional(sampled_rendering.camera)
    sampled_rendering_lidar = unpack_optional(sampled_rendering.lidar)
    assert (
        planned_rendering_camera.rays.shape
        == reference_rendering_camera.rays.shape
        == sampled_rendering_camera.rays.shape
        == (1, 135, 240, 6)
    )
    assert (
        planned_rendering_lidar.rays.shape
        == reference_rendering_lidar.rays.shape
        == sampled_rendering_lidar.rays.shape
        == (1, 128, 3600, 6)
    )
    for planned, reference in (
        (planned_rendering_camera, reference_rendering_camera),
        (planned_rendering_lidar, reference_rendering_lidar),
    ):
        torch.testing.assert_close(planned.rays, reference.rays, rtol=0, atol=0)
        torch.testing.assert_close(planned.poses_tquat_startend, reference.poses_tquat_startend, rtol=0, atol=0)
        torch.testing.assert_close(planned.timestamps_startend_us, reference.timestamps_startend_us, rtol=0, atol=0)
        torch.testing.assert_close(
            planned.timestamps_startend_us_cpu,
            reference.timestamps_startend_us_cpu,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            unpack_optional(planned.rays_timestamps_us),
            unpack_optional(reference.rays_timestamps_us),
            rtol=0,
            atol=0,
        )
        assert len(planned.sensor_model_parameters) == len(reference.sensor_model_parameters) == 1
        planned_sensor_model = planned.sensor_model_parameters[0]
        reference_sensor_model = reference.sensor_model_parameters[0]
        assert planned_sensor_model is not None
        assert reference_sensor_model is not None
        assert type(planned_sensor_model) is type(reference_sensor_model)
        assert planned_sensor_model.to_json() == reference_sensor_model.to_json()
    torch.testing.assert_close(
        planned_rendering_camera.ray_footprints,
        reference_rendering_camera.ray_footprints,
        rtol=0,
        atol=0,
    )


def _make_ply_viewer_style_datasource() -> DataSourceSummary:
    """Mirror the dummy `DataSourceSummary` the PLY viewer builds when no real datasource is available."""
    state = DataSourceSummary.State(
        n_frames_per_camera=np.array([1], dtype=np.int32),
        n_frames_per_lidar=np.zeros(0, dtype=np.int32),
        sequence_tracks_all=None,
        sequence_tracks_dynamic=None,
        rig_trajectories=None,
        xform_matrices=np.zeros((0, 4), dtype=np.float32),
        aabb_blb=np.zeros((1, 3), dtype=np.float32),
        aabb_trf=np.ones((1, 3), dtype=np.float32),
    )
    return DataSourceSummary(state=state, datasource=None)


def test_skip_calib_factory_without_rig_trajectories() -> None:
    """The PLY viewer constructs a `GaussiansComposite` with `skip-calib` and no rig_trajectories.
    Regression for the crash where `BaseCalib.factory` rejected this case and the assertion in
    `FreePoseCalib.__init__` required at least one view geometry."""
    datasource = _make_ply_viewer_style_datasource()

    calib_config = CalibConfig.model_validate(
        {
            "name": "skip-calib",
            "lidar": {"enabled": False},
            "camera": {"enabled": False},
        }
    )

    calib = BaseCalib.factory(
        name=calib_config.name,
        config=calib_config,
        trainer_config=_trainer_config(),
        datasource=datasource,
    )
    assert isinstance(calib, FreePoseCalib)
    assert calib.enabled is False
    assert calib.camera_view_geometry is None
    assert calib.lidar_view_geometry is None


def test_calib_interp_with_rig(small_clipgt_dataset: tuple[NCORETrainDataset, NuRecConfig]) -> None:
    train_dataset, _ = small_clipgt_dataset

    rig_trajectories = train_dataset.datasource.get_rig_trajectories()

    # For the test case (and most of the cases), we expect `interp_with_rig=True` to be the same as `interp_with_rig=False`
    # I.e. They contain the same poses.
    view_geometry1 = CameraFreePoseViewGeometry.from_rig_trajectories(rig_trajectories, interp_with_rig=True)
    view_geometry2 = CameraFreePoseViewGeometry.from_rig_trajectories(rig_trajectories, interp_with_rig=False)

    torch.testing.assert_close(
        view_geometry1.T_sensor_world_startend_allviews,
        view_geometry2.T_sensor_world_startend_allviews,
    )


def _make_rig_calib_for_unit_test() -> RigCalib:
    calib = RigCalib.__new__(RigCalib)
    torch.nn.Module.__init__(calib)
    calib.config = CalibConfig.model_validate(
        {
            "name": "rig-calib",
            "camera": {"enabled": True},
            "lidar": {"enabled": True},
            "rig_trajectory": {
                "enabled": True,
                "spline_degree": 1,
                "knot_spacing_seconds": 1.0,
                "max_translation_m": 1.0,
                "max_rotation_degrees": 45.0,
            },
            "camera_extrinsics": {
                "enabled": True,
                "max_translation_m": 1.0,
                "max_rotation_degrees": 45.0,
            },
        }
    )
    calib.nurec_scale = 1.0
    calib.rig_spline = PackedRigSpline(
        torch.tensor([1_000_000], dtype=torch.int64),
        torch.tensor([2_000_000], dtype=torch.int64),
        degree=1,
        knot_spacing_seconds=1.0,
        skip_first_control=False,
        max_translation_m=1.0,
        max_rotation_rad=math.radians(45.0),
    )
    calib.camera_extrinsics = CameraExtrinsicResiduals(
        torch.tensor([0, 0], dtype=torch.int64),
        num_trajectories=1,
        max_translation_m=1.0,
        max_rotation_rad=math.radians(45.0),
    )
    sensor_to_rig = residual_vectors_to_transforms(
        torch.tensor(
            [
                [0.4, -0.2, 0.1, 0.0, 0.0, math.radians(30.0)],
                [-0.3, 0.1, 0.2, math.radians(10.0), 0.0, 0.0],
            ]
        )
    )
    calib.camera_trajectory_indices = torch.nn.Buffer(torch.tensor([0, 0], dtype=torch.int64), persistent=False)
    calib.camera_sensor_to_rig = torch.nn.Buffer(sensor_to_rig, persistent=False)
    calib.lidar_trajectory_indices = torch.nn.Buffer(torch.tensor([0], dtype=torch.int64), persistent=False)
    calib.lidar_sensor_to_rig = torch.nn.Buffer(sensor_to_rig[:1], persistent=False)
    calib.camera_extrinsic_indices = torch.nn.Buffer(torch.tensor([0, 1], dtype=torch.int64), persistent=False)
    calib._camera_sensor_id_to_index = {"camera_0": 0, "camera_1": 1}
    calib._lidar_sensor_id_to_index = {"lidar_0": 0}
    calib._camera_ids_by_extrinsic_index = ["camera_0", "camera_1"]
    return calib


def _rig_components(calib: RigCalib) -> tuple[PackedRigSpline, CameraExtrinsicResiduals]:
    assert calib.rig_spline is not None
    assert calib.camera_extrinsics is not None
    return calib.rig_spline, calib.camera_extrinsics


def test_rig_calib_forward_skip_gradients_and_checkpoint(
    small_clipgt_dataset: tuple[NCORETrainDataset, NuRecConfig],
) -> None:
    train_dataset, config = small_clipgt_dataset
    camera_data_batch = train_dataset.get_camera_data_batch(
        sensor_id="camera_front_wide_120fov",
        sensor_frame_idx=2,
        sampled_pixels=RectSubsampled(
            width=200,
            height=100,
            subsample_factor=2.0,
            original_width=480,
            original_height=270,
            j=20,
        ),
    )
    lidar_data_batch = train_dataset.get_lidar_data_batch(
        sensor_id="lidar_gt_top_p128",
        sensor_frame_idx=2,
    )
    data_batch = DataBatch(camera=camera_data_batch, lidar=lidar_data_batch)
    calib_config = CalibConfig.model_validate(
        {
            "name": "rig-calib",
            "camera": {"enabled": True},
            "lidar": {"enabled": True},
            "rig_trajectory": {
                "enabled": True,
                "spline_degree": 3,
                "knot_spacing_seconds": 0.1,
                "max_translation_m": 0.25,
                "max_rotation_degrees": 3.0,
            },
            "camera_extrinsics": {"enabled": False},
        }
    )
    datasource_summary = DataSourceSummary.from_datasource(train_dataset.datasource)
    calib = BaseCalib.factory(
        name=calib_config.name,
        config=calib_config,
        trainer_config=config.trainer,
        datasource=datasource_summary,
    )
    assert isinstance(calib, RigCalib)
    assert calib.rig_spline is not None

    device = torch.device("cuda")
    calib = calib.to(device)
    rig_spline = calib.rig_spline
    assert rig_spline is not None
    data_batch = data_batch.to(device=device)
    raw_camera = calib.camera_view_geometry.to_rendering_data(
        unpack_optional(data_batch.camera), cache_sensor_params=True, skip_calib=True
    )
    raw_lidar = calib.lidar_view_geometry.to_rendering_data(
        unpack_optional(data_batch.lidar), cache_sensor_params=True, skip_calib=True
    )
    skipped = calib(data_batch, skip_calib=True)
    assert torch.equal(unpack_optional(skipped.camera).poses_tquat_startend, raw_camera.poses_tquat_startend)
    assert torch.equal(unpack_optional(skipped.camera).rays, raw_camera.rays)
    assert torch.equal(unpack_optional(skipped.lidar).poses_tquat_startend, raw_lidar.poses_tquat_startend)
    assert torch.equal(unpack_optional(skipped.lidar).rays, raw_lidar.rays)

    with torch.no_grad():
        ramp = torch.linspace(0.0, 1.0, rig_spline.controls.shape[0], device=device)
        rig_spline.controls[:, 0] = 0.05 * ramp.square()
        rig_spline.controls[:, 5] = math.radians(0.5) * ramp.square()
    rendering = calib(data_batch)
    assert not torch.equal(unpack_optional(rendering.camera).poses_tquat_startend, raw_camera.poses_tquat_startend)
    assert not torch.equal(unpack_optional(rendering.lidar).poses_tquat_startend, raw_lidar.poses_tquat_startend)
    assert not torch.equal(unpack_optional(rendering.camera).rays, raw_camera.rays)
    assert not torch.equal(unpack_optional(rendering.lidar).rays, raw_lidar.rays)
    pose_and_ray_loss = (
        unpack_optional(rendering.camera).poses_tquat_startend.sum()
        + unpack_optional(rendering.lidar).poses_tquat_startend.sum()
        + unpack_optional(rendering.camera).rays.sum()
        + unpack_optional(rendering.lidar).rays.sum()
    )
    (gradient,) = torch.autograd.grad(pose_and_ray_loss, calib.params)
    assert torch.isfinite(gradient).all()
    assert (gradient != 0).any()

    clone = BaseCalib.factory(
        name=calib_config.name,
        config=calib_config,
        trainer_config=config.trainer,
        datasource=datasource_summary,
    ).to(device)
    clone.load_state_dict(calib.state_dict())
    assert isinstance(clone, RigCalib)
    assert clone.rig_spline is not None
    torch.testing.assert_close(clone.rig_spline.controls, rig_spline.controls)


def test_rig_calib_transform_order_gradients_skip_and_checkpoint() -> None:
    calib = _make_rig_calib_for_unit_test()
    rig_spline, camera_extrinsics = _rig_components(calib)
    rig_residual = torch.tensor([0.08, -0.03, 0.02, 0.04, -0.02, 0.03])
    extrinsic_residual = torch.tensor([-0.02, 0.05, 0.01, -0.03, 0.01, 0.02])
    with torch.no_grad():
        rig_spline.controls[:] = rig_residual
        camera_extrinsics.residuals[0] = extrinsic_residual
        camera_extrinsics.residuals[1] = -extrinsic_residual

    assert list(calib.state_dict()) == ["rig_spline.controls", "camera_extrinsics.residuals"]
    timestamps = torch.tensor([1_200_000, 1_800_000], dtype=torch.int64)
    base = residual_vectors_to_transforms(
        torch.tensor(
            [
                [0.1, 0.2, -0.1, 0.02, 0.01, -0.03],
                [-0.2, 0.1, 0.3, -0.01, 0.03, 0.02],
            ]
        )
    )
    rig_transform = residual_vectors_to_transforms(rig_residual).expand(2, 4, 4)
    extrinsic_transform = residual_vectors_to_transforms(extrinsic_residual)
    sensor_to_rig = calib.camera_sensor_to_rig[0]
    expected_rig_correction = torch.linalg.inv(sensor_to_rig) @ rig_transform @ sensor_to_rig

    spline_only = calib._correct_sensor_poses(
        base,
        timestamps,
        0,
        sensor_type="camera",
        apply_rig_trajectory=True,
        apply_camera_extrinsics=False,
    )
    extrinsics_only = calib._correct_sensor_poses(
        base,
        timestamps,
        0,
        sensor_type="camera",
        apply_rig_trajectory=False,
        apply_camera_extrinsics=True,
    )
    combined = calib._correct_sensor_poses(
        base,
        timestamps,
        0,
        sensor_type="camera",
        apply_rig_trajectory=True,
        apply_camera_extrinsics=True,
    )
    torch.testing.assert_close(spline_only, base @ expected_rig_correction)
    torch.testing.assert_close(extrinsics_only, base @ extrinsic_transform)
    torch.testing.assert_close(combined, base @ expected_rig_correction @ extrinsic_transform)
    assert not torch.allclose(combined, base @ extrinsic_transform @ expected_rig_correction)

    rig_gradient, extrinsic_gradient = torch.autograd.grad(
        combined.square().sum(),
        [rig_spline.controls, camera_extrinsics.residuals],
        retain_graph=True,
    )
    assert (rig_gradient != 0.0).any()
    assert (extrinsic_gradient != 0.0).any()

    skipped = calib._pose_correction("camera", skip_calib=True)(
        base,
        timestamps,
        FrameMeta(unique_sensor_idx=0, unique_frame_idx=0, training_frame_idx=0),
    )
    assert torch.equal(skipped, base)
    skipped_gradients = torch.autograd.grad(
        skipped.sum(),
        [rig_spline.controls, camera_extrinsics.residuals],
    )
    assert all(torch.equal(gradient, torch.zeros_like(gradient)) for gradient in skipped_gradients)

    clone = _make_rig_calib_for_unit_test()
    clone.load_state_dict(calib.state_dict())
    clone_rig_spline, clone_camera_extrinsics = _rig_components(clone)
    torch.testing.assert_close(clone_rig_spline.controls, rig_spline.controls)
    torch.testing.assert_close(clone_camera_extrinsics.residuals, camera_extrinsics.residuals)


def _make_rig_trajectories_for_baking(target_scale: float) -> RigTrajectories:
    timestamps = torch.tensor([1_000_000, 2_000_000], dtype=torch.int64)
    camera_timestamps = timestamps.repeat(2, 1)
    coarse_poses = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    coarse_poses[1, 0, 3] = 1.0
    camera_frame_poses = torch.eye(4, dtype=torch.float64).repeat(2, 2, 1, 1)
    return RigTrajectories(
        T_world_base=torch.eye(4, dtype=torch.float64),
        world_to_nurec=FrameConversion.from_origin_scale_axis(
            np.zeros(3, dtype=np.float64),
            target_scale,
            [0, 1, 2],
            dtype=np.float64,
        ),
        rig_trajectories=[
            RigTrajectories.RigTrajectory(
                sequence_id="sequence",
                rig_bbox=None,
                cameras_linear_start_frame_indices={"camera_0": 0, "camera_1": 2},
                lidars_linear_start_frame_indices={"lidar_0": 0},
                cameras_frame_timestamps_us={
                    "camera_0": camera_timestamps.clone(),
                    "camera_1": camera_timestamps.clone(),
                },
                lidars_frame_timestamps_us={"lidar_0": camera_timestamps.clone()},
                T_rig_worlds=coarse_poses,
                T_rig_world_timestamps_us=timestamps,
                cameras_frame_T_rig_worlds={
                    "camera_0": camera_frame_poses.clone(),
                    "camera_1": camera_frame_poses.clone(),
                },
            )
        ],
        camera_calibrations=OrderedDict(
            {
                "camera_0": RigTrajectories.CameraCalibration(
                    sequence_id="sequence",
                    logical_sensor_name="camera_0",
                    T_sensor_rig=torch.eye(4),
                    camera_model_parameters=cast(Any, "intrinsics_0"),
                    unique_sensor_idx=0,
                ),
                "camera_1": RigTrajectories.CameraCalibration(
                    sequence_id="sequence",
                    logical_sensor_name="camera_1",
                    T_sensor_rig=torch.eye(4),
                    camera_model_parameters=cast(Any, "intrinsics_1"),
                    unique_sensor_idx=1,
                ),
            }
        ),
        lidar_calibrations=OrderedDict(
            {
                "lidar_0": RigTrajectories.LidarCalibration(
                    sequence_id="sequence",
                    logical_sensor_name="lidar_0",
                    T_sensor_rig=torch.eye(4),
                    lidar_model_parameters=None,
                    unique_sensor_idx=0,
                )
            }
        ),
    )


def test_rig_calib_bakes_detached_copy_without_mutating_source_and_refreshes() -> None:
    calib = _make_rig_calib_for_unit_test()
    calib.nurec_scale = 2.5
    rig_spline, camera_extrinsics = _rig_components(calib)
    source = _make_rig_trajectories_for_baking(calib.nurec_scale)
    source.camera_calibrations["camera_0"].T_sensor_rig = calib.camera_sensor_to_rig[0].clone()
    source.camera_calibrations["camera_1"].T_sensor_rig = calib.camera_sensor_to_rig[1].clone()
    source.lidar_calibrations["lidar_0"].T_sensor_rig = calib.lidar_sensor_to_rig[0].clone()
    source_trajectory = source.rig_trajectories[0]
    source_camera_frame_poses = unpack_optional(source_trajectory.cameras_frame_T_rig_worlds)
    source_coarse = source_trajectory.T_rig_worlds.clone()
    source_camera_frames = source_camera_frame_poses["camera_0"].clone()
    source_camera_extrinsic = source.camera_calibrations["camera_0"].T_sensor_rig.clone()
    rig_residual = torch.tensor([0.08, -0.03, 0.02, 0.04, -0.02, 0.03])
    extrinsic_residual = torch.tensor([-0.02, 0.05, 0.01, -0.03, 0.01, 0.02])
    with torch.no_grad():
        rig_spline.controls[:] = rig_residual
        camera_extrinsics.residuals[0] = extrinsic_residual
        camera_extrinsics.residuals[1] = -extrinsic_residual

    baked = calib.apply_to_rig_trajectories(source)
    baked_trajectory = baked.rig_trajectories[0]
    baked_camera_frame_poses = unpack_optional(baked_trajectory.cameras_frame_T_rig_worlds)
    rig_transform = residual_vectors_to_transforms(rig_residual).to(torch.float64)
    extrinsic_transform = residual_vectors_to_transforms(extrinsic_residual)
    torch.testing.assert_close(baked_trajectory.T_rig_worlds, source_coarse @ rig_transform)
    torch.testing.assert_close(
        baked_camera_frame_poses["camera_0"],
        source_camera_frames @ rig_transform,
    )
    torch.testing.assert_close(
        baked.camera_calibrations["camera_0"].T_sensor_rig,
        source_camera_extrinsic @ extrinsic_transform,
    )
    torch.testing.assert_close(
        baked.lidar_calibrations["lidar_0"].T_sensor_rig,
        source.lidar_calibrations["lidar_0"].T_sensor_rig,
    )
    np.testing.assert_array_equal(baked.world_to_nurec.matrix, source.world_to_nurec.matrix)
    assert baked_trajectory.cameras_linear_start_frame_indices == {"camera_0": 0, "camera_1": 2}
    assert baked_trajectory.lidars_linear_start_frame_indices == {"lidar_0": 0}
    torch.testing.assert_close(baked.T_world_base, source.T_world_base)
    assert baked_trajectory.T_rig_worlds.device.type == "cpu"
    assert baked_trajectory.T_rig_worlds.requires_grad is False

    # Baked camera poses are the same poses seen by the training-time correction path.
    source_camera_pose = source_camera_frames @ source_camera_extrinsic.to(torch.float64)
    source_camera_pose[..., :3, 3] *= calib.nurec_scale
    training_pose = calib._correct_sensor_poses(
        source_camera_pose.to(torch.float32),
        source_trajectory.cameras_frame_timestamps_us["camera_0"],
        0,
        sensor_type="camera",
        apply_rig_trajectory=True,
        apply_camera_extrinsics=True,
    )
    baked_camera_pose = baked_camera_frame_poses["camera_0"] @ baked.camera_calibrations["camera_0"].T_sensor_rig.to(
        torch.float64
    )
    baked_camera_pose[..., :3, 3] *= calib.nurec_scale
    torch.testing.assert_close(baked_camera_pose.to(torch.float32), training_pose)

    torch.testing.assert_close(source_trajectory.T_rig_worlds, source_coarse)
    torch.testing.assert_close(source_camera_frame_poses["camera_0"], source_camera_frames)
    torch.testing.assert_close(source.camera_calibrations["camera_0"].T_sensor_rig, source_camera_extrinsic)

    with torch.no_grad():
        rig_spline.controls[:, 0] += 0.1
    refreshed = calib.apply_to_rig_trajectories(source)
    assert not torch.equal(refreshed.rig_trajectories[0].T_rig_worlds, baked_trajectory.T_rig_worlds)
    torch.testing.assert_close(source_trajectory.T_rig_worlds, source_coarse)
