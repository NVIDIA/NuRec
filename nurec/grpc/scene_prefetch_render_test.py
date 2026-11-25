# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check scene loading and shared-scene camera batches through real CUDA rendering."""

import gc
import io
import weakref
import zipfile

from collections import OrderedDict
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from omegaconf import OmegaConf

import nurec.models.gaussians.renderers as gaussian_renderers

from ncore.data import OpenCVPinholeCameraModelParameters, RowOffsetStructuredSpinningLidarModelParameters, ShutterType
from nurec.artifact import Artifact
from nurec.config.model import ModelConfig
from nurec.config.trainer import TrainerConfig
from nurec.datasets.summary import DataSourceSummary
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.grpc.cache_keys import BackendCacheKey, SceneCacheKey
from nurec.grpc.pose import tquat_to_grpc_pose
from nurec.grpc.protos.sensorsim_pb2 import CameraRenderRequest, CameraSpec, DynamicObject, PosePair
from nurec.grpc.serve import Backend, BackendCache, PreparedBackend, _CameraBatchDispatch
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import SHGaussianModel
from nurec.render.render import CameraRenderOptions, LidarRayBundle, PoseRange, RayBundle
from nurec.utils.lidar_model import LidarModelBundle
from nurec.utils.types import FrameConversion, RigTrajectories


@pytest.fixture
def dynamic_artifact(tmp_path: Path) -> Artifact:
    """Export three Gaussians on one moving track, with per-track Fourier appearance."""
    camera = OpenCVPinholeCameraModelParameters(
        resolution=np.array([16, 16], dtype=np.uint64),
        focal_length=np.array([16, 16], dtype=np.float32),
        principal_point=np.array([8, 8], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        shutter_type=ShutterType.GLOBAL,
    )
    lidar = RowOffsetStructuredSpinningLidarModelParameters(
        spinning_frequency_hz=10.0,
        spinning_direction="ccw",
        n_rows=4,
        n_columns=8,
        row_elevations_rad=np.linspace(0.1, -0.1, 4).astype(np.float32),
        column_azimuths_rad=np.linspace(-0.1, 0.1, 8).astype(np.float32),
        row_azimuth_offsets_rad=np.zeros(4, dtype=np.float32),
    )
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    poses[:, 0, 3] = 5.0
    poses[1, 1, 3] = 0.5
    tracks = CuboidTracks.Factory.from_numpy(
        ["car"],
        [poses],
        [np.array([0, 1_000_000], dtype=np.int64)],
        ["automobile"],
        [TrackFlags.DYNAMIC],
        [np.array([2.0, 2.0, 2.0], dtype=np.float32)],
        device=torch.device("cpu"),
    )
    rig = RigTrajectories(
        T_world_base=torch.eye(4, dtype=torch.float64),
        world_to_nurec=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
        rig_trajectories=[
            RigTrajectories.RigTrajectory(
                sequence_id="scene",
                rig_bbox=None,
                cameras_frame_timestamps_us={"camera": torch.tensor([[0, 0], [1_000_000, 1_000_000]])},
                lidars_frame_timestamps_us={"lidar": torch.tensor([[0, 100_000], [800_000, 900_000]])},
                T_rig_worlds=torch.eye(4, dtype=torch.float64).repeat(2, 1, 1),
                T_rig_world_timestamps_us=torch.tensor([0, 1_000_000]),
            )
        ],
        camera_calibrations=OrderedDict(
            camera=RigTrajectories.CameraCalibration(
                sequence_id="scene",
                logical_sensor_name="camera",
                unique_sensor_idx=0,
                T_sensor_rig=torch.eye(4),
                camera_model_parameters=camera,
            )
        ),
        lidar_calibrations=OrderedDict(
            lidar=RigTrajectories.LidarCalibration(
                sequence_id="scene",
                logical_sensor_name="lidar",
                unique_sensor_idx=0,
                T_sensor_rig=torch.eye(4),
                lidar_model_parameters=lidar,
            )
        ),
    )
    summary = DataSourceSummary.State(
        n_frames_per_camera=np.array([2], dtype=np.int32),
        n_training_frames_per_camera=np.array([2], dtype=np.int32),
        n_frames_per_lidar=np.array([2], dtype=np.int32),
        sequence_tracks_all={"scene": tracks},
        sequence_tracks_dynamic={"scene": tracks},
        rig_trajectories=rig,
        xform_matrices=np.zeros((0, 4), dtype=np.float32),
        aabb_blb=np.array([[-10, -10, -10]], dtype=np.float32),
        aabb_trf=np.array([[10, 10, 10]], dtype=np.float32),
    )
    config = ModelConfig.model_validate(
        OmegaConf.create(
            {
                "name": "gaussians-composite",
                "train_num_rays": 1,
                "calib": {"name": "skip-calib", "camera": {"enabled": False}, "lidar": {"enabled": False}},
                "background": {"name": "skip-background"},
                "renderer": {"name": "3dgut-gsplat"},
                "strategy": {"name": "mcmc"},
                "post_processing": {},
                "layers": {
                    "dynamic_rigids": {
                        "name": "rigid-gaussians",
                        "density_activation": "sigmoid",
                        "scale_activation": "exp",
                        "rotation_activation": "normalize",
                        "fourier_features_dim": 3,
                        "fourier_freeze_until_step": 10,
                        "time_embed": {"name": "individual-remap-time-input-embedding", "remap_min": 0, "remap_max": 1},
                        "progressive_training": {
                            "init_n_features": 0,
                            "max_n_features": 3,
                            "increase_frequency": 1000,
                            "increase_step": 1,
                        },
                        "particle": {
                            "radiance_sph_degree": 3,
                            "radiance_sph_O0": True,
                            "density_kernel_planar": False,
                            "density_kernel_degree": 2,
                            "density_kernel_min_response": 0.0113,
                            "ray_spread_filter_enabled": False,
                            "extra_signal_dim": 0,
                            "camera_extra_signal_dim": 0,
                            "lidar_extra_signal_dim": 0,
                            "lidar_extra_signal_sph_degree": 0,
                        },
                        "initialization": {"name": "random", "local_debug_dir": None, "scale_multiplier": 1.0},
                        "tracks": {"is_dynamic": True},
                        "tracks_calib": {"name": "skip-tracks-calib", "visualize": False},
                        "optimize_track_albedo": False,
                        "optimize_track_scale": False,
                        "track_albedo": {},
                        "track_scale": {},
                    }
                },
            }
        )
    )
    trainer = TrainerConfig(
        max_epochs=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        precision="32",
    )
    model = GaussiansComposite(config, trainer, DataSourceSummary(state=summary, datasource=None), False)
    node = model.gaussians_nodes["dynamic_rigids"]
    assert isinstance(node, SHGaussianModel)
    for name, tensor in {
        "positions": torch.tensor([[0.0, -0.3, 0.0], [0.0, 0.0, 0.0], [0.0, 0.3, 0.0]]),
        "rotations": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        "scales": torch.full((3, 3), -0.5),
        "densities": torch.full((3, 1), 8.0),
        "features_albedo": torch.tensor([[[0.5, 0.3, 0.7], [0.1, 0.05, 0.1], [0.05, 0.1, 0.05]]]).repeat(3, 1, 1),
        "features_specular": torch.zeros(3, 45),
        "extra_signal": torch.empty(3, 0),
        "camera_extra_signal": torch.empty(3, 0),
        "lidar_extra_signal": torch.empty(3, 0),
    }.items():
        setattr(node, name, torch.nn.Parameter(tensor))
    node.gaussian_cuboid_ids = torch.nn.Buffer(torch.zeros(3, dtype=torch.int32))
    node.n_active_features = 0
    checkpoint = io.BytesIO()
    torch.save(
        {"state_dict": {f"model.{key}": value for key, value in model.state_dict().items()}, "load_in_place": False},
        checkpoint,
    )
    path = tmp_path / "scene.usdz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.yaml", "scene_id: scene\n")
        archive.writestr(
            "parsed_config.yaml",
            OmegaConf.to_yaml(
                OmegaConf.create(
                    {
                        "model": config.model_dump(),
                        "trainer": trainer.model_dump(),
                    }
                )
            ),
        )
        archive.writestr("datasource_summary.json", summary.to_json())
        archive.writestr("rig_trajectories.json", rig.to_json())
        archive.writestr("checkpoint.ckpt", checkpoint.getvalue())
    return Artifact(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA rendering required")
@pytest.mark.parametrize("explicit_intrinsics,edit_actors", [(False, False), (True, False), (True, True)])
def test_shared_camera_batch_matches_sequential_rendering(
    dynamic_artifact: Artifact, explicit_intrinsics: bool, edit_actors: bool
) -> None:
    key = BackendCacheKey(SceneCacheKey(scene_id="scene"), dynamic_artifact.sha256_footprint)
    backend = Backend.from_artifact(dynamic_artifact, Backend.RendererType.DEFAULT, 1, backend_cache_key=key)
    model = backend.renderable_model._model
    assert isinstance(model, GaussiansComposite)
    intrinsics = (
        next(backend.camera_bank.get_available_cameras()).intrinsics
        if explicit_intrinsics
        else CameraSpec(logical_id="camera")
    )
    actor_start = tquat_to_grpc_pose(torch.tensor([5.0, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0]))
    actor_end = tquat_to_grpc_pose(torch.tensor([5.0, 0.8, 0.0, 0.0, 0.0, 0.0, 1.0]))
    actors = (
        [DynamicObject(track_id="car", pose_pair=PosePair(start_pose=actor_start, end_pose=actor_end))]
        if edit_actors
        else []
    )
    reference_images: list[torch.Tensor] = []
    for timestamp, resolution in ((250_000, 16), (750_000, 16), (750_000, 8), (250_000, 16)):
        requests = []
        for offset in (-0.5, 0.5):
            pose = tquat_to_grpc_pose(torch.tensor([0.0, offset, 0.0, 0.0, 2**-0.5, 0.0, 2**-0.5]))
            requests.append(
                CameraRenderRequest(
                    scene_id="scene",
                    camera_intrinsics=intrinsics,
                    sensor_pose=PosePair(start_pose=pose, end_pose=pose),
                    resolution_h=resolution,
                    resolution_w=resolution,
                    frame_start_us=timestamp,
                    frame_end_us=timestamp + 10_000,
                    dynamic_objects=actors,
                )
            )
        expected: list[torch.Tensor] = []
        for request in requests:
            image = backend.render_camera_request(request, enable_editing_actors=edit_actors)[0].color_image
            assert image is not None and image.count_nonzero() > 0
            assert image.shape == (resolution, resolution, 3)
            expected.append(image)
        with (
            mock.patch.object(model, "_collect_sensor", wraps=model._collect_sensor) as collect,
            mock.patch.object(
                gaussian_renderers.gsplat, "rasterization", wraps=gaussian_renderers.gsplat.rasterization
            ) as rasterize,
        ):
            frames, dispatch = backend.render_camera_batch_shared_scene(requests, enable_editing_actors=edit_actors)
        assert dispatch == _CameraBatchDispatch.NATIVE_BATCHED_RAYS
        assert collect.call_count == 1
        assert rasterize.call_count == 1
        assert frames is not None and len(frames) == len(requests)
        for frame, image in zip(frames, expected):
            torch.testing.assert_close(frame.color_image, image)
        assert not torch.equal(expected[0], expected[1])
        reference_images.append(expected[0])
        reversed_frames, _ = backend.render_camera_batch_shared_scene(
            list(reversed(requests)), enable_editing_actors=edit_actors
        )
        assert reversed_frames is not None
        for frame, image in zip(reversed_frames, reversed(expected)):
            torch.testing.assert_close(frame.color_image, image)
    assert not torch.equal(reference_images[0], reference_images[1])
    torch.testing.assert_close(reference_images[0], reference_images[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA rendering required")
def test_prepared_backend_matches_synchronous_camera_and_lidar(dynamic_artifact: Artifact) -> None:
    artifact = dynamic_artifact
    key = BackendCacheKey(SceneCacheKey(scene_id="scene"), artifact.sha256_footprint)
    prepared = PreparedBackend.from_artifact(artifact, Backend.RendererType.DEFAULT, 1, backend_cache_key=key)
    assert prepared.renderable_model.device.type == "cpu"
    assert isinstance(prepared.renderable_model._model, GaussiansComposite)
    assert prepared.renderable_model._model.cuboid_track_devices() == {torch.device("cpu")}
    activated = prepared.activate()
    synchronous = Backend.from_artifact(artifact, Backend.RendererType.DEFAULT, 1, backend_cache_key=key)
    rig = RigTrajectories.from_dict(artifact.rig_trajectories)
    camera = rig.camera_calibrations["camera"].camera_model_parameters
    lidar = rig.lidar_calibrations["lidar"].lidar_model_parameters
    assert lidar is not None
    camera_pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 2**-0.5, 0.0, 2**-0.5])
    lidar_pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    reference_images: list[torch.Tensor] = []
    for timestamp in (250_000, 750_000):
        camera_rays = RayBundle.build(
            camera, PoseRange(camera_pose, camera_pose, timestamp, timestamp), activated.world_to_nurec
        )
        lidar_rays = LidarRayBundle.build(
            LidarModelBundle.load_from_config(lidar.to_dict()),
            PoseRange(lidar_pose, lidar_pose, timestamp, timestamp + 100_000),
            activated.world_to_nurec,
        )
        images = [
            backend.renderable_model.render_camera_frame_from_ray_bundle(
                camera_rays, CameraRenderOptions(color_image=True, opacity_image=True)
            )
            for backend in (activated, synchronous)
        ]
        actual, expected = images
        assert actual.color_image is not None and expected.color_image is not None
        assert actual.opacity_image is not None and expected.opacity_image is not None
        assert actual.color_image.count_nonzero() > 0
        assert actual.opacity_image.max() > 0.1
        torch.testing.assert_close(actual.color_image, expected.color_image)
        torch.testing.assert_close(actual.opacity_image, expected.opacity_image)
        reference_images.append(expected.color_image)
        scans = [
            backend.renderable_model.render_lidar_frame_from_ray_bundle(
                lidar_rays, opacity_threshold=0.01, enable_distance_filter=False
            )
            for backend in (activated, synchronous)
        ]
        assert scans[0].point_positions.numel() > 0
        torch.testing.assert_close(scans[0].point_positions, scans[1].point_positions)
    assert not torch.equal(*reference_images)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA model required")
def test_eager_eviction_releases_real_model_before_activation(dynamic_artifact: Artifact) -> None:
    key = BackendCacheKey(SceneCacheKey(scene_id="scene"), dynamic_artifact.sha256_footprint)
    backend = Backend.from_artifact(dynamic_artifact, Backend.RendererType.DEFAULT, 1, backend_cache_key=key)
    assert isinstance(backend.renderable_model._model, GaussiansComposite)
    parameter = weakref.ref(backend.renderable_model._model.gaussians_nodes["dynamic_rigids"].positions)
    cache = BackendCache(maxsize=1)
    cache.put(key.key_hash, backend)
    cache.checkin(backend)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        del backend
        reservation = cache.reserve("replacement", scene_id="scene")
        assert cache.evict_reserved_spare(reservation)
        assert parameter() is None
        cache.release_reservation(reservation)
    finally:
        if was_enabled:
            gc.enable()
