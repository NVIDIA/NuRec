# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NCORE point-cloud coloring and training frame mapping."""

import inspect
import unittest

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import torch

from PIL import Image

import nurec.datasets.ncore as ncore_module

from nurec.datasets.base import ActiveTileMappingPrewarmEntry, ActiveTileMappingPrewarmPlan
from nurec.datasets.ncore import (
    NCOREDataSource,
    NCORETrainDataset,
    UniqueSensorId,
    _CameraRGBMode,
    _fov_to_lo_hi,
    compute_effective_samples_per_epoch,
)
from nurec.datasets.samplers.base import BaseBatchSampler, BaseCameraPixelSampler
from nurec.datasets.samplers.holdout import compute_holdout_training_frame_mask
from nurec.datasets.samplers.image_crop import ImageCropCameraPixelSampler
from nurec.utils.batch import RectSubsampled
from nurec.utils.types import RayFlags


def test_dataset_camera_mask_uses_resolved_backend_and_camera_override(tmp_path: Path) -> None:
    datasource = NCOREDataSource.__new__(NCOREDataSource)
    datasource.resolved_dataset = MagicMock()
    mask_image = Image.fromarray(np.asarray([[0, 255]], dtype=np.uint8))
    datasource.resolved_dataset.backend.get_camera_mask_and_resolution.return_value = (
        mask_image,
        np.asarray([2, 1]),
    )
    override_path = tmp_path / "camera-mask.png"
    Image.fromarray(np.asarray([[255, 0]], dtype=np.uint8)).save(override_path)
    datasource.camera_mask_overrides = {"camera": str(override_path)}
    camera_sensor = MagicMock()

    mask = datasource._get_dataset_camera_mask("camera", camera_sensor)

    datasource.resolved_dataset.backend.get_camera_mask_and_resolution.assert_called_once_with(camera_sensor)
    assert mask is not None
    npt.assert_array_equal(mask, np.asarray([[True, False]]))


def test_metric_z_depth_method_names_are_recognized() -> None:
    method: str | None
    for method in ("MetricDepthAnythingV2", "vggt-z-depth"):
        assert ncore_module.is_metric_z_depth_method(method)

    for method in ("DepthAnythingV2", None):
        assert not ncore_module.is_metric_z_depth_method(method)


class TestEffectiveSamplesPerEpoch(unittest.TestCase):
    def test_scaling_rounds_up(self):
        self.assertEqual(
            compute_effective_samples_per_epoch(
                configured_n_samples_per_epoch=10,
                eligible_pairs=1,
                total_pairs=3,
                scale_samples_per_epoch_by_frame_selection_fraction=True,
            ),
            4,
        )

    def test_disabled_scaling_preserves_configured_count(self):
        self.assertEqual(
            compute_effective_samples_per_epoch(
                configured_n_samples_per_epoch=10,
                eligible_pairs=1,
                total_pairs=3,
                scale_samples_per_epoch_by_frame_selection_fraction=False,
            ),
            10,
        )

    def test_all_eligible_preserves_configured_count(self):
        self.assertEqual(
            compute_effective_samples_per_epoch(
                configured_n_samples_per_epoch=10,
                eligible_pairs=3,
                total_pairs=3,
                scale_samples_per_epoch_by_frame_selection_fraction=True,
            ),
            10,
        )

    def test_non_positive_total_raises(self):
        for total_pairs in (0, -1):
            with self.subTest(total_pairs=total_pairs), self.assertRaisesRegex(ValueError, "total_pairs"):
                compute_effective_samples_per_epoch(
                    configured_n_samples_per_epoch=10,
                    eligible_pairs=1,
                    total_pairs=total_pairs,
                    scale_samples_per_epoch_by_frame_selection_fraction=True,
                )

    def test_invalid_eligible_count_raises(self):
        for eligible_pairs in (0, -1, 4):
            with self.subTest(eligible_pairs=eligible_pairs), self.assertRaisesRegex(ValueError, "eligible_pairs"):
                compute_effective_samples_per_epoch(
                    configured_n_samples_per_epoch=10,
                    eligible_pairs=eligible_pairs,
                    total_pairs=3,
                    scale_samples_per_epoch_by_frame_selection_fraction=True,
                )

    def test_dataset_applies_and_exposes_effective_count(self):
        dataset = object.__new__(NCORETrainDataset)
        dataset.configured_n_samples_per_epoch = 10
        dataset.effective_n_samples_per_epoch = 10

        result = dataset.apply_samples_per_epoch_scaling(
            eligible_pairs=1,
            total_pairs=4,
            scale_samples_per_epoch_by_frame_selection_fraction=True,
        )

        self.assertEqual(result, 3)
        self.assertEqual(dataset.n_samples_per_epoch, 3)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.configured_n_samples_per_epoch, 10)


class TestCameraDataBatchWithoutRGB(unittest.TestCase):
    @staticmethod
    def _dataset() -> tuple[NCORETrainDataset, MagicMock]:
        dataset = object.__new__(NCORETrainDataset)
        datasource = MagicMock()
        datasource.camera_sensors = {"cam": MagicMock()}
        datasource.camera_sensors["cam"].get_frame_timestamp_us.return_value = 10
        datasource.camera_models = {"cam": SimpleNamespace(resolution=torch.tensor([6, 4]))}
        datasource.camera_linear_start_frame_indices = {"cam": 0}
        datasource.camera_frame_ranges = {"cam": range(1)}
        datasource.camera_unique_ids = {"cam": UniqueSensorId(id="cam", idx=0)}
        datasource.get_training_frame_idx_camera.return_value = 0
        datasource.cameras_frame_valid_pixels_masks = {
            "cam": {0: SimpleNamespace(unpacked=lambda: np.ones((4, 6), dtype=np.bool_))}
        }
        datasource.aux_loader = None
        dataset.datasource = datasource
        dataset.load_aux_depth = False
        dataset.load_aux_normal = False
        return dataset, datasource

    def test_skips_rgb_decode_and_uses_sampled_shape(self) -> None:
        dataset, datasource = self._dataset()
        sampled = RectSubsampled(
            i=0,
            j=0,
            height=2,
            width=3,
            subsample_factor=2.0,
            original_width=6,
            original_height=4,
        )

        batch = dataset.get_camera_data_batch("cam", 0, sampled, load_rgb=False)

        datasource.decode_image_cpu.assert_not_called()
        self.assertIsNone(batch.labels.rgb)
        flags = batch.labels.flags
        assert flags is not None
        self.assertEqual(flags.shape, (1, 2, 3, 1))
        self.assertFalse(torch.any(flags & RayFlags.RGB_LABEL.value))
        self.assertEqual(batch.labels.n_valid_rgb, 0)

    def test_skips_rgb_decode_and_uses_full_camera_shape(self) -> None:
        dataset, datasource = self._dataset()

        batch = dataset.get_camera_data_batch("cam", 0, load_rgb=False)

        datasource.decode_image_cpu.assert_not_called()
        self.assertIsNone(batch.labels.rgb)
        flags = batch.labels.flags
        assert flags is not None
        self.assertEqual(flags.shape, (1, 4, 6, 1))
        self.assertFalse(torch.any(flags & RayFlags.RGB_LABEL.value))
        self.assertEqual(batch.labels.n_valid_rgb, 0)

    def test_default_path_still_decodes_rgb(self) -> None:
        dataset, datasource = self._dataset()
        datasource.decode_image_cpu.return_value = np.full((4, 6, 3), 255, dtype=np.uint8)

        batch = dataset.get_camera_data_batch("cam", 0)

        datasource.decode_image_cpu.assert_called_once()
        rgb = batch.labels.rgb
        assert rgb is not None
        self.assertTrue(torch.equal(rgb, torch.ones((1, 4, 6, 3))))
        flags = batch.labels.flags
        assert flags is not None
        self.assertTrue(torch.all(flags & RayFlags.RGB_LABEL.value))

    def test_aux_geometry_loading_is_independent_of_rgb(self) -> None:
        dataset, datasource = self._dataset()
        dataset.load_aux_depth = True
        datasource.sensor_sky_class_ids = {"cam": 1}
        datasource.sensor_road_class_ids = {"cam": 2}
        datasource.sensor_vehicle_classes_ids = {"cam": [3]}
        datasource.camera_model_parameters = {"cam": SimpleNamespace(resolution=(6, 4))}
        datasource.world_to_nurec = SimpleNamespace(target_scale=1.0)
        aux_loader = MagicMock()
        aux_loader.get_semantic_segmentation.return_value = np.zeros((4, 6), dtype=np.uint8)
        aux_loader.has_depth.return_value = True
        aux_loader.get_depth.return_value = np.ones((4, 6), dtype=np.float32)
        aux_loader.get_depth_meta.return_value = {"method": "MetricDepthAnythingV2"}
        datasource.aux_loader = aux_loader
        dataset.camera_rig_module = MagicMock()
        dataset.camera_rig_module._compute_elements_and_sensor_rays.return_value = (
            torch.zeros((4, 6, 2)),
            torch.ones((4, 6, 3)),
        )
        sampled = RectSubsampled(
            i=0,
            j=0,
            height=4,
            width=6,
            subsample_factor=1.0,
            original_width=6,
            original_height=4,
        )

        batch = dataset.get_camera_data_batch("cam", 0, sampled, load_rgb=False)

        datasource.decode_image_cpu.assert_not_called()
        self.assertIsNone(batch.labels.rgb)
        self.assertIsNotNone(batch.labels.metric_distance)
        aux_loader.get_depth.assert_called_once()

        aux_loader.reset_mock()
        batch = dataset.get_camera_data_batch(
            "cam",
            0,
            sampled,
            load_rgb=False,
            load_aux_geometry=False,
        )

        self.assertIsNone(batch.labels.metric_distance)
        aux_loader.has_depth.assert_not_called()
        aux_loader.get_depth.assert_not_called()

    def test_rgb_path_still_applies_requested_sample_rectangle(self) -> None:
        dataset, datasource = self._dataset()
        datasource.decode_image_cpu.return_value = np.full((2, 3, 3), 255, dtype=np.uint8)
        sampled = RectSubsampled(
            i=1,
            j=0,
            height=2,
            width=2,
            subsample_factor=2.0,
            original_width=6,
            original_height=4,
        )

        batch = dataset.get_camera_data_batch("cam", 0, sampled)

        rgb = batch.labels.rgb
        assert rgb is not None
        self.assertEqual(rgb.shape, (1, 2, 2, 3))
        self.assertEqual(datasource.decode_image_cpu.call_args.kwargs["subsample_factor"], 2.0)


def _lidar_point_cloud_datasource() -> MagicMock:
    datasource = MagicMock(spec=NCOREDataSource)
    datasource._maybe_init_worker = MagicMock()
    datasource._compute_lidarpoints_dynamic = MagicMock()
    datasource._sensor_ids_to_unique_ids.return_value = ["lidar"]
    datasource.all_lidar_ids = ["lidar"]
    datasource.all_camera_ids = []
    datasource.lidar_unique_ids = {"lidar": UniqueSensorId(id="lidar", idx=0)}
    datasource.lidar_frame_ranges = {"lidar": range(1)}
    datasource.tqdm_disabled = True

    rows = torch.arange(4, dtype=torch.float32)
    xyz = torch.stack((rows, rows + 1, rows + 2), dim=-1)
    lidar_sensor = MagicMock()
    lidar_sensor.get_frame_point_cloud.return_value = SimpleNamespace(
        xyz_m_start=xyz.numpy(),
        xyz_m_end=(xyz + 0.5).numpy(),
    )
    lidar_sensor.get_frame_ray_bundle_return_intensity.return_value = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    # A V4 bundle can retain no-return emissions, so the point-cloud path filters on
    # the return-valid mask. These fixtures are all-returns, so the mask is all-true.
    lidar_sensor.get_frame_ray_bundle_return_valid_mask.return_value = np.ones(4, dtype=np.bool_)
    lidar_sensor.get_frames_T_sensor_target.return_value = np.eye(4, dtype=np.float32)
    lidar_sensor.get_frame_timestamp_us.return_value = 0
    lidar_sensor.get_frame_generic_data.return_value = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.uint8
    )
    datasource.lidar_sensors = {"lidar": lidar_sensor}

    aux_loader = MagicMock()
    aux_loader.has_lidar_semantic_segmentation.return_value = True
    aux_loader.get_lidar_semantic_segmentation_meta.return_value = {"ignore_label": -1}
    aux_loader.get_lidar_semantic_segmentation.return_value = np.array([10, 20, 30, 40])
    datasource.aux_loader = aux_loader

    datasource.world_to_nurec = MagicMock()
    datasource.world_to_nurec.transform_poses.side_effect = lambda pose: pose
    datasource.world_to_nurec.target_scale = 1.0
    datasource.sensor_road_class_ids = {"lidar": 10}
    datasource.sensor_sky_class_ids = {"lidar": 30}
    datasource.sensor_vehicle_classes_ids = {"lidar": [20]}
    datasource.get_semantic_colormap.return_value = np.arange(256 * 3, dtype=np.uint8).reshape(256, 3)

    colors = torch.tensor(
        [[10, 11, 12], [0, 0, 0], [30, 31, 32], [0, 0, 0]],
        dtype=torch.uint8,
    )
    color_valid_mask = torch.tensor([True, False, True, False])
    scales = torch.tensor([1.0, 0.0, 3.0, 0.0])
    datasource._project_lidar_camera_rgb = MagicMock(return_value=(colors, color_valid_mask, scales))
    datasource._color_lidar_point_cloud_from_cameras = NCOREDataSource._color_lidar_point_cloud_from_cameras.__get__(
        datasource, NCOREDataSource
    )
    datasource._color_lidar_point_cloud = NCOREDataSource._color_lidar_point_cloud.__get__(datasource, NCOREDataSource)
    datasource._transform_lidar_point_cloud_to_nurec = NCOREDataSource._transform_lidar_point_cloud_to_nurec.__get__(
        datasource, NCOREDataSource
    )
    datasource._apply_lidar_semantics = NCOREDataSource._apply_lidar_semantics.__get__(datasource, NCOREDataSource)
    datasource.get_lidar_point_clouds = NCOREDataSource.get_lidar_point_clouds.__get__(
        datasource,
        NCOREDataSource,
    )
    return datasource


def test_get_point_clouds_camera_rgb_modes_preserve_alignment_and_report_validity() -> None:
    rows = torch.arange(4, dtype=torch.float32)
    expected_xyz_end = torch.stack((rows, rows + 1, rows + 2), dim=-1) + 0.5
    legacy = list(
        NCOREDataSource.get_point_clouds(
            _lidar_point_cloud_datasource(),
            device=torch.device("cpu"),
            lidar_ids=["lidar"],
            valid_points_only=False,
            non_dynamic_points_only=False,
            color_type="camera-rgb",
        )
    )[0]
    torch.testing.assert_close(legacy.xyz_end, expected_xyz_end[torch.tensor([0, 2])])
    torch.testing.assert_close(legacy.semantic_class_id, torch.tensor([10, 30]))
    torch.testing.assert_close(legacy.intensity, torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(legacy.color, torch.tensor([[10, 11, 12], [30, 31, 32]], dtype=torch.uint8))
    torch.testing.assert_close(legacy.camera_footprint_scale, torch.tensor([1.0, 3.0]))
    assert legacy.n_points == 2
    assert legacy.flags is not None
    assert not bool(torch.any(legacy.flags & RayFlags.RGB_LABEL.value))

    valid_pixels = list(
        NCOREDataSource.get_point_clouds(
            _lidar_point_cloud_datasource(),
            device=torch.device("cpu"),
            lidar_ids=["lidar"],
            valid_points_only=False,
            non_dynamic_points_only=False,
            color_type="camera-rgb-frame-valid-pixels",
        )
    )[0]
    torch.testing.assert_close(valid_pixels.xyz_end, expected_xyz_end)
    torch.testing.assert_close(valid_pixels.semantic_class_id, torch.tensor([10, 20, 30, 40]))
    torch.testing.assert_close(valid_pixels.intensity, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(
        valid_pixels.color,
        torch.tensor([[10, 11, 12], [0, 0, 0], [30, 31, 32], [0, 0, 0]], dtype=torch.uint8),
    )
    torch.testing.assert_close(valid_pixels.camera_footprint_scale, torch.tensor([1.0, 0.0, 3.0, 0.0]))
    assert valid_pixels.n_points == 4
    assert valid_pixels.flags is not None
    torch.testing.assert_close(
        (valid_pixels.flags & RayFlags.RGB_LABEL.value) != 0,
        torch.tensor([True, False, True, False]),
    )


def test_get_point_clouds_generic_and_semantic_color_modes_preserve_alignment() -> None:
    generic = list(
        NCOREDataSource.get_point_clouds(
            _lidar_point_cloud_datasource(),
            device=torch.device("cpu"),
            lidar_ids=["lidar"],
            valid_points_only=False,
            non_dynamic_points_only=False,
            color_type="generic-data",
        )
    )[0]
    torch.testing.assert_close(
        generic.color,
        torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.uint8),
    )
    torch.testing.assert_close(generic.semantic_class_id, torch.tensor([10, 20, 30, 40]))
    torch.testing.assert_close(generic.intensity, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    semantics = list(
        NCOREDataSource.get_point_clouds(
            _lidar_point_cloud_datasource(),
            device=torch.device("cpu"),
            lidar_ids=["lidar"],
            valid_points_only=False,
            non_dynamic_points_only=False,
            color_type="semantics",
        )
    )[0]
    torch.testing.assert_close(
        semantics.color,
        torch.tensor([[30, 31, 32], [60, 61, 62], [90, 91, 92], [120, 121, 122]], dtype=torch.uint8),
    )
    torch.testing.assert_close(semantics.semantic_class_id, torch.tensor([10, 20, 30, 40]))
    assert semantics.flags is not None
    assert bool(torch.all(semantics.flags & RayFlags.VALID_SEMANTIC.value))
    assert bool(semantics.flags[0] & RayFlags.ROAD_SEMANTIC.value)
    assert bool(semantics.flags[1] & RayFlags.VEHICLE_SEMANTIC.value)
    assert bool(semantics.flags[2] & RayFlags.SKY_SEMANTIC.value)


def _camera_projection_datasource(
    camera_order: tuple[str, ...],
    *,
    valid_masks: dict[str, bool],
    ego_valid_masks: dict[str, bool] | None = None,
    camera_footprints: dict[str, float] | None = None,
) -> MagicMock:
    datasource = MagicMock(spec=NCOREDataSource)
    datasource.camera_sensors = {}
    datasource.camera_models = {}
    datasource.cameras_all_footprints = {}
    datasource.cameras_valid_pixels_ego_masks = {}
    datasource.cameras_frame_valid_pixels_masks = {}
    datasource.camera_frame_ranges = {}

    camera_colors = {
        "wide": np.array([[[255, 0, 0]]], dtype=np.uint8),
        "tele": np.array([[[0, 0, 255]]], dtype=np.uint8),
    }
    camera_footprints = camera_footprints or {"wide": 2.0, "tele": 1.0}
    for camera_id in camera_order:
        sensor = MagicMock()
        sensor.get_closest_frame_index.return_value = 0
        sensor.get_frame_handle.return_value = camera_id
        sensor.get_frames_T_source_sensor.return_value = np.stack(
            (np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32))
        )
        datasource.camera_sensors[camera_id] = sensor
        datasource.camera_frame_ranges[camera_id] = range(1)

        projection = MagicMock()
        projection.image_points = torch.tensor([[0.0, 0.0]])
        projection.valid_indices = torch.tensor([0])
        camera_model = MagicMock()
        camera_model.world_points_to_image_points_shutter_pose.return_value = projection
        datasource.camera_models[camera_id] = camera_model
        datasource.cameras_all_footprints[camera_id] = np.array([[camera_footprints[camera_id]]], dtype=np.float32)

        frame_valid_mask = MagicMock()
        frame_valid_mask.unpacked.return_value = np.array([[valid_masks[camera_id]]], dtype=bool)
        ego_valid_mask = MagicMock()
        ego_valid_mask.unpacked.return_value = np.array([[(ego_valid_masks or valid_masks)[camera_id]]], dtype=bool)
        datasource.cameras_valid_pixels_ego_masks[camera_id] = ego_valid_mask
        datasource.cameras_frame_valid_pixels_masks[camera_id] = [frame_valid_mask]

    datasource.decode_image_cpu.side_effect = camera_colors.__getitem__
    return datasource


def test_valid_camera_rgb_overlap_uses_finer_valid_observation() -> None:
    xyz = torch.tensor([[0.0, 0.0, 1.0]])
    for camera_order in (("wide", "tele"), ("tele", "wide")):
        datasource = _camera_projection_datasource(
            camera_order,
            valid_masks={"wide": True, "tele": True},
        )
        color, color_valid_mask, scale = NCOREDataSource._project_lidar_camera_rgb(
            datasource,
            xyz,
            np.eye(4, dtype=np.float32),
            0,
            torch.device("cpu"),
            _CameraRGBMode.FRAME_VALID_PIXELS,
        )

        torch.testing.assert_close(color, torch.tensor([[0, 0, 255]], dtype=torch.uint8))
        torch.testing.assert_close(color_valid_mask, torch.tensor([True]))
        torch.testing.assert_close(scale, torch.tensor([1.0]))


def test_valid_camera_rgb_equal_footprints_keep_datasource_order() -> None:
    xyz = torch.tensor([[0.0, 0.0, 1.0]])
    for camera_order, expected_color in (
        (("wide", "tele"), torch.tensor([[255, 0, 0]], dtype=torch.uint8)),
        (("tele", "wide"), torch.tensor([[0, 0, 255]], dtype=torch.uint8)),
    ):
        datasource = _camera_projection_datasource(
            camera_order,
            valid_masks={"wide": True, "tele": True},
            camera_footprints={"wide": 1.0, "tele": 1.0},
        )
        color, color_valid_mask, scale = NCOREDataSource._project_lidar_camera_rgb(
            datasource,
            xyz,
            np.eye(4, dtype=np.float32),
            0,
            torch.device("cpu"),
            _CameraRGBMode.FRAME_VALID_PIXELS,
        )

        torch.testing.assert_close(color, expected_color)
        torch.testing.assert_close(color_valid_mask, torch.tensor([True]))
        torch.testing.assert_close(scale, torch.tensor([1.0]))


def test_valid_camera_rgb_overlap_falls_back_from_masked_finer_observation() -> None:
    datasource = _camera_projection_datasource(
        ("wide", "tele"),
        valid_masks={"wide": True, "tele": False},
    )
    color, color_valid_mask, scale = NCOREDataSource._project_lidar_camera_rgb(
        datasource,
        torch.tensor([[0.0, 0.0, 1.0]]),
        np.eye(4, dtype=np.float32),
        0,
        torch.device("cpu"),
        _CameraRGBMode.FRAME_VALID_PIXELS,
    )

    torch.testing.assert_close(color, torch.tensor([[255, 0, 0]], dtype=torch.uint8))
    torch.testing.assert_close(color_valid_mask, torch.tensor([True]))
    torch.testing.assert_close(scale, torch.tensor([2.0]))


def test_valid_camera_rgb_leaves_points_uncolored_when_every_observation_is_masked() -> None:
    datasource = _camera_projection_datasource(
        ("wide", "tele"),
        valid_masks={"wide": False, "tele": False},
    )
    color, color_valid_mask, scale = NCOREDataSource._project_lidar_camera_rgb(
        datasource,
        torch.tensor([[0.0, 0.0, 1.0]]),
        np.eye(4, dtype=np.float32),
        0,
        torch.device("cpu"),
        _CameraRGBMode.FRAME_VALID_PIXELS,
    )

    torch.testing.assert_close(color, torch.zeros((1, 3), dtype=torch.uint8))
    torch.testing.assert_close(color_valid_mask, torch.tensor([False]))
    torch.testing.assert_close(scale, torch.tensor([0.0]))


def test_frame_valid_camera_rgb_skips_closest_frame_outside_configured_range() -> None:
    datasource = _camera_projection_datasource(
        ("wide",),
        valid_masks={"wide": True},
    )
    datasource.camera_sensors["wide"].get_closest_frame_index.return_value = 1

    color, color_valid_mask, scale = NCOREDataSource._project_lidar_camera_rgb(
        datasource,
        torch.tensor([[0.0, 0.0, 1.0]]),
        np.eye(4, dtype=np.float32),
        0,
        torch.device("cpu"),
        _CameraRGBMode.FRAME_VALID_PIXELS,
    )

    datasource.decode_image_cpu.assert_not_called()
    torch.testing.assert_close(color, torch.zeros((1, 3), dtype=torch.uint8))
    torch.testing.assert_close(color_valid_mask, torch.tensor([False]))
    torch.testing.assert_close(scale, torch.tensor([0.0]))


def test_legacy_camera_rgb_preserves_last_camera_wins_and_ignores_validity_masks() -> None:
    datasource = _camera_projection_datasource(
        ("wide", "tele"),
        valid_masks={"wide": True, "tele": False},
    )
    color, color_valid_mask, scale = NCOREDataSource._project_lidar_camera_rgb(
        datasource,
        torch.tensor([[0.0, 0.0, 1.0]]),
        np.eye(4, dtype=np.float32),
        0,
        torch.device("cpu"),
        _CameraRGBMode.ALL_PIXELS,
    )

    torch.testing.assert_close(color, torch.tensor([[0, 0, 255]], dtype=torch.uint8))
    torch.testing.assert_close(color_valid_mask, torch.tensor([True]))
    torch.testing.assert_close(scale, torch.tensor([1.0]))


def test_ego_valid_camera_rgb_ignores_masked_housing_observations() -> None:
    datasource = _camera_projection_datasource(
        ("wide", "tele"),
        valid_masks={"wide": False, "tele": True},
        ego_valid_masks={"wide": True, "tele": False},
    )
    color, color_valid_mask, scale = NCOREDataSource._project_lidar_camera_rgb(
        datasource,
        torch.tensor([[0.0, 0.0, 1.0]]),
        np.eye(4, dtype=np.float32),
        0,
        torch.device("cpu"),
        _CameraRGBMode.EGO_VALID_PIXELS,
    )

    torch.testing.assert_close(color, torch.tensor([[255, 0, 0]], dtype=torch.uint8))
    torch.testing.assert_close(color_valid_mask, torch.tensor([True]))
    torch.testing.assert_close(scale, torch.tensor([2.0]))


class TestComputeHoldoutTrainingFrameMask(unittest.TestCase):
    """Tests for the compute_holdout_training_frame_mask() pure function."""

    def test_no_holdout_all_training(self):
        """With no holdout config, every frame is a training frame."""
        mask = compute_holdout_training_frame_mask(
            n_frames=5,
            frame_start=None,
            frame_step=None,
            exclude_frame_start=None,
            exclude_frame_step=None,
        )
        npt.assert_array_equal(mask, [True, True, True, True, True])

    def test_val_frame_step_from_start_0(self):
        """val at [0::2] → validation at indices 0, 2, 4; training at 1, 3."""
        mask = compute_holdout_training_frame_mask(
            n_frames=5,
            frame_start=0,
            frame_step=2,
            exclude_frame_start=None,
            exclude_frame_step=None,
        )
        npt.assert_array_equal(mask, [False, True, False, True, False])

    def test_val_frame_step_with_custom_start(self):
        """val at [1::2] → validation at indices 1, 3; training at 0, 2, 4."""
        mask = compute_holdout_training_frame_mask(
            n_frames=5,
            frame_start=1,
            frame_step=2,
            exclude_frame_start=None,
            exclude_frame_step=None,
        )
        npt.assert_array_equal(mask, [True, False, True, False, True])

    def test_val_frame_step_3(self):
        """val at [0::3] → validation at 0, 3; training at 1, 2, 4."""
        mask = compute_holdout_training_frame_mask(
            n_frames=5,
            frame_start=0,
            frame_step=3,
            exclude_frame_start=None,
            exclude_frame_step=None,
        )
        npt.assert_array_equal(mask, [False, True, True, False, True])

    def test_val_frame_step_default_start(self):
        """frame_start=None defaults to 0."""
        mask = compute_holdout_training_frame_mask(
            n_frames=5,
            frame_start=None,
            frame_step=2,
            exclude_frame_start=None,
            exclude_frame_step=None,
        )
        npt.assert_array_equal(mask, [False, True, False, True, False])

    def test_val_exclude_frame_step(self):
        """training = [0::2] → training at 0, 2, 4; validation at 1, 3."""
        mask = compute_holdout_training_frame_mask(
            n_frames=5,
            frame_start=None,
            frame_step=None,
            exclude_frame_start=0,
            exclude_frame_step=2,
        )
        npt.assert_array_equal(mask, [True, False, True, False, True])

    def test_val_exclude_frame_step_with_custom_start(self):
        """training = [1::2] → training at 1, 3; validation at 0, 2, 4."""
        mask = compute_holdout_training_frame_mask(
            n_frames=5,
            frame_start=None,
            frame_step=None,
            exclude_frame_start=1,
            exclude_frame_step=2,
        )
        npt.assert_array_equal(mask, [False, True, False, True, False])

    def test_val_exclude_frame_step_default_start(self):
        """exclude_frame_start=None defaults to 0."""
        mask = compute_holdout_training_frame_mask(
            n_frames=5,
            frame_start=None,
            frame_step=None,
            exclude_frame_start=None,
            exclude_frame_step=2,
        )
        npt.assert_array_equal(mask, [True, False, True, False, True])

    def test_frame_step_1_means_no_holdout(self):
        """frame_step=1 does not satisfy > 1 guard, so all frames are training."""
        mask = compute_holdout_training_frame_mask(
            n_frames=5,
            frame_start=0,
            frame_step=1,
            exclude_frame_start=None,
            exclude_frame_step=None,
        )
        npt.assert_array_equal(mask, [True, True, True, True, True])

    def test_single_frame(self):
        """Edge case: a single frame with holdout."""
        mask = compute_holdout_training_frame_mask(
            n_frames=1,
            frame_start=0,
            frame_step=2,
            exclude_frame_start=None,
            exclude_frame_step=None,
        )
        npt.assert_array_equal(mask, [False])

    def test_zero_frames(self):
        """Edge case: zero frames produces an empty mask."""
        mask = compute_holdout_training_frame_mask(
            n_frames=0,
            frame_start=0,
            frame_step=2,
            exclude_frame_start=None,
            exclude_frame_step=None,
        )
        self.assertEqual(len(mask), 0)


class TestSetupTrainingFrameMapping(unittest.TestCase):
    """Integration tests for NCOREDataSource.setup_training_frame_mapping()
    and the no-holdout fallback path.

    Uses a lightly mocked NCOREDataSource with only the attributes needed,
    avoiding heavy ncore dependencies.
    """

    def _make_datasource(
        self,
        camera_ids: list[str],
        camera_frame_counts: list[int],
        camera_unique_indices: list[int],
        lidar_ids: list[str] | None = None,
        lidar_frame_counts: list[int] | None = None,
    ) -> MagicMock:
        """Create a minimal mock NCOREDataSource with the attributes needed by setup_training_frame_mapping."""

        ds = MagicMock(spec=NCOREDataSource)
        ds._maybe_init_worker = MagicMock()

        # Camera setup
        ds.all_camera_ids = camera_ids
        ds.camera_frame_ranges = {}
        ds.camera_linear_start_frame_indices = {}
        ds.camera_unique_ids = {}

        linear_start = 0
        for cam_id, n_frames, unique_idx in zip(camera_ids, camera_frame_counts, camera_unique_indices, strict=True):
            ds.camera_frame_ranges[cam_id] = range(n_frames)
            ds.camera_linear_start_frame_indices[cam_id] = linear_start
            ds.camera_unique_ids[cam_id] = UniqueSensorId(id=cam_id, idx=unique_idx)
            linear_start += n_frames

        # Lidar setup
        lidar_ids = lidar_ids or []
        lidar_frame_counts = lidar_frame_counts or []
        ds.all_lidar_ids = lidar_ids
        ds.lidar_frame_ranges = {}
        ds.lidar_linear_start_frame_indices = {}

        linear_start = 0
        for lidar_id, n_frames in zip(lidar_ids, lidar_frame_counts, strict=True):
            ds.lidar_frame_ranges[lidar_id] = range(n_frames)
            ds.lidar_linear_start_frame_indices[lidar_id] = linear_start
            linear_start += n_frames

        return ds

    @staticmethod
    def _group(
        camera_frame_sampler: dict | None = None,
        lidar_frame_sampler: dict | None = None,
        ratio_camera_samples: float = 1.0,
        ratio_lidar_samples: float = 1.0,
    ) -> dict:
        """A raw batch-sampler group entry, as found in ``dataset.samplers``."""
        return {
            "camera_frame_sampler": camera_frame_sampler or {"name": "uniform"},
            "lidar_frame_sampler": lidar_frame_sampler or {"name": "uniform"},
            "ratio_camera_samples": ratio_camera_samples,
            "ratio_lidar_samples": ratio_lidar_samples,
        }

    def _call_setup(self, ds: MagicMock, sampler_groups: list[dict]) -> None:
        """Call the real setup_training_frame_mapping on the mock datasource."""
        NCOREDataSource.setup_training_frame_mapping(ds, sampler_groups)

    def test_no_holdout_fallback_single_camera(self):
        """No holdout → don't call setup, fallbacks produce identity mapping."""

        ds = self._make_datasource(
            camera_ids=["front"],
            camera_frame_counts=[5],
            camera_unique_indices=[0],
        )
        # Explicitly set to None to simulate the no-holdout path (MagicMock
        # auto-creates attributes, so we must override them).
        ds._global_to_training_camera_frame_idx = None
        ds._n_training_frames_per_camera = None

        # Verify scalar fallback returns identity mapping.
        for i in range(5):
            self.assertEqual(NCOREDataSource.get_training_frame_idx_camera(ds, i), i)

        # Verify array accessor materializes the identity mapping.
        npt.assert_array_equal(
            NCOREDataSource.get_global_to_training_camera_frame_mapping(ds),
            [0, 1, 2, 3, 4],
        )

    def test_no_holdout_fallback_two_cameras(self):
        """No holdout, two cameras → fallback identity mapping across both."""

        ds = self._make_datasource(
            camera_ids=["front", "back"],
            camera_frame_counts=[3, 2],
            camera_unique_indices=[0, 1],
        )
        ds._global_to_training_camera_frame_idx = None

        # front: unique_frame_idx 0,1,2 → training 0,1,2
        # back: unique_frame_idx 3,4 → training 3,4
        for i in range(5):
            self.assertEqual(NCOREDataSource.get_training_frame_idx_camera(ds, i), i)

        npt.assert_array_equal(
            NCOREDataSource.get_global_to_training_camera_frame_mapping(ds),
            [0, 1, 2, 3, 4],
        )

    def test_holdout_exclude_mode_two_cameras(self):
        """Holdout sampler excluding [0::3] → correct compaction with -1 gaps."""
        ds = self._make_datasource(
            camera_ids=["front", "back"],
            camera_frame_counts=[3, 2],
            camera_unique_indices=[0, 1],
        )
        holdout = {"name": "holdout", "exclude_frames_start": 0, "exclude_every_n_frames": 3}
        self._call_setup(ds, [self._group(camera_frame_sampler=holdout, ratio_lidar_samples=0.0)])

        # front (3 frames): sampler excludes local 0 → mask [F,T,T]
        #   global 0 → -1,  global 1 → 0,  global 2 → 1
        # back (2 frames): sampler excludes local 0 → mask [F,T]
        #   global 3 → -1,  global 4 → 2
        npt.assert_array_equal(ds._global_to_training_camera_frame_idx, [-1, 0, 1, -1, 2])
        npt.assert_array_equal(ds._n_training_frames_per_camera, [2, 1])

        # Array accessor returns the same compacted mapping.
        npt.assert_array_equal(
            NCOREDataSource.get_global_to_training_camera_frame_mapping(ds),
            [-1, 0, 1, -1, 2],
        )

    def test_holdout_include_mode_two_cameras(self):
        """Holdout sampler including only [0::2] across two cameras."""
        ds = self._make_datasource(
            camera_ids=["front", "back"],
            camera_frame_counts=[4, 3],
            camera_unique_indices=[0, 1],
        )
        holdout = {"name": "holdout", "include_frames_start": 0, "include_every_n_frames": 2}
        self._call_setup(ds, [self._group(camera_frame_sampler=holdout, ratio_lidar_samples=0.0)])

        # front (4 frames): training at [0,2] → mask [T,F,T,F]
        #   global 0 → 0,  global 1 → -1,  global 2 → 1,  global 3 → -1
        # back (3 frames): training at [0,2] → mask [T,F,T]
        #   global 4 → 2,  global 5 → -1,  global 6 → 3
        npt.assert_array_equal(ds._global_to_training_camera_frame_idx, [0, -1, 1, -1, 2, -1, 3])
        npt.assert_array_equal(ds._n_training_frames_per_camera, [2, 2])

    def test_holdout_lidar_mapping(self):
        """Lidar frames get their own independent training index compaction."""
        ds = self._make_datasource(
            camera_ids=["front"],
            camera_frame_counts=[3],
            camera_unique_indices=[0],
            lidar_ids=["top_lidar"],
            lidar_frame_counts=[4],
        )
        holdout = {"name": "holdout", "exclude_frames_start": 0, "exclude_every_n_frames": 2}
        self._call_setup(ds, [self._group(lidar_frame_sampler=holdout)])

        # Camera: unrestricted (uniform) → all training
        npt.assert_array_equal(ds._global_to_training_camera_frame_idx, [0, 1, 2])
        # Lidar (4 frames): sampler excludes [0::2] → mask [F,T,F,T]
        #   global 0 → -1,  global 1 → 0,  global 2 → -1,  global 3 → 1
        npt.assert_array_equal(ds._global_to_training_lidar_frame_idx, [-1, 0, -1, 1])

        # Array accessors return the same results.
        npt.assert_array_equal(NCOREDataSource.get_global_to_training_camera_frame_mapping(ds), [0, 1, 2])
        npt.assert_array_equal(NCOREDataSource.get_global_to_training_lidar_frame_mapping(ds), [-1, 0, -1, 1])

    def test_no_holdout_lidar_array_accessor(self):
        """No holdout for lidar → array accessor returns identity."""

        ds = self._make_datasource(
            camera_ids=["front"],
            camera_frame_counts=[3],
            camera_unique_indices=[0],
            lidar_ids=["top_lidar"],
            lidar_frame_counts=[4],
        )
        ds._global_to_training_camera_frame_idx = None
        ds._global_to_training_lidar_frame_idx = None

        npt.assert_array_equal(NCOREDataSource.get_global_to_training_lidar_frame_mapping(ds), [0, 1, 2, 3])

    def test_training_sensor_frame_indices_preserve_raw_ranges_and_filter_holdouts(self):
        """Typed camera/lidar accessors return raw sensor indices, not global or compact indices."""
        ds = NCOREDataSource.__new__(NCOREDataSource)
        ds.camera_frame_ranges = {"front": range(5, 9), "back": range(10, 13)}
        ds.camera_linear_start_frame_indices = {"front": 0, "back": 4}
        ds.lidar_frame_ranges = {"lidar": range(20, 24)}
        ds.lidar_linear_start_frame_indices = {"lidar": 0}
        ds._global_to_training_camera_frame_idx = np.asarray([-1, 0, -1, 1, 2, -1, 3], dtype=np.int32)
        ds._global_to_training_lidar_frame_idx = np.asarray([0, -1, 1, -1], dtype=np.int32)

        front = ds.get_training_camera_sensor_frame_indices("front")
        back = ds.get_training_camera_sensor_frame_indices("back")
        lidar = ds.get_training_lidar_sensor_frame_indices("lidar")

        npt.assert_array_equal(front, [6, 8])
        npt.assert_array_equal(back, [10, 12])
        npt.assert_array_equal(lidar, [20, 22])
        self.assertEqual(front.dtype, np.dtype(np.int64))
        self.assertEqual(back.dtype, np.dtype(np.int64))
        self.assertEqual(lidar.dtype, np.dtype(np.int64))

    def test_holdout_training_counter_is_global_across_cameras(self):
        """The training counter increments globally, not per-camera."""
        ds = self._make_datasource(
            camera_ids=["cam_a", "cam_b", "cam_c"],
            camera_frame_counts=[2, 1, 2],
            camera_unique_indices=[0, 1, 2],
        )
        holdout = {"name": "holdout", "exclude_frames_start": 1, "exclude_every_n_frames": 3}
        self._call_setup(ds, [self._group(camera_frame_sampler=holdout, ratio_lidar_samples=0.0)])

        # Per camera the sampler excludes local [1::3]:
        # cam_a (2 frames): [T,F] → global 0 → 0
        # cam_b (1 frame):  [T]   → global 2 → 1  (counter continues globally)
        # cam_c (2 frames): [T,F] → global 3 → 2
        npt.assert_array_equal(ds._global_to_training_camera_frame_idx, [0, -1, 1, 2, -1])
        npt.assert_array_equal(ds._n_training_frames_per_camera, [1, 1, 1])

    def test_unrestricted_samplers_keep_identity_mapping(self):
        """uniform/timestamp samplers reach every frame → mapping stays unset."""
        ds = self._make_datasource(
            camera_ids=["front"],
            camera_frame_counts=[3],
            camera_unique_indices=[0],
        )
        self._call_setup(ds, [self._group()])
        self.assertIsNone(ds._global_to_training_camera_frame_idx)
        self.assertIsNone(ds._n_training_frames_per_camera)

    def test_union_across_groups_restores_identity(self):
        """A restrictive group unioned with an unrestricted one → every frame
        is a training frame, so the mapping stays identity. (Under the old
        validation-config derivation the restrictive group would have marked
        frames -1 that the unrestricted group can sample — the leak crash.)"""
        ds = self._make_datasource(
            camera_ids=["front"],
            camera_frame_counts=[4],
            camera_unique_indices=[0],
        )
        holdout = {"name": "holdout", "include_frames_start": 0, "include_every_n_frames": 2}
        self._call_setup(ds, [self._group(camera_frame_sampler=holdout), self._group()])
        self.assertIsNone(ds._global_to_training_camera_frame_idx)

    def test_single_included_frame_gets_training_index_zero(self):
        """A training sampler that explicitly includes exactly one frame must
        map that frame to training index 0 — never to -1, regardless of any
        validation holdout configuration (validation may deliberately
        re-render the same frame for an overfit diagnostic)."""
        ds = self._make_datasource(
            camera_ids=["front"],
            camera_frame_counts=[5],
            camera_unique_indices=[0],
        )
        fixed = {"name": "fixed", "frame_index": 2}
        self._call_setup(ds, [self._group(camera_frame_sampler=fixed, ratio_lidar_samples=0.0)])

        npt.assert_array_equal(ds._global_to_training_camera_frame_idx, [-1, -1, 0, -1, -1])
        npt.assert_array_equal(ds._n_training_frames_per_camera, [1])
        self.assertEqual(NCOREDataSource.get_training_frame_idx_camera(ds, 2), 0)

    def test_fixed_sampler_multiple_cameras_contiguous_indices(self):
        """Per-camera frame ranges: the fixed frame of every camera receives
        a contiguous training index."""
        ds = self._make_datasource(
            camera_ids=["front", "left", "right"],
            camera_frame_counts=[4, 3, 5],
            camera_unique_indices=[0, 1, 2],
        )
        fixed = {"name": "fixed", "frame_index": 1}
        self._call_setup(ds, [self._group(camera_frame_sampler=fixed, ratio_lidar_samples=0.0)])

        # front global [0..3], left [4..6], right [7..11]; local 1 each.
        npt.assert_array_equal(
            ds._global_to_training_camera_frame_idx,
            [-1, 0, -1, -1, -1, 1, -1, -1, 2, -1, -1, -1],
        )
        npt.assert_array_equal(ds._n_training_frames_per_camera, [1, 1, 1])

    def test_zero_ratio_group_contributes_no_frames(self):
        """ratio_camera_samples == 0 substitutes a SkipFrameSampler — the
        group's camera sampler cannot produce frames."""
        ds = self._make_datasource(
            camera_ids=["front"],
            camera_frame_counts=[3],
            camera_unique_indices=[0],
            lidar_ids=["top_lidar"],
            lidar_frame_counts=[2],
        )
        holdout = {"name": "holdout", "include_frames_start": 0, "include_every_n_frames": 2}
        camera_only = self._group(camera_frame_sampler=holdout, ratio_lidar_samples=0.0)
        lidar_only = self._group(ratio_camera_samples=0.0)
        self._call_setup(ds, [camera_only, lidar_only])

        # Camera frames come from the holdout group only ([0, 2]); the
        # lidar-only group's uniform lidar sampler reaches every lidar frame.
        npt.assert_array_equal(ds._global_to_training_camera_frame_idx, [0, -1, 1])
        npt.assert_array_equal(ds._global_to_training_lidar_frame_idx, [0, 1])

    def test_timestamp_share_rule_uses_effective_camera_sampler(self):
        """A timestamp camera sampler is shared with the lidar side —
        including the Skip substitution when the group draws no camera
        samples. The group's own (restrictive) lidar sampler config is
        ignored in that case, mirroring BaseBatchSampler.__init__."""
        ds = self._make_datasource(
            camera_ids=["front"],
            camera_frame_counts=[2],
            camera_unique_indices=[0],
            lidar_ids=["top_lidar"],
            lidar_frame_counts=[3],
        )
        # Group 1: timestamp camera sampler with zero camera ratio — its
        # lidar side shares the substituted SkipFrameSampler, so its
        # "fixed frame 1" lidar config must contribute NOTHING.
        group1 = self._group(
            camera_frame_sampler={"name": "timestamp"},
            lidar_frame_sampler={"name": "fixed", "frame_index": 1},
            ratio_camera_samples=0.0,
        )
        holdout = {"name": "holdout", "include_frames_start": 0, "include_every_n_frames": 2}
        group2 = self._group(camera_frame_sampler=holdout, lidar_frame_sampler=holdout)
        self._call_setup(ds, [group1, group2])

        # Lidar frames come from group 2's holdout only: [0, 2]. Had
        # group 1's fixed lidar sampler leaked through, frame 1 would be
        # a training frame too (making the union the identity).
        npt.assert_array_equal(ds._global_to_training_lidar_frame_idx, [0, -1, 1])
        npt.assert_array_equal(ds._global_to_training_camera_frame_idx, [0, -1])

    def test_no_sampler_groups_keeps_identity(self):
        """Pure-validation runs configure no sampler groups — identity."""
        ds = self._make_datasource(
            camera_ids=["front"],
            camera_frame_counts=[3],
            camera_unique_indices=[0],
        )
        self._call_setup(ds, [])
        self.assertIsNone(ds._global_to_training_camera_frame_idx)
        self.assertIsNone(ds._n_training_frames_per_camera)


class CameraTrackAuxLoaderTest(unittest.TestCase):
    """Lidar-free camera-track init must keep aux_loader under PYTHONOPTIMIZE=1."""

    def _datasource(self, aux_loader):
        """Build a minimal NCOREDataSource stub bound to the real get_camera_track_point_clouds."""
        datasource = MagicMock(spec=NCOREDataSource)
        datasource._maybe_init_worker = MagicMock()
        datasource._sensor_ids_to_unique_ids.side_effect = lambda ids, _type: ids
        datasource.aux_loader = aux_loader
        datasource.camera_sensors = {}
        datasource.get_camera_track_point_clouds = NCOREDataSource.get_camera_track_point_clouds.__get__(
            datasource, NCOREDataSource
        )
        return datasource

    def test_empty_camera_ids_skips_aux_requirement(self):
        """An empty camera list yields no point clouds without ever requiring aux data."""
        datasource = self._datasource(None)
        self.assertEqual(
            list(datasource.get_camera_track_point_clouds([], MagicMock(), device=torch.device("cpu"))),
            [],
        )

    def test_missing_aux_loader_raises_assertion_error(self):
        """Requesting a camera without any loaded aux data fails loudly rather than silently."""
        datasource = self._datasource(None)
        with self.assertRaisesRegex(AssertionError, "aux data was not loaded"):
            list(datasource.get_camera_track_point_clouds(["cam0"], MagicMock(), device=torch.device("cpu")))

    def test_missing_depth_skips_camera(self):
        """A camera without depth data is skipped for point-cloud init, not a hard failure."""
        aux_loader = MagicMock()
        aux_loader.has_depth.return_value = False
        datasource = self._datasource(aux_loader)
        datasource.camera_sensors = {"cam0": MagicMock()}
        datasource.camera_unique_ids = {"cam0": UniqueSensorId(id="cam0", idx=0)}
        self.assertEqual(
            list(datasource.get_camera_track_point_clouds(["cam0"], MagicMock(), device=torch.device("cpu"))),
            [],
        )
        aux_loader.has_depth.assert_called_with("cam0")

    def test_empty_cameras_use_aux_loader_without_unbound_local(self):
        """aux_loader stays bound when no sensor matches, so the loop exits without UnboundLocalError."""
        aux_loader = MagicMock()
        aux_loader.has_depth.return_value = True
        datasource = self._datasource(aux_loader)
        self.assertEqual(
            list(datasource.get_camera_track_point_clouds(["cam0"], MagicMock(), device=torch.device("cpu"))),
            [],
        )

    def test_camera_track_init_does_not_bind_aux_loader_in_assert(self):
        """aux_loader must be assigned outside the assert so PYTHONOPTIMIZE=1 cannot strip it."""
        try:
            source = inspect.getsource(NCOREDataSource.get_camera_track_point_clouds)
        except OSError:
            self.skipTest("source is not available next to the compiled module")
        self.assertNotIn("assert (aux_loader :=", source)
        self.assertIn("aux_loader = self.aux_loader", source)


class _StubFov:
    """Minimal stand-in for ncore.impl.data.util.FOV (start_rad, span_rad, direction)."""

    def __init__(self, start_rad: float, span_rad: float, direction: str):
        self.start_rad = start_rad
        self.span_rad = span_rad
        self.direction = direction


class TestFovToLoHi(unittest.TestCase):
    """_fov_to_lo_hi must respect the ncore FOV direction (cw vs ccw).

    ncore's FOV.start_rad is the FIRST element and span sweeps in `direction`:
    cw sweeps DOWN (start is the high end -> lo = start - span); ccw sweeps UP
    (start is the low end -> hi = start + span). Vertical FOV is always cw;
    horizontal follows the sensor spin direction, so the ccw case must NOT be
    treated as cw (the bug this guards against).
    """

    def test_cw_start_is_high_end(self):
        # cw: start=0.3 (high), span=0.5 -> [-0.2, 0.3]
        lo, hi = _fov_to_lo_hi(_StubFov(start_rad=0.3, span_rad=0.5, direction="cw"))
        npt.assert_allclose([lo, hi], [-0.2, 0.3], atol=1e-9)

    def test_ccw_start_is_low_end(self):
        # ccw: start=-0.2 (low), span=0.5 -> [-0.2, 0.3]  (would be WRONG under cw)
        lo, hi = _fov_to_lo_hi(_StubFov(start_rad=-0.2, span_rad=0.5, direction="ccw"))
        npt.assert_allclose([lo, hi], [-0.2, 0.3], atol=1e-9)

    def test_returns_ordered_lo_hi(self):
        # Regardless of direction, the result is ordered lo <= hi.
        lo, hi = _fov_to_lo_hi(_StubFov(start_rad=1.0, span_rad=0.4, direction="cw"))
        self.assertLessEqual(lo, hi)
        lo, hi = _fov_to_lo_hi(_StubFov(start_rad=-1.0, span_rad=0.4, direction="ccw"))
        self.assertLessEqual(lo, hi)

    def test_missing_direction_defaults_cw(self):
        # A FOV object without a `direction` attribute falls back to cw.
        class _NoDir:
            start_rad = 0.3
            span_rad = 0.5

        lo, hi = _fov_to_lo_hi(_NoDir())
        npt.assert_allclose([lo, hi], [-0.2, 0.3], atol=1e-9)


def _make_active_tile_prewarm_dataset() -> tuple[
    NCORETrainDataset,
    ImageCropCameraPixelSampler,
    SimpleNamespace,
    NCOREDataSource,
]:
    pixel_sampler = ImageCropCameraPixelSampler.__new__(ImageCropCameraPixelSampler)
    pixel_sampler.crop_type = "full_image"
    pixel_sampler.subsample = 2
    pixel_sampler.crop_width_ratio = pixel_sampler.crop_height_ratio = 1.0
    batch_sampler = SimpleNamespace(
        n_train_sample_camera_rays=1,
        n_train_sample_lidar_rays=1,
        camera_pixel_sampler=pixel_sampler,
    )

    camera_a_timestamps = np.zeros((4, 2), dtype=np.int64)
    camera_a_timestamps[:, 0] = [0, 390, 400, 460]
    camera_b_timestamps = np.zeros((4, 2), dtype=np.int64)
    camera_b_timestamps[:, 0] = [0, 700, 405, 800]
    lidar_timestamps = np.zeros((7, 2), dtype=np.int64)
    lidar_timestamps[:, 0] = np.arange(7) * 100

    datasource = NCOREDataSource.__new__(NCOREDataSource)
    datasource.sequence_id = "sequence"
    datasource.camera_unique_ids = {
        "camera_a": UniqueSensorId(id="camera_a", idx=4),
        "camera_b": UniqueSensorId(id="camera_b", idx=1),
    }
    datasource.lidar_unique_ids = {"lidar": UniqueSensorId(id="lidar", idx=9)}
    datasource.camera_frame_ranges = {"camera_a": range(1, 4), "camera_b": range(1, 4)}
    datasource.lidar_frame_ranges = {"lidar": range(2, 7)}
    datasource.camera_linear_start_frame_indices = {"camera_a": 0, "camera_b": 3}
    datasource.lidar_linear_start_frame_indices = {"lidar": 0}
    datasource.camera_sensors = {
        "camera_a": SimpleNamespace(frames_timestamps_us=camera_a_timestamps),
        "camera_b": SimpleNamespace(frames_timestamps_us=camera_b_timestamps),
    }
    datasource.lidar_sensors = {"lidar": SimpleNamespace(frames_timestamps_us=lidar_timestamps)}
    datasource.camera_model_parameters = {
        "camera_a": cast(Any, SimpleNamespace(resolution=np.asarray((8, 6), dtype=np.int64))),
        "camera_b": cast(Any, SimpleNamespace(resolution=np.asarray((8, 6), dtype=np.int64))),
    }
    # LiDAR unique frames 0, 2, 4 are trainable, so frame 4 (timestamp 400)
    # is the central training frame. Camera A's exact-timestamp frame is a
    # holdout, forcing the nearest trainable frame at timestamp 390.
    datasource._global_to_training_lidar_frame_idx = np.asarray([10, -1, 12, -1, 14], dtype=np.int32)
    datasource._global_to_training_camera_frame_idx = np.asarray([20, -1, 22, 23, 24, 25], dtype=np.int32)

    dataset = NCORETrainDataset.__new__(NCORETrainDataset)
    dataset.batch_samplers = [cast(BaseBatchSampler, batch_sampler)]
    dataset.camera_ids = ["camera_a", "camera_b"]
    dataset.lidar_ids = ["lidar"]
    dataset.datasource = cast(NCOREDataSource, datasource)
    dataset.config = cast(Any, SimpleNamespace(use_real_lidar_rays=False))
    return dataset, pixel_sampler, batch_sampler, datasource


class TestActiveTileMappingPrewarmPlan(unittest.TestCase):
    def test_plan_requires_exactly_entries_or_skip_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "must contain entries or a skip reason"):
            ActiveTileMappingPrewarmPlan()

        entry = cast(ActiveTileMappingPrewarmEntry, MagicMock())
        with self.assertRaisesRegex(ValueError, "cannot contain both entries and a skip reason"):
            ActiveTileMappingPrewarmPlan(entries=(entry,), skip_reason="unsupported")

    def test_plan_is_metadata_only_sorted_and_uses_nearest_training_frames(self) -> None:
        dataset, _, _, _ = _make_active_tile_prewarm_dataset()

        plan = NCORETrainDataset.get_active_tile_mapping_prewarm_plan(dataset)

        self.assertIsNone(plan.skip_reason)
        self.assertEqual([entry.camera_sensor_idx for entry in plan.entries], [1, 4])

        camera_b = plan.entries[0].data_batch.camera
        camera_a = plan.entries[1].data_batch.camera
        assert camera_b is not None
        assert camera_a is not None
        camera_b_meta = camera_b.meta[0]
        camera_a_meta = camera_a.meta[0]
        self.assertEqual(
            (
                camera_b_meta.unique_sensor_idx,
                camera_b_meta.unique_frame_idx,
                camera_b_meta.training_frame_idx,
            ),
            (1, 4, 24),
        )
        self.assertEqual(
            (
                camera_a_meta.unique_sensor_idx,
                camera_a_meta.unique_frame_idx,
                camera_a_meta.training_frame_idx,
            ),
            (4, 0, 20),
        )
        for entry in plan.entries:
            assert entry.data_batch.camera is not None
            assert entry.data_batch.lidar is not None
            camera_meta = entry.data_batch.camera.meta[0]
            lidar_meta = entry.data_batch.lidar.meta[0]
            self.assertEqual(
                (lidar_meta.unique_sensor_idx, lidar_meta.unique_frame_idx, lidar_meta.training_frame_idx),
                (9, 2, 12),
            )
            assert camera_meta.subsample is not None
            self.assertEqual(
                (
                    camera_meta.subsample.i,
                    camera_meta.subsample.j,
                    camera_meta.subsample.width,
                    camera_meta.subsample.height,
                    camera_meta.subsample.subsample_factor,
                    camera_meta.subsample.original_width,
                    camera_meta.subsample.original_height,
                ),
                (0, 0, 4, 3, 2.0, 8, 6),
            )
            self.assertIsNone(entry.data_batch.camera.labels.rgb)
            self.assertIsNone(entry.data_batch.lidar.labels.distance)

    def test_fixed_full_image_geometry_matches_normal_sampling(self) -> None:
        _, pixel_sampler, _, _ = _make_active_tile_prewarm_dataset()
        for subsample in (2, True):
            with self.subTest(subsample=subsample):
                pixel_sampler.subsample = subsample
                sampled_pixels = pixel_sampler.sample_camera_pixels(
                    rng=np.random.default_rng(0),
                    batch_idx=0,
                    frame_range=range(4),
                    n_frame_pixel_samples=1,
                    frame_all_pixels=np.empty((0, 2), dtype=np.int64),
                    frame_valid_pixels_mask=np.ones((6, 8), dtype=np.bool_),
                    unique_camera_id="camera_a",
                    camera_frame_idx=1,
                ).sampled_pixels

                self.assertEqual(
                    sampled_pixels,
                    pixel_sampler.fixed_sampled_pixels(original_width=8, original_height=6),
                )

    def test_entry_rejects_a_sensor_index_that_disagrees_with_its_batch(self) -> None:
        dataset, _, _, _ = _make_active_tile_prewarm_dataset()
        (entry, _) = NCORETrainDataset.get_active_tile_mapping_prewarm_plan(dataset).entries

        with self.assertRaisesRegex(ValueError, "camera_sensor_idx must match"):
            ActiveTileMappingPrewarmEntry(
                camera_sensor_idx=entry.camera_sensor_idx + 1,
                data_batch=entry.data_batch,
            )

    def test_plan_reports_every_sampler_geometry_skip_gate(self) -> None:
        dataset, _, _, _ = _make_active_tile_prewarm_dataset()
        dataset.batch_samplers = []
        self.assertEqual(
            NCORETrainDataset.get_active_tile_mapping_prewarm_plan(dataset).skip_reason,
            "requires_one_batch_sampler",
        )

        dataset, _, batch_sampler, _ = _make_active_tile_prewarm_dataset()
        batch_sampler.n_train_sample_camera_rays = 0
        self.assertEqual(
            NCORETrainDataset.get_active_tile_mapping_prewarm_plan(dataset).skip_reason,
            "requires_camera_and_lidar_training",
        )

        dataset, _, batch_sampler, _ = _make_active_tile_prewarm_dataset()
        batch_sampler.n_train_sample_lidar_rays = 0
        self.assertEqual(
            NCORETrainDataset.get_active_tile_mapping_prewarm_plan(dataset).skip_reason,
            "requires_camera_and_lidar_training",
        )

        dataset, _, batch_sampler, _ = _make_active_tile_prewarm_dataset()
        batch_sampler.camera_pixel_sampler = MagicMock(spec=BaseCameraPixelSampler)
        batch_sampler.camera_pixel_sampler.fixed_sampled_pixels.return_value = None
        self.assertEqual(
            NCORETrainDataset.get_active_tile_mapping_prewarm_plan(dataset).skip_reason,
            "requires_fixed_full_image_sampling",
        )

        dataset, pixel_sampler, _, _ = _make_active_tile_prewarm_dataset()
        pixel_sampler.crop_type = "random"
        self.assertEqual(
            NCORETrainDataset.get_active_tile_mapping_prewarm_plan(dataset).skip_reason,
            "requires_fixed_full_image_sampling",
        )

        for invalid_subsample in (False, 0, 1.5):
            dataset, pixel_sampler, _, _ = _make_active_tile_prewarm_dataset()
            pixel_sampler.subsample = invalid_subsample
            with self.subTest(invalid_subsample=invalid_subsample):
                self.assertEqual(
                    NCORETrainDataset.get_active_tile_mapping_prewarm_plan(dataset).skip_reason,
                    "requires_fixed_full_image_sampling",
                )

    def test_plan_reports_every_sensor_selection_skip_gate(self) -> None:
        dataset, _, _, _ = _make_active_tile_prewarm_dataset()
        dataset.camera_ids = []
        self.assertEqual(
            NCORETrainDataset.get_active_tile_mapping_prewarm_plan(dataset).skip_reason,
            "requires_training_cameras",
        )

        dataset, _, _, _ = _make_active_tile_prewarm_dataset()
        dataset.lidar_ids = []
        self.assertEqual(
            NCORETrainDataset.get_active_tile_mapping_prewarm_plan(dataset).skip_reason,
            "requires_one_training_lidar",
        )

    def test_plan_skips_without_training_lidar_frames(self) -> None:
        dataset, _, _, datasource = _make_active_tile_prewarm_dataset()
        lidar_training_indices = datasource._global_to_training_lidar_frame_idx
        assert lidar_training_indices is not None
        lidar_training_indices.fill(-1)
        self.assertEqual(
            NCORETrainDataset.get_active_tile_mapping_prewarm_plan(dataset).skip_reason,
            "requires_training_lidar_frames",
        )

    def test_plan_reports_camera_frame_and_subsample_skip_gates(self) -> None:
        dataset, _, _, datasource = _make_active_tile_prewarm_dataset()
        camera_training_indices = datasource._global_to_training_camera_frame_idx
        assert camera_training_indices is not None
        camera_training_indices.fill(-1)
        self.assertEqual(
            NCORETrainDataset.get_active_tile_mapping_prewarm_plan(dataset).skip_reason,
            "camera_without_training_frames:camera_b",
        )

        dataset, _, _, datasource = _make_active_tile_prewarm_dataset()
        datasource.camera_model_parameters["camera_b"].resolution[0] = 7
        self.assertEqual(
            NCORETrainDataset.get_active_tile_mapping_prewarm_plan(dataset).skip_reason,
            "requires_fixed_full_image_sampling",
        )

    def test_plan_skips_real_lidar_rays(self) -> None:
        dataset, _, _, _ = _make_active_tile_prewarm_dataset()
        dataset.config.use_real_lidar_rays = True

        plan = NCORETrainDataset.get_active_tile_mapping_prewarm_plan(dataset)

        self.assertEqual(plan.skip_reason, "requires_modeled_lidar_rays")
        self.assertEqual(plan.entries, ())


if __name__ == "__main__":
    unittest.main()
