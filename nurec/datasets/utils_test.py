# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import types
import unittest

from collections import defaultdict
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation

import nurec.datasets.utils as utils_module

from ncore.data import BBox3, FrameTimepoint, OpenCVPinholeCameraModelParameters, ShutterType
from ncore.impl.common import transformations as ncore_transformations
from ncore.sensors import camera_model_from_parameters
from nurec.config import ValidPixelsFrameMaskConfig, ValidPixelsSceneFlowConfig, ValidPixelsTrafficLightConfig
from nurec.config.dataset import CuboidTracksConfig
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.datasets.utils import (
    BBox3Provider,
    PackedMask,
    _transform_cuboid_observations_native,
    compute_camera_visible_intervals_cuboid_tracks,
    compute_cameras_valid_pixels_frame_mask,
    compute_cuboid_tracks,
    compute_point_cloud_inside_tracks_mask,
    compute_points_outside_tracks,
    compute_valid_lidarpoints_trafficlight_cameravisible,
    compute_valid_pixels_ego,
    compute_valid_pixels_sceneflow,
    compute_valid_pixels_trafficlight,
    consolidate_cuboid_tracks,
    get_sampled_pixels,
)
from nurec.utils.batch import RectSubsampled, generate_grid_2d_indices
from nurec.utils.types import HalfClosedInterval, PointCloud


def _pixel_grid(width: int, height: int) -> np.ndarray:
    yy, xx = np.indices((height, width))
    return np.stack((xx, yy), axis=-1).reshape(-1, 2)


def test_get_sampled_pixels_returns_unstructured_pixels_unchanged() -> None:
    sampled = np.array([[1, 2], [3, 4]])

    pixels, half_pixel_offset = get_sampled_pixels(sampled, (6, 4), _pixel_grid(6, 4))

    assert pixels is sampled
    assert half_pixel_offset is False


@pytest.mark.parametrize(
    ("subsample_factor", "resolution", "crop_offset", "expected_pixels", "expected_half_pixel_offset"),
    [
        (
            2.0,
            (6, 4),
            (1, 0),
            [[2, 0], [4, 0], [2, 2], [4, 2]],
            True,
        ),
        (
            3.0,
            (6, 6),
            (0, 0),
            [[1, 1], [4, 1], [1, 4], [4, 4]],
            False,
        ),
    ],
)
def test_get_sampled_pixels_characterizes_structured_subsampling(
    subsample_factor: float,
    resolution: tuple[int, int],
    crop_offset: tuple[int, int],
    expected_pixels: list[list[int]],
    expected_half_pixel_offset: bool,
) -> None:
    width, height = resolution
    i, j = crop_offset
    sampled = RectSubsampled.from_subsample(
        original_width=width,
        original_height=height,
        width=2,
        height=2,
        i=i,
        j=j,
        subsample_factor=subsample_factor,
    )

    pixels, half_pixel_offset = get_sampled_pixels(sampled, resolution, _pixel_grid(width, height))

    np.testing.assert_array_equal(pixels, expected_pixels)
    assert half_pixel_offset is expected_half_pixel_offset


def test_get_sampled_pixels_rejects_non_integer_subsample_factor() -> None:
    sampled = RectSubsampled.from_subsample(
        original_width=6,
        original_height=3,
        width=2,
        height=2,
        subsample_factor=1.5,
    )

    with pytest.raises(AssertionError, match="must be an integer"):
        get_sampled_pixels(sampled, (6, 3), _pixel_grid(6, 3))


def test_get_sampled_pixels_rejects_mismatched_camera_resolution() -> None:
    sampled = RectSubsampled.from_subsample(
        original_width=6,
        original_height=4,
        width=3,
        height=2,
        subsample_factor=2.0,
    )

    with pytest.raises(AssertionError, match="does not match the camera resolution"):
        get_sampled_pixels(sampled, (8, 4), _pixel_grid(8, 4))


def test_get_sampled_pixels_rejects_unknown_sample_type() -> None:
    with pytest.raises(ValueError, match="Unexpected sampled pixels type"):
        get_sampled_pixels([], (6, 4), _pixel_grid(6, 4))  # type: ignore[arg-type]


@pytest.mark.parametrize("factor", [49, 93])
def test_get_sampled_pixels_preserves_integer_stride(factor: int) -> None:
    sampled = RectSubsampled.from_subsample(
        original_width=factor * 2, original_height=factor, width=2, height=1, subsample_factor=float(factor)
    )
    pixels, half_offset = get_sampled_pixels(sampled, (factor * 2, factor), _pixel_grid(factor * 2, factor))
    offset = (factor - 1) // 2
    np.testing.assert_array_equal(pixels, [[offset, offset], [factor + offset, offset]])
    assert not half_offset


class TestPackedMask(unittest.TestCase):
    def test_random(self):
        # make sure that random masks are packed / unpacked correctly
        NUM_RUNS = 20
        MAX_DIMS = 4
        MAX_SIZE = 5
        for n in range(NUM_RUNS):
            for d in range(MAX_DIMS):  # number of dimensions
                dims = [random.randrange(MAX_SIZE) for _ in range(d)]

                mask_ref = np.random.choice([True, False], size=dims)

                packed_mask = PackedMask(mask_ref)

                np.testing.assert_array_equal(mask_ref, packed_mask.unpacked())


class TestNativeCuboidAssociation(unittest.TestCase):
    def test_bbox_transforms_match_ncore_reference(self):
        rng = np.random.default_rng(7)
        bboxes = rng.normal(size=(16, 9)).astype(np.float32)
        bboxes[:, 3:6] = np.abs(bboxes[:, 3:6]) + 0.1
        bboxes[:4, 6:9] = np.array(
            [
                [0.3, np.pi / 2 - 1e-6, -0.7],
                [-1.2, -np.pi / 2 + 1e-6, 2.4],
                [np.pi - 1e-5, np.pi / 2, -np.pi + 1e-5],
                [-2.8, -np.pi / 2, 2.9],
            ],
            dtype=np.float32,
        )
        transforms = np.repeat(np.eye(4, dtype=np.float32)[None], len(bboxes), axis=0)
        transforms[:, :3, :3] = (
            Rotation.from_euler("xyz", rng.normal(size=(len(bboxes), 3))).as_matrix().astype(np.float32)
        )
        transforms[:, :3, 3] = rng.normal(size=(len(bboxes), 3))
        rig_valid = np.ones(len(bboxes), dtype=bool)

        expected = np.stack(
            [
                ncore_transformations.bbox_pose(ncore_transformations.transform_bbox(bbox, transform))
                for bbox, transform in zip(bboxes, transforms)
            ]
        )

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=device):
                keep, poses = _transform_cuboid_observations_native(
                    bboxes,
                    transforms,
                    transforms,
                    rig_valid,
                    track_min_centroid_rig_dist_m=0.0,
                    device=device,
                )
                np.testing.assert_array_equal(keep, np.ones(len(bboxes), dtype=bool))
                np.testing.assert_allclose(poses, expected, rtol=2e-6, atol=2e-6)

    def test_compute_cuboid_tracks_preserves_serialized_default_and_native_opt_out(self):
        """Filtering retains the caller's serialized/native observation representation."""
        cuboids_df = pd.DataFrame()
        tracks = object()
        params = CuboidTracksConfig(track_label_sources=None, track_min_centroid_rig_dist_m=0.0)

        with (
            patch("nurec.datasets.utils.compute_cuboid_df", return_value=cuboids_df) as compute_df,
            patch("nurec.datasets.utils.consolidate_cuboid_tracks", return_value={}),
            patch("nurec.datasets.utils.CuboidTracks.Factory.from_numpy", return_value=tracks),
        ):
            default_result = compute_cuboid_tracks(
                sequence_loader=Mock(),
                time_range_us=HalfClosedInterval(0, 1),
                cuboid_tracks_params=params,
                tqdm_disabled=True,
            )
            native_result = compute_cuboid_tracks(
                sequence_loader=Mock(),
                time_range_us=HalfClosedInterval(0, 1),
                cuboid_tracks_params=params,
                tqdm_disabled=True,
                serialize_observation=False,
            )

        assert default_result[0] is tracks
        assert default_result[1] is cuboids_df
        assert native_result[0] is tracks
        assert native_result[1] is cuboids_df
        assert compute_df.call_args_list[0].kwargs["serialize_observation"] is True
        assert compute_df.call_args_list[1].kwargs["serialize_observation"] is False

    def test_rig_distance_filter_preserves_nan_reference_behavior(self):
        bboxes = np.zeros((1, 9), dtype=np.float64)
        bboxes[0, 0] = np.nan
        transforms = np.eye(4, dtype=np.float64)[None]

        keep, _ = _transform_cuboid_observations_native(
            bboxes,
            transforms,
            transforms,
            np.ones(1, dtype=bool),
            track_min_centroid_rig_dist_m=1.0,
            device=torch.device("cpu"),
        )

        np.testing.assert_array_equal(keep, np.ones(1, dtype=bool))

    def test_consolidation_uses_bounded_cpu_batches_after_source_filtering(self):
        bbox = BBox3((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        rows = [
            {
                "track_id": f"keep-{idx}",
                "timestamp_us": idx,
                "reference_frame_id": "sensor",
                "reference_frame_timestamp_us": idx,
                "bbox3": bbox,
                "class_id": "car",
                "source": types.SimpleNamespace(name="keep"),
                "source_version": None,
            }
            for idx in range(5)
        ]
        rows.extend(
            {
                "track_id": f"drop-{idx}",
                "timestamp_us": 100 + idx,
                "reference_frame_id": "sensor",
                "reference_frame_timestamp_us": 100 + idx,
                "bbox3": bbox,
                "class_id": "car",
                "source": types.SimpleNamespace(name="drop"),
                "source_version": None,
            }
            for idx in range(2)
        )
        sequence_loader = Mock()
        sequence_loader.pose_graph.evaluate_poses.side_effect = lambda _source, target, _timestamp: (
            np.eye(4, dtype=np.float64) if target == "world" else (_ for _ in ()).throw(KeyError(target))
        )
        batch_calls = []

        def transform_spy(*args, **kwargs):
            batch_calls.append((len(args[0]), kwargs["device"]))
            return _transform_cuboid_observations_native(*args, **kwargs)

        with (
            patch(
                "nurec.datasets.utils.ncore.data.CuboidTrackObservation",
                side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs),
            ),
            patch("nurec.datasets.utils._transform_cuboid_observations_native", side_effect=transform_spy),
        ):
            tracks = consolidate_cuboid_tracks(
                cuboids_df=pd.DataFrame.from_records(rows),
                sequence_loader=sequence_loader,
                track_label_sources=["keep"],
                track_min_centroid_rig_dist_m=0.0,
                T_world_world_base=None,
                tqdm_disabled=True,
                association_batch_size=2,
            )

        assert [batch_size for batch_size, _ in batch_calls] == [2, 2, 1]
        assert all(device == torch.device("cpu") for _, device in batch_calls)
        assert set(tracks) == {f"keep-{idx}" for idx in range(5)}


class TestComputeFunctions(unittest.TestCase):
    def generate_mask(self):
        self.camera_name = "dummy_camera_name"
        self.lidar_name = "dummy_lidar_name"
        self.dims = (16, 20)
        self.num_lidar_points = 200
        self.mask_ref = np.random.choice([True, False], size=self.dims)
        self.packed_mask = PackedMask(self.mask_ref)
        self.num_frames = 2

    def test_compute_valid_pixels_ego(self):
        self.generate_mask()

        masks = compute_valid_pixels_ego(
            camera_frame_ranges={self.camera_name: range(self.num_frames)},
            cameras_valid_pixels_ego_masks={self.camera_name: self.packed_mask},
        )

        assert len(masks[self.camera_name]) == self.num_frames

        for i in range(self.num_frames):
            np.testing.assert_array_equal(self.mask_ref, masks[self.camera_name][i].unpacked())

    def test_compute_cameras_valid_pixels_frame_mask(self):
        self.generate_mask()

        base_mask = np.random.choice([True, False], size=self.dims)
        tree_mask = np.random.choice([True, False], size=self.dims)
        n_dilation_iterations = 2
        config = ValidPixelsFrameMaskConfig(
            n_dilation_iterations=n_dilation_iterations,
            classes=["road", "trees"],
        )

        def get_frame_generic_data(continuous_frame_index: int, name: str) -> np.ndarray:
            if name == "trees":
                return tree_mask
            return base_mask

        camera_sensor = Mock()
        camera_sensor.get_frame_generic_data = get_frame_generic_data

        masks = compute_valid_pixels_ego(
            camera_frame_ranges={self.camera_name: range(self.num_frames)},
            cameras_valid_pixels_ego_masks={self.camera_name: self.packed_mask},
        )

        masks = compute_cameras_valid_pixels_frame_mask(
            camera_sensors={self.camera_name: camera_sensor},
            camera_frame_ranges={self.camera_name: range(self.num_frames)},
            valid_pixels_frame_mask_params=config,
            cameras_frame_valid_pixels_masks=masks,
            tqdm_disabled=False,
        )

        # combine & dilate masks
        include_mask = np.logical_or(base_mask, tree_mask)
        exclude_mask = ndimage.binary_dilation(
            np.logical_not(include_mask),
            iterations=n_dilation_iterations,
        )
        include_mask = np.logical_not(exclude_mask)

        for i in range(self.num_frames):
            np.testing.assert_array_equal(self.mask_ref & include_mask, masks[self.camera_name][i].unpacked())

    def test_compute_valid_pixels_sceneflow(self):
        self.generate_mask()

        masks = compute_valid_pixels_ego(
            camera_frame_ranges={self.camera_name: range(self.num_frames)},
            cameras_valid_pixels_ego_masks={self.camera_name: self.packed_mask},
        )

        flow = np.zeros(self.dims, np.float32)
        flow[1:3, 2:5] = 5.0
        # This area will be dilated and undilated
        # 0 0 0 0 0 0 0 ...
        # 0 0 1 1 1 0 0 ...
        # 0 0 1 1 1 0 0 ...
        # 0 0 0 0 0 0 0 ...
        # After that it changes to this:
        mask_copy = self.mask_ref.copy()
        mask_copy[2:4, 2:6] = False
        # 0 0 0 0 0 0 0 ...
        # 0 0 0 0 0 0 0 ...
        # 0 0 1 1 1 1 0 ...
        # 0 0 1 1 1 1 0 ...
        # 0 0 0 0 0 0 0 ...

        aux_loader = Mock()
        aux_loader.get_scene_flow_magnitude = lambda camera_id, timestamp: flow
        camera_sensor = Mock()
        camera_sensor.get_frame_timestamp_us = lambda x: x * 33333
        camera_model = Mock()
        camera_model.resolution = torch.tensor([self.dims[1], self.dims[0]], dtype=torch.int32)
        frame_range = range(self.num_frames)
        valid_pixels_scene_flow_config = ValidPixelsSceneFlowConfig(
            flow_min_speed_ms=0.1,
            flow_dilate_radius=1,
            flow_downsample_scale=2,
        )

        masks = compute_valid_pixels_sceneflow(
            aux_loader,
            camera_sensors={self.camera_name: camera_sensor},
            camera_models={self.camera_name: camera_model},
            camera_frame_ranges={self.camera_name: frame_range},
            valid_pixels_scene_flow_config=valid_pixels_scene_flow_config,
            cameras_frame_valid_pixels_masks=masks,
        )

        for i in range(self.num_frames):
            np.testing.assert_array_equal(mask_copy, masks[self.camera_name][i].unpacked())

    def test_compute_valid_pixels_trafficlight(self):
        self.generate_mask()

        masks = compute_valid_pixels_ego(
            camera_frame_ranges={self.camera_name: range(self.num_frames)},
            cameras_valid_pixels_ego_masks={self.camera_name: self.packed_mask},
        )

        class_ids = [0, 16, 20]
        trafficlight_class_id = 16
        semantic_segmentation = np.random.choice(class_ids, size=self.dims)

        aux_loader = Mock()
        # Semantic Segmentation is a PILImage.Image of class ids
        aux_loader.get_semantic_segmentation = lambda c, t: Image.fromarray(semantic_segmentation.astype(np.uint8))

        camera_sensor = Mock()
        camera_sensor.get_frame_timestamp_us = lambda frame_idx: 33333 * frame_idx
        camera_frame_range = range(self.num_frames)
        valid_pixels_traffic_light_params = ValidPixelsTrafficLightConfig(seg_dilate_radius=1)

        masks = compute_valid_pixels_trafficlight(
            aux_loader,
            camera_sensors={self.camera_name: camera_sensor},
            camera_frame_ranges={self.camera_name: camera_frame_range},
            valid_pixels_traffic_light_params=valid_pixels_traffic_light_params,
            sensor_trafficlight_class_ids={self.camera_name: trafficlight_class_id},
            cameras_frame_valid_pixels_masks=masks,
            tqdm_disabled=False,
        )

        for i in range(self.num_frames):
            light_mask = semantic_segmentation == trafficlight_class_id
            assert np.all(np.logical_not(masks[self.camera_name][i].unpacked()[light_mask]))

    def test_compute_valid_lidarpoints_trafficlight_cameravisible(self):
        self.generate_mask()

        class_ids = [0, 16, 20]
        trafficlight_class_id = 16
        semantic_segmentation = np.random.choice(class_ids, size=(self.num_lidar_points,))
        visibility_mask = np.random.choice([True, False], size=(self.num_lidar_points,))

        aux_loader = Mock()
        aux_loader.get_lidar_semantic_segmentation = lambda lidar_id, frame_timestamps_us: semantic_segmentation
        aux_loader.get_lidar_camera_visibility = lambda lidar_id, frame_timestamps_us, camera_ids: {
            self.camera_name: visibility_mask
        }

        all_camera_ids = [self.camera_name]

        camera_sensor = Mock()
        camera_sensor.get_frame_timestamp_us = lambda frame_idx: 33333 * frame_idx

        lidar_sensor = Mock()
        lidar_sensor.get_frames_timestamps_us = lambda: np.array(
            [frame_idx * 100000 for frame_idx in range(self.num_frames)]
        )
        lidar_sensor.get_frame_timestamp = lambda frame_idx: 100000 * frame_idx

        time_range_us = HalfClosedInterval(0, 100000)

        masks: dict[str, dict[int, PackedMask]] = defaultdict(dict)

        masks = compute_valid_lidarpoints_trafficlight_cameravisible(
            aux_loader,
            lidar_sensors={self.lidar_name: lidar_sensor},
            all_camera_ids=all_camera_ids,
            time_range_us=time_range_us,
            sensor_trafficlight_class_ids={self.lidar_name: trafficlight_class_id},
            lidars_frame_valid_points_masks=masks,
            tqdm_disabled=False,
        )

        assert len(masks[self.lidar_name]) == self.num_frames

        for i in range(self.num_frames):
            expected_result = np.logical_and(semantic_segmentation != trafficlight_class_id, visibility_mask)
            np.testing.assert_array_equal(expected_result, masks[self.lidar_name][i].unpacked())

    def test_compute_points_outside_tracks(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Case 1: all points outside (zero-sized box)
        tracks_all_outside = [BBox3((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
        with self.subTest("all_outside"):
            outside_mask, cuboid_T, track_dim = compute_points_outside_tracks(
                points,
                tracks_all_outside,
                [0.0, 0.0, 0.0],
            )
            np.testing.assert_array_equal(outside_mask.cpu().numpy(), np.array([True, True, True, True]))
            assert cuboid_T.shape == (1, 4, 4) or cuboid_T.shape == (4, 4)
            np.testing.assert_allclose(track_dim.cpu().numpy(), np.array([[0.0, 0.0, 0.0]], dtype=np.float32))

        # Case 2: all points inside (large box)
        tracks_all_inside = [BBox3((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (0.0, 0.0, 0.0))]
        with self.subTest("all_inside"):
            outside_mask, cuboid_T, track_dim = compute_points_outside_tracks(
                points,
                tracks_all_inside,
                [0.0, 0.0, 0.0],
            )
            np.testing.assert_array_equal(outside_mask.cpu().numpy(), np.array([False, False, False, False]))
            assert cuboid_T.shape == (1, 4, 4) or cuboid_T.shape == (4, 4)
            np.testing.assert_allclose(track_dim.cpu().numpy(), np.array([[10.0, 10.0, 10.0]], dtype=np.float32))

        # Case 3: some points outside (box size 2, half-extent 1)
        tracks_some_outside = [BBox3((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))]
        with self.subTest("some_outside"):
            outside_mask, cuboid_T, track_dim = compute_points_outside_tracks(
                points,
                tracks_some_outside,
                [0.0, 0.0, 0.0],
            )
            np.testing.assert_array_equal(outside_mask.cpu().numpy(), np.array([False, False, True, True]))
            assert cuboid_T.shape == (1, 4, 4) or cuboid_T.shape == (4, 4)
            np.testing.assert_allclose(track_dim.cpu().numpy(), np.array([[2.0, 2.0, 2.0]], dtype=np.float32))

        # Padding increases box; with extent 3 (half-extent 1.5), x=2 remains outside and x=3 remains outside
        with self.subTest("some_outside_with_padding"):
            outside_mask_padded, _, track_dim_padded = compute_points_outside_tracks(
                points,
                tracks_some_outside,
                [3.0, 3.0, 3.0],
            )
            np.testing.assert_array_equal(outside_mask_padded.cpu().numpy(), np.array([False, False, False, True]))
            np.testing.assert_allclose(track_dim_padded.cpu().numpy(), np.array([[5.0, 5.0, 5.0]], dtype=np.float32))

    def test_compute_points_outside_tracks_preserves_bbox_property_error(self):
        class MissingBBox:
            @property
            def bbox3(self) -> BBox3:
                raise AttributeError("bbox3 is unavailable")

        labels: list[BBox3Provider] = [MissingBBox()]
        with self.assertRaisesRegex(AttributeError, "bbox3 is unavailable"):
            compute_points_outside_tracks(
                np.zeros((1, 3), dtype=np.float32),
                labels,
                [0.0, 0.0, 0.0],
            )

    def test_compute_points_outside_tracks_rejects_unsupported_labels(self):
        with self.assertRaisesRegex(TypeError, "unsupported bbox_label type"):
            compute_points_outside_tracks(
                np.zeros((1, 3), dtype=np.float32),
                [object()],  # type: ignore[list-item]
                [0.0, 0.0, 0.0],
            )

    def test_compute_point_cloud_inside_tracks_mask(self):
        # Build a simple point cloud on CUDA to match implementation device
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        pc = PointCloud(xyz_start=points.clone(), xyz_end=points.clone())

        time_range_us = HalfClosedInterval(0, 200000)

        def make_cuboid_tracks(
            center_xyz: tuple[float, float, float], dims_lwh: tuple[float, float, float]
        ) -> CuboidTracks:
            # Two identical poses to satisfy Tracks.Factory requirement (>=2 poses per track)
            T = np.eye(4, dtype=np.float32)
            T[:3, 3] = np.array(center_xyz, dtype=np.float32)
            poses = np.stack([T, T], axis=0).astype(np.float32)
            timestamps = np.array([0, 100000], dtype=np.int64)
            return CuboidTracks.Factory.from_numpy(
                tracks_id=["track_0"],
                tracks_poses=[poses],
                tracks_timestamps_us=[timestamps],
                tracks_label_class=["car"],
                tracks_flags=[TrackFlags.DYNAMIC],
                cuboids_dims=[np.array(dims_lwh, dtype=np.float32)],
            )

        # Case 1: all points outside tracks => all False
        with self.subTest("all_outside"):
            cuboidtracks_dynamic = make_cuboid_tracks(center_xyz=(10.0, 0.0, 0.0), dims_lwh=(1.0, 1.0, 1.0))
            mask = compute_point_cloud_inside_tracks_mask(
                pc,
                time_range_us=time_range_us,
                cuboidtracks_dynamic=cuboidtracks_dynamic,
                track_padding_m=[0.0, 0.0, 0.0],
                batch_size=1,
                tqdm_disabled=True,
            )
            torch.testing.assert_close(
                mask, torch.tensor([False, False, False, False], dtype=torch.bool, device="cuda")
            )

        # Case 2: all points inside tracks => all True
        with self.subTest("all_inside"):
            cuboidtracks_dynamic = make_cuboid_tracks(center_xyz=(1.5, 0.0, 0.0), dims_lwh=(10.0, 10.0, 10.0))
            mask = compute_point_cloud_inside_tracks_mask(
                pc,
                time_range_us=time_range_us,
                cuboidtracks_dynamic=cuboidtracks_dynamic,
                track_padding_m=[0.0, 0.0, 0.0],
                batch_size=1,
                tqdm_disabled=True,
            )
            torch.testing.assert_close(mask, torch.tensor([True, True, True, True], dtype=torch.bool, device="cuda"))

        # Case 3: mixed outside/inside => some False, some True
        with self.subTest("some_inside"):
            # Box of length 2 centered at 0.5 covers x in [-0.5, 1.5] -> points 0 and 1 inside
            cuboidtracks_dynamic = make_cuboid_tracks(center_xyz=(0.5, 0.0, 0.0), dims_lwh=(2.0, 2.0, 2.0))
            mask = compute_point_cloud_inside_tracks_mask(
                pc,
                time_range_us=time_range_us,
                cuboidtracks_dynamic=cuboidtracks_dynamic,
                track_padding_m=[0.0, 0.0, 0.0],
                batch_size=1,
                tqdm_disabled=True,
            )
            torch.testing.assert_close(mask, torch.tensor([True, True, False, False], dtype=torch.bool, device="cuda"))

        # Case 4: semantic class restriction - only test specified classes
        with self.subTest("semantic_class_restriction"):
            # Create point cloud with semantic classes: [0=dynamic, 1=static, 0=dynamic, 1=static]
            semantic_class_id = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="cuda")
            pc_semantic = PointCloud(
                xyz_start=points.clone(), xyz_end=points.clone(), semantic_class_id=semantic_class_id
            )
            # All points are geometrically inside the track
            cuboidtracks_dynamic = make_cuboid_tracks(center_xyz=(1.5, 0.0, 0.0), dims_lwh=(10.0, 10.0, 10.0))
            mask = compute_point_cloud_inside_tracks_mask(
                pc_semantic,
                time_range_us=time_range_us,
                cuboidtracks_dynamic=cuboidtracks_dynamic,
                track_padding_m=[0.0, 0.0, 0.0],
                batch_size=1,
                tqdm_disabled=True,
                restrict_to_class_ids=[0],  # Only test class 0 (dynamic)
            )
            # Only points with class 0 are tested and marked as inside; class 1 always False
            torch.testing.assert_close(mask, torch.tensor([True, False, True, False], dtype=torch.bool, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the CUDA intersection binding")
# 15 x 11 does not divide by subsample 2 or 3.
@pytest.mark.parametrize("width,height", [(16, 12), (15, 11)])
@pytest.mark.parametrize("subsample", [1, 2, 3])
def test_batched_visible_intervals_match_per_frame_path(subsample, width, height):
    params = OpenCVPinholeCameraModelParameters(
        resolution=np.array([width, height], dtype=np.uint64),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
        principal_point=np.array([width / 2, height / 2], dtype=np.float32),
        focal_length=np.array([12, 12], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )
    model = camera_model_from_parameters(params)
    pixels = generate_grid_2d_indices((width, height)).numpy()
    rays = model.pixels_to_camera_rays(torch.from_numpy(pixels)).reshape(height, width, 3).cpu().numpy()

    # More than 16 frames, so the batched path launches several chunks.
    n_frames = 20

    def poses(target, index, timepoint):
        index = np.atleast_1d(index)
        transforms = np.repeat(np.eye(4)[None], len(index), 0)
        transforms[:, 0, 3] = index * 0.1 + (0.05 if timepoint == FrameTimepoint.END else 0.0)
        return transforms if len(index) > 1 else transforms[0]

    def frame_time(index, timepoint):
        return int(index) * 100_000 + (33_333 if timepoint == FrameTimepoint.END else 0)

    sensor = Mock(get_frames_T_sensor_target=poses, get_frame_timestamp_us=frame_time)

    def track(x, z, times):
        transforms = np.repeat(np.eye(4)[None], len(times), 0)
        transforms[:, 0, 3], transforms[:, 2, 3] = x, z
        transforms[-1, 1, 3] = 0.5
        return transforms.astype(np.float32)

    # The second track starts inside frame 10's exposure, so its intervals depend on lifetime clipping.
    track_times = [np.array([0, 2_000_000], dtype=np.int64), np.array([1_016_000, 1_900_000], dtype=np.int64)]
    tracks = CuboidTracks.Factory.from_numpy(
        tracks_id=["a", "b"],
        tracks_poses=[track(0.5, 6.0, track_times[0]), track(1.5, 8.0, track_times[1])],
        tracks_timestamps_us=track_times,
        tracks_label_class=["car", "car"],
        tracks_flags=[TrackFlags(0), TrackFlags(0)],
        cuboids_dims=[np.full(3, 3.0, dtype=np.float32), np.full(3, 2.0, dtype=np.float32)],
        device=torch.device("cuda"),
    )
    common = dict(
        cuboidtracks=tracks,
        camera_sensors={"camera": sensor},
        camera_models={"camera": model},
        camera_frame_ranges={"camera": range(n_frames)},
        cameras_all_pixels={"camera": pixels},
        cameras_all_rays={"camera": rays},
        subsample=subsample,
        tqdm_disabled=True,
    )
    with patch(
        "nurec.datasets.utils._camera_visible_intervals_batched",
        wraps=utils_module._camera_visible_intervals_batched,
    ) as batched_spy:
        batched = compute_camera_visible_intervals_cuboid_tracks(**common)
        # Requesting track labels forces the per-frame path.
        per_frame = compute_camera_visible_intervals_cuboid_tracks(**common, cameras_frame_track_idxs={"camera": {}})
    assert batched_spy.call_count == 1
    assert len(batched) == len(per_frame) == 2
    assert all(len(intervals) for intervals in batched)
    for got, want in zip(batched, per_frame, strict=True):
        np.testing.assert_array_equal(got, want)
