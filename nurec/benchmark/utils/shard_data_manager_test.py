# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Shard Data Manager utility."""

import unittest

from unittest.mock import MagicMock, Mock, patch

import numpy as np

from nurec.benchmark.utils.shard_data_manager import ShardDataManager


class MockBBox3:
    """Mock 3D bounding box for testing."""

    def __init__(self, center, dim, rot):
        """Initialize mock bbox.

        Args:
            center: Center coordinates [x, y, z]
            dim: Dimensions [length, width, height]
            rot: Rotation [roll, pitch, yaw] in radians
        """
        self._data = np.concatenate([center, dim, rot])

    def to_array(self):
        """Return bbox as array."""
        return self._data


class TestShardDataManager(unittest.TestCase):
    """Test cases for ShardDataManager class."""

    def test_box2corners_simple_box(self) -> None:
        """Test converting 3D bbox to corner points."""
        # Create a simple box: 2x2x2 centered at origin, no rotation
        center = np.array([0.0, 0.0, 0.0])
        dim = np.array([2.0, 2.0, 2.0])
        rot = np.array([0.0, 0.0, 0.0])  # No rotation

        bbox3 = MockBBox3(center, dim, rot)
        corners = ShardDataManager.box2corners(bbox3)

        # Should have 8 corners
        self.assertEqual(corners.shape, (8, 3))

        # Check that corners are at expected positions
        # For a 2x2x2 box at origin, corners should be at ±1 in each dim
        expected_corners = np.array(
            [
                [1, 1, 1],  # front-top-right
                [-1, 1, 1],  # front-top-left
                [-1, -1, 1],  # front-bottom-left
                [1, -1, 1],  # front-bottom-right
                [1, -1, -1],  # back-bottom-right
                [-1, -1, -1],  # back-bottom-left
                [-1, 1, -1],  # back-top-left
                [1, 1, -1],  # back-top-right
            ]
        )

        np.testing.assert_array_almost_equal(corners, expected_corners)

    def test_box2corners_translated_box(self) -> None:
        """Test bbox corners with translation."""
        center = np.array([10.0, 5.0, 2.0])
        dim = np.array([2.0, 2.0, 2.0])
        rot = np.array([0.0, 0.0, 0.0])

        bbox3 = MockBBox3(center, dim, rot)
        corners = ShardDataManager.box2corners(bbox3)

        # All corners should be within ±1 of center
        np.testing.assert_array_almost_equal(corners.mean(axis=0), center)

    def test_box2corners_different_dimensions(self) -> None:
        """Test bbox corners with non-uniform dimensions."""
        center = np.array([0.0, 0.0, 0.0])
        dim = np.array([4.0, 2.0, 1.0])  # length=4, width=2, height=1
        rot = np.array([0.0, 0.0, 0.0])

        bbox3 = MockBBox3(center, dim, rot)
        corners = ShardDataManager.box2corners(bbox3)

        # Check dimensions: x should span ±2, y should span ±1, z should span ±0.5
        self.assertAlmostEqual(corners[:, 0].max() - corners[:, 0].min(), 4.0)
        self.assertAlmostEqual(corners[:, 1].max() - corners[:, 1].min(), 2.0)
        self.assertAlmostEqual(corners[:, 2].max() - corners[:, 2].min(), 1.0)

    def test_box2corners_with_rotation(self) -> None:
        """Test bbox corners with 90-degree yaw rotation."""
        center = np.array([0.0, 0.0, 0.0])
        dim = np.array([4.0, 2.0, 1.0])
        rot = np.array([0.0, 0.0, np.pi / 2])  # 90-degree yaw

        bbox3 = MockBBox3(center, dim, rot)
        corners = ShardDataManager.box2corners(bbox3)

        # After 90-degree yaw, x and y extents swap
        self.assertAlmostEqual(corners[:, 0].max() - corners[:, 0].min(), 2.0)
        self.assertAlmostEqual(corners[:, 1].max() - corners[:, 1].min(), 4.0)
        self.assertAlmostEqual(corners[:, 2].max() - corners[:, 2].min(), 1.0)

    @staticmethod
    def _create_mock_observation(
        reference_frame_timestamp_us: int, track_id: str, class_id: str, timestamp_us: int | None = None
    ) -> MagicMock:
        """Create a mock cuboid track observation.

        ``timestamp_us`` defaults to the reference-frame timestamp (V4-export behavior);
        pass a different value to model V3 per-label in-frame observation times.
        """
        observation = MagicMock()
        observation.reference_frame_timestamp_us = reference_frame_timestamp_us
        observation.timestamp_us = reference_frame_timestamp_us if timestamp_us is None else timestamp_us
        observation.track_id = track_id
        observation.class_id = class_id
        observation.bbox3 = MockBBox3(np.zeros(3), np.array([2.0, 2.0, 2.0]), np.zeros(3))
        return observation

    def _create_mock_loader(self, camera_ids=("camera_front",), observations=None, camera_timestamps=None) -> MagicMock:
        """Create a mock sequence loader implementing the protocol surface used here."""
        loader = MagicMock()
        loader.camera_ids = list(camera_ids)

        camera_sensor = MagicMock()
        camera_sensor.model_parameters = {}
        if camera_timestamps is None:
            camera_timestamps = np.array([1000, 2000, 3000], dtype=np.uint64)
        camera_sensor.get_frames_timestamps_us.return_value = np.asarray(camera_timestamps, dtype=np.uint64)
        loader.get_camera_sensor.return_value = camera_sensor

        if observations is None:
            observations = [self._create_mock_observation(1000, "7", "pedestrian")]
        loader.get_cuboid_track_observations.return_value = iter(observations)
        return loader

    @patch("nurec.benchmark.utils.shard_data_manager.camera_model_from_parameters")
    @patch("nurec.benchmark.utils.shard_data_manager.require_explicit_sequence_input_provider")
    def test_initialization_v3_glob(self, mock_require_provider: Mock, mock_camera_model: Mock) -> None:
        """Test initialization from an explicit path pattern."""
        mock_loader = self._create_mock_loader()
        mock_require_provider.return_value.open_group.return_value = (
            MagicMock(),
            mock_loader,
            ("shard1.zarr", "shard2.zarr"),
        )
        mock_camera_model.return_value = MagicMock()

        manager = ShardDataManager("shard*.zarr", "camera_front")

        self.assertEqual(manager.camera_id, "camera_front")
        self.assertEqual(manager.shard_pattern, "shard*.zarr")
        mock_require_provider.return_value.open_group.assert_called_once_with(
            "shard*.zarr",
            open_consolidated=True,
            loader_max_workers=None,
        )

        # Camera sensor is the loader's protocol sensor (no adapter)
        self.assertIs(manager.camera_sensor, mock_loader.get_camera_sensor.return_value)
        mock_camera_model.assert_called_once()

        # No meta file -> no interval restriction on label loading
        _, obs_kwargs = mock_loader.get_cuboid_track_observations.call_args
        self.assertIsNone(obs_kwargs["timestamp_interval_us"])

    @patch("nurec.benchmark.utils.shard_data_manager.require_explicit_sequence_input_provider")
    def test_initialization_no_shard_files(self, mock_require_provider: Mock) -> None:
        """Test initialization fails with no shard files."""
        mock_require_provider.return_value.open_group.side_effect = ValueError("No shard files found")

        with self.assertRaises(ValueError) as context:
            ShardDataManager("nonexistent*.zarr", "camera_front")

        self.assertIn("No shard files found", str(context.exception))

    @patch("nurec.benchmark.utils.shard_data_manager.camera_model_from_parameters")
    @patch("nurec.benchmark.utils.shard_data_manager.require_explicit_sequence_input_provider")
    def test_initialization_invalid_camera(self, mock_require_provider: Mock, mock_camera_model: Mock) -> None:
        """Test initialization fails with invalid camera ID."""
        mock_require_provider.return_value.open_group.return_value = (
            MagicMock(),
            self._create_mock_loader(camera_ids=("camera_left", "camera_right")),
            ("shard1.zarr",),
        )

        with self.assertRaises(ValueError) as context:
            ShardDataManager("shard*.zarr", "camera_front")

        self.assertIn("not found in shard", str(context.exception))
        # Should fail before loading camera model
        mock_camera_model.assert_not_called()

    @patch("nurec.benchmark.utils.shard_data_manager.camera_model_from_parameters")
    @patch("nurec.benchmark.utils.shard_data_manager.require_explicit_sequence_input_provider")
    def test_initialization_no_observations(self, mock_require_provider: Mock, mock_camera_model: Mock) -> None:
        """Test that a label-less clip loads with an empty timeline (empty metrics, not an error)."""
        mock_require_provider.return_value.open_group.return_value = (
            MagicMock(),
            self._create_mock_loader(observations=[]),
            ("shard1.zarr",),
        )
        mock_camera_model.return_value = MagicMock()

        with self.assertLogs("nurec.benchmark.utils.shard_data_manager", level="WARNING"):
            manager = ShardDataManager("shard*.zarr", "camera_front")

        self.assertEqual(manager.label_timestamps_us.size, 0)
        self.assertEqual(manager.compile_3d_labels(), {})

    @patch("nurec.benchmark.utils.shard_data_manager.camera_model_from_parameters")
    @patch("nurec.benchmark.utils.shard_data_manager.ncore_utils")
    @patch("nurec.benchmark.utils.shard_data_manager.resolve_dataset")
    def test_initialization_dataset_meta_json(
        self, mock_resolve_dataset: Mock, mock_ncore_utils: Mock, mock_camera_model: Mock
    ) -> None:
        """Test loading from a dataset meta json and compiling labels from observations."""
        mock_loader = self._create_mock_loader(
            observations=[
                self._create_mock_observation(2000, "42", "automobile"),
                self._create_mock_observation(1000, "7", "pedestrian"),
            ]
        )
        mock_resolved = mock_resolve_dataset.return_value
        mock_resolved.metadata.time_range_start_us = 500
        mock_resolved.metadata.time_range_end_us = 2500
        mock_resolved.create_sequence_loader.return_value = mock_loader
        # Identity transform keeps the mocked observations
        mock_ncore_utils.transform_cuboid_track_observations.side_effect = lambda observations, **kwargs: observations
        mock_camera_model.return_value = MagicMock()

        manager = ShardDataManager("clipgt-abc.json", "camera_front")

        mock_resolve_dataset.assert_called_once()

        # The meta file's clip interval bounds label loading
        _, obs_kwargs = mock_loader.get_cuboid_track_observations.call_args
        self.assertEqual(obs_kwargs["timestamp_interval_us"].start, 500)
        self.assertEqual(obs_kwargs["timestamp_interval_us"].stop, 2500)

        # Label timeline is sorted by reference-frame timestamp and lidar-free
        np.testing.assert_array_equal(manager.label_timestamps_us, np.array([1000, 2000], dtype=np.int64))

        # Camera frames are bounded to the clip interval too: default mock camera END
        # timestamps [1000, 2000, 3000] against interval [500, 2500) keep frames 0..1
        self.assertEqual(manager.camera_frame_index_range, (0, 2))

        labels = manager.compile_3d_labels()
        self.assertEqual(set(labels.keys()), {0, 1})
        self.assertEqual(labels[0]["7"]["class"], "pedestrian")
        self.assertEqual(labels[1]["42"]["class"], "automobile")
        self.assertEqual(labels[1]["42"]["vertices"].shape, (8, 3))

        # Labels are transformed to the world frame via the shared helper
        for call in mock_ncore_utils.transform_cuboid_track_observations.call_args_list:
            self.assertEqual(call.kwargs["target_frame_id"], "world")

    @patch("nurec.benchmark.utils.shard_data_manager.camera_model_from_parameters")
    @patch("nurec.benchmark.utils.shard_data_manager.require_explicit_sequence_input_provider")
    def test_label_timeline_keyed_by_reference_frame_time(
        self, mock_require_provider: Mock, mock_camera_model: Mock
    ) -> None:
        """Timeline keys follow the reference-frame timestamp, one entry per label frame.

        V3-style data: labels within one lidar frame share reference_frame_timestamp_us but
        may carry accurate per-label observation times. Keying by the reference timestamp
        keeps the timeline at the label frame rate instead of fragmenting it per label,
        which would collapse the downstream median-interval gap guard.
        """
        observations = [
            # One label frame (reference ts 2000) with per-label in-frame observation times
            self._create_mock_observation(2000, "7", "pedestrian", timestamp_us=1980),
            self._create_mock_observation(2000, "8", "automobile", timestamp_us=1990),
            self._create_mock_observation(2000, "9", "automobile", timestamp_us=2000),
            # Second label frame
            self._create_mock_observation(3000, "7", "pedestrian", timestamp_us=2985),
        ]
        mock_require_provider.return_value.open_group.return_value = (
            MagicMock(),
            self._create_mock_loader(observations=observations),
            ("shard1.zarr",),
        )
        mock_camera_model.return_value = MagicMock()

        manager = ShardDataManager("shard*.zarr", "camera_front")

        # One timeline entry per label frame, keyed by the frame's reference timestamp
        np.testing.assert_array_equal(manager.label_timestamps_us, np.array([2000, 3000], dtype=np.int64))
        self.assertEqual(len(manager._observations_by_timestamp_us[2000]), 3)
        self.assertEqual(len(manager._observations_by_timestamp_us[3000]), 1)

    @patch("nurec.benchmark.utils.shard_data_manager.camera_model_from_parameters")
    @patch("nurec.benchmark.utils.shard_data_manager.require_explicit_sequence_input_provider")
    def test_camera_range_full_without_interval(self, mock_require_provider: Mock, mock_camera_model: Mock) -> None:
        """Without a clip interval (V3 glob), the camera range spans the full store."""
        mock_require_provider.return_value.open_group.return_value = (
            MagicMock(),
            self._create_mock_loader(camera_timestamps=np.array([1000, 2000, 3000, 4000], dtype=np.uint64)),
            ("shard1.zarr",),
        )
        mock_camera_model.return_value = MagicMock()

        manager = ShardDataManager("shard*.zarr", "camera_front")

        self.assertEqual(manager.camera_frame_index_range, (0, 4))

    @patch("nurec.benchmark.utils.shard_data_manager.camera_model_from_parameters")
    @patch("nurec.benchmark.utils.shard_data_manager.resolve_dataset")
    def test_clip_interval_without_camera_frames_rejected(
        self, mock_resolve_dataset: Mock, mock_camera_model: Mock
    ) -> None:
        """A meta whose interval contains no camera frames is a broken dataset."""
        mock_resolved = mock_resolve_dataset.return_value
        mock_resolved.metadata.time_range_start_us = 50
        mock_resolved.metadata.time_range_end_us = 90
        mock_resolved.create_sequence_loader.return_value = self._create_mock_loader(
            camera_timestamps=np.array([1000, 2000, 3000], dtype=np.uint64)
        )
        mock_camera_model.return_value = MagicMock()

        with self.assertRaisesRegex(ValueError, "clip interval"):
            ShardDataManager("clipgt-abc.json", "camera_front")

    @patch("nurec.benchmark.utils.shard_data_manager.camera_model_from_parameters")
    @patch("nurec.benchmark.utils.shard_data_manager.require_explicit_sequence_input_provider")
    def test_load_new_shard(self, mock_require_provider: Mock, mock_camera_model: Mock) -> None:
        """Test loading a new shard pattern."""
        mock_require_provider.return_value.open_group.side_effect = [
            (MagicMock(), self._create_mock_loader(), ("shard1.zarr",)),
            (MagicMock(), self._create_mock_loader(), ("shard2.zarr",)),
        ]
        mock_camera_model.return_value = MagicMock()

        manager = ShardDataManager("shard1*.zarr", "camera_front")
        self.assertEqual(manager.shard_pattern, "shard1*.zarr")

        # Load new shard
        manager.load("shard2*.zarr")

        self.assertEqual(manager.shard_pattern, "shard2*.zarr")
        self.assertEqual(mock_require_provider.return_value.open_group.call_count, 2)


if __name__ == "__main__":
    unittest.main()
