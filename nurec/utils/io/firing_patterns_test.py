# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import tempfile
import unittest
import zipfile

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import ncore.data

from nurec.artifact.artifact import Artifact
from nurec.config.checkpoint import ArtifactConfig
from nurec.utils.io.artifact_cache import ArtifactCache
from nurec.utils.io.firing_patterns import serialize_firing_patterns
from nurec.utils.types import FrameConversion, RigTrajectories


class _FakeSensor:
    """A record that lists every slot fired, so one frame is the whole pattern.

    Structurally what ``recover_firing_pattern`` reads: frame spans, per-frame directions and
    timestamps, and the return distance it uses to tell which kind of record this is.
    """

    def __init__(self, n_rays: int = 8, n_frames: int = 4):
        self.frames_count = n_frames
        self._n_rays = n_rays
        # (N, 2) even when empty, as ncore guarantees; a bare (N,) would fail differently.
        self.frames_timestamps_us = np.array(
            [[i * 1000, i * 1000 + 900] for i in range(n_frames)], dtype=np.int64
        ).reshape(n_frames, 2)

    def get_frame_ray_bundle_direction(self, frame_index: int) -> np.ndarray:
        angles = np.linspace(-0.4, 0.4, self._n_rays)
        return np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1).astype(np.float32)

    def get_frame_ray_bundle_timestamp_us(self, frame_index: int) -> np.ndarray:
        return self.frames_timestamps_us[frame_index][0] + np.arange(self._n_rays, dtype=np.int64) * 100

    def get_frame_ray_bundle_return_distance_m(self, frame_index: int, return_index: int) -> np.ndarray:
        # Half the slots came back; the rest are the non-returning ones that mark this record.
        distances = np.full(self._n_rays, 10.0, dtype=np.float64)
        distances[::2] = np.nan
        return distances


def _lidar_model_parameters() -> ncore.data.ConcreteLidarModelParametersUnion:
    """Minimal spinning lidar intrinsics; only their presence matters here."""
    return ncore.data.RowOffsetStructuredSpinningLidarModelParameters(
        spinning_frequency_hz=10.0,
        spinning_direction="ccw",
        n_rows=2,
        n_columns=4,
        row_elevations_rad=np.linspace(0.1, -0.1, 2).astype(np.float32),
        column_azimuths_rad=np.linspace(-0.5, 0.5, 4).astype(np.float32),
        row_azimuth_offsets_rad=np.zeros(2, dtype=np.float32),
    )


def _rig_trajectories(*lidars: tuple[str, bool]) -> RigTrajectories:
    """Calibrations for ``(logical_id, has_model_parameters)``, keyed by unique sensor id."""
    calibrations: OrderedDict[str, RigTrajectories.LidarCalibration] = OrderedDict()
    for idx, (logical_id, has_model) in enumerate(lidars):
        calibrations[f"{logical_id}@sequence{idx}"] = RigTrajectories.LidarCalibration(
            sequence_id="sequence",
            logical_sensor_name=logical_id,
            unique_sensor_idx=idx,
            T_sensor_rig=torch.eye(4, dtype=torch.float32),
            lidar_model_parameters=_lidar_model_parameters() if has_model else None,
        )
    return RigTrajectories(
        T_world_base=torch.eye(4, dtype=torch.float64),
        world_to_nurec=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
        rig_trajectories=[],
        camera_calibrations=OrderedDict(),
        lidar_calibrations=calibrations,
    )


class TestSerializeFiringPatterns(unittest.TestCase):
    def test_writes_one_npz_per_model_free_lidar(self):
        members = serialize_firing_patterns(_rig_trajectories(("lidar_fu", False)), {"lidar_fu": _FakeSensor()})

        self.assertEqual([m.filename for m in members], ["firing_patterns/lidar_fu.npz"])
        with np.load(io.BytesIO(members[0].serialized)) as data:
            self.assertEqual(data["directions_sensor"].shape, (8, 3))
            self.assertEqual(data["timestamp_offsets_us"].shape, (8,))
            # Frames start every 1000us and carry 900us of rays: the frame, not the sweep.
            self.assertEqual(int(data["frame_duration_us"]), 1000)
            self.assertEqual(int(data["timestamp_offsets_us"].max()), 700)

    def test_skips_a_lidar_that_has_a_sensor_model(self):
        # Serving synthesises those rays from the model, which covers the whole grid.
        members = serialize_firing_patterns(
            _rig_trajectories(("lidar_modelled", True), ("lidar_free", False)),
            {"lidar_modelled": _FakeSensor(), "lidar_free": _FakeSensor()},
        )

        self.assertEqual([m.filename for m in members], ["firing_patterns/lidar_free.npz"])

    def test_duplicate_logical_ids_are_refused(self):
        # The member name is the logical id, so a second one would write a duplicate zip entry.
        rig = _rig_trajectories(("lidar_top", False), ("lidar_top", False))
        with self.assertRaisesRegex(ValueError, "share logical id"):
            serialize_firing_patterns(rig, {"lidar_top": _FakeSensor()})

    def test_refuses_a_lidar_the_datasource_does_not_carry(self):
        with self.assertRaisesRegex(ValueError, "no datasource sensor.*lidar_fu"):
            serialize_firing_patterns(_rig_trajectories(("lidar_fu", False)), {})

    def test_refuses_an_id_that_cannot_name_an_artifact_member(self):
        with self.assertRaisesRegex(ValueError, "contains '/'"):
            serialize_firing_patterns(
                _rig_trajectories(("bad/id", False)),
                {"bad/id": _FakeSensor()},
            )

    def test_refuses_a_pattern_recovery_failure(self):
        with self.assertRaisesRegex(ValueError, "could not recover.*lidar_fu.*no full sweep"):
            serialize_firing_patterns(
                _rig_trajectories(("lidar_fu", False)),
                {"lidar_fu": _FakeSensor(n_frames=0)},
            )

    def test_reports_every_missing_required_pattern_together(self):
        with self.assertRaises(ValueError) as caught:
            serialize_firing_patterns(
                _rig_trajectories(("lidar_left", False), ("lidar_right", False)),
                {},
            )

        self.assertIn("lidar_left", str(caught.exception))
        self.assertIn("lidar_right", str(caught.exception))


class TestFiringPatternsRoundTrip(unittest.TestCase):
    """What the export writes is what the artifact reader finds."""

    def test_patterns_reach_the_usdz_and_read_back(self):
        cache = ArtifactCache(artifact_config=ArtifactConfig())
        cache.firing_patterns = serialize_firing_patterns(
            _rig_trajectories(("lidar_fu", False), ("lidar:top:center", False)),
            {"lidar_fu": _FakeSensor(n_rays=8), "lidar:top:center": _FakeSensor(n_rays=5)},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.usdz"
            cache.write_to_usdz(path)

            # ArtifactCache.archive_members picks the field up by type, with no wiring of its own.
            members = zipfile.ZipFile(path).namelist()
            self.assertIn("firing_patterns/lidar_fu.npz", members)

            patterns = Artifact(path).firing_patterns
            self.assertEqual(sorted(patterns), ["lidar:top:center", "lidar_fu"])
            with np.load(io.BytesIO(patterns["lidar_fu"])) as data:
                self.assertEqual(data["directions_sensor"].shape, (8, 3))
                self.assertEqual(int(data["frame_duration_us"]), 1000)


if __name__ == "__main__":
    unittest.main()
