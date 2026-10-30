# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for object data loader utilities."""

import unittest

import numpy as np

from nurec.benchmark.utils.object_data_loader import (
    _corners_in_front_counts,
    index_label_tracks,
    select_track_labels,
)


def _labels(observations):
    """(timestamp_us, track_id) pairs -> (labels_3d, label_timestamps_us), grouped by timestamp."""
    by_ts = {}
    for ts, track_id in observations:
        by_ts.setdefault(ts, {})[track_id] = {"class": "automobile", "track": track_id, "ts": ts}
    timestamps = sorted(by_ts)
    return {i: by_ts[ts] for i, ts in enumerate(timestamps)}, np.asarray(timestamps, dtype=np.int64)


def _select(observations, query_us, fallback_us=0.0):
    labels_3d, label_ts = _labels(observations)
    tracks, interval_us = index_label_tracks(labels_3d, label_ts)
    fallback = interval_us if interval_us is not None else fallback_us
    return {k: v["ts"] for k, v in select_track_labels(tracks, labels_3d, query_us, fallback).items()}


class TestSelectTrackLabels(unittest.TestCase):
    TRACK = [(t, "a") for t in (0, 100_000, 200_000, 300_000)]

    def test_exact_hit(self) -> None:
        self.assertEqual(_select(self.TRACK, 100_000), {"a": 100_000})

    def test_nearer_neighbour_inside_the_span(self) -> None:
        self.assertEqual(_select(self.TRACK, 140_000), {"a": 100_000})
        self.assertEqual(_select(self.TRACK, 160_000), {"a": 200_000})

    def test_jittered_gap_is_still_covered(self) -> None:
        # 150_000 is more than half the median interval from both neighbours; it is covered only
        # because the 100_500 gap between them still counts as consecutive.
        track = [(0, "a"), (99_993, "a"), (200_493, "a"), (300_000, "a")]
        self.assertEqual(_select(track, 150_000), {"a": 99_993})

    def test_missed_sample_is_not_bridged(self) -> None:
        track = [(t, "a") for t in (0, 100_000, 300_000, 400_000, 500_000)]
        self.assertEqual(_select(track, 200_000), {})
        self.assertEqual(_select(track, 140_000), {"a": 100_000})  # still within its own cell
        self.assertEqual(_select(track, 300_000), {"a": 300_000})

    def test_sparse_track_does_not_bridge_frames_that_omit_it(self) -> None:
        # Its own gaps are 300 ms; the clip is labelled every 100 ms, so it is absent in between.
        observations = [(t, "steady") for t in (0, 100_000, 200_000, 300_000)] + [(0, "sparse"), (300_000, "sparse")]
        self.assertEqual(_select(observations, 100_000), {"steady": 100_000})
        self.assertEqual(_select(observations, 200_000), {"steady": 200_000})
        self.assertEqual(_select(observations, 290_000), {"steady": 300_000, "sparse": 300_000})

    def test_edges_within_half_an_interval(self) -> None:
        self.assertEqual(_select(self.TRACK, -49_000), {"a": 0})
        self.assertEqual(_select(self.TRACK, -51_000), {})
        self.assertEqual(_select(self.TRACK, 349_000), {"a": 300_000})
        self.assertEqual(_select(self.TRACK, 351_000), {})

    def test_single_observation_track_uses_the_clip_interval(self) -> None:
        observations = self.TRACK + [(150_000, "b")]
        self.assertEqual(_select(observations, 190_000), {"a": 200_000, "b": 150_000})
        self.assertEqual(_select(observations, 210_000), {"a": 200_000})

    def test_single_observations_only_use_the_fallback(self) -> None:
        self.assertEqual(_select([(1000, "a")], 1000), {"a": 1000})
        self.assertEqual(_select([(1000, "a")], 1001), {})
        self.assertEqual(_select([(1000, "a")], 1030, fallback_us=66), {"a": 1000})
        self.assertEqual(_select([(1000, "a")], 1040, fallback_us=66), {})

    def test_empty(self) -> None:
        self.assertEqual(index_label_tracks({}, np.array([], dtype=np.int64)), ({}, None))
        self.assertEqual(_select([], 0), {})

    def test_staggered_per_object_timestamps(self) -> None:
        # PAI-converted clips: every observation has its own reference time, so each label frame
        # holds one object. Every track present at a camera time must still be selected.
        rng = np.random.default_rng(0)
        spans = {k: (0, 20_000_000) for k in range(17)}
        spans[3] = (1_000_000, 20_000_000)
        spans[5] = (0, 19_000_000)
        observations = []
        for k, (start, stop) in spans.items():
            for t in range(start + k * 100_000 // 17, stop, 100_000):
                observations.append((t + int(rng.integers(-50, 51)), str(k)))
        labels_3d, label_ts = _labels(observations)
        self.assertTrue(all(len(frame) == 1 for frame in labels_3d.values()))
        tracks, interval_us = index_label_tracks(labels_3d, label_ts)
        assert interval_us is not None
        per_track = {k: np.asarray(sorted(t for t, tr in observations if tr == k)) for k in {str(k) for k in spans}}
        for camera_us in range(50_000, 19_950_000, 33_333):
            selected = select_track_labels(tracks, labels_3d, camera_us, interval_us)
            expected = {k for k, ts in per_track.items() if ts[0] - 50_000 <= camera_us <= ts[-1] + 50_000}
            self.assertEqual(set(selected), expected, camera_us)
            for track_id, box in selected.items():
                self.assertLessEqual(abs(box["ts"] - camera_us), 51_000)

    def test_shared_reference_frames_match_the_nearest_frame(self) -> None:
        # VGGT and V3 exports share one reference time per frame: the selection must equal the
        # nearest frame's boxes, including around a track's missing frame.
        frame_ts = [i * 100_000 for i in range(10)]
        observations = []
        for i, t in enumerate(frame_ts):
            observations.append((t, "steady"))
            if i >= 3:
                observations.append((t, "enters"))
            if i <= 6:
                observations.append((t, "leaves"))
            if i != 5:
                observations.append((t, "holed"))
        observations.append((frame_ts[4], "once"))
        labels_3d, label_ts = _labels(observations)
        tracks, interval_us = index_label_tracks(labels_3d, label_ts)
        assert interval_us is not None
        for camera_us in range(0, frame_ts[-1] + 1, 33_333):
            nearest = int(np.argmin(np.abs(label_ts - camera_us)))
            selected = select_track_labels(tracks, labels_3d, camera_us, interval_us)
            expected = labels_3d[nearest]
            self.assertEqual(set(selected), set(expected), camera_us)
            for track_id, box in selected.items():
                self.assertIs(box, expected[track_id])


class TestCornersInFrontCounts(unittest.TestCase):
    """Tests for the image-bounds-independent in-front occluder cull."""

    @staticmethod
    def _box(center: np.ndarray, half: float = 1.0) -> np.ndarray:
        """Axis-aligned 8-corner box around ``center`` in world coordinates."""
        offsets = np.array(
            [[sx, sy, sz] for sx in (-half, half) for sy in (-half, half) for sz in (-half, half)],
            dtype=np.float64,
        )
        return center + offsets

    def test_empty_input(self) -> None:
        self.assertEqual(_corners_in_front_counts([], np.eye(4)).tolist(), [])

    def test_box_fully_in_front(self) -> None:
        # Identity world->sensor: camera at origin looking down +z; box at z=5.
        counts = _corners_in_front_counts([self._box(np.array([0.0, 0.0, 5.0]))], np.eye(4))
        self.assertEqual(counts.tolist(), [8])

    def test_box_fully_behind_is_culled(self) -> None:
        counts = _corners_in_front_counts([self._box(np.array([0.0, 0.0, -5.0]))], np.eye(4))
        self.assertEqual(counts.tolist(), [0])

    def test_box_enclosing_camera_kept(self) -> None:
        # Camera (origin) inside the box: corners straddle z=0, so some are in front.
        counts = _corners_in_front_counts([self._box(np.array([0.0, 0.0, 0.0]), half=2.0)], np.eye(4))
        self.assertGreaterEqual(int(counts[0]), 1)

    def test_large_near_box_offscreen_but_in_front_kept(self) -> None:
        # Far off to the side (huge x) but in front (z>0): image bounds are ignored,
        # so it must still count as in front and be kept as an occluder.
        counts = _corners_in_front_counts([self._box(np.array([1000.0, 0.0, 3.0]))], np.eye(4))
        self.assertEqual(counts.tolist(), [8])

    def test_rotated_translated_camera(self) -> None:
        # Camera at world (10,0,0) looking toward world -x (forward = [-1,0,0]).
        sensor_to_world = np.array(
            [
                [0.0, 0.0, -1.0, 10.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        t_world_sensor = np.linalg.inv(sensor_to_world)
        in_front = self._box(np.array([5.0, 0.0, 0.0]))  # x < 10 -> in front
        behind = self._box(np.array([15.0, 0.0, 0.0]))  # x > 10 -> behind
        counts = _corners_in_front_counts([in_front, behind], t_world_sensor)
        self.assertEqual(int(counts[0]), 8)
        self.assertEqual(int(counts[1]), 0)


if __name__ == "__main__":
    unittest.main()
