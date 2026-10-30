# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ActorTracks and ActorsSnapshot (nurec/render/actors.py)."""

import unittest

from unittest.mock import MagicMock

import numpy as np
import torch

from nurec.render.actors import ActorsSnapshot, ActorTracks, clip_window_to_cuboid_track


def _make_cuboid_track(track_id: str, t_min: int, t_max: int, dims=(2.0, 2.0, 2.0)):
    """Return a mock CuboidTracks with a single ID and a two-keyframe timestamp span."""
    track = MagicMock()
    track.tracks_id = [track_id]
    track.tracks_timestamps_us = torch.tensor([t_min, t_max], dtype=torch.int64)
    track.cuboids_dims = torch.tensor([list(dims)], dtype=torch.float32)
    return track


def _make_interpolator():
    """Return a mock PoseInterpolator whose interpolation always returns two identity SE3 poses."""
    interp = MagicMock()
    identity_se3 = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)])  # (2, 4, 4)
    interp.interpolate_to_timestamps.return_value = identity_se3
    return interp


def _actor_tracks(*specs):
    """Build an ActorTracks from (track_id, t_min, t_max) specs."""
    cuboid_tracks = [_make_cuboid_track(tid, t0, t1) for tid, t0, t1 in specs]
    interpolators = [_make_interpolator() for _ in specs]
    return ActorTracks(_cuboid_tracks_list=cuboid_tracks, _interpolator_list=interpolators)


def _make_cuboid_track_empty():
    track = MagicMock()
    track.tracks_timestamps_us = torch.empty(0, dtype=torch.int64)
    return track


class TestClipWindowToCuboidTrack(unittest.TestCase):
    def test_empty_track_returns_none(self):
        self.assertIsNone(clip_window_to_cuboid_track(_make_cuboid_track_empty(), 0, 100))

    def test_no_overlap_frame_before_track(self):
        self.assertIsNone(clip_window_to_cuboid_track(_make_cuboid_track("a", 200, 400), 0, 100))

    def test_no_overlap_frame_after_track(self):
        self.assertIsNone(clip_window_to_cuboid_track(_make_cuboid_track("a", 0, 100), 200, 300))

    def test_full_containment_frame_inside_track(self):
        self.assertEqual(clip_window_to_cuboid_track(_make_cuboid_track("a", 0, 1000), 100, 200), (100, 200))

    def test_partial_overlap_start(self):
        self.assertEqual(clip_window_to_cuboid_track(_make_cuboid_track("a", 100, 500), 0, 200), (100, 200))

    def test_partial_overlap_end(self):
        self.assertEqual(clip_window_to_cuboid_track(_make_cuboid_track("a", 0, 300), 200, 500), (200, 300))

    def test_frame_wider_than_track(self):
        self.assertEqual(clip_window_to_cuboid_track(_make_cuboid_track("a", 100, 200), 0, 500), (100, 200))

    def test_boundary_touch_at_track_end(self):
        self.assertEqual(clip_window_to_cuboid_track(_make_cuboid_track("a", 0, 100), 100, 200), (100, 100))

    def test_boundary_touch_at_track_start(self):
        self.assertIsNone(clip_window_to_cuboid_track(_make_cuboid_track("a", 100, 200), 0, 100))

    def test_point_query_at_track_start_included(self):
        # Zero-length window (instant shutter) exactly at track_min must include the track.
        self.assertEqual(clip_window_to_cuboid_track(_make_cuboid_track("a", 100, 600), 100, 100), (100, 100))

    def test_point_query_inside_track_included(self):
        self.assertEqual(clip_window_to_cuboid_track(_make_cuboid_track("a", 0, 1000), 500, 500), (500, 500))

    def test_point_query_at_track_end_included(self):
        self.assertEqual(clip_window_to_cuboid_track(_make_cuboid_track("a", 0, 100), 100, 100), (100, 100))

    def test_point_query_before_track_excluded(self):
        self.assertIsNone(clip_window_to_cuboid_track(_make_cuboid_track("a", 100, 200), 99, 99))

    def test_point_query_after_track_excluded(self):
        self.assertIsNone(clip_window_to_cuboid_track(_make_cuboid_track("a", 0, 100), 101, 101))


class TestGetSnapshotAtFrame(unittest.TestCase):
    """Tests for ActorTracks.get_snapshot_at_frame — especially the overlap-based inclusion logic."""

    def test_no_tracks_returns_empty(self):
        snap = ActorTracks().get_snapshot_at_frame(0, 100)
        self.assertEqual(snap.num_actors(), 0)

    def test_frame_entirely_before_track_returns_empty(self):
        snap = _actor_tracks(("a", 200, 400)).get_snapshot_at_frame(0, 100)
        self.assertEqual(snap.num_actors(), 0)

    def test_frame_entirely_after_track_returns_empty(self):
        snap = _actor_tracks(("a", 0, 100)).get_snapshot_at_frame(200, 300)
        self.assertEqual(snap.num_actors(), 0)

    def test_frame_inside_track_span_returns_actor(self):
        snap = _actor_tracks(("a", 0, 1000)).get_snapshot_at_frame(100, 200)
        self.assertEqual(snap.num_actors(), 1)
        self.assertEqual(snap.actor_ids[0], "a")

    def test_frame_overlaps_track_start_clamped(self):
        """Frame starts before track_min; interpolation receives [track_min, frame_end]."""
        tracks = _actor_tracks(("a", 100, 500))
        snap = tracks.get_snapshot_at_frame(0, 200)
        self.assertEqual(snap.num_actors(), 1)
        tracks._interpolator_list[0].interpolate_to_timestamps.assert_called_once_with([100, 200])

    def test_frame_overlaps_track_end_clamped(self):
        """Frame ends after track_max; interpolation receives [frame_start, track_max]."""
        tracks = _actor_tracks(("a", 0, 300))
        snap = tracks.get_snapshot_at_frame(200, 500)
        self.assertEqual(snap.num_actors(), 1)
        tracks._interpolator_list[0].interpolate_to_timestamps.assert_called_once_with([200, 300])

    def test_frame_fully_contains_track_clamped_to_track_span(self):
        """Frame window wider than track; interpolation receives [track_min, track_max]."""
        tracks = _actor_tracks(("a", 100, 200))
        snap = tracks.get_snapshot_at_frame(0, 500)
        self.assertEqual(snap.num_actors(), 1)
        tracks._interpolator_list[0].interpolate_to_timestamps.assert_called_once_with([100, 200])

    def test_frame_touches_track_end_exactly(self):
        """Frame starts exactly at track_max — boundary overlap, actor included."""
        snap = _actor_tracks(("a", 0, 100)).get_snapshot_at_frame(100, 200)
        self.assertEqual(snap.num_actors(), 1)

    def test_frame_touches_track_start_exactly(self):
        """Frame ends exactly at track_min — request window is half-open, so no overlap."""
        snap = _actor_tracks(("a", 100, 200)).get_snapshot_at_frame(0, 100)
        self.assertEqual(snap.num_actors(), 0)

    def test_boundary_touch_clamps_both_timestamps_equal(self):
        """When frame starts exactly at track_max, both clamped timestamps equal track_max."""
        tracks = _actor_tracks(("a", 0, 100))
        snap = tracks.get_snapshot_at_frame(100, 200)
        self.assertEqual(snap.num_actors(), 1)
        tracks._interpolator_list[0].interpolate_to_timestamps.assert_called_once_with([100, 100])

    def test_point_query_at_track_start_returns_actor(self):
        """Zero-length window (instant shutter) with track_min == frame_start == frame_end must
        include the actor, interpolated at the single instant."""
        tracks = _actor_tracks(("a", 100, 600))
        snap = tracks.get_snapshot_at_frame(100, 100)
        self.assertEqual(snap.num_actors(), 1)
        self.assertEqual(snap.actor_ids[0], "a")
        tracks._interpolator_list[0].interpolate_to_timestamps.assert_called_once_with([100, 100])

    def test_multiple_tracks_partial_overlap(self):
        """One track overlaps the frame; the other does not."""
        snap = _actor_tracks(("in", 0, 1000), ("out", 2000, 3000)).get_snapshot_at_frame(100, 500)
        self.assertEqual(snap.num_actors(), 1)
        self.assertEqual(snap.actor_ids[0], "in")

    def test_snapshot_cuboid_dims_populated(self):
        """Returned snapshot must carry cuboid_dims for bbox overlay use."""
        snap = _actor_tracks(("a", 0, 1000)).get_snapshot_at_frame(100, 200)
        self.assertIsNotNone(snap.cuboid_dims)
        self.assertEqual(tuple(snap.cuboid_dims.shape), (1, 3))

    def test_snapshot_actor_poses_shape(self):
        snap = _actor_tracks(("a", 0, 1000)).get_snapshot_at_frame(100, 200)
        self.assertEqual(tuple(snap.actor_poses.shape), (1, 2, 7))

    def test_invalid_timestamps_raises(self):
        with self.assertRaises(ValueError):
            _actor_tracks(("a", 0, 1000)).get_snapshot_at_frame(500, 100)  # start > end


class TestActorsSnapshot(unittest.TestCase):
    """Tests for ActorsSnapshot invariant checks."""

    def test_empty_snapshot(self):
        snap = ActorsSnapshot.empty()
        self.assertEqual(snap.num_actors(), 0)
        self.assertIsNone(snap.cuboid_dims)

    def test_cuboid_dims_shape_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            ActorsSnapshot(
                actor_ids=["a", "b"],
                actor_poses=torch.zeros((2, 2, 7)),
                cuboid_dims=torch.zeros((3, 3)),  # wrong: 3 rows for 2 actors
            )

    def test_empty_actor_ids_wrong_pose_shape_raises(self):
        """actor_ids=[] with wrong actor_poses shape must raise even though there are no actors."""
        with self.assertRaises(AssertionError):
            ActorsSnapshot(
                actor_ids=[],
                actor_poses=torch.zeros((0, 3, 7)),  # wrong: middle dim is 3 not 2
                cuboid_dims=None,
            )

    def test_cuboid_dims_none_allowed(self):
        snap = ActorsSnapshot(
            actor_ids=["a"],
            actor_poses=torch.zeros((1, 2, 7)),
            cuboid_dims=None,
        )
        self.assertIsNone(snap.cuboid_dims)


if __name__ == "__main__":
    unittest.main()
