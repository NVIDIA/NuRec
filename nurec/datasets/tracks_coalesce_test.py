# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import pytest
import torch

from nurec.datasets.tracks import CuboidTracks, TrackFlags


class TestCoalesceShortTimestampGaps:
    @staticmethod
    def _make_tracks(timestamps_by_track: list[list[int]]) -> CuboidTracks:
        tracks_poses = []
        for track_idx, timestamps_us in enumerate(timestamps_by_track):
            poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], len(timestamps_us), axis=0)
            poses[:, 0, 3] = np.arange(len(timestamps_us), dtype=np.float32)
            poses[:, 1, 3] = float(track_idx)
            tracks_poses.append(poses)

        return CuboidTracks.Factory.from_numpy(
            tracks_id=[f"track_{idx}" for idx in range(len(timestamps_by_track))],
            tracks_poses=tracks_poses,
            tracks_timestamps_us=[np.asarray(timestamps_us, dtype=np.int64) for timestamps_us in timestamps_by_track],
            tracks_label_class=[f"class_{idx}" for idx in range(len(timestamps_by_track))],
            tracks_flags=[
                TrackFlags.DYNAMIC if idx == 0 else TrackFlags.NONE for idx in range(len(timestamps_by_track))
            ],
            cuboids_dims=[np.full(3, idx + 1, dtype=np.float32) for idx in range(len(timestamps_by_track))],
            device=torch.device("cpu"),
        )

    @pytest.mark.parametrize("min_gap_us", [0, -1])
    def test_rejects_nonpositive_minimum_gap(self, min_gap_us: int) -> None:
        tracks = CuboidTracks.Factory.empty(device=torch.device("cpu"))

        with pytest.raises(ValueError, match="min_gap_us must be positive"):
            CuboidTracks.Ops.coalesce_short_timestamp_gaps(tracks, min_gap_us)

    def test_empty_tracks_return_an_independent_empty_container(self) -> None:
        tracks = CuboidTracks.Factory.empty(device=torch.device("cpu"))

        result = CuboidTracks.Ops.coalesce_short_timestamp_gaps(tracks, min_gap_us=10)

        assert result is not tracks
        assert result.n_tracks == 0
        assert result.max_track_n_poses == 0
        torch.testing.assert_close(result.tracks_packinfo, tracks.tracks_packinfo)
        torch.testing.assert_close(result.tracks_poses.data, tracks.tracks_poses.data)
        torch.testing.assert_close(result.tracks_timestamps_us, tracks.tracks_timestamps_us)

    @pytest.mark.parametrize("timestamps_us", [[0, 0], [10, 9]])
    def test_rejects_nonincreasing_timestamps(self, timestamps_us: list[int]) -> None:
        tracks = self._make_tracks([timestamps_us])

        with pytest.raises(ValueError, match="track timestamps must be strictly increasing"):
            CuboidTracks.Ops.coalesce_short_timestamp_gaps(tracks, min_gap_us=10)

    def test_preserves_tracks_when_all_gaps_meet_the_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        tracks = self._make_tracks([[0, 10, 20], [100, 120]])

        with caplog.at_level(logging.INFO, logger="nurec.datasets.tracks"):
            result = CuboidTracks.Ops.coalesce_short_timestamp_gaps(tracks, min_gap_us=10)

        assert "No cuboid-track timestamp gaps" in caplog.text
        assert result is not tracks
        assert result.tracks_id == tracks.tracks_id
        assert result.tracks_label_class == tracks.tracks_label_class
        assert result.max_track_n_poses == tracks.max_track_n_poses
        torch.testing.assert_close(result.tracks_packinfo, tracks.tracks_packinfo)
        torch.testing.assert_close(result.tracks_poses.data, tracks.tracks_poses.data)
        torch.testing.assert_close(result.tracks_timestamps_us, tracks.tracks_timestamps_us)
        torch.testing.assert_close(result.tracks_flags, tracks.tracks_flags)
        torch.testing.assert_close(result.cuboids_dims, tracks.cuboids_dims)
        torch.testing.assert_close(
            result.max_positive_timestamp_gap_us,
            torch.tensor([10, 20], dtype=torch.int64),
        )

    def test_selects_endpoint_and_midpoint_representatives_per_track(self, caplog: pytest.LogCaptureFixture) -> None:
        tracks = self._make_tracks(
            [
                [0, 5, 9, 30, 34, 39, 60, 65, 69],
                [100, 110, 120],
            ]
        )

        with caplog.at_level(logging.INFO, logger="nurec.datasets.tracks"):
            result = CuboidTracks.Ops.coalesce_short_timestamp_gaps(tracks, min_gap_us=10)

        assert "Coalesced short cuboid-track timestamp gaps" in caplog.text
        # The first, interior, and final short-gap groups retain indices 0, 4, and 8.
        # Gaps exactly equal to the threshold in the second track are not coalesced.
        expected_source_indices = torch.tensor([0, 4, 8, 9, 10, 11])
        torch.testing.assert_close(result.tracks_poses.data, tracks.tracks_poses.data[expected_source_indices])
        torch.testing.assert_close(
            result.tracks_timestamps_us,
            torch.tensor([0, 34, 69, 100, 110, 120], dtype=torch.int64),
        )
        torch.testing.assert_close(
            result.tracks_packinfo,
            torch.tensor([[0, 3], [3, 3]], dtype=torch.int32),
        )
        assert result.max_track_n_poses == 3
        assert result.tracks_id == tracks.tracks_id
        assert result.tracks_label_class == tracks.tracks_label_class
        torch.testing.assert_close(result.tracks_flags, tracks.tracks_flags)
        torch.testing.assert_close(result.cuboids_dims, tracks.cuboids_dims)
        torch.testing.assert_close(
            result.max_positive_timestamp_gap_us,
            torch.tensor([35, 10], dtype=torch.int64),
        )

        # The operation is not in-place.
        assert tracks.tracks_poses.shape[0] == 12
        torch.testing.assert_close(
            tracks.tracks_timestamps_us,
            torch.tensor([0, 5, 9, 30, 34, 39, 60, 65, 69, 100, 110, 120], dtype=torch.int64),
        )

    def test_keeps_first_and_last_pose_when_a_track_would_collapse(self, caplog: pytest.LogCaptureFixture) -> None:
        tracks = self._make_tracks([[0, 3, 7, 9]])

        with caplog.at_level(logging.WARNING, logger="nurec.datasets.tracks"):
            result = CuboidTracks.Ops.coalesce_short_timestamp_gaps(tracks, min_gap_us=10)

        assert "would collapse below two poses" in caplog.text
        assert "remain after fusion" in caplog.text
        torch.testing.assert_close(result.tracks_poses.data, tracks.tracks_poses.data[[0, 3]])
        torch.testing.assert_close(result.tracks_timestamps_us, torch.tensor([0, 9], dtype=torch.int64))
        torch.testing.assert_close(result.tracks_packinfo, torch.tensor([[0, 2]], dtype=torch.int32))
        torch.testing.assert_close(result.max_positive_timestamp_gap_us, torch.tensor([9], dtype=torch.int64))
        assert result.max_track_n_poses == 2
