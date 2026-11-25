# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from nurec.lidar.firing_patterns.recovery import choose_frame, recover_firing_pattern


class _FakeSensor:
    def __init__(self, counts: list[int], non_returning: list[int], spans: list[int] | None = None):
        self.frames_count = len(counts)
        starts = np.arange(self.frames_count, dtype=np.int64) * 1_000
        spans = spans or [700] * self.frames_count
        self.frames_timestamps_us = np.stack([starts, starts + spans], axis=1)
        self._counts = counts
        self._non_returning = non_returning

    def get_frame_ray_bundle_timestamp_us(self, frame_index: int):
        start = self.frames_timestamps_us[frame_index, 0]
        span = self.frames_timestamps_us[frame_index, 1] - start
        return start + np.linspace(0, span, self._counts[frame_index]).astype(np.int64)

    def get_frame_ray_bundle_return_distance_m(self, *, frame_index: int, return_index: int):
        del return_index
        result = np.ones(self._counts[frame_index], dtype=np.float32)
        result[: self._non_returning[frame_index]] = 0
        return result

    def get_frame_ray_bundle_direction(self, frame_index: int):
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (self._counts[frame_index], 1))


class TestChooseFrame:
    def test_skips_densest_ineligible_frame(self):
        sensor = _FakeSensor([10, 8], [0, 1])

        assert choose_frame(sensor) == 1

    def test_equal_counts_prefer_earliest_eligible_frame(self):
        sensor = _FakeSensor([8, 8], [1, 1])

        assert choose_frame(sensor) == 0

    def test_reports_when_all_full_sweeps_are_ineligible(self):
        sensor = _FakeSensor([10, 8], [0, 0])

        with pytest.raises(ValueError, match="dense rays including non-returning"):
            choose_frame(sensor)

    def test_reports_when_there_is_no_full_sweep(self):
        sensor = _FakeSensor([8] * 5, [1] * 5, spans=[100, 100, 700, 700, 700])

        with pytest.raises(ValueError, match="no full sweep"):
            choose_frame(sensor, scan_frames=2)

    def test_scan_limit_excludes_later_eligible_frame(self):
        sensor = _FakeSensor([10, 8], [0, 1])

        with pytest.raises(ValueError, match="first 1 frames"):
            choose_frame(sensor, scan_frames=1)

    def test_explicit_ineligible_frame_is_still_rejected(self):
        sensor = _FakeSensor([10, 8], [0, 1])

        with pytest.raises(ValueError, match="dense rays including non-returning"):
            recover_firing_pattern(sensor, frame=0)


@pytest.mark.parametrize("scan_frames", [2, 3])
def test_truncated_majority_does_not_lower_full_sweep_threshold(scan_frames):
    sensor = _FakeSensor([20, 20, 8], [1, 1, 1], spans=[100, 100, 700])

    if scan_frames == 2:
        with pytest.raises(ValueError, match="no full sweep"):
            choose_frame(sensor, scan_frames=scan_frames)
    else:
        assert choose_frame(sensor, scan_frames=scan_frames) == 2


def test_longest_ray_span_precedes_ray_count_within_complete_candidates():
    sensor = _FakeSensor([20, 8], [1, 1], spans=[650, 700])

    assert choose_frame(sensor) == 1


def test_equal_ray_spans_prefer_more_slots_then_earliest_frame():
    sensor = _FakeSensor([8, 20, 20], [1, 1, 1])

    assert choose_frame(sensor) == 1


def test_explicit_truncated_frame_warns_even_when_truncated_frames_dominate(caplog):
    sensor = _FakeSensor([20, 20, 8], [1, 1, 1], spans=[100, 100, 700])

    recover_firing_pattern(sensor, frame=0)

    assert "truncated sweep" in caplog.text
