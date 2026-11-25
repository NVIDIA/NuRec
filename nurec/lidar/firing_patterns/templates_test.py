# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import numpy as np
import pytest

from nurec.lidar.firing_patterns.schema import LidarFiringPatternTemplatesData
from nurec.lidar.firing_patterns.templates import LidarFiringPatternTemplate, LidarFiringPatternTemplateSet


def _template(
    n: int = 8,
    *,
    span_us: int = 70_000,
    azimuth_offset: float = 0.0,
) -> LidarFiringPatternTemplate:
    azimuths = np.linspace(-0.5, 0.5, n) + azimuth_offset
    directions = np.stack([np.cos(azimuths), np.sin(azimuths), np.zeros(n)], axis=-1).astype(np.float32)
    return LidarFiringPatternTemplate(
        directions_sensor=directions,
        timestamp_offsets_us=np.linspace(0, span_us, n).astype(np.int64),
        packet_ray_offsets=np.array([0, n // 2, n], dtype=np.int64),
    )


def _template_data(template_count: int = 4) -> LidarFiringPatternTemplatesData:
    templates = [_template(n=8, azimuth_offset=phase * 0.001) for phase in range(template_count)]
    return LidarFiringPatternTemplatesData(
        directions_sensor=np.concatenate([template.directions_sensor for template in templates]),
        timestamp_offsets_us=np.concatenate([template.timestamp_offsets_us for template in templates]),
        template_ray_offsets=np.arange(0, (template_count + 1) * 8, 8, dtype=np.int64),
        packet_ray_offsets=np.arange(0, (template_count * 2 + 1) * 4, 4, dtype=np.int64),
        template_packet_offsets=np.arange(0, (template_count + 1) * 2, 2, dtype=np.int64),
        reference_frame_start_us=1_000_000,
        frame_period_us=100_000,
        frame_start_tolerance_us=200,
    )


class TestValidation:
    def test_rejects_mismatched_timestamp_count(self):
        with pytest.raises(ValueError, match="matching directions_sensor"):
            LidarFiringPatternTemplate(
                directions_sensor=np.zeros((4, 3), dtype=np.float32),
                timestamp_offsets_us=np.zeros(3, dtype=np.int64),
                packet_ray_offsets=np.array([0, 4], dtype=np.int64),
            )

    def test_rejects_packet_offsets_that_do_not_cover_the_rays(self):
        with pytest.raises(ValueError, match="cover every template ray"):
            LidarFiringPatternTemplate(
                directions_sensor=np.ones((2, 3), dtype=np.float32) / np.sqrt(3.0),
                timestamp_offsets_us=np.array([0, 1], dtype=np.int64),
                packet_ray_offsets=np.array([0, 1], dtype=np.int64),
            )

    def test_rejects_firing_span_past_frame_period_tolerance(self):
        template = _template(n=4, span_us=150_000)

        with pytest.raises(ValueError, match="cannot outlast"):
            LidarFiringPatternTemplateSet(
                templates=(template,),
                reference_frame_start_us=0,
                frame_period_us=100_000,
                frame_start_tolerance_us=100,
            )

    def test_packet_order_does_not_require_monotonic_timestamps(self):
        template = LidarFiringPatternTemplate(
            directions_sensor=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            timestamp_offsets_us=np.array([100, 0], dtype=np.int64),
            packet_ray_offsets=np.array([0, 1, 2], dtype=np.int64),
        )

        np.testing.assert_array_equal(template.ray_timestamps_us(1_000), [1_100, 1_000])

    def test_packet_layout_arrays_are_jointly_optional(self):
        with pytest.raises(ValueError, match="must either both be present or both absent"):
            dataclasses.replace(_template_data(), template_packet_offsets=None)

    def test_variable_ray_counts_round_trip_to_runtime(self):
        data = dataclasses.replace(
            _template_data(template_count=2),
            template_ray_offsets=np.array([0, 7, 16], dtype=np.int64),
            packet_ray_offsets=None,
            template_packet_offsets=None,
        )
        loaded = LidarFiringPatternTemplateSet.from_data(data)
        assert [t.n_rays for t in loaded.templates] == [7, 9]
        assert loaded.select(loaded.reference_frame_start_us).n_rays == 7
        assert loaded.select(loaded.reference_frame_start_us + loaded.frame_period_us).n_rays == 9

    def test_export_rejects_timing_that_runtime_would_reject(self):
        with pytest.raises(ValueError, match="outlast"):
            dataclasses.replace(_template_data(), frame_period_us=100, frame_start_tolerance_us=0)

    def test_accepts_frame_window_between_firing_span_and_period(self) -> None:
        template = _template(span_us=60_000)

        template.check_window(requested_us=60_000, frame_period_us=100_000)
        template.check_window(requested_us=69_659, frame_period_us=100_000)
        template.check_window(requested_us=105_000, frame_period_us=100_000)

    @pytest.mark.parametrize("requested_us", [-1, 0, 57_000, 59_999, 105_001, 110_000])
    def test_rejects_frame_window_outside_firing_span_and_period(self, requested_us: int) -> None:
        template = _template(span_us=60_000)

        with pytest.raises(ValueError, match=r"request duration.*shorter of firing span and frame period"):
            template.check_window(requested_us=requested_us, frame_period_us=100_000)

    def test_accepts_nominal_period_when_firing_span_slightly_overruns_period(self) -> None:
        template = _template(span_us=100_011)

        template.check_window(requested_us=99_988, frame_period_us=99_988)

    def test_zero_span_pattern_accepts_zero_duration(self) -> None:
        _template(span_us=0).check_window(requested_us=0, frame_period_us=100_000)


class TestTemplateSet:
    def test_selects_phase_from_timestamp_before_and_after_the_recording(self):
        template_set = LidarFiringPatternTemplateSet.from_data(_template_data())
        start = template_set.reference_frame_start_us
        period = template_set.frame_period_us

        assert template_set.select_index(start) == 0
        assert template_set.select_index(start + period) == 1
        assert template_set.select_index(start + 6 * period) == 2
        assert template_set.select_index(start - period) == 3

    def test_selects_nearest_phase_without_requiring_the_recording_timestamp_grid(self):
        template_set = LidarFiringPatternTemplateSet.from_data(_template_data())
        start = template_set.reference_frame_start_us

        assert template_set.select_index(start + 100_000 + 200) == 1
        assert template_set.select_index(start + 100_000 + 201) == 1

    def test_pattern_id_is_stable_and_covers_packet_layout(self):
        template = _template()
        same = dataclasses.replace(template)
        packetless = dataclasses.replace(template, packet_ray_offsets=None)

        assert template.pattern_id(100_000) == same.pattern_id(100_000)
        assert template.pattern_id(100_000).startswith("sha256:")
        assert template.pattern_id(100_000) != packetless.pattern_id(100_000)

    def test_pattern_ids_distinguish_frame_periods_and_lookup_rejects_other_period(self):
        template_set = LidarFiringPatternTemplateSet.from_data(_template_data())
        slower_set = dataclasses.replace(template_set, frame_period_us=200_000)
        template = template_set.templates[0]
        pattern_id = template.pattern_id(template_set.frame_period_us)
        slower_id = template.pattern_id(slower_set.frame_period_us)

        assert pattern_id != slower_id
        assert template_set.get_by_id(pattern_id) is template
        assert slower_set.get_by_id(slower_id) is template
        with pytest.raises(KeyError, match="Unknown firing pattern id"):
            template_set.get_by_id(slower_id)
        with pytest.raises(KeyError, match="Unknown firing pattern id"):
            slower_set.get_by_id(pattern_id)

    def test_explicit_id_lookup_selects_the_requested_template(self):
        template_set = LidarFiringPatternTemplateSet.from_data(_template_data())
        requested = template_set.templates[2]

        assert template_set.get_by_id(requested.pattern_id(template_set.frame_period_us)) is requested
        with pytest.raises(KeyError, match="Unknown firing pattern id"):
            template_set.get_by_id("sha256:unknown")


class TestGrid:
    """The grid is derived from the firing times, so a pattern never stores a stale copy."""

    @staticmethod
    def _column_wise(n_columns: int, n_rows: int, step_us: int = 49) -> LidarFiringPatternTemplate:
        """A unit that fires a whole column at once, one column every ``step_us``."""
        n = n_columns * n_rows
        return LidarFiringPatternTemplate(
            directions_sensor=np.tile(np.array([1.0, 0.0, 0.0], np.float32), (n, 1)),
            timestamp_offsets_us=np.repeat(np.arange(n_columns, dtype=np.int64) * step_us, n_rows),
        )

    def test_reads_the_grid_off_the_firing_times(self):
        pattern = self._column_wise(1214, 360)

        assert pattern.grid == (1214, 360)
        # The addressing the proto documents, for a ray in column 7, line 42.
        n_rows = pattern.grid[1]
        assert divmod(7 * n_rows + 42, n_rows) == (7, 42)

    def test_a_pattern_that_fires_at_one_instant_has_no_grid(self):
        flat = LidarFiringPatternTemplate(
            directions_sensor=np.tile(np.array([1.0, 0.0, 0.0], np.float32), (24, 1)),
            timestamp_offsets_us=np.zeros(24, dtype=np.int64),
        )
        assert flat.grid is None

    def test_a_pattern_whose_order_was_destroyed_has_no_grid(self):
        # What a clustering merge produces: the right times in cluster order rather than sweep
        # order. It must not be mistaken for a grid.
        offsets = np.repeat(np.arange(8, dtype=np.int64) * 49, 6)
        shuffled = LidarFiringPatternTemplate(
            directions_sensor=np.tile(np.array([1.0, 0.0, 0.0], np.float32), (48, 1)),
            timestamp_offsets_us=np.random.default_rng(0).permutation(offsets),
        )
        assert shuffled.grid is None
