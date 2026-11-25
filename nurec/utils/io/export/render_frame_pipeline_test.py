# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic wall-clock tests requiring neither CUDA nor a rendering dependency."""

import unittest

from contextlib import contextmanager
from unittest.mock import Mock

from nurec.utils.io.export.render_frame_pipeline import FramePipelineRecorder


class FakeClock:
    """Advance only for simulated application work; its integer unit is one nanosecond."""

    def __init__(self):
        """Initialize deterministic time and the observed NVTX stack."""
        self.now = 0
        self.stack = []
        self.events = []

    def __call__(self):
        """Return monotonic nanoseconds without advancing time during instrumentation."""
        return self.now

    def advance(self, milliseconds):
        """Account for a simulated operation in milliseconds."""
        self.now += round(milliseconds * 1_000_000)

    @contextmanager
    def range(self, name):
        """Verify NVTX contexts close in stack order even if measured work raises."""
        self.events.append(("enter", name, self.now))
        self.stack.append(name)
        try:
            yield
        finally:
            assert self.stack.pop() == name
            self.events.append(("exit", name, self.now))


class FramePipelineTest(unittest.TestCase):
    def setUp(self):
        """Bind a deterministic clock and collector for independent repetition samples."""
        self.clock = FakeClock()
        self.collector = Mock()
        self.recorder = FramePipelineRecorder(
            enabled=True,
            range_factory=self.clock.range,
            clock=self.clock,
            collector=self.collector,
            sequence_id="sequence",
        )

    def test_warmup_exclusion_keeps_preparation_and_consumption(self):
        """Exclude warm-up while retaining the measured work on both sides of it."""
        with self.recorder.measure("left", 7, iteration=1) as timing:
            self.clock.advance(2)
            with timing.warmup():
                self.clock.advance(100)
            self.clock.advance(11)
        self.collector.record_frame.assert_called_once_with(
            sequence_id="sequence", sensor_id="left", frame=7, iteration=1, is_lidar=False, frame_wall_ms=13
        )
        self.assertEqual(self.clock.stack, [])

    def test_repetitions_keep_independent_values_and_iteration_numbers(self):
        """Preserve per-repetition variation without summing or averaging the frame's repetitions."""
        for iteration, cost in enumerate((13, 29, 7), 1):
            with self.recorder.measure("left", 7, iteration=iteration):
                self.clock.advance(cost)
        samples = [call.kwargs for call in self.collector.record_frame.call_args_list]
        self.assertEqual([row["frame_wall_ms"] for row in samples], [13, 29, 7])
        self.assertEqual([row["iteration"] for row in samples], [1, 2, 3])
        self.assertEqual([row["frame"] for row in samples], [7, 7, 7])

    def test_lidar_sample_keeps_modality_and_repetition_identity(self):
        """Route LiDAR samples to the same frame and iteration as their render-call metrics."""
        recorder = FramePipelineRecorder(
            enabled=True,
            range_factory=self.clock.range,
            clock=self.clock,
            collector=self.collector,
            sequence_id="sequence",
            is_lidar=True,
        )
        with recorder.measure("lidar_front", 9, iteration=2):
            self.clock.advance(6)
        self.collector.record_frame.assert_called_once_with(
            sequence_id="sequence", sensor_id="lidar_front", frame=9, iteration=2, is_lidar=True, frame_wall_ms=6
        )

    def test_failed_repetition_is_not_recorded(self):
        """Preserve earlier successful samples while unwinding a later failed repetition."""
        with self.recorder.measure("left", 9, iteration=1):
            self.clock.advance(5)
        with self.assertRaisesRegex(RuntimeError, "render failed"):
            with self.recorder.measure("left", 9, iteration=2):
                self.clock.advance(50)
                raise RuntimeError("render failed")
        self.assertEqual(self.collector.record_frame.call_count, 1)
        self.assertEqual(self.clock.stack, [])

    def test_warmup_failure_unwinds_ranges_without_a_sample(self):
        """Propagate warm-up failures and leave the annotation stack balanced."""
        with self.assertRaisesRegex(RuntimeError, "warmup failed"):
            with self.recorder.measure("left", 0, iteration=1) as timing:
                with timing.warmup():
                    self.clock.advance(20)
                    raise RuntimeError("warmup failed")
        self.collector.record_frame.assert_not_called()
        self.assertEqual(self.clock.stack, [])

    def test_disabled_reporting_never_reads_clock(self):
        """Keep repetition and warm-up annotations without sampling or reading the clock."""

        def forbidden():
            """Reject any clock access while reporting is disabled."""
            raise AssertionError("No wall-clock reads without reporting")

        recorder = FramePipelineRecorder(
            enabled=False,
            range_factory=self.clock.range,
            clock=forbidden,
            collector=self.collector,
        )
        for iteration in (1, 2):
            with recorder.measure("left", 0, iteration=iteration) as timing:
                with timing.warmup():
                    self.clock.advance(50)
                self.clock.advance(7)
        self.collector.record_frame.assert_not_called()
        self.assertEqual(self.clock.stack, [])
        self.assertTrue(any("iteration=2" in event[1] for event in self.clock.events))
        self.assertTrue(any("reason=warmup" in event[1] for event in self.clock.events))

    def test_collector_insertion_is_outside_each_sample(self):
        """Exclude this metric's insertion overhead and avoid adding custom report metadata."""

        def record(**sample):
            """Charge the insertion after measurement has finished."""
            self.clock.advance(200)

        self.collector.record_frame.side_effect = record
        for iteration, cost in enumerate((23, 47), 1):
            with self.recorder.measure("left", 0, iteration=iteration):
                self.clock.advance(cost)
        samples = [call.kwargs for call in self.collector.record_frame.call_args_list]
        self.assertEqual([row["frame_wall_ms"] for row in samples], [23, 47])
        self.collector.set_metric_metadata.assert_not_called()

    def test_invalid_iterations_and_nested_exclusion_are_rejected(self):
        """Reject nonpositive repetition numbers and overlapping warm-up pauses."""
        for iteration in (0, -1):
            with self.assertRaisesRegex(ValueError, "iteration must be positive"):
                with self.recorder.measure("left", 0, iteration=iteration):
                    pass
        with self.recorder.measure("left", 0, iteration=1) as timing:
            with timing.warmup():
                with self.assertRaises(RuntimeError):
                    with timing.warmup():
                        pass
        self.assertEqual(self.clock.stack, [])


if __name__ == "__main__":
    unittest.main()
