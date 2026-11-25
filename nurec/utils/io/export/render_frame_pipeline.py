# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-repetition pipeline wall timing for one camera frame or LiDAR scan."""

from __future__ import annotations

import time

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import cast


class FramePipelineTiming:
    """Own one repetition's clock and NVTX scopes without synchronizing CUDA.

    Render-call timers complete GPU work inside this interval. Warm-up calls may
    temporarily pause measurement; fixed calibration and final exports stay outside.
    """

    def __init__(
        self, label: str, range_factory: Callable[[str], AbstractContextManager], clock: Callable[[], int] | None
    ) -> None:
        """Bind the sample identity and optional monotonic nanosecond clock."""
        self._label = label
        self._range_factory = range_factory
        # None retains NVTX annotations while disabling every wall-clock read.
        self._clock = clock
        # Excluded warm-up duration in nanoseconds, local to this repetition.
        self._excluded_ns = 0
        # Segment numbering identifies the measured intervals on either side of warm-up.
        self._segment_index = 0
        self._segment_context: AbstractContextManager | None = None
        self._paused = False

    def start(self) -> None:
        """Start the repetition envelope before current-frame request preparation."""
        self._started_ns = self._clock() if self._clock is not None else None
        self._lifecycle = self._range_factory(
            f"nurec.render.frame_pipeline_lifecycle({self._label}, includes_warmup=true)"
        )
        self._lifecycle.__enter__()
        self._start_segment()

    def _start_segment(self) -> None:
        """Open a measured NVTX segment sharing the repetition's identity."""
        index = self._segment_index
        self._segment_index += 1
        self._segment_context = self._range_factory(
            f"nurec.render.frame_pipeline_measured({self._label}, segment={index}, metric=frame_wall_ms, warmup=false)"
        )
        self._segment_context.__enter__()

    def _stop_segment(self) -> None:
        """Close the active NVTX segment without changing renderer or CUDA state."""
        cast(AbstractContextManager, self._segment_context).__exit__(None, None, None)
        self._segment_context = None

    @contextmanager
    def warmup(self) -> Iterator[None]:
        """Exclude warm-up rendering, validation and release without changing their lifetime."""
        if self._paused:
            raise RuntimeError("Frame pipeline warm-up exclusions cannot be nested")
        self._stop_segment()
        self._paused = True
        started = self._clock() if self._clock is not None else None
        try:
            with self._range_factory(f"nurec.render.frame_pipeline_excluded({self._label}, reason=warmup)"):
                yield
        finally:
            if self._clock is not None:
                assert started is not None
                self._excluded_ns += self._clock() - started
            self._paused = False
            self._start_segment()

    def finish(self) -> float | None:
        """Close all ranges and return milliseconds excluding warm-up, or None without reporting."""
        self._stop_segment()
        self._lifecycle.__exit__(None, None, None)
        if self._clock is None:
            return None
        assert self._started_ns is not None
        return (self._clock() - self._started_ns - self._excluded_ns) / 1_000_000


class FramePipelineRecorder:
    """Insert one frame_wall_ms sample per successful measured repetition into the shared collector.

    Each sample identifies one camera frame or LiDAR scan using the same source
    frame and repetition number as the render-call metrics. This recorder owns
    no files or retained timing records.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        range_factory: Callable[[str], AbstractContextManager],
        clock: Callable[[], int] = time.perf_counter_ns,
        collector=None,
        sequence_id: str = "",
        is_lidar: bool = False,
    ) -> None:
        """Bind report collection, sequence/modality identity and an optional nanosecond clock."""
        self._range_factory = range_factory
        self._clock = clock if enabled else None
        self._collector = collector if enabled else None
        self._sequence_id = sequence_id
        # Route samples to per_lidar instead of per_camera when measuring LiDAR scans.
        self._is_lidar = is_lidar

    @contextmanager
    def measure(self, sensor_id: str, frame: int, *, iteration: int) -> Iterator[FramePipelineTiming]:
        """Measure one camera/LiDAR sensor's source frame and positive, one-based repetition.

        Failed repetitions unwind annotations and propagate without inserting a sample.
        Successful samples are inserted after measurement has stopped.
        """
        if iteration < 1:
            raise ValueError("Frame pipeline iteration must be positive")
        sensor_kind = "lidar" if self._is_lidar else "camera"
        label = f"{sensor_kind}={sensor_id}, frame={frame}, iteration={iteration}"
        timing = FramePipelineTiming(label, self._range_factory, self._clock)
        complete = False
        timing.start()
        try:
            yield timing
            complete = True
        finally:
            milliseconds = timing.finish()
            if complete and milliseconds is not None and self._collector is not None:
                # Stop timing before inserting this metric so collection cannot charge itself.
                self._collector.record_frame(
                    sequence_id=self._sequence_id,
                    sensor_id=sensor_id,
                    frame=frame,
                    iteration=iteration,
                    is_lidar=self._is_lidar,
                    frame_wall_ms=milliseconds,
                )
