# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-event timing of a rendering dependency chain."""

from __future__ import annotations

import logging
import math

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch


log = logging.getLogger(__name__)


class CudaEventTimer:
    """Time GPU work and explicitly join any additional producer streams.

    The RGB path submits its rendering operations to the current CUDA stream.
    Pass additional producer streams to the constructor through ``streams``;
    no later registration API is available. Each producer waits on the start
    event and is joined before completion.
    Native forward paths must join their internal streams back to the current
    stream before returning. Record completion after those dependencies, then
    wait on that event. CUDA events include any CPU submission gaps.
    """

    def __init__(self, label: str = "nurec.render.forward", streams: Sequence[torch.cuda.Stream] = ()) -> None:
        self.milliseconds: list[float] = []
        self.valid: list[bool] = []
        self._label = label
        self._streams = tuple(streams)
        # Reuse our own timing pair and join events for each device/stream set.
        self._events: dict[
            tuple[int, tuple[int, ...]], tuple[torch.cuda.Event, torch.cuda.Event, tuple[torch.cuda.Event, ...]]
        ] = {}

    @contextmanager
    def __call__(self, device: torch.device, *, label: str | None = None, record_sample: bool = True) -> Iterator[None]:
        """Time a completed render; warmups may override the label and discard the sample."""
        label = self._label if label is None else label
        with torch.cuda.device(device):
            device_index = torch.cuda.current_device()
            stream = torch.cuda.current_stream(device)
            producers = tuple(worker for worker in self._streams if worker != stream)
            if any(worker.device != stream.device for worker in producers):
                raise ValueError("All rendering streams must belong to the measured CUDA device")
            event_key = (device_index, tuple(id(worker) for worker in producers))
            if event_key not in self._events:
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                done = tuple(torch.cuda.Event(enable_timing=False) for _ in producers)
                # PyTorch creates CUDA events lazily. Materialize every event
                # before starting measurement, without executing a render.
                for event in (start, stop, *done):
                    event.record(stream)
                stop.synchronize()
                self._events[event_key] = (start, stop, done)
            start, stop, done = self._events[event_key]

            # Drain preparation/warm-up work before recording the first marker.
            torch.cuda.synchronize(device)
            # NVTX labels the host call in Nsight; CUDA events supply the clock.
            torch.cuda.nvtx.range_push(label)
            try:
                start.record(stream)
                for worker in producers:
                    worker.wait_event(start)
                yield
                # Make the stop stream wait for any explicitly registered producers.
                for worker, completion in zip(producers, done, strict=True):
                    completion.record(worker)
                    stream.wait_event(completion)
                # Queue the marker after the final output dependencies; waiting
                # on the host first would add a host round trip to the interval.
                stop.record(stream)
                stop.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
            milliseconds = float(start.elapsed_time(stop))
            if not record_sample:
                return
            valid = math.isfinite(milliseconds) and milliseconds > 0
            if not valid:
                log.warning(
                    "Invalid CUDA rendering duration for %s: %s ms; recording 0 ms with valid=False",
                    label,
                    milliseconds,
                )
            self.milliseconds.append(milliseconds if valid else 0.0)
            self.valid.append(valid)
