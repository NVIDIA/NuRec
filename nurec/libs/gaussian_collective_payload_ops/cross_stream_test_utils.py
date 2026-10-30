# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


_PENDING_FREE_STATES = ("active_pending_free", "active_awaiting_free")


@dataclass(frozen=True, slots=True)
class CrossStreamAllocationTestHarness:
    """Hold a CUDA consumer behind a gate while allocation lifetimes are inspected."""

    producer: torch.cuda.Stream
    consumer: torch.cuda.Stream
    gate: torch.cuda.Stream
    producer_ready: torch.cuda.Event
    release_consumer: torch.cuda.Event

    @classmethod
    def create(cls) -> CrossStreamAllocationTestHarness:
        return cls(
            producer=torch.cuda.Stream(),
            consumer=torch.cuda.Stream(),
            gate=torch.cuda.Stream(),
            producer_ready=torch.cuda.Event(),
            release_consumer=torch.cuda.Event(),
        )

    def enqueue_gated(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> torch.cuda.Event:
        """Enqueue ``operation`` after producer work and a deliberately long gate."""
        with torch.cuda.stream(self.gate):
            torch.cuda._sleep(2_000_000_000)
            self.release_consumer.record()

        with torch.cuda.stream(self.consumer):
            self.consumer.wait_event(self.producer_ready)
            self.consumer.wait_event(self.release_consumer)
            operation(*args, **kwargs)
            consumer_done = torch.cuda.Event()
            consumer_done.record()
        return consumer_done

    def assert_allocations_pending(self, *pointers: int) -> None:
        """Assert that released allocations remain owned by the gated stream."""
        if self.release_consumer.query():
            raise AssertionError("consumer gate completed before allocation release")

        snapshot = torch.cuda.memory_snapshot()
        for pointer in pointers:
            blocks = [block for segment in snapshot for block in segment["blocks"] if block["address"] == pointer]
            if len(blocks) != 1:
                raise AssertionError(f"expected one allocator block for {pointer:#x}, got {len(blocks)}")
            if blocks[0]["state"] not in _PENDING_FREE_STATES:
                raise AssertionError(
                    f"allocation {pointer:#x} has state {blocks[0]['state']!r}, expected one of {_PENDING_FREE_STATES}"
                )
