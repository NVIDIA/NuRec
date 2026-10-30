# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import torch


class _AsyncWork(Protocol):
    """Minimal interface shared by ``torch.distributed.Work`` and test doubles."""

    def wait(self) -> object:
        """Wait until the asynchronous collective has completed."""
        ...


@dataclass(slots=True)
class AsyncVisibilityUnion:
    """An in-flight global visibility union for one optimizer step.

    ``reduction`` stores one uint8 entry per globally gathered Gaussian.
    Reducing with ``MAX`` makes the normal case a cross-rank OR. A rank without
    a usable visibility source contributes an all-visible mask, forcing the
    global result to the safe dense fallback.
    """

    reduction: torch.Tensor
    local_ranges: tuple[tuple[int, int], ...] | None
    local_source_missing: bool
    _work: _AsyncWork | None = field(default=None, init=False, repr=False)
    _launched: bool = field(default=False, init=False)
    _resolved: torch.Tensor | None = field(default=None, init=False, repr=False)

    def launch(self) -> None:
        """Launch the prepared union after rank-local forward errors are coordinated."""
        if self._launched:
            raise RuntimeError("SelectiveAdam visibility union was launched more than once")

        if (
            self.reduction.numel() == 0
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() <= 1
        ):
            self._launched = True
            return
        if self.reduction.device.type != "cuda":
            raise RuntimeError("multi-rank SelectiveAdam visibility union requires a CUDA tensor")
        if torch.distributed.get_backend() != torch.distributed.Backend.NCCL:
            raise RuntimeError("multi-rank SelectiveAdam visibility union requires the NCCL backend")

        self._work = torch.distributed.all_reduce(
            self.reduction,
            op=torch.distributed.ReduceOp.MAX,
            async_op=True,
        )
        self._launched = True

    def resolve(self) -> torch.Tensor:
        """Wait for the union and return this rank's Boolean owner mask."""
        if self._resolved is not None:
            return self._resolved
        if not self._launched:
            raise RuntimeError("SelectiveAdam visibility union must be launched before it is resolved")

        if self._work is not None:
            self._work.wait()
            self._work = None

        owner_visibility = self.reduction
        if self.local_ranges is not None:
            views = tuple(owner_visibility.narrow(0, start, length) for start, length in self.local_ranges)
            owner_visibility = (
                owner_visibility[:0] if not views else views[0] if len(views) == 1 else torch.cat(views, dim=0)
            )
        self._resolved = owner_visibility.bool()
        return self._resolved


def prepare_async_visibility_union(
    visibility_sources: Sequence[torch.Tensor],
    *,
    element_count: int,
    device: torch.device,
    local_ranges: tuple[tuple[int, int], ...] | None,
    positive_only: bool,
) -> AsyncVisibilityUnion:
    """Prepare a rank-global visibility OR without changing renderer outputs.

    Camera and LiDAR visibility are unioned locally into uint8. The caller
    launches the asynchronous NCCL ``MAX`` only after coordinated forward-error
    handling, so a rank-local forward failure cannot reorder WORLD collectives.
    An empty source set or a malformed provided source contributes an
    all-visible mask instead of being interpreted as all-false. Every rank must
    call this function with the same globally gathered ``element_count``. The
    upstream Gaussian gather owns that rank-uniform layout; the coordinated
    forward-error gate only ensures that every rank reaches
    :meth:`AsyncVisibilityUnion.launch` together.
    """
    if element_count < 0:
        raise ValueError(f"element_count must be non-negative, got {element_count}")

    reduction = torch.empty(element_count, device=device, dtype=torch.uint8)
    source_missing = not visibility_sources or any(source.numel() != element_count for source in visibility_sources)

    if source_missing:
        reduction.fill_(1)
    else:
        for source_index, source in enumerate(visibility_sources):
            source_mask = source.detach().reshape(-1)
            source_mask = source_mask > 0 if positive_only else source_mask.bool()
            if source_index == 0:
                reduction.copy_(source_mask)
            else:
                reduction.logical_or_(source_mask)

    return AsyncVisibilityUnion(
        reduction=reduction,
        local_ranges=local_ranges,
        local_source_missing=source_missing,
    )
