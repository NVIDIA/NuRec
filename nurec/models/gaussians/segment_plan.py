# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side segment planning for the sparse collect / render path.

A segment is a window of the collected Gaussian rows. The host decides which windows
are live for a frame, the collector writes only those rows (in place, see
``LayerSegments`` in collector.slang), and the renderer draws the same window list.
Unchanged source tensors require no device readback: their host tables are reused, and
a sparse frame marshals its collect and draw ranges with one host-to-device copy.

The time rule is a superset of the pose interpolation's acceptance test
(``fetch_tracks`` in collector.slang): a track is live when the frame window
``[frame_start, frame_end]`` overlaps ``[first - tol, last + tol]`` with
``tol = max_positive_timestamp_gap + frame_exposure``. Tracks the kernel then rejects
still get density 0 through ``keep_mask``, so the rendered image is the dense one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, NamedTuple, Optional, Sequence

import numpy as np
import torch

from nurec.models.gaussians.collect import (
    LayerSegments,
    TrackTimeData,
    whole_layer_segments,
    windowed_layer_segments,
)


_INT32_MAX = torch.iinfo(torch.int32).max


def _exclusive_prefix(counts: np.ndarray) -> np.ndarray:
    """``[W + 1]`` exclusive prefix sum of ``counts`` ([W]); the last entry is the total."""
    return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))


def assert_sparse_segments_supported(
    *,
    has_extra_signals: bool,
    has_active_tile_mask: bool,
    grad_enabled: bool,
) -> None:
    """Raise when a sparse draw plan cannot be used for this frame.

    These are the three unsupported features a config load cannot see: the layers
    config owns the extra signal widths, and the tile mask and the gradient mode belong
    to the frame. Everything else the sparse path does not support is fixed for the run
    -- the renderer name, packed, sparse_grad, absgrad, scene visibility and
    unstructured LiDAR -- and ``GSplatRendererConfig.validate_sparse_segments_support``
    rejects it when the config is loaded.

    The composite asks before it builds a plan, with the values it collects for; the
    renderer asks before it draws one, with the values of the call it received. Reading
    one list means the two cannot come to different answers as the list grows or shrinks.
    """
    if has_extra_signals:
        raise ValueError("sparse_segments does not support extra signals.")
    if has_active_tile_mask:
        raise ValueError("sparse_segments does not support active tile masks.")
    if grad_enabled:
        raise ValueError("sparse_segments supports inference without gradients only.")


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """A gsplat draw list over collected Gaussian rows.

    ``segments`` has shape ``[2, segment_count]``: row 0 holds start rows, row 1 row
    counts. ``segment_offsets`` has shape ``[segment_count + 1]`` and holds the first
    packed row of each segment. ``active_count`` is the drawn row count, ``total_rows``
    the collected row count.

    This is the renderer's half of what ``upload_segments`` uploads; the per-layer
    ``LayerSegments`` is the collector's. They stay two types because they are two
    values, not two views of one: this list is in collected-scene rows and gsplat wants
    starts and counts stacked, that one is in layer-local rows and the collect kernel
    reads starts only.
    """

    segments: torch.Tensor
    segment_offsets: torch.Tensor
    active_count: int
    total_rows: int


@dataclass(slots=True)
class LayerSegmentTable:
    """Host metadata used to select contiguous row ranges for one layer."""

    track_ids: torch.Tensor
    row_count: int
    _ranges: np.ndarray
    _range_tracks: np.ndarray
    _time_bounds: Optional[np.ndarray]
    _time_source_refs: tuple[torch.Tensor, ...] = ()

    # Rows grouped by track give one window per track. Rows that are not (checkpoints
    # predating the per-track Morton sort, compact artifacts) give one per run, which is
    # why the composite re-sorts them at load. A little slack over one window per track
    # keeps the check from tripping on a small layer that a few extra runs cost nothing.
    MAX_WINDOWS_PER_TRACK = 8
    MIN_WINDOWS_ALLOWED = 64

    @classmethod
    def window_limit(cls, track_count: int) -> int:
        """Most windows a layer whose rows are grouped by track can have."""
        return max(cls.MIN_WINDOWS_ALLOWED, cls.MAX_WINDOWS_PER_TRACK * track_count)

    @staticmethod
    def _time_source(time_data: Optional[TrackTimeData]) -> tuple[torch.Tensor, ...]:
        """The tensors the live ranges are derived from; empty without pose times."""
        if time_data is None:
            return ()
        return (time_data.tracks_timestamps_us, time_data.tracks_packinfo, time_data.max_positive_timestamp_gap_us)

    @staticmethod
    def _time_ranges(data: Optional[TrackTimeData]) -> Optional[np.ndarray]:
        """``[3, num_tracks]`` first, last, and endpoint tolerance per track, on the host."""
        if data is None:
            return None
        packinfo = data.tracks_packinfo.to(torch.int64)
        start, length = packinfo[:, 0], packinfo[:, 1]
        has_poses = length > 0
        timestamps = data.tracks_timestamps_us
        if timestamps.numel() == 0:
            assert not bool(has_poses.any()), "a track claims poses but the layer has no pose timestamps"
            # Nothing to gather. last < first is what ``select`` reads as dead.
            first, last = torch.zeros_like(start), torch.full_like(start, -1)
        else:
            # A pose-less track's start index can point past the end of the list, so send
            # those lanes to row 0 and throw the gathered value away. A live track's index
            # is used as-is: an out-of-range one is a corrupt packinfo, not something to
            # clamp into silence.
            zero = torch.zeros_like(start)
            first = torch.where(has_poses, timestamps[torch.where(has_poses, start, zero)], 0)
            last = torch.where(has_poses, timestamps[torch.where(has_poses, start + length - 1, zero)], -1)
        return torch.stack((first, last, data.max_positive_timestamp_gap_us.to(torch.int64))).cpu().numpy()

    @classmethod
    def for_layer(
        cls,
        table: Optional[LayerSegmentTable],
        track_ids: torch.Tensor,
        time_data: Optional[TrackTimeData],
        *,
        layer_name: str = "",
    ) -> LayerSegmentTable:
        """The table for this frame's tensors, reusing whatever is still valid.

        New track ids rebuild everything; new pose timestamps alone only redo the live
        ranges, which is a gather over the tracks rather than over every row.
        """
        if table is None or table.track_ids is not track_ids:
            return cls.build(track_ids, time_data, layer_name=layer_name)
        source, refs = cls._time_source(time_data), table._time_source_refs
        if len(source) == len(refs) and all(new is old for new, old in zip(source, refs)):
            return table
        table._time_bounds = table._time_ranges(time_data)
        table._time_source_refs = source
        return table

    @classmethod
    def build(
        cls,
        track_ids: torch.Tensor,
        time_data: Optional[TrackTimeData],
        *,
        layer_name: str = "",
    ) -> LayerSegmentTable:
        assert track_ids.ndim == 1, f"track ids must be one value per layer row, got shape {tuple(track_ids.shape)}"
        tracks, counts = torch.unique_consecutive(track_ids, return_counts=True)
        range_tracks, run_sizes = torch.stack((tracks.to(torch.int64), counts.to(torch.int64))).cpu().numpy()
        ranges = np.column_stack((_exclusive_prefix(run_sizes)[:-1], run_sizes))
        row_count = int(track_ids.numel())
        time_bounds = cls._time_ranges(time_data)
        track_count = 0 if time_bounds is None else int(time_bounds.shape[1])
        if track_count == 0 and range_tracks.size:
            track_count = int(range_tracks.max()) + 1
        # GaussiansComposite._ensure_tracked_rows_grouped re-sorts any layer that
        # fails this before the first sparse frame and verifies the sort worked, so
        # reaching it here means a rigid layer's rows were reordered behind that repair.
        if ranges.shape[0] > cls.window_limit(track_count):
            raise ValueError(
                f"layer {layer_name!r} has {ranges.shape[0]} track runs for {track_count} tracks; "
                "sparse_segments needs its rows grouped by track. Re-capture the checkpoint with "
                "checkpoint.morton_sort.sort_track_layers, or turn sparse_segments off."
            )
        return cls(
            track_ids=track_ids,
            row_count=row_count,
            _ranges=ranges,
            _range_tracks=range_tracks,
            _time_bounds=time_bounds,
            _time_source_refs=cls._time_source(time_data),
        )

    def select(self, frame_start_us: int, frame_end_us: int, frame_exposure_us: int) -> np.ndarray:
        """Return ordered, non-overlapping layer-local ``[start, count]`` ranges."""
        if self._time_bounds is None:
            return whole_layer_ranges(self.row_count)
        first_us, last_us, endpoint_tolerance_us = self._time_bounds
        tolerance = endpoint_tolerance_us + int(frame_exposure_us)
        live_tracks = (
            (last_us >= first_us)
            & (last_us + tolerance >= int(frame_start_us))
            & (first_us - tolerance <= int(frame_end_us))
        )
        live = live_tracks[self._range_tracks]
        if live.all():
            return whole_layer_ranges(self.row_count)
        return self._ranges[live]


def whole_layer_ranges(row_count: int) -> np.ndarray:
    """Return the canonical range list for a whole layer."""
    if row_count == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array([[0, row_count]], dtype=np.int64)


def collect_ranges(
    selections: Optional[Sequence[np.ndarray]],
    layer_sizes: Sequence[int],
    variant_ids: Collection[int],
    cold: bool,
) -> list[np.ndarray]:
    """The layer-local rows this frame's collect writes: one ``[count, 2]`` array per layer.

    The collect list in host form, and the one definition of it. A variant layer is
    written through its selected ranges, or whole when there are no selections (the
    dense path); a cacheable layer is written whole on a cold frame and not at all on a
    warm one.

    ``upload_segments`` marshals exactly this to the device. The composite also hands it
    to ``post_process``, because a read-modify-write over collected rows has to see the
    rows this frame wrote rather than the whole layer -- rows it did not write already
    carry a previous frame's transform.
    """
    variant = {int(i) for i in variant_ids}
    ranges: list[np.ndarray] = []
    for i, size in enumerate(layer_sizes):
        if i in variant and selections is not None:
            ranges.append(selections[i])
        elif i in variant or cold:
            ranges.append(whole_layer_ranges(int(size)))
        else:
            ranges.append(np.zeros((0, 2), dtype=np.int64))
    return ranges


def covers_all(ranges: np.ndarray, row_count: int) -> bool:
    """Whether ``ranges`` is the single range spanning every one of ``row_count`` rows."""
    return bool(ranges.shape[0] == 1 and ranges[0, 0] == 0 and ranges[0, 1] == row_count)


def rows_grouped_by_track(track_ids: torch.Tensor, track_count: int) -> bool:
    """Whether a tracked layer's rows form few enough runs for per-track windows. One device sync.

    This is the same limit ``LayerSegmentTable.build`` asserts on; the composite checks it
    at load so the assert can never fire mid-render.
    """
    assert track_ids.ndim == 1, f"track ids must be one value per layer row, got shape {tuple(track_ids.shape)}"
    return int(torch.unique_consecutive(track_ids).numel()) <= LayerSegmentTable.window_limit(track_count)


class _Pending(NamedTuple):
    """A window list waiting on the upload: where its two arrays land in it, and the rows they cover."""

    windows: int
    prefix: int
    active_count: int


def upload_segments(
    device: torch.device,
    selections: Sequence[np.ndarray],
    layer_sizes: Sequence[int],
    layer_offsets: Sequence[int],
    variant_ids: Collection[int],
    cold: bool,
) -> tuple[list[LayerSegments], Optional[SegmentPlan]]:
    """Marshal one frame's row ranges into a single device copy.

    ``selections`` holds this frame's live ranges for every layer, in layer-local rows:
    one ordered, non-overlapping ``[segment_count, 2]`` int64 array per layer, as
    ``LayerSegmentTable.select`` returns it. Dense frames are handled by the caller and
    do not enter this function.
    Two lists come out of it. The draw list is the selections of every layer, shifted
    into collected-scene rows and merged where they touch; it does not depend on the
    cache, and it is ``None`` when every row is live. The collect list says what this
    frame writes, and it does depend on the cache: a variant layer is written through
    its windows, a cacheable layer whole on a cold frame and not at all on a warm one.

    Every array of both lists goes into one int32 buffer and one copy. On CUDA, the
    temporary host tensor uses PyTorch's caching pinned allocator; ``copy_`` records
    its stream use so the allocator does not reuse the block before the transfer ends.
    """
    total_rows = int(sum(layer_sizes))
    if total_rows > _INT32_MAX:
        raise ValueError(f"the segment plan must fit in int32, got {total_rows} rows")

    parts: list[np.ndarray] = []

    def add(values: np.ndarray) -> int:
        parts.append(np.ascontiguousarray(values, dtype=np.int32).reshape(-1))
        return len(parts) - 1

    collected: list[LayerSegments | _Pending] = []
    for ranges, size in zip(collect_ranges(selections, layer_sizes, variant_ids, cold), layer_sizes, strict=True):
        if ranges.shape[0] == 0:
            # Nothing to write: a warm cacheable layer, or one whose tracks are all dead.
            collected.append(whole_layer_segments(0))
        elif covers_all(ranges, int(size)):
            collected.append(whole_layer_segments(int(size)))
        else:
            prefix = _exclusive_prefix(ranges[:, 1])
            collected.append(_Pending(add(ranges[:, 0]), add(prefix), int(prefix[-1])))

    # Ranges arrive in row order and do not overlap; one that starts where the previous
    # one ends extends it, so a run ends with its last range.
    shifted = [r + np.array([int(o), 0], dtype=np.int64) for r, o in zip(selections, layer_offsets, strict=True)]
    rows = np.concatenate(shifted) if shifted else np.zeros((0, 2), dtype=np.int64)
    ends = rows[:, 0] + rows[:, 1]
    opens_run = np.ones(rows.shape[0], dtype=bool)
    opens_run[1:] = rows[1:, 0] != ends[:-1]
    closes_run = np.ones(rows.shape[0], dtype=bool)
    closes_run[:-1] = opens_run[1:]
    draw = np.column_stack((rows[opens_run, 0], ends[closes_run] - rows[opens_run, 0]))

    drawn = None
    if not covers_all(draw, total_rows):
        prefix = _exclusive_prefix(draw[:, 1])
        drawn = _Pending(add(draw.T), add(prefix), int(prefix[-1]))

    # Whole/nothing collect entries and a fully live draw need no uploaded arrays.
    flat = _upload_int32(np.concatenate(parts), device) if parts else None
    views = () if flat is None else torch.split(flat, [int(part.size) for part in parts])

    segments = [
        entry
        if isinstance(entry, LayerSegments)
        else windowed_layer_segments(views[entry.windows], views[entry.prefix], entry.active_count)
        for entry in collected
    ]
    draw_plan = (
        None
        if drawn is None
        else SegmentPlan(
            segments=views[drawn.windows].reshape(2, draw.shape[0]),
            segment_offsets=views[drawn.prefix],
            active_count=drawn.active_count,
            total_rows=total_rows,
        )
    )
    return segments, draw_plan


def _upload_int32(values: np.ndarray, device: torch.device) -> torch.Tensor:
    """Upload through PyTorch's event-aware caching pinned allocator."""
    flat = np.ascontiguousarray(values, dtype=np.int32).reshape(-1)
    if device.type != "cuda":
        return torch.from_numpy(flat.copy()).to(device)

    host = torch.empty(flat.size, dtype=torch.int32, pin_memory=True)
    host.numpy()[:] = flat
    return host.to(device=device, non_blocking=True)


__all__ = [
    "LayerSegmentTable",
    "SegmentPlan",
    "assert_sparse_segments_supported",
    "collect_ranges",
    "covers_all",
    "rows_grouped_by_track",
    "upload_segments",
    "whole_layer_ranges",
]
