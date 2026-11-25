# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover packetless firing-pattern data from a dense recorded frame.

A model-free lidar can replay the directions and firing times preserved in a dense
recording. Recovery reads one observed frame in its original ray order, without
averaging directions or reconstructing missing emissions. The recording must retain
non-returning emissions as well as returns; a record of returns alone is rejected.

Artifact export uses this as a fallback when native firing-pattern templates are
unavailable. Packet boundaries cannot be recovered from these frame arrays.
"""

from __future__ import annotations

import logging

from typing import Protocol

import numpy as np
import numpy.typing as npt


log = logging.getLogger(__name__)

# Frames inspected when choosing the complete frame to export.
SCAN_FRAMES_DEFAULT = 50


class _RecoverableLidarSensor(Protocol):
    """Narrow sensor surface required for observed-frame pattern recovery."""

    @property
    def frames_count(self) -> int: ...

    @property
    def frames_timestamps_us(self) -> npt.NDArray[np.integer]: ...

    def get_frame_ray_bundle_direction(self, frame_index: int) -> npt.NDArray[np.floating]: ...

    def get_frame_ray_bundle_timestamp_us(self, frame_index: int) -> npt.NDArray[np.integer]: ...

    def get_frame_ray_bundle_return_distance_m(
        self,
        frame_index: int,
        return_index: int = 0,
    ) -> npt.NDArray[np.floating]: ...


def _full_sweep_mask(sensor: _RecoverableLidarSensor) -> npt.NDArray[np.bool_]:
    """Return the frames that have a full sweep duration."""

    spans = np.diff(np.asarray(sensor.frames_timestamps_us), axis=1).reshape(-1)
    return spans >= 0.9 * float(np.max(spans))


def frame_duration_us(sensor: _RecoverableLidarSensor) -> int | None:
    """Estimate the frame period from the median interval between frame starts.

    A frame's start-to-end span covers its recorded sweep, which can be shorter than
    the period between sweeps. Use successive starts to preserve that distinction.

    Args:
        sensor: Lidar sensor exposing ``frames_timestamps_us`` as an array of shape
            ``(n_frames, 2)`` containing frame start and end times in microseconds.

    Returns:
        The median interval in integer microseconds, or ``None`` when fewer than
        two frames are available. The caller must check that the period is positive.
    """

    starts = np.asarray(sensor.frames_timestamps_us, dtype=np.int64)[:, 0]
    if starts.size < 2:
        return None
    return int(np.median(np.diff(starts)))


def _non_returning_slots(sensor: _RecoverableLidarSensor, frame_index: int) -> int:
    """Count fired slots retained despite having no valid return."""

    distances = np.asarray(
        sensor.get_frame_ray_bundle_return_distance_m(frame_index=frame_index, return_index=0),
        dtype=np.float64,
    )
    return int((~(np.isfinite(distances) & (distances > 0))).sum())


def _read_frame(
    sensor: _RecoverableLidarSensor,
    frame_index: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Read one frame's ordered directions and timestamps relative to its start."""

    directions = np.ascontiguousarray(sensor.get_frame_ray_bundle_direction(frame_index), dtype=np.float32)
    timestamps = np.asarray(sensor.get_frame_ray_bundle_timestamp_us(frame_index), dtype=np.int64)
    if directions.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"directions ({directions.shape[0]}) and timestamps ({timestamps.shape[0]}) "
            f"disagree for frame {frame_index}"
        )
    if not np.isfinite(directions).all():
        raise ValueError(f"frame {frame_index} has non-finite ray directions")

    return directions, timestamps - int(sensor.frames_timestamps_us[frame_index][0])


def recover_firing_pattern(
    sensor: _RecoverableLidarSensor,
    *,
    frame: int | None = None,
    scan_frames: int = SCAN_FRAMES_DEFAULT,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], int]:
    """Read one observed firing pattern without changing its ray order or timing.

    Recovery requires a dense recording that retains every emitted ray, including
    rays without a valid return. Directions are read as recorded, without averaging
    or normalization, and timestamps become offsets from the selected frame's start.

    Args:
        sensor: Lidar sensor exposing ``frames_count``, ``frames_timestamps_us``,
            and per-frame ray directions, timestamps, and first-return distances.
            Ray arrays must retain non-returning emissions in their original order.
        frame: Zero-based frame index to read. If ``None``, use ``choose_frame``.
            An explicitly selected truncated sweep produces a warning because its
            offsets may omit the missing lead-in; it is not rejected for truncation.
        scan_frames: Maximum number of initial frames considered when ``frame`` is
            ``None``. Ignored when an explicit frame index is supplied.

    Returns:
        A tuple of sensor-frame directions as a contiguous ``(n_rays, 3)`` float32
        array, offsets from frame start as an ``(n_rays,)`` int64 array in
        microseconds, and the positive frame period in integer microseconds.

    Raises:
        ValueError: If automatic selection finds no recoverable full sweep, the
            explicit frame index is out of bounds, or the selected frame contains
            no non-returning emissions. Also raised when fewer than two frames are
            available, the estimated period is non-positive, direction and timestamp
            counts differ, directions are non-finite, or a ray predates frame start.
    """

    if frame is None:
        frame = choose_frame(sensor, scan_frames)
    elif not 0 <= frame < sensor.frames_count:
        raise ValueError(f"frame {frame} is outside this lidar's 0..{sensor.frames_count - 1}")
    elif not _full_sweep_mask(sensor)[frame]:
        log.warning(
            "frame %d is a truncated sweep: its offsets are short by the missing lead-in, "
            "which shifts every ray once the pose is interpolated.",
            frame,
        )

    if not _non_returning_slots(sensor, frame):
        raise ValueError(
            f"frame {frame} carries only slots that returned, so it cannot prove that the record "
            "retains the complete firing schedule. Recovery needs dense rays including "
            "non-returning emissions."
        )

    duration_us = frame_duration_us(sensor)
    if duration_us is None:
        raise ValueError(
            "this record holds a single frame, so it cannot determine the frame period required "
            "to replay the firing pattern"
        )
    if duration_us <= 0:
        raise ValueError(f"the recovered frame period must be positive, got {duration_us}")

    directions, offsets = _read_frame(sensor, frame)
    if np.any(offsets < 0):
        raise ValueError(f"frame {frame} contains ray timestamps before its declared start")
    return directions, offsets, duration_us


def choose_frame(sensor: _RecoverableLidarSensor, scan_frames: int = SCAN_FRAMES_DEFAULT) -> int:
    """Choose the longest recoverable complete frame among the first ``scan_frames``.

    A full sweep spans at least 90% of the longest recorded sweep duration, so a
    majority of truncated frames cannot lower the completeness threshold. Among
    full sweeps retaining non-returning emissions, prefer the longest recorded ray
    timestamp span, then the most ray slots; ties select the earliest frame.

    Args:
        sensor: Lidar sensor exposing ``frames_count``, ``frames_timestamps_us``,
            and per-frame ray timestamps and first-return distances.
        scan_frames: Maximum number of initial frames to inspect. Must be positive;
            fewer frames are inspected when the recording is shorter.

    Returns:
        The zero-based index of the selected frame.

    Raises:
        ValueError: If there are no frames to inspect, no full sweep occurs within
            the scan limit, or none of those full sweeps retains non-returning rays.
    """

    scanned = min(scan_frames, sensor.frames_count)
    if scanned <= 0:
        raise ValueError("the lidar record contains no frames")
    full = _full_sweep_mask(sensor)
    candidates = [index for index in range(scanned) if full[index]]
    if not candidates:
        raise ValueError(f"no full sweep among the first {scanned} frames; scan more of them")

    def rank(index: int) -> tuple[int, int]:
        timestamps = np.asarray(sensor.get_frame_ray_bundle_timestamp_us(index), dtype=np.int64)
        span = int(timestamps.max()) - int(timestamps.min()) if timestamps.size else 0
        return -span, -timestamps.size

    ranked = sorted(candidates, key=rank)
    for frame in ranked:
        if _non_returning_slots(sensor, frame) > 0:
            count = len(sensor.get_frame_ray_bundle_timestamp_us(frame))
            log.info(
                "frame %d chosen: longest ray span (%d slots) among %d candidate full sweeps",
                frame,
                count,
                len(candidates),
            )
            return frame
    raise ValueError(
        f"none of the {len(candidates)} full sweeps among the first {scanned} frames retains "
        "dense rays including non-returning emissions"
    )


def describe_pattern(
    directions: npt.NDArray[np.float32],
    offsets: npt.NDArray[np.int64],
    frame_duration_us: int,
) -> str:
    """Describe one recovered pattern for the export log.

    Args:
        directions: Non-empty ``(n_rays, 3)`` array of unit ray directions in the
            sensor frame, ordered consistently with ``offsets``.
        offsets: Non-empty ``(n_rays,)`` array of firing-time offsets from frame
            start, in microseconds.
        frame_duration_us: Period between successive frame starts, in microseconds.

    Returns:
        A single line containing the ray count, latest firing offset and frame
        period in milliseconds, and azimuth and elevation ranges in degrees.

    Raises:
        ValueError: If either array is empty and its extent cannot be computed.
    """

    azimuth = np.degrees(np.arctan2(directions[:, 1], directions[:, 0]))
    elevation = np.degrees(np.arcsin(np.clip(directions[:, 2], -1.0, 1.0)))
    return (
        f"{directions.shape[0]} rays, sweep {offsets.max() / 1000.0:.1f} ms of a "
        f"{frame_duration_us / 1000.0:.1f} ms frame, "
        f"az [{azimuth.min():.1f}, {azimuth.max():.1f}] deg, "
        f"el [{elevation.min():.1f}, {elevation.max():.1f}] deg"
    )
