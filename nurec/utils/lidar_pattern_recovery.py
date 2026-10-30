# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading a lidar's firing pattern out of what a capture recorded.

A unit without a parametric sensor model is served by replaying the directions it was
observed to fire. A record that retains every slot fired -- returning or not -- already
holds that in a single frame, so the pattern is read from one frame as measured: nothing
is reconstructed, nothing is averaged. A record of returns alone under-covers what the
sensor fires and is rejected rather than approximated.

Used by the artifact export, so a model-free unit carries its own pattern.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt


log = logging.getLogger(__name__)

# Frames looked at when picking the frame to export.
SCAN_FRAMES_DEFAULT = 50


def _full_sweep_mask(sensor) -> npt.NDArray[np.bool_]:
    """Which frames ran a full sweep, by duration.

    The 0.9 threshold sits in a wide gap: a full sweep varies by 0.08% (P128), 0.15% (ET30X)
    and 2.0% (M1P) of the median, while a truncated one is 21% of it.
    """
    # frames_timestamps_us is (n_frames, 2) = [start, end], so this is end - start per frame.
    spans = np.diff(np.asarray(sensor.frames_timestamps_us), axis=1).reshape(-1)
    return spans >= 0.9 * float(np.median(spans))


def frame_duration_us(sensor) -> int | None:
    """How long one frame lasts, i.e. how often the unit starts a sweep.

    ``frames_timestamps_us`` brackets the rays a frame carries, so its own span is the sweep, not
    the frame (ET30X: 59.7 of 100.0 ms), and no record states a nominal rate. The median over
    successive frame starts ignores the truncated first and last frames.

    ``None`` when there is no interval to read -- one frame cannot say how far apart frames are.
    """
    starts = np.asarray(sensor.frames_timestamps_us, dtype=np.int64)[:, 0]
    if starts.size < 2:
        return None
    return int(np.median(np.diff(starts)))


def _non_returning_slots(sensor, frame_index: int) -> int:
    """Slots this frame carries that never came back.

    Non-zero means the record lists every direction fired, so one frame is the whole pattern.
    """
    dist = np.asarray(
        sensor.get_frame_ray_bundle_return_distance_m(frame_index=frame_index, return_index=0), dtype=np.float64
    )
    return int((~(np.isfinite(dist) & (dist > 0))).sum())


def _read_frame(sensor, frame_index: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """One frame's fired directions and their offsets from that frame's own start.

    Offsets are relative to the frame because the pattern describes one sweep: how long after
    starting it the unit fires each element, which is the same for every frame it runs.
    """
    d = np.ascontiguousarray(sensor.get_frame_ray_bundle_direction(frame_index), dtype=np.float32)
    t = np.asarray(sensor.get_frame_ray_bundle_timestamp_us(frame_index), dtype=np.int64)
    if d.shape[0] != t.shape[0]:
        raise ValueError(f"directions ({d.shape[0]}) and timestamps ({t.shape[0]}) disagree for frame {frame_index}")
    if not np.isfinite(d).all():
        raise ValueError(f"frame {frame_index} has non-finite ray directions")

    return d, t - int(sensor.frames_timestamps_us[frame_index][0])


def recover_firing_pattern(
    sensor,
    *,
    frame: int | None = None,
    scan_frames: int = SCAN_FRAMES_DEFAULT,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], int]:
    """One observed pattern as ``(directions, offsets, frame_duration_us)``.

    A record that lists every slot the sensor fired -- returning or not -- already holds the
    whole pattern in one frame, so one frame is read as-is. Nothing is reconstructed and
    nothing is averaged: the exported directions are the measured ones.

    Raises:
        ValueError: if the sensor has no frame that can stand in for a full sweep, or if the
            record carries returns alone, in which case one frame under-covers what was fired.
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
            f"frame {frame} carries only slots that returned, so it under-covers what the sensor "
            f"fired and cannot stand in for its firing pattern. This needs a record that retains "
            f"non-returning emissions."
        )

    duration_us = frame_duration_us(sensor)
    if duration_us is None:
        raise ValueError(
            "this record holds a single frame, so it cannot say how long a frame lasts. The "
            "pattern is replayed over a frame, so serving needs that duration."
        )

    directions, offsets = _read_frame(sensor, frame)
    return directions, offsets, duration_us


def choose_frame(sensor, scan_frames: int = SCAN_FRAMES_DEFAULT) -> int:
    """The frame that best stands in for one sweep, among the first ``scan_frames``.

    A truncated sweep dates its rays from its own first ray, so its offsets are short by the
    missing lead-in; it cannot stand in for one frame's sweep however much it fired.
    """
    scanned = min(scan_frames, sensor.frames_count)
    full = _full_sweep_mask(sensor)
    candidates = [i for i in range(scanned) if full[i]]
    if not candidates:
        raise ValueError(f"no full sweep among the first {scanned} frames; scan more of them")
    # Slots fired, which is what gets exported -- not slots that returned, which describes
    # the scene rather than the sensor.
    counts = [len(sensor.get_frame_ray_bundle_timestamp_us(i)) for i in candidates]
    frame = candidates[int(np.argmax(counts))]
    log.info(f"frame {frame} chosen: most slots fired ({max(counts)}) of {len(candidates)} full sweeps")
    return frame


def describe_pattern(
    directions: npt.NDArray[np.float32], offsets: npt.NDArray[np.int64], frame_duration_us: int
) -> str:
    """One line of the pattern's extent, for the export log."""
    azimuth = np.degrees(np.arctan2(directions[:, 1], directions[:, 0]))
    elevation = np.degrees(np.arcsin(np.clip(directions[:, 2], -1.0, 1.0)))
    return (
        f"{directions.shape[0]} rays, sweep {offsets.max() / 1000.0:.1f} ms of a "
        f"{frame_duration_us / 1000.0:.1f} ms frame, "
        f"az [{azimuth.min():.1f}, {azimuth.max():.1f}] deg, el [{elevation.min():.1f}, {elevation.max():.1f}] deg"
    )
