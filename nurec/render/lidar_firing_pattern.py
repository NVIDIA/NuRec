# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observed firing patterns for rendering lidars that have no nominal sensor model.

A model-free lidar (a MEMS unit, say) has no fitted row/column spinning model, so serving
cannot synthesise its rays from intrinsics the way the structured path does. What it can
use instead is an *observed firing pattern*: the sensor-frame direction and in-frame
firing time of every element, measured from real captures.

The pattern is pose-independent -- it says only *where and when* the sensor fires, not
where the sensor is -- so one pattern serves every request; the request's start/end poses
turn it into world rays.
"""

from __future__ import annotations

import logging
import math
import threading

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:
    import torch

    from nurec.utils.batch import UnstructuredLidarTilingData
    from nurec.utils.unstructured_lidar import UnstructuredFov


log = logging.getLogger(__name__)

# Widen the tiling bounds slightly beyond the observed directions. The bounds are a
# pre-cull: a ray on the boundary must stay inside it, and floating-point conversion to
# az/el must not push it out.
_FOV_MARGIN_RAD = np.deg2rad(0.5)
# How far the request's frame duration may sit from the pattern's own sweep before the two
# describe different sensors. A real unit's frames vary by well under this.
_MAX_DURATION_MISMATCH = 0.05
# No sweep runs this long: 1 Hz is an order of magnitude slower than the slowest unit here
# (the H10 P128 sweeps in 100 ms), so an offset past it is not one measured within a frame.
_MAX_SWEEP_US = 1_000_000


@dataclass(frozen=True)
class LidarFiringPattern:
    """One lidar's measured firing geometry.

    Attributes:
        directions_sensor: Unit ray directions in the sensor frame, ``(N, 3)`` float32.
        timestamp_offsets_us: Firing time of each ray measured from the start of the sweep,
            ``(N,)`` int64 and non-negative. Offset to each request's frame start at use, so
            the schedule stays the sensor's own rather than the request's.
        frame_duration_us: How long one frame lasts. NOT the sweep, which is ``span_us``: a
            unit that idles between sweeps fires over 59.7 of its 100.0 ms frame (ET30X).
    """

    directions_sensor: npt.NDArray[np.float32]
    timestamp_offsets_us: npt.NDArray[np.int64]
    frame_duration_us: int

    # Tilings built from this pattern, keyed by device: every input is the unit's own, and
    # the only per-request value is the frame start, which is restamped as metadata.
    _tiling_cache: dict = field(default_factory=dict, repr=False, compare=False)
    _tiling_lock: "threading.Lock" = field(default_factory=lambda: threading.Lock(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.directions_sensor.ndim != 2 or self.directions_sensor.shape[1] != 3:
            raise ValueError(f"directions_sensor must be (N, 3), got {self.directions_sensor.shape}")
        if self.timestamp_offsets_us.shape != (self.directions_sensor.shape[0],):
            raise ValueError(
                f"timestamp_offsets_us must be (N,) matching directions_sensor's N="
                f"{self.directions_sensor.shape[0]}, got {self.timestamp_offsets_us.shape}"
            )
        if self.directions_sensor.shape[0] == 0:
            raise ValueError("firing pattern is empty")
        if not np.isfinite(self.directions_sensor).all():
            raise ValueError("directions_sensor must contain only finite vectors")
        norms = np.linalg.norm(self.directions_sensor, axis=1)
        if not np.isfinite(norms).all():
            raise ValueError("directions_sensor must have finite vector norms")
        if np.any(norms <= 1e-9):
            raise ValueError("directions_sensor must not contain zero-length vectors")
        if self.timestamp_offsets_us.min() < 0:
            raise ValueError("timestamp_offsets_us must be measured from the sweep start, so non-negative")
        if self.timestamp_offsets_us.max() > _MAX_SWEEP_US:
            raise ValueError(
                f"timestamp_offsets_us reaches {int(self.timestamp_offsets_us.max())}us, longer than "
                f"any sweep ({_MAX_SWEEP_US}us): these look like absolute timestamps rather than "
                "offsets measured from the start of the sweep."
            )
        if not 0 < self.frame_duration_us <= _MAX_SWEEP_US:
            raise ValueError(f"frame_duration_us must be in [1, {_MAX_SWEEP_US}], got {self.frame_duration_us}")
        # The margin is jitter, not idle time: a spinning unit sweeps its whole frame and can
        # overrun it slightly (P128: 0.27%).
        if self.timestamp_offsets_us.max() > (1.0 + _MAX_DURATION_MISMATCH) * self.frame_duration_us:
            raise ValueError(
                f"timestamp_offsets_us reaches {int(self.timestamp_offsets_us.max())}us, more than "
                f"{_MAX_DURATION_MISMATCH:.0%} past the {self.frame_duration_us}us frame it sweeps within"
            )

    @property
    def n_rays(self) -> int:
        return int(self.directions_sensor.shape[0])

    @property
    def span_us(self) -> int:
        """Time from the first to the last firing."""
        return int(self.timestamp_offsets_us.max())

    @classmethod
    def load(cls, source: Path | BinaryIO) -> LidarFiringPattern:
        """Load a pattern from an ``.npz`` holding directions, offsets, and frame duration.

        This is the layout produced and read by the observed-pattern tooling. ``source`` is
        a path or an open binary stream, so a pattern carried inside an artifact loads without a
        temp file.
        """
        with np.load(source) as data:
            missing = {"directions_sensor", "timestamp_offsets_us", "frame_duration_us"} - set(data.files)
            if missing:
                raise ValueError(f"{source}: firing pattern is missing {sorted(missing)}; found {sorted(data.files)}")
            directions = np.ascontiguousarray(data["directions_sensor"], dtype=np.float32)
            offsets = np.ascontiguousarray(data["timestamp_offsets_us"], dtype=np.int64)
            duration = np.asarray(data["frame_duration_us"])

        if duration.shape != ():
            raise ValueError(f"{source}: frame_duration_us must be a scalar, got shape {duration.shape}")
        frame_duration_us = int(duration)

        norms = np.linalg.norm(directions, axis=1)
        if not np.isfinite(directions).all():
            raise ValueError(f"{source}: directions_sensor must contain only finite vectors")
        if not np.isfinite(norms).all():
            raise ValueError(f"{source}: directions_sensor must have finite vector norms")
        if np.any(norms <= 1e-9):
            raise ValueError(f"{source}: directions_sensor must not contain zero-length vectors")
        if not np.allclose(norms, 1.0, atol=1e-3):
            log.warning(
                "%s: ray directions are not unit length (min %.4f, max %.4f); normalising.",
                source,
                float(norms.min()),
                float(norms.max()),
            )
            directions = (directions / norms[:, None]).astype(np.float32)

        return cls(
            directions_sensor=directions,
            timestamp_offsets_us=offsets,
            frame_duration_us=frame_duration_us,
        )

    def check_window(self, requested_us: int) -> None:
        """Reject a request that is neither this unit's sweep nor its frame.

        Both are accepted: a client timing frames by the sensor's rate sends the frame, one
        reading the record's own brackets sends the sweep. Either is cropped to the sweep.
        """
        near_sweep = abs(requested_us - self.span_us) <= _MAX_DURATION_MISMATCH * self.span_us
        near_frame = abs(requested_us - self.frame_duration_us) <= _MAX_DURATION_MISMATCH * self.frame_duration_us
        if not (near_sweep or near_frame):
            raise ValueError(
                f"the request asks for a {requested_us / 1000.0:.1f} ms window, which is neither this "
                f"unit's {self.span_us / 1000.0:.1f} ms sweep nor its "
                f"{self.frame_duration_us / 1000.0:.1f} ms frame. Replaying a sweep at another rate "
                "describes a different sensor: the ray origins would cover a different arc of the "
                "pose than the tiling's relative times say they do."
            )

    def ray_timestamps_us(self, frame_start_us: int) -> npt.NDArray[np.int64]:
        """The pattern's firings, offset to a request's frame start.

        The offsets are the sensor's own: how long after starting a sweep it fires each
        element. That is a property of the unit, so it is carried through as measured rather
        than stretched onto the requested window -- a sweep of a different length would be a
        different sensor.
        """
        return frame_start_us + self.timestamp_offsets_us

    def fov(self) -> "UnstructuredFov":
        """The field of view covering every ray, as the tiling consumes it.

        Derived from the pattern itself rather than configured separately -- serving has no
        datasource to ask -- so the tiling bounds cannot disagree with the rays they are
        meant to bound, and units with different fields of view get different bounds. The
        periodic / seam rules are the shared ones, so a full-circle pattern is recognised as
        periodic rather than given a hard edge at +/-pi. The margin is only applied to a
        limited FOV: a periodic one already spans the circle.
        """
        from nurec.utils.unstructured_lidar import PERIODIC_AZ_SPAN_RAD, fov_from_bounds

        d = self.directions_sensor.astype(np.float64)
        az = np.arctan2(d[:, 1], d[:, 0])
        el = np.arcsin(np.clip(d[:, 2] / np.clip(np.linalg.norm(d, axis=1), 1e-9, None), -1.0, 1.0))
        sorted_az = np.sort(az)
        internal_gaps = np.diff(sorted_az)
        wrap_gap = float(2.0 * math.pi - (sorted_az[-1] - sorted_az[0]))
        largest_internal_gap = float(internal_gaps.max()) if internal_gaps.size else 0.0
        circular_span = 2.0 * math.pi - max(largest_internal_gap, wrap_gap)

        if circular_span >= PERIODIC_AZ_SPAN_RAD:
            az_lo, az_hi = -math.pi, math.pi
        elif largest_internal_gap > wrap_gap + 1e-6:
            raise ValueError(
                "the firing pattern's limited azimuth coverage crosses the +/-180 deg seam. "
                "The unstructured azimuth binning is linear with no wrap and cannot represent "
                "this pattern; use a pattern whose limited FOV does not cross the seam."
            )
        else:
            az_lo, az_hi = float(sorted_az[0]), float(sorted_az[-1])
            az_lo = max(az_lo - _FOV_MARGIN_RAD, -math.pi)
            az_hi = min(az_hi + _FOV_MARGIN_RAD, math.pi)
        return fov_from_bounds(
            az_lo,
            az_hi,
            float(el.min()) - _FOV_MARGIN_RAD,
            float(el.max()) + _FOV_MARGIN_RAD,
            source="the firing pattern's measured directions",
        )


def build_unstructured_tiling(
    firing_pattern: LidarFiringPattern,
    frame_start_us: int,
    device: "torch.device",
) -> "UnstructuredLidarTilingData":
    """The render tiling for a pattern's rays, cached per unit and device.

    Mirrors what the dataloader precomputes during training, so serving hands the renderer the
    same structure. Building it dominates the request -- several times the render it feeds --
    and every input is a property of the unit, so one tiling serves every request.
    """
    import dataclasses

    import torch

    from nurec.utils.unstructured_lidar import build_tiling

    cache_key = str(device)

    with firing_pattern._tiling_lock:
        cached = firing_pattern._tiling_cache.get(cache_key)
        if cached is None:
            # Bounds are left to gsplat: the offsets run from 0 to the sweep, which is the
            # window the rays are placed in once the request has been cropped to it.
            cached = build_tiling(
                torch.from_numpy(firing_pattern.directions_sensor),
                torch.from_numpy(firing_pattern.ray_timestamps_us(0)),
                firing_pattern.fov(),
            ).to(device)
            firing_pattern._tiling_cache[cache_key] = cached
            log.info(
                "Built unstructured tiling for a %d-ray pattern sweeping %.1f ms on %s (cached).",
                firing_pattern.n_rays,
                firing_pattern.span_us / 1000.0,
                device,
            )

    # The absolute window is metadata rather than an input, so restamping it leaves the
    # tensors shared.
    meta = dict(cached.meta)
    meta["frame_start_us"] = float(frame_start_us)
    meta["frame_end_us"] = float(frame_start_us + firing_pattern.span_us)
    return dataclasses.replace(cached, meta=meta)
