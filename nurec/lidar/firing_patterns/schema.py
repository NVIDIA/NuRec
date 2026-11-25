# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, sensor-agnostic lidar firing-pattern template data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


SCHEMA_VERSION = 1
MAX_DURATION_MISMATCH = 0.05

DIRECTIONS_ARRAY_NAME = "lidar_firing_pattern_directions_sensor"
TIMESTAMP_OFFSETS_ARRAY_NAME = "lidar_firing_pattern_timestamp_offsets_us"
TEMPLATE_RAY_OFFSETS_ARRAY_NAME = "lidar_firing_pattern_template_ray_offsets"
PACKET_RAY_OFFSETS_ARRAY_NAME = "lidar_firing_pattern_packet_ray_offsets"
TEMPLATE_PACKET_OFFSETS_ARRAY_NAME = "lidar_firing_pattern_template_packet_offsets"


def _require_plain_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


def validate_template_grid(
    spans_us: Sequence[int],
    reference_frame_start_us: int,
    frame_period_us: int,
    frame_start_tolerance_us: int,
) -> None:
    """Validate the shared export and runtime frame-grid contract.

    Args:
        spans_us: Maximum firing offset of each nonempty template, in phase order.
        reference_frame_start_us: Integer timestamp of phase zero in microseconds.
        frame_period_us: Positive integer interval between frame starts in microseconds.
        frame_start_tolerance_us: Nonnegative integer fit tolerance, below half a period.

    Returns:
        ``None`` when every template fits within 105% of the frame period.

    Raises:
        ValueError: If the cycle is empty, timing scalars are invalid, or a firing
            span exceeds the period tolerance. Ray counts may differ by phase.
    """
    if not spans_us:
        raise ValueError("a firing-pattern template set cannot be empty")
    _require_plain_int("reference_frame_start_us", reference_frame_start_us)
    _require_plain_int("frame_period_us", frame_period_us)
    _require_plain_int("frame_start_tolerance_us", frame_start_tolerance_us)
    if frame_period_us <= 0:
        raise ValueError("frame_period_us must be positive")
    if not 0 <= frame_start_tolerance_us < frame_period_us / 2:
        raise ValueError("frame_start_tolerance_us must be less than half a period")
    if any(span < 0 or span > (1.0 + MAX_DURATION_MISMATCH) * frame_period_us for span in spans_us):
        raise ValueError("a firing-pattern template cannot outlast its frame period")


@dataclass(frozen=True)
class LidarFiringPatternTemplatesData:
    """Flattened firing-pattern templates with optional packet boundaries.

    Args:
        directions_sensor: Float32 unit vectors of shape ``(total_rays, 3)`` in
            sensor coordinates, concatenated in template order.
        timestamp_offsets_us: Non-negative int64 firing offsets of shape
            ``(total_rays,)``, relative to each template's frame start.
        template_ray_offsets: Increasing int64 prefix offsets of shape
            ``(template_count + 1,)``, from zero through ``total_rays``.
            Ray counts may differ between templates.
        reference_frame_start_us: Integer timestamp of phase zero, in microseconds.
        frame_period_us: Positive integer interval between frame starts, in microseconds.
        frame_start_tolerance_us: Non-negative integer tolerance used when fitting
            the recorded frame grid, in microseconds; less than half a period.
        packet_ray_offsets: Optional increasing int64 packet boundaries in the
            concatenated ray arrays, from zero through ``total_rays``.
        template_packet_offsets: Optional increasing int64 prefix offsets into the
            packet sequence, one boundary per template plus the final packet count.
            Both packet arrays must be supplied together, with aligned boundaries.

    Raises:
        ValueError: If scalar types, array shapes or dtypes, unit directions,
            timing values, frame-period limits, or prefix boundaries are invalid.
    """

    directions_sensor: npt.NDArray[np.float32]
    timestamp_offsets_us: npt.NDArray[np.int64]
    template_ray_offsets: npt.NDArray[np.int64]
    reference_frame_start_us: int
    frame_period_us: int
    frame_start_tolerance_us: int
    packet_ray_offsets: npt.NDArray[np.int64] | None = None
    template_packet_offsets: npt.NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        if self.directions_sensor.dtype != np.float32:
            raise ValueError("directions_sensor must be float32")
        if self.directions_sensor.ndim != 2 or self.directions_sensor.shape[1] != 3:
            raise ValueError(f"directions_sensor must have shape [total_rays, 3], got {self.directions_sensor.shape}")
        total_rays = len(self.directions_sensor)
        if total_rays == 0:
            raise ValueError("a firing-pattern template set cannot be empty")
        if not np.all(np.isfinite(self.directions_sensor)):
            raise ValueError("directions_sensor must contain only finite values")
        if not np.allclose(
            np.linalg.norm(self.directions_sensor, axis=1),
            1.0,
            rtol=0.0,
            atol=1e-4,
        ):
            raise ValueError("directions_sensor must contain unit vectors")

        if self.timestamp_offsets_us.dtype != np.int64:
            raise ValueError("timestamp_offsets_us must be int64")
        if self.timestamp_offsets_us.shape != (total_rays,):
            raise ValueError("timestamp_offsets_us must have shape [total_rays] matching directions_sensor")
        if np.any(self.timestamp_offsets_us < 0):
            raise ValueError("timestamp_offsets_us must be non-negative")

        for name, offsets in (("template_ray_offsets", self.template_ray_offsets),):
            if offsets.dtype != np.int64:
                raise ValueError(f"{name} must be int64")
            if offsets.ndim != 1 or len(offsets) < 2:
                raise ValueError(f"{name} must be a one-dimensional non-empty prefix array")
            if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
                raise ValueError(f"{name} must start at zero and be strictly increasing")

        template_count = len(self.template_ray_offsets) - 1
        if self.template_ray_offsets[-1] != total_rays:
            raise ValueError("template_ray_offsets must cover every ray")
        validate_template_grid(
            [
                int(self.timestamp_offsets_us[begin:end].max())
                for begin, end in zip(self.template_ray_offsets[:-1], self.template_ray_offsets[1:])
            ],
            self.reference_frame_start_us,
            self.frame_period_us,
            self.frame_start_tolerance_us,
        )

        if (self.packet_ray_offsets is None) != (self.template_packet_offsets is None):
            raise ValueError(
                "packet_ray_offsets and template_packet_offsets must either both be present or both absent"
            )
        if self.packet_ray_offsets is not None and self.template_packet_offsets is not None:
            for name, offsets in (
                ("packet_ray_offsets", self.packet_ray_offsets),
                ("template_packet_offsets", self.template_packet_offsets),
            ):
                if offsets.dtype != np.int64:
                    raise ValueError(f"{name} must be int64")
                if offsets.ndim != 1 or len(offsets) < 2:
                    raise ValueError(f"{name} must be a one-dimensional non-empty prefix array")
                if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
                    raise ValueError(f"{name} must start at zero and be strictly increasing")

            if len(self.template_packet_offsets) != template_count + 1:
                raise ValueError("template_ray_offsets and template_packet_offsets disagree on template count")
            total_packets = len(self.packet_ray_offsets) - 1
            if self.packet_ray_offsets[-1] != total_rays:
                raise ValueError("packet_ray_offsets must cover every ray")
            if self.template_packet_offsets[-1] != total_packets:
                raise ValueError("template_packet_offsets must cover every packet")

            packet_boundary_rays = self.packet_ray_offsets[self.template_packet_offsets]
            if not np.array_equal(packet_boundary_rays, self.template_ray_offsets):
                raise ValueError("a decoded packet crosses a firing-pattern template boundary")

    @property
    def template_count(self) -> int:
        """Count the templates represented by the ray prefix offsets.

        Returns:
            Number of templates in the repeating sequence.
        """
        return len(self.template_ray_offsets) - 1

    @property
    def packet_count(self) -> int | None:
        """Count the packets represented by the optional packet boundaries.

        Returns:
            Total packet count across all templates, or ``None`` when layout is absent.
        """
        return None if self.packet_ray_offsets is None else len(self.packet_ray_offsets) - 1

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        """Return a version-1 NPZ payload, including packet layout when available.

        Returns:
            NPZ field names mapped to arrays, including int64 scalar metadata and
            optional packet boundaries. Existing ray and boundary arrays are shared,
            not copied; callers must not modify them through this mapping.
        """

        arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
            "reference_frame_start_us": np.asarray(self.reference_frame_start_us, dtype=np.int64),
            "frame_period_us": np.asarray(self.frame_period_us, dtype=np.int64),
            "frame_start_tolerance_us": np.asarray(self.frame_start_tolerance_us, dtype=np.int64),
            DIRECTIONS_ARRAY_NAME: self.directions_sensor,
            TIMESTAMP_OFFSETS_ARRAY_NAME: self.timestamp_offsets_us,
            TEMPLATE_RAY_OFFSETS_ARRAY_NAME: self.template_ray_offsets,
        }
        if self.packet_ray_offsets is not None and self.template_packet_offsets is not None:
            arrays[PACKET_RAY_OFFSETS_ARRAY_NAME] = self.packet_ray_offsets
            arrays[TEMPLATE_PACKET_OFFSETS_ARRAY_NAME] = self.template_packet_offsets
        return arrays


def grid_from_offsets(offsets: npt.NDArray[np.int64]) -> tuple[int, int] | None:
    """Recover ``(n_columns, n_rows)`` from the firing times, or ``None`` if there is no grid.

    A unit that fires a whole column at once leaves the offsets constant across each column,
    so the first run length is the column height and the rest follows. Reading it off the
    times rather than taking it from the caller means a pattern only claims a grid when the
    record actually shows one -- and the checks here are what a consumer relies on when it
    addresses a ray as ``column_id = i // n_rows, row_id = i % n_rows``. That addressing is a
    property of the layout, so the order the columns arrived in is not checked.

    A merged pattern is emitted in cluster order rather than sweep order, so it fails these
    checks on its own; the caller does not have to know how the pattern was built.

    Args:
        offsets: One-dimensional int64 firing offsets in microseconds, in ray order.

    Returns:
        ``(n_columns, n_rows)`` when equally sized groups of simultaneous rays
        describe a grid, or ``None`` when those checks do not establish one.
    """
    n = int(offsets.size)
    change = np.flatnonzero(np.diff(offsets) != 0) + 1
    if change.size == 0:
        return None  # every ray at one instant: no column structure to read
    n_rows = int(change[0])
    if n_rows <= 1 or n % n_rows != 0:
        return None  # one ray per instant is a flat list, not a grid
    n_columns = n // n_rows
    blocks = offsets.reshape(n_columns, n_rows)
    if not bool((blocks.max(axis=1) == blocks.min(axis=1)).all()):
        return None  # a column spans several instants
    return n_columns, n_rows
