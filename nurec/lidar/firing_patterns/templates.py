# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable firing-pattern templates and repeating phase selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np
import numpy.typing as npt

from nurec.lidar.firing_patterns.schema import (
    MAX_DURATION_MISMATCH,
    LidarFiringPatternTemplatesData,
    grid_from_offsets,
    validate_template_grid,
)


@dataclass(frozen=True)
class LidarFiringPatternTemplate:
    """One ordered lidar ray schedule with optional packet boundaries.

    Args:
        directions_sensor: Non-empty float32 array of shape ``(n_rays, 3)`` with
            finite unit directions in sensor coordinates, in firing order.
        timestamp_offsets_us: Non-negative int64 offsets of shape ``(n_rays,)``
            from physical frame start, in microseconds, aligned with the directions.
        packet_ray_offsets: Optional int64 prefix array delimiting source packets;
            starts at zero, strictly increases, and ends at ``n_rays``.

    Arrays must remain unchanged after construction because their digest and
    renderer tiling are cached; the frozen dataclass does not freeze array contents.

    Raises:
        ValueError: If arrays have invalid dtypes or shapes, directions are not
            finite unit vectors, offsets are negative, or packet boundaries are invalid.
    """

    directions_sensor: npt.NDArray[np.float32]
    timestamp_offsets_us: npt.NDArray[np.int64]
    packet_ray_offsets: npt.NDArray[np.int64] | None = None

    _ray_digest: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.directions_sensor.dtype != np.float32:
            raise ValueError("directions_sensor must be float32")
        if self.directions_sensor.ndim != 2 or self.directions_sensor.shape[1] != 3:
            raise ValueError(f"directions_sensor must be (N, 3), got {self.directions_sensor.shape}")
        ray_count = len(self.directions_sensor)
        if ray_count == 0:
            raise ValueError("firing-pattern template is empty")
        if self.timestamp_offsets_us.dtype != np.int64:
            raise ValueError("timestamp_offsets_us must be int64")
        if self.timestamp_offsets_us.shape != (ray_count,):
            raise ValueError(
                f"timestamp_offsets_us must be (N,) matching directions_sensor's N={ray_count}, "
                f"got {self.timestamp_offsets_us.shape}"
            )
        if np.any(self.timestamp_offsets_us < 0):
            raise ValueError("timestamp_offsets_us must be non-negative")
        if self.packet_ray_offsets is not None:
            if self.packet_ray_offsets.dtype != np.int64:
                raise ValueError("packet_ray_offsets must be int64")
            if self.packet_ray_offsets.ndim != 1 or len(self.packet_ray_offsets) < 2:
                raise ValueError("packet_ray_offsets must describe at least one packet")
            if self.packet_ray_offsets[0] != 0 or np.any(np.diff(self.packet_ray_offsets) <= 0):
                raise ValueError("packet_ray_offsets must start at zero and be strictly increasing")
            if self.packet_ray_offsets[-1] != ray_count:
                raise ValueError("packet_ray_offsets must cover every template ray")

        if not np.all(np.isfinite(self.directions_sensor)):
            raise ValueError("directions_sensor must contain only finite vectors")
        if not np.allclose(
            np.linalg.norm(self.directions_sensor, axis=1),
            1.0,
            rtol=0.0,
            atol=1e-4,
        ):
            raise ValueError("directions_sensor must contain unit vectors")

        digest = sha256(b"nurec-lidar-firing-pattern-v1\0")
        digest.update(np.asarray(self.directions_sensor.shape, dtype="<u8").tobytes())
        digest.update(np.asarray(self.directions_sensor, dtype="<f4", order="C").tobytes())
        digest.update(np.asarray(self.timestamp_offsets_us, dtype="<i8", order="C").tobytes())
        if self.packet_ray_offsets is None:
            digest.update(b"packet-layout-absent")
        else:
            digest.update(b"packet-layout-present\0")
            digest.update(np.asarray(self.packet_ray_offsets, dtype="<i8", order="C").tobytes())
        object.__setattr__(self, "_ray_digest", digest.digest())

    def pattern_id(self, frame_period_us: int) -> str:
        """Content ID covering the ray schedule and its physical repetition period.

        Args:
            frame_period_us: Positive integer repetition period in microseconds,
                representable as an unsigned 64-bit value; booleans are not accepted.

        Returns:
            A ``sha256:``-prefixed identifier covering directions, firing offsets,
            optional packet layout, and the supplied frame period.

        Raises:
            ValueError: If the period is not a positive integer.
            OverflowError: If the period cannot be encoded in eight unsigned bytes.
        """

        if isinstance(frame_period_us, bool) or not isinstance(frame_period_us, int) or frame_period_us <= 0:
            raise ValueError("frame_period_us must be a positive integer")
        # Hash the large ray arrays once; the period belongs to the enclosing template set.
        digest = sha256(self._ray_digest)
        digest.update(frame_period_us.to_bytes(8, byteorder="little"))
        return f"sha256:{digest.hexdigest()}"

    @property
    def n_rays(self) -> int:
        """Count the emitted rays in this template.

        Returns:
            The number of directions, including rays without returns.
        """
        return len(self.directions_sensor)

    @property
    def n_packets(self) -> int | None:
        """Count source packets when packet boundaries are available.

        Returns:
            The packet count, or ``None`` when the template has no packet layout.
        """
        return None if self.packet_ray_offsets is None else len(self.packet_ray_offsets) - 1

    @property
    def grid(self) -> tuple[int, int] | None:
        """``(n_columns, n_rows)`` when the unit fires on a grid, else ``None``.

        Derived from ``timestamp_offsets_us`` rather than stored beside them: a unit that
        fires a column at once leaves the offsets constant across it and advancing between
        columns, so the grid is a function of data the pattern already carries. Deriving it
        means a pattern written before grids existed gains one without being re-exported,
        and there is no second copy to fall out of step. Costs ~0.4 ms for 437k rays.

        Rays are then addressable as ``column_id = i // n_rows, row_id = i % n_rows``.

        Returns:
            ``(n_columns, n_rows)`` inferred from equal timestamp groups, or ``None``
            when those checks do not establish a grid.
        """
        return grid_from_offsets(self.timestamp_offsets_us)

    @property
    def span_us(self) -> int:
        """Offset of the final firing from the physical frame start.

        Returns:
            Largest firing offset in microseconds, including any idle lead-in.
        """

        return int(self.timestamp_offsets_us.max())

    def check_window(self, requested_us: int, frame_period_us: int) -> None:
        """Check that the supplied pose interval matches the template timing.

        Args:
            requested_us: Requested pose interval length in microseconds.
            frame_period_us: Physical repetition period in microseconds.

        Returns:
            ``None`` when the interval is at least the smaller of the firing
            span and frame period, and does not exceed the frame period by
            more than 5%.

        Raises:
            ValueError: If the interval falls outside the tolerated timing range.
        """

        # Span and period are inferred independently, so revolution jitter can make
        # the measured span slightly exceed the nominal period.
        minimum_us = min(self.span_us, frame_period_us)
        maximum_us = (1.0 + MAX_DURATION_MISMATCH) * frame_period_us
        if not minimum_us <= requested_us <= maximum_us:
            raise ValueError(
                f"the request duration is {requested_us} us, outside this template's allowed "
                f"range of {minimum_us:g} us (shorter of firing span and frame period) through "
                f"{maximum_us:g} us (frame period plus tolerance)"
            )

    def ray_timestamps_us(self, frame_start_us: int) -> npt.NDArray[np.int64]:
        """Offset this template's fixed firing schedule to one requested frame.

        Args:
            frame_start_us: Physical start of the requested frame, in microseconds.
                The start and resulting timestamps must fit in int64.

        Returns:
            An int64 array of shape ``(n_rays,)`` containing absolute firing times
            in microseconds, in the template's original ray order.
        """

        return frame_start_us + self.timestamp_offsets_us


@dataclass(frozen=True)
class LidarFiringPatternTemplateSet:
    """Templates in repeating phase order plus their physical frame grid.

    Args:
        templates: Non-empty tuple of templates in repeating phase order. All
            templates may have different ray counts and must fit within 105% of the period.
        reference_frame_start_us: Integer timestamp of phase zero, in microseconds.
        frame_period_us: Positive integer interval between frame starts, in microseconds.
        frame_start_tolerance_us: Non-negative integer grid-fit tolerance in
            microseconds, less than half a period. Records provenance and does
            not restrict timestamps accepted for phase selection.

    Raises:
        ValueError: If templates are empty, a template outlasts
            the allowed period, or timing scalars have invalid types or values.
    """

    templates: tuple[LidarFiringPatternTemplate, ...]
    reference_frame_start_us: int
    frame_period_us: int
    frame_start_tolerance_us: int

    def __post_init__(self) -> None:
        validate_template_grid(
            [template.span_us for template in self.templates],
            self.reference_frame_start_us,
            self.frame_period_us,
            self.frame_start_tolerance_us,
        )

    @classmethod
    def from_data(cls, data: LidarFiringPatternTemplatesData) -> "LidarFiringPatternTemplateSet":
        """Build runtime templates from the flattened artifact representation.

        Args:
            data: Validated arrays and frame-grid metadata for one lidar. Packet
                boundaries, when present, are converted to template-local ray offsets.

        Returns:
            A template set preserving phase order and frame-grid metadata. Contiguous
            ray slices may share storage with ``data`` and must remain unchanged.

        Raises:
            ValueError: If a runtime template or its frame period fails validation,
                including a firing span longer than 105% of the period.
        """
        templates = []
        for template_index in range(data.template_count):
            ray_begin = int(data.template_ray_offsets[template_index])
            ray_end = int(data.template_ray_offsets[template_index + 1])
            packet_ray_offsets = None
            if data.packet_ray_offsets is not None and data.template_packet_offsets is not None:
                packet_begin = int(data.template_packet_offsets[template_index])
                packet_end = int(data.template_packet_offsets[template_index + 1])
                packet_ray_offsets = np.ascontiguousarray(
                    data.packet_ray_offsets[packet_begin : packet_end + 1] - ray_begin
                )
            templates.append(
                LidarFiringPatternTemplate(
                    directions_sensor=np.ascontiguousarray(data.directions_sensor[ray_begin:ray_end]),
                    timestamp_offsets_us=np.ascontiguousarray(data.timestamp_offsets_us[ray_begin:ray_end]),
                    packet_ray_offsets=packet_ray_offsets,
                )
            )
        return cls(
            templates=tuple(templates),
            reference_frame_start_us=data.reference_frame_start_us,
            frame_period_us=data.frame_period_us,
            frame_start_tolerance_us=data.frame_start_tolerance_us,
        )

    def select_index(self, frame_start_us: int) -> int:
        """Select a phase statelessly from the nearest frame-grid point.

        The artifact's grid is fitted to recording timestamps, while callers may use a
        different simulation clock.  Its fit tolerance is therefore provenance for the
        estimator, not a request-admission constraint.

        Args:
            frame_start_us: Integer frame-start timestamp in microseconds. Timestamps
                before the reference are allowed; halfway ties round to the later grid point.

        Returns:
            Zero-based template index after wrapping the nearest grid ordinal
            through the repeating template sequence.

        Raises:
            ValueError: If the timestamp is not an integer, or is a boolean.
        """

        if isinstance(frame_start_us, bool) or not isinstance(frame_start_us, int):
            raise ValueError("frame_start_us must be an integer")
        if len(self.templates) == 1:
            return 0
        delta_us = frame_start_us - self.reference_frame_start_us
        quotient, remainder = divmod(delta_us, self.frame_period_us)
        ordinal = quotient if remainder * 2 < self.frame_period_us else quotient + 1
        return ordinal % len(self.templates)

    def select(self, frame_start_us: int) -> LidarFiringPatternTemplate:
        """Select the repeating template for a requested physical frame.

        Args:
            frame_start_us: Integer timestamp in microseconds used by ``select_index``.

        Returns:
            The existing template object at the selected phase; no copy is made.

        Raises:
            ValueError: If the timestamp is not an integer, or is a boolean.
        """
        return self.templates[self.select_index(frame_start_us)]

    def get_by_id(self, pattern_id: str) -> LidarFiringPatternTemplate:
        """Resolve an opaque content ID within this lidar's artifact pattern set.

        Args:
            pattern_id: Exact opaque ID produced with this set's frame period.

        Returns:
            The first template with the requested content ID; no copy is made.

        Raises:
            KeyError: If no template in this set has the requested ID.
            OverflowError: If the set's period cannot be encoded in eight unsigned bytes.
        """

        for template in self.templates:
            if template.pattern_id(self.frame_period_us) == pattern_id:
                return template
        raise KeyError(f"Unknown firing pattern id '{pattern_id}'")
