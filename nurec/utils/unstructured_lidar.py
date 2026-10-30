# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pieces of the model-free unstructured LiDAR render input.

Training resolves a sensor's field of view from config or its model and builds the tiling
in the dataloader; serving has no datasource, so it derives the field of view from a
firing pattern's own directions and builds the tiling once per unit. The field of view
type, the rule that turns raw bounds into one, and the directions-to-tiling step are the
same either way and live here, so the azimuth seam is handled in one place rather than
two.
"""

from __future__ import annotations

import math

from typing import NamedTuple

import torch

from gsplat.cuda._lidar import FOV, SpinningDirection, UnstructuredModelParameters
from gsplat.lidar_context import UnstructuredLidarFrameContext

from nurec.utils.batch import UnstructuredLidarTilingData


# A span this wide is a full spin: atan2 azimuth is in (-pi, pi], so measured bounds
# would clip the wrap-around rather than describe it. Public so serving can ask the same
# question of a pattern's measured bounds before handing them over.
PERIODIC_AZ_SPAN_RAD = math.radians(350.0)


class UnstructuredFov(NamedTuple):
    """Resolved per-sensor field of view for the unstructured LiDAR path (radians).

    ``az_lo/az_hi`` and ``el_lo/el_hi`` bound the azimuth and elevation grids
    (az-first, matching the ``n_bins_azimuth, n_bins_elevation`` ordering used by
    gsplat's ``build_tiling`` / ``UnstructuredTiling`` / ``FiringLatticeEstimator``).
    ``periodic_az`` is True for a full-circle (spinning) sensor (then ``az_lo/az_hi``
    span the full ``[-pi, pi]``); False for a limited-FOV sensor, where
    ``az_lo/az_hi`` are its real (direction-aware) bounds and must lie within
    ``(-pi, pi]`` without crossing the +/-pi seam.

    CONSUMERS:
      * Firing lattice (get_firing_lattice -> FiringLatticeEstimator): uses BOTH
        az and el bounds, so its grid matches the sensor FOV exactly.
      * Render tiling (gsplat UnstructuredLidarFrameContext): consumes ``az_lo/az_hi``,
        ``el_lo/el_hi`` and ``periodic_az``, so the tiling grid also matches the
        sensor FOV exactly (empty-but-in-FOV azimuth/elevation bands are
        represented rather than clipped to the observed per-frame extent).
    """

    az_lo: float
    az_hi: float
    el_lo: float
    el_hi: float
    periodic_az: bool


def fov_from_bounds(az_lo: float, az_hi: float, el_lo: float, el_hi: float, *, source: str) -> UnstructuredFov:
    """Raw azimuth/elevation bounds (radians) -> :class:`UnstructuredFov`.

    A near-full azimuth span is periodic, and its bounds become the whole circle. A
    narrower span keeps its real ``[lo, hi]`` so the lattice and tiling do not waste bins
    on unobserved azimuth -- but the binning is linear with a hard clamp and NO wrap, so
    such a span must lie inside atan2's ``(-pi, pi]``. One that straddles the seam (a rear
    arc, say) would alias its wrapped half into a single column, so it raises instead.
    ``source`` names what produced the bounds, for that message.
    """
    if el_lo > el_hi:
        # Elevation has no seam, so a descending pair is just an ordering.
        el_lo, el_hi = el_hi, el_lo
    if az_lo > az_hi:
        # Azimuth does have a seam, so this is not an ordering: [170, -170] deg means the 20 deg
        # arc through 180, and swapping would silently turn it into the 340 deg arc through 0.
        raise ValueError(
            f"{source} gives azimuth [{math.degrees(az_lo):.1f}, {math.degrees(az_hi):.1f}] deg, "
            f"which runs backwards and so crosses the +/-180 deg seam. The unstructured azimuth "
            f"binning is linear with no wrap and cannot represent that. Express the FOV within "
            f"[-180, 180] deg, or use a full-360 (periodic) sensor."
        )

    if (az_hi - az_lo) >= PERIODIC_AZ_SPAN_RAD:
        return UnstructuredFov(az_lo=-math.pi, az_hi=math.pi, el_lo=el_lo, el_hi=el_hi, periodic_az=True)

    if az_lo < -math.pi - 1e-6 or az_hi > math.pi + 1e-6:
        raise ValueError(
            f"{source} spans [{math.degrees(az_lo):.1f}, {math.degrees(az_hi):.1f}] deg in azimuth, which "
            f"leaves the [-180, 180] deg range. A limited (non-periodic) FOV that crosses the +/-180 deg "
            f"seam is not supported by the unstructured azimuth binning (no wrap). Express the FOV within "
            f"[-180, 180] deg, or use a full-360 (periodic) sensor."
        )
    return UnstructuredFov(az_lo=az_lo, az_hi=az_hi, el_lo=el_lo, el_hi=el_hi, periodic_az=False)


def unstructured_model_from_fov(fov: UnstructuredFov):
    """Build a gsplat ``UnstructuredModelParameters`` from an :class:`UnstructuredFov`.

    The context reconstructs the tiling bounds as ``[min(start,end), max(start,end)]`` per
    axis and infers ``periodic_azimuth`` from ``isclose(fov_horiz.span, 2*pi)``, so we only
    need FOVs whose ``[start, end]`` bracket the desired ``[lo, hi]`` and whose azimuth span
    is exactly ``2*pi`` when periodic. Directions follow the COUNTER_CLOCKWISE (azimuth) /
    CLOCKWISE (elevation) conventions of ``UnstructuredModelParameters.from_directions``.
    """
    if fov.periodic_az:
        az_start, az_span = -math.pi, 2.0 * math.pi
    else:
        az_start, az_span = fov.az_lo, (fov.az_hi - fov.az_lo)
    return UnstructuredModelParameters(
        fov_horiz_rad=FOV(
            start=az_start,
            span=az_span,
            direction=SpinningDirection.COUNTER_CLOCKWISE,
        ),
        fov_vert_rad=FOV(
            start=fov.el_hi,
            span=(fov.el_hi - fov.el_lo),
            direction=SpinningDirection.CLOCKWISE,
        ),
        fov_eps_rad=4.0 * float(torch.finfo(torch.float32).eps),
    )


def build_tiling(
    directions_sensor: torch.Tensor,
    timestamps_us: torch.Tensor,
    fov: UnstructuredFov,
    *,
    frame_start_us: int | None = None,
    frame_end_us: int | None = None,
) -> UnstructuredLidarTilingData:
    """Tiling tensors for one frame's rays, as the unstructured render consumes them.

    Builds a gsplat ``UnstructuredLidarFrameContext`` (analytic projection + occupancy
    tiling + auto-resolution) and keeps only its tensors: the context is rebuilt at zero
    compute by ``from_tensors`` on the render side.
    """
    context = UnstructuredLidarFrameContext(
        directions_sensor,
        timestamps_us=timestamps_us,
        model=unstructured_model_from_fov(fov),
        frame_start_us=frame_start_us,
        frame_end_us=frame_end_us,
    )
    return UnstructuredLidarTilingData(**context.to_tensors())
