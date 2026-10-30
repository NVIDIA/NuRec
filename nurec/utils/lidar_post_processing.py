# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from typing import Optional

import torch

from nurec.utils.misc import unpack_optional
from nurec.utils.types import GaussiansRenderReturn


log = logging.getLogger(__name__)


def distance_based_filter(
    rendered: GaussiansRenderReturn,
    model_elements: torch.Tensor,
    valid_mask_pred: torch.Tensor,
    filter_threshold: float,
    n_vertical_bins: int,
    n_horizontal_bins: int,
) -> torch.Tensor:
    """Apply distance-based filtering to LiDAR rays.

    Args:
        rendered: Rendered LiDAR data with distance and optional raydrop signals
        model_elements: 2D indices for ray elements
        valid_mask_pred: Boolean mask indicating which rays returned a valid measurement
        filter_threshold: Threshold for distance discontinuity filtering
        n_vertical_bins: Number of vertical bins in the LiDAR range view
        n_horizontal_bins: Number of horizontal bins in the LiDAR range view

    Returns:
        Boolean mask indicating which rays passed the filter
    """
    device = model_elements.device
    dist_pred = unpack_optional(rendered.distance)

    rangeview_range_pred = torch.full((n_vertical_bins, n_horizontal_bins), torch.inf, device=device)
    rangeview_range_factor = torch.full((n_vertical_bins, n_horizontal_bins), torch.inf, device=device)

    rangeview_range_pred[model_elements[..., 0][valid_mask_pred], model_elements[..., 1][valid_mask_pred]] = dist_pred[
        valid_mask_pred
    ]
    rangeview_range_factor[model_elements[..., 0][valid_mask_pred], model_elements[..., 1][valid_mask_pred]] = 1.0 / (
        dist_pred[valid_mask_pred] + 1e-6
    )
    rangeview_range_factor[rangeview_range_factor == torch.inf] = 0

    range_diff_left = torch.abs(rangeview_range_pred[:, 1:-1] - rangeview_range_pred[:, :-2])
    range_diff_right = torch.abs(rangeview_range_pred[:, 1:-1] - rangeview_range_pred[:, 2:])
    range_diff = torch.min(range_diff_left, range_diff_right)
    range_diff[range_diff == torch.inf] = 0
    range_diff *= rangeview_range_factor[:, 1:-1]
    range_filter_mask = range_diff > filter_threshold
    range_filter_mask_full = torch.full((n_vertical_bins, n_horizontal_bins), 0.0, device=device)
    range_filter_mask_full[:, 1:-1] = range_filter_mask

    filter_mask = range_filter_mask_full.reshape(-1) == 1.0

    return filter_mask


# Row-labelling thresholds (see `_auto_elevation_rows`).
_ROW_FLOOR_DEG = 1e-4  # gap floor for the ratio test; below any real ring spacing, above fp32 noise
_ROW_MIN_JUMP_RATIO = 5.0  # the between/within gap clusters must differ by at least this factor to split
_SINGLE_ROW_MAX_SPREAD_DEG = 0.5  # with no ring split, an elevation spread <= this is treated as one row


def _auto_elevation_rows(el_deg: torch.Tensor) -> torch.Tensor:
    """Assign each ray to a scanline/ring row from its ELEVATION alone (for DISCRETE-ring lidar).

    Rays are sorted by elevation; the sorted-elevation gaps split into a WITHIN-ring cluster (small)
    and a BETWEEN-ring cluster (large). We find the boundary between the two as the largest
    multiplicative JUMP in the sorted gaps and start a new row at every gap above it. This labels
    rings DIRECTLY rather than quantizing elevation at a fixed resolution, so it handles NON-UNIFORM
    beam spacing (e.g. dense near the horizon, sparse at the edges) natively: every between-ring gap,
    whatever its size, lands above the boundary and becomes its own split. The jump is scale-free, so
    it works for any number of azimuths per ring (the old percentile heuristic only fired below ~10).

    FAIL-LOUD: if the gaps show no clear split -- the two clusters are not separated by at least
    `_ROW_MIN_JUMP_RATIO` (e.g. ET30X, whose scan interleaves azimuth and elevation so a ring's
    elevation drifts by ~the ring spacing, or a genuinely continuous elevation sweep) -- ring
    assignment is ambiguous, so we RAISE rather than silently mis-binning (which would merge adjacent
    rings and reintroduce the elevation-gradient false positives this filter exists to avoid). Pass
    `elevation_resolution_deg` explicitly (~FOV_el / n_beams) for such sensors.

    Returns a full-length Long tensor of row indices (0-based, increasing with elevation).
    """
    n = el_deg.numel()
    device = el_deg.device
    zeros = torch.zeros(n, dtype=torch.long, device=device)
    if n <= 2:
        return zeros

    order = torch.argsort(el_deg)
    el_s = el_deg[order]
    d = el_s[1:] - el_s[:-1]  # sorted-elevation gaps, >= 0, length n-1
    spread = float(el_s[-1] - el_s[0])

    # Largest multiplicative jump between consecutive sorted gaps = the split between the within-ring
    # and between-ring clusters. The floor keeps ratios finite across (near-)zero intra-ring gaps.
    ds = torch.sort(d).values
    lo = ds[:-1].clamp_min(_ROW_FLOOR_DEG)
    ratios = ds[1:] / lo
    jump_ratio, j = torch.max(ratios, dim=0)

    if float(jump_ratio) < _ROW_MIN_JUMP_RATIO:
        # No bimodal split: one ring (tight elevation spread) or a continuous sweep (wide spread).
        if spread <= _SINGLE_ROW_MAX_SPREAD_DEG:
            return zeros
        raise ValueError(
            "angle_gradient_filter: elevation shows no discrete ring gaps (largest gap jump "
            f"{float(jump_ratio):.2f}x over a {spread:.3g} deg spread); cannot auto-assign scanline "
            "rows. Set floating_point_filter.elevation_resolution_deg explicitly (~FOV_el / n_beams) "
            "for this sensor (e.g. ET30X, whose rings interleave in elevation)."
        )

    # Threshold sits (geometrically) between the top of the within-ring cluster and the bottom of the
    # between-ring cluster; every gap above it is a ring boundary.
    thr = float((ds[j].clamp_min(_ROW_FLOOR_DEG) * ds[j + 1]) ** 0.5)
    is_boundary = d > thr
    row_sorted = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(is_boundary.long(), dim=0)])
    row = torch.empty(n, dtype=torch.long, device=device)
    row[order] = row_sorted
    return row


def angle_gradient_filter(
    rendered: GaussiansRenderReturn,
    ray_direction_sensor: torch.Tensor,
    valid_mask_pred: torch.Tensor,
    filter_threshold: float,
    elevation_resolution_deg: Optional[float] = None,
    azimuth_max_gap_deg: Optional[float] = None,
) -> torch.Tensor:
    """Sensor-agnostic distance-gradient filter using ONLY ray angles (no range-view).

    Rays are grouped into ELEVATION rows (each row ≈ a scanline/ring): when `elevation_resolution_deg`
    is given, elevation is quantized at that spacing; when None, rows are auto-assigned by labelling
    rings directly from the elevation-gap structure (`_auto_elevation_rows`), which handles
    non-uniform beam spacing and RAISES if the rings are not separable in elevation (rather than
    silently mis-binning). Within each row the rays are SORTED by azimuth, and the relative range
    gradient is computed only along the AZIMUTH axis (the geometrically safe axis: on the ground,
    azimuth is an iso-range direction, so grazing surfaces are not falsely flagged — unlike the
    elevation axis). A ray's azimuth neighbour is only used if the azimuth gap to it is <=
    `azimuth_max_gap_deg` (auto = 3x the native azimuth spacing), which skips scan holes. Same
    min(left,right) / relative-`filter_threshold` semantics as `distance_based_filter`.

    NOTE: the azimuth-only axis is deliberate but not universal. On a locally planar surface azimuth
    is the iso-range direction (so it cannot manufacture a false gradient), and empirically the
    azimuth axis Pareto-dominates the elevation axis on ET30X (higher precision at every recall). That
    empirical result is one sensor/scene, so treat "azimuth is the safe axis" as validated for ET30X,
    not settled in general. If elevation filtering is added later, the extensible shape is NOT a new
    `neighbor_source` value but promoting the axis to config (a per-axis (resolution, max_gap) pair or
    an explicit `axes` list, azimuth-only as the default).

    NOTE: auto row assignment needs elevation rows separable in elevation. It fits DISCRETE-ring
    spinning lidars; for sensors whose scan interleaves azimuth and elevation within a firing step
    (so elevation rows overlap — e.g. ET30X), pass `elevation_resolution_deg` explicitly from the
    datasheet (≈ FOV_el / n_beams), otherwise auto assignment raises.

    NOTE: the ±180° azimuth seam is never wrapped, so the two rays either side of it are row
    boundaries (see below) and are never flagged; a flying pixel exactly at the seam is missed. This
    is negligible (two rays per row) and consistent with the row-boundary exclusion.

    Returns a FULL-LENGTH per-ray boolean mask (over all input rays, not only the valid ones):
    True = flagged flying point.
    """
    device = ray_direction_sensor.device
    # fp32 for the angle math so row binning is identical under fp16 autocast (gRPC render) and the
    # fp32 validation path; a small angular error near a row edge would otherwise flip the row index.
    d = ray_direction_sensor.float()
    norm = d.norm(dim=1).clamp_min(1e-9)
    el = torch.rad2deg(torch.asin((d[:, 2] / norm).clamp(-1.0, 1.0)))
    az = torch.rad2deg(torch.atan2(d[:, 1], d[:, 0]))

    if elevation_resolution_deg is not None:
        row = torch.round((el - el.min()) / max(elevation_resolution_deg, 1e-6)).long()
    else:
        row = _auto_elevation_rows(el)

    # Lexsort by (row, azimuth): within-row azimuth-ascending order.
    order = torch.argsort(az, stable=True)
    order = order[torch.argsort(row[order], stable=True)]
    dist = unpack_optional(rendered.distance).reshape(-1)
    inf = torch.tensor(float("inf"), device=device)
    R = torch.where(valid_mask_pred, dist, inf)
    R_s, az_s, row_s = R[order], az[order], row[order]
    n = R_s.shape[0]

    if azimuth_max_gap_deg is not None:
        az_gap = azimuth_max_gap_deg
    else:
        # native azimuth spacing = median consecutive same-row azimuth gap; guard = 3x.
        same = row_s[1:] == row_s[:-1]
        g = (az_s[1:] - az_s[:-1])[same]
        g = g[g > 1e-6]
        az_gap = 3.0 * float(g.median()) if g.numel() > 0 else 1.0

    same_l = row_s[1:] == row_s[:-1]
    valid_l = same_l & ((az_s[1:] - az_s[:-1]) <= az_gap)  # pair (i-1, i) valid, indexed 0..n-2
    dR_left = torch.full((n,), float("inf"), device=device)
    dR_right = torch.full((n,), float("inf"), device=device)
    dR_left[1:] = torch.where(valid_l, (R_s[1:] - R_s[:-1]).abs(), inf)
    dR_right[:-1] = torch.where(valid_l, (R_s[:-1] - R_s[1:]).abs(), inf)
    grad = torch.minimum(dR_left, dR_right)
    grad[~torch.isfinite(grad)] = 0.0
    relative = grad / R_s.clamp_min(1e-6)

    # A row-boundary ray (first/last in its azimuth row) has only one in-row neighbour, so the
    # min(left, right) test degenerates to a single-neighbour gradient and could falsely flag a
    # genuine surface edge. Exclude these, mirroring `distance_based_filter`, which never flags the
    # first/last range-view column (`range_filter_mask_full[:, 1:-1]`). Interior rays adjacent to a
    # scan hole keep the single-neighbour behaviour (the range-view path does too).
    is_row_start = torch.ones(n, dtype=torch.bool, device=device)
    is_row_start[1:] = row_s[1:] != row_s[:-1]
    is_row_end = torch.ones(n, dtype=torch.bool, device=device)
    is_row_end[:-1] = row_s[1:] != row_s[:-1]
    row_boundary = is_row_start | is_row_end

    flag_s = torch.isfinite(R_s) & ~row_boundary & (relative > filter_threshold)

    flag = torch.empty(n, dtype=torch.bool, device=device)
    flag[order] = flag_s
    return flag & valid_mask_pred
