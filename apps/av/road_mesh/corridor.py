# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ego-corridor masks for the 2.5D ground mesh.

The corridor is the rig path buffered by ``half_width_m`` in BEV, optionally banded in Z by
``z_band_m``. Callers use it to mark which triangles carry road under the ego vehicle:

* :func:`corridor_point_mask` / :func:`corridor_triangle_mask` -- membership tests.
* :func:`select_components_covering_corridor` -- component retention rule that keeps every
  face-connected component carrying corridor area, instead of only the largest.

These only classify existing geometry; nothing here creates vertices or triangles.
"""

from __future__ import annotations

import logging

from typing import Optional, Tuple

import numpy as np

from numpy.typing import NDArray
from scipy.spatial import cKDTree


logger = logging.getLogger(__name__)

DEFAULT_CORRIDOR_HALF_WIDTH_M = 2.5
"""Default corridor half-width in metres: lane half-width plus a lateral pose margin.

Over-claims where the road is narrower than 5 m; override per clip when the width is known.
"""

DEFAULT_CORRIDOR_MAX_EDGE_LENGTH_M = 6.0
"""Default effective max BEV edge length for a protected corridor triangle.

Above the 5 m corridor width, so a protected face may still span the corridor laterally, while
bounding how far one exemption reaches along the path (12.36 m unbounded on a 42%-grade clip).
Coverage-neutral across 469 swept clips, and it never increases boundary-loop count; on the two
worst perimeter-fragmentation clips it cuts them 58 -> 3 and 34 -> 2. Coverage-neutrality relies on
component retention: where a sparse band is wider than this, the exemption protects nothing and
retention is what keeps the road.
"""

RECOMMENDED_CORRIDOR_Z_BAND_M = 2.0
"""Suggested value for ``z_band_m``, which is off by default.

Above measured rig-to-road elevation offsets (p90 <= 0.26 m) and below realistic deck separation
(7.54 m), so it keeps the driven surface and excludes the deck above.
"""

_TRAJECTORY_RESAMPLE_STEP_M = 0.25
"""Polyline resample step in metres. Nearest-sample distance overestimates the true perpendicular
distance by at most step/2, so this stays well below the corridor half-width."""


def resample_polyline(
    polyline: NDArray[np.floating],
    step_m: float = _TRAJECTORY_RESAMPLE_STEP_M,
) -> NDArray[np.float64]:
    """Resample an open 2D or 3D polyline to approximately uniform ``step_m`` spacing in XY.

    Parameterised by horizontal arclength, so a 3D input has Z interpolated along that parameter.
    Consecutive duplicate vertices (a stationary ego) are collapsed first.
    """
    points = np.asarray(polyline, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(f"Expected polyline with shape (N, 2) or (N, 3), got {points.shape}")
    if len(points) == 0:
        raise ValueError("Cannot resample an empty polyline")
    if not np.isfinite(points).all():
        raise ValueError("Polyline contains non-finite coordinates")
    if not np.isfinite(step_m) or step_m <= 0.0:
        raise ValueError(f"Resample step must be positive and finite, got {step_m}")

    if len(points) > 1:
        keep = np.concatenate(([True], np.any(np.diff(points[:, :2], axis=0) != 0.0, axis=1)))
        points = points[keep]
    if len(points) < 2:
        return points.copy()

    arclength = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))))
    total = float(arclength[-1])
    n_samples = max(int(np.ceil(total / step_m)) + 1, 2)
    sample_s = np.linspace(0.0, total, n_samples)
    return np.column_stack([np.interp(sample_s, arclength, points[:, c]) for c in range(points.shape[1])])


def nearest_trajectory_sample(
    query_xy: NDArray[np.floating],
    trajectory: NDArray[np.floating],
    resample_step_m: float = _TRAJECTORY_RESAMPLE_STEP_M,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """BEV distance from each query point to the trajectory, plus the nearest resampled sample.

    Returning the sample lets callers compare elevation against the trajectory at that point.
    """
    query = np.asarray(query_xy, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError(f"Expected query_xy with shape (N, 2), got {query.shape}")
    dense = resample_polyline(trajectory, step_m=resample_step_m)
    if len(query) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, dense.shape[1]), dtype=np.float64)
    distances, indices = cKDTree(dense[:, :2]).query(query, k=1)
    return np.asarray(distances, dtype=np.float64), dense[np.asarray(indices)]


def corridor_point_mask(
    points: NDArray[np.floating],
    trajectory: NDArray[np.floating],
    half_width_m: float = DEFAULT_CORRIDOR_HALF_WIDTH_M,
    z_band_m: Optional[float] = None,
) -> NDArray[np.bool_]:
    """Mask of points within ``half_width_m`` of the trajectory in BEV.

    With ``z_band_m``, a point must also lie within that vertical distance of the trajectory
    elevation at its nearest sample. ``points`` and ``trajectory`` may each be 2D or 3D; Z gating
    needs 3D on both sides and is skipped with a warning otherwise.
    """
    if not np.isfinite(half_width_m) or half_width_m <= 0.0:
        raise ValueError(f"Corridor half-width must be positive and finite, got {half_width_m}")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"Expected points with shape (N, 2) or (N, 3), got {pts.shape}")
    if z_band_m is not None and (not np.isfinite(z_band_m) or z_band_m <= 0.0):
        raise ValueError(f"Corridor z-band must be positive and finite, got {z_band_m}")

    distances, nearest = nearest_trajectory_sample(pts[:, :2], trajectory)
    inside = distances <= half_width_m
    if z_band_m is None:
        return inside

    if pts.shape[1] != 3 or nearest.shape[1] != 3:
        logger.warning(
            f"Corridor z-band of {z_band_m:g} m requested but elevation is missing "
            f"(points {pts.shape}, trajectory samples {nearest.shape}); using BEV-only masking. "
            "Supply (N, 3) points and an (M, 3) trajectory to gate vertically."
        )
        return inside
    return inside & (np.abs(pts[:, 2] - nearest[:, 2]) <= z_band_m)


def corridor_triangle_mask(
    vertices: NDArray[np.floating],
    triangles: NDArray[np.integer],
    trajectory: NDArray[np.floating],
    half_width_m: float = DEFAULT_CORRIDOR_HALF_WIDTH_M,
    z_band_m: Optional[float] = None,
    max_edge_length_m: Optional[float] = DEFAULT_CORRIDOR_MAX_EDGE_LENGTH_M,
    elevation_axis: int = 2,
) -> NDArray[np.bool_]:
    """Mask of triangles whose centroid lies inside the corridor.

    Centroid rather than any-vertex membership, so a triangle qualifies when its area is largely in
    the corridor rather than when one distant vertex clips it.

    ``max_edge_length_m`` additionally requires the triangle's longest BEV edge to be shorter than
    the cap, so it is the effective maximum edge length inside the corridor. ``None`` means no bound.
    It only has effect above the caller's own long-edge threshold: below that, every face it would
    protect is already kept unprotected. Above it, a gap wider than the cap is not protected at all
    and the mesh still splits -- component retention, which ignores the cap, is what preserves
    coverage there.
    """
    triangles = np.asarray(triangles)
    if len(triangles) == 0:
        return np.zeros(0, dtype=bool)
    vertices = np.asarray(vertices, dtype=np.float64)
    # Reorder to (lateral, lateral, elevation); identity when elevation_axis is 2.
    lateral_axes = [axis for axis in range(3) if axis != elevation_axis]
    centroids = vertices[triangles].mean(axis=1)[:, lateral_axes + [elevation_axis]]
    mask = corridor_point_mask(centroids, trajectory, half_width_m=half_width_m, z_band_m=z_band_m)
    if max_edge_length_m is not None:
        corner_xy = vertices[triangles][:, :, lateral_axes]
        longest_bev_edge = np.maximum.reduce(
            [np.linalg.norm(corner_xy[:, (i + 1) % 3] - corner_xy[:, i], axis=1) for i in range(3)]
        )
        mask &= longest_bev_edge < float(max_edge_length_m)
    return mask


def select_components_covering_corridor(
    vertices: NDArray[np.floating],
    triangles: NDArray[np.integer],
    trajectory: NDArray[np.floating],
    half_width_m: float = DEFAULT_CORRIDOR_HALF_WIDTH_M,
    z_band_m: Optional[float] = None,
    elevation_axis: int = 2,
) -> NDArray[np.bool_]:
    """Per-triangle mask keeping every face-connected component that carries corridor area.

    Falls back to the largest component when none intersects the corridor.

    Membership here is deliberately independent of any edge-length cap: the cap bounds how far one
    *protected* face may span, whereas a component is kept because it carries road under the ego
    path. Applying the cap here would delete a sparse-but-real stretch outright, which is the failure
    this function exists to prevent.
    """
    triangles = np.asarray(triangles)
    if len(triangles) == 0:
        return np.zeros(0, dtype=bool)

    # Imported here rather than at module scope: ground_mesh is the caller of this module.
    from apps.av.road_mesh.ground_mesh import _triangle_components

    labels = _triangle_components(triangles.astype(np.int32, copy=False))
    in_corridor = corridor_triangle_mask(
        vertices,
        triangles,
        trajectory,
        half_width_m=half_width_m,
        z_band_m=z_band_m,
        max_edge_length_m=None,
        elevation_axis=elevation_axis,
    )

    corridor_labels = np.unique(labels[in_corridor])
    if len(corridor_labels) == 0:
        counts = np.bincount(labels)
        keep = labels == int(np.argmax(counts))
        logger.warning(
            "select_components_covering_corridor: no component intersects the corridor; "
            f"falling back to the largest ({int(keep.sum())}/{len(triangles)} triangles)"
        )
        return keep

    keep = np.isin(labels, corridor_labels)
    logger.info(
        f"select_components_covering_corridor: kept {len(corridor_labels)}/{labels.max() + 1} components "
        f"({int(keep.sum())}/{len(triangles)} triangles) carrying corridor coverage"
    )
    return keep
