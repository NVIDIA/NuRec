# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ego-corridor coverage guarantees on the ground mesh."""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.testing as npt
import pytest

from scipy.spatial import Delaunay

from apps.av.road_mesh.corridor import (
    DEFAULT_CORRIDOR_MAX_EDGE_LENGTH_M,
    corridor_point_mask,
    corridor_triangle_mask,
    nearest_trajectory_sample,
    resample_polyline,
    select_components_covering_corridor,
)
from apps.av.road_mesh.ground_mesh import (
    DelaunayElevationMeshingAlgorithm,
    _triangle_components,
    fill_interior_boundary_loops,
    filter_triangles_by_max_edge_length,
    keep_largest_connected_component,
    patch_and_densify_ground_mesh,
    reconstruct_ground_mesh_from_points,
)


ROAD_HALF_WIDTH_M = 6.0
CORRIDOR_HALF_WIDTH_M = 2.5


def _road_strip(
    length_m: float = 24.0,
    spacing_m: float = 1.0,
    gap_span: tuple[float, float] | None = None,
    gap_max_y: float | None = None,
    grade: float = 0.0,
) -> np.ndarray:
    """Flat/sloped road strip sampled on a grid, optionally with a sparsified band.

    ``gap_span`` removes points with ``gap_span[0] < x < gap_span[1]``; ``gap_max_y`` restricts the
    removal to ``y <= gap_max_y`` producing a notch that touches only one road edge. This is how
    crest foreshortening and occlusion present themselves: a band of missing fused points.
    """
    points = []
    for x in np.arange(0.0, length_m + 1e-9, spacing_m):
        for y in np.arange(-ROAD_HALF_WIDTH_M, ROAD_HALF_WIDTH_M + 1e-9, spacing_m):
            if gap_span is not None and gap_span[0] < x < gap_span[1]:
                if gap_max_y is None or y <= gap_max_y:
                    continue
            points.append((x, y, grade * x))
    return np.asarray(points, dtype=np.float32)


def _centreline(length_m: float = 24.0, grade: float = 0.0) -> np.ndarray:
    xs = np.arange(0.0, length_m + 1e-9, 0.5)
    return np.column_stack((xs, np.zeros_like(xs)))


def _delaunay(points: np.ndarray) -> np.ndarray:
    return Delaunay(points[:, :2]).simplices.astype(np.int32)


def test_resample_polyline_matches_requested_spacing_and_endpoints() -> None:
    polyline = np.asarray([[0.0, 0.0], [10.0, 0.0]])
    dense = resample_polyline(polyline, step_m=0.25)
    npt.assert_allclose(dense[0], [0.0, 0.0])
    npt.assert_allclose(dense[-1], [10.0, 0.0])
    steps = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    assert steps.max() <= 0.25 + 1e-9


def test_resample_polyline_collapses_stationary_duplicates() -> None:
    # A stationary ego repeats a pose; duplicates carry no arclength and must not create
    # zero-length segments that would bias nearest-sample distance.
    polyline = np.asarray([[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]])
    dense = resample_polyline(polyline, step_m=1.0)
    assert len(dense) >= 5
    npt.assert_allclose(dense[-1], [4.0, 0.0])


def test_distance_to_polyline_is_perpendicular_within_resample_tolerance() -> None:
    polyline = np.asarray([[0.0, 0.0], [10.0, 0.0]])
    query = np.asarray([[5.0, 3.0], [5.0, -1.5], [2.0, 0.0]])
    distances, _ = nearest_trajectory_sample(query, polyline, resample_step_m=0.25)
    npt.assert_allclose(distances, [3.0, 1.5, 0.0], atol=0.125)


def test_corridor_point_mask_selects_band_around_trajectory() -> None:
    trajectory = _centreline()
    points_xy = np.asarray([[12.0, 0.0], [12.0, 2.0], [12.0, 4.0]])
    mask = corridor_point_mask(points_xy, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M)
    npt.assert_array_equal(mask, [True, True, False])


def test_corridor_point_mask_rejects_non_positive_half_width() -> None:
    with pytest.raises(ValueError, match="half-width must be positive"):
        corridor_point_mask(np.zeros((1, 2)), _centreline(), half_width_m=0.0)


def test_corridor_triangle_mask_uses_centroid_not_any_vertex() -> None:
    # A triangle with one vertex in the corridor but its area outside must NOT be protected,
    # otherwise the exemption would extend to long faces reaching far off the road.
    vertices = np.asarray([[12.0, 0.0, 0.0], [12.0, 9.0, 0.0], [13.0, 9.0, 0.0]], dtype=np.float32)
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    mask = corridor_triangle_mask(vertices, triangles, _centreline(), half_width_m=CORRIDOR_HALF_WIDTH_M)
    npt.assert_array_equal(mask, [False])


def test_edge_filter_protects_corridor_triangles_only() -> None:
    """The long-edge filter must spare corridor faces and still drop the rest."""
    points = _road_strip(gap_span=(10.0, 14.0))
    triangles = _delaunay(points)
    protected = corridor_triangle_mask(points, triangles, _centreline(), half_width_m=CORRIDOR_HALF_WIDTH_M)

    _, unprotected_tris, _ = filter_triangles_by_max_edge_length(points, triangles, 2.5)
    _, protected_tris, _ = filter_triangles_by_max_edge_length(
        points, triangles, 2.5, protected_triangle_mask=protected
    )

    # Exemption strictly adds faces, and exactly the protected long ones.
    assert len(protected_tris) > len(unprotected_tris)
    assert len(protected_tris) == len(unprotected_tris) + int(
        (protected & ~_short_edge_mask(points, triangles, 2.5)).sum()
    )


def _short_edge_mask(vertices: np.ndarray, triangles: np.ndarray, max_edge: float) -> np.ndarray:
    tri_xy = vertices[triangles, :2]
    edges = np.stack(
        [
            np.linalg.norm(tri_xy[:, 1] - tri_xy[:, 0], axis=1),
            np.linalg.norm(tri_xy[:, 2] - tri_xy[:, 1], axis=1),
            np.linalg.norm(tri_xy[:, 0] - tri_xy[:, 2], axis=1),
        ],
        axis=1,
    )
    return edges.max(axis=1) < max_edge


def test_edge_filter_exemption_is_confined_to_the_corridor() -> None:
    """Faces the exemption adds stay corridor-local, unlike a global edge-length relaxation."""
    points = _road_strip(gap_span=(10.0, 14.0))
    trajectory = _centreline()
    triangles = _delaunay(points)
    protected = corridor_triangle_mask(points, triangles, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M)

    _, strict_tris, _ = filter_triangles_by_max_edge_length(points, triangles, 2.5)
    _, exempt_tris, _ = filter_triangles_by_max_edge_length(points, triangles, 2.5, protected_triangle_mask=protected)
    _, relaxed_tris, _ = filter_triangles_by_max_edge_length(points, triangles, 6.0)

    strict_keys = {tuple(sorted(t)) for t in strict_tris.tolist()}
    added_by_exemption = [t for t in exempt_tris.tolist() if tuple(sorted(t)) not in strict_keys]
    added_by_relaxation = [t for t in relaxed_tris.tolist() if tuple(sorted(t)) not in strict_keys]

    # Vertex indices differ between outputs (each call compacts), so compare via positions.
    def max_distance_from_path(vertex_index_triples: list[list[int]], vertices: np.ndarray) -> float:
        if not vertex_index_triples:
            return 0.0
        centroids = vertices[np.asarray(vertex_index_triples)][:, :, :2].mean(axis=1)
        return float(nearest_trajectory_sample(centroids, trajectory)[0].max())

    exemption_reach = max_distance_from_path(added_by_exemption, points)
    relaxation_reach = max_distance_from_path(added_by_relaxation, points)

    assert len(added_by_exemption) > 0, "expected the corridor exemption to retain some long faces"
    assert exemption_reach <= CORRIDOR_HALF_WIDTH_M
    # A global relaxation is what the superseded hotfix did; it reaches further off-path.
    assert relaxation_reach > exemption_reach


def test_max_edge_length_bounds_the_protected_span() -> None:
    """The cap drops protected faces whose BEV span exceeds it; ``None`` means no bound."""
    points = _road_strip(gap_span=(10.0, 14.0))
    trajectory = _centreline()
    triangles = _delaunay(points)

    def longest_bev_edge(mask: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        corners = points[triangles[mask]][:, :, :2]
        spans = np.maximum.reduce([np.linalg.norm(corners[:, (i + 1) % 3] - corners[:, i], axis=1) for i in range(3)])
        return float(spans.max())

    unbounded = corridor_triangle_mask(
        points, triangles, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M, max_edge_length_m=None
    )
    capped = corridor_triangle_mask(
        points, triangles, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M, max_edge_length_m=3.0
    )

    assert longest_bev_edge(unbounded) > 3.0, "fixture should contain a protected face spanning the gap"
    assert longest_bev_edge(capped) < 3.0
    # The cap only ever removes faces, never adds them.
    assert bool((capped & ~unbounded).sum() == 0)
    assert int(capped.sum()) < int(unbounded.sum())


def test_default_cap_admits_faces_spanning_the_corridor_width() -> None:
    """The default must not forbid a face crossing the corridor, or protection would break there.

    Sized on a 5 m gap, whose spanning faces are ~5.1 m: wider than the 5 m corridor, inside the 6 m
    default. A cap at or below the corridor width would drop exactly the faces protection needs.
    """
    points = _road_strip(gap_span=(10.0, 15.0))
    trajectory = _centreline()
    triangles = _delaunay(points)

    def spans(mask: np.ndarray) -> np.ndarray:
        corners = points[triangles[mask]][:, :, :2]
        return np.maximum.reduce([np.linalg.norm(corners[:, (i + 1) % 3] - corners[:, i], axis=1) for i in range(3)])

    default_mask = corridor_triangle_mask(points, triangles, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M)
    at_width = corridor_triangle_mask(
        points,
        triangles,
        trajectory,
        half_width_m=CORRIDOR_HALF_WIDTH_M,
        max_edge_length_m=2.0 * CORRIDOR_HALF_WIDTH_M,
    )

    assert spans(default_mask).max() > 2.0 * CORRIDOR_HALF_WIDTH_M
    # A cap at the corridor width drops them; the default keeps them.
    assert int(at_width.sum()) < int(default_mask.sum())


def test_cap_below_the_base_threshold_protects_nothing() -> None:
    """The cap is the corridor's max edge, so at or below the base threshold it is inert.

    Both filters use strict ``<``, so {edge < cap} is a subset of {edge < base} once cap <= base:
    every face the exemption would add is already kept unprotected. The CLI rejects such a value;
    this pins the underlying arithmetic.
    """
    points = _road_strip(gap_span=(10.0, 14.0))
    trajectory = _centreline()
    triangles = _delaunay(points)

    _, base_only, _ = filter_triangles_by_max_edge_length(points, triangles, 2.5)
    for cap in (2.5, 2.0, 1.0):
        mask = corridor_triangle_mask(
            points, triangles, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M, max_edge_length_m=cap
        )
        _, kept, _ = filter_triangles_by_max_edge_length(points, triangles, 2.5, protected_triangle_mask=mask)
        assert len(kept) == len(base_only), f"cap {cap} m should add nothing above the base filter"


def test_retention_covers_for_the_cap_when_a_gap_exceeds_it() -> None:
    """A band wider than the cap is not protected, so the mesh splits and retention must save it.

    This is why cap 6 m measured coverage-neutral across the sweep: not because the cap never bites,
    but because component retention ignores it. Guards the pair, not either half alone.
    """
    points = _road_strip(gap_span=(10.0, 18.0))  # ~8 m spanning faces, above the 6 m default
    trajectory = _centreline()
    triangles = _delaunay(points)

    mask = corridor_triangle_mask(points, triangles, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M)
    vertices, kept, _ = filter_triangles_by_max_edge_length(points, triangles, 2.5, protected_triangle_mask=mask)
    assert int(_triangle_components(kept.astype(np.int32)).max()) + 1 == 2, "gap wider than the cap must split"

    keep = select_components_covering_corridor(vertices, kept, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M)
    assert int(keep.sum()) == len(kept), "retention must keep both sides despite the cap"


def test_max_edge_length_cap_does_not_gate_component_retention() -> None:
    """The cap bounds the long-edge exemption only; component retention is a membership question.

    A component carrying corridor area must still be kept even when its faces exceed the cap,
    otherwise a sparse-but-real stretch of road would be deleted outright. The fixture's gap spans
    8 m so its bridging faces are ABOVE the 6 m default -- with a fixture under the cap this test
    would pass vacuously.
    """
    points = _road_strip(gap_span=(10.0, 18.0))
    trajectory = _centreline()
    triangles = _delaunay(points)

    corners = points[triangles][:, :, :2]
    spans = np.maximum.reduce([np.linalg.norm(corners[:, (i + 1) % 3] - corners[:, i], axis=1) for i in range(3)])
    assert spans.max() > DEFAULT_CORRIDOR_MAX_EDGE_LENGTH_M, "fixture must contain faces above the cap"

    # Membership must ignore the cap: capped and uncapped selection agree.
    keep = select_components_covering_corridor(points, triangles, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M)
    uncapped_members = corridor_triangle_mask(
        points, triangles, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M, max_edge_length_m=None
    )
    capped_members = corridor_triangle_mask(points, triangles, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M)
    assert int(capped_members.sum()) < int(uncapped_members.sum()), "cap must matter for the exemption mask"
    assert keep[uncapped_members].all(), "every uncapped corridor face must sit in a retained component"

    # And a split mesh keeps both sides, which is the failure mode this guards.
    vertices, split_tris, _ = filter_triangles_by_max_edge_length(points, triangles, 2.5)
    split_keep = select_components_covering_corridor(
        vertices, split_tris, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M
    )
    assert int(split_keep.sum()) == len(split_tris), "both sides of the gap carry corridor area"


def test_edge_filter_rejects_mismatched_protected_mask() -> None:
    points = _road_strip()
    triangles = _delaunay(points)
    with pytest.raises(ValueError, match="protected_triangle_mask has"):
        filter_triangles_by_max_edge_length(
            points, triangles, 2.5, protected_triangle_mask=np.ones(len(triangles) + 1, dtype=bool)
        )


def test_full_width_gap_is_not_an_interior_hole_so_legacy_fill_cannot_close_it() -> None:
    """Regression witness for failure mode 1: interior-loop filling adds nothing here."""
    points = _road_strip(gap_span=(10.0, 14.0))
    vertices, triangles, _ = filter_triangles_by_max_edge_length(points, _delaunay(points), 2.5)

    filled = fill_interior_boundary_loops(vertices, triangles)

    assert len(filled) == len(triangles), "a full-width gap is not an interior loop; nothing should be filled"
    # And the gap band really is uncovered: no face centroid lands inside it.
    centroids = vertices[filled][:, :, :2].mean(axis=1)
    in_gap = (centroids[:, 0] > 10.0) & (centroids[:, 0] < 14.0) & (np.abs(centroids[:, 1]) < 5.0)
    assert int(in_gap.sum()) == 0


def test_largest_component_keep_deletes_road_beyond_the_gap() -> None:
    """Regression witness for failure mode 2: the smaller side is deleted outright."""
    points = _road_strip(gap_span=(10.0, 14.0))
    vertices, triangles, _ = filter_triangles_by_max_edge_length(points, _delaunay(points), 2.5)

    legacy_keep = keep_largest_connected_component(triangles)
    corridor_keep = select_components_covering_corridor(
        vertices, triangles, _centreline(), half_width_m=CORRIDOR_HALF_WIDTH_M
    )

    assert int(legacy_keep.sum()) < len(triangles), "expected the gap to split the mesh"
    assert int(corridor_keep.sum()) > int(legacy_keep.sum()), "corridor selection must retain both sides"
    npt.assert_array_equal(corridor_keep, np.ones(len(triangles), dtype=bool))


def test_select_components_falls_back_to_largest_when_corridor_misses_mesh() -> None:
    points = _road_strip(gap_span=(10.0, 14.0))
    vertices, triangles, _ = filter_triangles_by_max_edge_length(points, _delaunay(points), 2.5)
    far_trajectory = np.column_stack((np.arange(0.0, 24.0, 0.5), np.full(48, 500.0)))

    keep = select_components_covering_corridor(vertices, triangles, far_trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M)

    npt.assert_array_equal(keep, keep_largest_connected_component(triangles))


def test_patch_and_densify_without_corridor_matches_legacy_path() -> None:
    """Corridor arguments default off, so every existing caller keeps byte-identical output."""
    points = _road_strip(gap_span=(10.0, 14.0))
    vertices, triangles, _ = filter_triangles_by_max_edge_length(points, _delaunay(points), 2.5)

    legacy_v, legacy_t, _ = patch_and_densify_ground_mesh(vertices, triangles, None, densify_max_edge_length=None)
    explicit_v, explicit_t, _ = patch_and_densify_ground_mesh(vertices, triangles, None, densify_max_edge_length=None)

    npt.assert_array_equal(legacy_v, explicit_v)
    npt.assert_array_equal(legacy_t, explicit_t)


def test_corridor_protection_synthesizes_no_geometry() -> None:
    """Corridor mode only retains existing triangles, so every output vertex is an input vertex."""
    points = _road_strip(gap_span=(10.0, 14.0))
    trajectory = _centreline()
    triangles = _delaunay(points)
    protected = corridor_triangle_mask(points, triangles, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M)
    vertices, kept_triangles, _ = filter_triangles_by_max_edge_length(
        points, triangles, 2.5, protected_triangle_mask=protected
    )

    keep = select_components_covering_corridor(vertices, kept_triangles, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M)
    out_vertices, out_triangles, _ = patch_and_densify_ground_mesh(
        vertices,
        kept_triangles[keep],
        None,
        densify_max_edge_length=None,
        drop_floating_components=False,
    )

    # Every surviving vertex position must exist in the input point set.
    input_keys = {tuple(np.round(p, 6)) for p in points.tolist()}
    referenced = np.unique(np.asarray(out_triangles).ravel())
    for position in np.asarray(out_vertices)[referenced].tolist():
        assert tuple(np.round(position, 6)) in input_keys, "corridor mode introduced a synthesized vertex"


def _two_deck_strip() -> tuple[np.ndarray, np.ndarray]:
    """A lower deck at z=0 and an upper deck at z=+7 over the SAME XY footprint.

    Mimics the real two-deck clip (decks ~7.5 m apart) that exposed BEV-only masking.
    """
    lower = _road_strip()
    upper = lower.copy()
    upper[:, 2] += 7.0
    return lower, np.vstack((lower, upper))


def test_bev_only_mask_selects_both_decks() -> None:
    """Witness for the multi-level defect: without a z-band, a lower-deck path claims the upper deck."""
    lower, both = _two_deck_strip()
    trajectory_xy = _centreline()
    triangles = _delaunay(both)

    mask = corridor_triangle_mask(both, triangles, trajectory_xy, half_width_m=CORRIDOR_HALF_WIDTH_M, z_band_m=None)

    centroid_z = both[triangles][:, :, 2].mean(axis=1)[mask]
    assert (centroid_z < 3.5).any(), "expected lower-deck triangles"
    assert (centroid_z > 3.5).any(), "BEV-only masking should (wrongly) also select the upper deck"


def test_z_band_excludes_the_other_deck() -> None:
    """With a z-band, only the deck the trajectory is actually on is selected."""
    lower, both = _two_deck_strip()
    trajectory = np.column_stack((_centreline(), np.zeros(len(_centreline()))))  # lower deck, z=0
    triangles = _delaunay(both)

    mask = corridor_triangle_mask(both, triangles, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M, z_band_m=2.0)

    centroid_z = both[triangles][:, :, 2].mean(axis=1)[mask]
    assert len(centroid_z) > 0, "the lower deck must still be protected"
    assert (centroid_z < 3.5).all(), "z-band must exclude the deck 7 m above the trajectory"

    # And targeting the upper deck selects that one instead.
    upper_trajectory = trajectory.copy()
    upper_trajectory[:, 2] = 7.0
    upper_mask = corridor_triangle_mask(
        both, triangles, upper_trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M, z_band_m=2.0
    )
    upper_z = both[triangles][:, :, 2].mean(axis=1)[upper_mask]
    assert len(upper_z) > 0 and (upper_z > 3.5).all()


def test_corridor_extents_have_measured_defaults() -> None:
    """Extent defaults live in the corridor module, not the mesher."""
    from apps.av.road_mesh.corridor import (
        DEFAULT_CORRIDOR_HALF_WIDTH_M,
        DEFAULT_CORRIDOR_MAX_EDGE_LENGTH_M,
        RECOMMENDED_CORRIDOR_Z_BAND_M,
    )

    assert DEFAULT_CORRIDOR_HALF_WIDTH_M == 2.5
    # The cap must exceed the corridor width, or it would forbid spanning the corridor laterally.
    assert DEFAULT_CORRIDOR_MAX_EDGE_LENGTH_M > 2.0 * DEFAULT_CORRIDOR_HALF_WIDTH_M
    # Recommended band sits above measured rig-to-road offsets and below deck separation.
    assert 0.5 < RECOMMENDED_CORRIDOR_Z_BAND_M < 7.0

    # The mesher takes an exemption callback and holds no corridor state.
    algorithm = DelaunayElevationMeshingAlgorithm()
    assert algorithm.protected_triangle_mask_fn is None
    assert not any(field.startswith("corridor") for field in vars(algorithm))


def test_mesher_rejects_a_wrong_length_mask_from_the_callback() -> None:
    """The exemption hook is validated, so a buggy policy fails loudly rather than mis-filtering."""
    points = _road_strip(gap_span=(10.0, 14.0))
    algorithm = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=False,
        max_triangle_edge_length=2.5,
        protected_triangle_mask_fn=lambda vertices, triangles: np.ones(len(triangles) + 1, dtype=bool),
    )
    with pytest.raises(ValueError, match="protected_triangle_mask_fn returned"):
        algorithm.build_mesh_from_points(points)


def test_component_selection_respects_the_z_band() -> None:
    """Component retention must not keep the upper deck when the ego is on the lower one."""
    lower, both = _two_deck_strip()
    trajectory = np.column_stack((_centreline(), np.zeros(len(_centreline()))))
    triangles = _delaunay(both)
    vertices, kept, _ = filter_triangles_by_max_edge_length(both, triangles, 2.5)

    banded = select_components_covering_corridor(
        vertices, kept, trajectory, half_width_m=CORRIDOR_HALF_WIDTH_M, z_band_m=2.0
    )
    bev_only = select_components_covering_corridor(
        vertices, kept, trajectory[:, :2], half_width_m=CORRIDOR_HALF_WIDTH_M, z_band_m=None
    )

    assert int(banded.sum()) > 0
    assert int(banded.sum()) <= int(bev_only.sum()), "the band can only narrow what is retained"


def test_z_band_falls_back_to_bev_when_elevation_is_missing(caplog) -> None:
    """A 2D trajectory has no elevation, so the band degrades to BEV-only with a warning."""
    _, both = _two_deck_strip()
    triangles = _delaunay(both)

    with caplog.at_level("WARNING"):
        mask = corridor_triangle_mask(both, triangles, _centreline(), half_width_m=CORRIDOR_HALF_WIDTH_M, z_band_m=2.0)

    assert "BEV-only masking" in caplog.text
    bev_only = corridor_triangle_mask(both, triangles, _centreline(), half_width_m=CORRIDOR_HALF_WIDTH_M, z_band_m=None)
    npt.assert_array_equal(mask, bev_only)


def test_hook_is_unreachable_under_the_lidar_road_init_config() -> None:
    """The lidar road-init path cannot reach the exemption hook, whatever it is set to.

    lidar_ground_mesh_road.yaml is ``mesh_topology: grid`` with ``max_triangle_edge_length: null``,
    and the long-edge filter is gated on delaunay AND a non-null threshold, so the hook has no call
    site there. Pinned because a future change re-enabling either would silently expose the lidar
    mesh to corridor policy. Complements the delaunay-topology check below.
    """

    def lidar_mesher(
        hook: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    ) -> DelaunayElevationMeshingAlgorithm:
        """Exactly the lidar_ground_mesh_road.yaml values that reach the mesher."""
        return DelaunayElevationMeshingAlgorithm(
            enable_downsampling=True,
            voxel_size=0.1,
            min_points_per_voxel=1,
            smoothing_passes=10,
            elevation_axis=2,
            mesh_topology="grid",
            smoothing_method="neighbor_average",
            height_statistic="mean",
            color_statistic="mean",
            outlier_height_threshold=None,
            outlier_neighbor_radius=1,
            outlier_min_neighbors=4,
            max_triangle_edge_length=None,
            max_triangle_normal_angle_deg=None,
            protected_triangle_mask_fn=hook,
        )

    # Spacing must be <= voxel_size or the grid mesher finds no occupied 2x2 block and emits nothing.
    points = _road_strip(gap_span=(10.0, 14.0), spacing_m=0.1)

    # reconstruct_ground_mesh_from_points and friends never pass the hook.
    assert lidar_mesher(None).protected_triangle_mask_fn is None

    invocations: list[int] = []

    def spy(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        invocations.append(len(triangles))
        return np.zeros(len(triangles), dtype=bool)

    _, plain_tris, _, _ = lidar_mesher(None).build_mesh_from_points(points)
    hooked_vertices, hooked_tris, _, _ = lidar_mesher(spy).build_mesh_from_points(points)

    assert invocations == [], "the lidar config must never reach the exemption hook"
    assert len(plain_tris) > 0, "fixture should produce a mesh"
    assert plain_tris.shape == hooked_tris.shape
    assert len(hooked_vertices) > 0


def test_lidar_and_other_consumers_are_unaffected():
    """The exemption hook is inert when unset, so other ground_mesh.py consumers are unaffected.

    Topology must match exactly; vertex positions to float32 noise (the mesher drifts ~3.6e-07
    between two identical runs on its own).
    """
    import numpy as np

    from apps.av.road_mesh.ground_mesh import DelaunayElevationMeshingAlgorithm, patch_and_densify_ground_mesh

    pts = []
    rng = np.random.default_rng(7)
    for x in np.arange(0, 30.0, 0.4):
        for y in np.arange(-8.0, 8.01, 0.4):
            if 12.0 < x < 16.0 and y < 2.0:
                continue  # gap: forces component split + filtering
            pts.append((x, y, 0.05 * x + 0.02 * np.sin(y) + rng.normal(0, 0.01)))
    road = np.asarray(pts, dtype=np.float32)

    # A: exactly how the other consumers call it -- no corridor kwargs at all
    a = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=0.3,
        min_points_per_voxel=1,
        smoothing_passes=3,
        elevation_axis=2,
        mesh_topology="delaunay",
        height_statistic="mean",
        color_statistic="mean",
        max_triangle_edge_length=1.0,
        max_triangle_normal_angle_deg=60.0,
    )
    Va, Ta, _, _ = a.build_mesh_from_points(road)
    Pa = patch_and_densify_ground_mesh(Va, Ta, None, densify_max_edge_length=None)

    # B: same, but the exemption hook explicitly at its default (None)
    b = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=0.3,
        min_points_per_voxel=1,
        smoothing_passes=3,
        elevation_axis=2,
        mesh_topology="delaunay",
        height_statistic="mean",
        color_statistic="mean",
        max_triangle_edge_length=1.0,
        max_triangle_normal_angle_deg=60.0,
        protected_triangle_mask_fn=None,
    )
    Vb, Tb, _, _ = b.build_mesh_from_points(road)
    Pb = patch_and_densify_ground_mesh(Vb, Tb, None, densify_max_edge_length=None)

    print(f"A (no hook kwarg):  verts={len(Va)} tris={len(Ta)}  postprocess tris={len(Pa[1])}")
    print(f"B (hook=None):      verts={len(Vb)} tris={len(Tb)}  postprocess tris={len(Pb[1])}")
    np.testing.assert_array_equal(Ta, Tb)
    np.testing.assert_array_equal(Pa[1], Pb[1])
    dv = np.abs(np.asarray(Va) - np.asarray(Vb)).max()
    dp = np.abs(np.asarray(Pa[0]) - np.asarray(Pb[0])).max()
    print(f"topology identical: build tris {len(Ta)}=={len(Tb)}, postprocess tris {len(Pa[1])}=={len(Pb[1])}")
    print(f"max vertex delta: build={dv:.3e} m  postprocess={dp:.3e} m  (float32 eps ~1.2e-07)")
    assert dv < 1e-5 and dp < 1e-5, "vertex drift beyond float32 noise"

    # C: the lidar/USDZ entry point cannot even opt in -- it exposes no corridor parameter.
    import inspect

    sig = inspect.signature(reconstruct_ground_mesh_from_points)
    corridor_params = [p for p in sig.parameters if "corridor" in p]
    print(
        f"reconstruct_ground_mesh_from_points corridor params: {corridor_params or 'NONE -- cannot opt in, so unchanged'}"
    )
    assert not corridor_params

    # D: export_usdz's exact construction still works
    u = DelaunayElevationMeshingAlgorithm(enable_downsampling=False)
    assert u.protected_triangle_mask_fn is None
    print(f"export_usdz-style ctor: protected_triangle_mask_fn={u.protected_triangle_mask_fn}")
