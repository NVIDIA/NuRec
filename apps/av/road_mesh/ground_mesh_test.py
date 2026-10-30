# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import apps.av.road_mesh.ground_mesh as ground_mesh_module

from apps.av.road_mesh.ground_mesh import (
    DelaunayElevationMeshingAlgorithm,
    GroundMeshReconstructionSettings,
    _generate_ground_points_from_cuboids_unfiltered,
    _GridMeshCudaResult,
    _triangulate_simple_polygon_xy,
    build_grid_constrained_mesh,
    calculate_edge_lengths,
    compute_mesh_normals_in_lidar_spin,
    compute_mesh_normals_in_lidar_spin_torch,
    construct_lidar_spin_mesh,
    downsample_3d_points_on_2d_grid,
    downsample_3d_points_on_2d_grid_no_clamp,
    fill_interior_boundary_loops,
    filter_local_height_outliers,
    filter_triangles_by_max_edge_length,
    filter_triangles_by_max_normal_angle,
    find_boundary_loops,
    generate_ground_points_from_cuboids,
    get_mesh_edges,
    get_nominal_ground_point_under_lidar,
    get_vertex_neighbors,
    keep_largest_connected_component,
    mesh_smoothing_along_axis,
    smooth_mesh_along_axis_torch,
)
from nurec.datasets.tracks import CuboidTracks
from nurec.utils.types import TrackFlags


def test_ground_mesh_settings_accept_spin_mesh_device() -> None:
    settings = GroundMeshReconstructionSettings.from_mapping({"spin_mesh_device": "cuda"})

    assert settings.spin_mesh_device == "cuda"


def test_ground_mesh_settings_preserve_legacy_delaunay_default() -> None:
    settings = GroundMeshReconstructionSettings.from_mapping({})

    assert settings.mesh_topology == "delaunay"


def test_ground_mesh_settings_reject_invalid_spin_mesh_device() -> None:
    with pytest.raises(ValueError, match="spin_mesh_device"):
        GroundMeshReconstructionSettings.from_mapping({"spin_mesh_device": "cdua"})


def test_single_threaded_blas_context_restores_absent_environment(monkeypatch: Any) -> None:
    variable_names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    for name in variable_names:
        monkeypatch.delenv(name, raising=False)

    with ground_mesh_module._single_threaded_blas_context():
        assert all(os.environ[name] == "1" for name in variable_names)

    assert all(name not in os.environ for name in variable_names)


def test_single_threaded_blas_context_restores_existing_environment(monkeypatch: Any) -> None:
    previous_values = {
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "6",
        "OPENBLAS_NUM_THREADS": "4",
    }
    for name, value in previous_values.items():
        monkeypatch.setenv(name, value)

    with ground_mesh_module._single_threaded_blas_context():
        assert all(os.environ[name] == "1" for name in previous_values)

    assert {name: os.environ[name] for name in previous_values} == previous_values


def test_mesher_default_preserves_sparse_delaunay_ground_coverage() -> None:
    rng = np.random.default_rng(123)
    points = np.column_stack(
        (
            rng.uniform(-500.0, 500.0, 10_000),
            rng.uniform(-500.0, 500.0, 10_000),
            np.zeros(10_000),
        )
    ).astype(np.float32)

    _, triangles, vertices, _ = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=False,
        smoothing_passes=0,
    ).build_mesh_from_points(points)

    assert len(triangles) > 19_000
    assert np.ptp(vertices[:, 0]) > 990.0
    assert np.ptp(vertices[:, 1]) > 990.0


def test_get_mesh_edges() -> None:
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    edges = get_mesh_edges(triangles)
    print(edges)
    assert np.all(edges == np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]], dtype=np.int32))


def test_calculate_edge_lengths() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 12.0]], dtype=np.float32)
    edges = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32)

    edge_lengths = calculate_edge_lengths(vertices, edges)

    assert np.allclose(edge_lengths, np.array([5.0, 12.0, 13.0], dtype=np.float32))


def test_fill_invalid_road_colors_preserves_observed_and_avoids_black() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [
            [90, 100, 110],
            [0, 0, 0],
            [0, 0, 0],
            [110, 120, 130],
        ],
        dtype=np.uint8,
    )
    valid = np.array([True, False, False, True])
    road = np.ones(4, dtype=bool)

    filled = ground_mesh_module._fill_invalid_road_colors(points, colors, valid, road)

    np.testing.assert_array_equal(filled[valid], colors[valid])
    assert np.all(filled[~valid] > 0)
    assert np.all(filled[~valid] >= np.array([90, 100, 110], dtype=np.uint8))
    assert np.all(filled[~valid] <= np.array([110, 120, 130], dtype=np.uint8))


def test_fill_invalid_road_colors_does_not_cross_height_layers() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],  # lower-road donor
            [0.1, 0.0, 0.0],  # lower-road masked
            [0.1, 0.0, 2.0],  # upper-road donor, close in XY
        ],
        dtype=np.float32,
    )
    colors = np.array([[80, 90, 100], [0, 0, 0], [240, 20, 240]], dtype=np.uint8)
    valid = np.array([True, False, True])
    road = np.ones(3, dtype=bool)

    filled = ground_mesh_module._fill_invalid_road_colors(
        points,
        colors,
        valid,
        road,
        neighbor_count=2,
    )

    np.testing.assert_array_equal(filled[1], colors[0])


def test_fill_invalid_road_colors_uses_neutral_gray_when_all_road_is_masked(caplog) -> None:
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[10, 20, 30]], dtype=np.uint8)

    caplog.set_level("WARNING", logger=ground_mesh_module.__name__)
    filled = ground_mesh_module._fill_invalid_road_colors(
        points,
        colors,
        np.array([False]),
        np.array([True]),
    )

    np.testing.assert_array_equal(filled, np.array([[128, 128, 128]], dtype=np.uint8))
    assert not np.shares_memory(filled, colors)
    assert "no directly observed road colors are available" in caplog.text


def test_get_vertex_neighbors() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 12.0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)

    neighbors, distances = get_vertex_neighbors(vertices, triangles)

    assert neighbors == [[1, 2], [0, 2], [0, 1]]
    assert np.allclose(distances[0], [5.0, 12.0])
    assert np.allclose(distances[1], [5.0, 13.0])
    assert np.allclose(distances[2], [12.0, 13.0])


def test_pooled_worker_honors_spin_mesh_device(monkeypatch: Any) -> None:
    seen_devices: list[str] = []

    def fake_compute_mesh_normals_in_lidar_spin(
        points: np.ndarray,
        verbose: bool = False,
        n_elevation_bins: int | None = None,
        n_azimuth_bins: int | None = None,
        device: str = "cpu",
    ) -> tuple[np.ndarray, np.ndarray]:
        del verbose, n_elevation_bins, n_azimuth_bins
        seen_devices.append(device)
        normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(points), 1))
        triangles = np.zeros((0, 3), dtype=np.int32)
        return normals, triangles

    monkeypatch.setattr(
        ground_mesh_module,
        "compute_mesh_normals_in_lidar_spin",
        fake_compute_mesh_normals_in_lidar_spin,
    )
    config = ground_mesh_module._PerSpinConfig(
        min_ray_length=0.0,
        ground_control_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        ground_compat_max_distance=1.0,
        ground_compat_max_angle_deg=45.0,
        num_plane_hypotheses=1,
        plane_max_distance=1.0,
        plane_max_angle_deg=45.0,
        plane_min_eval_points=999,
        plane_max_eval_points=1000,
        plane_min_inliers=1,
        enable_plane_extension=False,
        spin_mesh_device="cuda",
        transformation=None,
    )

    ground_mesh_module._process_single_spin_worker(
        0,
        np.zeros((3, 3), dtype=np.float32),
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
        np.array([True, True, True], dtype=np.bool_),
        None,
        config,
    )

    assert seen_devices == ["cuda"]


def test_smooth_mesh_along_axis_torch() -> None:
    vertices = np.array([[0, 3, 10], [0, 0, 60], [4, 3, 30], [4, 0, 50]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    axis = 2
    smooth_vertices = smooth_mesh_along_axis_torch(vertices, triangles, axis=axis, smoothness=1.0, num_passes=1)
    # Coordinates along the non-smoothing axes are unchanged.
    fixed_axes = [a for a in (0, 1, 2) if a != axis]
    assert np.all(smooth_vertices[:, fixed_axes] == vertices[:, fixed_axes])
    # Each smoothed coordinate lies strictly between the min and max of its 1-ring neighbors.
    assert smooth_vertices[0, axis] > np.min(vertices[[1, 2], axis])
    assert smooth_vertices[0, axis] < np.max(vertices[[1, 2], axis])
    assert smooth_vertices[1, axis] > np.min(vertices[[0, 2, 3], axis])
    assert smooth_vertices[1, axis] < np.max(vertices[[0, 2, 3], axis])
    assert smooth_vertices[2, axis] > np.min(vertices[[0, 1, 3], axis])
    assert smooth_vertices[2, axis] < np.max(vertices[[0, 1, 3], axis])
    assert smooth_vertices[3, axis] > np.min(vertices[[1, 2], axis])
    assert smooth_vertices[3, axis] < np.max(vertices[[1, 2], axis])


def test_smooth_mesh_along_axis_torch_isolated_vertex_preserves_height() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 20.0],
            [0.0, 1.0, 30.0],
            [5.0, 5.0, 99.0],
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)

    out = smooth_mesh_along_axis_torch(vertices, triangles, axis=2, smoothness=1.0, num_passes=2)

    assert np.isfinite(out).all()
    assert np.isclose(out[3, 2], 99.0, atol=1e-5)


def test_smooth_mesh_along_axis_local_plane_projects_to_neighbor_plane() -> None:
    """Local-plane smoothing preserves slope instead of averaging neighbor heights."""
    vertices = np.array(
        [
            [0.0, 0.0, 10.0],  # wrong height; target plane is z = 2x + 3y
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 5.0],
        ],
        dtype=np.float32,
    )
    neighbors = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    distances = [
        [1.0, 1.0, np.sqrt(2.0)],
        [1.0, np.sqrt(2.0), 1.0],
        [1.0, np.sqrt(2.0), 1.0],
        [np.sqrt(2.0), 1.0, 1.0],
    ]

    smooth_vertices = mesh_smoothing_along_axis(
        vertices,
        neighbors,
        distances,
        axis=2,
        smoothness=1.0,
        method="local_plane",
    )

    assert np.isclose(smooth_vertices[0, 2], 0.0, atol=1e-5)
    assert np.all(smooth_vertices[:, :2] == vertices[:, :2])


def test_downsample_3d_points_on_2d_grid() -> None:
    points = np.array(
        [
            # Cell (0,0) with center (2.0, 0.0)
            [1.0, -1.0, 11.1],  # Bottom left point of the grid (bbox)
            [2.5, 0.5, 13.3],
            # Cell (1,1) with center (3.0, 2.0)
            [4.0, 2.0, -3.3],
            # Cell (3,2) with center (8.0, 4.0)
            [7.0, 3.0, 4.4],  # Top right point of the grid (bbox)
            # Cell (2,0) with center (6, 0.0)
            [5.5, -0.5, 2.0],
            [5.5, 0.5, 3.0],
            [6.5, 0.5, 4.0],
        ]
    )
    expected_points = np.array([[2.0, 0.0, 12.2], [6.0, 0.0, 3.0], [4.0, 2.0, -3.3], [8.0, 4.0, 4.4]])
    downsampled_points, downsampled_colors = downsample_3d_points_on_2d_grid(points, cell_size=2.0)
    assert downsampled_points.shape == (4, 3)
    assert downsampled_colors is None
    assert np.allclose(expected_points, downsampled_points, rtol=0.0, atol=1e-4)


def test_downsample_3d_points_on_2d_grid_no_clamp_averages_occupied_cells() -> None:
    points = np.array(
        [
            [-0.9, -0.8, 2.0],
            [-0.2, -0.4, 4.0],
            [0.1, 0.2, 1.0],
            [0.7, 0.8, 3.0],
            [1.2, 0.3, 5.0],
        ],
        dtype=np.float32,
    )

    merged_points = downsample_3d_points_on_2d_grid_no_clamp(points, cell_size=1.0)

    # These are the original XYZ means; their XY values deliberately differ from the grid-cell centers.
    expected_points = np.array(
        [
            [-0.55, -0.6, 3.0],
            [0.4, 0.5, 2.0],
            [1.2, 0.3, 5.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(merged_points, expected_points)


def test_downsample_3d_points_on_2d_grid_no_clamp_handles_empty_input() -> None:
    merged_points = downsample_3d_points_on_2d_grid_no_clamp(np.empty((0, 3), dtype=np.float32), cell_size=0.1)

    assert merged_points.shape == (0, 3)
    assert merged_points.dtype == np.float32


def test_downsample_3d_points_on_2d_grid_with_colors() -> None:
    """Test that per-point colors are averaged per cell during downsampling."""
    # Same 7-point layout as test_downsample_3d_points_on_2d_grid (4 cells, cell_size=2.0)
    points = np.array(
        [
            # Cell (0,0): 2 points
            [1.0, -1.0, 11.1],
            [2.5, 0.5, 13.3],
            # Cell (1,1): 1 point
            [4.0, 2.0, -3.3],
            # Cell (3,2): 1 point
            [7.0, 3.0, 4.4],
            # Cell (2,0): 3 points
            [5.5, -0.5, 2.0],
            [5.5, 0.5, 3.0],
            [6.5, 0.5, 4.0],
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [
            [100, 0, 50],  # Cell (0,0)
            [200, 100, 50],  # Cell (0,0)
            [10, 20, 30],  # Cell (1,1)
            [255, 255, 0],  # Cell (3,2)
            [60, 90, 120],  # Cell (2,0)
            [90, 90, 180],  # Cell (2,0)
            [150, 120, 0],  # Cell (2,0)
        ],
        dtype=np.uint8,
    )
    downsampled_points, downsampled_colors = downsample_3d_points_on_2d_grid(points, cell_size=2.0, colors=colors)

    assert downsampled_colors is not None
    assert downsampled_colors.shape == (4, 3)
    assert downsampled_colors.dtype == np.uint8

    # Build expected colors in the same cell order as downsampled_points.
    # Cell (0,0): mean([100,200], [0,100], [50,50]) = [150.0, 50.0, 50.0] -> uint8 [150, 50, 50]
    # Cell (2,0): mean([60,90,150], [90,90,120], [120,180,0]) = [100.0, 100.0, 100.0] -> [100, 100, 100]
    # Cell (1,1): single point -> [10, 20, 30]
    # Cell (3,2): single point -> [255, 255, 0]
    expected_colors = np.array(
        [
            [150, 50, 50],  # Cell (0,0)
            [100, 100, 100],  # Cell (2,0)
            [10, 20, 30],  # Cell (1,1)
            [255, 255, 0],  # Cell (3,2)
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(downsampled_colors, expected_colors)


def test_build_mesh_colors_propagated() -> None:
    """Test that colors survive downsampling and Delaunay meshing."""
    rng = np.random.RandomState(42)
    n_points = 50
    # Random XY in [0, 10], small Z variation
    points = np.column_stack(
        [rng.uniform(0, 10, n_points), rng.uniform(0, 10, n_points), rng.uniform(-0.1, 0.1, n_points)]
    ).astype(np.float32)
    colors = rng.randint(0, 256, (n_points, 3)).astype(np.uint8)

    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=5.0,  # Large voxel to force merging
        smoothing_passes=0,
    )
    smooth_vertices, triangles, vertices, vertex_colors = mesher.build_mesh_from_points(points, colors=colors)

    assert vertex_colors is not None
    assert vertex_colors.shape[0] == smooth_vertices.shape[0]
    assert vertex_colors.shape[1] == 3
    assert vertex_colors.dtype == np.uint8
    assert triangles.shape[1] == 3


def test_build_mesh_no_downsampling_colors_passthrough() -> None:
    """Without downsampling, colors pass through unchanged."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]], dtype=np.uint8)

    mesher = DelaunayElevationMeshingAlgorithm(enable_downsampling=False, smoothing_passes=0)
    _, _, _, vertex_colors = mesher.build_mesh_from_points(points, colors=colors)

    assert vertex_colors is not None
    assert np.array_equal(vertex_colors, colors)


def test_build_mesh_colors_none() -> None:
    """No colors passed -> vertex_colors is None."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

    mesher = DelaunayElevationMeshingAlgorithm(enable_downsampling=False, smoothing_passes=0)
    _, _, _, vertex_colors = mesher.build_mesh_from_points(points)

    assert vertex_colors is None


def test_downsample_with_median_height_drops_spike() -> None:
    """Using median height aggregation ignores a single outlier spike inside a cell."""
    # Single cell (cell_size=10) with 4 near-zero points and one large spike. Mean would be pulled
    # toward the spike; median ignores it.
    points = np.array(
        [
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.05],
            [3.0, 3.0, -0.05],
            [4.0, 4.0, 0.0],
            [5.0, 5.0, 5.0],  # spike outlier
        ],
        dtype=np.float32,
    )
    mean_result, _ = downsample_3d_points_on_2d_grid(points, cell_size=10.0)
    median_result, _ = downsample_3d_points_on_2d_grid(points, cell_size=10.0, height_statistic="median")

    assert mean_result.shape == (1, 3)
    assert median_result.shape == (1, 3)
    assert np.isclose(mean_result[0, 2], 1.0, atol=1e-3)  # pulled toward spike
    assert np.isclose(median_result[0, 2], 0.0, atol=1e-3)  # ignores spike


def test_downsample_min_points_per_cell_drops_sparse() -> None:
    """``min_points_per_cell=3`` drops cells that don't meet the threshold."""
    points = np.array(
        [
            # Cell A: 3 points -> kept
            [0.5, 0.5, 0.0],
            [0.6, 0.6, 0.0],
            [0.7, 0.7, 0.0],
            # Cell B: 1 point -> dropped
            [10.5, 10.5, 0.0],
        ],
        dtype=np.float32,
    )
    keep_all, _ = downsample_3d_points_on_2d_grid(points, cell_size=1.0)
    keep_three_plus, _ = downsample_3d_points_on_2d_grid(points, cell_size=1.0, min_points_per_cell=3)

    assert keep_all.shape == (2, 3)
    assert keep_three_plus.shape == (1, 3)
    # Surviving cell is cell A. Its center = bbox_min + (idx + 0.5) * cell_size
    # = 0.5 + (0 + 0.5) * 1.0 = 1.0 along each XY axis.
    assert np.isclose(keep_three_plus[0, 0], 1.0, atol=1e-3)
    assert np.isclose(keep_three_plus[0, 1], 1.0, atol=1e-3)


def test_filter_local_height_outliers_drops_spike() -> None:
    """A spike cell surrounded by flat neighbors gets dropped."""
    # 3x3 grid at cell_size=1. Center is a spike; eight neighbors at z=0.
    vertices = np.array(
        [
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [2.5, 0.5, 0.0],
            [0.5, 1.5, 0.0],
            [1.5, 1.5, 5.0],  # spike at center
            [2.5, 1.5, 0.0],
            [0.5, 2.5, 0.0],
            [1.5, 2.5, 0.0],
            [2.5, 2.5, 0.0],
        ],
        dtype=np.float32,
    )
    keep = filter_local_height_outliers(vertices, cell_size=1.0, threshold=0.3, min_neighbors=4, radius=1)
    assert keep.sum() == 8
    assert not keep[4]  # the spike vertex


def test_filter_local_height_outliers_keeps_sparse_boundary() -> None:
    """Cells with fewer than ``min_neighbors`` neighbors are kept regardless of deviation."""
    # Two isolated cells far apart; each has at most 1 neighbor (zero, actually). Even with
    # extreme z deviation neither should be dropped because there's no support to call it an outlier.
    vertices = np.array(
        [
            [0.5, 0.5, 0.0],
            [50.5, 50.5, 10.0],
        ],
        dtype=np.float32,
    )
    keep = filter_local_height_outliers(vertices, cell_size=1.0, threshold=0.3, min_neighbors=4, radius=1)
    assert keep.all()


def test_filter_local_height_outliers_disabled_with_zero_threshold() -> None:
    vertices = np.array([[0.5, 0.5, 0.0], [1.5, 0.5, 10.0]], dtype=np.float32)
    keep = filter_local_height_outliers(vertices, cell_size=1.0, threshold=0.0, min_neighbors=4)
    assert keep.all()


def test_filter_triangles_by_max_edge_length_drops_long() -> None:
    """Long bridging triangles get dropped; short triangles and their vertices are retained."""
    # Three short triangles forming a small patch, plus one synthetic long-bridge triangle.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [1.0, 1.0, 0.0],  # 3
            [100.0, 0.0, 0.0],  # 4 - far vertex
        ],
        dtype=np.float32,
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
            [0, 1, 4],  # long bridging triangle
        ],
        dtype=np.int32,
    )
    new_v, new_t, keep_mask = filter_triangles_by_max_edge_length(vertices, triangles, max_edge_length=2.0)
    assert len(new_t) == 2
    assert len(new_v) == 4  # vertex 4 (the far one) gets compacted out
    assert not keep_mask[4]
    # New triangle indices should be valid against new vertex array.
    assert int(new_t.max()) < len(new_v)


def test_filter_triangles_by_max_edge_length_all_dropped() -> None:
    """When every triangle exceeds the threshold the result is an empty mesh."""
    vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    new_v, new_t, keep_mask = filter_triangles_by_max_edge_length(vertices, triangles, max_edge_length=1.0)
    assert len(new_t) == 0
    assert len(new_v) == 0
    assert not keep_mask.any()


def test_filter_triangles_by_max_edge_length_empty_input() -> None:
    vertices = np.array([[0, 0, 0]], dtype=np.float32)
    triangles = np.zeros((0, 3), dtype=np.int32)
    new_v, new_t, keep_mask = filter_triangles_by_max_edge_length(vertices, triangles, max_edge_length=1.0)
    assert len(new_t) == 0
    assert new_v.shape == (1, 3)
    assert keep_mask.all()


def test_filter_triangles_by_max_normal_angle_drops_vertical_curtain() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.1, 10.0],  # short XY edge, large Z jump -> near-vertical face
        ],
        dtype=np.float32,
    )
    triangles = np.array(
        [
            [0, 1, 2],  # horizontal, kept
            [0, 1, 3],  # curtain, dropped
        ],
        dtype=np.int32,
    )

    new_v, new_t, keep_mask = filter_triangles_by_max_normal_angle(vertices, triangles, max_angle_deg=60.0)

    assert len(new_t) == 1
    assert len(new_v) == 3
    assert not keep_mask[3]
    assert np.allclose(new_v[new_t[0], 2], 0.0)


def test_build_grid_constrained_mesh_does_not_bridge_separate_patches() -> None:
    vertices = np.array(
        [
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [0.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
            [10.5, 10.5, 0.0],
            [11.5, 10.5, 0.0],
            [10.5, 11.5, 0.0],
            [11.5, 11.5, 0.0],
        ],
        dtype=np.float32,
    )

    new_v, new_t, keep_mask = build_grid_constrained_mesh(vertices, cell_size=1.0)
    edges = get_mesh_edges(new_t)
    edge_lengths = calculate_edge_lengths(new_v, edges)

    assert len(new_t) == 4
    assert keep_mask.all()
    assert edge_lengths.max() <= np.sqrt(2.0) + 1e-5


def test_build_grid_constrained_mesh_keeps_three_corner_normals_up() -> None:
    patterns = [
        # Missing v00
        [[1.5, 0.5, 0.0], [0.5, 1.5, 0.0], [1.5, 1.5, 0.0]],
        # Missing v10
        [[0.5, 0.5, 0.0], [0.5, 1.5, 0.0], [1.5, 1.5, 0.0]],
        # Missing v01
        [[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [1.5, 1.5, 0.0]],
        # Missing v11
        [[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [0.5, 1.5, 0.0]],
    ]

    for pattern in patterns:
        vertices = np.array(pattern, dtype=np.float32)
        new_v, new_t, _ = build_grid_constrained_mesh(vertices, cell_size=1.0)
        assert len(new_t) == 1
        tri = new_v[new_t[0]]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        assert normal[2] > 0


def test_build_mesh_grid_topology_does_not_bridge_separate_patches() -> None:
    points = np.array(
        [
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [0.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
            [10.5, 10.5, 0.0],
            [11.5, 10.5, 0.0],
            [10.5, 11.5, 0.0],
            [11.5, 11.5, 0.0],
        ],
        dtype=np.float32,
    )
    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=False,
        voxel_size=1.0,
        mesh_topology="grid",
        smoothing_passes=0,
    )

    vertices, triangles, _, _ = mesher.build_mesh_from_points(points)
    edge_lengths = calculate_edge_lengths(vertices, get_mesh_edges(triangles))

    assert len(triangles) == 4
    assert edge_lengths.max() <= np.sqrt(2.0) + 1e-5


def test_build_mesh_grid_mean_uses_cuda_fast_path(monkeypatch: Any) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    colors = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    fast_vertices = np.array([[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [0.5, 1.5, 0.0]], dtype=np.float32)
    fast_triangles = np.array([[0, 1, 2]], dtype=np.int32)
    fast_colors = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    calls: list[tuple[np.ndarray, np.ndarray | None, float, int, int]] = []

    def fake_fast_path(
        in_points: np.ndarray,
        colors: np.ndarray | None,
        cell_size: float,
        min_points_per_cell: int,
        hole_fill_min_neighbors: int,
        smoothing_passes: int,
        elevation_axis: int,
    ) -> _GridMeshCudaResult:
        calls.append((in_points, colors, cell_size, min_points_per_cell, hole_fill_min_neighbors))
        return _GridMeshCudaResult(
            vertices=fast_vertices,
            triangles=fast_triangles,
            vertex_colors=fast_colors,
            smooth_vertices=None,
        )

    monkeypatch.setattr(ground_mesh_module, "_try_build_grid_mesh_mean_cuda", fake_fast_path)
    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=1.0,
        mesh_topology="grid",
        smoothing_passes=0,
        height_statistic="mean",
        color_statistic="mean",
    )

    vertices, triangles, raw_vertices, vertex_colors = mesher.build_mesh_from_points(points, colors=colors)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], points)
    assert calls[0][1] is not None
    np.testing.assert_array_equal(calls[0][1], colors)
    assert calls[0][2] == 1.0
    assert calls[0][3] == 1
    assert calls[0][4] == 6
    np.testing.assert_array_equal(vertices, fast_vertices)
    np.testing.assert_array_equal(raw_vertices, fast_vertices)
    np.testing.assert_array_equal(triangles, fast_triangles)
    assert vertex_colors is not None
    np.testing.assert_array_equal(vertex_colors, fast_colors)


def test_build_mesh_grid_cuda_fast_path_honors_local_plane_smoothing(monkeypatch: Any) -> None:
    points = np.array(
        [
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 5.0],
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    seen_smoothing_passes: list[int] = []
    seen_methods: list[str] = []

    def fake_fast_path(
        in_points: np.ndarray,
        colors: np.ndarray | None,
        cell_size: float,
        min_points_per_cell: int,
        hole_fill_min_neighbors: int,
        smoothing_passes: int,
        elevation_axis: int,
    ) -> _GridMeshCudaResult:
        del colors, cell_size, min_points_per_cell, hole_fill_min_neighbors, elevation_axis
        seen_smoothing_passes.append(smoothing_passes)
        return _GridMeshCudaResult(
            vertices=in_points.copy(),
            triangles=triangles,
            vertex_colors=None,
            smooth_vertices=None,
        )

    def fake_local_plane_smoothing(
        vertices: np.ndarray,
        vertex_neighbors: list[list[int]],
        vertex_distances: list[list[float]],
        **kwargs: Any,
    ) -> np.ndarray:
        del vertex_neighbors, vertex_distances
        seen_methods.append(kwargs["method"])
        smoothed = vertices.copy()
        smoothed[:, 2] = 42.0
        return smoothed

    monkeypatch.setattr(ground_mesh_module, "_try_build_grid_mesh_mean_cuda", fake_fast_path)
    monkeypatch.setattr(ground_mesh_module, "mesh_smoothing_along_axis", fake_local_plane_smoothing)
    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=1.0,
        mesh_topology="grid",
        smoothing_passes=1,
        smoothing_method="local_plane",
        height_statistic="mean",
        color_statistic="mean",
    )

    smooth_vertices, _, _, _ = mesher.build_mesh_from_points(points)

    assert seen_smoothing_passes == [0]
    assert seen_methods == ["local_plane"]
    np.testing.assert_array_equal(smooth_vertices[:, 2], np.full(4, 42.0, dtype=np.float32))


def test_build_mesh_grid_cuda_smoothing_recomputes_after_normal_filter(monkeypatch: Any) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 10.0],
        ],
        dtype=np.float32,
    )
    original_triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    filtered_triangles = np.array([[0, 1, 2]], dtype=np.int32)
    keep_mask = np.array([True, True, True, False])
    smooth_input_sizes: list[int] = []
    filter_calls = 0

    def fake_fast_path(
        in_points: np.ndarray,
        colors: np.ndarray | None,
        cell_size: float,
        min_points_per_cell: int,
        hole_fill_min_neighbors: int,
        smoothing_passes: int,
        elevation_axis: int,
    ) -> _GridMeshCudaResult:
        del colors, cell_size, min_points_per_cell, hole_fill_min_neighbors, smoothing_passes, elevation_axis
        pre_smoothed = in_points.copy()
        pre_smoothed[:, 2] = 99.0
        return _GridMeshCudaResult(
            vertices=in_points.copy(),
            triangles=original_triangles,
            vertex_colors=None,
            smooth_vertices=pre_smoothed,
        )

    def fake_normal_filter(
        vertices: np.ndarray,
        triangles: np.ndarray,
        max_angle_deg: float,
        axis: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal filter_calls
        del max_angle_deg, axis
        filter_calls += 1
        if filter_calls == 1:
            assert len(vertices) == 4
            return vertices[keep_mask].copy(), filtered_triangles, keep_mask
        return vertices.copy(), triangles.copy(), np.ones(len(vertices), dtype=bool)

    def fake_smoothing(vertices: np.ndarray, *_args: Any, **_kwargs: Any) -> np.ndarray:
        smooth_input_sizes.append(len(vertices))
        return vertices.copy()

    monkeypatch.setattr(ground_mesh_module, "_try_build_grid_mesh_mean_cuda", fake_fast_path)
    monkeypatch.setattr(ground_mesh_module, "filter_triangles_by_max_normal_angle", fake_normal_filter)
    monkeypatch.setattr(ground_mesh_module, "smooth_mesh_along_axis_torch", fake_smoothing)
    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=1.0,
        mesh_topology="grid",
        smoothing_passes=1,
        height_statistic="mean",
        color_statistic="mean",
        max_triangle_normal_angle_deg=60.0,
    )

    smooth_vertices, triangles, vertices, _ = mesher.build_mesh_from_points(points)

    assert smooth_input_sizes == [3]
    assert len(smooth_vertices) == 3
    assert len(vertices) == 3
    np.testing.assert_array_equal(triangles, filtered_triangles)


def test_build_mesh_grid_outlier_filter_skips_cuda_fast_path(monkeypatch: Any) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    def fail_fast_path(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("CUDA fast path should not run when local height filtering is enabled")

    monkeypatch.setattr(ground_mesh_module, "_try_build_grid_mesh_mean_cuda", fail_fast_path)
    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=1.0,
        mesh_topology="grid",
        smoothing_passes=0,
        height_statistic="mean",
        color_statistic="mean",
        outlier_height_threshold=0.5,
    )

    _, triangles, _, _ = mesher.build_mesh_from_points(points)

    assert len(triangles) > 0


def test_build_mesh_grid_falls_back_to_cpu_when_cuda_fast_path_declines(monkeypatch: Any, caplog) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    calls = 0

    def declining_fast_path(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(ground_mesh_module, "_try_build_grid_mesh_mean_cuda", declining_fast_path)
    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=1.0,
        mesh_topology="grid",
        smoothing_passes=0,
        height_statistic="mean",
        color_statistic="mean",
    )

    caplog.set_level("INFO", logger=ground_mesh_module.__name__)
    smooth_vertices, triangles, vertices, _ = mesher.build_mesh_from_points(points)

    assert calls == 1
    assert "Grid-constrained triangulation" in caplog.text
    assert len(vertices) == 4
    assert len(triangles) > 0
    np.testing.assert_array_equal(smooth_vertices, vertices)


def test_build_mesh_grid_cuda_smoothing_is_not_recomputed_on_host(monkeypatch: Any, caplog) -> None:
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 4.0],
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    def fake_fast_path(*args: Any, **kwargs: Any) -> _GridMeshCudaResult:
        pre_smoothed = points.copy()
        pre_smoothed[:, 2] = 7.0
        return _GridMeshCudaResult(
            vertices=points.copy(),
            triangles=triangles,
            vertex_colors=None,
            smooth_vertices=pre_smoothed,
        )

    def fail_smoothing(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("host-side smoothing must not run when the CUDA fast path already smoothed")

    monkeypatch.setattr(ground_mesh_module, "_try_build_grid_mesh_mean_cuda", fake_fast_path)
    monkeypatch.setattr(ground_mesh_module, "smooth_mesh_along_axis_torch", fail_smoothing)
    monkeypatch.setattr(ground_mesh_module, "mesh_smoothing_along_axis", fail_smoothing)
    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=1.0,
        mesh_topology="grid",
        smoothing_passes=2,
        height_statistic="mean",
        color_statistic="mean",
    )

    caplog.set_level("INFO", logger=ground_mesh_module.__name__)
    smooth_vertices, out_triangles, vertices, _ = mesher.build_mesh_from_points(points)

    assert "Mesh smoothing over" not in caplog.text
    np.testing.assert_array_equal(smooth_vertices[:, 2], np.full(4, 7.0, dtype=np.float32))
    np.testing.assert_array_equal(vertices, points)
    np.testing.assert_array_equal(out_triangles, triangles)


def test_detect_dominant_plane_skips_native_path_without_cuda(monkeypatch: Any) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(points), 1))

    ext = MagicMock()
    # Now that fast-path errors propagate, this side effect would surface as a test failure rather
    # than being absorbed into a warning, so it enforces the guard on its own.
    ext.detect_dominant_plane.side_effect = AssertionError("native path must not run without CUDA")
    monkeypatch.setattr(ground_mesh_module, "_try_load_ground_mesh_extension", lambda: ext)
    monkeypatch.setattr(ground_mesh_module.torch.cuda, "is_available", lambda: False)

    plane, is_inlier, _ = ground_mesh_module.detect_dominant_plane(
        points, normals, distance_threshold=0.1, angle_threshold_deg=10.0, num_hypotheses=4
    )

    ext.detect_dominant_plane.assert_not_called()
    assert np.isclose(abs(float(plane.normal[2])), 1.0)
    assert np.all(is_inlier)


_SMOOTHING_VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]], dtype=np.float32)
_SMOOTHING_EDGES = np.array([[0, 1]], dtype=np.int32)


@pytest.mark.parametrize(
    "exc",
    [
        # Recoverable in principle, but the caller is no longer allowed to decide that silently.
        ground_mesh_module.torch.cuda.OutOfMemoryError("CUDA out of memory"),
        # Version skew: the type an extension built before this kernel existed raises.
        AttributeError("module has no attribute 'smooth_mesh_along_axis'"),
        # A programming error, e.g. a pybind11 signature mismatch.
        TypeError("incompatible function arguments"),
    ],
    ids=["out_of_memory", "missing_kernel", "programming_error"],
)
def test_cuda_fast_path_failure_propagates(exc: Exception, monkeypatch: Any) -> None:
    """A failing CUDA kernel must reach the caller rather than degrading to the slow path.

    Driven through the smoothing entry point because it takes its device as an argument, so the
    extension call is reachable without a GPU. The remaining native fast paths hard-code
    ``device="cuda"`` and share this no-``except`` structure.
    """
    ext = MagicMock()
    ext.smooth_mesh_along_axis.side_effect = exc
    monkeypatch.setattr(ground_mesh_module, "_try_load_ground_mesh_extension", lambda: ext)

    with pytest.raises(type(exc)):
        ground_mesh_module._smooth_mesh_along_axis_torch_cuda(
            _SMOOTHING_VERTICES,
            _SMOOTHING_EDGES,
            axis=2,
            smoothness=1.0,
            num_passes=1,
            device=ground_mesh_module.torch.device("cpu"),
        )


def test_smoothing_uses_torch_path_when_extension_is_absent(monkeypatch: Any) -> None:
    """``ext is None`` is a capability check, not error handling, so a host without the extension
    must still get an answer. Guards against the loud-failure policy regressing CPU-only hosts."""
    monkeypatch.setattr(ground_mesh_module, "_try_load_ground_mesh_extension", lambda: None)

    smoothed = ground_mesh_module._smooth_mesh_along_axis_torch_cuda(
        _SMOOTHING_VERTICES,
        _SMOOTHING_EDGES,
        axis=2,
        smoothness=1.0,
        num_passes=1,
        device=ground_mesh_module.torch.device("cpu"),
    )

    expected = ground_mesh_module._smooth_mesh_along_axis_numpy(_SMOOTHING_VERTICES, _SMOOTHING_EDGES, 2, 1.0, 1)
    np.testing.assert_allclose(smoothed, expected, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_smoothing_without_extension_is_repeatable_on_cuda(monkeypatch: Any) -> None:
    monkeypatch.setattr(ground_mesh_module, "_try_load_ground_mesh_extension", lambda: None)
    rng = np.random.default_rng(4802)
    # A high-degree center amplifies any order-dependent floating-point roundoff.
    vertices = rng.uniform(1.0, 10.0, size=(257, 3)).astype(np.float32)
    edges = np.column_stack([np.zeros(256, dtype=np.int32), np.arange(1, 257, dtype=np.int32)])
    # The NumPy path provides the expected smoothed values independently of CUDA.
    expected = ground_mesh_module._smooth_mesh_along_axis_numpy(vertices, edges, 2, 0.35, 10)
    first = ground_mesh_module._smooth_mesh_along_axis_torch_cuda(vertices, edges, 2, 0.35, 10, torch.device("cuda"))
    np.testing.assert_allclose(first, expected, rtol=1e-6, atol=1e-6)
    # Permutations vary parallel insertion order while preserving the graph itself.
    for _ in range(10):
        repeated = ground_mesh_module._smooth_mesh_along_axis_torch_cuda(
            vertices, rng.permutation(edges), 2, 0.35, 10, torch.device("cuda")
        )
        np.testing.assert_array_equal(first, repeated)


def test_find_ground_compatible_points_uses_numpy_without_extension(monkeypatch: Any) -> None:
    """The numpy filter stays reachable in production whenever the CUDA capability guard declines."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 5.0]], dtype=np.float32)
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(points), 1))
    ground_control_point = np.zeros(3, dtype=np.float32)
    up_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(ground_mesh_module, "_try_load_ground_mesh_extension", lambda: None)

    mask = ground_mesh_module.find_ground_compatible_points(
        points, normals, ground_control_point, ref_normal=up_vector, max_distance=0.5, max_angle_deg=10.0
    )

    expected = ground_mesh_module._find_ground_compatible_points_numpy(
        points, normals, ground_control_point, up_vector, 0.5, 10.0
    )
    np.testing.assert_array_equal(mask, expected)


def test_build_mesh_grid_topology_normal_angle_filter_drops_edge_curtain() -> None:
    points = np.array(
        [
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [2.5, 0.5, 0.0],
            [0.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
            [2.5, 1.5, 10.0],  # road-edge outlier adjacent in BEV
        ],
        dtype=np.float32,
    )
    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=False,
        voxel_size=1.0,
        mesh_topology="grid",
        smoothing_passes=0,
        max_triangle_normal_angle_deg=60.0,
    )

    vertices, triangles, _, _ = mesher.build_mesh_from_points(points)

    assert len(triangles) == 2
    assert vertices[:, 2].max() == 0.0


def test_build_mesh_initial_normal_angle_filter_falls_back_when_empty(caplog) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 10.0],
        ],
        dtype=np.float32,
    )
    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=False,
        smoothing_passes=1,
        max_triangle_normal_angle_deg=60.0,
    )

    caplog.set_level("WARNING", logger=ground_mesh_module.__name__)
    smooth_vertices, triangles, vertices, vertex_colors = mesher.build_mesh_from_points(points)

    assert "Normal-angle triangle filter would empty mesh" in caplog.text
    assert len(smooth_vertices) == 3
    assert len(triangles) == 1
    assert len(vertices) == 3
    assert vertex_colors is None


def test_build_mesh_reapplies_normal_angle_filter_after_smoothing(monkeypatch, caplog) -> None:
    points = np.array(
        [
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [0.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
        ],
        dtype=np.float32,
    )

    def _make_smoothed_mesh_steep(vertices, *_args, **_kwargs):
        smoothed = vertices.copy()
        smoothed[3, 2] = 10.0
        return smoothed

    monkeypatch.setattr(ground_mesh_module, "smooth_mesh_along_axis_torch", _make_smoothed_mesh_steep)
    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=False,
        voxel_size=1.0,
        mesh_topology="grid",
        smoothing_passes=1,
        max_triangle_normal_angle_deg=60.0,
    )

    caplog.set_level("WARNING", logger=ground_mesh_module.__name__)
    smooth_vertices, triangles, vertices, vertex_colors = mesher.build_mesh_from_points(points)

    assert "Post-smoothing normal-angle triangle filter would empty mesh" in caplog.text
    assert len(smooth_vertices) == 4
    assert len(triangles) == 2
    assert len(vertices) == 4
    assert smooth_vertices[:, 2].max() == 10.0
    assert vertex_colors is None


def test_build_mesh_neighbor_average_smoothing_is_repeatable_with_auto_device(monkeypatch) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    selected_devices: list[torch.device | None] = []
    original = ground_mesh_module.smooth_mesh_along_axis_torch

    # Record the forwarded argument while preserving the production device selection.
    def record_device(vertices, triangles, axis=2, smoothness=1.0, num_passes=1, device=None):
        selected_devices.append(device)
        return original(vertices, triangles, axis, smoothness, num_passes, device)

    monkeypatch.setattr(ground_mesh_module, "smooth_mesh_along_axis_torch", record_device)
    mesher = DelaunayElevationMeshingAlgorithm(enable_downsampling=False, smoothing_passes=2)

    # Repeat the full mesher path, where smoothing normally selects its own backend.
    first, first_triangles, _, _ = mesher.build_mesh_from_points(points)
    second, second_triangles, _, _ = mesher.build_mesh_from_points(points)

    # Verify the mesher integration invariants:
    # - backend selection remains automatic
    # - vertices and topology repeat exactly
    # - smoothing preserves horizontal coordinates and produces finite values
    assert selected_devices == [None, None]
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_triangles, second_triangles)
    np.testing.assert_array_equal(first[:, :2], points[:, :2])
    assert np.isfinite(first).all()


def test_smooth_mesh_along_axis_isolated_vertex_preserves_height() -> None:
    """A vertex with no neighbors keeps its original height under smoothing."""
    vertices = np.array(
        [
            [0, 0, 10.0],  # connected
            [1, 0, 20.0],  # connected
            [0, 1, 30.0],  # connected
            [50, 50, 99.0],  # ISOLATED (not part of any triangle)
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    neighbors, distances = get_vertex_neighbors(vertices, triangles)
    out = mesh_smoothing_along_axis(vertices, neighbors, distances, axis=2, smoothness=1.0)
    assert np.isclose(out[3, 2], 99.0, atol=1e-5)


def test_build_mesh_robust_pipeline_e2e() -> None:
    """End-to-end check: median + outlier filter + long-edge filter combine without crashing
    and produce a non-empty mesh."""
    rng = np.random.RandomState(0)
    # Dense grid of road points at z=0, plus a single spike point and a stray far-away point.
    xs = np.linspace(0.0, 5.0, 60)
    ys = np.linspace(0.0, 5.0, 60)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), rng.uniform(-0.02, 0.02, xx.size)])
    spike = np.array([[2.5, 2.5, 3.0]])  # local spike, will get medianed/filtered
    stray = np.array([[100.0, 100.0, 0.0]])  # creates long bridging triangle
    points = np.vstack([grid_points, spike, stray]).astype(np.float32)

    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=0.5,
        height_statistic="median",
        outlier_height_threshold=0.3,
        max_triangle_edge_length=2.0,
        smoothing_passes=2,
    )
    smooth_vertices, triangles, _, _ = mesher.build_mesh_from_points(points)

    assert len(smooth_vertices) > 0
    assert len(triangles) > 0
    # Stray point at (100, 100) should have been removed by long-edge filter (no neighbor closer than 95m).
    max_xy = smooth_vertices[:, :2].max(axis=0)
    assert max_xy[0] < 50.0
    assert max_xy[1] < 50.0
    # No surviving vertex should be at the spike height.
    assert smooth_vertices[:, 2].max() < 1.0


def test_build_mesh_returns_empty_for_too_few_vertices() -> None:
    """If the outlier filter empties the point cloud, build_mesh_from_points returns an empty mesh
    instead of crashing in Delaunay."""
    # Two flat neighbors + one extreme outlier; with an aggressive threshold and high min_points,
    # everything gets filtered.
    points = np.array(
        [
            [0.5, 0.5, 0.0],
            [50.0, 50.0, 99.0],
        ],
        dtype=np.float32,
    )
    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=1.0,
        min_points_per_voxel=5,  # both cells have <5 points -> all filtered
        smoothing_passes=0,
    )
    smooth_vertices, triangles, _, _ = mesher.build_mesh_from_points(points)
    assert len(triangles) == 0
    assert len(smooth_vertices) == 0


def test_build_mesh_collinear_points_returns_empty() -> None:
    """Three or more collinear points pass the <3 count guard but make Delaunay throw QhullError;
    the catch in build_mesh_from_points should turn that into an empty-mesh return."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mesher = DelaunayElevationMeshingAlgorithm(enable_downsampling=False, smoothing_passes=0)
    smooth_vertices, triangles, _, _ = mesher.build_mesh_from_points(points)
    assert len(triangles) == 0
    assert len(smooth_vertices) == 0


def test_build_mesh_long_edge_filter_falls_back_when_empty(caplog) -> None:
    """If every Delaunay triangle exceeds max_triangle_edge_length, keep the original mesh."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ],
        dtype=np.float32,
    )
    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=False,
        mesh_topology="delaunay",
        max_triangle_edge_length=1.0,  # all edges are 10m+, drops everything
        smoothing_passes=2,
    )

    caplog.set_level("WARNING", logger=ground_mesh_module.__name__)
    smooth_vertices, triangles, vertices, vertex_colors = mesher.build_mesh_from_points(points)

    assert "Long-edge triangle filter would empty mesh" in caplog.text
    assert len(smooth_vertices) == 3
    assert len(triangles) == 1
    assert len(vertices) == 3
    assert vertex_colors is None


# --- Tests for get_nominal_ground_point_under_lidar ---


def _make_mock_datasource(sensor_height: float) -> MagicMock:
    """Create a mock NCOREDataSource with a single lidar at the given Z height in rig frame."""
    T_sensor_rig = np.eye(4, dtype=np.float32)
    T_sensor_rig[2, 3] = sensor_height  # lidar Z in rig frame

    lidar_sensor = MagicMock()
    lidar_sensor.T_sensor_rig = T_sensor_rig

    datasource = MagicMock()
    datasource.lidar_sensors = {"lidar_top": lidar_sensor}
    return datasource


def test_get_nominal_ground_point_under_lidar_default_z_offset() -> None:
    """With default rig_z_offset=0, ground point Z = -sensor_height."""
    sensor_height = 1.87
    datasource = _make_mock_datasource(sensor_height)

    ground_point = get_nominal_ground_point_under_lidar(datasource)

    assert ground_point.shape == (3,)
    assert ground_point.dtype == np.float32
    assert np.isclose(ground_point[0], 0.0)
    assert np.isclose(ground_point[1], 0.0)
    assert np.isclose(ground_point[2], -sensor_height)


def test_get_nominal_ground_point_under_lidar_with_z_offset() -> None:
    """With non-zero rig_z_offset, ground point Z = rig_z_offset - sensor_height."""
    sensor_height = 1.87
    rig_z_offset = -0.3  # ground is 30cm below rig origin
    datasource = _make_mock_datasource(sensor_height)

    ground_point = get_nominal_ground_point_under_lidar(datasource, rig_z_offset=rig_z_offset)

    expected_z = rig_z_offset - sensor_height  # -0.3 - 1.87 = -2.17
    assert np.isclose(ground_point[2], expected_z)
    assert np.isclose(ground_point[0], 0.0)
    assert np.isclose(ground_point[1], 0.0)


def test_get_nominal_ground_point_under_lidar_positive_z_offset() -> None:
    """Positive rig_z_offset (ground above rig origin) shifts ground point up."""
    sensor_height = 2.0
    rig_z_offset = 0.5
    datasource = _make_mock_datasource(sensor_height)

    ground_point = get_nominal_ground_point_under_lidar(datasource, rig_z_offset=rig_z_offset)

    expected_z = 0.5 - 2.0  # -1.5
    assert np.isclose(ground_point[2], expected_z)


def test_get_nominal_ground_point_under_lidar_uses_selected_lidar() -> None:
    """Multiple LiDAR datasets use the requested sensor as the ground reference."""
    lidar_front = _make_mock_datasource(sensor_height=2.5).lidar_sensors["lidar_top"]
    lidar_top = _make_mock_datasource(sensor_height=1.5).lidar_sensors["lidar_top"]
    datasource = MagicMock()
    datasource.lidar_sensors = {"lidar_front": lidar_front, "lidar_top": lidar_top}

    ground_point = get_nominal_ground_point_under_lidar(datasource, lidar_id="lidar_top")

    np.testing.assert_array_equal(ground_point, np.array([0.0, 0.0, -1.5], dtype=np.float32))


def test_get_nominal_ground_point_under_lidar_defaults_to_first_sorted_lidar() -> None:
    """The default ground reference matches the datasource's sorted sensor order."""
    lidar_front = _make_mock_datasource(sensor_height=2.5).lidar_sensors["lidar_top"]
    lidar_top = _make_mock_datasource(sensor_height=1.5).lidar_sensors["lidar_top"]
    datasource = MagicMock()
    datasource.lidar_sensors = {"lidar_front": lidar_front, "lidar_top": lidar_top}

    ground_point = get_nominal_ground_point_under_lidar(datasource)

    np.testing.assert_array_equal(ground_point, np.array([0.0, 0.0, -2.5], dtype=np.float32))


def test_get_nominal_ground_point_under_lidar_non_affine_raises() -> None:
    """Should raise ValueError when T_sensor_rig is not affine."""
    T_bad = np.eye(4, dtype=np.float32)
    T_bad[3, 0] = 1.0  # break affine assumption

    lidar_sensor = MagicMock()
    lidar_sensor.T_sensor_rig = T_bad

    datasource = MagicMock()
    datasource.lidar_sensors = {"lidar_top": lidar_sensor}

    try:
        get_nominal_ground_point_under_lidar(datasource)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "affine" in str(e).lower()


def _make_synthetic_spin_points(n_rows: int = 32, n_az: int = 720, sensor_height: float = 2.0) -> np.ndarray:
    """Synthetic single-sensor LiDAR returns on a wavy ground surface.

    Each row has a fixed elevation; rays sweep through ``n_az`` azimuth
    samples; the returned point is where the ray hits a gentle terrain
    surface ``z = 0.4 sin(0.07 x) cos(0.05 y)``. Mirrors the shape of real
    automotive spinning LiDAR data without needing a recorded clip.
    """
    sensor = np.array([0.0, 0.0, sensor_height], dtype=np.float32)
    elevations = np.linspace(-np.pi / 8, -0.02, n_rows, dtype=np.float32)
    azimuths = np.linspace(-np.pi, np.pi, n_az, endpoint=False, dtype=np.float32)
    el_grid, az_grid = np.meshgrid(elevations, azimuths, indexing="ij")
    el_flat = el_grid.ravel()
    az_flat = az_grid.ravel()
    # Range to a horizontal terrain at z=0 (small wave added below).
    dist = np.clip(sensor_height / np.maximum(-np.sin(el_flat), 0.05), 1.0, 80.0)
    ends = np.stack(
        [
            dist * np.cos(el_flat) * np.cos(az_flat),
            dist * np.cos(el_flat) * np.sin(az_flat),
            sensor[2] + dist * np.sin(el_flat),
        ],
        axis=1,
    ).astype(np.float32)
    ends[:, 2] += (0.4 * np.sin(0.07 * ends[:, 0]) * np.cos(0.05 * ends[:, 1])).astype(np.float32)
    return ends - sensor


def test_construct_lidar_spin_mesh_returns_consistent_arrays() -> None:
    """Output shape contract: triangles index into points; ``azimuth_elevation`` shape mirrors input."""
    points = _make_synthetic_spin_points(n_rows=16, n_az=180)
    triangles, az_el = construct_lidar_spin_mesh(points, n_elevation_bins=32, n_azimuth_bins=512)
    assert triangles.dtype == np.int32
    assert triangles.ndim == 2 and triangles.shape[1] == 3
    assert az_el.shape == (points.shape[0], 3)
    if triangles.size > 0:
        assert int(triangles.min()) >= 0
        assert int(triangles.max()) < points.shape[0]


def test_construct_lidar_spin_mesh_dense_grid_emits_all_quads() -> None:
    """A perfectly regular ``n_rows × n_az`` scan with one point per cell should
    yield exactly ``2 * (n_rows - 1) * (n_az - 1)`` triangles when the bin
    grid matches the scan grid. Topological-completeness check for the
    scanline mesher."""
    n_rows, n_az = 8, 32
    # Pure (az, el) grid with no terrain perturbation, so each point's
    # recovered (az, el) under ``xyz_to_spherical`` falls exactly into its
    # own bin. Azimuths are shifted off the ``-π``/``+π`` seam by half a
    # bin so ``arctan2``'s seam behavior (``arctan2(0, -1) == +π``, never
    # ``-π``) does not collapse the first/last column into the same bin.
    elevations = np.linspace(-np.pi / 8, -0.02, n_rows, dtype=np.float32)
    az_step = 2.0 * np.pi / n_az
    azimuths = (np.arange(n_az, dtype=np.float32) * az_step - np.pi + az_step * 0.5).astype(np.float32)
    el_grid, az_grid = np.meshgrid(elevations, azimuths, indexing="ij")
    el_flat = el_grid.ravel()
    az_flat = az_grid.ravel()
    # Unit-radius points on the (az, el) sphere — keeps ``xyz_to_spherical``
    # round-trips numerically clean.
    points = np.stack(
        [
            np.cos(el_flat) * np.cos(az_flat),
            np.cos(el_flat) * np.sin(az_flat),
            np.sin(el_flat),
        ],
        axis=1,
    ).astype(np.float32)

    triangles, _ = construct_lidar_spin_mesh(
        points,
        n_elevation_bins=n_rows,
        n_azimuth_bins=n_az,
    )
    expected = 2 * (n_rows - 1) * (n_az - 1)
    assert triangles.shape[0] == expected, f"got {triangles.shape[0]} triangles, expected exactly {expected}"


def test_compute_mesh_normals_in_lidar_spin_gives_upward_normals_on_flat_ground() -> None:
    """On a near-horizontal patch, vertex normals should mostly point up after
    the ``flip_normals_towards_viewpoint`` step (the sensor sits above the
    surface at z = +2 m). This validates that the scanline mesher produces
    a usable triangulation for downstream pcu normal estimation."""
    n_rows, n_az = 32, 480
    points = _make_synthetic_spin_points(n_rows=n_rows, n_az=n_az)
    # Bin counts sized to this synthetic scene's scan density. Production
    # callers use the defaults (256 × 4096) tuned to real LiDAR resolutions.
    normals, triangles = compute_mesh_normals_in_lidar_spin(
        points,
        verbose=False,
        n_elevation_bins=n_rows,
        n_azimuth_bins=n_az,
    )
    assert normals.shape == points.shape
    assert triangles.ndim == 2 and triangles.shape[1] == 3
    # The point at the sensor centerline shouldn't have a degenerate normal.
    # Require a clear majority of normals to have positive Z (pointing up toward
    # the sensor, which is above the surface).
    up_fraction = float(np.mean(normals[:, 2] > 0.5))
    assert up_fraction > 0.7, f"only {up_fraction:.2%} of normals point up; expected > 70%"


def test_construct_lidar_spin_mesh_auto_detects_bin_counts() -> None:
    """When ``n_elevation_bins`` / ``n_azimuth_bins`` are left as ``None``,
    the mesher must size its grid to the input scan structure rather than
    to a fixed default — otherwise a typical 128-line LiDAR with default
    bins denser than the scan grid would land every other elevation bin
    empty and emit zero triangles."""
    n_rows, n_az = 128, 1800
    elevations = np.linspace(-np.pi / 8, -0.02, n_rows, dtype=np.float32)
    az_step = 2.0 * np.pi / n_az
    azimuths = (np.arange(n_az, dtype=np.float32) * az_step - np.pi + az_step * 0.5).astype(np.float32)
    el_grid, az_grid = np.meshgrid(elevations, azimuths, indexing="ij")
    el_flat = el_grid.ravel()
    az_flat = az_grid.ravel()
    points = np.stack(
        [
            np.cos(el_flat) * np.cos(az_flat),
            np.cos(el_flat) * np.sin(az_flat),
            np.sin(el_flat),
        ],
        axis=1,
    ).astype(np.float32)

    triangles, _ = construct_lidar_spin_mesh(points)  # auto-detect
    assert triangles.shape[0] > 0, (
        "auto-detect must size the grid to the actual scan structure; "
        "got 0 triangles which means the bin grid was denser than the scan grid"
    )
    interior_quads = (n_rows - 1) * (n_az - 1)
    assert triangles.shape[0] >= int(0.5 * 2 * interior_quads), (
        f"auto-detect produced suspiciously few triangles: "
        f"got {triangles.shape[0]}, expected at least {int(0.5 * 2 * interior_quads)} "
        f"on a clean {n_rows}×{n_az} scan grid"
    )


def test_compute_mesh_normals_in_lidar_spin_torch_matches_numpy() -> None:
    """The torch path must reproduce the CPU path within float tolerance.

    Equivalence is asserted on a synthetic ``(az, el)`` grid (no terrain
    perturbation, unit-radius points). On such input every bin is filled
    exactly once and the binning step is well-conditioned: both backends
    produce the same triangle list. Normal-vector tolerance is set to
    ``1e-3`` on per-component max abs diff."""
    import torch as _torch

    n_rows, n_az = 24, 360
    elevations = np.linspace(-np.pi / 8, -0.02, n_rows, dtype=np.float32)
    az_step = 2.0 * np.pi / n_az
    azimuths = (np.arange(n_az, dtype=np.float32) * az_step - np.pi + az_step * 0.5).astype(np.float32)
    el_grid, az_grid = np.meshgrid(elevations, azimuths, indexing="ij")
    el_flat = el_grid.ravel()
    az_flat = az_grid.ravel()
    points = np.stack(
        [
            np.cos(el_flat) * np.cos(az_flat),
            np.cos(el_flat) * np.sin(az_flat),
            np.sin(el_flat),
        ],
        axis=1,
    ).astype(np.float32)

    cpu_normals, cpu_triangles = compute_mesh_normals_in_lidar_spin(
        points, verbose=False, n_elevation_bins=n_rows, n_azimuth_bins=n_az, device="cpu"
    )

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    points_t = _torch.as_tensor(points, dtype=_torch.float32, device=device)
    torch_normals_t, torch_triangles_t = compute_mesh_normals_in_lidar_spin_torch(
        points_t, n_elevation_bins=n_rows, n_azimuth_bins=n_az
    )
    torch_normals = torch_normals_t.detach().cpu().numpy()
    torch_triangles = torch_triangles_t.detach().cpu().numpy()

    assert torch_triangles.shape == cpu_triangles.shape, (
        f"triangle counts differ on clean grid: numpy={cpu_triangles.shape}, torch={torch_triangles.shape}"
    )
    assert np.array_equal(torch_triangles, cpu_triangles), (
        "triangle lists must match exactly between numpy and torch paths on a clean grid"
    )

    diff = np.abs(torch_normals - cpu_normals).max()
    assert diff < 1e-3, f"normal max abs diff between numpy and torch = {diff:.3e}, expected < 1e-3"


def test_compute_mesh_normals_in_lidar_spin_torch_flat_ground_quality() -> None:
    """End-to-end quality check for the torch path on noisy data: on a
    near-horizontal patch with terrain perturbation, the torch path should
    produce mostly-upward normals."""
    import torch as _torch

    n_rows, n_az = 32, 480
    points = _make_synthetic_spin_points(n_rows=n_rows, n_az=n_az)
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    points_t = _torch.as_tensor(points, dtype=_torch.float32, device=device)
    normals_t, _ = compute_mesh_normals_in_lidar_spin_torch(points_t, n_elevation_bins=n_rows, n_azimuth_bins=n_az)
    normals = normals_t.detach().cpu().numpy()
    up_fraction = float(np.mean(normals[:, 2] > 0.5))
    assert up_fraction > 0.7, f"only {up_fraction:.2%} of torch normals point up; expected > 70%"


def test_compute_mesh_normals_in_lidar_spin_cuda_dispatch_returns_numpy() -> None:
    """``device='cuda'`` returns numpy arrays so existing call sites are
    unaffected. Falls back to CPU when CUDA is unavailable."""
    n_rows, n_az = 16, 180
    points = _make_synthetic_spin_points(n_rows=n_rows, n_az=n_az)
    normals, triangles = compute_mesh_normals_in_lidar_spin(
        points, verbose=False, n_elevation_bins=n_rows, n_azimuth_bins=n_az, device="cuda"
    )
    assert isinstance(normals, np.ndarray)
    assert isinstance(triangles, np.ndarray)
    assert normals.shape == points.shape
    assert triangles.ndim == 2 and triangles.shape[1] == 3
    assert normals.dtype == np.float32
    assert triangles.dtype == np.int32


def _polygon_area_xy(pts: np.ndarray) -> float:
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    return float(0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _triangulated_area_xy(pts: np.ndarray, tris: np.ndarray) -> float:
    p = pts.astype(np.float64)
    total = 0.0
    for a, b, c in tris:
        total += 0.5 * abs((p[b, 0] - p[a, 0]) * (p[c, 1] - p[a, 1]) - (p[b, 1] - p[a, 1]) * (p[c, 0] - p[a, 0]))
    return total


def test_find_boundary_loops_finds_outer_and_interior_hole() -> None:
    # Square annulus: outer corners 0..3, inner hole corners 4..7, triangulated ring.
    triangles = np.array(
        [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32
    )
    loops = find_boundary_loops(triangles)
    loop_vertex_sets = sorted(tuple(sorted(set(loop))) for loop in loops)
    assert loop_vertex_sets == [(0, 1, 2, 3), (4, 5, 6, 7)]


def test_find_boundary_loops_empty() -> None:
    assert find_boundary_loops(np.zeros((0, 3), dtype=np.int32)) == []


def test_triangulate_simple_polygon_convex_square() -> None:
    pts = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
    tris = _triangulate_simple_polygon_xy(pts)
    assert len(tris) == 2  # n - 2
    assert np.isclose(_triangulated_area_xy(pts, tris), _polygon_area_xy(pts))


def test_triangulate_simple_polygon_concave_L_shape() -> None:
    pts = np.array([[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]], dtype=np.float32)
    tris = _triangulate_simple_polygon_xy(pts)
    assert len(tris) == 4  # n - 2
    # Exact interior area (7.0) -> no over-fill across the concave notch, no spurious tris.
    assert np.isclose(_triangulated_area_xy(pts, tris), _polygon_area_xy(pts))


def test_triangulate_simple_polygon_world_space_precision() -> None:
    # World-space absolute coordinates (~1e5): float32 orientation math would collapse to a
    # zero-area degeneracy and fill nothing; the local-shift + float64 must still triangulate.
    base = 100000.0
    pts = np.array(
        [[base + 2, base + 1], [base + 1, base + 1], [base + 1, base + 2], [base + 2, base + 2]], dtype=np.float32
    )
    tris = _triangulate_simple_polygon_xy(pts)
    assert len(tris) == 2
    assert np.isclose(_triangulated_area_xy(pts, tris), 1.0)


def test_triangulate_simple_polygon_drops_consecutive_duplicate_vertex() -> None:
    pts = np.array([[0, 0], [1, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)  # vertex (1,0) repeated
    tris = _triangulate_simple_polygon_xy(pts)
    assert np.isclose(_triangulated_area_xy(pts, tris), 1.0)


def test_triangulate_simple_polygon_degenerate_returns_empty() -> None:
    assert len(_triangulate_simple_polygon_xy(np.array([[0, 0], [1, 0]], dtype=np.float32))) == 0


def test_triangulate_simple_polygon_non_simple_loop_returns_empty(caplog) -> None:
    # A pinched/self-touching loop is not a simple polygon: ear clipping cannot consume it
    # fully. It must warn AND return nothing -- a partial fan would leave invalid topology
    # for densify/smoothing, so the whole loop is discarded rather than half-filled.
    pts = np.array([[0, 0], [2, 0], [1, 1], [2, 2], [0, 2], [1, 1]], dtype=np.float32)
    with caplog.at_level("WARNING"):
        tris = _triangulate_simple_polygon_xy(pts)
    assert len(tris) == 0
    assert any("non-simple" in rec.message for rec in caplog.records)


def test_fill_interior_boundary_loops_per_component_keeps_patch_outlines() -> None:
    # Two disjoint square patches (no shared edges). Each patch's outer boundary is its own
    # component's outer hull -- neither should be triangulated as a "hole". A single global
    # largest-loop rule would wrongly fill the smaller patch's outline.
    vertices = np.array(
        [
            [0, 0, 0],
            [4, 0, 0],
            [4, 4, 0],
            [0, 4, 0],  # big patch (verts 0..3)
            [10, 10, 0],
            [11, 10, 0],
            [11, 11, 0],
            [10, 11, 0],  # small patch (verts 4..7)
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int32)
    out = fill_interior_boundary_loops(vertices, triangles)
    # Neither patch has an interior hole, so nothing is added.
    assert len(out) == len(triangles)


def test_fill_interior_boundary_loops_fills_hole_in_one_of_two_components() -> None:
    # Two disjoint components; only the first has an interior hole (square annulus). The hole
    # must be filled while the hole-free second patch is left untouched.
    vertices = np.array(
        [
            [0, 0, 0],
            [3, 0, 0],
            [3, 3, 0],
            [0, 3, 0],  # annulus outer (0..3)
            [1, 1, 0],
            [2, 1, 0],
            [2, 2, 0],
            [1, 2, 0],  # annulus inner hole (4..7)
            [10, 0, 0],
            [11, 0, 0],
            [11, 1, 0],  # separate solid triangle (8..10)
        ],
        dtype=np.float32,
    )
    annulus = np.array(
        [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32
    )
    solid = np.array([[8, 9, 10]], dtype=np.int32)
    triangles = np.concatenate([annulus, solid], axis=0)
    out = fill_interior_boundary_loops(vertices, triangles)
    # The 4-vertex inner hole adds n-2 = 2 triangles; the solid patch adds none.
    assert len(out) == len(triangles) + 2
    # The added triangles must close the INNER hole (verts 4..7), not the outer ring (0..3)
    # or the solid patch (8..10) -- counts alone would not catch a wrong-loop classification.
    added = out[len(triangles) :]
    assert set(added.ravel().tolist()) <= {4, 5, 6, 7}


def test_fill_interior_boundary_loops_world_coords_fills_hole_not_outer() -> None:
    # Single annulus at world-space coords (~1e5): the outer ring (area 100) must beat the
    # inner hole (area 16) when picking the outer hull. A float32 shoelace area collapses both
    # to the same value here, so this locks the float64 area computation -- without it the
    # exterior can be triangulated as the "hole".
    base = 100000.0
    vertices = np.array(
        [
            [base, base, 0],
            [base + 10, base, 0],
            [base + 10, base + 10, 0],
            [base, base + 10, 0],  # outer
            [base + 3, base + 3, 0],
            [base + 7, base + 3, 0],
            [base + 7, base + 7, 0],
            [base + 3, base + 7, 0],  # hole
        ],
        dtype=np.float32,
    )
    triangles = np.array(
        [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32
    )
    out = fill_interior_boundary_loops(vertices, triangles)
    assert len(out) == len(triangles) + 2
    added = out[len(triangles) :]
    assert set(added.ravel().tolist()) <= {4, 5, 6, 7}


def test_keep_largest_connected_component_two_patches() -> None:
    # Big patch (2 tris) + small patch (1 tri), no shared edges -> only the big patch kept.
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int32)
    keep = keep_largest_connected_component(triangles)
    assert keep.tolist() == [True, True, False]


# --- Tests for generate_ground_points_from_cuboids ---


def _make_cuboid_tracks_stationary(n_tracks: int = 1) -> CuboidTracks:
    """Build n_tracks cuboid tracks, each with 2 poses at the same stationary position.

    Track i has both poses at (i*5, 0, 0) — tracks are spaced 5 m apart along x, but each
    individual track does not move (both poses share the same world position). All tracks are
    flagged DYNAMIC and have dims [2.0, 1.0, 1.5] m (length × width × height). No other
    flags are set.
    """
    cpu = torch.device("cpu")
    tracks_id = [f"track_{i}" for i in range(n_tracks)]
    poses = [np.eye(4, dtype=np.float32)[np.newaxis].repeat(2, axis=0) for _ in range(n_tracks)]
    for i, T in enumerate(poses):
        T[:, 0, 3] = float(i * 5)
    timestamps = [np.array([0, 1], dtype=np.int64) for _ in range(n_tracks)]
    label_class = ["vehicle"] * n_tracks
    flags_list = [TrackFlags.DYNAMIC] * n_tracks
    dims = [np.array([2.0, 1.0, 1.5], dtype=np.float32) for _ in range(n_tracks)]
    return CuboidTracks.Factory.from_numpy(tracks_id, poses, timestamps, label_class, flags_list, dims, device=cpu)


def test_generate_ground_points_from_cuboids_returns_correct_shape() -> None:
    n_tracks = 3
    cuboid_tracks = _make_cuboid_tracks_stationary(n_tracks=n_tracks)
    pts = generate_ground_points_from_cuboids(cuboid_tracks, num_samples=4, cuboid_point_merge_cell_size=0.1)
    # The two identical poses per track are merged into the same 4x4 occupied cells.
    assert pts.ndim == 2
    assert pts.shape[1] == 3
    assert pts.dtype == np.float32
    expected_n = n_tracks * 4 * 4
    assert pts.shape[0] == expected_n


def test_generate_ground_points_from_cuboids_unfiltered_returns_source_track_indices() -> None:
    n_tracks = 3
    cuboid_tracks = _make_cuboid_tracks_stationary(n_tracks=n_tracks)

    points, track_indices = _generate_ground_points_from_cuboids_unfiltered(
        cuboid_tracks,
        num_samples=4,
        cuboids_base_margin=0.0,
    )

    samples_per_track = 2 * 4 * 4  # 2 poses x 4x4 samples per pose
    assert points.shape == (n_tracks * samples_per_track, 3)
    np.testing.assert_array_equal(track_indices, np.repeat(np.arange(n_tracks), samples_per_track))


def test_generate_ground_points_from_cuboids_base_margin() -> None:
    # With dims [2, 1, 1.5] m and a 0.5 m margin, the footprint spans [-1.5, 1.5]
    # along X and [-1.0, 1.0] along Y, while its base remains at z = -0.75.
    cuboid_tracks = _make_cuboid_tracks_stationary(n_tracks=1)
    pts = generate_ground_points_from_cuboids(
        cuboid_tracks, num_samples=3, cuboids_base_margin=0.5, cuboid_point_merge_cell_size=0.1
    )
    xy_lower = np.array([-1.5, -1.0])
    xy_upper = np.array([1.5, 1.0])

    # All samples remain at the expected cuboid base height.
    np.testing.assert_allclose(pts[:, 2], -0.75, atol=1e-5)
    # All samples lie within the margin-expanded footprint.
    assert np.all(pts[:, :2] >= xy_lower - 1e-5)
    assert np.all(pts[:, :2] <= xy_upper + 1e-5)
    # Samples reach both ends of each axis and therefore span the full footprint.
    np.testing.assert_allclose(np.ptp(pts[:, :2], axis=0), xy_upper - xy_lower, atol=1e-5)


def test_generate_ground_points_from_cuboids_1nn_filter() -> None:
    cuboid_tracks = _make_cuboid_tracks_stationary()
    # Place one scanned point 0.01 m from the cuboid sample at [-1, -0.5, -0.75]. It is inside that
    # sample's 0.1 m exclusion radius and outside the radius of the other three cuboid samples.
    selected_cuboid_point = np.array([-1.0, -0.5, -0.75], dtype=np.float32)
    scanned_ground_points = selected_cuboid_point[None] + np.array([[0.01, 0.0, 0.0]], dtype=np.float32)

    # Generate 2x2 cuboid samples. With k=1, the nearby scanned point should remove only the selected sample.
    filtered_points = generate_ground_points_from_cuboids(
        cuboid_tracks,
        scanned_ground_points=scanned_ground_points,
        num_samples=2,
        cuboids_base_margin=0.0,
        exclude_points_radius=0.1,
        exclude_points_knns=1,
    )
    assert len(filtered_points) == 3
    expected_points = np.array(
        [
            [-1.0, 0.5, -0.75],
            [1.0, -0.5, -0.75],
            [1.0, 0.5, -0.75],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(filtered_points, expected_points)


def test_generate_ground_points_from_cuboids_2nn_keep() -> None:
    cuboid_tracks = _make_cuboid_tracks_stationary()
    # Place one scanned point inside and one outside the 0.1 m radius of the cuboid sample at
    # [-1, -0.5, -0.75]. Both scanned points are outside the radius of the other three samples.
    selected_cuboid_point = np.array([-1.0, -0.5, -0.75], dtype=np.float32)
    scanned_ground_points_one_outside = selected_cuboid_point + np.array(
        [
            [-0.01, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Generate 2x2 cuboid samples. With k=2, one scanned point outside the radius should retain all samples.
    retained_points = generate_ground_points_from_cuboids(
        cuboid_tracks,
        scanned_ground_points=scanned_ground_points_one_outside,
        num_samples=2,
        cuboids_base_margin=0.0,
        exclude_points_radius=0.1,
        exclude_points_knns=2,
    )
    assert len(retained_points) == 4
    expected_points = np.array(
        [
            [-1.0, -0.5, -0.75],
            [-1.0, 0.5, -0.75],
            [1.0, -0.5, -0.75],
            [1.0, 0.5, -0.75],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(retained_points, expected_points)


def test_generate_ground_points_from_cuboids_2nn_filter() -> None:
    cuboid_tracks = _make_cuboid_tracks_stationary()

    # Place two scanned points inside the 0.1 m radius of the cuboid sample at [-1, -0.5, -0.75].
    # Both scanned points are outside the radius of the other three cuboid samples.
    selected_cuboid_point = np.array([-1.0, -0.5, -0.75], dtype=np.float32)
    scanned_ground_points_both_inside = selected_cuboid_point + np.array(
        [
            [-0.01, 0.0, 0.0],
            [0.01, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Generate 2x2 cuboid samples. With k=2, both scanned points inside the radius should remove only that sample.
    filtered_points = generate_ground_points_from_cuboids(
        cuboid_tracks,
        scanned_ground_points=scanned_ground_points_both_inside,
        num_samples=2,
        cuboids_base_margin=0.0,
        exclude_points_radius=0.1,
        exclude_points_knns=2,
    )
    assert len(filtered_points) == 3
    expected_points = np.array(
        [
            [-1.0, 0.5, -0.75],
            [1.0, -0.5, -0.75],
            [1.0, 0.5, -0.75],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(filtered_points, expected_points)


def test_generate_ground_points_from_cuboids_insufficient_nns(caplog: Any) -> None:
    cuboid_tracks = _make_cuboid_tracks_stationary()
    selected_cuboid_point = np.array([-1.0, -0.5, -0.75], dtype=np.float32)
    scanned_ground_points = selected_cuboid_point[None] + np.array([[0.01, 0.0, 0.0]], dtype=np.float32)

    # Request k=2 with only one nearby scanned point. The function should skip filtering because
    # the second-nearest scanned point does not exist, retaining all four cuboid samples.
    retained_points = generate_ground_points_from_cuboids(
        cuboid_tracks,
        scanned_ground_points=scanned_ground_points,
        num_samples=2,
        cuboids_base_margin=0.0,
        exclude_points_radius=0.1,
        exclude_points_knns=2,
    )
    assert len(retained_points) == 4
    expected_points = np.array(
        [
            [-1.0, -0.5, -0.75],
            [-1.0, 0.5, -0.75],
            [1.0, -0.5, -0.75],
            [1.0, 0.5, -0.75],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(retained_points, expected_points)
    assert (
        "Skipping cuboid ground point filtering: requested 2 nearest scanned points, but only 1 available"
        in caplog.text
    )
