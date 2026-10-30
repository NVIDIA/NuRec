# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import numpy as np
import torch

from nurec.benchmark.eval_ground_mesh import (
    _rasterize_points,
    blend_colorized_masks,
    calculate_cuboid_coverage_by_ground_mesh,
    calculate_ground_mesh_metrics,
    calculate_point_cloud_hull_coverage_metrics,
    calculate_vehicle_trajectory_vs_ground_consistency_metrics,
    generate_track_colors,
)
from nurec.datasets.tracks import CuboidTracks
from nurec.utils.types import TrackFlags


def _flat_square_mesh():
    vertices = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return vertices, faces


def _linear_trajectory(n: int = 20) -> np.ndarray:
    # 20 samples along the x axis: x = 5.01, 6.01, ..., 24.01, y = 5, z = 0.2.
    # The +0.01 x-offset keeps all samples away from mesh boundaries (diagonal at x=y and right edge at x=10).
    # Exactly 5 samples (x=5.01..9.01) fall inside the [0,10]×[0,10] mesh → 25% trajectory coverage.
    x = np.arange(n, dtype=np.float32) + 5.01
    y = np.full(n, 5.0, dtype=np.float32)
    z = np.full(n, 0.2, dtype=np.float32)
    positions = np.stack([x, y, z], axis=1)
    identity_rot = np.eye(3, dtype=np.float32)
    T = np.zeros((n, 4, 4), dtype=np.float32)
    T[:, :3, :3] = identity_rot
    T[:, :3, 3] = positions
    T[:, 3, 3] = 1.0
    return T


class TestRasterizePoints(unittest.TestCase):
    def test_basic_coverage(self):
        # 3x3 grid: xs=[0,1,2], ys=[2,1,0] (Y decreases down rows — row 0 = max Y, row 2 = min Y).
        n_xs, n_ys = 3, 3
        xs = np.arange(n_xs, dtype=np.float32)
        ys = np.arange(n_ys - 1, -1, -1, dtype=np.float32)  # [2, 1, 0]
        gx, gy = np.meshgrid(xs, ys)
        grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

        # Point at (1, 1) → col=1, row=1 (y_max=2, row=(2-1)/1=1) → flat index 1*3+1=4.
        points = np.array([[1.0, 1.0]], dtype=np.float32)
        mask = _rasterize_points(points, grid_pts, 1.0)
        self.assertEqual(mask.shape, (9,))
        self.assertTrue(mask[4])  # cell (row=1, col=1) = world (1, 1)
        self.assertEqual(mask.sum(), 1)

    def test_empty_points(self):
        xs = np.arange(3, dtype=np.float32)
        ys = np.arange(2, -1, -1, dtype=np.float32)  # [2, 1, 0]
        gx, gy = np.meshgrid(xs, ys)
        grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
        mask = _rasterize_points(np.zeros((0, 2), dtype=np.float32), grid_pts, 1.0)
        self.assertFalse(mask.any())


class TestBlendColorizedMasks(unittest.TestCase):
    def test_output_shape_and_dtype(self):
        h, w = 4, 4
        m1 = np.ones((h, w), dtype=bool)
        m2 = np.zeros((h, w), dtype=bool)
        result = blend_colorized_masks([m1, m2], [(255, 0, 0), (0, 255, 0)], [1.0, 1.0])
        self.assertEqual(result.shape, (h, w, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_pixel_range(self):
        h, w = 8, 8
        mask = np.zeros((h, w), dtype=bool)
        mask[: h // 2, :] = True  # top half covered, bottom half uncovered
        result = blend_colorized_masks([mask], [(100, 150, 200)], [1.0])
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 255)

    def test_background_filled_for_uncovered(self):
        h, w = 4, 4
        mask = np.zeros((h, w), dtype=bool)
        bg = (128, 64, 32)
        result = blend_colorized_masks([mask], [(255, 0, 0)], [1.0], background_color=bg)
        np.testing.assert_array_equal(result[0, 0], bg)


class TestGenerateTrackColors(unittest.TestCase):
    def test_deterministic(self):
        ids = ["track_a", "track_b"]
        colors1 = generate_track_colors(ids)
        colors2 = generate_track_colors(ids)
        np.testing.assert_array_equal(colors1["track_a"], colors2["track_a"])
        np.testing.assert_array_equal(colors1["track_b"], colors2["track_b"])

    def test_shape_and_dtype(self):
        colors = generate_track_colors(["x"])
        self.assertEqual(colors["x"].shape, (3,))
        self.assertEqual(colors["x"].dtype, np.uint8)


class TestCalculatePointCloudHullCoverageMetrics(unittest.TestCase):
    def setUp(self):
        self.mesh_vertices, self.mesh_faces = _flat_square_mesh()
        # 4 corners of an axis-aligned square [5,15]×[5,15] and a 5th point inside.
        xy = np.array(
            [[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0], [10.0, 10.0]],
            dtype=np.float32,
        )
        self.input_points = np.concatenate([xy, np.zeros((5, 1), dtype=np.float32)], axis=1)
        self.ego_trajectory = _linear_trajectory()[:, :3, 3]

    def test_returns_expected_keys(self):
        metrics = calculate_point_cloud_hull_coverage_metrics(
            self.mesh_vertices,
            self.mesh_faces,
            self.input_points,
            self.ego_trajectory,
            hull_margins_meters=[0, 2, 4],
        )
        self.assertIn("point_cloud_hull_ground_coverage_percentage_0m", metrics)
        self.assertIn("point_cloud_hull_ground_coverage_percentage_2m", metrics)
        self.assertIn("point_cloud_hull_ground_coverage_percentage_4m", metrics)
        self.assertIn("mesh_area_2d_outside_point_cloud_hull_percentage", metrics)

    def test_metric_values(self):
        metrics = calculate_point_cloud_hull_coverage_metrics(
            self.mesh_vertices,
            self.mesh_faces,
            self.input_points,
            self.ego_trajectory,
            hull_margins_meters=[0, 2, 4],
        )
        # The hull [5,15]×[5,15] overlaps the mesh [0,10]×[0,10] in [5,10]×[5,10].
        # At 0.5 m grid resolution: 11×11 = 121 overlap cells out of 21×21 = 441 hull cells.
        self.assertAlmostEqual(
            metrics["point_cloud_hull_ground_coverage_percentage_0m"],
            100.0 * 121.0 / 441.0,
            delta=0.01,
        )
        # Expanding the hull margin adds uncovered area outside the mesh, decreasing the covered fraction.
        self.assertGreater(
            metrics["point_cloud_hull_ground_coverage_percentage_0m"],
            metrics["point_cloud_hull_ground_coverage_percentage_2m"],
        )
        self.assertGreater(
            metrics["point_cloud_hull_ground_coverage_percentage_2m"],
            metrics["point_cloud_hull_ground_coverage_percentage_4m"],
        )
        # Mesh area beyond the hull: total = 100 m², inside hull ≈ 121 × 0.25 = 30.25 m² → 69.75% beyond.
        self.assertAlmostEqual(metrics["mesh_area_2d_outside_point_cloud_hull_percentage"], 69.75, delta=0.01)

    def test_too_few_points_returns_zeros(self):
        metrics = calculate_point_cloud_hull_coverage_metrics(
            self.mesh_vertices,
            self.mesh_faces,
            self.input_points[:2],
            self.ego_trajectory,
            hull_margins_meters=[0],
        )
        self.assertEqual(metrics.get("point_cloud_hull_ground_coverage_percentage_0m", 0.0), 0.0)

    def test_diagnostic_files_exported(self):
        with tempfile.TemporaryDirectory() as output_dir:
            calculate_point_cloud_hull_coverage_metrics(
                self.mesh_vertices,
                self.mesh_faces,
                self.input_points,
                self.ego_trajectory,
                hull_margins_meters=[0, 2, 4],
                output_dir=output_dir,
            )
            for filename in [
                "occupancy_mesh.png",
                "occupancy_point_cloud_hull.png",
                "occupancy_input_points.png",
                "occupancy_input_points_vs_hull.png",
                "occupancy_input_points_vs_mesh.png",
                "occupancy_point_cloud_hull_extended_0m.png",
                "occupancy_mesh_vs_point_cloud_hull_extended_0m.png",
                "occupancy_point_cloud_hull_extended_2m.png",
                "occupancy_mesh_vs_point_cloud_hull_extended_2m.png",
                "occupancy_point_cloud_hull_extended_4m.png",
                "occupancy_mesh_vs_point_cloud_hull_extended_4m.png",
            ]:
                self.assertTrue(os.path.isfile(os.path.join(output_dir, filename)), filename)


class TestCalculateGroundMeshMetrics(unittest.TestCase):
    def test_mesh_metrics_exact_values(self):
        vertices, faces = _flat_square_mesh()
        T_rig_worlds = _linear_trajectory()
        metrics = calculate_ground_mesh_metrics(vertices, faces, T_rig_worlds)
        self.assertAlmostEqual(metrics["mesh_area_m2"], 100.0, places=3)
        self.assertEqual(metrics["mesh_vertex_count"], 4)
        self.assertEqual(metrics["mesh_face_count"], 2)
        self.assertEqual(metrics["mesh_boundary_count"], 1)

    def test_empty_trajectory_does_not_raise(self):
        vertices, faces = _flat_square_mesh()
        T_rig_worlds = np.zeros((0, 4, 4), dtype=np.float32)
        metrics = calculate_ground_mesh_metrics(vertices, faces, T_rig_worlds)
        self.assertIn("mesh_area_m2", metrics)

    def test_empty_trajectory_with_input_points_does_not_raise(self):
        vertices, faces = _flat_square_mesh()
        T_rig_worlds = np.zeros((0, 4, 4), dtype=np.float32)
        rng = np.random.default_rng(1)
        xy = rng.uniform(1, 9, (50, 2)).astype(np.float32)
        points = np.concatenate([xy, np.zeros((50, 1), dtype=np.float32)], axis=1)
        metrics = calculate_ground_mesh_metrics(vertices, faces, T_rig_worlds, input_points=points)
        self.assertIn("mesh_area_m2", metrics)


class TestCalculateVehicleTrajectoryVsGroundConsistencyMetrics(unittest.TestCase):
    def setUp(self):
        self.mesh_vertices, self.mesh_faces = _flat_square_mesh()
        self.T_rig_worlds = _linear_trajectory()

    def test_metric_values(self):
        metrics = calculate_vehicle_trajectory_vs_ground_consistency_metrics(
            self.mesh_vertices, self.mesh_faces, self.T_rig_worlds
        )
        # 20-sample trajectory: x=5.01..24.01, y=5, z=0.2 (unit steps in x, 19 segments).
        self.assertEqual(metrics["trajectory_sample_count"], 20)
        self.assertAlmostEqual(metrics["trajectory_length_meters"], 19.0, places=3)
        # Samples x=5.01..9.01 (i=0..4) are inside the [0,10]×[0,10] mesh → 5/20 = 25%.
        self.assertAlmostEqual(metrics["trajectory_ground_coverage_percentage"], 25.0, places=3)
        # All covered samples are at z=0.2; mesh is flat at z=0 → distance is 0.2 everywhere.
        self.assertAlmostEqual(metrics["trajectory_ground_distance_max_meters"], 0.2, places=3)
        self.assertAlmostEqual(metrics["trajectory_ground_distance_med_meters"], 0.2, places=3)

    def test_diagnostic_files_exported(self):
        with tempfile.TemporaryDirectory() as output_dir:
            calculate_vehicle_trajectory_vs_ground_consistency_metrics(
                self.mesh_vertices,
                self.mesh_faces,
                self.T_rig_worlds,
                output_dir=output_dir,
            )
            for filename in [
                "rig_positions.ply",
                "rig_positions_up_shifted.ply",
                "trajectory_ground_projections.ply",
            ]:
                self.assertTrue(os.path.isfile(os.path.join(output_dir, filename)), filename)


def _make_cuboid_tracks(n_tracks: int = 1, dynamic: bool = True) -> CuboidTracks:
    """Create a minimal CuboidTracks with n_tracks tracks on the CPU."""
    cpu = torch.device("cpu")
    flags = TrackFlags.DYNAMIC | TrackFlags.CONTROLLABLE if dynamic else TrackFlags(0)
    tracks_id = [f"track_{i}" for i in range(n_tracks)]
    # Tracks step along x with step=1.5 starting at x=8.0 (y=5, z=0, 3 stationary poses each).
    # Track 0 (x=8): base samples x=[7..9] — fully inside mesh → covered.
    # Track 1 (x=9.5): base samples reach x=10.5 — crosses right mesh boundary → not covered.
    # Tracks 2+ (x=11, 12.5, …): fully outside mesh → not covered.
    # With 5 tracks × 3 poses = 15 total poses, 3 covered → 20% coverage.
    poses = []
    for i in range(n_tracks):
        T = np.eye(4, dtype=np.float32)[np.newaxis].repeat(3, axis=0)
        T[:, 0, 3] = 8.0 + i * 1.5
        T[:, 1, 3] = 5.0
        poses.append(T)
    timestamps = [np.array([0, 1, 2], dtype=np.int64) for _ in range(n_tracks)]
    label_class = ["vehicle"] * n_tracks
    flags_list = [flags] * n_tracks
    dims = [np.array([2.0, 1.0, 1.5], dtype=np.float32) for _ in range(n_tracks)]
    return CuboidTracks.Factory.from_numpy(tracks_id, poses, timestamps, label_class, flags_list, dims, device=cpu)


class TestCalculateCuboidCoverageByGroundMesh(unittest.TestCase):
    def setUp(self):
        self.mesh_vertices, self.mesh_faces = _flat_square_mesh()

    def test_dynamic_cuboids(self):
        # 5 dynamic tracks × 3 poses = 15 poses; track 0 (x=8) is the only covered one → 3/15 = 20%.
        # Covered base samples shoot upward from z=-0.75 and hit the flat mesh at z=0 → distance 0.75 m.
        cuboid_tracks = _make_cuboid_tracks(n_tracks=5, dynamic=True)
        metrics = calculate_cuboid_coverage_by_ground_mesh(cuboid_tracks, self.mesh_vertices, self.mesh_faces)
        self.assertEqual(metrics["cuboid_tracks_count"], 5)
        self.assertEqual(metrics["cuboid_poses_count"], 15)
        self.assertAlmostEqual(metrics["cuboid_ground_coverage_percentage"], 20.0, places=1)
        self.assertAlmostEqual(metrics["cuboid_ground_distance_med_meters"], 0.75, places=2)
        self.assertAlmostEqual(metrics["cuboid_ground_distance_max_meters"], 0.75, places=2)
        self.assertEqual(metrics["dynamic_cuboid_tracks_count"], 5)
        self.assertEqual(metrics["dynamic_cuboid_poses_count"], 15)
        self.assertAlmostEqual(metrics["dynamic_cuboid_ground_coverage_percentage"], 20.0, places=1)
        self.assertAlmostEqual(metrics["dynamic_cuboid_ground_distance_med_meters"], 0.75, places=2)
        self.assertAlmostEqual(metrics["dynamic_cuboid_ground_distance_max_meters"], 0.75, places=2)

    def test_static_cuboids(self):
        # Same layout as test_dynamic_cuboids but all tracks are static → dynamic counts are zero.
        cuboid_tracks = _make_cuboid_tracks(n_tracks=5, dynamic=False)
        metrics = calculate_cuboid_coverage_by_ground_mesh(cuboid_tracks, self.mesh_vertices, self.mesh_faces)
        self.assertEqual(metrics["cuboid_tracks_count"], 5)
        self.assertEqual(metrics["cuboid_poses_count"], 15)
        self.assertAlmostEqual(metrics["cuboid_ground_coverage_percentage"], 20.0, places=1)
        self.assertAlmostEqual(metrics["cuboid_ground_distance_med_meters"], 0.75, places=2)
        self.assertAlmostEqual(metrics["cuboid_ground_distance_max_meters"], 0.75, places=2)
        self.assertEqual(metrics["dynamic_cuboid_tracks_count"], 0)
        self.assertEqual(metrics["dynamic_cuboid_poses_count"], 0)
        self.assertEqual(metrics["dynamic_cuboid_ground_coverage_percentage"], 100.0)
        self.assertEqual(metrics["dynamic_cuboid_ground_distance_med_meters"], 0.0)
        self.assertEqual(metrics["dynamic_cuboid_ground_distance_max_meters"], 0.0)

    def test_empty_tracks_returns_perfect_coverage(self):
        empty = CuboidTracks.Factory.empty(device=torch.device("cpu"))
        metrics = calculate_cuboid_coverage_by_ground_mesh(empty, self.mesh_vertices, self.mesh_faces)
        self.assertEqual(metrics["cuboid_tracks_count"], 0)
        self.assertEqual(metrics["cuboid_poses_count"], 0)
        self.assertEqual(metrics["cuboid_ground_coverage_percentage"], 100.0)
        self.assertEqual(metrics["cuboid_ground_distance_med_meters"], 0.0)
        self.assertEqual(metrics["cuboid_ground_distance_max_meters"], 0.0)
        self.assertEqual(metrics["dynamic_cuboid_tracks_count"], 0)
        self.assertEqual(metrics["dynamic_cuboid_poses_count"], 0)
        self.assertEqual(metrics["dynamic_cuboid_ground_coverage_percentage"], 100.0)
        self.assertEqual(metrics["dynamic_cuboid_ground_distance_med_meters"], 0.0)
        self.assertEqual(metrics["dynamic_cuboid_ground_distance_max_meters"], 0.0)

    def test_diagnostic_files_exported(self):
        with tempfile.TemporaryDirectory() as output_dir:
            cuboid_tracks = _make_cuboid_tracks(n_tracks=1)
            calculate_cuboid_coverage_by_ground_mesh(
                cuboid_tracks,
                self.mesh_vertices,
                self.mesh_faces,
                output_dir=output_dir,
            )
            for filename in [
                "cuboids_dynamic.ply",
                "cuboid_base_samples.ply",
                "cuboid_projected_samples.ply",
                "cuboid_stats.csv",
            ]:
                self.assertTrue(os.path.isfile(os.path.join(output_dir, filename)), filename)


if __name__ == "__main__":
    unittest.main()
