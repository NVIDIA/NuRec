# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import torch

from apps.av.road_mesh.ground_mesh import (
    _find_ground_compatible_points_numpy,
    build_grid_constrained_mesh,
    downsample_3d_points_on_2d_grid,
    get_mesh_edges,
)
from nurec.libs.ground_mesh.interface import ground_mesh  # type: ignore


@unittest.skipUnless(torch.cuda.is_available(), "ground_mesh CUDA extension requires a GPU")
class GroundMeshKernelTest(unittest.TestCase):
    def test_build_grid_mesh_mean_matches_python_reference(self) -> None:
        points = np.array(
            [
                [0.05, 0.05, 0.0],
                [0.15, 0.05, 2.0],  # same cell as first point; mean z = 1.0
                [1.05, 0.05, 3.0],
                [0.05, 1.05, 4.0],
                [1.05, 1.05, 5.0],
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [0, 10, 20],
                [100, 110, 120],
                [30, 40, 50],
                [60, 70, 80],
                [90, 100, 110],
            ],
            dtype=np.uint8,
        )

        ref_vertices, ref_colors = downsample_3d_points_on_2d_grid(points, 1.0, colors=colors)
        ref_vertices, ref_triangles, keep_mask = build_grid_constrained_mesh(ref_vertices, 1.0)
        assert ref_colors is not None
        ref_colors = ref_colors[keep_mask]

        points_t = torch.as_tensor(points, dtype=torch.float32, device="cuda")
        colors_t = torch.as_tensor(colors, dtype=torch.uint8, device="cuda")
        cu_vertices_t, cu_triangles_t, cu_colors_t = ground_mesh.build_grid_mesh_mean(
            points_t,
            colors_t,
            cell_size=1.0,
            min_points_per_cell=1,
        )

        np.testing.assert_allclose(cu_vertices_t.cpu().numpy(), ref_vertices, rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(cu_triangles_t.cpu().numpy(), ref_triangles)
        np.testing.assert_array_equal(cu_colors_t.cpu().numpy(), ref_colors)

    def test_build_grid_mesh_mean_fills_single_cell_hole_with_at_least_six_neighbors(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 4.0],
                [0.0, 1.0, 6.0],
                # center cell intentionally empty
                [2.0, 1.0, 8.0],
                [0.0, 2.0, 10.0],
                [1.0, 2.0, 12.0],
                [2.0, 2.0, 14.0],
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [10, 0, 0],
                [20, 0, 0],
                [30, 0, 0],
                [40, 0, 0],
                [50, 0, 0],
                [60, 0, 0],
                [70, 0, 0],
                [80, 0, 0],
            ],
            dtype=np.uint8,
        )

        vertices_t, triangles_t, colors_t = ground_mesh.build_grid_mesh_mean(
            torch.as_tensor(points, dtype=torch.float32, device="cuda"),
            torch.as_tensor(colors, dtype=torch.uint8, device="cuda"),
            cell_size=1.0,
            min_points_per_cell=1,
            hole_fill_min_neighbors=6,
        )

        vertices = vertices_t.cpu().numpy()
        center_idx = np.flatnonzero(np.all(np.isclose(vertices[:, :2], [1.5, 1.5], atol=1e-6), axis=1))
        self.assertEqual(len(center_idx), 1)
        np.testing.assert_allclose(vertices[center_idx[0]], [1.5, 1.5, 7.0], rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(colors_t.cpu().numpy()[center_idx[0]], [45, 0, 0])
        self.assertEqual(len(vertices), 9)
        self.assertEqual(len(triangles_t), 8)

    def test_build_grid_mesh_mean_fills_bracketed_interior_hole(self) -> None:
        points = np.array(
            [[float(x), float(y), float(x + y)] for y in range(5) for x in range(5) if x in (0, 4) or y in (0, 4)],
            dtype=np.float32,
        )

        vertices_t, triangles_t, _ = ground_mesh.build_grid_mesh_mean(
            torch.as_tensor(points, dtype=torch.float32, device="cuda"),
            cell_size=1.0,
            min_points_per_cell=1,
            hole_fill_min_neighbors=0,
        )

        vertices = vertices_t.cpu().numpy()
        self.assertEqual(len(vertices), 25)
        self.assertEqual(len(triangles_t), 32)
        for y in (1.5, 2.5, 3.5):
            for x in (1.5, 2.5, 3.5):
                self.assertTrue(np.any(np.all(np.isclose(vertices[:, :2], [x, y], atol=1e-6), axis=1)))

    def test_build_grid_mesh_mean_requires_hole_fill_threshold(self) -> None:
        points = torch.zeros((3, 3), dtype=torch.float32, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "hole_fill_min_neighbors must be in \\[0, 8\\]"):
            ground_mesh.build_grid_mesh_mean(points, cell_size=1.0, hole_fill_min_neighbors=9)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires at least two CUDA devices")
    def test_all_entry_points_use_input_device_not_current_device(self) -> None:
        device = torch.device("cuda:1")
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        normals = torch.tensor([[0.0, 0.0, 1.0]] * 4, dtype=torch.float32, device=device)
        triangles = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32, device=device)
        edges = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=torch.int64, device=device)
        indices = torch.arange(4, dtype=torch.int64, device=device)
        pack_info = torch.tensor([[0, 4, 0, 4, 0, 4]], dtype=torch.int64, device=device)

        with torch.cuda.device(0):
            outputs = (
                ground_mesh.build_grid_mesh_mean(points, cell_size=1.0, min_points_per_cell=1),
                ground_mesh.smooth_mesh_along_axis(points, edges),
                ground_mesh.compute_vertex_normals(points, triangles),
                ground_mesh.find_ground_compatible_points(
                    points,
                    normals,
                    torch.tensor([0.0, 0.0, 0.0], device=device),
                    torch.tensor([0.0, 0.0, 1.0], device=device),
                    0.5,
                    60.0,
                ),
                ground_mesh.detect_dominant_plane(points, normals, indices, indices, 0.3, 30.0),
                ground_mesh.detect_dominant_plane_batched(points, normals, indices, indices, pack_info, 0.3, 30.0),
            )

        for output_group in outputs:
            tensors = output_group if isinstance(output_group, tuple) else (output_group,)
            for tensor in tensors:
                if tensor is not None:
                    self.assertEqual(tensor.device, device)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires at least two CUDA devices")
    def test_all_tensor_arguments_reject_mixed_devices(self) -> None:
        primary_device = torch.device("cuda:0")
        other_device = torch.device("cuda:1")
        points = torch.zeros((4, 3), dtype=torch.float32, device=primary_device)
        normals = torch.tensor([[0.0, 0.0, 1.0]] * 4, dtype=torch.float32, device=primary_device)
        triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=primary_device)
        edges = torch.tensor([[0, 1]], dtype=torch.int64, device=primary_device)
        indices = torch.arange(4, dtype=torch.int64, device=primary_device)
        vector = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=primary_device)
        pack_info = torch.tensor([[0, 4, 0, 4, 0, 4]], dtype=torch.int64, device=primary_device)

        cases = (
            (
                "colors",
                lambda: ground_mesh.build_grid_mesh_mean(
                    points, torch.zeros((4, 3), dtype=torch.uint8, device=other_device)
                ),
            ),
            ("edges", lambda: ground_mesh.smooth_mesh_along_axis(points, edges.to(other_device))),
            ("triangles", lambda: ground_mesh.compute_vertex_normals(points, triangles.to(other_device))),
            (
                "normals",
                lambda: ground_mesh.find_ground_compatible_points(
                    points, normals.to(other_device), vector, vector, 0.5, 60.0
                ),
            ),
            (
                "ground_control_point",
                lambda: ground_mesh.find_ground_compatible_points(
                    points, normals, vector.to(other_device), vector, 0.5, 60.0
                ),
            ),
            (
                "ref_normal",
                lambda: ground_mesh.find_ground_compatible_points(
                    points, normals, vector, vector.to(other_device), 0.5, 60.0
                ),
            ),
            (
                "hypothesis_indices",
                lambda: ground_mesh.detect_dominant_plane(
                    points, normals, indices.to(other_device), indices, 0.3, 30.0
                ),
            ),
            (
                "eval_indices",
                lambda: ground_mesh.detect_dominant_plane(
                    points, normals, indices, indices.to(other_device), 0.3, 30.0
                ),
            ),
            (
                "pack_info",
                lambda: ground_mesh.detect_dominant_plane_batched(
                    points, normals, indices, indices, pack_info.to(other_device), 0.3, 30.0
                ),
            ),
        )
        for argument_name, call in cases:
            with self.subTest(argument_name=argument_name):
                with self.assertRaisesRegex(RuntimeError, f"{argument_name} must be on the same CUDA device"):
                    call()

    def test_out_of_bounds_indices_are_ignored_without_host_validation(self) -> None:
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device="cuda",
        )
        normals = torch.tensor([[0.0, 0.0, 1.0]] * 3, dtype=torch.float32, device="cuda")

        smoothed = ground_mesh.smooth_mesh_along_axis(
            points,
            torch.tensor([[-1, 1], [0, 99]], dtype=torch.int64, device="cuda"),
        )
        computed_normals = ground_mesh.compute_vertex_normals(
            points,
            torch.tensor([[-1, 1, 2], [0, 1, 99]], dtype=torch.int32, device="cuda"),
        )
        _, _, inliers = ground_mesh.detect_dominant_plane(
            points,
            normals,
            torch.tensor([0], dtype=torch.int64, device="cuda"),
            torch.tensor([-1, 0, 99], dtype=torch.int64, device="cuda"),
            0.3,
            30.0,
        )

        np.testing.assert_array_equal(smoothed.cpu().numpy(), points.cpu().numpy())
        np.testing.assert_array_equal(computed_normals.cpu().numpy(), np.zeros((3, 3), dtype=np.float32))
        self.assertEqual(inliers.shape, (3,))

    def test_smooth_mesh_along_axis_matches_torch_reference(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 2.0],
                [1.0, 1.0, 4.0],
            ],
            dtype=np.float32,
        )
        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        edges = get_mesh_edges(triangles).astype(np.int64)

        vertices_t = torch.as_tensor(vertices, dtype=torch.float32, device="cuda")
        edges_t = torch.as_tensor(edges, dtype=torch.int64, device="cuda")
        smoothed_t = ground_mesh.smooth_mesh_along_axis(vertices_t, edges_t, axis=2, smoothness=1.0, num_passes=3)

        # Reference mirrors nurec.utils.io.ground_mesh._smooth_mesh_along_axis_numpy without importing a private helper.
        i = edges[:, 0].astype(np.int64)
        j = edges[:, 1].astype(np.int64)
        diff = vertices[i] - vertices[j]
        weights = (1.0 / np.sqrt(np.sum(diff * diff, axis=1))).astype(np.float32)
        indices = np.concatenate([i, j])
        weights_concat = np.concatenate([weights, weights])
        weight_sum = np.bincount(indices, weights=weights_concat, minlength=len(vertices)).astype(np.float32)
        connected = weight_sum > 0
        heights = vertices[:, 2].copy()
        for _ in range(3):
            values = np.concatenate([weights * heights[j], weights * heights[i]])
            weighted_sum = np.bincount(indices, weights=values, minlength=len(vertices)).astype(np.float32)
            target = heights.copy()
            target[connected] = weighted_sum[connected] / weight_sum[connected]
            heights = target
        expected = vertices.copy()
        expected[:, 2] = heights

        np.testing.assert_allclose(smoothed_t.cpu().numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_smoothing_is_repeatable_for_irregular_neighborhoods(self) -> None:
        rng = np.random.default_rng(4821)
        vertices = rng.uniform(1.0, 20.0, size=(1025, 3)).astype(np.float32)
        edges = rng.integers(0, 1024, size=(4096, 2), dtype=np.int64)
        # Extend the random graph with the row-construction cases most likely to
        # expose ordering or bounds bugs:
        # - a vertex with 512 neighbors
        # - duplicate and self edges
        # - endpoints outside the vertex range
        edges = np.concatenate(
            [
                edges,
                np.column_stack([np.zeros(512, dtype=np.int64), np.arange(1, 513)]),
                [[0, 1], [0, 1], [1, 1], [-1, 2], [2, len(vertices)]],
            ]
        )
        vertices_t = torch.as_tensor(vertices, device="cuda")
        edges_t = torch.as_tensor(edges, device="cuda")
        # The CUDA kernel ignores invalid endpoints, so omit them from the CPU reference.
        valid_edges = edges[np.all((edges >= 0) & (edges < len(vertices)), axis=1)]
        i, j = valid_edges.T
        weights = 1.0 / np.maximum(np.linalg.norm(vertices[i] - vertices[j], axis=1), 1e-8)
        indices = np.concatenate([i, j])
        weight_sum = np.bincount(indices, weights=np.concatenate([weights, weights]), minlength=len(vertices)).astype(
            np.float32
        )
        connected = weight_sum > 0

        for axis in range(3):
            with self.subTest(axis=axis):
                # Reproduce ten passes with weights fixed from the input vertices.
                reference_heights = vertices[:, axis].copy()
                for _ in range(10):
                    total = np.bincount(
                        indices,
                        weights=np.concatenate([weights * reference_heights[j], weights * reference_heights[i]]),
                        minlength=len(vertices),
                    ).astype(np.float32)
                    target = reference_heights.copy()
                    target[connected] = total[connected] / weight_sum[connected]
                    reference_heights = 0.35 * target + 0.65 * reference_heights
                # Establish the CUDA result for this graph before varying edge order.
                expected = ground_mesh.smooth_mesh_along_axis(
                    vertices_t, edges_t, axis=axis, smoothness=0.35, num_passes=10
                )
                np.testing.assert_allclose(expected[:, axis].cpu().numpy(), reference_heights, rtol=1e-6, atol=1e-6)
                # Each permutation changes parallel insertion order; sorting must recover
                # the same neighbor-index order before the reduction.
                for _ in range(20):
                    shuffled_edges = torch.as_tensor(rng.permutation(edges), device="cuda")
                    actual = ground_mesh.smooth_mesh_along_axis(
                        vertices_t, shuffled_edges, axis=axis, smoothness=0.35, num_passes=10
                    )
                    self.assertTrue(torch.equal(expected, actual))
                # Preserve invariants beyond repeatability:
                # - the intentionally isolated final vertex remains unchanged
                # - unselected coordinates remain unchanged
                # - every output value remains finite
                self.assertTrue(torch.equal(expected[-1], vertices_t[-1]))
                unchanged_axes = [coordinate for coordinate in range(3) if coordinate != axis]
                self.assertTrue(torch.equal(expected[:, unchanged_axes], vertices_t[:, unchanged_axes]))
                self.assertTrue(torch.isfinite(expected).all())

    def test_compute_vertex_normals_matches_python_reference(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        vertices_t = torch.as_tensor(vertices, dtype=torch.float32, device="cuda")
        triangles_t = torch.as_tensor(triangles, dtype=torch.int32, device="cuda")

        normals_t = ground_mesh.compute_vertex_normals(vertices_t, triangles_t)

        v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-8)
        expected = np.zeros_like(vertices)
        for i in range(3):
            np.add.at(expected, triangles[:, i], face_normals)
        expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-8)

        np.testing.assert_allclose(normals_t.cpu().numpy(), expected, rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires at least two CUDA devices")
    def test_compute_vertex_normals_uses_input_device_not_current_device(self) -> None:
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device="cuda:1",
        )
        triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda:1")

        with torch.cuda.device(0):
            normals = ground_mesh.compute_vertex_normals(vertices, triangles)

        self.assertEqual(normals.device, vertices.device)
        np.testing.assert_allclose(normals.cpu().numpy(), [[0.0, 0.0, 1.0]] * 3, atol=1e-6)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires at least two CUDA devices")
    def test_compute_vertex_normals_rejects_mixed_devices(self) -> None:
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device="cuda:0",
        )
        triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda:1")

        with self.assertRaisesRegex(RuntimeError, "triangles must be on the same CUDA device"):
            ground_mesh.compute_vertex_normals(vertices, triangles)

    def test_find_ground_compatible_points_matches_python_reference(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        ground_control_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ref_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        max_distance = 0.5
        max_angle_deg = 60.0

        mask_t = ground_mesh.find_ground_compatible_points(
            torch.as_tensor(points, dtype=torch.float32, device="cuda"),
            torch.as_tensor(normals, dtype=torch.float32, device="cuda"),
            torch.as_tensor(ground_control_point, dtype=torch.float32, device="cuda"),
            torch.as_tensor(ref_normal, dtype=torch.float32, device="cuda"),
            max_distance=max_distance,
            max_angle_deg=max_angle_deg,
        )

        # The numpy fallback is called directly rather than the public
        # ``find_ground_compatible_points``, which dispatches to this very kernel whenever CUDA is
        # available and would therefore compare the kernel against itself.
        expected = _find_ground_compatible_points_numpy(
            points,
            normals,
            ground_control_point,
            ref_normal,
            max_distance=max_distance,
            max_angle_deg=max_angle_deg,
        )
        np.testing.assert_array_equal(mask_t.cpu().numpy(), expected)

    def test_detect_dominant_plane_matches_reference(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        hypothesis_indices = np.array([0, 4], dtype=np.int64)
        eval_indices = np.arange(len(points), dtype=np.int64)

        normal_t, offset_t, inlier_t = ground_mesh.detect_dominant_plane(
            torch.as_tensor(points, device="cuda"),
            torch.as_tensor(normals, device="cuda"),
            torch.as_tensor(hypothesis_indices, device="cuda"),
            torch.as_tensor(eval_indices, device="cuda"),
            distance_threshold=0.3,
            angle_threshold_deg=30.0,
        )

        np.testing.assert_allclose(normal_t.cpu().numpy(), np.array([0.0, 0.0, 1.0]), atol=1e-6)
        np.testing.assert_allclose(offset_t.cpu().numpy(), np.array([0.0]), atol=1e-6)
        np.testing.assert_array_equal(inlier_t.cpu().numpy(), np.array([True, True, True, True, False]))

    def test_detect_dominant_plane_matches_batched_numpy_scoring(self) -> None:
        rng = np.random.default_rng(123)
        points = rng.uniform(-10.0, 10.0, size=(2000, 3)).astype(np.float32)
        points[:1800, 2] = rng.normal(0.0, 0.02, size=1800).astype(np.float32)
        normals = rng.normal(size=(2000, 3)).astype(np.float32)
        normals[:1800] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        hypothesis_indices = np.concatenate([np.arange(0, 15, dtype=np.int64), np.arange(1800, 1820, dtype=np.int64)])
        eval_indices = rng.choice(len(points), size=1000, replace=False).astype(np.int64)
        distance_threshold = 0.3
        angle_threshold_deg = 30.0

        normal_t, offset_t, inlier_t = ground_mesh.detect_dominant_plane(
            torch.as_tensor(points, device="cuda"),
            torch.as_tensor(normals, device="cuda"),
            torch.as_tensor(hypothesis_indices, device="cuda"),
            torch.as_tensor(eval_indices, device="cuda"),
            distance_threshold=distance_threshold,
            angle_threshold_deg=angle_threshold_deg,
        )

        plane_normals = normals[hypothesis_indices]
        sample_points = points[hypothesis_indices]
        plane_offsets = -np.sum(plane_normals * sample_points, axis=1)
        normal_lengths = np.linalg.norm(plane_normals, axis=1)
        eval_points = points[eval_indices]
        eval_normals = normals[eval_indices]
        distances = np.abs(eval_points @ plane_normals.T + plane_offsets[None, :]) / normal_lengths[None, :]
        distance_weights = 0.5 + 0.5 * np.cos(np.pi * np.clip(distances / distance_threshold, a_min=-1.0, a_max=1.0))
        dot_products = np.clip(eval_normals @ (plane_normals / normal_lengths[:, None]).T, a_min=-1.0, a_max=1.0)
        angles_deg = np.degrees(np.arccos(dot_products))
        angle_weights = 0.5 + 0.5 * np.cos(np.pi * np.clip(angles_deg / angle_threshold_deg, a_min=-1.0, a_max=1.0))
        best_idx = int(np.argmax(np.sum(distance_weights * angle_weights, axis=0)))
        best_normal = plane_normals[best_idx]
        best_offset = plane_offsets[best_idx]
        full_distances = np.abs(points @ best_normal + best_offset)
        full_angles = np.degrees(np.arccos(np.clip(normals @ best_normal, a_min=-1.0, a_max=1.0)))
        expected_inliers = np.logical_and(full_distances <= distance_threshold, full_angles <= angle_threshold_deg)

        np.testing.assert_allclose(normal_t.cpu().numpy(), best_normal, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(offset_t.cpu().numpy(), np.array([best_offset]), rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(inlier_t.cpu().numpy(), expected_inliers)

    def test_detect_dominant_plane_tie_selects_first_hypothesis(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 100.0],
            ],
            dtype=np.float32,
        )
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        normal_t, offset_t, _ = ground_mesh.detect_dominant_plane(
            torch.as_tensor(points, device="cuda"),
            torch.as_tensor(normals, device="cuda"),
            torch.as_tensor(np.array([0, 1], dtype=np.int64), device="cuda"),
            torch.as_tensor(np.array([2], dtype=np.int64), device="cuda"),
            distance_threshold=0.3,
            angle_threshold_deg=30.0,
        )

        np.testing.assert_array_equal(normal_t.cpu().numpy(), np.array([0.0, 0.0, 1.0], dtype=np.float32))
        np.testing.assert_array_equal(offset_t.cpu().numpy(), np.array([0.0], dtype=np.float32))

    def test_detect_dominant_plane_batched_matches_single_calls(self) -> None:
        first_points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
            dtype=np.float32,
        )
        first_normals = np.array(
            [[0.0, 0.0, 1.0]] * 4 + [[1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        second_points = np.array(
            [[0.0, 0.0, 3.0], [1.0, 0.0, 3.0], [0.0, 1.0, 3.0], [1.0, 1.0, 3.0]],
            dtype=np.float32,
        )
        second_normals = np.array([[0.0, 0.0, 1.0]] * 4, dtype=np.float32)
        points = np.concatenate([first_points, second_points])
        normals = np.concatenate([first_normals, second_normals])
        hypothesis_indices = np.array([0, 4, 5], dtype=np.int64)
        eval_indices = np.arange(len(points), dtype=np.int64)
        pack_info = np.array(
            [
                [0, len(first_points), 0, 2, 0, len(first_points)],
                [len(first_points), len(second_points), 2, 1, len(first_points), len(second_points)],
            ],
            dtype=np.int64,
        )

        normals_t, offsets_t, inliers_t = ground_mesh.detect_dominant_plane_batched(
            torch.as_tensor(points, device="cuda"),
            torch.as_tensor(normals, device="cuda"),
            torch.as_tensor(hypothesis_indices, device="cuda"),
            torch.as_tensor(eval_indices, device="cuda"),
            torch.as_tensor(pack_info, device="cuda"),
            distance_threshold=0.3,
            angle_threshold_deg=30.0,
        )

        expected_normals = []
        expected_offsets = []
        expected_inliers = []
        for point_start, point_count, hypothesis_start, hypothesis_count, eval_start, eval_count in pack_info:
            normal_t, offset_t, inlier_t = ground_mesh.detect_dominant_plane(
                torch.as_tensor(points[point_start : point_start + point_count], device="cuda"),
                torch.as_tensor(normals[point_start : point_start + point_count], device="cuda"),
                torch.as_tensor(
                    hypothesis_indices[hypothesis_start : hypothesis_start + hypothesis_count] - point_start,
                    device="cuda",
                ),
                torch.as_tensor(
                    eval_indices[eval_start : eval_start + eval_count] - point_start,
                    device="cuda",
                ),
                distance_threshold=0.3,
                angle_threshold_deg=30.0,
            )
            expected_normals.append(normal_t.cpu().numpy())
            expected_offsets.append(offset_t.item())
            expected_inliers.append(inlier_t.cpu().numpy())

        np.testing.assert_allclose(normals_t.cpu().numpy(), np.stack(expected_normals), rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(offsets_t.cpu().numpy(), np.array(expected_offsets), rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(inliers_t.cpu().numpy(), np.concatenate(expected_inliers))


if __name__ == "__main__":
    unittest.main()
