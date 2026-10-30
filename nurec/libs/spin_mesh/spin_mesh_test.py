# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import torch

from nurec.libs.spin_mesh.interface import spin_mesh  # type: ignore


def _grid_points(n_rows: int, n_az: int) -> np.ndarray:
    """Unit-sphere points on a regular (az, el) grid offset off the ±π seam.

    Using this kind of input gives a one-to-one mapping between scan
    positions and bin cells, which lets us assert exact mesh-shape
    properties without worrying about floating-point bin-edge cases.
    """
    elevations = np.linspace(-np.pi / 8, -0.02, n_rows, dtype=np.float32)
    az_step = 2.0 * np.pi / n_az
    azimuths = (np.arange(n_az, dtype=np.float32) * az_step - np.pi + az_step * 0.5).astype(np.float32)
    el_grid, az_grid = np.meshgrid(elevations, azimuths, indexing="ij")
    el_flat = el_grid.ravel()
    az_flat = az_grid.ravel()
    return np.stack(
        [
            np.cos(el_flat) * np.cos(az_flat),
            np.cos(el_flat) * np.sin(az_flat),
            np.sin(el_flat),
        ],
        axis=1,
    ).astype(np.float32)


def _wavy_ground_points(n_rows: int = 32, n_az: int = 480) -> np.ndarray:
    """LiDAR-like returns from a sensor at z = +2 m firing onto a gently
    wavy ground surface. Stable enough that downstream normal directions
    are mostly upward but with enough variation to exercise the kernel
    on realistic-shape data."""
    sensor_height = 2.0
    elevations = np.linspace(-np.pi / 8, -0.02, n_rows, dtype=np.float32)
    azimuths = np.linspace(-np.pi, np.pi, n_az, endpoint=False, dtype=np.float32)
    el_grid, az_grid = np.meshgrid(elevations, azimuths, indexing="ij")
    el_flat = el_grid.ravel()
    az_flat = az_grid.ravel()
    dist = np.clip(sensor_height / np.maximum(-np.sin(el_flat), 0.05), 1.0, 80.0)
    ends = np.stack(
        [
            dist * np.cos(el_flat) * np.cos(az_flat),
            dist * np.cos(el_flat) * np.sin(az_flat),
            sensor_height + dist * np.sin(el_flat),
        ],
        axis=1,
    ).astype(np.float32)
    ends[:, 2] += (0.4 * np.sin(0.07 * ends[:, 0]) * np.cos(0.05 * ends[:, 1])).astype(np.float32)
    sensor = np.array([0.0, 0.0, sensor_height], dtype=np.float32)
    return ends - sensor


def _elevation_bounds(points_np: np.ndarray) -> tuple[float, float]:
    radii = np.linalg.norm(points_np.astype(np.float64, copy=False), axis=1)
    elevations = np.arcsin(np.clip(points_np[:, 2] / np.maximum(radii, 1e-12), -1.0, 1.0))
    return float(elevations.min()), float(elevations.max())


@unittest.skipUnless(torch.cuda.is_available(), "spin_mesh CUDA extension requires a GPU")
class SpinMeshKernelTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires at least two CUDA devices")
    def test_uses_input_device_not_current_device(self) -> None:
        n_rows, n_az = 8, 32
        points_np = _grid_points(n_rows=n_rows, n_az=n_az)
        points = torch.as_tensor(points_np, dtype=torch.float32, device="cuda:1")
        el_min, el_max = _elevation_bounds(points_np)

        with torch.cuda.device(0):
            normals, triangles = spin_mesh.compute_mesh_normals_in_lidar_spin(
                points,
                n_elevation_bins=n_rows,
                n_azimuth_bins=n_az,
                el_min=el_min,
                el_max=el_max,
            )

        self.assertEqual(normals.device, points.device)
        self.assertEqual(triangles.device, points.device)
        self.assertEqual(int(triangles.cpu().shape[0]), 2 * (n_rows - 1) * (n_az - 1))

    def test_dense_grid_emits_all_quads(self) -> None:
        """On a clean ``n_rows × n_az`` scan grid with one point per cell, the
        kernel must emit exactly ``2 * (n_rows - 1) * (n_az - 1)`` triangles
        when bin counts match the scan grid."""
        n_rows, n_az = 8, 32
        points_np = _grid_points(n_rows=n_rows, n_az=n_az)
        points = torch.as_tensor(points_np, dtype=torch.float32, device="cuda")
        normals, triangles = spin_mesh.compute_mesh_normals_in_lidar_spin(
            points, n_elevation_bins=n_rows, n_azimuth_bins=n_az
        )
        expected = 2 * (n_rows - 1) * (n_az - 1)
        self.assertEqual(
            int(triangles.shape[0]), expected, f"got {int(triangles.shape[0])} triangles, expected exactly {expected}"
        )
        self.assertEqual(tuple(normals.shape), points_np.shape)
        self.assertEqual(normals.dtype, torch.float32)
        self.assertEqual(triangles.dtype, torch.int32)

    def test_triangle_indices_are_in_range(self) -> None:
        """No triangle should index outside the input point set."""
        n_rows, n_az = 32, 480
        points_np = _wavy_ground_points(n_rows=n_rows, n_az=n_az)
        points = torch.as_tensor(points_np, dtype=torch.float32, device="cuda")
        # Bin counts sized to this synthetic scene's scan density.
        _, triangles = spin_mesh.compute_mesh_normals_in_lidar_spin(
            points, n_elevation_bins=n_rows, n_azimuth_bins=n_az
        )
        self.assertGreater(int(triangles.shape[0]), 0, "expected some triangles on wavy ground")
        tri_max = int(triangles.max().item())
        tri_min = int(triangles.min().item())
        self.assertGreaterEqual(tri_min, 0)
        self.assertLess(tri_max, points_np.shape[0])

    def test_normals_point_up_on_flat_ground(self) -> None:
        """Sensor sits above the (nearly) flat surface, so the majority of
        flipped-toward-origin normals should have positive z."""
        n_rows, n_az = 32, 480
        points_np = _wavy_ground_points(n_rows=n_rows, n_az=n_az)
        points = torch.as_tensor(points_np, dtype=torch.float32, device="cuda")
        normals, _ = spin_mesh.compute_mesh_normals_in_lidar_spin(points, n_elevation_bins=n_rows, n_azimuth_bins=n_az)
        up_fraction = float((normals[:, 2] > 0.5).float().mean().item())
        self.assertGreater(up_fraction, 0.7, f"only {up_fraction:.2%} of normals point up")

    def test_unit_norm_normals(self) -> None:
        """Every returned normal must be unit-length (the kernel either
        normalizes or installs the +z fallback)."""
        n_rows, n_az = 32, 480
        points_np = _wavy_ground_points(n_rows=n_rows, n_az=n_az)
        points = torch.as_tensor(points_np, dtype=torch.float32, device="cuda")
        normals, _ = spin_mesh.compute_mesh_normals_in_lidar_spin(points, n_elevation_bins=n_rows, n_azimuth_bins=n_az)
        norms = torch.linalg.norm(normals, dim=1)
        max_dev = float((norms - 1.0).abs().max().item())
        self.assertLess(max_dev, 1e-4, f"max |‖n‖ − 1| = {max_dev:.3e}")

    def test_matches_torch_reference(self) -> None:
        """On the clean grid, the kernel reproduces the torch reference
        path's triangle list exactly and its normals within float
        tolerance."""
        from apps.av.road_mesh.ground_mesh import compute_mesh_normals_in_lidar_spin_torch

        n_rows, n_az = 24, 360
        points_np = _grid_points(n_rows=n_rows, n_az=n_az)
        points = torch.as_tensor(points_np, dtype=torch.float32, device="cuda")
        el_min, el_max = _elevation_bounds(points_np)
        cu_normals, cu_triangles = spin_mesh.compute_mesh_normals_in_lidar_spin(
            points, n_elevation_bins=n_rows, n_azimuth_bins=n_az, el_min=el_min, el_max=el_max
        )
        ref_normals, ref_triangles = compute_mesh_normals_in_lidar_spin_torch(
            points, n_elevation_bins=n_rows, n_azimuth_bins=n_az
        )

        # Triangle sets must match (order may differ — sort each by row-major
        # vertex-id triple for the comparison).
        def _canon(t: torch.Tensor) -> torch.Tensor:
            t = t.to(torch.int64)
            t, _ = torch.sort(t, dim=1)
            idx = (t[:, 0] * 1_000_000_000 + t[:, 1] * 1_000 + t[:, 2]).argsort()
            return t[idx]

        self.assertEqual(
            cu_triangles.shape,
            ref_triangles.shape,
            f"triangle counts differ: cu={tuple(cu_triangles.shape)}, ref={tuple(ref_triangles.shape)}",
        )
        self.assertTrue(
            torch.equal(_canon(cu_triangles), _canon(ref_triangles)),
            "triangle sets must match between the .cu kernel and torch reference",
        )

        diff = (cu_normals - ref_normals).abs().max().item()
        self.assertLess(diff, 1e-3, f"normal max abs diff between .cu kernel and torch ref = {diff:.3e}")


if __name__ == "__main__":
    unittest.main()
