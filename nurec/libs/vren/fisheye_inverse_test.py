# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import torch

from ncore.data import OpenCVFisheyeCameraModelParameters, ShutterType
from nurec.libs.vren.interface import image_points_to_world_rays_shutter_pose, to_vren


@unittest.skipUnless(torch.cuda.is_available(), "VREN camera kernels require CUDA")
class FisheyeInverseTest(unittest.TestCase):
    def setUp(self):
        # g(theta)=theta-theta**3/3 has a known inverse on its optical-axis branch.
        self.parameters = OpenCVFisheyeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array([320.0, 320.0], dtype=np.float32),
            radial_coeffs=np.array([-1.0 / 3.0, 0.0, 0.0, 0.0], dtype=np.float32),
            max_angle=2.0,
        )
        self.poses = torch.tensor([[0, 0, 0, 0, 0, 0, 1]] * 2, dtype=torch.float32, device="cuda")
        self.timestamps = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    def _rays(self, radii):
        radii = torch.tensor(radii, dtype=torch.float32, device="cuda")
        points = torch.stack([320.0 + 320.0 * radii, torch.full_like(radii, 240.0)], dim=1)
        return self._rays_for_points(points)

    def _rays_for_points(self, points):
        return image_points_to_world_rays_shutter_pose(self.parameters, points, self.poses, self.timestamps).world_rays[
            :, 3:
        ]

    def test_valid_rays_match_analytic_inverse_near_singular_seed(self):
        radii = np.array([0.1, 0.49, 0.5, 0.51, 0.6, 0.65], dtype=np.float32)
        theta = 2.0 * np.sin(np.arcsin(1.5 * radii.astype(np.float64)) / 3.0)
        expected = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1)
        rays = self._rays(radii).cpu().numpy()
        np.testing.assert_allclose(rays, expected, atol=2e-6, rtol=0.0)

    def test_outside_domain_and_nonfinite_pixels_have_finite_sentinel(self):
        rays = self._rays([0.0, 0.7, 0.8, float("inf"), float("nan")])
        expected = torch.tensor([[0.0, 0.0, 1.0]], device="cuda").expand_as(rays)
        torch.testing.assert_close(rays, expected, atol=0.0, rtol=0.0)

    def test_mixed_nonfinite_coordinates_have_finite_sentinel(self):
        # A finite nonzero other coordinate exposes fminf/fmaxf hiding a NaN in the norm.
        points = torch.tensor(
            [[float("nan"), 272.0], [352.0, float("nan")], [float("inf"), 272.0], [352.0, -float("inf")]],
            dtype=torch.float32,
            device="cuda",
        )
        rays = self._rays_for_points(points)
        expected = torch.tensor([[0.0, 0.0, 1.0]], device="cuda").expand_as(rays)
        torch.testing.assert_close(rays, expected, atol=0.0, rtol=0.0)

    def test_near_domain_boundary_matches_analytic_inverse(self):
        points = torch.tensor([[532.0, 240.0], [533.0, 240.0], [533.25, 240.0]], device="cuda")
        radius = (points[:, 0].cpu().numpy().astype(np.float64) - 320.0) / 320.0
        a = -3.0 * float(self.parameters.radial_coeffs[0])
        theta = 2.0 / np.sqrt(a) * np.sin(np.arcsin(1.5 * radius * np.sqrt(a)) / 3.0)
        expected = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1)
        # Near the stationary boundary, float32 radius rounding is magnified by 1/g'(theta).
        np.testing.assert_allclose(self._rays_for_points(points).cpu().numpy(), expected, atol=1e-5, rtol=0.0)

    def test_poor_initial_seed_matches_independent_bisection(self):
        self.parameters = OpenCVFisheyeCameraModelParameters(
            resolution=np.array([960, 540], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([960.0, 540.0], dtype=np.float32),
            focal_length=np.array([1000.0, 1000.0], dtype=np.float32),
            radial_coeffs=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            max_angle=np.deg2rad(70.0),
        )
        xx, yy = np.meshgrid(np.linspace(0.5, 959.5, 17), np.linspace(0.5, 539.5, 11))
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        uv = (points.astype(np.float64) - self.parameters.principal_point) / self.parameters.focal_length
        radius = np.linalg.norm(uv, axis=-1)
        lower, upper = np.zeros_like(radius), np.full_like(radius, self.parameters.max_angle)
        k = self.parameters.radial_coeffs.astype(np.float64)
        for _ in range(64):
            theta = 0.5 * (lower + upper)
            projected = theta + sum(k[i] * theta ** (2 * i + 3) for i in range(4))
            lower, upper = np.where(projected < radius, theta, lower), np.where(projected < radius, upper, theta)
        theta = 0.5 * (lower + upper)
        expected = np.column_stack([uv * (np.sin(theta) / radius)[:, None], np.cos(theta)])
        actual = self._rays_for_points(torch.from_numpy(points).to("cuda")).cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0.0)

    def test_native_parameter_conversion_also_limits_domain_without_mutation(self):
        native = to_vren(self.parameters)
        expected_max_angle = native.max_angle
        native.max_angle = 2.0
        converted = to_vren(native)
        self.assertEqual(native.max_angle, 2.0)
        self.assertEqual(converted.max_angle, expected_max_angle)
        self.assertLess(converted.max_angle, 1.0)
        # The common helper leaves a small slope margin inside the analytic boundary at theta=1.
        self.assertLess(1.0 - converted.max_angle, 3e-6)


if __name__ == "__main__":
    unittest.main()
