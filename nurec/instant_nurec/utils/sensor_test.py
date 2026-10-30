# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import torch

from ncore.impl.data.types import FThetaCameraModelParameters, OpenCVPinholeCameraModelParameters, ShutterType
from ncore.impl.sensors.camera import camera_model_from_parameters
from nurec.instant_nurec.utils.sensor import to_simple_pinhole_model_parameters


WIDTH = 200
HALF_WIDTH = WIDTH / 2.0  # horizontal pinhole_pixel_distance for a centered principal point


def _ftheta_with_linear_backward_poly(angle_per_pixel: float, max_angle_deg: float) -> FThetaCameraModelParameters:
    """Build a 200x200 FTheta camera whose pixel->angle map is linear: angle = angle_per_pixel * radius.

    The horizontal edge pixels sit HALF_WIDTH px from the principal point, so they map to a ray at
    ``angle_per_pixel * HALF_WIDTH`` radians regardless of ``max_angle`` (the model extrapolates the
    polynomial rather than clamping rays to ``max_angle``). ``max_angle_deg`` is the stored valid half-FOV
    that ``to_simple_pinhole_model_parameters`` bounds the probed FOV against.
    """
    return FThetaCameraModelParameters(
        resolution=np.array([WIDTH, WIDTH], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([HALF_WIDTH, HALF_WIDTH], dtype=np.float32),
        reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
        pixeldist_to_angle_poly=np.array([0.0, angle_per_pixel, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        angle_to_pixeldist_poly=np.array([0.0, 1.0 / angle_per_pixel, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        max_angle=float(np.deg2rad(max_angle_deg)),
        linear_cde=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )


def _probed_horizontal_half_fov_rad(params: FThetaCameraModelParameters) -> float:
    """Replicate the FOV probe inside ``to_simple_pinhole_model_parameters`` (method='horizontal', reduce='min')."""
    camera_model = camera_model_from_parameters(params, device="cpu")
    cy = float(params.principal_point[1])
    image_points = torch.tensor([[0.0, cy], [float(params.resolution[0]), cy]])
    rays = torch.nn.functional.normalize(camera_model.image_points_to_camera_rays(image_points.float()), dim=-1)
    return float(torch.min(torch.arccos(rays[:, 2])).item())


class ToSimplePinholeModelParametersTest(unittest.TestCase):
    def test_edge_beyond_max_angle_is_bounded_to_max_angle(self) -> None:
        """Regression test (real H81 failure): when border pixels extrapolate past the camera's valid FOV,
        the probed half-FOV must be bounded by ``max_angle`` so the focal length stays finite and positive
        instead of going negative and tripping OpenCVPinholeCameraModelParameters' assertion."""
        # Edge maps to ~114.6deg, well beyond max_angle=40deg (mirrors the degenerate rear-70fov camera).
        max_angle_deg = 40.0
        params = _ftheta_with_linear_backward_poly(angle_per_pixel=0.02, max_angle_deg=max_angle_deg)

        # Precondition: the raw probe really does exceed max_angle (otherwise the test would be vacuous).
        self.assertGreater(np.rad2deg(_probed_horizontal_half_fov_rad(params)), max_angle_deg)

        result = to_simple_pinhole_model_parameters(params, method="horizontal", reduce="min")

        # Bounded to max_angle -> focal = (width / 2) / tan(max_angle), finite and positive.
        expected_focal = HALF_WIDTH / np.tan(np.deg2rad(max_angle_deg))
        self.assertTrue(np.all(np.isfinite(result.focal_length)), f"focal not finite: {result.focal_length}")
        self.assertGreater(result.focal_length[0], 0.0)
        np.testing.assert_allclose(result.focal_length[0], expected_focal, rtol=1e-4)

    def test_lens_wider_than_90deg_is_bounded_below_90deg(self) -> None:
        """A lens whose valid FOV is itself >= 90deg (e.g. fisheye) cannot be a pinhole; bound below 90deg."""
        params = _ftheta_with_linear_backward_poly(angle_per_pixel=0.02, max_angle_deg=130.0)
        self.assertGreater(np.rad2deg(_probed_horizontal_half_fov_rad(params)), 90.0)

        result = to_simple_pinhole_model_parameters(params, method="horizontal", reduce="min")

        self.assertTrue(np.all(np.isfinite(result.focal_length)), f"focal not finite: {result.focal_length}")
        self.assertGreater(result.focal_length[0], 0.0)
        np.testing.assert_allclose(result.focal_length[0], HALF_WIDTH / np.tan(np.deg2rad(89.0)), rtol=1e-4)

    def test_normal_camera_focal_is_not_clamped(self) -> None:
        """A well-behaved camera (edge pixels inside the valid FOV) must be unaffected by the bound."""
        # Edge maps to ~28.6deg, inside max_angle=60deg.
        params = _ftheta_with_linear_backward_poly(angle_per_pixel=0.005, max_angle_deg=60.0)
        probed = _probed_horizontal_half_fov_rad(params)
        self.assertLess(np.rad2deg(probed), 60.0)

        result = to_simple_pinhole_model_parameters(params, method="horizontal", reduce="min")

        self.assertGreater(result.focal_length[0], 0.0)
        np.testing.assert_allclose(result.focal_length[0], HALF_WIDTH / np.tan(probed), rtol=1e-4)

    def test_pinhole_camera_is_not_bounded(self) -> None:
        """A pinhole has no distortion to extrapolate; it takes the no-bound branch and round-trips its focal."""
        fx = 150.0
        params = OpenCVPinholeCameraModelParameters(
            resolution=np.array([WIDTH, WIDTH], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            external_distortion_parameters=None,
            principal_point=np.array([HALF_WIDTH, HALF_WIDTH], dtype=np.float32),
            focal_length=np.array([fx, fx], dtype=np.float32),
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )

        result = to_simple_pinhole_model_parameters(params, method="horizontal", reduce="min")

        # No bound applied: a centered, undistorted pinhole maps back to its own focal length.
        self.assertGreater(result.focal_length[0], 0.0)
        np.testing.assert_allclose(result.focal_length[0], fx, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
