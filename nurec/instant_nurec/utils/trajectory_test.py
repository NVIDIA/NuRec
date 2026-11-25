# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ncore.impl.data.types import OpenCVPinholeCameraModelParameters, ShutterType
from nurec.instant_nurec.utils.trajectory import SensorOverride


def test_sensor_override_preserves_uniform_scale_for_rounded_resolution() -> None:
    camera_parameters = OpenCVPinholeCameraModelParameters(
        resolution=np.array([3840, 2160], dtype=np.uint64),
        focal_length=np.array([3840.0, 2160.0], dtype=np.float32),
        principal_point=np.array([1920.0, 1080.0], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        shutter_type=ShutterType.GLOBAL,
    )
    override = SensorOverride(
        sensor_id="camera_front_wide_120fov",
        height=1000,
        translation_offset=(0.0, 0.0, 0.0),
        rotation_offset=(0.0, 0.0, 0.0),
        rotation_first=False,
        force_pinhole=False,
    )

    transformed = override.apply_camera_model_parameters(camera_parameters)

    assert isinstance(transformed, OpenCVPinholeCameraModelParameters)
    np.testing.assert_array_equal(transformed.resolution, [1777, 1000])
    np.testing.assert_allclose(transformed.focal_length, [3840 * 1000 / 2160, 1000.0])
    np.testing.assert_allclose(transformed.principal_point, [1920 * 1000 / 2160, 500.0])
