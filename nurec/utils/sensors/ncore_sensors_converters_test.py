# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host calibration reuse and gradient preservation in fisheye conversion."""

import unittest

from unittest import mock

import numpy as np
import torch

from ncore.data import OpenCVFisheyeCameraModelParameters, ShutterType
from ncore.sensors import OpenCVFisheyeCameraModel
from nurec.libs.sensors.kernels.cameras.parameters import OpenCVFisheyeProjection
from nurec.utils.sensors.ncore_sensors_converters import CameraModelConverter


class FisheyeConverterTest(unittest.TestCase):
    def _model(self):
        parameters = OpenCVFisheyeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array([320.0, 320.0], dtype=np.float32),
            radial_coeffs=np.array([-1.0 / 3.0, 0.0, 0.0, 0.0], dtype=np.float32),
            max_angle=2.0,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return OpenCVFisheyeCameraModel(parameters, device=device), parameters, device

    def test_current_host_snapshot_avoids_additional_tensor_readback(self):
        model, host_parameters, device = self._model()
        reference = CameraModelConverter.convert(model, device=device)
        with mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected readback")):
            converted = CameraModelConverter.convert(model, device=device, host_parameters=host_parameters)
        self.assertEqual(converted.resolution, (640, 480))
        assert isinstance(converted.projection, OpenCVFisheyeProjection)
        assert isinstance(reference.projection, OpenCVFisheyeProjection)
        self.assertEqual(converted.projection.configured_max_angle, 2.0)
        self.assertEqual(converted.projection.max_angle, reference.projection.max_angle)
        torch.testing.assert_close(converted.projection.intrinsics, reference.projection.intrinsics)

    def test_differentiable_source_retains_tensor_gradient_path(self):
        model, host_parameters, device = self._model()
        model.forward_poly.requires_grad_(True)
        with mock.patch.object(
            OpenCVFisheyeProjection, "from_host_components", side_effect=AssertionError("lost gradient")
        ):
            converted = CameraModelConverter.convert(model, device=device, host_parameters=host_parameters)
        assert isinstance(converted.projection, OpenCVFisheyeProjection)
        converted.projection.forward_poly.sum().backward()
        expected = torch.zeros_like(model.forward_poly)
        expected[[3, 5, 7, 9]] = 1.0
        torch.testing.assert_close(model.forward_poly.grad, expected)


if __name__ == "__main__":
    unittest.main()
