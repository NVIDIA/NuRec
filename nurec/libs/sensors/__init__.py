# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensors module - Camera and LiDAR projection with GPU acceleration.

This module provides the sensors architecture with two layers:
- Layer 0 (kernels): GPU kernels implemented in Slang with Python bindings
- Layer 2 (models): Stateful nn.Module wrappers with learnable parameters

The module is organized into:
- kernels.cameras: Camera projection kernels and parameters
- kernels.lidars: LiDAR projection kernels and parameters
- models.cameras: Camera model classes (OpenCVPinhole, OpenCVFisheye, FTheta)
- models.lidars: LiDAR model classes
- models.frame: Frame structures for sensor observations

Example usage:
    # Layer 0: Direct kernel access
    from nurec.libs.sensors.kernels import cameras, lidars

    image_points, valid = cameras.camera_rays_to_image_points(
        camera_rays, projection, external_distortion
    )

    # Layer 2: High-level model API
    from nurec.libs.sensors import models

    camera = models.OpenCVPinholeCameraModel.from_ncore(ncore_camera_model)
    result = camera.world_points_to_image_points_shutter_pose(
        world_points, dynamic_pose
    )
"""

from nurec.libs.sensors import kernels, models


__all__ = [
    "kernels",
    "models",
]
