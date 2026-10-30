# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiDAR kernels package - Layer 0 GPU operations for LiDAR projection."""

# Pre-load dynamic torch dependencies, otherwise runtime-lookup will fail for torch-specific .so's
import torch  # noqa: F401

import nurec.libs.sensors.liblidar_slang_cc as lidar_slang  # type: ignore

from nurec.libs.sensors.kernels.lidars.bindings import (
    SpinningDirection,
    elements_to_sensor_angles,
    generate_spinning_lidar_rays,
    inverse_project_spinning_lidar,
    sensor_angles_to_sensor_rays,
    sensor_rays_to_sensor_angles,
)
from nurec.libs.sensors.kernels.lidars.parameters import (
    LidarProjection,
    RowOffsetStructuredSpinningLidarProjection,
)


__all__ = [
    # Slang module
    "lidar_slang",
    # Enums
    "SpinningDirection",
    # LiDAR projections
    "RowOffsetStructuredSpinningLidarProjection",
    "LidarProjection",
    # Kernel functions
    "generate_spinning_lidar_rays",
    "elements_to_sensor_angles",
    "inverse_project_spinning_lidar",
    "sensor_rays_to_sensor_angles",
    "sensor_angles_to_sensor_rays",
]
