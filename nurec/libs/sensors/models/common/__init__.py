# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities for sensor models.

This module provides:
- Frame: Base class for sensor frames
- Return type dataclasses for model operations
- Utility functions shared between camera and LiDAR models
"""

from nurec.libs.sensors.models.common.frame import Frame
from nurec.libs.sensors.models.common.return_types import (
    ImagePointsReturn,
    PixelsReturn,
    SensorAnglesReturn,
    SensorRayReturn,
    WorldPointsToImagePointsReturn,
    WorldPointsToPixelsReturn,
    WorldPointsToSensorAnglesReturn,
    WorldRaysReturn,
)
from nurec.libs.sensors.models.common.utils import (
    TensorLike,
    batched_quat_slerp,
    compute_scaled_resolution,
    filter_by_validity,
    poses_to_matrix,
    to_torch,
    valid_flags_to_indices,
)


__all__ = [
    # Frame base class
    "Frame",
    # Return types
    "ImagePointsReturn",
    "PixelsReturn",
    "WorldPointsToImagePointsReturn",
    "WorldPointsToPixelsReturn",
    "WorldRaysReturn",
    "SensorAnglesReturn",
    "SensorRayReturn",
    "WorldPointsToSensorAnglesReturn",
    # Utils
    "TensorLike",
    "to_torch",
    "poses_to_matrix",
    "valid_flags_to_indices",
    "batched_quat_slerp",
    "compute_scaled_resolution",
    "filter_by_validity",
]
