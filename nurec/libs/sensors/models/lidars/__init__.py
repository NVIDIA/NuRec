# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiDAR models package - Layer 2 stateful LiDAR models.

This module provides:
- LidarModel: Abstract base class for all LiDAR models
- RowOffsetStructuredSpinningLidarModel: Row-offset structured spinning LiDAR model
- LidarFrame: Frame class for LiDAR observations
"""

from nurec.libs.sensors.kernels.lidars import (
    LidarProjection,
    RowOffsetStructuredSpinningLidarProjection,
)
from nurec.libs.sensors.models.lidars.lidar_frame import LidarFrame, LidarFrameSet
from nurec.libs.sensors.models.lidars.lidar_model import LidarModel
from nurec.libs.sensors.models.lidars.row_offset_structured_spinning import RowOffsetStructuredSpinningLidarModel


__all__ = [
    # LiDAR models
    "LidarModel",
    "RowOffsetStructuredSpinningLidarModel",
    # Frame types
    "LidarFrame",
    "LidarFrameSet",
    # Re-exported Layer 0 projection types
    "LidarProjection",
    "RowOffsetStructuredSpinningLidarProjection",
]
