# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common types shared between camera and LiDAR kernels.

This module provides:
- Pose types: Pose, Trajectory, DynamicPose (Python dataclasses mirroring Slang structs)
"""

from nurec.libs.sensors.kernels.common.pose import DynamicPose, Pose, Trajectory


__all__ = [
    "Pose",
    "Trajectory",
    "DynamicPose",
]
