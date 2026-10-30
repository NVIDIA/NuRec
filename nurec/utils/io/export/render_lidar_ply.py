# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared intensity-colored LiDAR PLY export for local and gRPC rendering."""

from pathlib import Path

import numpy as np
import point_cloud_utils as pcu

from nurec.utils.visualize import scalar2rgb


def save_lidar_ply(path: str | Path, positions: np.ndarray, intensities: np.ndarray | None = None) -> None:
    """Write sensor-space XYZ with intensity mapped to vertex colors, without changing positions."""
    mesh = pcu.TriangleMesh()
    mesh.vertex_data.positions = positions
    if intensities is not None and intensities.size:
        # Empty scans have no intensity range to normalize.
        mesh.vertex_data.colors = scalar2rgb(intensities)
    mesh.save(str(path))
