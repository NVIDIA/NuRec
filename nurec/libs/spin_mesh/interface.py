# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python entry point for the nurec/libs/spin_mesh CUDA extension.

The extension provides ``compute_mesh_normals_in_lidar_spin(points,
n_elevation_bins, n_azimuth_bins)`` which fuses the scanline mesh
triangulation and area-weighted vertex-normal pipeline into a small set
of hand-written CUDA kernels. Returns ``(normals, triangles)`` as CUDA
tensors on the same device as the input.
"""

import importlib

from typing import Any

# Pre-load dynamic torch dependencies before importing the .so, otherwise
# torch's runtime symbol lookup will fail.
import torch  # noqa: F401  (kept for the side-effect of loading libtorch)


spin_mesh: Any = importlib.import_module("nurec.libs.spin_mesh.libspin_mesh_cc")
