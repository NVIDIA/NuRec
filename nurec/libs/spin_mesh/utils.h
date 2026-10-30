// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ku/common_host.h>
#include <torch/extension.h>

// Fused per-spin LiDAR mesh + vertex normals on CUDA.
//
// Input:
//   points: (N, 3) float32 CUDA tensor of cartesian points in the sensor-
//           centered frame (sensor at origin).
//   n_elevation_bins, n_azimuth_bins: dimensions of the scan-pattern grid
//           used to quantize (azimuth, elevation) into cells.
//
// Returns:
//   normals:   (N, 3) float32 CUDA tensor of area-weighted vertex normals
//              flipped toward the origin. Vertices with no incident
//              triangle fall back to (0, 0, 1).
//   triangles: (T, 3) int32 CUDA tensor of vertex-index triplets into the
//              input points. T <= 2 * (n_elevation_bins - 1) *
//              (n_azimuth_bins - 1).
std::tuple<torch::Tensor, torch::Tensor> compute_mesh_normals_in_lidar_spin_cu(
    const torch::Tensor& points,
    const int n_elevation_bins,
    const int n_azimuth_bins,
    const double el_min,
    const double el_max);
