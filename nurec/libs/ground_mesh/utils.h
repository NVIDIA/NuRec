// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ku/common_host.h>
#include <torch/extension.h>

// CUDA fast path for the road ground-mesh production case:
//   - mean height/color aggregation in BEV cells
//   - grid-constrained 2.5D topology
//   - compact unreferenced vertices after triangle emission
//
// Input:
//   points: (N, 3) float32 CUDA tensor in world coordinates.
//   colors: optional (N, 3) uint8 CUDA tensor; pass an undefined tensor for no color.
//
// Returns:
//   vertices:      (V, 3) float32 CUDA tensor after topology compaction.
//   triangles:    (T, 3) int32 CUDA tensor indexing ``vertices``.
//   vertex_colors:(V, 3) uint8 CUDA tensor, or an undefined tensor if no colors were supplied.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> build_grid_mesh_mean_cu(
    const torch::Tensor& points,
    const torch::Tensor& colors,
    const double cell_size,
    const int min_points_per_cell,
    const int hole_fill_min_neighbors);

torch::Tensor smooth_mesh_along_axis_cu(
    const torch::Tensor& vertices,
    const torch::Tensor& edges,
    const int axis,
    const double smoothness,
    const int num_passes);

torch::Tensor compute_vertex_normals_cu(
    const torch::Tensor& vertices,
    const torch::Tensor& triangles);

torch::Tensor find_ground_compatible_points_cu(
    const torch::Tensor& points,
    const torch::Tensor& normals,
    const torch::Tensor& ground_control_point,
    const torch::Tensor& ref_normal,
    const double max_distance,
    const double max_angle_deg);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> detect_dominant_plane_cu(
    const torch::Tensor& points,
    const torch::Tensor& normals,
    const torch::Tensor& hypothesis_indices,
    const torch::Tensor& eval_indices,
    const double distance_threshold,
    const double angle_threshold_deg);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> detect_dominant_plane_batched_cu(
    const torch::Tensor& points,
    const torch::Tensor& normals,
    const torch::Tensor& hypothesis_indices,
    const torch::Tensor& eval_indices,
    const torch::Tensor& pack_info,
    const double distance_threshold,
    const double angle_threshold_deg);
