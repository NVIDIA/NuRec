// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "utils.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> build_grid_mesh_mean(
    const torch::Tensor& points,
    const py::object& colors_obj,
    const double cell_size,
    const int min_points_per_cell,
    const int hole_fill_min_neighbors) {
    torch::Tensor colors;
    if (!colors_obj.is_none()) {
        colors = colors_obj.cast<torch::Tensor>();
    }
    return build_grid_mesh_mean_cu(points, colors, cell_size, min_points_per_cell, hole_fill_min_neighbors);
}

torch::Tensor smooth_mesh_along_axis(
    const torch::Tensor& vertices,
    const torch::Tensor& edges,
    const int axis,
    const double smoothness,
    const int num_passes) {
    return smooth_mesh_along_axis_cu(vertices, edges, axis, smoothness, num_passes);
}

torch::Tensor compute_vertex_normals(
    const torch::Tensor& vertices,
    const torch::Tensor& triangles) {
    return compute_vertex_normals_cu(vertices, triangles);
}

torch::Tensor find_ground_compatible_points(
    const torch::Tensor& points,
    const torch::Tensor& normals,
    const torch::Tensor& ground_control_point,
    const torch::Tensor& ref_normal,
    const double max_distance,
    const double max_angle_deg) {
    return find_ground_compatible_points_cu(
        points, normals, ground_control_point, ref_normal, max_distance, max_angle_deg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> detect_dominant_plane(
    const torch::Tensor& points,
    const torch::Tensor& normals,
    const torch::Tensor& hypothesis_indices,
    const torch::Tensor& eval_indices,
    const double distance_threshold,
    const double angle_threshold_deg) {
    return detect_dominant_plane_cu(
        points, normals, hypothesis_indices, eval_indices, distance_threshold, angle_threshold_deg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> detect_dominant_plane_batched(
    const torch::Tensor& points,
    const torch::Tensor& normals,
    const torch::Tensor& hypothesis_indices,
    const torch::Tensor& eval_indices,
    const torch::Tensor& pack_info,
    const double distance_threshold,
    const double angle_threshold_deg) {
    return detect_dominant_plane_batched_cu(
        points,
        normals,
        hypothesis_indices,
        eval_indices,
        pack_info,
        distance_threshold,
        angle_threshold_deg);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_grid_mesh_mean",
          &build_grid_mesh_mean,
          py::arg("points"),
          py::arg("colors")                  = py::none(),
          py::arg("cell_size")               = 0.1,
          py::arg("min_points_per_cell")     = 1,
          py::arg("hole_fill_min_neighbors") = 6);
    m.def("smooth_mesh_along_axis",
          &smooth_mesh_along_axis,
          py::arg("vertices"),
          py::arg("edges"),
          py::arg("axis")       = 2,
          py::arg("smoothness") = 1.0,
          py::arg("num_passes") = 1);
    m.def("compute_vertex_normals",
          &compute_vertex_normals,
          py::arg("vertices"),
          py::arg("triangles"));
    m.def("find_ground_compatible_points",
          &find_ground_compatible_points,
          py::arg("points"),
          py::arg("normals"),
          py::arg("ground_control_point"),
          py::arg("ref_normal"),
          py::arg("max_distance"),
          py::arg("max_angle_deg"));
    m.def("detect_dominant_plane",
          &detect_dominant_plane,
          py::arg("points"),
          py::arg("normals"),
          py::arg("hypothesis_indices"),
          py::arg("eval_indices"),
          py::arg("distance_threshold"),
          py::arg("angle_threshold_deg"));
    m.def("detect_dominant_plane_batched",
          &detect_dominant_plane_batched,
          py::arg("points"),
          py::arg("normals"),
          py::arg("hypothesis_indices"),
          py::arg("eval_indices"),
          py::arg("pack_info"),
          py::arg("distance_threshold"),
          py::arg("angle_threshold_deg"));
}
