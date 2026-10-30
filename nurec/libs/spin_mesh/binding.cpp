// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "utils.h"

#include <limits>

std::tuple<torch::Tensor, torch::Tensor> compute_mesh_normals_in_lidar_spin(
    const torch::Tensor& points,
    const int n_elevation_bins,
    const int n_azimuth_bins,
    const double el_min,
    const double el_max) {
    CHECK_INPUT(points);
    return compute_mesh_normals_in_lidar_spin_cu(points, n_elevation_bins, n_azimuth_bins, el_min, el_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_mesh_normals_in_lidar_spin",
          &compute_mesh_normals_in_lidar_spin,
          py::arg("points"),
          py::arg("n_elevation_bins") = 256,
          py::arg("n_azimuth_bins")   = 4096,
          py::arg("el_min")           = std::numeric_limits<double>::quiet_NaN(),
          py::arg("el_max")           = std::numeric_limits<double>::quiet_NaN());
}
