// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include "knn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_check_version", &KnnCheckVersion);
    m.def("knn_points_idx", &KNearestNeighborIdx);
    m.def("knn_points_backward", &KNearestNeighborBackward);
}
