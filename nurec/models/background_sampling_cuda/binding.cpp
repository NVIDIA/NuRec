// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "pad_cubemap_forward",
        &pad_cubemap_forward_cuda,
        py::arg("cubemap"),
        py::arg("border_source_indices"),
        py::arg("border_destination_indices"),
        py::arg("corner_destination_indices"),
        "Materialize a one-texel seamless border around CUDA cubemap faces.");
    module.def(
        "pad_cubemap_backward",
        &pad_cubemap_backward_cuda,
        py::arg("grad_output"),
        py::arg("single_border_source_indices"),
        py::arg("single_border_destination_indices"),
        py::arg("repeated_border_source_indices"),
        py::arg("repeated_border_destination_indices"),
        py::arg("repeated_edge_source_count"),
        "Accumulate a padded cubemap gradient into the unpadded CUDA cubemap.");
}
