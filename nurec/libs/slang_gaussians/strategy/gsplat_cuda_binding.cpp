// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gsplat_cuda.h"

#include <ku/common_host.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("update_gradient_buffers", &update_gradient_buffers_cuda, "Update gradient accumulation buffers for GSplat densification (CUDA)");
}
