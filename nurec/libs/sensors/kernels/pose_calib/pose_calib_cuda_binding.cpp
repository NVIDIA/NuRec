// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "pose_calib_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pose_calib_forward_cuda, "Pose calibration forward (CUDA)");
    m.def("backward", &pose_calib_backward_cuda, "Pose calibration backward (CUDA)");
}
