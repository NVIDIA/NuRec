// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

// Inference-only contiguous CUDA FP32 [20,H,W] -> FP32 [H,W].
void categorical_entropy_out(at::Tensor const& logits, at::Tensor const& output);

// Write fresh labels/mask and optionally replace uncertain logits in place.
void mask2former_postprocess_out(at::Tensor const& logits, at::Tensor const& pred_labels, int64_t building_id,
                                 bool estimate_logits, at::Tensor const& entropy, at::Tensor const& labels,
                                 at::Tensor const& mask);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("categorical_entropy_out", &categorical_entropy_out);
    m.def("mask2former_postprocess_out", &mask2former_postprocess_out);
}
