// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

torch::Tensor pad_cubemap_forward_cuda(
    torch::Tensor const& cubemap,
    torch::Tensor const& border_source_indices,
    torch::Tensor const& border_destination_indices,
    torch::Tensor const& corner_destination_indices);

torch::Tensor pad_cubemap_backward_cuda(
    torch::Tensor const& grad_output,
    torch::Tensor const& single_border_source_indices,
    torch::Tensor const& single_border_destination_indices,
    torch::Tensor const& repeated_border_source_indices,
    torch::Tensor const& repeated_border_destination_indices,
    int64_t repeated_edge_source_count);
