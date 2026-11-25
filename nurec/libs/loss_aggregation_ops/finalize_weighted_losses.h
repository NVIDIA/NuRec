// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <ATen/core/Tensor.h>

namespace nurec::loss_aggregation_ops {

void launch_finalize_accumulated_losses_fwd(
    const at::Tensor& weighted_vector,
    at::TensorList scalar_values,
    const std::vector<double>& scalar_weights,
    at::Tensor& total,
    at::Tensor& weighted_values);

void launch_finalize_accumulated_losses_bwd(
    const at::Tensor* v_total,
    const at::Tensor* v_weighted_values,
    const std::vector<double>& scalar_weights,
    at::Tensor& v_weighted_vector,
    at::TensorList v_scalar_values);

} // namespace nurec::loss_aggregation_ops
