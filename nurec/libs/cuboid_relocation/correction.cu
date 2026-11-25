// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cuboid_relocation.h"

#include <cfloat>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {
constexpr int kThreads        = 256;
constexpr int64_t kIndexLimit = int64_t{1} << 31;
template <typename T>
using View = torch::PackedTensorAccessor64<T, 1>;
template <typename T>
using View2 = torch::PackedTensorAccessor64<T, 2>;
using cuboid_relocation::check_vector;

// Global row IDs are disjoint across tracks: one histogram counts all donors.
__global__ void count_and_keep_kernel(View<int64_t> order, View<bool> sorted_dead,
                                      View<int64_t> group, View2<int64_t> metadata,
                                      View<int64_t> sources, View<int> occurrences, View<int64_t> representatives, View<bool> keep) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < sources.size(0) && atomicAdd(&occurrences[sources[i]], 1) == 0)
        representatives[sources[i]] = i;
    if (i < order.size(0))
        keep[order[i]] = !(sorted_dead[i] && metadata[group[i]][1]);
}

// Correction arithmetic adapted from gsplat/cuda/csrc/RelocationCUDA.cu:
// Copyright 2025 the Regents of the University of California, Nerfstudio Team
// and contributors. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this code except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
__global__ void correct_relocation_kernel(View<int64_t> sources, View<int> occurrences, View<int64_t> representatives,
                                          View<float> opacities, View2<float> scales,
                                          View2<float> binoms, float min_opacity,
                                          View<float> revised_opacities, View2<float> revised_scales) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= sources.size(0))
        return;
    int64_t source = sources[i];
    if (representatives[source] != i)
        return;
    int n = int(min(int64_t(occurrences[source]) + 1, binoms.size(0)));
    // Match gsplat's arithmetic, including its stable geometric-sum opacity.
    float opacity         = fmaxf(opacities[source], min_opacity);
    float complement_root = powf(1.0f - opacity, 1.0f / n);
    float root_power = 1.0f, geometric_sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        geometric_sum += root_power;
        root_power *= complement_root;
    }
    float new_opacity    = opacity / geometric_sum;
    revised_opacities[i] = new_opacity;
    // Match gsplat's positive-integrand quadrature; the alternating binomial
    // series loses precision and can produce negative scales near opacity 1.
    // Tiny opacities use the leading term to avoid fast-math flush-to-zero.
    float denom_sum = n * new_opacity;
    if (denom_sum > FLT_EPSILON) {
        constexpr int num_steps = 24;
        constexpr float step    = 0.25f;
        denom_sum               = 0.0f;
        for (int j = 0; j <= num_steps; ++j) {
            float u = j * step;
            float x = new_opacity * expf(-u * u);
            float f = -expm1f(n * log1pf(-x));
            denom_sum += (j == 0) ? 0.5f * f : f;
        }
        denom_sum *= step * 1.1283791670955126f; // 2 / sqrt(pi)
    }
    float coeff = opacity / denom_sum;
    for (int j = 0; j < 3; ++j)
        revised_scales[i][j] = coeff * scales[source][j];
}

// The correction launch completes before expansion; representative outputs
// are read-only here, so repeated donors never race with their source output.
__global__ void expand_corrections_kernel(View<int64_t> sources, View<int64_t> representatives,
                                          View<float> opacities, View2<float> scales) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= sources.size(0))
        return;
    int64_t representative = representatives[sources[i]];
    if (representative == i)
        return;
    opacities[i] = opacities[representative];
    for (int j = 0; j < 3; ++j)
        scales[i][j] = scales[representative][j];
}

} // namespace

void correct_relocation(const torch::Tensor& order, const torch::Tensor& sorted_dead,
                        const torch::Tensor& group, const torch::Tensor& metadata,
                        const torch::Tensor& sources, const torch::Tensor& opacities,
                        const torch::Tensor& scales, const torch::Tensor& binoms,
                        double min_opacity, const torch::Tensor& occurrences,
                        const torch::Tensor& keep, const torch::Tensor& revised_opacities,
                        const torch::Tensor& revised_scales) {
    check_vector(order, order, torch::kInt64, "order");
    check_vector(sorted_dead, order, torch::kBool, "sorted_dead");
    check_vector(group, order, torch::kInt64, "group");
    check_vector(sources, order, torch::kInt64, "sources");
    check_vector(opacities, order, torch::kFloat32, "opacities");
    check_vector(occurrences, order, torch::kInt32, "occurrences");
    check_vector(keep, order, torch::kBool, "keep");
    check_vector(revised_opacities, order, torch::kFloat32, "revised_opacities");
    for (const auto& value : {sorted_dead, group, opacities, occurrences, keep})
        TORCH_CHECK(value.numel() == order.numel(), "relocation row lengths must match");
    TORCH_CHECK(revised_opacities.numel() == sources.numel(), "revised opacity length must match sources");
    auto check_matrix = [&](const torch::Tensor& value, at::ScalarType dtype,
                            int64_t rows, int64_t columns) {
        TORCH_CHECK(value.is_cuda() && value.device() == order.device() &&
                        value.layout() == torch::kStrided && value.scalar_type() == dtype &&
                        value.dim() == 2 && value.size(0) == rows && value.size(1) == columns,
                    "invalid relocation matrix device, dtype or shape");
    };
    TORCH_CHECK(metadata.dim() == 2 && binoms.dim() == 2, "expected matrix inputs");
    check_matrix(metadata, torch::kInt64, metadata.size(0), 2);
    check_matrix(scales, torch::kFloat32, order.numel(), 3);
    check_matrix(binoms, torch::kFloat32, binoms.size(0), binoms.size(0));
    check_matrix(revised_scales, torch::kFloat32, sources.numel(), 3);
    TORCH_CHECK(binoms.size(0) > 0 && binoms.size(0) < kIndexLimit, "invalid binomial table size");
    TORCH_CHECK(min_opacity > 0.0 && min_opacity < 1.0, "min_opacity must be in (0,1)");
    TORCH_CHECK(order.numel() > 0 || sources.numel() == 0, "empty rows require empty sources");
    TORCH_CHECK(order.numel() == 0 || metadata.size(0) > 0, "rows require track metadata");
    const c10::cuda::CUDAGuard guard(order.device());
    int64_t work = std::max(order.numel(), sources.numel());
    if (work == 0)
        return;
    auto representatives = torch::empty_like(order);
    count_and_keep_kernel<<<(work + kThreads - 1) / kThreads, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        order.packed_accessor64<int64_t, 1>(), sorted_dead.packed_accessor64<bool, 1>(),
        group.packed_accessor64<int64_t, 1>(), metadata.packed_accessor64<int64_t, 2>(),
        sources.packed_accessor64<int64_t, 1>(), occurrences.packed_accessor64<int, 1>(),
        representatives.packed_accessor64<int64_t, 1>(), keep.packed_accessor64<bool, 1>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (sources.numel() == 0)
        return;
    correct_relocation_kernel<<<(sources.numel() + kThreads - 1) / kThreads, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        sources.packed_accessor64<int64_t, 1>(), occurrences.packed_accessor64<int, 1>(),
        representatives.packed_accessor64<int64_t, 1>(),
        opacities.packed_accessor64<float, 1>(), scales.packed_accessor64<float, 2>(),
        binoms.packed_accessor64<float, 2>(), float(min_opacity),
        revised_opacities.packed_accessor64<float, 1>(), revised_scales.packed_accessor64<float, 2>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    expand_corrections_kernel<<<(sources.numel() + kThreads - 1) / kThreads, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        sources.packed_accessor64<int64_t, 1>(), representatives.packed_accessor64<int64_t, 1>(),
        revised_opacities.packed_accessor64<float, 1>(), revised_scales.packed_accessor64<float, 2>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
