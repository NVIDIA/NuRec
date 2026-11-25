// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cuboid_relocation.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {
constexpr int kThreads = 256;
template <typename T>
using View = torch::PackedTensorAccessor64<T, 1>;
template <typename T>
using View2 = torch::PackedTensorAccessor64<T, 2>;

using cuboid_relocation::check_vector;

int64_t check_offsets(const torch::Tensor& offsets, const torch::Tensor& reference) {
    check_vector(offsets, reference, torch::kInt64, "offsets");
    TORCH_CHECK(offsets.numel() >= 1, "offsets must contain at least one boundary");
    return offsets.numel() - 1;
}

// One block per track; suppressed tracks keep their dead mask and have no donors.
__global__ void prepare_tracks_kernel(View<int64_t> order, View<int64_t> offsets,
                                      View<float> weights, View<bool> dead,
                                      View<bool> sorted_dead, View2<int64_t> metadata,
                                      View<int64_t> group) {
    unsigned int t = threadIdx.x, g = blockIdx.x;
    int64_t begin = offsets[g], end = offsets[g + 1];
    __shared__ int dead_counts[kThreads], nonzero_weights[kThreads];
    int ndead = 0, nonzero = 0;
    for (int64_t i = begin + t; i < end; i += kThreads) {
        bool d         = dead[order[i]];
        sorted_dead[i] = d;
        group[i]       = int64_t(g);
        ndead += d ? 1 : 0;
        nonzero |= (!d && weights[order[i]] != 0.0) ? 1 : 0;
    }
    dead_counts[t]     = ndead;
    nonzero_weights[t] = nonzero;
    __syncthreads();
    for (unsigned int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            dead_counts[t] += dead_counts[t + stride];
            nonzero_weights[t] |= nonzero_weights[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        metadata[g][0] = dead_counts[0];
        metadata[g][1] = nonzero_weights[0];
    }
}

// Chunked inclusive scan with a double-precision carry for each track.
__global__ void track_cdf_kernel(View<float> weights, View<int64_t> offsets, View<double> cdf) {
    unsigned int t = threadIdx.x, g = blockIdx.x;
    int64_t begin = offsets[g], end = offsets[g + 1];
    __shared__ double values[kThreads];
    double carry = 0.0;
    for (int64_t chunk = begin; chunk < end; chunk += kThreads) {
        int64_t i = chunk + t;
        values[t] = i < end ? double(weights[i]) : 0.0;
        __syncthreads();
        for (unsigned int step = 1; step < kThreads; step <<= 1) {
            double previous = t >= step ? values[t - step] : 0.0;
            __syncthreads();
            values[t] += previous;
            __syncthreads();
        }
        if (i < end)
            cdf[i] = carry + values[t];
        carry += values[kThreads - 1];
        __syncthreads();
    }
}

// Inverse-CDF sampling from supplied uniforms, independently within each track.
__global__ void sample_tracks_kernel(View<int64_t> rows, View<double> cdf,
                                     View<int64_t> offsets, View<int64_t> draw_offsets,
                                     View<double> uniforms, View<int64_t> sources) {
    unsigned int g = blockIdx.x;
    if (draw_offsets[g] == draw_offsets[g + 1])
        return;
    int64_t begin = offsets[g], end = offsets[g + 1];
    if (end <= begin)
        return;
    double total = cdf[end - 1];
    for (int64_t i = draw_offsets[g] + threadIdx.x; i < draw_offsets[g + 1]; i += kThreads) {
        double target = uniforms[i] * total;
        int64_t lo = begin, hi = end;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            // A rounded upper endpoint must select the last positive interval,
            // never a trailing zero-weight row.
            bool below = target >= total ? cdf[mid] < total : cdf[mid] <= target;
            if (below)
                lo = mid + 1;
            else
                hi = mid;
        }
        sources[i] = rows[lo < end - 1 ? lo : end - 1];
    }
}

} // namespace

void prepare_tracks(const torch::Tensor& order, const torch::Tensor& offsets,
                    const torch::Tensor& weights, const torch::Tensor& dead,
                    const torch::Tensor& sorted_dead, const torch::Tensor& metadata,
                    const torch::Tensor& group) {
    check_vector(order, order, torch::kInt64, "order");
    const int64_t groups = check_offsets(offsets, order);
    check_vector(weights, order, torch::kFloat32, "weights");
    check_vector(dead, order, torch::kBool, "dead");
    check_vector(sorted_dead, order, torch::kBool, "sorted_dead");
    check_vector(group, order, torch::kInt64, "group");
    TORCH_CHECK(weights.numel() == order.numel() && dead.numel() == order.numel() &&
                    sorted_dead.numel() == order.numel() && group.numel() == order.numel(),
                "prepare_tracks row lengths must match");
    TORCH_CHECK(metadata.is_cuda() && metadata.device() == order.device() &&
                    metadata.layout() == torch::kStrided && metadata.scalar_type() == torch::kInt64 &&
                    metadata.dim() == 2 && metadata.size(0) == groups && metadata.size(1) == 2,
                "metadata must be same-device CUDA int64[G,2]");
    TORCH_CHECK(groups <= order.numel(), "prepare_tracks requires nonempty partitions");
    if (groups == 0) {
        TORCH_CHECK(order.numel() == 0, "zero tracks require zero rows");
        return;
    }
    const c10::cuda::CUDAGuard guard(order.device());
    prepare_tracks_kernel<<<groups, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        order.packed_accessor64<int64_t, 1>(), offsets.packed_accessor64<int64_t, 1>(),
        weights.packed_accessor64<float, 1>(), dead.packed_accessor64<bool, 1>(),
        sorted_dead.packed_accessor64<bool, 1>(), metadata.packed_accessor64<int64_t, 2>(),
        group.packed_accessor64<int64_t, 1>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void track_cdf(const torch::Tensor& weights, const torch::Tensor& offsets,
               const torch::Tensor& cdf) {
    check_vector(weights, weights, torch::kFloat32, "weights");
    const int64_t groups = check_offsets(offsets, weights);
    check_vector(cdf, weights, torch::kFloat64, "cdf");
    TORCH_CHECK(cdf.numel() == weights.numel(), "CDF and weight lengths must match");
    if (groups == 0) {
        TORCH_CHECK(weights.numel() == 0, "zero tracks require zero candidates");
        return;
    }
    const c10::cuda::CUDAGuard guard(weights.device());
    track_cdf_kernel<<<groups, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        weights.packed_accessor64<float, 1>(), offsets.packed_accessor64<int64_t, 1>(),
        cdf.packed_accessor64<double, 1>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sample_tracks(const torch::Tensor& rows, const torch::Tensor& cdf,
                   const torch::Tensor& offsets, const torch::Tensor& draw_offsets,
                   const torch::Tensor& uniforms, const torch::Tensor& sources) {
    check_vector(rows, rows, torch::kInt64, "rows");
    const int64_t groups = check_offsets(offsets, rows);
    check_vector(cdf, rows, torch::kFloat64, "cdf");
    check_vector(draw_offsets, rows, torch::kInt64, "draw_offsets");
    check_vector(uniforms, rows, torch::kFloat64, "uniforms");
    check_vector(sources, rows, torch::kInt64, "sources");
    TORCH_CHECK(cdf.numel() == rows.numel(), "CDF and row lengths must match");
    TORCH_CHECK(draw_offsets.numel() == offsets.numel(), "partition counts must match");
    TORCH_CHECK(sources.numel() == uniforms.numel(), "source and uniform lengths must match");
    if (groups == 0) {
        TORCH_CHECK(rows.numel() == 0 && uniforms.numel() == 0, "zero tracks require zero rows and draws");
        return;
    }
    const c10::cuda::CUDAGuard guard(rows.device());
    sample_tracks_kernel<<<groups, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        rows.packed_accessor64<int64_t, 1>(), cdf.packed_accessor64<double, 1>(),
        offsets.packed_accessor64<int64_t, 1>(), draw_offsets.packed_accessor64<int64_t, 1>(),
        uniforms.packed_accessor64<double, 1>(), sources.packed_accessor64<int64_t, 1>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
