// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "track_frame_slots.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ku/common.cuh>

namespace {

// Accept tensor views without allocating contiguous copies.
struct StridedInt64 {
    const int64_t* data;
    int64_t stride;
    __device__ __forceinline__ int64_t operator[](int64_t i) const { return data[i * stride]; }
};

__global__ void nearest_track_frame_slots_kernel(
    int64_t target, int64_t num_frames, int64_t num_tracks,
    StridedInt64 timestamps, StridedInt64 starts, StridedInt64 counts,
    int64_t* slots, int64_t slot_stride) {
    const int64_t track = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (track >= num_tracks)
        return;
    const int64_t count = counts[track];
    if (count <= 1 || num_frames == 0) {
        slots[track * slot_stride] = 0;
        return;
    }
    const int64_t start = starts[track];
    int64_t lo          = start;
    int64_t hi          = start + count;
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (timestamps[mid] < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    const int64_t right        = min(lo, start + count - 1);
    const int64_t left         = max(right - 1, start);
    const int64_t left_diff    = target - timestamps[left];
    const int64_t right_diff   = timestamps[right] - target;
    const int64_t left_delta   = left_diff < 0 ? -left_diff : left_diff;
    const int64_t right_delta  = right_diff < 0 ? -right_diff : right_diff;
    slots[track * slot_stride] = (left_delta <= right_delta ? left : right) - start;
}

} // namespace

void nearest_track_frame_slots_cu(
    int64_t target, const torch::Tensor& timestamps, const torch::Tensor& packinfo, const torch::Tensor& slots) {
    const auto check_tensor = [&](const torch::Tensor& tensor, const char* name, int64_t ndim) {
        TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(tensor.device() == timestamps.device(), name, " must be on the same CUDA device as timestamps");
        TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must have dtype int64");
        TORCH_CHECK(tensor.dim() == ndim, name, " must have rank ", ndim);
    };
    check_tensor(timestamps, "timestamps", 1);
    check_tensor(packinfo, "packinfo", 2);
    check_tensor(slots, "slots", 1);
    TORCH_CHECK(packinfo.size(1) == 2, "packinfo must have shape [T, 2]");
    TORCH_CHECK(packinfo.size(0) == slots.numel(), "packinfo and slots must have the same length");
    TORCH_CHECK(slots.numel() <= 1 || slots.stride(0) > 0, "slots must not overlap internally");
    if (slots.numel() == 0)
        return;

    const c10::cuda::CUDAGuard guard(timestamps.device());
    const auto blocks = div_round_up<int64_t>(slots.numel(), TRACK_FRAME_SLOTS_THREADS_PER_BLOCK);
    nearest_track_frame_slots_kernel<<<blocks, TRACK_FRAME_SLOTS_THREADS_PER_BLOCK, 0,
                                       c10::cuda::getCurrentCUDAStream()>>>(
        target, timestamps.numel(), slots.numel(),
        {timestamps.data_ptr<int64_t>(), timestamps.stride(0)},
        {packinfo.data_ptr<int64_t>(), packinfo.stride(0)},
        {packinfo.data_ptr<int64_t>() + packinfo.stride(1), packinfo.stride(0)},
        slots.data_ptr<int64_t>(), slots.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
