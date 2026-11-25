// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cuboid_relocation.h"
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {
template <typename T, int N>
using View = torch::PackedTensorAccessor64<T, N>;

__global__ void constrain_perturbation_kernel(
    View<float, 2> positions, View<float, 2> proposal, View<int64_t, 1> ids,
    View<int64_t, 1> tracks, View<float, 2> centers, View<float, 3> rotations,
    View<float, 2> dimensions, float expansion) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= positions.size(0))
        return;
    int64_t lo = 0, hi = tracks.size(0);
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (tracks[mid] < ids[i])
            lo = mid + 1;
        else
            hi = mid;
    }
    // Geometry and row identities are validated when the model cache is built.
    float delta[3];
    bool blocked = false;
    for (int axis = 0; axis < 3; ++axis) {
        float local = 0.0f, change = 0.0f;
        for (int j = 0; j < 3; ++j) {
            local += (positions[i][j] - centers[lo][j]) * rotations[lo][j][axis];
            change += (proposal[i][j] - positions[i][j]) * rotations[lo][j][axis];
        }
        float half    = dimensions[lo][axis] * (0.5f * expansion);
        float current = fabsf(local) - half, proposed = fabsf(local + change) - half;
        bool reject = (current < 0 && proposed >= 0) || (current >= 0 && proposed > current);
        delta[axis] = reject ? 0.0f : change;
        blocked |= reject;
    }
    for (int j = 0; j < 3; ++j) {
        float change = 0.0f;
        for (int axis = 0; axis < 3; ++axis)
            change += rotations[lo][j][axis] * delta[axis];
        positions[i][j] = blocked ? positions[i][j] + change : proposal[i][j];
    }
}
} // namespace

void constrain_perturbation(const torch::Tensor& positions, const torch::Tensor& proposal,
                            const torch::Tensor& ids, const torch::Tensor& tracks,
                            const torch::Tensor& centers, const torch::Tensor& rotations,
                            const torch::Tensor& dimensions, double expansion) {
    using cuboid_relocation::check_vector;
    check_vector(ids, positions, torch::kInt64, "ids");
    check_vector(tracks, positions, torch::kInt64, "tracks");
    for (const auto& value : {positions, proposal, centers, rotations, dimensions})
        TORCH_CHECK(value.is_cuda() && value.device() == positions.device() &&
                        value.layout() == torch::kStrided && value.scalar_type() == torch::kFloat32,
                    "cuboid perturbation requires same-device CUDA float32 tensors");
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3 && proposal.sizes() == positions.sizes() &&
                    ids.numel() == positions.size(0),
                "invalid perturbation row shapes");
    TORCH_CHECK(centers.dim() == 2 && centers.size(0) == tracks.numel() && centers.size(1) == 3 &&
                    dimensions.sizes() == centers.sizes() && rotations.dim() == 3 &&
                    rotations.size(0) == tracks.numel() && rotations.size(1) == 3 && rotations.size(2) == 3,
                "invalid cuboid geometry shapes");
    TORCH_CHECK(tracks.numel() > 0 && expansion > 0, "cuboid geometry must be nonempty and expansion positive");
    if (positions.size(0) == 0)
        return;
    const c10::cuda::CUDAGuard guard(positions.device());
    constrain_perturbation_kernel<<<(positions.size(0) + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        positions.packed_accessor64<float, 2>(), proposal.packed_accessor64<float, 2>(),
        ids.packed_accessor64<int64_t, 1>(), tracks.packed_accessor64<int64_t, 1>(),
        centers.packed_accessor64<float, 2>(), rotations.packed_accessor64<float, 3>(),
        dimensions.packed_accessor64<float, 2>(), float(expansion));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
