// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "utils.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

namespace {

constexpr int64_t kCubemapFaceCount         = 6;
constexpr int64_t kCubemapEdgesPerFace      = 4;
constexpr int64_t kCubemapCornersPerFace    = 4;
constexpr int64_t kCubemapCornerSourceCount = 3;
constexpr int64_t kSeamlessPaddingPerAxis   = 2;
constexpr int64_t kThreadsPerBlock          = 256;
// Match the ATen reference's left-associated additions and CUDA scalar
// division-by-3 lowering. The fp32 parity tests intentionally require this
// corner arithmetic to remain bit-exact.
constexpr float kCubemapCornerScale = 1.0F / static_cast<float>(kCubemapCornerSourceCount);

__global__ void pad_cubemap_forward_kernel(
    float const* __restrict__ cubemap,
    int64_t const* __restrict__ border_source_indices,
    int64_t const* __restrict__ border_destination_indices,
    int64_t const* __restrict__ corner_destination_indices,
    float* __restrict__ padded,
    int64_t cubemap_size,
    int64_t n_channels) {
    int64_t const padded_size     = cubemap_size + kSeamlessPaddingPerAxis;
    int64_t const interior_blocks = kCubemapFaceCount * cubemap_size;
    int64_t const edge_blocks     = kCubemapFaceCount * kCubemapEdgesPerFace;
    int64_t const block_index     = blockIdx.x;

    if (block_index < interior_blocks) {
        int64_t const face         = block_index / cubemap_size;
        int64_t const row          = block_index - face * cubemap_size;
        int64_t const input_offset = (face * cubemap_size * cubemap_size + row * cubemap_size) * n_channels;
        int64_t const output_offset =
            (face * padded_size * padded_size + (row + 1) * padded_size + 1) * n_channels;
        for (int64_t element = threadIdx.x; element < cubemap_size * n_channels; element += blockDim.x) {
            padded[output_offset + element] = cubemap[input_offset + element];
        }
        return;
    }

    int64_t const border_block = block_index - interior_blocks;
    if (border_block < edge_blocks) {
        int64_t const edge_offset = border_block * cubemap_size;
        for (int64_t element = threadIdx.x; element < cubemap_size * n_channels; element += blockDim.x) {
            int64_t const edge_texel = edge_offset + element / n_channels;
            int64_t const channel    = element % n_channels;
            padded[border_destination_indices[edge_texel] * n_channels + channel] =
                cubemap[border_source_indices[edge_texel] * n_channels + channel];
        }
        return;
    }

    int64_t const edge_texel_count = edge_blocks * cubemap_size;
    int64_t const corner_count     = kCubemapFaceCount * kCubemapCornersPerFace;
    for (int64_t element = threadIdx.x; element < corner_count * n_channels; element += blockDim.x) {
        int64_t const corner        = element / n_channels;
        int64_t const channel       = element % n_channels;
        int64_t const source_offset = edge_texel_count + corner * kCubemapCornerSourceCount;
        float const first           = cubemap[border_source_indices[source_offset] * n_channels + channel];
        float const second          = cubemap[border_source_indices[source_offset + 1] * n_channels + channel];
        float const third           = cubemap[border_source_indices[source_offset + 2] * n_channels + channel];
        float const first_pair_sum  = __fadd_rn(first, second);
        float const corner_sum      = __fadd_rn(first_pair_sum, third);
        padded[corner_destination_indices[corner] * n_channels + channel] =
            __fmul_rn(corner_sum, kCubemapCornerScale);
    }
}

__global__ void copy_cubemap_interior_gradient_kernel(
    float const* __restrict__ grad_output,
    float* __restrict__ grad_cubemap,
    int64_t cubemap_size,
    int64_t n_channels) {
    int64_t const face        = blockIdx.x / cubemap_size;
    int64_t const row         = blockIdx.x - face * cubemap_size;
    int64_t const padded_size = cubemap_size + kSeamlessPaddingPerAxis;
    int64_t const input_offset =
        (face * padded_size * padded_size + (row + 1) * padded_size + 1) * n_channels;
    int64_t const output_offset = (face * cubemap_size * cubemap_size + row * cubemap_size) * n_channels;
    for (int64_t element = threadIdx.x; element < cubemap_size * n_channels; element += blockDim.x) {
        grad_cubemap[output_offset + element] = grad_output[input_offset + element];
    }
}

__global__ void accumulate_cubemap_border_gradient_kernel(
    float const* __restrict__ grad_output,
    int64_t const* __restrict__ single_border_source_indices,
    int64_t const* __restrict__ single_border_destination_indices,
    int64_t const* __restrict__ repeated_border_source_indices,
    int64_t const* __restrict__ repeated_border_destination_indices,
    float* __restrict__ grad_cubemap,
    int64_t n_single_sources,
    int64_t n_repeated_sources,
    int64_t repeated_source_count,
    int64_t repeated_edge_source_count,
    int64_t n_channels) {
    int64_t const element        = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const total_elements = (n_single_sources + n_repeated_sources) * n_channels;
    if (element >= total_elements) {
        return;
    }

    int64_t const source_group = element / n_channels;
    int64_t const channel      = element % n_channels;
    if (source_group < n_single_sources) {
        int64_t const source      = single_border_source_indices[source_group];
        int64_t const destination = single_border_destination_indices[source_group];
        grad_cubemap[source * n_channels + channel] += grad_output[destination * n_channels + channel];
        return;
    }

    int64_t const repeated_group     = source_group - n_single_sources;
    int64_t const source             = repeated_border_source_indices[repeated_group];
    int64_t const destination_offset = repeated_group * repeated_source_count;
    float accumulated                = 0.0F;
    for (int64_t contribution = 0; contribution < repeated_source_count; ++contribution) {
        int64_t const destination = repeated_border_destination_indices[destination_offset + contribution];
        float value               = grad_output[destination * n_channels + channel];
        if (contribution >= repeated_edge_source_count) {
            value = __fmul_rn(value, kCubemapCornerScale);
        }
        accumulated = __fadd_rn(accumulated, value);
    }
    grad_cubemap[source * n_channels + channel] += accumulated;
}

void check_cuda_float_tensor(
    char const* function,
    torch::TensorArg const& argument,
    torch::Tensor const& tensor) {
    torch::checkContiguous(function, argument);
    torch::checkScalarType(function, argument, torch::kFloat);
    torch::checkDeviceType(function, {tensor}, torch::kCUDA);
}

void check_cuda_index_tensor(
    char const* function,
    torch::TensorArg const& argument,
    torch::Tensor const& tensor) {
    torch::checkContiguous(function, argument);
    torch::checkScalarType(function, argument, torch::kLong);
    torch::checkDeviceType(function, {tensor}, torch::kCUDA);
}

int64_t blocks_for(int64_t elements) {
    return (elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
}

void record_stream(torch::Tensor const& tensor, c10::cuda::CUDAStream const& stream) {
    if (tensor.numel() > 0) {
        c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
}

} // namespace

torch::Tensor pad_cubemap_forward_cuda(
    torch::Tensor const& cubemap,
    torch::Tensor const& border_source_indices,
    torch::Tensor const& border_destination_indices,
    torch::Tensor const& corner_destination_indices) {
    torch::TensorArg cubemap_arg{cubemap, "cubemap", 1};
    torch::TensorArg border_source_arg{border_source_indices, "border_source_indices", 2};
    torch::TensorArg border_destination_arg{border_destination_indices, "border_destination_indices", 3};
    torch::TensorArg corner_destination_arg{corner_destination_indices, "corner_destination_indices", 4};
    torch::checkDim(__func__, cubemap_arg, 4);
    torch::checkDim(__func__, border_source_arg, 1);
    torch::checkDim(__func__, border_destination_arg, 1);
    torch::checkDim(__func__, corner_destination_arg, 1);
    check_cuda_float_tensor(__func__, cubemap_arg, cubemap);
    check_cuda_index_tensor(__func__, border_source_arg, border_source_indices);
    check_cuda_index_tensor(__func__, border_destination_arg, border_destination_indices);
    check_cuda_index_tensor(__func__, corner_destination_arg, corner_destination_indices);
    torch::checkAllSameGPU(
        __func__, {cubemap_arg, border_source_arg, border_destination_arg, corner_destination_arg});

    TORCH_CHECK(cubemap.size(0) == kCubemapFaceCount, "cubemap must have six faces");
    TORCH_CHECK(cubemap.size(1) > 0 && cubemap.size(1) == cubemap.size(2), "cubemap faces must be non-empty squares");
    TORCH_CHECK(cubemap.size(3) > 0, "cubemap must have at least one channel");
    int64_t const cubemap_size       = cubemap.size(1);
    int64_t const n_channels         = cubemap.size(3);
    int64_t const edge_texel_count   = kCubemapFaceCount * kCubemapEdgesPerFace * cubemap_size;
    int64_t const corner_count       = kCubemapFaceCount * kCubemapCornersPerFace;
    int64_t const border_entry_count = edge_texel_count + corner_count * kCubemapCornerSourceCount;
    TORCH_CHECK(border_source_indices.numel() == border_entry_count, "unexpected border source topology size");
    TORCH_CHECK(border_destination_indices.numel() == border_entry_count, "unexpected border destination topology size");
    TORCH_CHECK(corner_destination_indices.numel() == corner_count, "unexpected corner destination topology size");

    int64_t const padded_size = cubemap_size + kSeamlessPaddingPerAxis;
    torch::Tensor padded      = torch::empty(
        {kCubemapFaceCount, padded_size, padded_size, n_channels}, cubemap.options());
    c10::cuda::OptionalCUDAGuard const device_guard(torch::device_of(cubemap));
    c10::cuda::CUDAStream const stream = c10::cuda::getCurrentCUDAStream(cubemap.get_device());
    record_stream(cubemap, stream);
    record_stream(border_source_indices, stream);
    record_stream(border_destination_indices, stream);
    record_stream(corner_destination_indices, stream);
    record_stream(padded, stream);
    int64_t const blocks = kCubemapFaceCount * cubemap_size + kCubemapFaceCount * kCubemapEdgesPerFace + 1;
    pad_cubemap_forward_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        cubemap.data_ptr<float>(),
        border_source_indices.data_ptr<int64_t>(),
        border_destination_indices.data_ptr<int64_t>(),
        corner_destination_indices.data_ptr<int64_t>(),
        padded.data_ptr<float>(),
        cubemap_size,
        n_channels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return padded;
}

torch::Tensor pad_cubemap_backward_cuda(
    torch::Tensor const& grad_output,
    torch::Tensor const& single_border_source_indices,
    torch::Tensor const& single_border_destination_indices,
    torch::Tensor const& repeated_border_source_indices,
    torch::Tensor const& repeated_border_destination_indices,
    int64_t repeated_edge_source_count) {
    torch::TensorArg grad_output_arg{grad_output, "grad_output", 1};
    torch::TensorArg single_source_arg{single_border_source_indices, "single_border_source_indices", 2};
    torch::TensorArg single_destination_arg{single_border_destination_indices, "single_border_destination_indices", 3};
    torch::TensorArg repeated_source_arg{repeated_border_source_indices, "repeated_border_source_indices", 4};
    torch::TensorArg repeated_destination_arg{
        repeated_border_destination_indices, "repeated_border_destination_indices", 5};
    torch::checkDim(__func__, grad_output_arg, 4);
    torch::checkDim(__func__, single_source_arg, 1);
    torch::checkDim(__func__, single_destination_arg, 1);
    torch::checkDim(__func__, repeated_source_arg, 1);
    torch::checkDim(__func__, repeated_destination_arg, 2);
    check_cuda_float_tensor(__func__, grad_output_arg, grad_output);
    check_cuda_index_tensor(__func__, single_source_arg, single_border_source_indices);
    check_cuda_index_tensor(__func__, single_destination_arg, single_border_destination_indices);
    check_cuda_index_tensor(__func__, repeated_source_arg, repeated_border_source_indices);
    check_cuda_index_tensor(__func__, repeated_destination_arg, repeated_border_destination_indices);
    torch::checkAllSameGPU(
        __func__,
        {grad_output_arg,
         single_source_arg,
         single_destination_arg,
         repeated_source_arg,
         repeated_destination_arg});

    TORCH_CHECK(grad_output.size(0) == kCubemapFaceCount, "grad_output must have six faces");
    TORCH_CHECK(
        grad_output.size(1) >= kSeamlessPaddingPerAxis + 1 && grad_output.size(1) == grad_output.size(2),
        "grad_output faces must be padded non-empty squares");
    TORCH_CHECK(grad_output.size(3) > 0, "grad_output must have at least one channel");
    TORCH_CHECK(
        single_border_source_indices.numel() == single_border_destination_indices.numel(),
        "single-source topology tensors must have equal lengths");
    TORCH_CHECK(
        repeated_border_source_indices.numel() == repeated_border_destination_indices.size(0),
        "repeated-source topology tensors must have matching leading dimensions");
    TORCH_CHECK(
        repeated_edge_source_count >= 0 && repeated_edge_source_count <= repeated_border_destination_indices.size(1),
        "repeated edge source count is outside the topology width");

    int64_t const cubemap_size = grad_output.size(1) - kSeamlessPaddingPerAxis;
    int64_t const n_channels   = grad_output.size(3);
    int64_t const expected_border_entry_count =
        kCubemapFaceCount * kCubemapEdgesPerFace * cubemap_size + kCubemapFaceCount * kCubemapCornersPerFace * kCubemapCornerSourceCount;
    int64_t const actual_border_entry_count =
        single_border_source_indices.numel() + repeated_border_source_indices.numel() * repeated_border_destination_indices.size(1);
    TORCH_CHECK(
        actual_border_entry_count == expected_border_entry_count,
        "backward topology size does not match grad_output cubemap size");
    torch::Tensor grad_cubemap = torch::empty(
        {kCubemapFaceCount, cubemap_size, cubemap_size, n_channels}, grad_output.options());
    c10::cuda::OptionalCUDAGuard const device_guard(torch::device_of(grad_output));
    c10::cuda::CUDAStream const stream = c10::cuda::getCurrentCUDAStream(grad_output.get_device());
    record_stream(grad_output, stream);
    record_stream(single_border_source_indices, stream);
    record_stream(single_border_destination_indices, stream);
    record_stream(repeated_border_source_indices, stream);
    record_stream(repeated_border_destination_indices, stream);
    record_stream(grad_cubemap, stream);
    copy_cubemap_interior_gradient_kernel<<<
        kCubemapFaceCount * cubemap_size, kThreadsPerBlock, 0, stream>>>(
        grad_output.data_ptr<float>(), grad_cubemap.data_ptr<float>(), cubemap_size, n_channels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int64_t const n_single_sources   = single_border_source_indices.numel();
    int64_t const n_repeated_sources = repeated_border_source_indices.numel();
    int64_t const border_elements    = (n_single_sources + n_repeated_sources) * n_channels;
    if (border_elements > 0) {
        accumulate_cubemap_border_gradient_kernel<<<blocks_for(border_elements), kThreadsPerBlock, 0, stream>>>(
            grad_output.data_ptr<float>(),
            single_border_source_indices.data_ptr<int64_t>(),
            single_border_destination_indices.data_ptr<int64_t>(),
            repeated_border_source_indices.data_ptr<int64_t>(),
            repeated_border_destination_indices.data_ptr<int64_t>(),
            grad_cubemap.data_ptr<float>(),
            n_single_sources,
            n_repeated_sources,
            repeated_border_destination_indices.size(1),
            repeated_edge_source_count,
            n_channels);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return grad_cubemap;
}
