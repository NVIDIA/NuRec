// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gaussian_collective_payload_ops.h"

#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>

#include <cstdint>
#include <limits>
#include <utility>

namespace {

// A missing optional gradient intentionally zero-fills its referenced output
// segment. Keep that state distinct from a present tensor with zero elements:
// the latter must still fail the trusted-metadata range check when referenced.
constexpr int64_t kMissingGradientNumel = -1;

// Each trusted metadata row stores destination offset, length, input index,
// source offset, and cumulative block end, all as int64 values.
constexpr int64_t kValuesPerSegment = 5;

// CTA width for gradient packing. Each thread walks the independently tunable
// kGradientPackValuesPerBlock chunk with a blockDim.x stride.
constexpr int kThreadsPerBlock = 256;

template <typename scalar_t>
struct GradientPackInputs {
    const scalar_t* pointers[kMaxGradientPackInputs];
    int64_t numels[kMaxGradientPackInputs];
};

static_assert(
    sizeof(GradientPackInputs<float>) == 2048,
    "gradient-pack input descriptor must match the documented 2,048-byte ABI table");
static_assert(
    sizeof(GradientPackInputs<float>) + 2 * sizeof(void*) + 4 * sizeof(int64_t) == 2096,
    "gradient-pack launch arguments must match the documented 2,096-byte ABI payload");

template <typename scalar_t>
__global__ void pack_gradients_kernel(
    scalar_t* __restrict__ output,
    int64_t output_numel,
    const int64_t* __restrict__ segments,
    int64_t num_segments,
    int64_t num_blocks,
    int64_t num_inputs,
    GradientPackInputs<scalar_t> inputs) {
    const int64_t block_index    = blockIdx.x;
    const bool valid_block_index = block_index < num_blocks;
    if (C10_UNLIKELY(!valid_block_index)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted gradient-pack plan: invalid block index");
        }
        return;
    }

    // A deficient launch can otherwise process every segment reachable by the
    // launched CTAs while silently leaving trailing output unwritten. Every
    // thread in CTA 0 evaluates the same predicate so this rejecting return is
    // CTA-uniform even when device assertions compile away.
    if (block_index == 0) {
        const bool valid_final_block_count =
            num_segments > 0 && segments[(num_segments - 1) * kValuesPerSegment + 4] == num_blocks;
        if (C10_UNLIKELY(!valid_final_block_count)) {
            if (threadIdx.x == 0) {
                CUDA_KERNEL_ASSERT_MSG(
                    false, "malformed trusted gradient-pack plan: block count does not match final copy row");
            }
            return;
        }
    }

    // Find the first segment whose cumulative block end is greater than this
    // block. Every thread follows the same uniform search for its CTA's block;
    // large fields can use many CTAs without expanding the segment table.
    int64_t low  = 0;
    int64_t high = num_segments;
    while (low < high) {
        const int64_t middle    = low + (high - low) / 2;
        const int64_t block_end = segments[middle * kValuesPerSegment + 4];
        if (block_end <= block_index) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    const bool found_segment = low < num_segments;
    if (C10_UNLIKELY(!found_segment)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted gradient-pack plan: missing copy row");
        }
        return;
    }

    const int64_t segment_index     = low;
    const int64_t* row              = segments + segment_index * kValuesPerSegment;
    const int64_t segment_length    = row[1];
    const int64_t input_index       = row[2];
    const bool valid_segment_length = segment_length > 0 && segment_length <= output_numel;
    if (C10_UNLIKELY(!valid_segment_length)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted gradient-pack plan: invalid copy-row length");
        }
        return;
    }
    const int64_t blocks_in_segment = 1 + (segment_length - 1) / kGradientPackValuesPerBlock;
    const int64_t block_end         = row[4];
    const bool valid_block_range =
        block_end >= blocks_in_segment && block_end <= num_blocks && block_index >= block_end - blocks_in_segment &&
        block_index < block_end;
    if (C10_UNLIKELY(!valid_block_range)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted gradient-pack plan: invalid copy-row block range");
        }
        return;
    }
    const int64_t first_block     = block_end - blocks_in_segment;
    const int64_t chunk_offset    = (block_index - first_block) * kGradientPackValuesPerBlock;
    const bool valid_chunk_offset = chunk_offset >= 0 && chunk_offset < segment_length;
    if (C10_UNLIKELY(!valid_chunk_offset)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted gradient-pack plan: invalid copy-row chunk");
        }
        return;
    }
    const int64_t remaining = segment_length - chunk_offset;
    const int64_t length =
        remaining < kGradientPackValuesPerBlock ? remaining : kGradientPackValuesPerBlock;
    const int64_t destination_start = row[0];
    const bool valid_destination =
        destination_start >= 0 && destination_start <= output_numel && chunk_offset <= output_numel - destination_start &&
        length <= output_numel - destination_start - chunk_offset;
    if (C10_UNLIKELY(!valid_destination)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted gradient-pack plan: invalid destination range");
        }
        return;
    }
    const int64_t destination_offset = destination_start + chunk_offset;

    const scalar_t* source       = nullptr;
    const bool valid_input_index = input_index >= -1 && input_index < num_inputs;
    if (C10_UNLIKELY(!valid_input_index)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted gradient-pack plan: invalid input index");
        }
        return;
    }
    if (input_index >= 0) {
        const int64_t source_numel = inputs.numels[input_index];
        if (source_numel != kMissingGradientNumel) {
            const int64_t source_start = row[3];
            const bool valid_source =
                source_start >= 0 && source_start <= source_numel && chunk_offset <= source_numel - source_start &&
                length <= source_numel - source_start - chunk_offset;
            if (C10_UNLIKELY(!valid_source)) {
                if (threadIdx.x == 0) {
                    CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted gradient-pack plan: invalid source range");
                }
                return;
            }
            source = inputs.pointers[input_index];
            if (C10_UNLIKELY(source == nullptr)) {
                if (threadIdx.x == 0) {
                    CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted gradient-pack plan: null source pointer");
                }
                return;
            }
            source += source_start + chunk_offset;
        }
    }

    for (int64_t index = threadIdx.x; index < length; index += blockDim.x) {
        output[destination_offset + index] = source == nullptr ? scalar_t(0) : source[index];
    }
}

} // namespace

torch::Tensor pack_gradients_cuda(
    torch::Tensor output,
    const std::vector<std::optional<torch::Tensor>>& gradients,
    const torch::Tensor& copy_rows,
    int64_t num_blocks) {
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(output.dim() == 1, "output must be one-dimensional, got ", output.dim(), " dimensions");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(!output.requires_grad(), "output must not require gradients");
    TORCH_CHECK(!output.is_inference(), "output must not be an inference tensor");
    TORCH_CHECK(
        output.is_floating_point(),
        "output must have a floating-point dtype, got ",
        output.scalar_type());
    TORCH_CHECK(is_supported_payload_dtype(output.scalar_type()), "unsupported gradient output dtype ", output.scalar_type());

    TORCH_CHECK(copy_rows.is_cuda(), "copy_rows must be a CUDA tensor");
    TORCH_CHECK(copy_rows.scalar_type() == torch::kLong, "copy_rows must have dtype torch.int64");
    TORCH_CHECK(
        copy_rows.dim() == 2 && copy_rows.size(1) == kValuesPerSegment,
        "copy_rows must have shape [N, 5], got ",
        copy_rows.sizes());
    TORCH_CHECK(copy_rows.is_contiguous(), "copy_rows must be contiguous");
    TORCH_CHECK(
        copy_rows.get_device() == output.get_device(),
        "copy_rows and output must be on the same CUDA device");
    at::assert_no_overlap(output, copy_rows);
    TORCH_CHECK(
        static_cast<int64_t>(gradients.size()) <= kMaxGradientPackInputs,
        "pack_gradients",
        " supports at most ",
        kMaxGradientPackInputs,
        " input tensors, got ",
        gradients.size());

    const int64_t num_segments = copy_rows.size(0);
    TORCH_CHECK(
        num_blocks >= 0 && num_blocks <= std::numeric_limits<int32_t>::max(),
        "invalid gradient-pack block count: ",
        num_blocks);
    TORCH_CHECK(
        (output.numel() == 0) == (num_segments == 0 && num_blocks == 0),
        "empty output, copy rows, and block count must be supplied together");
    TORCH_CHECK(output.numel() == 0 || (num_segments > 0 && num_blocks > 0), "nonempty output requires work");

    for (size_t index = 0; index < gradients.size(); ++index) {
        if (!gradients[index].has_value()) {
            continue;
        }
        const torch::Tensor& input = gradients[index].value();
        TORCH_CHECK(input.defined(), "gradient ", index, " is undefined; pass None instead");
        TORCH_CHECK(input.is_cuda(), "gradient ", index, " must be a CUDA tensor");
        TORCH_CHECK(input.is_contiguous(), "gradient ", index, " must be contiguous");
        TORCH_CHECK(
            input.get_device() == output.get_device(),
            "gradient ",
            index,
            " is on CUDA device ",
            input.get_device(),
            ", expected ",
            output.get_device());
        TORCH_CHECK(
            input.scalar_type() == output.scalar_type(),
            "gradient ",
            index,
            " has dtype ",
            input.scalar_type(),
            ", expected ",
            output.scalar_type());
        at::assert_no_overlap(output, input);
    }

    if (output.numel() == 0) {
        output.unsafeGetTensorImpl()->bump_version();
        return output;
    }

    const c10::cuda::OptionalCUDAGuard device_guard(torch::device_of(output));
    const auto stream = c10::cuda::getCurrentCUDAStream(output.get_device());

    // The kernel is asynchronous. Record every allocation it touches on the
    // launch stream so callers may release inputs immediately after this
    // function returns, including when a producer used a different stream.
    c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(copy_rows.storage().data_ptr(), stream);
    for (const auto& input : gradients) {
        if (input.has_value()) {
            c10::cuda::CUDACachingAllocator::recordStream(input->storage().data_ptr(), stream);
        }
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(),
        "pack_gradients_cuda",
        [&] {
            GradientPackInputs<scalar_t> descriptors{};
            for (size_t index = 0; index < gradients.size(); ++index) {
                if (gradients[index].has_value()) {
                    descriptors.pointers[index] = gradients[index]->const_data_ptr<scalar_t>();
                    descriptors.numels[index]   = gradients[index]->numel();
                } else {
                    descriptors.numels[index] = kMissingGradientNumel;
                }
            }

            pack_gradients_kernel<scalar_t><<<num_blocks, kThreadsPerBlock, 0, stream>>>(
                output.data_ptr<scalar_t>(),
                output.numel(),
                copy_rows.const_data_ptr<int64_t>(),
                num_segments,
                num_blocks,
                gradients.size(),
                descriptors);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    output.unsafeGetTensorImpl()->bump_version();
    return output;
}
