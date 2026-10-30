// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gaussian_collective_payload_ops.h"

#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>

#include <cstdint>
#include <limits>

namespace {

// One fixed signature supports one to eight outputs without uploading pointer tables.
// Eight output pointers precede eight output sizes, then six common arguments
// (176 bytes in total).
static_assert(
    kMaxGroupedPayloadLayers == 8,
    "grouped payload layer capacity must match the fixed unpack launch signature");

// Both trusted unpack formats contain five int64 values per row. A grouped row
// is [output, destination, length, source, block end]; a field-major row is
// [destination, length, source, field width, block end].
constexpr int64_t kValuesPerSegment = 5;

// CTA width for field-major unpack. The kernel precomputes row/column stepping
// from this value while its loop strides by blockDim.x, so the launch width
// must remain this exact constant. It is independent of values-per-block.
constexpr int kThreadsPerBlock = 256;

// Separate CTA width for grouped unpack, whose segments are contiguous copies
// and do not require the field-major row/column remapping above.
// An 8-256-thread sweep across 4-32 GPUs found no consistent winner,
// so keep 64 as a stable default.
constexpr int kGroupedPayloadUnpackThreads = 64;

__device__ __forceinline__ bool nonnegative_offset_range_within(
    int64_t base,
    int64_t offset,
    int64_t length,
    int64_t extent) {
    if (base < 0 || offset < 0 || length <= 0 || extent < 0 || base > extent) {
        return false;
    }
    const int64_t remaining_after_base = extent - base;
    return offset <= remaining_after_base && length <= remaining_after_base - offset;
}

/**
 * Copy cached grouped payload segments into up to eight output buffers.
 *
 * Args:
 *   output0, output1, output2, output3, output4, output5, output6, output7:
 *     Contiguous CUDA arrays of shape [output_numelN] and dtype scalar_t.
 *     Empty outputs and slots N >= num_outputs use nullptr with size zero.
 *   output_numel0, output_numel1, output_numel2, output_numel3,
 *   output_numel4, output_numel5, output_numel6, output_numel7:
 *     Nonnegative element counts for the corresponding output buffers.
 *   input: Contiguous CUDA source, viewed as [input_numel] scalar_t values.
 *   input_numel: Number of source elements available for bounds checks.
 *   segments: Contiguous CUDA int64 array of shape [num_segments, 5]. Each
 *     row is [output index, destination offset, length, source offset,
 *     exclusive cumulative block end]; offsets and lengths count elements.
 *   num_segments: Positive number of cached copy rows.
 *   num_blocks: Positive grid size, equal to the final row's block end.
 *   num_outputs: Number of supplied output slots, in [1, 8].
 *
 * Returns: Nothing; writes the ranges described by segments into the outputs.
 * Raises: CUDA device assertions reject malformed selected metadata or bounds.
 *
 * Invariants: All arrays share a CUDA device; scalar_t is float, double, half,
 * or bfloat16. Input, outputs, and metadata do not overlap. Output copy ranges
 * must not overlap each other. Rows have positive lengths, increasing block
 * ends, and in-bounds source/destination ranges targeting nonempty outputs.
 * Each block copies at most kPayloadUnpackValuesPerBlock elements from one
 * row; a row occupies ceil(length / kPayloadUnpackValuesPerBlock) blocks.
 * The host wrapper skips the launch when all outputs are empty.
 */
template <typename scalar_t>
__global__ void unpack_grouped_payload_kernel(
    scalar_t* output0,
    scalar_t* output1,
    scalar_t* output2,
    scalar_t* output3,
    scalar_t* output4,
    scalar_t* output5,
    scalar_t* output6,
    scalar_t* output7,
    int64_t output_numel0,
    int64_t output_numel1,
    int64_t output_numel2,
    int64_t output_numel3,
    int64_t output_numel4,
    int64_t output_numel5,
    int64_t output_numel6,
    int64_t output_numel7,
    const scalar_t* __restrict__ input,
    int64_t input_numel,
    const int64_t* __restrict__ segments,
    int64_t num_segments,
    int64_t num_blocks,
    int64_t num_outputs) {
    const int64_t block_index = blockIdx.x;
    if (C10_UNLIKELY(block_index < 0 || block_index >= num_blocks)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted grouped payload metadata: invalid block index");
        }
        return;
    }

    // One thread resolves this CTA's cached copy row. The metadata remains on
    // device and grows with fields * ranks while the kernel argument block is
    // constant-size for every process-group size.
    __shared__ int64_t selected_output_index;
    __shared__ int64_t copy_geometry[3];
    if (threadIdx.x == 0) {
        selected_output_index = -1;
        copy_geometry[2]      = 0;

        bool metadata_valid = true;
        if (C10_UNLIKELY(
                block_index == 0 &&
                (num_segments <= 0 || segments[(num_segments - 1) * kValuesPerSegment + 4] != num_blocks))) {
            CUDA_KERNEL_ASSERT_MSG(
                false, "malformed trusted grouped payload metadata: block count does not match final segment");
            metadata_valid = false;
        }

        int64_t low  = 0;
        int64_t high = num_segments;
        while (metadata_valid && low < high) {
            const int64_t middle    = low + (high - low) / 2;
            const int64_t block_end = segments[middle * kValuesPerSegment + 4];
            if (block_end <= block_index) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }

        if (C10_UNLIKELY(metadata_valid && low >= num_segments)) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted grouped payload metadata: missing copy segment");
            metadata_valid = false;
        }
        if (metadata_valid) {
            const int64_t* row           = segments + low * kValuesPerSegment;
            const int64_t output_index   = row[0];
            const int64_t destination    = row[1];
            const int64_t segment_length = row[2];
            const int64_t source         = row[3];
            if (C10_UNLIKELY(segment_length <= 0 || segment_length > input_numel)) {
                CUDA_KERNEL_ASSERT_MSG(
                    false, "malformed trusted grouped payload metadata: invalid segment length");
            } else {
                const int64_t blocks_in_segment =
                    1 + (segment_length - 1) / kPayloadUnpackValuesPerBlock;
                const int64_t block_end = row[4];
                const bool valid_block_range =
                    block_end >= blocks_in_segment && block_end <= num_blocks &&
                    block_index >= block_end - blocks_in_segment && block_index < block_end;
                if (C10_UNLIKELY(!valid_block_range)) {
                    CUDA_KERNEL_ASSERT_MSG(
                        false, "malformed trusted grouped payload metadata: invalid segment block range");
                } else {
                    const int64_t first_block = block_end - blocks_in_segment;
                    const int64_t chunk_offset =
                        (block_index - first_block) * kPayloadUnpackValuesPerBlock;
                    const bool valid_chunk_offset = chunk_offset >= 0 && chunk_offset < segment_length;
                    if (C10_UNLIKELY(!valid_chunk_offset)) {
                        CUDA_KERNEL_ASSERT_MSG(
                            false, "malformed trusted grouped payload metadata: invalid segment chunk");
                    } else {
                        const int64_t remaining = segment_length - chunk_offset;
                        const int64_t length    = remaining < kPayloadUnpackValuesPerBlock
                                                      ? remaining
                                                      : kPayloadUnpackValuesPerBlock;

                        const bool valid_output_index = output_index >= 0 && output_index < num_outputs;
                        scalar_t* output              = nullptr;
                        int64_t output_numel          = 0;
                        if (valid_output_index) {
                            switch (output_index) {
                            case 0:
                                output       = output0;
                                output_numel = output_numel0;
                                break;
                            case 1:
                                output       = output1;
                                output_numel = output_numel1;
                                break;
                            case 2:
                                output       = output2;
                                output_numel = output_numel2;
                                break;
                            case 3:
                                output       = output3;
                                output_numel = output_numel3;
                                break;
                            case 4:
                                output       = output4;
                                output_numel = output_numel4;
                                break;
                            case 5:
                                output       = output5;
                                output_numel = output_numel5;
                                break;
                            case 6:
                                output       = output6;
                                output_numel = output_numel6;
                                break;
                            case 7:
                                output       = output7;
                                output_numel = output_numel7;
                                break;
                            }
                        }
                        const bool valid_source =
                            nonnegative_offset_range_within(source, chunk_offset, length, input_numel);
                        const bool valid_destination =
                            valid_output_index && output != nullptr &&
                            nonnegative_offset_range_within(destination, chunk_offset, length, output_numel);

                        if (C10_UNLIKELY(!valid_output_index)) {
                            CUDA_KERNEL_ASSERT_MSG(
                                false, "malformed trusted grouped payload metadata: invalid output index");
                        } else if (C10_UNLIKELY(!valid_source)) {
                            CUDA_KERNEL_ASSERT_MSG(
                                false, "malformed trusted grouped payload metadata: invalid source range");
                        } else if (C10_UNLIKELY(!valid_destination)) {
                            CUDA_KERNEL_ASSERT_MSG(
                                false, "malformed trusted grouped payload metadata: invalid destination range");
                        } else {
                            // The range predicates above prove these additions
                            // cannot overflow and the copy lies in-bounds.
                            const int64_t destination_offset = destination + chunk_offset;
                            const int64_t source_offset      = source + chunk_offset;
                            selected_output_index            = output_index;
                            copy_geometry[0]                 = destination_offset;
                            copy_geometry[1]                 = source_offset;
                            copy_geometry[2]                 = length;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    const int64_t length = copy_geometry[2];
    if (length <= 0) {
        return;
    }
    const int64_t destination = copy_geometry[0];
    const int64_t source      = copy_geometry[1];

    // Keep the selected kernel-argument pointer thread-local. Passing a global
    // pointer through shared memory loses its address-space provenance and can
    // lower the write below to a generic store instead of a global store.
    // The scalar switches retain explicit kernel pointer operands; a by-value
    // pointer array's generated stores and performance have not been compared.
    scalar_t* selected_output = nullptr;
    switch (selected_output_index) {
    case 0:
        selected_output = output0;
        break;
    case 1:
        selected_output = output1;
        break;
    case 2:
        selected_output = output2;
        break;
    case 3:
        selected_output = output3;
        break;
    case 4:
        selected_output = output4;
        break;
    case 5:
        selected_output = output5;
        break;
    case 6:
        selected_output = output6;
        break;
    case 7:
        selected_output = output7;
        break;
    default:
        CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted grouped payload metadata: invalid selected output");
        return;
    }
    for (int64_t index = threadIdx.x; index < length; index += blockDim.x) {
        selected_output[destination + index] = input[source + index];
    }
}

template <typename scalar_t>
__global__ void unpack_payload_kernel(
    scalar_t* __restrict__ output,
    int64_t output_numel,
    const scalar_t* __restrict__ input,
    int64_t input_numel,
    const int64_t* __restrict__ segments,
    int64_t num_segments,
    int64_t num_blocks,
    int64_t source_row_width) {
    const int64_t block_index    = blockIdx.x;
    const bool valid_block_index = block_index < num_blocks;
    if (C10_UNLIKELY(!valid_block_index)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted payload-unpack metadata: invalid block index");
        }
        // CTA-uniform: every thread shares blockIdx.x, so returning here cannot
        // desynchronize the __syncthreads() below.
        return;
    }

    // Find the field/rank segment owning this CTA. Large fields use multiple
    // CTAs without expanding the cached topology metadata. One thread does the
    // identical search and broadcasts the resolved copy geometry to the CTA.
    //
    // Malformed metadata fires CUDA_KERNEL_ASSERT_MSG, matching
    // pack_gradients_kernel in this package. That assert is noreturn on CUDA
    // but compiles away under ROCm, so every rejecting path also leaves the
    // zero-length sentinel in copy_geometry[1] and falls through to the
    // barrier, keeping the CTA convergent instead of returning from one thread.
    __shared__ int64_t copy_geometry[8];
    if (threadIdx.x == 0) {
        copy_geometry[1] = 0;

        bool metadata_valid = true;
        if (C10_UNLIKELY(
                block_index == 0 &&
                (num_segments <= 0 || segments[(num_segments - 1) * kValuesPerSegment + 4] != num_blocks))) {
            CUDA_KERNEL_ASSERT_MSG(
                false, "malformed trusted payload-unpack metadata: block count does not match final segment");
            metadata_valid = false;
        }

        int64_t low  = 0;
        int64_t high = num_segments;
        while (metadata_valid && low < high) {
            const int64_t middle    = low + (high - low) / 2;
            const int64_t block_end = segments[middle * kValuesPerSegment + 4];
            if (block_end <= block_index) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }

        const bool found_segment = metadata_valid && low < num_segments;
        if (C10_UNLIKELY(!found_segment)) {
            if (metadata_valid) {
                CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted payload-unpack metadata: missing block segment");
            }
        }
        if (found_segment) {
            const int64_t* row           = segments + low * kValuesPerSegment;
            const int64_t destination    = row[0];
            const int64_t segment_length = row[1];
            const int64_t source         = row[2];
            const int64_t field_width    = row[3];
            if (C10_UNLIKELY(segment_length <= 0 || segment_length > input_numel)) {
                CUDA_KERNEL_ASSERT_MSG(
                    false, "malformed trusted payload-unpack metadata: invalid segment length");
            } else {
                const int64_t blocks_in_segment =
                    1 + (segment_length - 1) / kPayloadUnpackValuesPerBlock;
                const int64_t block_end = row[4];
                const bool valid_block_range =
                    block_end >= blocks_in_segment && block_end <= num_blocks &&
                    block_index >= block_end - blocks_in_segment && block_index < block_end;
                if (C10_UNLIKELY(!valid_block_range)) {
                    CUDA_KERNEL_ASSERT_MSG(
                        false, "malformed trusted payload-unpack metadata: invalid segment block range");
                } else {
                    const int64_t first_block = block_end - blocks_in_segment;
                    const int64_t chunk_offset =
                        (block_index - first_block) * kPayloadUnpackValuesPerBlock;
                    const bool valid_chunk_offset = chunk_offset >= 0 && chunk_offset < segment_length;
                    // field_width <= source_row_width keeps the source index
                    // monotonic in the copy position, so bounding the final
                    // read below bounds every read and keeps the loop unguarded.
                    const bool valid_field_width =
                        field_width > 0 && field_width <= source_row_width &&
                        field_width <= std::numeric_limits<int32_t>::max() - kThreadsPerBlock;
                    if (C10_UNLIKELY(!valid_chunk_offset)) {
                        CUDA_KERNEL_ASSERT_MSG(
                            false, "malformed trusted payload-unpack metadata: invalid segment chunk");
                    } else if (C10_UNLIKELY(!valid_field_width)) {
                        CUDA_KERNEL_ASSERT_MSG(
                            false, "malformed trusted payload-unpack metadata: invalid field width");
                    } else {
                        const int64_t remaining = segment_length - chunk_offset;
                        const int64_t length    = remaining < kPayloadUnpackValuesPerBlock
                                                      ? remaining
                                                      : kPayloadUnpackValuesPerBlock;
                        const bool valid_destination =
                            nonnegative_offset_range_within(destination, chunk_offset, length, output_numel);
                        const int64_t destination_offset =
                            valid_destination ? destination + chunk_offset : 0;

                        const int64_t last_position = chunk_offset + length - 1;
                        const int64_t row_offset    = last_position / field_width;
                        const int64_t col_offset    = last_position % field_width;
                        bool valid_source           = source >= 0 && source < input_numel;
                        if (valid_source) {
                            const int64_t remaining_input = input_numel - source;
                            valid_source                  = row_offset <= (remaining_input - 1) / source_row_width;
                            if (valid_source) {
                                const int64_t row_start_offset = row_offset * source_row_width;
                                valid_source                   = col_offset < remaining_input - row_start_offset;
                            }
                        }

                        if (C10_UNLIKELY(!valid_destination)) {
                            CUDA_KERNEL_ASSERT_MSG(
                                false, "malformed trusted payload-unpack metadata: invalid destination range");
                        } else if (C10_UNLIKELY(!valid_source)) {
                            CUDA_KERNEL_ASSERT_MSG(
                                false, "malformed trusted payload-unpack metadata: invalid source range");
                        } else {
                            copy_geometry[0] = destination_offset;
                            copy_geometry[1] = length;
                            copy_geometry[2] = source;
                            copy_geometry[3] = field_width;
                            copy_geometry[4] = chunk_offset / field_width;
                            copy_geometry[5] = chunk_offset % field_width;
                            copy_geometry[6] = kThreadsPerBlock / field_width;
                            copy_geometry[7] = kThreadsPerBlock % field_width;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    const int64_t length = copy_geometry[1];
    if (length <= 0) {
        return;
    }

    const int64_t destination_offset = copy_geometry[0];
    const int64_t source             = copy_geometry[2];
    const int32_t field_width        = static_cast<int32_t>(copy_geometry[3]);
    const int32_t initial_col        = static_cast<int32_t>(copy_geometry[5]) + static_cast<int32_t>(threadIdx.x);
    int64_t source_row               = copy_geometry[4] + initial_col / field_width;
    int32_t source_col               = initial_col % field_width;
    const int32_t row_step           = static_cast<int32_t>(copy_geometry[6]);
    const int32_t col_step           = static_cast<int32_t>(copy_geometry[7]);
    for (int64_t index = threadIdx.x; index < length; index += blockDim.x) {
        output[destination_offset + index] = input[source + source_row * source_row_width + source_col];
        source_row += row_step;
        source_col += col_step;
        if (source_col >= field_width) {
            source_col -= field_width;
            ++source_row;
        }
    }
}

} // namespace

std::vector<torch::Tensor> unpack_grouped_payload_cuda(
    std::vector<torch::Tensor> outputs,
    const torch::Tensor& rank_major_grouped,
    const torch::Tensor& segments,
    int64_t num_blocks) {
    TORCH_CHECK(!outputs.empty(), "grouped payload unpack requires at least one output");
    TORCH_CHECK(
        outputs.size() <= kMaxGroupedPayloadLayers,
        "grouped payload unpack supports at most ",
        kMaxGroupedPayloadLayers,
        " outputs, got ",
        outputs.size());

    TORCH_CHECK(rank_major_grouped.defined(), "rank-major grouped input must be defined");
    TORCH_CHECK(rank_major_grouped.is_cuda(), "rank-major grouped input must be a CUDA tensor");
    TORCH_CHECK(
        rank_major_grouped.dim() == 1 || rank_major_grouped.dim() == 2,
        "rank-major grouped input must be one- or two-dimensional, got ",
        rank_major_grouped.dim(),
        " dimensions");
    TORCH_CHECK(rank_major_grouped.is_contiguous(), "rank-major grouped input must be contiguous");
    TORCH_CHECK(!rank_major_grouped.requires_grad(), "rank-major grouped input must not require gradients");
    TORCH_CHECK(
        rank_major_grouped.is_floating_point(),
        "rank-major grouped input must have a floating-point dtype, got ",
        rank_major_grouped.scalar_type());
    TORCH_CHECK(
        is_supported_payload_dtype(rank_major_grouped.scalar_type()),
        "unsupported rank-major grouped payload dtype ",
        rank_major_grouped.scalar_type());

    int64_t total_output_numel = 0;
    for (size_t layer = 0; layer < outputs.size(); ++layer) {
        const torch::Tensor& output = outputs[layer];
        TORCH_CHECK(output.defined(), "grouped payload output ", layer, " must be defined");
        TORCH_CHECK(output.is_cuda(), "grouped payload output ", layer, " must be a CUDA tensor");
        TORCH_CHECK(
            output.dim() == 1,
            "grouped payload output ",
            layer,
            " must be one-dimensional, got ",
            output.dim(),
            " dimensions");
        TORCH_CHECK(output.is_contiguous(), "grouped payload output ", layer, " must be contiguous");
        TORCH_CHECK(!output.requires_grad(), "grouped payload output ", layer, " must not require gradients");
        TORCH_CHECK(
            !output.is_inference() || c10::InferenceMode::is_enabled(),
            "an inference grouped payload output is only valid while inference mode is enabled");
        TORCH_CHECK(
            output.get_device() == rank_major_grouped.get_device(),
            "grouped payload output ",
            layer,
            " and input must be on the same CUDA device");
        TORCH_CHECK(
            output.scalar_type() == rank_major_grouped.scalar_type(),
            "grouped payload output ",
            layer,
            " dtype ",
            output.scalar_type(),
            " does not match input dtype ",
            rank_major_grouped.scalar_type());
        TORCH_CHECK(
            output.storage().unsafeGetStorageImpl() != rank_major_grouped.storage().unsafeGetStorageImpl(),
            "grouped payload outputs and input must use distinct storage");
        at::assert_no_overlap(output, rank_major_grouped);
        for (size_t previous = 0; previous < layer; ++previous) {
            TORCH_CHECK(
                output.storage().unsafeGetStorageImpl() != outputs[previous].storage().unsafeGetStorageImpl(),
                "grouped payload outputs must use distinct storage");
            at::assert_no_overlap(output, outputs[previous]);
        }
        TORCH_CHECK(
            output.numel() <= std::numeric_limits<int64_t>::max() - total_output_numel,
            "grouped payload output size overflows int64");
        total_output_numel += output.numel();
    }

    TORCH_CHECK(segments.defined(), "grouped payload segments must be defined");
    TORCH_CHECK(segments.is_cuda(), "grouped payload segments must be a CUDA tensor");
    TORCH_CHECK(segments.scalar_type() == torch::kLong, "grouped payload segments must have dtype torch.int64");
    TORCH_CHECK(
        segments.dim() == 2 && segments.size(1) == kValuesPerSegment,
        "grouped payload segments must have shape [N, 5], got ",
        segments.sizes());
    TORCH_CHECK(segments.is_contiguous(), "grouped payload segments must be contiguous");
    TORCH_CHECK(
        segments.get_device() == rank_major_grouped.get_device(),
        "grouped payload segments and input must be on the same CUDA device");
    at::assert_no_overlap(rank_major_grouped, segments);
    for (const torch::Tensor& output : outputs) {
        at::assert_no_overlap(output, segments);
    }

    const int64_t num_segments = segments.size(0);
    TORCH_CHECK(
        num_blocks >= 0 && num_blocks <= std::numeric_limits<int32_t>::max(),
        "invalid grouped payload unpack block count: ",
        num_blocks);
    TORCH_CHECK(
        (total_output_numel == 0) == (num_segments == 0 && num_blocks == 0),
        "empty grouped payload outputs, segments, and block count must be supplied together");
    TORCH_CHECK(
        total_output_numel == 0 || (num_segments > 0 && num_blocks > 0),
        "nonempty grouped payload outputs require work");

    if (total_output_numel == 0) {
        for (torch::Tensor& output : outputs) {
            if (!output.is_inference()) {
                output.unsafeGetTensorImpl()->bump_version();
            }
        }
        return outputs;
    }

    const c10::cuda::OptionalCUDAGuard device_guard(torch::device_of(rank_major_grouped));
    const auto stream = c10::cuda::getCurrentCUDAStream(rank_major_grouped.get_device());
    c10::cuda::CUDACachingAllocator::recordStream(rank_major_grouped.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(segments.storage().data_ptr(), stream);
    for (const torch::Tensor& output : outputs) {
        if (output.numel() > 0) {
            c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), stream);
        }
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        rank_major_grouped.scalar_type(),
        "unpack_grouped_payload_cuda",
        [&] {
            scalar_t* output0           = outputs.size() > 0 && outputs[0].numel() > 0 ? outputs[0].data_ptr<scalar_t>() : nullptr;
            scalar_t* output1           = outputs.size() > 1 && outputs[1].numel() > 0 ? outputs[1].data_ptr<scalar_t>() : nullptr;
            scalar_t* output2           = outputs.size() > 2 && outputs[2].numel() > 0 ? outputs[2].data_ptr<scalar_t>() : nullptr;
            scalar_t* output3           = outputs.size() > 3 && outputs[3].numel() > 0 ? outputs[3].data_ptr<scalar_t>() : nullptr;
            scalar_t* output4           = outputs.size() > 4 && outputs[4].numel() > 0 ? outputs[4].data_ptr<scalar_t>() : nullptr;
            scalar_t* output5           = outputs.size() > 5 && outputs[5].numel() > 0 ? outputs[5].data_ptr<scalar_t>() : nullptr;
            scalar_t* output6           = outputs.size() > 6 && outputs[6].numel() > 0 ? outputs[6].data_ptr<scalar_t>() : nullptr;
            scalar_t* output7           = outputs.size() > 7 && outputs[7].numel() > 0 ? outputs[7].data_ptr<scalar_t>() : nullptr;
            const int64_t output_numel0 = outputs.size() > 0 ? outputs[0].numel() : 0;
            const int64_t output_numel1 = outputs.size() > 1 ? outputs[1].numel() : 0;
            const int64_t output_numel2 = outputs.size() > 2 ? outputs[2].numel() : 0;
            const int64_t output_numel3 = outputs.size() > 3 ? outputs[3].numel() : 0;
            const int64_t output_numel4 = outputs.size() > 4 ? outputs[4].numel() : 0;
            const int64_t output_numel5 = outputs.size() > 5 ? outputs[5].numel() : 0;
            const int64_t output_numel6 = outputs.size() > 6 ? outputs[6].numel() : 0;
            const int64_t output_numel7 = outputs.size() > 7 ? outputs[7].numel() : 0;
            unpack_grouped_payload_kernel<scalar_t><<<num_blocks, kGroupedPayloadUnpackThreads, 0, stream>>>(
                output0,
                output1,
                output2,
                output3,
                output4,
                output5,
                output6,
                output7,
                output_numel0,
                output_numel1,
                output_numel2,
                output_numel3,
                output_numel4,
                output_numel5,
                output_numel6,
                output_numel7,
                rank_major_grouped.const_data_ptr<scalar_t>(),
                rank_major_grouped.numel(),
                segments.const_data_ptr<int64_t>(),
                num_segments,
                num_blocks,
                outputs.size());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (torch::Tensor& output : outputs) {
        if (!output.is_inference()) {
            output.unsafeGetTensorImpl()->bump_version();
        }
    }
    return outputs;
}

torch::Tensor unpack_payload_cuda(
    torch::Tensor output,
    const torch::Tensor& rank_major_aos,
    const torch::Tensor& segments,
    int64_t num_blocks,
    int64_t source_row_width) {
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(output.dim() == 1, "output must be one-dimensional, got ", output.dim(), " dimensions");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(!output.requires_grad(), "output must not require gradients");
    TORCH_CHECK(
        !output.is_inference() || c10::InferenceMode::is_enabled(),
        "an inference output is only valid while inference mode is enabled");
    TORCH_CHECK(output.is_floating_point(), "output must have a floating-point dtype, got ", output.scalar_type());
    TORCH_CHECK(is_supported_payload_dtype(output.scalar_type()), "unsupported payload output dtype ", output.scalar_type());

    TORCH_CHECK(rank_major_aos.is_cuda(), "rank-major AoS input must be a CUDA tensor");
    TORCH_CHECK(
        rank_major_aos.dim() == 2,
        "rank-major AoS input must be two-dimensional, got ",
        rank_major_aos.dim(),
        " dimensions");
    TORCH_CHECK(rank_major_aos.is_contiguous(), "rank-major AoS input must be contiguous");
    TORCH_CHECK(!rank_major_aos.requires_grad(), "rank-major AoS input must not require gradients");
    TORCH_CHECK(
        rank_major_aos.is_floating_point(),
        "rank-major AoS input must have a floating-point dtype, got ",
        rank_major_aos.scalar_type());
    TORCH_CHECK(
        rank_major_aos.get_device() == output.get_device(),
        "input and output must be on the same CUDA device");
    TORCH_CHECK(
        rank_major_aos.scalar_type() == output.scalar_type(),
        "input dtype ",
        rank_major_aos.scalar_type(),
        " does not match output dtype ",
        output.scalar_type());
    at::assert_no_overlap(output, rank_major_aos);
    TORCH_CHECK(
        source_row_width >= 0 && rank_major_aos.size(1) == source_row_width,
        "source row width ",
        source_row_width,
        " does not match input shape ",
        rank_major_aos.sizes());

    TORCH_CHECK(segments.is_cuda(), "segments must be a CUDA tensor");
    TORCH_CHECK(segments.scalar_type() == torch::kLong, "segments must have dtype torch.int64");
    TORCH_CHECK(
        segments.dim() == 2 && segments.size(1) == kValuesPerSegment,
        "segments must have shape [N, 5], got ",
        segments.sizes());
    TORCH_CHECK(segments.is_contiguous(), "segments must be contiguous");
    TORCH_CHECK(
        segments.get_device() == rank_major_aos.get_device(),
        "segments and input must be on the same CUDA device");
    at::assert_no_overlap(output, segments);

    const int64_t num_segments = segments.size(0);
    TORCH_CHECK(
        num_blocks >= 0 && num_blocks <= std::numeric_limits<int32_t>::max(),
        "invalid Gaussian payload unpack block count: ",
        num_blocks);
    TORCH_CHECK(
        (output.numel() == 0) == (num_segments == 0 && num_blocks == 0),
        "empty output, segments, and block count must be supplied together");
    TORCH_CHECK(output.numel() == 0 || (num_segments > 0 && num_blocks > 0), "nonempty output requires work");

    if (output.numel() == 0) {
        if (!output.is_inference()) {
            output.unsafeGetTensorImpl()->bump_version();
        }
        return output;
    }

    const c10::cuda::OptionalCUDAGuard device_guard(torch::device_of(output));
    const auto stream = c10::cuda::getCurrentCUDAStream(output.get_device());

    c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(rank_major_aos.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(segments.storage().data_ptr(), stream);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(),
        "unpack_payload_cuda",
        [&] {
            unpack_payload_kernel<scalar_t><<<num_blocks, kThreadsPerBlock, 0, stream>>>(
                output.data_ptr<scalar_t>(),
                output.numel(),
                rank_major_aos.const_data_ptr<scalar_t>(),
                rank_major_aos.numel(),
                segments.const_data_ptr<int64_t>(),
                num_segments,
                num_blocks,
                source_row_width);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!output.is_inference()) {
        output.unsafeGetTensorImpl()->bump_version();
    }
    return output;
}
