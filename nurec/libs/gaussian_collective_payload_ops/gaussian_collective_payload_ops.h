// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <vector>

// Maximum adjacent layers accepted by both grouped payload kernels. This is
// a native launch-ABI capacity: grouped unpack passes one pointer and one size
// per layer by value, while grouped pack reserves one possible padding segment
// per layer. Keep the fixed kernel signatures and descriptor sizing assertions
// in both CUDA translation units coupled to this shared value. Python reads
// this value from the extension to decide whether adjacent layers may use the
// grouped path. Grouped payload segment capacities remain private to their
// native entry points, which validate them before launch. The gradient-pack
// input capacity below is exported because Python must reject a rank-agreed
// schema before payload communication and plan construction.
inline constexpr int64_t kMaxGroupedPayloadLayers = 8;

constexpr bool is_supported_payload_dtype(at::ScalarType scalar_type) {
    return scalar_type == at::ScalarType::Half ||
           scalar_type == at::ScalarType::BFloat16 ||
           scalar_type == at::ScalarType::Float ||
           scalar_type == at::ScalarType::Double;
}

// Gradient packing.

// Maximum gradients held in the by-value kernel launch descriptor.  Each
// input occupies one 64-bit pointer and one int64_t length; 128 slots keep the
// descriptor at 2,048 bytes and the full launch payload at 2,096 bytes, below
// CUDA's legacy 4,096-byte parameter limit.  Python exports and validates this
// ABI capacity before building or caching a gradient-pack plan; the native
// entry point retains the final defense.
inline constexpr int64_t kMaxGradientPackInputs = 128;

// This is work-partition geometry, not an input-capacity or CUDA-ABI limit.
// Each CUDA block owns a contiguous chunk of at most 4,096 flattened tensor
// values, traversed cooperatively by the kernel's private launch width. Python
// reads this value from the extension to compute each segment's cumulative
// block-end metadata, so the host metadata and CUDA kernel must use the same
// value.
// Changing it alters the launch grid and work per block, but not the number of
// input descriptors. Its numerical equality to CUDA's legacy 4,096-byte
// kernel-parameter limit is coincidental: this constant counts values, not
// bytes.
inline constexpr int64_t kGradientPackValuesPerBlock = 4096;

// Internal trusted operation that executes a gradient-pack plan on the current
// CUDA stream.
// Each int64 metadata row is [destination offset, length, input index, source
// offset, cumulative block end]. An input index of -1, or a missing input
// tensor, writes zeros. Copy rows must exactly and monotonically cover the output,
// and the last cumulative block end must equal num_blocks. The host caller is
// responsible for validating those invariants before upload. This is a
// no-autograd out operation: output must be a
// non-inference tensor without gradients, must not alias any input or metadata,
// and has its version counter incremented after the asynchronous launch. Input
// allocations are recorded on the launch stream so they may be released after
// this function returns.
torch::Tensor pack_gradients_cuda(
    torch::Tensor output,
    const std::vector<std::optional<torch::Tensor>>& gradients,
    const torch::Tensor& copy_rows,
    int64_t num_blocks);

// Payload packing.

// Pack contiguous mixed-dtype fields into one caller-owned flat field-major
// payload on PyTorch's current CUDA stream. Each input has shape
// [local_count, widths[index]]. The one-dimensional output contains exactly
// max_count * sum(widths) values. Valid field payloads are concatenated in
// input order, cast to the floating output dtype, and followed by deterministic
// zero padding through output.numel().
//
// This is a no-autograd out operation. Output must be fresh, contiguous,
// floating point, and non-overlapping with every input. A normal output has its
// version incremented. An inference output is accepted only while inference
// mode is enabled. Every nonempty source allocation is recorded on the launch
// stream so temporary contiguous inputs may be released after this returns.
torch::Tensor pack_payload_cuda(
    torch::Tensor output,
    const std::vector<torch::Tensor>& inputs,
    const std::vector<int64_t>& widths,
    int64_t local_count,
    int64_t max_count);

// Pack adjacent layers directly into one caller-owned flat buffer on
// PyTorch's current CUDA stream. Each layer is laid out as its field-major
// valid values followed by zero padding through
// max_counts[layer] * sum(widths_by_layer[layer]). Counts, offsets, source
// pointers, and segment boundaries are rebuilt for every invocation.
//
// This is a no-autograd out operation with the same dtype, aliasing,
// inference-mode, and stream-lifetime contract as pack_payload_cuda. The
// native layer capacity is kMaxGroupedPayloadLayers. The single-layer entry
// point above remains independent.
torch::Tensor pack_grouped_payload_cuda(
    torch::Tensor output,
    const std::vector<std::vector<torch::Tensor>>& inputs_by_layer,
    const std::vector<std::vector<int64_t>>& widths_by_layer,
    const std::vector<int64_t>& local_counts,
    const std::vector<int64_t>& max_counts);

// Payload unpacking.

// This is work-partition geometry, not an output-layer or CUDA-ABI capacity.
// Each CUDA block owns at most 4,096 destination values; the grouped and
// field-major kernels choose their thread counts separately. Python reads this
// value from the extension to compute each segment's cumulative block-end
// metadata, so the host metadata and CUDA kernels must remain in lockstep.
// Changing it alters the launch grid and work per block. Its equality to the
// gradient-pack chunk size is incidental because the two kernels are tuned
// independently.
inline constexpr int64_t kPayloadUnpackValuesPerBlock = 4096;

// Unpack several adjacent field-major layers from one rank-major grouped
// payload using cached device-resident copy metadata. The launch descriptor
// stays constant-size as the process-group size grows.
//
// Each int64 metadata row is [output index, destination offset, length,
// source offset, cumulative block end]. Rows for every output are flattened
// into one ordered table. The host caller validates the exact source and
// destination cover before uploading the cached table.
//
// This is a no-autograd out operation. Outputs and input must not require
// gradients or alias one another or the metadata. Normal outputs have their
// version counters incremented; inference outputs are accepted only while
// inference mode is enabled. All touched allocations are recorded on the
// current CUDA stream.
std::vector<torch::Tensor> unpack_grouped_payload_cuda(
    std::vector<torch::Tensor> outputs,
    const torch::Tensor& rank_major_grouped,
    const torch::Tensor& segments,
    int64_t num_blocks);

// Internal trusted-metadata operation that unpacks a padded rank-major
// array-of-structures payload into one unpadded structure-of-arrays buffer on
// the current CUDA stream.
//
// Each int64 metadata row is [destination offset, length, source offset,
// field width, cumulative block end]. Rows must exactly cover the output in
// field-major order, and each source segment must address one rank's valid
// rows with `source_row_width` stride. The host caller is responsible for
// validating those invariants before upload.
//
// This is a no-autograd out operation. The Python caller owns the fresh output
// allocation; output and input must not require gradients or alias. A normal
// output has its version counter incremented. An inference output is accepted
// only while inference mode is enabled and has no version counter to update.
// Input allocations are recorded on the launch stream so callers may release
// the gathered payload after this function returns.
torch::Tensor unpack_payload_cuda(
    torch::Tensor output,
    const torch::Tensor& rank_major_aos,
    const torch::Tensor& segments,
    int64_t num_blocks,
    int64_t source_row_width);
