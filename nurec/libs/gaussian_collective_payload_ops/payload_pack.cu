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
#include <c10/util/TypeCast.h>

#include <cstdint>
#include <limits>

namespace {

// Source pointers, cumulative element/block ends, and dtypes are passed by
// value. A 128-entry PayloadInputs descriptor is 3,200 bytes and leaves room
// for the remaining arguments below CUDA's legacy 4 KiB parameter limit. This
// is a per-layer ABI capacity, not a model-, row-, or device-derived limit.
constexpr int64_t kMaxPayloadPackInputs = 128;

static_assert(
    kMaxGroupedPayloadLayers == 8,
    "grouped payload layer capacity must match the grouped pack descriptor ABI");

// One descriptor supports one to eight layers, with an aggregate budget of
// 128 fields distributed unevenly across them. Larger schemas remain valid
// via the native per-layer launch fallback.
constexpr int64_t kMaxGroupedPayloadPackFields = 128;

// CTA width shared by the single-layer and one-launch grouped pack kernels.
// Threads traverse each value chunk with a blockDim.x stride.
constexpr int kThreadsPerBlock = 256;

// Number of contiguous source or padding values owned by one CUDA block. C++
// constructs the corresponding cumulative block ends internally, so this
// launch-tuning value does not need a Python export.
constexpr int64_t kPayloadPackValuesPerBlock = 4096;

// A grouped request needs at most one segment per nonempty field plus one
// zero-padding segment per layer.
constexpr int64_t kMaxGroupedPayloadPackSegments =
    kMaxGroupedPayloadPackFields + kMaxGroupedPayloadLayers;

enum class PayloadInputDType : uint8_t {
    Bool,
    UInt8,
    Int8,
    Int16,
    Int32,
    Int64,
    Half,
    BFloat16,
    Float,
    Double,
};

// Small-descriptor dispatch threshold: up to 16 inputs passes a 400-byte
// descriptor, while 17-128 inputs use the 3,200-byte general descriptor. This
// selects a launch specialization; it is not an input-capacity limit.
inline constexpr int64_t kFastPayloadPackInputs = 16;

template <int64_t Capacity>
struct PayloadInputs {
    const void* pointers[Capacity];
    int64_t element_ends[Capacity];
    int64_t block_ends[Capacity];
    PayloadInputDType dtypes[Capacity];
};

static_assert(sizeof(PayloadInputs<kFastPayloadPackInputs>) == 400);
static_assert(
    sizeof(PayloadInputs<kMaxPayloadPackInputs>) == 3200,
    "PayloadInputs must leave room below CUDA's legacy 4 KiB kernel-parameter limit");
static_assert(sizeof(PayloadInputs<kMaxGroupedPayloadPackSegments>) == 3400);
static_assert(
    sizeof(PayloadInputs<kMaxGroupedPayloadPackSegments>) + sizeof(void*) + 3 * sizeof(int64_t) < 4096,
    "grouped payload launch arguments must fit below CUDA's legacy 4 KiB limit");

PayloadInputDType payload_input_dtype(at::ScalarType dtype) {
    switch (dtype) {
    case at::ScalarType::Bool:
        return PayloadInputDType::Bool;
    case at::ScalarType::Byte:
        return PayloadInputDType::UInt8;
    case at::ScalarType::Char:
        return PayloadInputDType::Int8;
    case at::ScalarType::Short:
        return PayloadInputDType::Int16;
    case at::ScalarType::Int:
        return PayloadInputDType::Int32;
    case at::ScalarType::Long:
        return PayloadInputDType::Int64;
    case at::ScalarType::Half:
        return PayloadInputDType::Half;
    case at::ScalarType::BFloat16:
        return PayloadInputDType::BFloat16;
    case at::ScalarType::Float:
        return PayloadInputDType::Float;
    case at::ScalarType::Double:
        return PayloadInputDType::Double;
    default:
        TORCH_CHECK(
            false,
            "payload input dtype ",
            dtype,
            " is unsupported; expected a real boolean, integral, or floating-point tensor");
    }
}

template <typename output_t, typename input_t>
__device__ __forceinline__ output_t load_and_cast(const void* pointer, int64_t index) {
    return c10::convert<output_t>(static_cast<const input_t*>(pointer)[index]);
}

template <typename output_t, int64_t Capacity>
__device__ __forceinline__ output_t load_payload_value(
    const PayloadInputs<Capacity>& inputs,
    int64_t input_index,
    int64_t source_index) {
    switch (inputs.dtypes[input_index]) {
    case PayloadInputDType::Bool:
        return load_and_cast<output_t, bool>(inputs.pointers[input_index], source_index);
    case PayloadInputDType::UInt8:
        return load_and_cast<output_t, uint8_t>(inputs.pointers[input_index], source_index);
    case PayloadInputDType::Int8:
        return load_and_cast<output_t, int8_t>(inputs.pointers[input_index], source_index);
    case PayloadInputDType::Int16:
        return load_and_cast<output_t, int16_t>(inputs.pointers[input_index], source_index);
    case PayloadInputDType::Int32:
        return load_and_cast<output_t, int32_t>(inputs.pointers[input_index], source_index);
    case PayloadInputDType::Int64:
        return load_and_cast<output_t, int64_t>(inputs.pointers[input_index], source_index);
    case PayloadInputDType::Half:
        return load_and_cast<output_t, c10::Half>(inputs.pointers[input_index], source_index);
    case PayloadInputDType::BFloat16:
        return load_and_cast<output_t, c10::BFloat16>(inputs.pointers[input_index], source_index);
    case PayloadInputDType::Float:
        return load_and_cast<output_t, float>(inputs.pointers[input_index], source_index);
    case PayloadInputDType::Double:
        return load_and_cast<output_t, double>(inputs.pointers[input_index], source_index);
    }
    CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted payload-pack metadata: invalid input dtype");
    return output_t(0);
}

template <typename output_t, int64_t Capacity>
__global__ void pack_payload_kernel(
    output_t* __restrict__ output,
    int64_t output_numel,
    int64_t num_inputs,
    int64_t num_blocks,
    PayloadInputs<Capacity> inputs) {
    const int64_t block_index = blockIdx.x;
    if (C10_UNLIKELY(block_index < 0 || block_index >= num_blocks)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted payload-pack metadata: invalid block index");
        }
        return;
    }

    int64_t low  = 0;
    int64_t high = num_inputs;
    while (low < high) {
        const int64_t middle = low + (high - low) / 2;
        if (inputs.block_ends[middle] <= block_index) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }

    const int64_t input_index = low < num_inputs ? low : -1;
    const int64_t segment_start =
        input_index > 0    ? inputs.element_ends[input_index - 1]
        : input_index == 0 ? 0
        : num_inputs > 0   ? inputs.element_ends[num_inputs - 1]
                           : 0;
    const int64_t block_start =
        input_index > 0    ? inputs.block_ends[input_index - 1]
        : input_index == 0 ? 0
        : num_inputs > 0   ? inputs.block_ends[num_inputs - 1]
                           : 0;
    int64_t segment_length =
        input_index >= 0 ? inputs.element_ends[input_index] - segment_start : output_numel - segment_start;

    if (input_index < 0) {
        const int64_t padding_length = output_numel - segment_start;
        const int64_t padding_blocks =
            padding_length == 0 ? 0 : 1 + (padding_length - 1) / kPayloadPackValuesPerBlock;
        if (C10_UNLIKELY(block_index >= block_start + padding_blocks)) {
            if (threadIdx.x == 0) {
                CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted payload-pack metadata: missing block segment");
            }
            return;
        }
        segment_length = padding_length;
    }

    const int64_t chunk_offset = (block_index - block_start) * kPayloadPackValuesPerBlock;
    if (C10_UNLIKELY(chunk_offset < 0 || chunk_offset >= segment_length)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted payload-pack metadata: invalid segment chunk");
        }
        return;
    }
    const int64_t remaining = segment_length - chunk_offset;
    const int64_t length =
        remaining < kPayloadPackValuesPerBlock ? remaining : kPayloadPackValuesPerBlock;
    const int64_t destination_offset = segment_start + chunk_offset;
    if (C10_UNLIKELY(
            destination_offset < 0 || destination_offset > output_numel ||
            length > output_numel - destination_offset)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted payload-pack metadata: invalid destination range");
        }
        return;
    }

    for (int64_t index = threadIdx.x; index < length; index += blockDim.x) {
        output[destination_offset + index] =
            input_index < 0 ? output_t(0) : load_payload_value<output_t>(inputs, input_index, chunk_offset + index);
    }
}

template <typename output_t>
__global__ void pack_grouped_payload_kernel(
    output_t* __restrict__ output,
    int64_t output_numel,
    int64_t num_segments,
    int64_t num_blocks,
    PayloadInputs<kMaxGroupedPayloadPackSegments> segments) {
    const int64_t block_index = blockIdx.x;
    if (C10_UNLIKELY(block_index < 0 || block_index >= num_blocks)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted grouped-payload metadata: invalid block index");
        }
        return;
    }

    int64_t low  = 0;
    int64_t high = num_segments;
    while (low < high) {
        const int64_t middle = low + (high - low) / 2;
        if (segments.block_ends[middle] <= block_index) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (C10_UNLIKELY(low >= num_segments)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted grouped-payload metadata: missing block segment");
        }
        return;
    }

    const int64_t segment_index  = low;
    const int64_t segment_start  = segment_index == 0 ? 0 : segments.element_ends[segment_index - 1];
    const int64_t segment_end    = segments.element_ends[segment_index];
    const int64_t block_start    = segment_index == 0 ? 0 : segments.block_ends[segment_index - 1];
    const int64_t segment_length = segment_end - segment_start;
    const int64_t chunk_offset   = (block_index - block_start) * kPayloadPackValuesPerBlock;
    if (C10_UNLIKELY(segment_length <= 0 || chunk_offset < 0 || chunk_offset >= segment_length)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted grouped-payload metadata: invalid segment chunk");
        }
        return;
    }

    const int64_t remaining          = segment_length - chunk_offset;
    const int64_t length             = remaining < kPayloadPackValuesPerBlock ? remaining : kPayloadPackValuesPerBlock;
    const int64_t destination_offset = segment_start + chunk_offset;
    if (C10_UNLIKELY(
            segment_end > output_numel || destination_offset < 0 || destination_offset > output_numel ||
            length > output_numel - destination_offset)) {
        if (threadIdx.x == 0) {
            CUDA_KERNEL_ASSERT_MSG(false, "malformed trusted grouped-payload metadata: invalid destination range");
        }
        return;
    }

    const void* source = segments.pointers[segment_index];
    for (int64_t index = threadIdx.x; index < length; index += blockDim.x) {
        output[destination_offset + index] = source == nullptr
                                                 ? output_t(0)
                                                 : load_payload_value<output_t>(segments, segment_index, chunk_offset + index);
    }
}

void validate_payload_output(const torch::Tensor& output, int64_t expected_dimensions) {
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(
        output.dim() == expected_dimensions,
        "output must be ",
        expected_dimensions == 1 ? "one" : "two",
        "-dimensional, got ",
        output.dim(),
        " dimensions");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(!output.requires_grad(), "output must not require gradients");
    TORCH_CHECK(
        !output.is_inference() || c10::InferenceMode::is_enabled(),
        "an inference output is only valid while inference mode is enabled");
    TORCH_CHECK(output.is_floating_point(), "output must have a floating-point dtype, got ", output.scalar_type());
    TORCH_CHECK(is_supported_payload_dtype(output.scalar_type()), "unsupported payload output dtype ", output.scalar_type());
}

void checked_accumulate(int64_t value, int64_t& total, const char* description) {
    TORCH_CHECK(value >= 0, description, " must be nonnegative, got ", value);
    TORCH_CHECK(value <= std::numeric_limits<int64_t>::max() - total, description, " overflows int64");
    total += value;
}

int64_t checked_product(int64_t left, int64_t right, const char* description) {
    TORCH_CHECK(left >= 0 && right >= 0, description, " factors must be nonnegative");
    TORCH_CHECK(left == 0 || right <= std::numeric_limits<int64_t>::max() / left, description, " overflows int64");
    return left * right;
}

int64_t validate_payload_pack(
    const torch::Tensor& output,
    const std::vector<torch::Tensor>& inputs,
    const std::vector<int64_t>& widths,
    int64_t local_count,
    int64_t max_count) {
    validate_payload_output(output, 1);
    TORCH_CHECK(!inputs.empty(), "payload packing requires at least one input");
    TORCH_CHECK(
        inputs.size() == widths.size(),
        "payload input count ",
        inputs.size(),
        " does not match width count ",
        widths.size());
    TORCH_CHECK(
        static_cast<int64_t>(inputs.size()) <= kMaxPayloadPackInputs,
        "payload packing supports at most ",
        kMaxPayloadPackInputs,
        " inputs, got ",
        inputs.size());

    TORCH_CHECK(
        max_count >= 0 && local_count >= 0 && local_count <= max_count,
        "local count ",
        local_count,
        " must be between zero and maximum count ",
        max_count);

    int64_t row_width  = 0;
    int64_t num_blocks = 0;
    for (size_t index = 0; index < inputs.size(); ++index) {
        const torch::Tensor& input = inputs[index];
        const int64_t width        = widths[index];
        TORCH_CHECK(width >= 0, "payload input ", index, " has negative width ", width);
        checked_accumulate(width, row_width, "payload row width");

        TORCH_CHECK(input.defined(), "payload input ", index, " is undefined");
        TORCH_CHECK(input.is_cuda(), "payload input ", index, " must be a CUDA tensor");
        TORCH_CHECK(
            input.get_device() == output.get_device(),
            "payload input ",
            index,
            " is on CUDA device ",
            input.get_device(),
            ", expected ",
            output.get_device());
        TORCH_CHECK(input.dim() == 2, "payload input ", index, " must be two-dimensional, got ", input.dim());
        TORCH_CHECK(
            input.size(0) == local_count && input.size(1) == width,
            "payload input ",
            index,
            " has shape ",
            input.sizes(),
            ", expected [",
            local_count,
            ", ",
            width,
            "]");
        TORCH_CHECK(input.is_contiguous(), "payload input ", index, " must be contiguous");
        payload_input_dtype(input.scalar_type());
        at::assert_no_overlap(output, input);

        if (input.numel() > 0) {
            const int64_t blocks = 1 + (input.numel() - 1) / kPayloadPackValuesPerBlock;
            TORCH_CHECK(
                blocks <= std::numeric_limits<int32_t>::max() - num_blocks,
                "payload packing block count overflows int32");
            num_blocks += blocks;
        }
    }
    const int64_t expected_numel = checked_product(max_count, row_width, "payload output size");
    TORCH_CHECK(
        output.numel() == expected_numel,
        "payload output has ",
        output.numel(),
        " values, expected ",
        expected_numel,
        " for maximum count ",
        max_count,
        " and row width ",
        row_width);
    const int64_t valid_numel   = checked_product(local_count, row_width, "payload valid size");
    const int64_t padding_numel = output.numel() - valid_numel;
    TORCH_CHECK(padding_numel >= 0, "payload valid rows exceed output size");
    if (padding_numel > 0) {
        const int64_t blocks = 1 + (padding_numel - 1) / kPayloadPackValuesPerBlock;
        TORCH_CHECK(
            blocks <= std::numeric_limits<int32_t>::max() - num_blocks,
            "payload packing block count overflows int32");
        num_blocks += blocks;
    }
    TORCH_CHECK(output.numel() == 0 || num_blocks > 0, "nonempty payload packing requires positive work");
    return num_blocks;
}

template <typename scalar_t, int64_t Capacity>
C10_ALWAYS_INLINE void fill_and_launch_payload_pack(
    torch::Tensor& output,
    const std::vector<torch::Tensor>& inputs,
    int64_t num_blocks,
    const c10::cuda::CUDAStream& stream) {
    PayloadInputs<Capacity> packed_inputs{};
    int64_t element_end = 0;
    int64_t block_end   = 0;
    for (size_t index = 0; index < inputs.size(); ++index) {
        const int64_t input_numel = inputs[index].numel();
        element_end += input_numel;
        if (input_numel > 0) {
            block_end += 1 + (input_numel - 1) / kPayloadPackValuesPerBlock;
        }
        packed_inputs.pointers[index]     = input_numel > 0 ? inputs[index].const_data_ptr() : nullptr;
        packed_inputs.element_ends[index] = element_end;
        packed_inputs.block_ends[index]   = block_end;
        packed_inputs.dtypes[index]       = payload_input_dtype(inputs[index].scalar_type());
    }
    pack_payload_kernel<scalar_t, Capacity><<<num_blocks, kThreadsPerBlock, 0, stream>>>(
        output.data_ptr<scalar_t>(),
        output.numel(),
        inputs.size(),
        num_blocks,
        packed_inputs);
}

void launch_payload_pack(
    torch::Tensor& output,
    const std::vector<torch::Tensor>& inputs,
    int64_t num_blocks) {
    if (output.numel() == 0) {
        return;
    }

    const c10::cuda::OptionalCUDAGuard device_guard(torch::device_of(output));
    const auto stream = c10::cuda::getCurrentCUDAStream(output.get_device());

    c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), stream);
    for (const torch::Tensor& input : inputs) {
        if (input.numel() > 0) {
            c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), stream);
        }
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(),
        "pack_payload_cuda",
        [&] {
            if (inputs.size() <= kFastPayloadPackInputs) {
                fill_and_launch_payload_pack<scalar_t, kFastPayloadPackInputs>(
                    output, inputs, num_blocks, stream);
            } else {
                fill_and_launch_payload_pack<scalar_t, kMaxPayloadPackInputs>(
                    output, inputs, num_blocks, stream);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

torch::Tensor pack_payload_cuda(
    torch::Tensor output,
    const std::vector<torch::Tensor>& inputs,
    const std::vector<int64_t>& widths,
    int64_t local_count,
    int64_t max_count) {
    const int64_t num_blocks = validate_payload_pack(output, inputs, widths, local_count, max_count);
    launch_payload_pack(output, inputs, num_blocks);
    if (!output.is_inference()) {
        output.unsafeGetTensorImpl()->bump_version();
    }
    return output;
}

torch::Tensor pack_grouped_payload_cuda(
    torch::Tensor output,
    const std::vector<std::vector<torch::Tensor>>& inputs_by_layer,
    const std::vector<std::vector<int64_t>>& widths_by_layer,
    const std::vector<int64_t>& local_counts,
    const std::vector<int64_t>& max_counts) {
    validate_payload_output(output, 1);

    const int64_t num_layers = inputs_by_layer.size();
    TORCH_CHECK(num_layers > 0, "grouped payload packing requires at least one layer");
    TORCH_CHECK(
        num_layers <= kMaxGroupedPayloadLayers,
        "grouped payload packing supports at most ",
        kMaxGroupedPayloadLayers,
        " layers, got ",
        num_layers);
    TORCH_CHECK(
        widths_by_layer.size() == inputs_by_layer.size(),
        "grouped payload layer count ",
        inputs_by_layer.size(),
        " does not match width layer count ",
        widths_by_layer.size());
    TORCH_CHECK(
        local_counts.size() == inputs_by_layer.size(),
        "grouped payload layer count ",
        inputs_by_layer.size(),
        " does not match local-count length ",
        local_counts.size());
    TORCH_CHECK(
        max_counts.size() == inputs_by_layer.size(),
        "grouped payload layer count ",
        inputs_by_layer.size(),
        " does not match max-count length ",
        max_counts.size());

    int64_t aggregate_fields = 0;
    for (const auto& inputs : inputs_by_layer) {
        TORCH_CHECK(!inputs.empty(), "each grouped payload layer requires at least one input");
        TORCH_CHECK(
            inputs.size() <= static_cast<size_t>(std::numeric_limits<int64_t>::max() - aggregate_fields),
            "grouped payload field count overflows int64");
        aggregate_fields += static_cast<int64_t>(inputs.size());
    }

    // Keep descriptor capacity private to this native operation. Schemas that
    // exceed the one-launch table remain valid when every individual layer
    // fits the independent per-layer pack ABI: pack each layer into its slice
    // of the same grouped buffer, then use the unchanged grouped collective
    // and unpack path.
    if (aggregate_fields > kMaxGroupedPayloadPackFields) {
        std::vector<torch::Tensor> layer_outputs;
        std::vector<int64_t> layer_num_blocks;
        layer_outputs.reserve(inputs_by_layer.size());
        layer_num_blocks.reserve(inputs_by_layer.size());
        int64_t output_offset = 0;
        for (size_t layer = 0; layer < inputs_by_layer.size(); ++layer) {
            const auto& inputs = inputs_by_layer[layer];
            const auto& widths = widths_by_layer[layer];
            TORCH_CHECK(
                inputs.size() == widths.size(),
                "grouped payload layer ",
                layer,
                " input count ",
                inputs.size(),
                " does not match width count ",
                widths.size());

            const int64_t local_count = local_counts[layer];
            const int64_t max_count   = max_counts[layer];
            TORCH_CHECK(
                local_count >= 0 && max_count >= local_count,
                "grouped payload layer ",
                layer,
                " local count ",
                local_count,
                " must be between zero and maximum count ",
                max_count);

            int64_t row_width = 0;
            for (size_t field = 0; field < widths.size(); ++field) {
                TORCH_CHECK(
                    widths[field] >= 0,
                    "grouped payload layer ",
                    layer,
                    " field ",
                    field,
                    " has negative width ",
                    widths[field]);
                checked_accumulate(widths[field], row_width, "grouped payload row width");
            }
            const int64_t layer_span = checked_product(max_count, row_width, "grouped payload layer span");
            TORCH_CHECK(
                output_offset <= output.numel() && layer_span <= output.numel() - output_offset,
                "grouped payload layer ",
                layer,
                " exceeds output size ",
                output.numel());
            torch::Tensor layer_output = output.narrow(0, output_offset, layer_span);
            layer_num_blocks.push_back(validate_payload_pack(layer_output, inputs, widths, local_count, max_count));
            for (const torch::Tensor& input : inputs) {
                at::assert_no_overlap(output, input);
            }
            layer_outputs.push_back(std::move(layer_output));
            output_offset += layer_span;
        }
        TORCH_CHECK(
            output_offset == output.numel(),
            "grouped payload output has ",
            output.numel(),
            " values, expected ",
            output_offset);
        for (size_t layer = 0; layer < inputs_by_layer.size(); ++layer) {
            launch_payload_pack(layer_outputs[layer], inputs_by_layer[layer], layer_num_blocks[layer]);
        }
        if (!output.is_inference()) {
            output.unsafeGetTensorImpl()->bump_version();
        }
        return output;
    }

    PayloadInputs<kMaxGroupedPayloadPackSegments> segments{};
    int64_t num_segments = 0;
    int64_t num_blocks   = 0;
    int64_t output_end   = 0;
    for (size_t layer = 0; layer < inputs_by_layer.size(); ++layer) {
        const auto& inputs = inputs_by_layer[layer];
        const auto& widths = widths_by_layer[layer];
        TORCH_CHECK(
            inputs.size() == widths.size(),
            "grouped payload layer ",
            layer,
            " input count ",
            inputs.size(),
            " does not match width count ",
            widths.size());

        const int64_t local_count = local_counts[layer];
        const int64_t max_count   = max_counts[layer];
        TORCH_CHECK(
            local_count >= 0 && max_count >= local_count,
            "grouped payload layer ",
            layer,
            " local count ",
            local_count,
            " must be between zero and maximum count ",
            max_count);

        int64_t row_width         = 0;
        int64_t layer_valid_numel = 0;
        for (size_t field = 0; field < inputs.size(); ++field) {
            const torch::Tensor& input = inputs[field];
            const int64_t width        = widths[field];
            TORCH_CHECK(width >= 0, "grouped payload layer ", layer, " field ", field, " has negative width ", width);
            checked_accumulate(width, row_width, "grouped payload row width");

            TORCH_CHECK(input.defined(), "grouped payload layer ", layer, " field ", field, " is undefined");
            TORCH_CHECK(input.is_cuda(), "grouped payload layer ", layer, " field ", field, " must be a CUDA tensor");
            TORCH_CHECK(
                input.get_device() == output.get_device(),
                "grouped payload layer ",
                layer,
                " field ",
                field,
                " is on CUDA device ",
                input.get_device(),
                ", expected ",
                output.get_device());
            TORCH_CHECK(
                input.dim() == 2,
                "grouped payload layer ",
                layer,
                " field ",
                field,
                " must be two-dimensional, got ",
                input.dim());
            TORCH_CHECK(
                input.size(0) == local_count && input.size(1) == width,
                "grouped payload layer ",
                layer,
                " field ",
                field,
                " has shape ",
                input.sizes(),
                ", expected [",
                local_count,
                ", ",
                width,
                "]");
            TORCH_CHECK(input.is_contiguous(), "grouped payload layer ", layer, " field ", field, " must be contiguous");
            const PayloadInputDType dtype = payload_input_dtype(input.scalar_type());
            at::assert_no_overlap(output, input);

            const int64_t input_numel = input.numel();
            checked_accumulate(input_numel, layer_valid_numel, "grouped payload valid size");
            if (input_numel == 0) {
                continue;
            }
            TORCH_CHECK(
                num_segments < kMaxGroupedPayloadPackSegments,
                "grouped payload segment count exceeds capacity ",
                kMaxGroupedPayloadPackSegments);
            checked_accumulate(input_numel, output_end, "grouped payload output size");
            const int64_t blocks = 1 + (input_numel - 1) / kPayloadPackValuesPerBlock;
            TORCH_CHECK(
                blocks <= std::numeric_limits<int32_t>::max() - num_blocks,
                "grouped payload packing block count overflows int32");
            num_blocks += blocks;
            segments.pointers[num_segments]     = input.const_data_ptr();
            segments.element_ends[num_segments] = output_end;
            segments.block_ends[num_segments]   = num_blocks;
            segments.dtypes[num_segments]       = dtype;
            ++num_segments;
        }

        const int64_t expected_valid_numel = checked_product(local_count, row_width, "grouped payload valid size");
        TORCH_CHECK(
            layer_valid_numel == expected_valid_numel,
            "grouped payload layer ",
            layer,
            " fields contain ",
            layer_valid_numel,
            " values, expected ",
            expected_valid_numel);
        const int64_t layer_span    = checked_product(max_count, row_width, "grouped payload layer span");
        const int64_t padding_numel = layer_span - expected_valid_numel;
        if (padding_numel > 0) {
            TORCH_CHECK(
                num_segments < kMaxGroupedPayloadPackSegments,
                "grouped payload segment count exceeds capacity ",
                kMaxGroupedPayloadPackSegments);
            checked_accumulate(padding_numel, output_end, "grouped payload output size");
            const int64_t blocks = 1 + (padding_numel - 1) / kPayloadPackValuesPerBlock;
            TORCH_CHECK(
                blocks <= std::numeric_limits<int32_t>::max() - num_blocks,
                "grouped payload packing block count overflows int32");
            num_blocks += blocks;
            segments.pointers[num_segments]     = nullptr;
            segments.element_ends[num_segments] = output_end;
            segments.block_ends[num_segments]   = num_blocks;
            ++num_segments;
        }
    }

    TORCH_CHECK(
        output.numel() == output_end,
        "grouped payload output has ",
        output.numel(),
        " values, expected ",
        output_end);
    if (output.numel() == 0) {
        TORCH_CHECK(num_segments == 0 && num_blocks == 0, "empty grouped payload must not contain work segments");
        if (!output.is_inference()) {
            output.unsafeGetTensorImpl()->bump_version();
        }
        return output;
    }
    TORCH_CHECK(num_segments > 0 && num_blocks > 0, "nonempty grouped payload packing requires positive work");

    const c10::cuda::OptionalCUDAGuard device_guard(torch::device_of(output));
    const auto stream = c10::cuda::getCurrentCUDAStream(output.get_device());
    c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), stream);
    for (const auto& inputs : inputs_by_layer) {
        for (const torch::Tensor& input : inputs) {
            if (input.numel() > 0) {
                c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), stream);
            }
        }
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(),
        "pack_grouped_payload_cuda",
        [&] {
            pack_grouped_payload_kernel<scalar_t><<<num_blocks, kThreadsPerBlock, 0, stream>>>(
                output.data_ptr<scalar_t>(),
                output.numel(),
                num_segments,
                num_blocks,
                segments);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!output.is_inference()) {
        output.unsafeGetTensorImpl()->bump_version();
    }
    return output;
}
