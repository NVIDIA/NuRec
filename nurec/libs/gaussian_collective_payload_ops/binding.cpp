// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gaussian_collective_payload_ops.h"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <optional>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<int64_t> cast_int_sequence(const py::sequence& values, const char* message) {
    std::vector<int64_t> result;
    result.reserve(values.size());
    for (const py::handle item : values) {
        try {
            result.emplace_back(py::cast<int64_t>(item));
        } catch (const py::cast_error&) {
            throw py::type_error(message);
        }
    }
    return result;
}

torch::Tensor pack_gradients(
    torch::Tensor output,
    const py::sequence& gradient_list,
    const torch::Tensor& copy_rows,
    int64_t num_blocks) {
    std::vector<std::optional<torch::Tensor>> gradients;
    gradients.reserve(gradient_list.size());
    for (const py::handle item : gradient_list) {
        if (item.is_none()) {
            gradients.emplace_back(std::nullopt);
            continue;
        }
        try {
            gradients.emplace_back(py::cast<torch::Tensor>(item));
        } catch (const py::cast_error&) {
            throw py::type_error("gradients must contain only torch.Tensor or None values");
        }
    }

    torch::Tensor result;
    {
        py::gil_scoped_release release;
        result = pack_gradients_cuda(std::move(output), gradients, copy_rows, num_blocks);
    }
    return result;
}

torch::Tensor pack_payload(
    torch::Tensor output,
    const py::sequence& field_list,
    const py::sequence& width_list,
    int64_t local_count,
    int64_t max_count) {
    std::vector<torch::Tensor> fields;
    fields.reserve(field_list.size());
    for (const py::handle item : field_list) {
        if (item.is_none()) {
            throw py::type_error("payload fields must contain only torch.Tensor values");
        }
        try {
            fields.emplace_back(py::cast<torch::Tensor>(item));
        } catch (const py::cast_error&) {
            throw py::type_error("payload fields must contain only torch.Tensor values");
        }
    }

    std::vector<int64_t> widths;
    widths.reserve(width_list.size());
    for (const py::handle item : width_list) {
        try {
            widths.emplace_back(py::cast<int64_t>(item));
        } catch (const py::cast_error&) {
            throw py::type_error("payload widths must contain only integer values");
        }
    }

    torch::Tensor result;
    {
        py::gil_scoped_release release;
        result = pack_payload_cuda(std::move(output), fields, widths, local_count, max_count);
    }
    return result;
}

torch::Tensor pack_grouped_payload(
    torch::Tensor output,
    const py::sequence& layer_field_lists,
    const py::sequence& layer_width_lists,
    const py::sequence& local_count_list,
    const py::sequence& max_count_list) {
    std::vector<std::vector<torch::Tensor>> fields_by_layer;
    fields_by_layer.reserve(layer_field_lists.size());
    for (const py::handle layer_item : layer_field_lists) {
        if (!py::isinstance<py::list>(layer_item) && !py::isinstance<py::tuple>(layer_item)) {
            throw py::type_error("grouped payload fields must be a sequence of tensor sequences");
        }
        const py::sequence field_list = py::reinterpret_borrow<py::sequence>(layer_item);
        std::vector<torch::Tensor> fields;
        fields.reserve(field_list.size());
        for (const py::handle item : field_list) {
            if (item.is_none()) {
                throw py::type_error("grouped payload fields must contain only torch.Tensor values");
            }
            try {
                fields.emplace_back(py::cast<torch::Tensor>(item));
            } catch (const py::cast_error&) {
                throw py::type_error("grouped payload fields must contain only torch.Tensor values");
            }
        }
        fields_by_layer.emplace_back(std::move(fields));
    }

    std::vector<std::vector<int64_t>> widths_by_layer;
    widths_by_layer.reserve(layer_width_lists.size());
    for (const py::handle layer_item : layer_width_lists) {
        if (!py::isinstance<py::list>(layer_item) && !py::isinstance<py::tuple>(layer_item)) {
            throw py::type_error("grouped payload widths must be a sequence of integer sequences");
        }
        const py::sequence width_list = py::reinterpret_borrow<py::sequence>(layer_item);
        std::vector<int64_t> widths;
        widths.reserve(width_list.size());
        for (const py::handle item : width_list) {
            try {
                widths.emplace_back(py::cast<int64_t>(item));
            } catch (const py::cast_error&) {
                throw py::type_error("grouped payload widths must contain only integer values");
            }
        }
        widths_by_layer.emplace_back(std::move(widths));
    }

    std::vector<int64_t> local_counts =
        cast_int_sequence(local_count_list, "grouped payload local counts must contain only integer values");
    std::vector<int64_t> max_counts =
        cast_int_sequence(max_count_list, "grouped payload maximum counts must contain only integer values");

    torch::Tensor result;
    {
        py::gil_scoped_release release;
        result = pack_grouped_payload_cuda(
            std::move(output),
            fields_by_layer,
            widths_by_layer,
            local_counts,
            max_counts);
    }
    return result;
}

torch::Tensor unpack_payload(
    torch::Tensor output,
    const torch::Tensor& rank_major_aos,
    const torch::Tensor& segments,
    int64_t num_blocks,
    int64_t source_row_width) {
    torch::Tensor result;
    {
        py::gil_scoped_release release;
        result = unpack_payload_cuda(
            std::move(output),
            rank_major_aos,
            segments,
            num_blocks,
            source_row_width);
    }
    return result;
}

py::tuple unpack_grouped_payload(
    const py::sequence& output_list,
    const torch::Tensor& rank_major_grouped,
    const torch::Tensor& segments,
    int64_t num_blocks) {
    std::vector<torch::Tensor> outputs;
    outputs.reserve(output_list.size());
    for (const py::handle item : output_list) {
        if (item.is_none()) {
            throw py::type_error("grouped payload outputs must contain only torch.Tensor values");
        }
        try {
            outputs.emplace_back(py::cast<torch::Tensor>(item));
        } catch (const py::cast_error&) {
            throw py::type_error("grouped payload outputs must contain only torch.Tensor values");
        }
    }

    {
        py::gil_scoped_release release;
        outputs = unpack_grouped_payload_cuda(
            std::move(outputs),
            rank_major_grouped,
            segments,
            num_blocks);
    }

    py::tuple result(outputs.size());
    for (size_t layer = 0; layer < outputs.size(); ++layer) {
        result[layer] = outputs[layer];
    }
    return result;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("_MAX_GROUPED_PAYLOAD_LAYERS")      = py::int_(kMaxGroupedPayloadLayers);
    m.attr("_MAX_GRADIENT_PACK_INPUTS")        = py::int_(kMaxGradientPackInputs);
    m.attr("_GRADIENT_PACK_VALUES_PER_BLOCK")  = py::int_(kGradientPackValuesPerBlock);
    m.attr("_PAYLOAD_UNPACK_VALUES_PER_BLOCK") = py::int_(kPayloadUnpackValuesPerBlock);
    m.def(
        "_pack_gradients_trusted",
        &pack_gradients,
        py::arg("output"),
        py::arg("gradients"),
        py::arg("copy_rows"),
        py::arg("num_blocks"),
        R"doc(
Pack optional flattened CUDA gradients into a rank-major output tensor.

``copy_rows`` is a contiguous CUDA int64 tensor with shape ``[N, 5]``. Each
row contains ``[destination_offset, length, input_index, source_offset,
cumulative_block_end]``.
``input_index == -1`` writes zeros. A row that selects a ``None`` gradient
also writes zeros, allowing topology metadata to be cached before backward.

This is an internal trusted-metadata entry point. Rows must form an exact,
ordered, non-overlapping cover of ``[0, output.numel())``; cumulative block
ends must be exact and the final end must equal ``num_blocks``. Callers must
validate these invariants on the host before uploading ``copy_rows``.

The caller owns output allocation. ``output`` must not require gradients, must
not be an inference tensor, and must not overlap ``copy_rows`` or any gradient
input. A nonempty operation launches one CUDA packing kernel on PyTorch's
current stream; the supported empty operation launches none. Both increment
``output``'s autograd version and return ``output``.
)doc");
    m.def(
        "_pack_payload_trusted",
        &pack_payload,
        py::arg("output"),
        py::arg("fields"),
        py::arg("widths"),
        py::arg("local_count"),
        py::arg("max_count"),
        R"doc(
Pack flattened CUDA fields into one rank-local field-major payload.

``fields[index]`` is a contiguous CUDA tensor with shape
``[local_count, widths[index]]``. The fresh output is one-dimensional, has
``max_count * sum(widths)`` values, and uses one of the supported floating
payload dtypes. Valid fields are concatenated contiguously in field order with
payload-dtype casts; the remaining output receives deterministic zero padding.

The caller owns output allocation. ``output`` must not require gradients or
overlap any field input. A nonempty operation launches one CUDA packing kernel
on PyTorch's current stream; the supported empty operation launches none.
Source pointers are passed by value on every launch and source allocations are
recorded on that stream. A normal output has its autograd version incremented;
an inference output is accepted only while inference mode remains enabled.
)doc");
    m.def(
        "_pack_grouped_payload_trusted",
        &pack_grouped_payload,
        py::arg("output"),
        py::arg("fields_by_layer"),
        py::arg("widths_by_layer"),
        py::arg("local_counts"),
        py::arg("max_counts"),
        R"doc(
Pack adjacent field-major layer payloads directly into one flat buffer.

For each layer, ``fields_by_layer[layer][field]`` is a contiguous mixed-dtype
CUDA tensor with shape ``[local_counts[layer], widths_by_layer[layer][field]]``.
The layer contributes its valid fields in field order, followed by deterministic
zero padding through ``max_counts[layer] * sum(widths_by_layer[layer])``. Layers
are placed back-to-back in the fresh floating-point output.

Counts, layer offsets, source pointers, and copy segments are rebuilt on every
call and passed by value. Schemas that fit the fixed grouped launch descriptor
use one CUDA kernel. Larger supported schemas use the native per-layer packing
path and launch at most one kernel per nonempty layer slice into the same
output. The implementation validates its native layer, field, segment, and
per-layer descriptor capacities before launching. Empty operations launch no
kernels. Source and output allocations are recorded on PyTorch's current
stream. The output must not require gradients or overlap any field. A normal
output has its version incremented once; an inference output is accepted only
while inference mode remains enabled.
)doc");
    m.def(
        "_unpack_grouped_payload_trusted",
        &unpack_grouped_payload,
        py::arg("outputs"),
        py::arg("rank_major_grouped"),
        py::arg("segments"),
        py::arg("num_blocks"),
        R"doc(
Unpack several outputs from one rank-major grouped payload with device metadata.

``segments`` is a contiguous CUDA int64 tensor with shape ``[N, 5]``. Each
row contains ``[output_index, destination_offset, length, source_offset,
cumulative_block_end]``. Rows for all outputs share one ordered table and one
nonempty invocation launches exactly one CUDA kernel. Because rank-dependent
metadata resides in this tensor rather than the launch arguments, process-group
size is not capped by CUDA's legacy 4 KiB kernel-parameter limit.

This is an internal trusted-metadata entry point. The caller validates exact
destination coverage and source geometry before uploading and caching the
table. Source, metadata, and outputs must be non-overlapping current-device
CUDA tensors. Every normal output has its autograd version incremented once,
including an empty output; inference outputs are accepted only while inference
mode remains enabled.
)doc");
    m.def(
        "_unpack_payload_trusted",
        &unpack_payload,
        py::arg("output"),
        py::arg("rank_major_aos"),
        py::arg("segments"),
        py::arg("num_blocks"),
        py::arg("source_row_width"),
        R"doc(
Unpack a padded rank-major AoS Gaussian payload into one contiguous SoA buffer.

``segments`` is a contiguous CUDA int64 tensor with shape ``[N, 5]``. Each
row contains ``[destination_offset, length, source_offset, field_width,
cumulative_block_end]``. Rows select valid rank/field rectangles from the
padded input and place them consecutively in field-major output order.

This row layout is unrelated to the ``_pack_gradients_trusted`` layout above
despite the shared width: the third column is a source offset here and an
input index there.

This is an internal trusted-metadata entry point. The caller validates exact
destination coverage and source geometry, and supplies a fresh output that
does not overlap the source or metadata. A nonempty operation launches one
CUDA unpack kernel on PyTorch's current stream; the supported empty
operation launches none. A normal output has its autograd version incremented;
an inference output is accepted only while inference mode remains enabled.
)doc");
}
