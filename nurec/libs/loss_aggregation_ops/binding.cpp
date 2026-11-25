// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "finalize_weighted_losses.h"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cmath>
#include <optional>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<at::Tensor> cast_tensor_sequence(const py::sequence& values, const char* message) {
    std::vector<at::Tensor> tensors;
    tensors.reserve(values.size());
    for (const py::handle value : values) {
        try {
            tensors.emplace_back(py::cast<at::Tensor>(value));
        } catch (const py::cast_error&) {
            throw py::type_error(message);
        }
    }
    return tensors;
}

std::vector<double> cast_weight_sequence(const py::sequence& values) {
    std::vector<double> weights;
    weights.reserve(values.size());
    for (const py::handle value : values) {
        try {
            weights.emplace_back(py::cast<double>(value));
        } catch (const py::cast_error&) {
            throw py::type_error("scalar_weights must contain only real numbers");
        }
    }
    return weights;
}

std::optional<at::Tensor> cast_optional_tensor(const py::object& value, const char* message) {
    if (value.is_none()) {
        return std::nullopt;
    }
    try {
        return py::cast<at::Tensor>(value);
    } catch (const py::cast_error&) {
        throw py::type_error(message);
    }
}

void check_cuda_float32(const at::Tensor& tensor, const at::Device device, const char* name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.device() == device, name, " must be on ", device, ", got ", tensor.device());
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32, got ", tensor.scalar_type());
}

void check_cuda_scalar_float(const at::Tensor& tensor, const at::Device device, const char* name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.device() == device, name, " must be on ", device, ", got ", tensor.device());
    const at::ScalarType dtype = tensor.scalar_type();
    TORCH_CHECK(
        dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16,
        name,
        " must be float16, bfloat16, or float32, got ",
        dtype);
}

void check_scalar_weights(const std::vector<double>& weights, const int64_t expected) {
    TORCH_CHECK(
        static_cast<int64_t>(weights.size()) == expected,
        "finalize_accumulated_losses: expected ",
        expected,
        " scalar weights, got ",
        weights.size());
    for (const double weight : weights) {
        TORCH_CHECK(std::isfinite(weight), "finalize_accumulated_losses: scalar weights must be finite, got ", weight);
    }
}

void check_weighted_vector(const at::Tensor& tensor, const at::Device device, const char* name) {
    check_cuda_float32(tensor, device, name);
    TORCH_CHECK(tensor.dim() == 1, name, " must be 1-D, got ", tensor.sizes());
}

void check_scalar_values(at::TensorList values, const at::Device device, const char* name) {
    for (const at::Tensor& tensor : values) {
        check_cuda_scalar_float(tensor, device, name);
        TORCH_CHECK(tensor.numel() == 1, name, " must contain one element, got ", tensor.numel());
    }
}

void check_weighted_values(const at::Tensor& tensor, const at::Device device, const int64_t numel, const char* name) {
    check_cuda_float32(tensor, device, name);
    TORCH_CHECK(
        tensor.dim() == 1 && tensor.size(0) == numel,
        name,
        " must have shape [",
        numel,
        "], got ",
        tensor.sizes());
}

void finalize_accumulated_losses_fwd(
    at::Tensor weighted_vector,
    const py::sequence& scalar_value_values,
    const py::sequence& scalar_weight_values,
    at::Tensor total,
    at::Tensor weighted_values) {
    std::vector<at::Tensor> scalar_values =
        cast_tensor_sequence(scalar_value_values, "scalar_values must contain only torch.Tensor values");
    std::vector<double> scalar_weights = cast_weight_sequence(scalar_weight_values);

    TORCH_CHECK(total.defined() && total.is_cuda(), "finalize_accumulated_losses_fwd: total must be a CUDA tensor");
    const at::Device device = total.device();
    check_cuda_float32(total, device, "total");
    TORCH_CHECK(total.dim() == 0, "finalize_accumulated_losses_fwd: total must be a scalar, got ", total.sizes());
    check_weighted_vector(weighted_vector, device, "weighted_vector");
    check_scalar_values(scalar_values, device, "scalar_values entry");
    check_scalar_weights(scalar_weights, static_cast<int64_t>(scalar_values.size()));
    check_weighted_values(
        weighted_values, device, weighted_vector.numel() + static_cast<int64_t>(scalar_values.size()), "weighted_values");

    py::gil_scoped_release release;
    nurec::loss_aggregation_ops::launch_finalize_accumulated_losses_fwd(
        weighted_vector, scalar_values, scalar_weights, total, weighted_values);
}

void finalize_accumulated_losses_bwd(
    const py::object& v_total_value,
    const py::object& v_weighted_values_value,
    const py::sequence& scalar_weight_values,
    at::Tensor v_weighted_vector,
    const py::sequence& v_scalar_value_values) {
    std::optional<at::Tensor> v_total = cast_optional_tensor(v_total_value, "v_total must be a torch.Tensor or None");
    std::optional<at::Tensor> v_weighted_values =
        cast_optional_tensor(v_weighted_values_value, "v_weighted_values must be a torch.Tensor or None");
    std::vector<double> scalar_weights = cast_weight_sequence(scalar_weight_values);
    std::vector<at::Tensor> v_scalar_values =
        cast_tensor_sequence(v_scalar_value_values, "v_scalar_values must contain only torch.Tensor values");

    TORCH_CHECK(
        v_weighted_vector.defined() && v_weighted_vector.is_cuda(),
        "finalize_accumulated_losses_bwd: v_weighted_vector must be a CUDA tensor");
    const at::Device device = v_weighted_vector.device();
    check_weighted_vector(v_weighted_vector, device, "v_weighted_vector");
    check_scalar_values(v_scalar_values, device, "v_scalar_values entry");
    check_scalar_weights(scalar_weights, static_cast<int64_t>(v_scalar_values.size()));
    if (v_total.has_value()) {
        check_cuda_float32(v_total.value(), device, "v_total");
        TORCH_CHECK(v_total->numel() == 1, "finalize_accumulated_losses_bwd: v_total must contain one element");
    }
    if (v_weighted_values.has_value()) {
        check_weighted_values(
            v_weighted_values.value(),
            device,
            v_weighted_vector.numel() + static_cast<int64_t>(v_scalar_values.size()),
            "v_weighted_values");
    }

    py::gil_scoped_release release;
    nurec::loss_aggregation_ops::launch_finalize_accumulated_losses_bwd(
        v_total.has_value() ? &v_total.value() : nullptr,
        v_weighted_values.has_value() ? &v_weighted_values.value() : nullptr,
        scalar_weights,
        v_weighted_vector,
        v_scalar_values);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "finalize_accumulated_losses_fwd",
        &finalize_accumulated_losses_fwd,
        py::arg("weighted_vector"),
        py::arg("scalar_values"),
        py::arg("scalar_weights"),
        py::arg("total"),
        py::arg("weighted_values"));
    module.def(
        "finalize_accumulated_losses_bwd",
        &finalize_accumulated_losses_bwd,
        py::arg("v_total"),
        py::arg("v_weighted_values"),
        py::arg("scalar_weights"),
        py::arg("v_weighted_vector"),
        py::arg("v_scalar_values"));
}
