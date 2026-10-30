// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Test-only pybind for the fused render-pack CUDA kernel. Kept separate from any
// transport module so the kernel's HWC-f32 -> CHW-u8 correctness can be unit-tested
// on GPU without pulling in transport-specific host libraries.

#include "render_pack.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

// src_hwc_f32 / dst_chw_u8 are raw device pointers (e.g. torch tensor data_ptr()).
void run_render_pack(uint64_t src_hwc_f32, uint64_t dst_chw_u8, uint32_t width, uint32_t height) {
    cudaError_t err = nurec_render_pack::launch_render_pack_hwc_f32_to_chw_u8(
        reinterpret_cast<const float*>(src_hwc_f32),
        reinterpret_cast<uint8_t*>(dst_chw_u8),
        width,
        height,
        /*stream=*/nullptr);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("render_pack launch failed: ") + cudaGetErrorString(err));
    err = cudaStreamSynchronize(nullptr);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("render_pack sync failed: ") + cudaGetErrorString(err));
}

} // namespace

PYBIND11_MODULE(_render_pack, m) {
    m.doc() = "Test-only binding for the fused render-pack CUDA kernel.";
    m.def(
        "run",
        &run_render_pack,
        py::arg("src_hwc_f32"),
        py::arg("dst_chw_u8"),
        py::arg("width"),
        py::arg("height"));
}
