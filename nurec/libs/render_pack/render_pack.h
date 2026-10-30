// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Fused conversion: (H, W, 3) float32 in [0, 1] → (3, H, W) uint8 in [0, 255].
// Replaces the PyTorch chain `(x * 255).clamp(0, 255).to(uint8).permute(2,0,1).contiguous()`
// (5 kernels + 2 intermediate allocations) with a single CUDA kernel that writes the
// result directly into the caller-provided destination (e.g. a CUDA-IPC-exported
// buffer, so no extra D2D copy is needed).

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace nurec_render_pack {

// Launches the fused conversion on `stream`. dst must point to at least
// width * height * 3 bytes of GPU memory. Returns cudaSuccess on launch
// success, otherwise the cudaError_t from the launch.
cudaError_t launch_render_pack_hwc_f32_to_chw_u8(
    const float* src_hwc_f32,
    uint8_t* dst_chw_u8,
    uint32_t width,
    uint32_t height,
    cudaStream_t stream);

} // namespace nurec_render_pack
