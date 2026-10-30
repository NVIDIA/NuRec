// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "render_pack.h"

namespace nurec_render_pack {

namespace {

// One thread per (x, y). Reads three interleaved float32 channels at (y, x, c),
// scales to 0–255, saturates, and writes three uint8 outputs to the planar
// destinations dst[c, y, x]. Coalesced reads (3 consecutive f32 per thread) and
// coalesced writes within each plane.
__global__ void render_pack_kernel(
    const float* __restrict__ src_hwc_f32,
    uint8_t* __restrict__ dst_chw_u8,
    uint32_t width,
    uint32_t height) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const uint32_t plane_size = width * height;
    const uint32_t pix_idx    = y * width + x;
    const uint32_t src_idx    = pix_idx * 3u;

    auto sat_u8 = [] __device__(float v) -> uint8_t {
        // Match torch.tensor(...).to(torch.uint8): clamp [0, 255] then truncate
        // toward zero. (Round-to-nearest would differ by ±1 on ~half the pixels.)
        v *= 255.0f;
        v = fminf(255.0f, fmaxf(0.0f, v));
        return static_cast<uint8_t>(__float2uint_rz(v));
    };

    dst_chw_u8[pix_idx]                   = sat_u8(src_hwc_f32[src_idx + 0u]);
    dst_chw_u8[pix_idx + plane_size]      = sat_u8(src_hwc_f32[src_idx + 1u]);
    dst_chw_u8[pix_idx + 2u * plane_size] = sat_u8(src_hwc_f32[src_idx + 2u]);
}

} // namespace

cudaError_t launch_render_pack_hwc_f32_to_chw_u8(
    const float* src_hwc_f32,
    uint8_t* dst_chw_u8,
    uint32_t width,
    uint32_t height,
    cudaStream_t stream) {
    if (width == 0u || height == 0u)
        return cudaSuccess;

    constexpr dim3 block(32u, 8u, 1u);
    dim3 grid((width + block.x - 1u) / block.x,
              (height + block.y - 1u) / block.y,
              1u);
    render_pack_kernel<<<grid, block, 0, stream>>>(
        src_hwc_f32, dst_chw_u8, width, height);
    return cudaGetLastError();
}

} // namespace nurec_render_pack
