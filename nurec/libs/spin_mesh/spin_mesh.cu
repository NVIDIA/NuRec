// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cassert>
#include <climits>
#include <cmath>
#include <cuda_runtime.h>

// Sentinel for empty cells. INT_MAX makes ``atomicMin`` "first (lowest-
// indexed) point wins" automatic without a CAS loop.
static constexpr int EMPTY_CELL = INT_MAX;

// Per-point spherical conversion + azimuth bin. (az, el) are written in
// float64 so the bin partition matches the CPU reference across edges.
__global__ void compute_spherical_and_az_bin_kernel(
    const float* __restrict__ points,   // (N, 3)
    double* __restrict__ azimuth_out,   // (N,)
    double* __restrict__ elevation_out, // (N,)
    int* __restrict__ az_bins_out,      // (N,)
    int n_points,
    int n_az_bins,
    double az_span_inv, // 1 / (2π + ε)
    int n_az_bins_minus_1) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_points)
        return;

    const float x       = points[idx * 3 + 0];
    const float y       = points[idx * 3 + 1];
    const float z       = points[idx * 3 + 2];
    const double radius = sqrt(static_cast<double>(x) * x +
                               static_cast<double>(y) * y +
                               static_cast<double>(z) * z);
    const double r_safe = max(radius, 1e-12);
    const double az     = atan2(static_cast<double>(y), static_cast<double>(x));
    double sin_arg      = static_cast<double>(z) / r_safe;
    if (sin_arg > 1.0)
        sin_arg = 1.0;
    if (sin_arg < -1.0)
        sin_arg = -1.0;
    const double el = asin(sin_arg);

    azimuth_out[idx]   = az;
    elevation_out[idx] = el;

    int az_bin = static_cast<int>(floor((az - (-M_PI)) * az_span_inv * n_az_bins));
    if (az_bin < 0)
        az_bin = 0;
    if (az_bin > n_az_bins_minus_1)
        az_bin = n_az_bins_minus_1;
    az_bins_out[idx] = az_bin;
}

// Fast path when the host already knows elevation bounds. Computes spherical
// coordinates only long enough to claim the scan-grid cell, avoiding temporary
// azimuth/elevation tensors and the GPU-to-host min/max reduction.
__global__ void bin_and_claim_cells_kernel(
    const float* __restrict__ points, // (N, 3)
    int* __restrict__ occupancy,      // (n_el_bins * n_az_bins,)
    int n_points,
    int n_el_bins,
    int n_az_bins,
    double el_min,
    double el_span_inv, // 1 / (el_max - el_min + ε)
    double az_span_inv, // 1 / (2π + ε)
    int n_el_bins_minus_1,
    int n_az_bins_minus_1) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_points)
        return;

    const float x       = points[idx * 3 + 0];
    const float y       = points[idx * 3 + 1];
    const float z       = points[idx * 3 + 2];
    const double radius = sqrt(static_cast<double>(x) * x +
                               static_cast<double>(y) * y +
                               static_cast<double>(z) * z);
    const double r_safe = max(radius, 1e-12);
    const double az     = atan2(static_cast<double>(y), static_cast<double>(x));
    double sin_arg      = static_cast<double>(z) / r_safe;
    if (sin_arg > 1.0)
        sin_arg = 1.0;
    if (sin_arg < -1.0)
        sin_arg = -1.0;
    const double el = asin(sin_arg);

    int az_bin = static_cast<int>(floor((az - (-M_PI)) * az_span_inv * n_az_bins));
    if (az_bin < 0)
        az_bin = 0;
    if (az_bin > n_az_bins_minus_1)
        az_bin = n_az_bins_minus_1;

    int el_bin = static_cast<int>(floor((el - el_min) * el_span_inv * n_el_bins));
    if (el_bin < 0)
        el_bin = 0;
    if (el_bin > n_el_bins_minus_1)
        el_bin = n_el_bins_minus_1;

    const int cell = el_bin * n_az_bins + az_bin;
    atomicMin(&occupancy[cell], idx);
}

// Compute el_bin (using the host-reduced el_min/el_span) and atomicMin
// the point's index into its (el_bin, az_bin) cell. Caller pre-fills
// ``occupancy`` with EMPTY_CELL.
__global__ void claim_cells_kernel(
    const double* __restrict__ elevations, // (N,)
    const int* __restrict__ az_bins,       // (N,)
    int* __restrict__ occupancy,           // (n_el_bins * n_az_bins,)
    int n_points,
    int n_el_bins,
    int n_az_bins,
    double el_min,
    double el_span_inv, // 1 / (el_max - el_min + ε)
    int n_el_bins_minus_1) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_points)
        return;

    const double el = elevations[idx];
    int el_bin      = static_cast<int>(floor((el - el_min) * el_span_inv * n_el_bins));
    if (el_bin < 0)
        el_bin = 0;
    if (el_bin > n_el_bins_minus_1)
        el_bin = n_el_bins_minus_1;

    const int az_bin = az_bins[idx];
    const int cell   = el_bin * n_az_bins + az_bin;
    atomicMin(&occupancy[cell], idx);
}

// One thread per (r, c) interior cell. Emits two triangles per quad whose
// four corners are all occupied. ``out_capacity`` = 2 * (H-1) * (W-1)
// is the tight geometric upper bound for this launch.
__global__ void emit_triangles_kernel(
    const int* __restrict__ occupancy, // (n_el_bins, n_az_bins) row-major
    int* __restrict__ triangles,       // (out_capacity, 3), sliced by caller
    int* __restrict__ out_count,       // (1,) atomic counter
    int n_el_bins,
    int n_az_bins,
    int out_capacity) {
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_el_bins - 1 || c >= n_az_bins - 1)
        return;

    const int p0 = occupancy[r * n_az_bins + c];
    const int p1 = occupancy[r * n_az_bins + (c + 1)];
    const int p2 = occupancy[(r + 1) * n_az_bins + c];
    const int p3 = occupancy[(r + 1) * n_az_bins + (c + 1)];
    if (p0 == EMPTY_CELL || p1 == EMPTY_CELL || p2 == EMPTY_CELL || p3 == EMPTY_CELL)
        return;

    const int write = atomicAdd(out_count, 2);
    assert(write + 1 < out_capacity);

    int* base1 = triangles + write * 3;
    base1[0]   = p0;
    base1[1]   = p1;
    base1[2]   = p2;
    int* base2 = triangles + (write + 1) * 3;
    base2[0]   = p1;
    base2[1]   = p3;
    base2[2]   = p2;
}

__device__ void add_face_normal(
    const float* __restrict__ points,
    int ia,
    int ib,
    int ic,
    float& nx,
    float& ny,
    float& nz) {
    const float ax = points[ia * 3 + 0];
    const float ay = points[ia * 3 + 1];
    const float az = points[ia * 3 + 2];
    const float bx = points[ib * 3 + 0];
    const float by = points[ib * 3 + 1];
    const float bz = points[ib * 3 + 2];
    const float cx = points[ic * 3 + 0];
    const float cy = points[ic * 3 + 1];
    const float cz = points[ic * 3 + 2];

    const float e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const float e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    nx += e1y * e2z - e1z * e2y;
    ny += e1z * e2x - e1x * e2z;
    nz += e1x * e2y - e1y * e2x;
}

__device__ bool quad_points(
    const int* __restrict__ occupancy,
    int n_az_bins,
    int r,
    int c,
    int& p0,
    int& p1,
    int& p2,
    int& p3) {
    p0 = occupancy[r * n_az_bins + c];
    p1 = occupancy[r * n_az_bins + (c + 1)];
    p2 = occupancy[(r + 1) * n_az_bins + c];
    p3 = occupancy[(r + 1) * n_az_bins + (c + 1)];
    return p0 != EMPTY_CELL && p1 != EMPTY_CELL && p2 != EMPTY_CELL && p3 != EMPTY_CELL;
}

__global__ void initialize_normals_kernel(
    float* __restrict__ vertex_normals, // (N, 3)
    int n_points) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_points)
        return;

    vertex_normals[idx * 3 + 0] = 0.f;
    vertex_normals[idx * 3 + 1] = 0.f;
    vertex_normals[idx * 3 + 2] = 1.f;
}

// One thread per occupied scan-grid cell computes that vertex's incident face
// normals from neighbouring quads. This replaces triangle-wise global
// atomics with register accumulation + one final write per vertex.
__global__ void compute_normals_from_occupancy_kernel(
    const float* __restrict__ points,   // (N, 3)
    const int* __restrict__ occupancy,  // (n_el_bins, n_az_bins) row-major
    float* __restrict__ vertex_normals, // (N, 3)
    int n_el_bins,
    int n_az_bins) {
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_el_bins || c >= n_az_bins)
        return;

    const int p = occupancy[r * n_az_bins + c];
    if (p == EMPTY_CELL)
        return;

    float nx = 0.f, ny = 0.f, nz = 0.f;
    int p0, p1, p2, p3;

    if (r < n_el_bins - 1 && c < n_az_bins - 1 &&
        quad_points(occupancy, n_az_bins, r, c, p0, p1, p2, p3)) {
        add_face_normal(points, p0, p1, p2, nx, ny, nz);
    }
    if (c > 0 && r < n_el_bins - 1 &&
        quad_points(occupancy, n_az_bins, r, c - 1, p0, p1, p2, p3)) {
        add_face_normal(points, p0, p1, p2, nx, ny, nz);
        add_face_normal(points, p1, p3, p2, nx, ny, nz);
    }
    if (r > 0 && c < n_az_bins - 1 &&
        quad_points(occupancy, n_az_bins, r - 1, c, p0, p1, p2, p3)) {
        add_face_normal(points, p0, p1, p2, nx, ny, nz);
        add_face_normal(points, p1, p3, p2, nx, ny, nz);
    }
    if (r > 0 && c > 0 &&
        quad_points(occupancy, n_az_bins, r - 1, c - 1, p0, p1, p2, p3)) {
        add_face_normal(points, p1, p3, p2, nx, ny, nz);
    }

    const float len2 = nx * nx + ny * ny + nz * nz;
    if (len2 < 1e-20f) {
        vertex_normals[p * 3 + 0] = 0.f;
        vertex_normals[p * 3 + 1] = 0.f;
        vertex_normals[p * 3 + 2] = 1.f;
        return;
    }

    const float inv_len = rsqrtf(len2);
    nx *= inv_len;
    ny *= inv_len;
    nz *= inv_len;

    const float px = points[p * 3 + 0];
    const float py = points[p * 3 + 1];
    const float pz = points[p * 3 + 2];
    const float d  = nx * px + ny * py + nz * pz;
    if (d > 0.f) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
    }

    vertex_normals[p * 3 + 0] = nx;
    vertex_normals[p * 3 + 1] = ny;
    vertex_normals[p * 3 + 2] = nz;
}

std::tuple<torch::Tensor, torch::Tensor> compute_mesh_normals_in_lidar_spin_cu(
    const torch::Tensor& points,
    const int n_elevation_bins,
    const int n_azimuth_bins,
    const double el_min,
    const double el_max) {
    at::cuda::CUDAGuard device_guard(points.device());
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
                "points must be (N, 3), got ", points.sizes());
    TORCH_CHECK(points.scalar_type() == torch::kFloat32, "points must be float32");
    TORCH_CHECK(n_elevation_bins >= 2 && n_azimuth_bins >= 2,
                "bin counts must each be >= 2, got ",
                n_elevation_bins, " x ", n_azimuth_bins);

    const auto device  = points.device();
    auto stream        = at::cuda::getCurrentCUDAStream();
    const int n_points = static_cast<int>(points.size(0));

    auto float_opts  = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto int_opts    = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto double_opts = torch::TensorOptions().dtype(torch::kFloat64).device(device);

    // Empty / under-3-point input: emit empty triangles + +z fallback normals.
    if (n_points < 3) {
        auto normals = torch::zeros({n_points, 3}, float_opts);
        if (n_points > 0) {
            auto fallback = torch::tensor({0.f, 0.f, 1.f}, float_opts);
            normals.copy_(fallback.unsqueeze(0).expand({n_points, 3}));
        }
        auto triangles = torch::empty({0, 3}, int_opts);
        return std::make_tuple(normals, triangles);
    }

    auto occupancy = torch::full({n_elevation_bins * n_azimuth_bins}, EMPTY_CELL, int_opts);

    const int threads        = 256;
    const int blocks_pt      = (n_points + threads - 1) / threads;
    const double az_span_inv = 1.0 / (2.0 * M_PI + 1e-9);

    if (std::isfinite(el_min) && std::isfinite(el_max)) {
        const double el_span_inv = 1.0 / std::max(el_max - el_min, 1e-9);
        bin_and_claim_cells_kernel<<<blocks_pt, threads, 0, stream>>>(
            points.data_ptr<float>(),
            occupancy.data_ptr<int>(),
            n_points,
            n_elevation_bins,
            n_azimuth_bins,
            el_min,
            el_span_inv,
            az_span_inv,
            n_elevation_bins - 1,
            n_azimuth_bins - 1);
    } else {
        auto azimuth   = torch::empty({n_points}, double_opts);
        auto elevation = torch::empty({n_points}, double_opts);
        auto az_bins   = torch::empty({n_points}, int_opts);

        compute_spherical_and_az_bin_kernel<<<blocks_pt, threads, 0, stream>>>(
            points.data_ptr<float>(),
            azimuth.data_ptr<double>(),
            elevation.data_ptr<double>(),
            az_bins.data_ptr<int>(),
            n_points,
            n_azimuth_bins,
            az_span_inv,
            n_azimuth_bins - 1);

        // Compatibility path when callers do not pass elevation bounds.
        // These scalar reductions synchronize with the host and serialize this
        // fallback. That is acceptable for its single-spin call sites.
        const double el_min_fallback      = elevation.min().item<double>();
        const double el_max_fallback      = elevation.max().item<double>();
        const double el_span_inv_fallback = 1.0 / std::max(el_max_fallback - el_min_fallback, 1e-9);

        claim_cells_kernel<<<blocks_pt, threads, 0, stream>>>(
            elevation.data_ptr<double>(),
            az_bins.data_ptr<int>(),
            occupancy.data_ptr<int>(),
            n_points,
            n_elevation_bins,
            n_azimuth_bins,
            el_min_fallback,
            el_span_inv_fallback,
            n_elevation_bins - 1);
    }

    // Allocate the geometric upper-bound triangle buffer; slice on actual count.
    const int max_triangles = 2 * (n_elevation_bins - 1) * (n_azimuth_bins - 1);
    auto triangles_full     = torch::empty({max_triangles, 3}, int_opts);
    auto out_count          = torch::zeros({1}, int_opts);

    {
        const int tx = 32, ty = 8;
        dim3 block(tx, ty);
        dim3 grid((n_azimuth_bins - 1 + tx - 1) / tx,
                  (n_elevation_bins - 1 + ty - 1) / ty);
        emit_triangles_kernel<<<grid, block, 0, stream>>>(
            occupancy.data_ptr<int>(),
            triangles_full.data_ptr<int>(),
            out_count.data_ptr<int>(),
            n_elevation_bins,
            n_azimuth_bins,
            max_triangles);
    }

    const int n_triangles = out_count.item<int>();
    auto triangles        = triangles_full.narrow(0, 0, n_triangles).contiguous();

    auto normals = torch::empty({n_points, 3}, float_opts);
    initialize_normals_kernel<<<blocks_pt, threads, 0, stream>>>(
        normals.data_ptr<float>(),
        n_points);
    {
        const int tx = 32, ty = 8;
        dim3 block(tx, ty);
        dim3 grid((n_azimuth_bins + tx - 1) / tx,
                  (n_elevation_bins + ty - 1) / ty);
        compute_normals_from_occupancy_kernel<<<grid, block, 0, stream>>>(
            points.data_ptr<float>(),
            occupancy.data_ptr<int>(),
            normals.data_ptr<float>(),
            n_elevation_bins,
            n_azimuth_bins);
    }

    return std::make_tuple(normals, triangles);
}
