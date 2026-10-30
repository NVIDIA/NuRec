// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <limits>

namespace {

// Portable one-dimensional CUDA grid limit; grid-stride kernels cover larger inputs.
inline constexpr int THREADS_PER_BLOCK       = 256;
inline constexpr int MAX_GRID_STRIDE_BLOCKS  = 65535;
inline constexpr float PI                    = 3.14159265358979323846f;
inline constexpr float HALF                  = 0.5f;
inline constexpr float ONE                   = 1.0f;
inline constexpr float NORM_EPSILON          = 1.0e-8f;
inline constexpr float BBOX_PAD_SCALE        = 1.0e-4f;
inline constexpr float RAD_TO_DEG            = 180.0f / PI;
inline constexpr float DEG_TO_RAD            = PI / 180.0f;
inline constexpr int PACK_INFO_COLUMNS       = 6;
inline constexpr int POINT_START_COLUMN      = 0;
inline constexpr int POINT_COUNT_COLUMN      = 1;
inline constexpr int HYPOTHESIS_START_COLUMN = 2;
inline constexpr int HYPOTHESIS_COUNT_COLUMN = 3;
inline constexpr int EVAL_START_COLUMN       = 4;
inline constexpr int EVAL_COUNT_COLUMN       = 5;

// The block size is a compile-time parameter, so the scoring kernels that reduce with this type
// must be launched with exactly THREADS_PER_BLOCK threads per block.
using BlockReduce = cub::BlockReduce<float, THREADS_PER_BLOCK>;

constexpr int blocks_for(const int64_t n) {
    return static_cast<int>((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
}

constexpr int grid_stride_blocks_for(const int64_t n) {
    const int64_t blocks = n / THREADS_PER_BLOCK + static_cast<int64_t>(n % THREADS_PER_BLOCK != 0);
    return static_cast<int>(blocks < MAX_GRID_STRIDE_BLOCKS ? blocks : MAX_GRID_STRIDE_BLOCKS);
}

void check_same_device(const torch::Tensor& primary, const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device() == primary.device(), name, " must be on the same CUDA device as the primary tensor");
}

__global__ void accumulate_grid_kernel(
    const float* __restrict__ points,
    const uint8_t* __restrict__ colors,
    float* __restrict__ z_sums,
    float* __restrict__ color_sums,
    int* __restrict__ counts,
    int64_t n_points,
    int width,
    int height,
    float min_x,
    float min_y,
    float inv_cell_size) {
    const int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_points) {
        return;
    }

    const float x = points[idx * 3 + 0];
    const float y = points[idx * 3 + 1];
    const float z = points[idx * 3 + 2];

    int cx = static_cast<int>(floorf((x - min_x) * inv_cell_size));
    int cy = static_cast<int>(floorf((y - min_y) * inv_cell_size));
    cx     = max(0, min(width - 1, cx));
    cy     = max(0, min(height - 1, cy));

    const int cell = cy * width + cx;
    atomicAdd(&counts[cell], 1);
    atomicAdd(&z_sums[cell], z);

    if (colors != nullptr) {
        atomicAdd(&color_sums[cell * 3 + 0], static_cast<float>(colors[idx * 3 + 0]));
        atomicAdd(&color_sums[cell * 3 + 1], static_cast<float>(colors[idx * 3 + 1]));
        atomicAdd(&color_sums[cell * 3 + 2], static_cast<float>(colors[idx * 3 + 2]));
    }
}

__global__ void build_vertices_kernel(
    const int64_t* __restrict__ occupied_ids,
    const int* __restrict__ counts,
    const float* __restrict__ z_sums,
    const float* __restrict__ color_sums,
    float* __restrict__ vertices,
    uint8_t* __restrict__ vertex_colors,
    int* __restrict__ cell_to_vertex,
    int64_t n_vertices,
    int width,
    float min_x,
    float min_y,
    float cell_size) {
    const int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_vertices) {
        return;
    }

    const int cell  = static_cast<int>(occupied_ids[idx]);
    const int count = max(counts[cell], 1);
    const int cx    = cell % width;
    const int cy    = cell / width;

    vertices[idx * 3 + 0] = cell_size * (static_cast<float>(cx) + HALF) + min_x;
    vertices[idx * 3 + 1] = cell_size * (static_cast<float>(cy) + HALF) + min_y;
    vertices[idx * 3 + 2] = z_sums[cell] / static_cast<float>(count);
    cell_to_vertex[cell]  = static_cast<int>(idx);

    if (vertex_colors != nullptr) {
        vertex_colors[idx * 3 + 0] = static_cast<uint8_t>(color_sums[cell * 3 + 0] / static_cast<float>(count));
        vertex_colors[idx * 3 + 1] = static_cast<uint8_t>(color_sums[cell * 3 + 1] / static_cast<float>(count));
        vertex_colors[idx * 3 + 2] = static_cast<uint8_t>(color_sums[cell * 3 + 2] / static_cast<float>(count));
    }
}

__global__ void mark_horizontal_brackets_kernel(
    const int* __restrict__ counts,
    const float* __restrict__ z_sums,
    const float* __restrict__ color_sums,
    uint8_t* __restrict__ horizontal_bracket,
    float* __restrict__ fill_z_sums,
    float* __restrict__ fill_color_sums,
    int* __restrict__ fill_weight_sums,
    int width,
    int height,
    int min_points_per_cell) {
    const int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= height) {
        return;
    }

    const int row_base = row * width;
    bool left_seen     = false;
    float left_z       = 0.0f;
    float left_color0  = 0.0f;
    float left_color1  = 0.0f;
    float left_color2  = 0.0f;
    for (int x = 0; x < width; ++x) {
        const int cell  = row_base + x;
        const int count = counts[cell];
        if (count >= min_points_per_cell) {
            const float inv_count = ONE / static_cast<float>(count);
            left_seen             = true;
            left_z                = z_sums[cell] * inv_count;
            if (fill_color_sums != nullptr) {
                left_color0 = color_sums[cell * 3 + 0] * inv_count;
                left_color1 = color_sums[cell * 3 + 1] * inv_count;
                left_color2 = color_sums[cell * 3 + 2] * inv_count;
            }
            horizontal_bracket[cell] = 0;
        } else if (left_seen) {
            horizontal_bracket[cell] = 1;
            fill_z_sums[cell]        = left_z;
            fill_weight_sums[cell]   = 1;
            if (fill_color_sums != nullptr) {
                fill_color_sums[cell * 3 + 0] = left_color0;
                fill_color_sums[cell * 3 + 1] = left_color1;
                fill_color_sums[cell * 3 + 2] = left_color2;
            }
        } else {
            horizontal_bracket[cell] = 0;
        }
    }

    bool right_seen    = false;
    float right_z      = 0.0f;
    float right_color0 = 0.0f;
    float right_color1 = 0.0f;
    float right_color2 = 0.0f;
    for (int x = width - 1; x >= 0; --x) {
        const int cell  = row_base + x;
        const int count = counts[cell];
        if (count >= min_points_per_cell) {
            const float inv_count = ONE / static_cast<float>(count);
            right_seen            = true;
            right_z               = z_sums[cell] * inv_count;
            if (fill_color_sums != nullptr) {
                right_color0 = color_sums[cell * 3 + 0] * inv_count;
                right_color1 = color_sums[cell * 3 + 1] * inv_count;
                right_color2 = color_sums[cell * 3 + 2] * inv_count;
            }
        } else if (horizontal_bracket[cell] != 0 && right_seen) {
            fill_z_sums[cell] += right_z;
            fill_weight_sums[cell] += 1;
            if (fill_color_sums != nullptr) {
                fill_color_sums[cell * 3 + 0] += right_color0;
                fill_color_sums[cell * 3 + 1] += right_color1;
                fill_color_sums[cell * 3 + 2] += right_color2;
            }
        } else {
            horizontal_bracket[cell] = 0;
        }
    }
}

__global__ void mark_vertical_brackets_kernel(
    const int* __restrict__ counts,
    const float* __restrict__ z_sums,
    const float* __restrict__ color_sums,
    const uint8_t* __restrict__ horizontal_bracket,
    uint8_t* __restrict__ vertical_bracket,
    uint8_t* __restrict__ fillable,
    float* __restrict__ fill_z_sums,
    float* __restrict__ fill_color_sums,
    int* __restrict__ fill_weight_sums,
    int width,
    int height,
    int min_points_per_cell,
    int hole_fill_min_neighbors) {
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col >= width) {
        return;
    }

    bool up_seen    = false;
    float up_z      = 0.0f;
    float up_color0 = 0.0f;
    float up_color1 = 0.0f;
    float up_color2 = 0.0f;
    for (int y = 0; y < height; ++y) {
        const int cell  = y * width + col;
        const int count = counts[cell];
        if (count >= min_points_per_cell) {
            const float inv_count = ONE / static_cast<float>(count);
            up_seen               = true;
            up_z                  = z_sums[cell] * inv_count;
            if (fill_color_sums != nullptr) {
                up_color0 = color_sums[cell * 3 + 0] * inv_count;
                up_color1 = color_sums[cell * 3 + 1] * inv_count;
                up_color2 = color_sums[cell * 3 + 2] * inv_count;
            }
            vertical_bracket[cell] = 0;
            continue;
        }

        if (hole_fill_min_neighbors > 0) {
            int neighbors = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                const int ny = y + dy;
                if (ny < 0 || ny >= height) {
                    continue;
                }
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const int nx = col + dx;
                    if (nx < 0 || nx >= width) {
                        continue;
                    }
                    neighbors += counts[ny * width + nx] >= min_points_per_cell ? 1 : 0;
                }
            }
            if (neighbors >= hole_fill_min_neighbors) {
                fillable[cell] = 1;
            }
        }

        if (horizontal_bracket[cell] != 0 && up_seen) {
            vertical_bracket[cell] = 1;
            fill_z_sums[cell] += up_z;
            fill_weight_sums[cell] += 1;
            if (fill_color_sums != nullptr) {
                fill_color_sums[cell * 3 + 0] += up_color0;
                fill_color_sums[cell * 3 + 1] += up_color1;
                fill_color_sums[cell * 3 + 2] += up_color2;
            }
        } else {
            vertical_bracket[cell] = 0;
        }
    }

    bool down_seen    = false;
    float down_z      = 0.0f;
    float down_color0 = 0.0f;
    float down_color1 = 0.0f;
    float down_color2 = 0.0f;
    for (int y = height - 1; y >= 0; --y) {
        const int cell  = y * width + col;
        const int count = counts[cell];
        if (count >= min_points_per_cell) {
            const float inv_count = ONE / static_cast<float>(count);
            down_seen             = true;
            down_z                = z_sums[cell] * inv_count;
            if (fill_color_sums != nullptr) {
                down_color0 = color_sums[cell * 3 + 0] * inv_count;
                down_color1 = color_sums[cell * 3 + 1] * inv_count;
                down_color2 = color_sums[cell * 3 + 2] * inv_count;
            }
        } else if (vertical_bracket[cell] != 0 && down_seen) {
            fill_z_sums[cell] += down_z;
            fill_weight_sums[cell] += 1;
            if (fill_color_sums != nullptr) {
                fill_color_sums[cell * 3 + 0] += down_color0;
                fill_color_sums[cell * 3 + 1] += down_color1;
                fill_color_sums[cell * 3 + 2] += down_color2;
            }
            fillable[cell] = 1;
        }
    }
}

__global__ void build_filled_vertices_kernel(
    const int64_t* __restrict__ filled_ids,
    const int* __restrict__ counts,
    const float* __restrict__ z_sums,
    const float* __restrict__ color_sums,
    const float* __restrict__ fill_z_sums,
    const float* __restrict__ fill_color_sums,
    const int* __restrict__ fill_weight_sums,
    float* __restrict__ vertices,
    uint8_t* __restrict__ vertex_colors,
    int* __restrict__ cell_to_vertex,
    int64_t n_filled,
    int64_t first_filled_vertex,
    int width,
    int height,
    int min_points_per_cell,
    float min_x,
    float min_y,
    float cell_size) {
    const int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_filled) {
        return;
    }

    const int cell = static_cast<int>(filled_ids[idx]);
    const int cx   = cell % width;
    const int cy   = cell / width;

    float z                   = 0.0f;
    float color0              = 0.0f;
    float color1              = 0.0f;
    float color2              = 0.0f;
    const int fill_weight_sum = fill_weight_sums != nullptr ? fill_weight_sums[cell] : 0;
    if (fill_weight_sum > 0) {
        const float inv_weight_sum = ONE / static_cast<float>(fill_weight_sum);
        z                          = fill_z_sums[cell] * inv_weight_sum;
        if (vertex_colors != nullptr) {
            color0 = fill_color_sums[cell * 3 + 0] * inv_weight_sum;
            color1 = fill_color_sums[cell * 3 + 1] * inv_weight_sum;
            color2 = fill_color_sums[cell * 3 + 2] * inv_weight_sum;
        }
    } else {
        float z_sum      = 0.0f;
        float color_sum0 = 0.0f;
        float color_sum1 = 0.0f;
        float color_sum2 = 0.0f;
        int neighbors    = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            const int ny = cy + dy;
            if (ny < 0 || ny >= height) {
                continue;
            }
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                const int nx = cx + dx;
                if (nx < 0 || nx >= width) {
                    continue;
                }
                const int neighbor = ny * width + nx;
                const int count    = counts[neighbor];
                if (count < min_points_per_cell) {
                    continue;
                }
                const float inv_count = ONE / static_cast<float>(count);
                z_sum += z_sums[neighbor] * inv_count;
                if (vertex_colors != nullptr) {
                    color_sum0 += color_sums[neighbor * 3 + 0] * inv_count;
                    color_sum1 += color_sums[neighbor * 3 + 1] * inv_count;
                    color_sum2 += color_sums[neighbor * 3 + 2] * inv_count;
                }
                ++neighbors;
            }
        }
        if (neighbors == 0) {
            return;
        }

        const float inv_neighbors = ONE / static_cast<float>(neighbors);
        z                         = z_sum * inv_neighbors;
        if (vertex_colors != nullptr) {
            color0 = color_sum0 * inv_neighbors;
            color1 = color_sum1 * inv_neighbors;
            color2 = color_sum2 * inv_neighbors;
        }
    }

    const int64_t vertex_idx     = first_filled_vertex + idx;
    vertices[vertex_idx * 3 + 0] = cell_size * (static_cast<float>(cx) + HALF) + min_x;
    vertices[vertex_idx * 3 + 1] = cell_size * (static_cast<float>(cy) + HALF) + min_y;
    vertices[vertex_idx * 3 + 2] = z;
    cell_to_vertex[cell]         = static_cast<int>(vertex_idx);

    if (vertex_colors != nullptr) {
        vertex_colors[vertex_idx * 3 + 0] = static_cast<uint8_t>(color0);
        vertex_colors[vertex_idx * 3 + 1] = static_cast<uint8_t>(color1);
        vertex_colors[vertex_idx * 3 + 2] = static_cast<uint8_t>(color2);
    }
}

// Marking the referenced vertices here rather than in a separate pass is safe because every index
// handed to this helper is a present vertex, and concurrent byte writes of the same value need no
// atomics.
__device__ inline void write_triangle(int* triangles, uint8_t* referenced, int tri_idx, int a, int b, int c) {
    triangles[tri_idx * 3 + 0] = a;
    triangles[tri_idx * 3 + 1] = b;
    triangles[tri_idx * 3 + 2] = c;
    referenced[a]              = 1;
    referenced[b]              = 1;
    referenced[c]              = 1;
}

__global__ void build_triangles_kernel(
    const int* __restrict__ cell_to_vertex,
    int* __restrict__ triangles,
    int* __restrict__ triangle_count,
    uint8_t* __restrict__ referenced,
    int64_t n_2x2_blocks,
    int width,
    int height) {
    const int64_t block_2x2_idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (block_2x2_idx >= n_2x2_blocks || width < 2 || height < 2) {
        return;
    }

    const int blocks_2x2_per_row = width - 1;
    const int cx                 = static_cast<int>(block_2x2_idx % blocks_2x2_per_row);
    const int cy                 = static_cast<int>(block_2x2_idx / blocks_2x2_per_row);
    if (cy >= height - 1) {
        return;
    }

    const int base = cy * width + cx;
    const int v00  = cell_to_vertex[base];
    const int v10  = cell_to_vertex[base + 1];
    const int v01  = cell_to_vertex[base + width];
    const int v11  = cell_to_vertex[base + width + 1];

    const bool p00    = v00 >= 0;
    const bool p10    = v10 >= 0;
    const bool p01    = v01 >= 0;
    const bool p11    = v11 >= 0;
    const int present = static_cast<int>(p00) + static_cast<int>(p10) + static_cast<int>(p01) + static_cast<int>(p11);

    if (present == 4) {
        const int out = atomicAdd(triangle_count, 2);
        write_triangle(triangles, referenced, out, v00, v10, v11);
        write_triangle(triangles, referenced, out + 1, v00, v11, v01);
    } else if (present == 3) {
        const int out = atomicAdd(triangle_count, 1);
        if (!p00) {
            write_triangle(triangles, referenced, out, v10, v11, v01);
        } else if (!p10) {
            write_triangle(triangles, referenced, out, v00, v11, v01);
        } else if (!p01) {
            write_triangle(triangles, referenced, out, v00, v10, v11);
        } else {
            write_triangle(triangles, referenced, out, v00, v10, v01);
        }
    }
}

__global__ void build_old_to_new_kernel(
    const int64_t* __restrict__ referenced_ids,
    int* __restrict__ old_to_new,
    int64_t n_referenced) {
    const int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_referenced) {
        return;
    }
    old_to_new[referenced_ids[idx]] = static_cast<int>(idx);
}

__global__ void remap_triangles_kernel(
    const int* __restrict__ old_to_new,
    int* __restrict__ triangles,
    int64_t n_indices) {
    const int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_indices) {
        return;
    }
    triangles[idx] = old_to_new[triangles[idx]];
}

// Rows up to this degree use transposed storage for adjacent vertex threads.
inline constexpr int SHORT_NEIGHBOR_CAPACITY = 8;
// Bound insertion sort's quadratic work independently of the storage layout.
inline constexpr int INSERTION_SORT_MAX_DEGREE = 32;
// Longer reductions accumulate in double to limit floating-point roundoff.
inline constexpr int FLOAT_SUM_MAX_DEGREE = 32;

struct SmoothingRowLayout {
    int64_t start;
    int64_t stride;

    __device__ __forceinline__ int64_t position(int64_t slot) const {
        return start + slot * stride;
    }
};

__device__ __forceinline__ SmoothingRowLayout smoothing_row_layout(
    int64_t vertex, int64_t begin, int64_t degree, int64_t n_vertices) {
    if (degree <= SHORT_NEIGHBOR_CAPACITY) {
        // Adjacent vertex threads access adjacent entries at the same neighbor slot.
        return {vertex, n_vertices};
    }
    // Long rows occupy contiguous spans after the transposed region.
    return {SHORT_NEIGHBOR_CAPACITY * n_vertices + begin, 1};
}

template <typename Index>
__device__ Index atomic_neighbor_count(Index* value, int delta) {
    if constexpr (sizeof(Index) == sizeof(int)) {
        return atomicAdd(value, delta);
    } else {
        // CUDA only exposes unsigned 64-bit atomicAdd; two's-complement addition
        // preserves the signed increment or decrement bit pattern used here.
        return static_cast<Index>(atomicAdd(reinterpret_cast<unsigned long long*>(value),
                                            static_cast<unsigned long long>(delta)));
    }
}

template <typename Index>
__global__ void count_smoothing_neighbors_kernel(
    const int64_t* __restrict__ edges,
    Index* __restrict__ counts,
    int64_t n_edges,
    int64_t n_vertices) {
    const int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_edges) {
        return;
    }

    const int64_t i = edges[idx * 2 + 0];
    const int64_t j = edges[idx * 2 + 1];
    if (i < 0 || i >= n_vertices || j < 0 || j >= n_vertices) {
        return;
    }
    // Shift degrees by one element so the inclusive scan yields each row's start.
    atomic_neighbor_count(counts + i + 1, 1);
    atomic_neighbor_count(counts + j + 1, 1);
}

template <typename Index>
__device__ __forceinline__ void insert_smoothing_neighbor(
    int64_t vertex, int64_t neighbor, const Index* offsets,
    Index* counts, Index* neighbors, int64_t n_vertices) {
    // Atomic countdowns assign each incident edge a unique slot within its row.
    const Index remaining                       = atomic_neighbor_count(counts + vertex + 1, -1);
    const Index degree                          = offsets[vertex + 1] - offsets[vertex];
    const auto row                              = smoothing_row_layout(vertex, offsets[vertex], degree, n_vertices);
    neighbors[row.position(degree - remaining)] = static_cast<Index>(neighbor);
}

template <typename Index>
__global__ void fill_smoothing_neighbors_kernel(
    const int64_t* __restrict__ edges,
    const Index* __restrict__ offsets,
    Index* __restrict__ counts,
    Index* __restrict__ neighbors,
    int64_t n_edges,
    int64_t n_vertices) {
    const int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_edges) {
        return;
    }

    const int64_t i = edges[idx * 2 + 0];
    const int64_t j = edges[idx * 2 + 1];
    if (i < 0 || i >= n_vertices || j < 0 || j >= n_vertices) {
        return;
    }
    // Each undirected edge contributes twice, including self-edges.
    insert_smoothing_neighbor(i, j, offsets, counts, neighbors, n_vertices);
    insert_smoothing_neighbor(j, i, offsets, counts, neighbors, n_vertices);
}

template <typename Index>
__device__ void sift_neighbor_heap(Index* row, Index root, Index size) {
    while (root < size / 2) {
        Index child = root * 2 + 1;
        if (child + 1 < size && row[child] < row[child + 1]) {
            ++child;
        }
        if (row[root] >= row[child]) {
            break;
        }
        const Index value = row[root];
        row[root]         = row[child];
        row[child]        = value;
        root              = child;
    }
}

template <typename Index>
__device__ __forceinline__ void sort_smoothing_neighbors(Index* row, Index size) {
    if (size <= INSERTION_SORT_MAX_DEGREE) {
        // Insertion sort avoids heap bookkeeping for small neighborhoods.
        for (Index k = 1; k < size; ++k) {
            const Index value = row[k];
            Index position    = k;
            while (position > 0 && row[position - 1] > value) {
                row[position] = row[position - 1];
                --position;
            }
            row[position] = value;
        }
    } else {
        // Heap sort bounds high-degree row work to O(degree log degree).
        for (Index k = size / 2; k > 0; --k) {
            sift_neighbor_heap<Index>(row, k - 1, size);
        }
        for (Index end = size - 1; end > 0; --end) {
            const Index value = row[0];
            row[0]            = row[end];
            row[end]          = value;
            sift_neighbor_heap<Index>(row, 0, end);
        }
    }
}

template <typename Index>
__global__ void prepare_smoothing_rows_kernel(
    const float* __restrict__ vertices,
    const Index* __restrict__ offsets,
    Index* __restrict__ neighbors,
    float* __restrict__ weights,
    float* __restrict__ weight_sums,
    float* __restrict__ heights,
    float* __restrict__ out,
    int64_t n_vertices,
    int axis) {
    const int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_vertices) {
        return;
    }
    const Index begin = offsets[idx];
    const Index size  = offsets[idx + 1] - begin;
    const auto layout = smoothing_row_layout(idx, begin, size, n_vertices);
    Index* row        = neighbors + layout.start;
    Index short_row[SHORT_NEIGHBOR_CAPACITY];
    if (size <= SHORT_NEIGHBOR_CAPACITY) {
        // Copy the transposed row into a contiguous per-thread array for sorting.
        for (Index k = 0; k < size; ++k) {
            short_row[k] = neighbors[layout.position(k)];
        }
        row = short_row;
    }
    // Sorting by neighbor index fixes the floating-point reduction order.
    sort_smoothing_neighbors(row, size);

    // Compute weights once from the input geometry, in sorted neighbor order.
    float sum          = 0.0f;
    double precise_sum = 0.0;
    for (Index k = 0; k < size; ++k) {
        const Index neighbor   = row[k];
        const int64_t position = layout.position(k);
        if (size <= SHORT_NEIGHBOR_CAPACITY) {
            neighbors[position] = neighbor;
        }
        const float dx     = vertices[idx * 3] - vertices[neighbor * 3];
        const float dy     = vertices[idx * 3 + 1] - vertices[neighbor * 3 + 1];
        const float dz     = vertices[idx * 3 + 2] - vertices[neighbor * 3 + 2];
        const float weight = ONE / fmaxf(sqrtf(dx * dx + dy * dy + dz * dz), NORM_EPSILON);
        weights[position]  = weight;
        if (size > FLOAT_SUM_MAX_DEGREE) {
            // Double accumulation limits roundoff in long rows before float storage.
            precise_sum += weight;
        } else {
            sum += weight;
        }
    }
    weight_sums[idx] = size > FLOAT_SUM_MAX_DEGREE ? static_cast<float>(precise_sum) : sum;
    heights[idx]     = vertices[idx * 3 + axis];
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        out[idx * 3 + coordinate] = vertices[idx * 3 + coordinate];
    }
}

template <typename Accumulator, typename Index>
__device__ __forceinline__ float weighted_neighbor_sum(
    SmoothingRowLayout layout, Index degree, const Index* neighbors,
    const float* weights, const float* heights) {
    Accumulator sum = 0;
    // Serial traversal preserves the sorted neighbor reduction order.
    for (Index k = 0; k < degree; ++k) {
        const int64_t position = layout.position(k);
        // Multiply in float before widening, preserving the product's rounding.
        sum += static_cast<Accumulator>(weights[position] * heights[neighbors[position]]);
    }
    return static_cast<float>(sum);
}

template <typename Index>
__global__ void smooth_neighbor_rows_kernel(
    const Index* __restrict__ offsets,
    const Index* __restrict__ neighbors,
    const float* __restrict__ weights,
    const float* __restrict__ weight_sums,
    const float* __restrict__ heights,
    float* __restrict__ next_heights,
    int64_t n_vertices,
    float smoothness,
    int output_stride) {
    const int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_vertices) {
        return;
    }

    float value = heights[idx];
    if (weight_sums[idx] > 0.0f) {
        const Index begin  = offsets[idx];
        const Index end    = offsets[idx + 1];
        const Index degree = end - begin;
        const auto layout  = smoothing_row_layout(idx, begin, degree, n_vertices);
        const float sum    = degree > FLOAT_SUM_MAX_DEGREE
                                 ? weighted_neighbor_sum<double>(layout, degree, neighbors, weights, heights)
                                 : weighted_neighbor_sum<float>(layout, degree, neighbors, weights, heights);
        value              = (sum / weight_sums[idx]) * smoothness + value * (ONE - smoothness);
    }
    next_heights[idx * output_stride] = value;
}

__global__ void accumulate_face_normals_kernel(
    const float* __restrict__ vertices,
    const int* __restrict__ triangles,
    float* __restrict__ vertex_normals,
    int64_t n_triangles,
    int64_t n_vertices) {
    const int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_triangles) {
        return;
    }

    const int i0 = triangles[idx * 3 + 0];
    const int i1 = triangles[idx * 3 + 1];
    const int i2 = triangles[idx * 3 + 2];
    if (i0 < 0 || i0 >= n_vertices || i1 < 0 || i1 >= n_vertices || i2 < 0 || i2 >= n_vertices) {
        return;
    }
    const float ax   = vertices[i1 * 3 + 0] - vertices[i0 * 3 + 0];
    const float ay   = vertices[i1 * 3 + 1] - vertices[i0 * 3 + 1];
    const float az   = vertices[i1 * 3 + 2] - vertices[i0 * 3 + 2];
    const float bx   = vertices[i2 * 3 + 0] - vertices[i0 * 3 + 0];
    const float by   = vertices[i2 * 3 + 1] - vertices[i0 * 3 + 1];
    const float bz   = vertices[i2 * 3 + 2] - vertices[i0 * 3 + 2];
    float nx         = ay * bz - az * by;
    float ny         = az * bx - ax * bz;
    float nz         = ax * by - ay * bx;
    const float norm = fmaxf(sqrtf(nx * nx + ny * ny + nz * nz), NORM_EPSILON);
    nx /= norm;
    ny /= norm;
    nz /= norm;

    atomicAdd(&vertex_normals[i0 * 3 + 0], nx);
    atomicAdd(&vertex_normals[i0 * 3 + 1], ny);
    atomicAdd(&vertex_normals[i0 * 3 + 2], nz);
    atomicAdd(&vertex_normals[i1 * 3 + 0], nx);
    atomicAdd(&vertex_normals[i1 * 3 + 1], ny);
    atomicAdd(&vertex_normals[i1 * 3 + 2], nz);
    atomicAdd(&vertex_normals[i2 * 3 + 0], nx);
    atomicAdd(&vertex_normals[i2 * 3 + 1], ny);
    atomicAdd(&vertex_normals[i2 * 3 + 2], nz);
}

__global__ void normalize_normals_kernel(float* __restrict__ normals, int64_t n_vertices) {
    const int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_vertices) {
        return;
    }
    float nx             = normals[idx * 3 + 0];
    float ny             = normals[idx * 3 + 1];
    float nz             = normals[idx * 3 + 2];
    const float norm     = fmaxf(sqrtf(nx * nx + ny * ny + nz * nz), NORM_EPSILON);
    normals[idx * 3 + 0] = nx / norm;
    normals[idx * 3 + 1] = ny / norm;
    normals[idx * 3 + 2] = nz / norm;
}

__global__ void ground_compatible_kernel(
    const float* __restrict__ points,
    const float* __restrict__ normals,
    const float* __restrict__ ground_control_point,
    const float* __restrict__ ref_normal,
    bool* __restrict__ out,
    int64_t n_points,
    float max_distance,
    float min_cos_angle) {
    const int64_t first_idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    const int64_t stride    = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t idx = first_idx; idx < n_points; idx += stride) {
        const float nx = normals[idx * 3 + 0];
        const float ny = normals[idx * 3 + 1];
        const float nz = normals[idx * 3 + 2];
        const float px = points[idx * 3 + 0];
        const float py = points[idx * 3 + 1];
        const float pz = points[idx * 3 + 2];

        const float plane_offset = -(nx * px + ny * py + nz * pz);
        const float distance     = fabsf(
            nx * ground_control_point[0] + ny * ground_control_point[1] + nz * ground_control_point[2] + plane_offset);
        const float cos_angle = nx * ref_normal[0] + ny * ref_normal[1] + nz * ref_normal[2];
        out[idx]              = distance <= max_distance && cos_angle > min_cos_angle;
    }
}

__device__ __forceinline__ float clamp_unit(const float value) {
    return fminf(fmaxf(value, -ONE), ONE);
}

__device__ __forceinline__ float hann_weight(const float value, const float threshold) {
    const float x = clamp_unit(value / threshold);
    return HALF + HALF * cosf(PI * x);
}

__global__ void build_plane_hypotheses_kernel(
    const float* __restrict__ points,
    const float* __restrict__ normals,
    const int64_t* __restrict__ hypothesis_indices,
    float* __restrict__ plane_normals,
    float* __restrict__ plane_offsets,
    float* __restrict__ normal_lengths,
    int64_t n_hypotheses,
    int64_t n_points) {
    const int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_hypotheses) {
        return;
    }

    const int64_t point_idx = hypothesis_indices[idx];
    if (point_idx < 0 || point_idx >= n_points) {
        plane_normals[idx * 3 + 0] = 0.0f;
        plane_normals[idx * 3 + 1] = 0.0f;
        plane_normals[idx * 3 + 2] = 0.0f;
        plane_offsets[idx]         = 0.0f;
        normal_lengths[idx]        = NORM_EPSILON;
        return;
    }
    const float nx             = normals[point_idx * 3 + 0];
    const float ny             = normals[point_idx * 3 + 1];
    const float nz             = normals[point_idx * 3 + 2];
    const float px             = points[point_idx * 3 + 0];
    const float py             = points[point_idx * 3 + 1];
    const float pz             = points[point_idx * 3 + 2];
    plane_normals[idx * 3 + 0] = nx;
    plane_normals[idx * 3 + 1] = ny;
    plane_normals[idx * 3 + 2] = nz;
    plane_offsets[idx]         = -(nx * px + ny * py + nz * pz);
    normal_lengths[idx]        = fmaxf(sqrtf(nx * nx + ny * ny + nz * nz), NORM_EPSILON);
}

__global__ void score_plane_hypotheses_kernel(
    const float* __restrict__ points,
    const float* __restrict__ normals,
    const int64_t* __restrict__ eval_indices,
    const float* __restrict__ plane_normals,
    const float* __restrict__ plane_offsets,
    const float* __restrict__ normal_lengths,
    float* __restrict__ scores,
    int64_t n_eval_points,
    int64_t n_points,
    float distance_threshold,
    float angle_threshold_deg) {
    const int64_t hypothesis_idx  = blockIdx.x;
    const float nx                = plane_normals[hypothesis_idx * 3 + 0];
    const float ny                = plane_normals[hypothesis_idx * 3 + 1];
    const float nz                = plane_normals[hypothesis_idx * 3 + 2];
    const float offset            = plane_offsets[hypothesis_idx];
    const float normal_length     = normal_lengths[hypothesis_idx];
    const float inv_normal_length = ONE / normal_length;

    float partial_score = 0.0f;
    for (int64_t eval_idx = threadIdx.x; eval_idx < n_eval_points; eval_idx += blockDim.x) {
        const int64_t point_idx = eval_indices[eval_idx];
        if (point_idx < 0 || point_idx >= n_points) {
            continue;
        }
        const float px        = points[point_idx * 3 + 0];
        const float py        = points[point_idx * 3 + 1];
        const float pz        = points[point_idx * 3 + 2];
        const float pnx       = normals[point_idx * 3 + 0];
        const float pny       = normals[point_idx * 3 + 1];
        const float pnz       = normals[point_idx * 3 + 2];
        const float distance  = fabsf(nx * px + ny * py + nz * pz + offset) * inv_normal_length;
        const float dot       = clamp_unit((pnx * nx + pny * ny + pnz * nz) * inv_normal_length);
        const float angle_deg = acosf(dot) * RAD_TO_DEG;
        partial_score += hann_weight(distance, distance_threshold) * hann_weight(angle_deg, angle_threshold_deg);
    }

    __shared__ typename BlockReduce::TempStorage temp_storage;
    const float block_score = BlockReduce(temp_storage).Sum(partial_score);
    if (threadIdx.x == 0) {
        scores[hypothesis_idx] = block_score;
    }
}

__global__ void select_best_plane_kernel(
    const float* __restrict__ plane_normals,
    const float* __restrict__ plane_offsets,
    const float* __restrict__ scores,
    float* __restrict__ best_normal,
    float* __restrict__ best_offset,
    int64_t n_hypotheses) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int64_t best_idx = 0;
    float best_score = scores[0];
    for (int64_t idx = 1; idx < n_hypotheses; ++idx) {
        const float score = scores[idx];
        if (score > best_score) {
            best_score = score;
            best_idx   = idx;
        }
    }
    best_normal[0] = plane_normals[best_idx * 3 + 0];
    best_normal[1] = plane_normals[best_idx * 3 + 1];
    best_normal[2] = plane_normals[best_idx * 3 + 2];
    best_offset[0] = plane_offsets[best_idx];
}

__global__ void mark_plane_inliers_kernel(
    const float* __restrict__ points,
    const float* __restrict__ normals,
    const float* __restrict__ best_normal,
    const float* __restrict__ best_offset,
    bool* __restrict__ inlier_mask,
    int64_t n_points,
    float distance_threshold,
    float angle_threshold_deg) {
    const int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_points) {
        return;
    }

    const float nx                = best_normal[0];
    const float ny                = best_normal[1];
    const float nz                = best_normal[2];
    const float inv_normal_length = ONE / fmaxf(sqrtf(nx * nx + ny * ny + nz * nz), NORM_EPSILON);
    const float px                = points[idx * 3 + 0];
    const float py                = points[idx * 3 + 1];
    const float pz                = points[idx * 3 + 2];
    const float pnx               = normals[idx * 3 + 0];
    const float pny               = normals[idx * 3 + 1];
    const float pnz               = normals[idx * 3 + 2];
    const float distance          = fabsf(nx * px + ny * py + nz * pz + best_offset[0]) * inv_normal_length;
    const float dot               = clamp_unit((pnx * nx + pny * ny + pnz * nz) * inv_normal_length);
    const float angle_deg         = acosf(dot) * RAD_TO_DEG;
    inlier_mask[idx]              = distance <= distance_threshold && angle_deg <= angle_threshold_deg;
}

__device__ __forceinline__ int64_t find_pack_for_index(
    const int64_t* __restrict__ pack_info,
    const int64_t n_packs,
    const int start_column,
    const int count_column,
    const int64_t index) {
    for (int64_t pack_idx = 0; pack_idx < n_packs; ++pack_idx) {
        const int64_t start = pack_info[pack_idx * PACK_INFO_COLUMNS + start_column];
        const int64_t count = pack_info[pack_idx * PACK_INFO_COLUMNS + count_column];
        if (index >= start && index < start + count) {
            return pack_idx;
        }
    }
    return -1;
}

__global__ void score_plane_hypotheses_batched_kernel(
    const float* __restrict__ points,
    const float* __restrict__ normals,
    const int64_t* __restrict__ eval_indices,
    const int64_t* __restrict__ pack_info,
    const float* __restrict__ plane_normals,
    const float* __restrict__ plane_offsets,
    const float* __restrict__ normal_lengths,
    float* __restrict__ scores,
    int64_t n_packs,
    int64_t n_points,
    float distance_threshold,
    float angle_threshold_deg) {
    const int64_t hypothesis_idx = blockIdx.x;
    const int64_t pack_idx       = find_pack_for_index(
        pack_info, n_packs, HYPOTHESIS_START_COLUMN, HYPOTHESIS_COUNT_COLUMN, hypothesis_idx);
    if (pack_idx < 0) {
        return;
    }

    const int64_t eval_start      = pack_info[pack_idx * PACK_INFO_COLUMNS + EVAL_START_COLUMN];
    const int64_t eval_count      = pack_info[pack_idx * PACK_INFO_COLUMNS + EVAL_COUNT_COLUMN];
    const float nx                = plane_normals[hypothesis_idx * 3 + 0];
    const float ny                = plane_normals[hypothesis_idx * 3 + 1];
    const float nz                = plane_normals[hypothesis_idx * 3 + 2];
    const float offset            = plane_offsets[hypothesis_idx];
    const float inv_normal_length = ONE / normal_lengths[hypothesis_idx];

    float partial_score = 0.0f;
    for (int64_t local_eval_idx = threadIdx.x; local_eval_idx < eval_count; local_eval_idx += blockDim.x) {
        const int64_t point_idx = eval_indices[eval_start + local_eval_idx];
        if (point_idx < 0 || point_idx >= n_points) {
            continue;
        }
        const float px        = points[point_idx * 3 + 0];
        const float py        = points[point_idx * 3 + 1];
        const float pz        = points[point_idx * 3 + 2];
        const float pnx       = normals[point_idx * 3 + 0];
        const float pny       = normals[point_idx * 3 + 1];
        const float pnz       = normals[point_idx * 3 + 2];
        const float distance  = fabsf(nx * px + ny * py + nz * pz + offset) * inv_normal_length;
        const float dot       = clamp_unit((pnx * nx + pny * ny + pnz * nz) * inv_normal_length);
        const float angle_deg = acosf(dot) * RAD_TO_DEG;
        partial_score += hann_weight(distance, distance_threshold) * hann_weight(angle_deg, angle_threshold_deg);
    }

    __shared__ typename BlockReduce::TempStorage temp_storage;
    const float block_score = BlockReduce(temp_storage).Sum(partial_score);
    if (threadIdx.x == 0) {
        scores[hypothesis_idx] = block_score;
    }
}

__global__ void select_best_planes_batched_kernel(
    const int64_t* __restrict__ pack_info,
    const float* __restrict__ plane_normals,
    const float* __restrict__ plane_offsets,
    const float* __restrict__ scores,
    float* __restrict__ best_normals,
    float* __restrict__ best_offsets,
    int64_t n_packs) {
    const int64_t pack_idx = blockIdx.x;
    if (pack_idx >= n_packs || threadIdx.x != 0) {
        return;
    }

    const int64_t hypothesis_start = pack_info[pack_idx * PACK_INFO_COLUMNS + HYPOTHESIS_START_COLUMN];
    const int64_t hypothesis_count = pack_info[pack_idx * PACK_INFO_COLUMNS + HYPOTHESIS_COUNT_COLUMN];
    int64_t best_idx               = hypothesis_start;
    float best_score               = scores[best_idx];
    for (int64_t local_idx = 1; local_idx < hypothesis_count; ++local_idx) {
        const int64_t idx = hypothesis_start + local_idx;
        const float score = scores[idx];
        if (score > best_score) {
            best_score = score;
            best_idx   = idx;
        }
    }
    best_normals[pack_idx * 3 + 0] = plane_normals[best_idx * 3 + 0];
    best_normals[pack_idx * 3 + 1] = plane_normals[best_idx * 3 + 1];
    best_normals[pack_idx * 3 + 2] = plane_normals[best_idx * 3 + 2];
    best_offsets[pack_idx]         = plane_offsets[best_idx];
}

__global__ void mark_plane_inliers_batched_kernel(
    const float* __restrict__ points,
    const float* __restrict__ normals,
    const int64_t* __restrict__ pack_info,
    const float* __restrict__ best_normals,
    const float* __restrict__ best_offsets,
    bool* __restrict__ inlier_mask,
    int64_t n_points,
    int64_t n_packs,
    float distance_threshold,
    float angle_threshold_deg) {
    const int64_t idx = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (idx >= n_points) {
        return;
    }
    const int64_t pack_idx =
        find_pack_for_index(pack_info, n_packs, POINT_START_COLUMN, POINT_COUNT_COLUMN, idx);
    if (pack_idx < 0) {
        inlier_mask[idx] = false;
        return;
    }

    const float nx                = best_normals[pack_idx * 3 + 0];
    const float ny                = best_normals[pack_idx * 3 + 1];
    const float nz                = best_normals[pack_idx * 3 + 2];
    const float inv_normal_length = ONE / fmaxf(sqrtf(nx * nx + ny * ny + nz * nz), NORM_EPSILON);
    const float px                = points[idx * 3 + 0];
    const float py                = points[idx * 3 + 1];
    const float pz                = points[idx * 3 + 2];
    const float pnx               = normals[idx * 3 + 0];
    const float pny               = normals[idx * 3 + 1];
    const float pnz               = normals[idx * 3 + 2];
    const float distance          = fabsf(nx * px + ny * py + nz * pz + best_offsets[pack_idx]) * inv_normal_length;
    const float dot               = clamp_unit((pnx * nx + pny * ny + pnz * nz) * inv_normal_length);
    const float angle_deg         = acosf(dot) * RAD_TO_DEG;
    inlier_mask[idx]              = distance <= distance_threshold && angle_deg <= angle_threshold_deg;
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> build_grid_mesh_mean_cu(
    const torch::Tensor& points,
    const torch::Tensor& colors,
    const double cell_size,
    const int min_points_per_cell,
    const int hole_fill_min_neighbors) {
    CHECK_INPUT(points);
    at::cuda::CUDAGuard device_guard(points.device());
    TORCH_CHECK(points.scalar_type() == torch::kFloat32, "points must be float32");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must have shape (N, 3)");
    TORCH_CHECK(cell_size > 0.0, "cell_size must be positive");
    TORCH_CHECK(min_points_per_cell >= 1, "min_points_per_cell must be >= 1");
    TORCH_CHECK(
        hole_fill_min_neighbors >= 0 && hole_fill_min_neighbors <= 8,
        "hole_fill_min_neighbors must be in [0, 8]");

    const bool has_colors = colors.defined();
    if (has_colors) {
        CHECK_INPUT(colors);
        check_same_device(points, colors, "colors");
        TORCH_CHECK(colors.scalar_type() == torch::kUInt8, "colors must be uint8");
        TORCH_CHECK(colors.sizes() == points.sizes(), "colors must have shape (N, 3)");
    }

    const int64_t n_points = points.size(0);
    auto point_options     = points.options();
    auto int_options       = point_options.dtype(torch::kInt32);
    auto byte_options      = point_options.dtype(torch::kUInt8);
    if (n_points == 0) {
        return {
            torch::empty({0, 3}, point_options),
            torch::empty({0, 3}, int_options),
            has_colors ? torch::empty({0, 3}, byte_options) : torch::Tensor(),
        };
    }

    auto xy                   = points.slice(1, 0, 2);
    auto bbox_min_t           = std::get<0>(xy.min(0));
    auto bbox_max_t           = std::get<0>(xy.max(0));
    const float min_x         = bbox_min_t[0].item<float>();
    const float min_y         = bbox_min_t[1].item<float>();
    const float max_x         = bbox_max_t[0].item<float>() + static_cast<float>(cell_size * BBOX_PAD_SCALE);
    const float max_y         = bbox_max_t[1].item<float>() + static_cast<float>(cell_size * BBOX_PAD_SCALE);
    const int width           = std::max(1, static_cast<int>(std::ceil((max_x - min_x) / cell_size)));
    const int height          = std::max(1, static_cast<int>(std::ceil((max_y - min_y) / cell_size)));
    const int64_t total_cells = static_cast<int64_t>(width) * static_cast<int64_t>(height);
    TORCH_CHECK(total_cells <= std::numeric_limits<int>::max(), "BEV grid is too large for int32 indexing");

    auto z_sums = torch::zeros({total_cells}, point_options);
    auto counts = torch::zeros({total_cells}, int_options);
    torch::Tensor color_sums;
    if (has_colors) {
        color_sums = torch::zeros({total_cells, 3}, point_options);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    accumulate_grid_kernel<<<blocks_for(n_points), THREADS_PER_BLOCK, 0, stream>>>(
        points.data_ptr<float>(),
        has_colors ? colors.data_ptr<uint8_t>() : nullptr,
        z_sums.data_ptr<float>(),
        has_colors ? color_sums.data_ptr<float>() : nullptr,
        counts.data_ptr<int>(),
        n_points,
        width,
        height,
        min_x,
        min_y,
        static_cast<float>(1.0 / cell_size));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto occupied_ids        = torch::nonzero(counts >= min_points_per_cell).flatten().contiguous();
    const int64_t n_occupied = occupied_ids.size(0);
    auto fillable_ids        = torch::empty({0}, occupied_ids.options());
    auto fill_z_sums         = torch::zeros({total_cells}, point_options);
    auto fill_weight_sums    = torch::zeros({total_cells}, int_options);
    torch::Tensor fill_color_sums;
    if (has_colors) {
        fill_color_sums = torch::zeros({total_cells, 3}, point_options);
    }
    auto fillable_mask      = torch::zeros({total_cells}, byte_options);
    auto horizontal_bracket = torch::empty({total_cells}, byte_options);
    auto vertical_bracket   = torch::empty({total_cells}, byte_options);
    mark_horizontal_brackets_kernel<<<blocks_for(height), THREADS_PER_BLOCK, 0, stream>>>(
        counts.data_ptr<int>(),
        z_sums.data_ptr<float>(),
        has_colors ? color_sums.data_ptr<float>() : nullptr,
        horizontal_bracket.data_ptr<uint8_t>(),
        fill_z_sums.data_ptr<float>(),
        has_colors ? fill_color_sums.data_ptr<float>() : nullptr,
        fill_weight_sums.data_ptr<int>(),
        width,
        height,
        min_points_per_cell);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    mark_vertical_brackets_kernel<<<blocks_for(width), THREADS_PER_BLOCK, 0, stream>>>(
        counts.data_ptr<int>(),
        z_sums.data_ptr<float>(),
        has_colors ? color_sums.data_ptr<float>() : nullptr,
        horizontal_bracket.data_ptr<uint8_t>(),
        vertical_bracket.data_ptr<uint8_t>(),
        fillable_mask.data_ptr<uint8_t>(),
        fill_z_sums.data_ptr<float>(),
        has_colors ? fill_color_sums.data_ptr<float>() : nullptr,
        fill_weight_sums.data_ptr<int>(),
        width,
        height,
        min_points_per_cell,
        hole_fill_min_neighbors);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fillable_ids           = torch::nonzero(fillable_mask).flatten().contiguous();
    const int64_t n_filled = fillable_ids.size(0);

    const int64_t n_vertices = n_occupied + n_filled;
    if (n_vertices < 3) {
        return {
            torch::empty({0, 3}, point_options),
            torch::empty({0, 3}, int_options),
            has_colors ? torch::empty({0, 3}, byte_options) : torch::Tensor(),
        };
    }

    auto vertices       = torch::empty({n_vertices, 3}, point_options);
    auto vertex_colors  = has_colors ? torch::empty({n_vertices, 3}, byte_options) : torch::Tensor();
    auto cell_to_vertex = torch::full({total_cells}, -1, int_options);
    build_vertices_kernel<<<blocks_for(n_occupied), THREADS_PER_BLOCK, 0, stream>>>(
        occupied_ids.data_ptr<int64_t>(),
        counts.data_ptr<int>(),
        z_sums.data_ptr<float>(),
        has_colors ? color_sums.data_ptr<float>() : nullptr,
        vertices.data_ptr<float>(),
        has_colors ? vertex_colors.data_ptr<uint8_t>() : nullptr,
        cell_to_vertex.data_ptr<int>(),
        n_occupied,
        width,
        min_x,
        min_y,
        static_cast<float>(cell_size));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (n_filled > 0) {
        build_filled_vertices_kernel<<<blocks_for(n_filled), THREADS_PER_BLOCK, 0, stream>>>(
            fillable_ids.data_ptr<int64_t>(),
            counts.data_ptr<int>(),
            z_sums.data_ptr<float>(),
            has_colors ? color_sums.data_ptr<float>() : nullptr,
            fill_z_sums.data_ptr<float>(),
            has_colors ? fill_color_sums.data_ptr<float>() : nullptr,
            fill_weight_sums.data_ptr<int>(),
            vertices.data_ptr<float>(),
            has_colors ? vertex_colors.data_ptr<uint8_t>() : nullptr,
            cell_to_vertex.data_ptr<int>(),
            n_filled,
            n_occupied,
            width,
            height,
            min_points_per_cell,
            min_x,
            min_y,
            static_cast<float>(cell_size));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    const int64_t n_2x2_blocks =
        static_cast<int64_t>(std::max(0, width - 1)) * static_cast<int64_t>(std::max(0, height - 1));
    auto triangle_buffer = torch::empty({2 * total_cells, 3}, int_options);
    auto triangle_count  = torch::zeros({1}, int_options);
    auto referenced_mask = torch::zeros({n_vertices}, byte_options);
    if (n_2x2_blocks > 0) {
        build_triangles_kernel<<<blocks_for(n_2x2_blocks), THREADS_PER_BLOCK, 0, stream>>>(
            cell_to_vertex.data_ptr<int>(),
            triangle_buffer.data_ptr<int>(),
            triangle_count.data_ptr<int>(),
            referenced_mask.data_ptr<uint8_t>(),
            n_2x2_blocks,
            width,
            height);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    const int n_triangles = triangle_count[0].item<int>();
    if (n_triangles == 0) {
        return {
            torch::empty({0, 3}, point_options),
            torch::empty({0, 3}, int_options),
            has_colors ? torch::empty({0, 3}, byte_options) : torch::Tensor(),
        };
    }

    auto triangles = triangle_buffer.narrow(0, 0, n_triangles).contiguous();

    auto referenced_ids        = torch::nonzero(referenced_mask).flatten().contiguous();
    const int64_t n_referenced = referenced_ids.size(0);
    auto old_to_new            = torch::full({n_vertices}, -1, int_options);
    build_old_to_new_kernel<<<blocks_for(n_referenced), THREADS_PER_BLOCK, 0, stream>>>(
        referenced_ids.data_ptr<int64_t>(),
        old_to_new.data_ptr<int>(),
        n_referenced);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    remap_triangles_kernel<<<blocks_for(static_cast<int64_t>(n_triangles) * 3), THREADS_PER_BLOCK, 0, stream>>>(
        old_to_new.data_ptr<int>(),
        triangles.data_ptr<int>(),
        static_cast<int64_t>(n_triangles) * 3);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    vertices = vertices.index_select(0, referenced_ids);
    if (has_colors) {
        vertex_colors = vertex_colors.index_select(0, referenced_ids);
    }

    return {vertices, triangles, vertex_colors};
}

template <typename Index>
torch::Tensor smooth_mesh_along_axis_impl(
    const torch::Tensor& vertices,
    const torch::Tensor& edges,
    const int axis,
    const double smoothness,
    const int num_passes) {
    const int64_t n_vertices = vertices.size(0);
    const int64_t n_edges    = edges.size(0);
    const auto index_options = edges.options().dtype(sizeof(Index) == sizeof(int) ? torch::kInt32 : torch::kInt64);
    auto counts              = torch::zeros({n_vertices + 1}, index_options);
    cudaStream_t stream      = at::cuda::getCurrentCUDAStream();
    count_smoothing_neighbors_kernel<<<blocks_for(n_edges), THREADS_PER_BLOCK, 0, stream>>>(
        edges.data_ptr<int64_t>(), counts.data_ptr<Index>(), n_edges, n_vertices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto offsets = torch::cumsum(counts, 0, counts.scalar_type());
    // Reserve a transposed slot for every short-row position plus a contiguous
    // tail large enough for every directed edge in long rows.
    const int64_t neighbor_capacity = SHORT_NEIGHBOR_CAPACITY * n_vertices + 2 * n_edges;
    auto neighbors                  = torch::empty({neighbor_capacity}, index_options);
    fill_smoothing_neighbors_kernel<<<blocks_for(n_edges), THREADS_PER_BLOCK, 0, stream>>>(
        edges.data_ptr<int64_t>(), offsets.data_ptr<Index>(), counts.data_ptr<Index>(),
        neighbors.data_ptr<Index>(), n_edges, n_vertices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto weights = torch::empty({neighbor_capacity}, vertices.options());
    // Store the three per-vertex work arrays in one contiguous allocation.
    auto state          = torch::empty({3, n_vertices}, vertices.options());
    float* heights      = state.data_ptr<float>();
    float* next_heights = heights + n_vertices;
    float* weight_sums  = next_heights + n_vertices;
    auto out            = torch::empty_like(vertices);
    prepare_smoothing_rows_kernel<<<blocks_for(n_vertices), THREADS_PER_BLOCK, 0, stream>>>(
        vertices.data_ptr<float>(), offsets.data_ptr<Index>(), neighbors.data_ptr<Index>(),
        weights.data_ptr<float>(), weight_sums, heights, out.data_ptr<float>(), n_vertices, axis);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int pass = 0; pass < num_passes; ++pass) {
        const bool last_pass = pass + 1 == num_passes;
        // - Intermediate passes write a contiguous scratch buffer.
        // - The last pass writes directly to the selected output column.
        smooth_neighbor_rows_kernel<<<blocks_for(n_vertices), THREADS_PER_BLOCK, 0, stream>>>(
            offsets.data_ptr<Index>(), neighbors.data_ptr<Index>(), weights.data_ptr<float>(),
            weight_sums, heights, last_pass ? out.data_ptr<float>() + axis : next_heights,
            n_vertices, static_cast<float>(smoothness), last_pass ? 3 : 1);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        std::swap(heights, next_heights);
    }
    return out;
}

torch::Tensor smooth_mesh_along_axis_cu(
    const torch::Tensor& vertices,
    const torch::Tensor& edges,
    const int axis,
    const double smoothness,
    const int num_passes) {
    CHECK_INPUT(vertices);
    at::cuda::CUDAGuard device_guard(vertices.device());
    CHECK_INPUT(edges);
    check_same_device(vertices, edges, "edges");
    TORCH_CHECK(vertices.scalar_type() == torch::kFloat32, "vertices must be float32");
    TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3, "vertices must have shape (N, 3)");
    TORCH_CHECK(edges.scalar_type() == torch::kInt64, "edges must be int64");
    TORCH_CHECK(edges.dim() == 2 && edges.size(1) == 2, "edges must have shape (E, 2)");
    TORCH_CHECK(axis >= 0 && axis < 3, "axis must be 0, 1, or 2");
    TORCH_CHECK(num_passes >= 0, "num_passes must be non-negative");

    const int64_t n_vertices = vertices.size(0);
    const int64_t n_edges    = edges.size(0);
    if (n_vertices == 0 || n_edges == 0 || num_passes == 0) {
        return vertices.clone();
    }

    // Select 32-bit adjacency only when:
    // - three coordinates per vertex fit a signed offset
    // - two directed entries per edge fit a signed offset
    if (n_vertices <= std::numeric_limits<int>::max() / 3 && n_edges <= std::numeric_limits<int>::max() / 2) {
        return smooth_mesh_along_axis_impl<int>(vertices, edges, axis, smoothness, num_passes);
    }
    return smooth_mesh_along_axis_impl<int64_t>(vertices, edges, axis, smoothness, num_passes);
}

torch::Tensor compute_vertex_normals_cu(
    const torch::Tensor& vertices,
    const torch::Tensor& triangles) {
    CHECK_INPUT(vertices);
    at::cuda::CUDAGuard device_guard(vertices.device());
    CHECK_INPUT(triangles);
    check_same_device(vertices, triangles, "triangles");
    TORCH_CHECK(vertices.scalar_type() == torch::kFloat32, "vertices must be float32");
    TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3, "vertices must have shape (N, 3)");
    TORCH_CHECK(triangles.scalar_type() == torch::kInt32, "triangles must be int32");
    TORCH_CHECK(triangles.dim() == 2 && triangles.size(1) == 3, "triangles must have shape (T, 3)");

    const int64_t n_vertices  = vertices.size(0);
    const int64_t n_triangles = triangles.size(0);
    auto normals              = torch::zeros_like(vertices);
    if (n_vertices == 0 || n_triangles == 0) {
        return normals;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    accumulate_face_normals_kernel<<<blocks_for(n_triangles), THREADS_PER_BLOCK, 0, stream>>>(
        vertices.data_ptr<float>(),
        triangles.data_ptr<int>(),
        normals.data_ptr<float>(),
        n_triangles,
        n_vertices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    normalize_normals_kernel<<<blocks_for(n_vertices), THREADS_PER_BLOCK, 0, stream>>>(
        normals.data_ptr<float>(),
        n_vertices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return normals;
}

torch::Tensor find_ground_compatible_points_cu(
    const torch::Tensor& points,
    const torch::Tensor& normals,
    const torch::Tensor& ground_control_point,
    const torch::Tensor& ref_normal,
    const double max_distance,
    const double max_angle_deg) {
    CHECK_INPUT(points);
    at::cuda::CUDAGuard device_guard(points.device());
    CHECK_INPUT(normals);
    CHECK_INPUT(ground_control_point);
    CHECK_INPUT(ref_normal);
    check_same_device(points, normals, "normals");
    check_same_device(points, ground_control_point, "ground_control_point");
    check_same_device(points, ref_normal, "ref_normal");
    TORCH_CHECK(points.scalar_type() == torch::kFloat32, "points must be float32");
    TORCH_CHECK(normals.scalar_type() == torch::kFloat32, "normals must be float32");
    TORCH_CHECK(ground_control_point.scalar_type() == torch::kFloat32, "ground_control_point must be float32");
    TORCH_CHECK(ref_normal.scalar_type() == torch::kFloat32, "ref_normal must be float32");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must have shape (N, 3)");
    TORCH_CHECK(normals.sizes() == points.sizes(), "normals must have shape (N, 3)");
    TORCH_CHECK(ground_control_point.numel() == 3, "ground_control_point must contain 3 values");
    TORCH_CHECK(ref_normal.numel() == 3, "ref_normal must contain 3 values");
    TORCH_CHECK(max_distance > 0.0, "max_distance must be positive");
    TORCH_CHECK(max_angle_deg > 0.0 && max_angle_deg <= 180.0, "max_angle_deg must be in (0, 180]");

    const int64_t n_points = points.size(0);
    auto out               = torch::empty({n_points}, points.options().dtype(torch::kBool));
    if (n_points == 0) {
        return out;
    }

    cudaStream_t stream       = at::cuda::getCurrentCUDAStream();
    const float min_cos_angle = cosf(static_cast<float>(max_angle_deg) * DEG_TO_RAD);
    ground_compatible_kernel<<<grid_stride_blocks_for(n_points), THREADS_PER_BLOCK, 0, stream>>>(
        points.data_ptr<float>(),
        normals.data_ptr<float>(),
        ground_control_point.contiguous().data_ptr<float>(),
        ref_normal.contiguous().data_ptr<float>(),
        out.data_ptr<bool>(),
        n_points,
        static_cast<float>(max_distance),
        min_cos_angle);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> detect_dominant_plane_cu(
    const torch::Tensor& points,
    const torch::Tensor& normals,
    const torch::Tensor& hypothesis_indices,
    const torch::Tensor& eval_indices,
    const double distance_threshold,
    const double angle_threshold_deg) {
    CHECK_INPUT(points);
    at::cuda::CUDAGuard device_guard(points.device());
    CHECK_INPUT(normals);
    CHECK_INPUT(hypothesis_indices);
    CHECK_INPUT(eval_indices);
    check_same_device(points, normals, "normals");
    check_same_device(points, hypothesis_indices, "hypothesis_indices");
    check_same_device(points, eval_indices, "eval_indices");
    TORCH_CHECK(points.scalar_type() == torch::kFloat32, "points must be float32");
    TORCH_CHECK(normals.scalar_type() == torch::kFloat32, "normals must be float32");
    TORCH_CHECK(hypothesis_indices.scalar_type() == torch::kInt64, "hypothesis_indices must be int64");
    TORCH_CHECK(eval_indices.scalar_type() == torch::kInt64, "eval_indices must be int64");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must have shape (N, 3)");
    TORCH_CHECK(normals.sizes() == points.sizes(), "normals must have shape (N, 3)");
    TORCH_CHECK(hypothesis_indices.dim() == 1, "hypothesis_indices must be 1D");
    TORCH_CHECK(eval_indices.dim() == 1, "eval_indices must be 1D");
    TORCH_CHECK(hypothesis_indices.numel() > 0, "hypothesis_indices must not be empty");
    TORCH_CHECK(eval_indices.numel() > 0, "eval_indices must not be empty");
    TORCH_CHECK(distance_threshold > 0.0, "distance_threshold must be positive");
    TORCH_CHECK(angle_threshold_deg > 0.0 && angle_threshold_deg <= 90.0,
                "angle_threshold_deg must be in (0, 90]");

    const int64_t n_points      = points.size(0);
    const int64_t n_hypotheses  = hypothesis_indices.size(0);
    const int64_t n_eval_points = eval_indices.size(0);
    auto float_options          = points.options();
    auto plane_normals          = torch::empty({n_hypotheses, 3}, float_options);
    auto plane_offsets          = torch::empty({n_hypotheses}, float_options);
    auto normal_lengths         = torch::empty({n_hypotheses}, float_options);
    auto scores                 = torch::empty({n_hypotheses}, float_options);
    auto best_normal            = torch::empty({3}, float_options);
    auto best_offset            = torch::empty({1}, float_options);
    auto inlier_mask            = torch::empty({n_points}, points.options().dtype(torch::kBool));
    cudaStream_t stream         = at::cuda::getCurrentCUDAStream();

    build_plane_hypotheses_kernel<<<blocks_for(n_hypotheses), THREADS_PER_BLOCK, 0, stream>>>(
        points.data_ptr<float>(),
        normals.data_ptr<float>(),
        hypothesis_indices.data_ptr<int64_t>(),
        plane_normals.data_ptr<float>(),
        plane_offsets.data_ptr<float>(),
        normal_lengths.data_ptr<float>(),
        n_hypotheses,
        n_points);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    score_plane_hypotheses_kernel<<<static_cast<int>(n_hypotheses), THREADS_PER_BLOCK, 0, stream>>>(
        points.data_ptr<float>(),
        normals.data_ptr<float>(),
        eval_indices.data_ptr<int64_t>(),
        plane_normals.data_ptr<float>(),
        plane_offsets.data_ptr<float>(),
        normal_lengths.data_ptr<float>(),
        scores.data_ptr<float>(),
        n_eval_points,
        n_points,
        static_cast<float>(distance_threshold),
        static_cast<float>(angle_threshold_deg));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    select_best_plane_kernel<<<1, 1, 0, stream>>>(
        plane_normals.data_ptr<float>(),
        plane_offsets.data_ptr<float>(),
        scores.data_ptr<float>(),
        best_normal.data_ptr<float>(),
        best_offset.data_ptr<float>(),
        n_hypotheses);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    mark_plane_inliers_kernel<<<blocks_for(n_points), THREADS_PER_BLOCK, 0, stream>>>(
        points.data_ptr<float>(),
        normals.data_ptr<float>(),
        best_normal.data_ptr<float>(),
        best_offset.data_ptr<float>(),
        inlier_mask.data_ptr<bool>(),
        n_points,
        static_cast<float>(distance_threshold),
        static_cast<float>(angle_threshold_deg));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {best_normal, best_offset, inlier_mask};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> detect_dominant_plane_batched_cu(
    const torch::Tensor& points,
    const torch::Tensor& normals,
    const torch::Tensor& hypothesis_indices,
    const torch::Tensor& eval_indices,
    const torch::Tensor& pack_info,
    const double distance_threshold,
    const double angle_threshold_deg) {
    CHECK_INPUT(points);
    at::cuda::CUDAGuard device_guard(points.device());
    CHECK_INPUT(normals);
    CHECK_INPUT(hypothesis_indices);
    CHECK_INPUT(eval_indices);
    CHECK_INPUT(pack_info);
    check_same_device(points, normals, "normals");
    check_same_device(points, hypothesis_indices, "hypothesis_indices");
    check_same_device(points, eval_indices, "eval_indices");
    check_same_device(points, pack_info, "pack_info");
    TORCH_CHECK(points.scalar_type() == torch::kFloat32, "points must be float32");
    TORCH_CHECK(normals.scalar_type() == torch::kFloat32, "normals must be float32");
    TORCH_CHECK(hypothesis_indices.scalar_type() == torch::kInt64, "hypothesis_indices must be int64");
    TORCH_CHECK(eval_indices.scalar_type() == torch::kInt64, "eval_indices must be int64");
    TORCH_CHECK(pack_info.scalar_type() == torch::kInt64, "pack_info must be int64");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must have shape (N, 3)");
    TORCH_CHECK(normals.sizes() == points.sizes(), "normals must have shape (N, 3)");
    TORCH_CHECK(hypothesis_indices.dim() == 1, "hypothesis_indices must be 1D");
    TORCH_CHECK(eval_indices.dim() == 1, "eval_indices must be 1D");
    TORCH_CHECK(pack_info.dim() == 2 && pack_info.size(1) == PACK_INFO_COLUMNS,
                "pack_info must have shape (B, 6)");
    TORCH_CHECK(pack_info.size(0) > 0, "pack_info must not be empty");
    TORCH_CHECK(hypothesis_indices.numel() > 0, "hypothesis_indices must not be empty");
    TORCH_CHECK(eval_indices.numel() > 0, "eval_indices must not be empty");
    TORCH_CHECK(distance_threshold > 0.0, "distance_threshold must be positive");
    TORCH_CHECK(angle_threshold_deg > 0.0 && angle_threshold_deg <= 90.0,
                "angle_threshold_deg must be in (0, 90]");

    const int64_t n_points     = points.size(0);
    const int64_t n_hypotheses = hypothesis_indices.size(0);
    const int64_t n_packs      = pack_info.size(0);
    const auto pack_info_cpu   = pack_info.cpu().contiguous();
    const auto* pack_data      = pack_info_cpu.data_ptr<int64_t>();
    for (int64_t pack_idx = 0; pack_idx < n_packs; ++pack_idx) {
        const auto* pack               = pack_data + pack_idx * PACK_INFO_COLUMNS;
        const int64_t point_start      = pack[POINT_START_COLUMN];
        const int64_t point_count      = pack[POINT_COUNT_COLUMN];
        const int64_t hypothesis_start = pack[HYPOTHESIS_START_COLUMN];
        const int64_t hypothesis_count = pack[HYPOTHESIS_COUNT_COLUMN];
        const int64_t eval_start       = pack[EVAL_START_COLUMN];
        const int64_t eval_count       = pack[EVAL_COUNT_COLUMN];
        TORCH_CHECK(
            point_start >= 0 && point_count >= 0 && point_start <= n_points - point_count,
            "pack_info row ",
            pack_idx,
            " has an invalid point range");
        TORCH_CHECK(
            hypothesis_start >= 0 && hypothesis_count > 0 && hypothesis_start <= n_hypotheses - hypothesis_count,
            "pack_info row ",
            pack_idx,
            " has an invalid hypothesis range");
        TORCH_CHECK(
            eval_start >= 0 && eval_count >= 0 && eval_start <= eval_indices.size(0) - eval_count,
            "pack_info row ",
            pack_idx,
            " has an invalid eval range");
    }
    auto float_options  = points.options();
    auto plane_normals  = torch::empty({n_hypotheses, 3}, float_options);
    auto plane_offsets  = torch::empty({n_hypotheses}, float_options);
    auto normal_lengths = torch::empty({n_hypotheses}, float_options);
    auto scores         = torch::empty({n_hypotheses}, float_options);
    auto best_normals   = torch::empty({n_packs, 3}, float_options);
    auto best_offsets   = torch::empty({n_packs}, float_options);
    auto inlier_mask    = torch::empty({n_points}, points.options().dtype(torch::kBool));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    build_plane_hypotheses_kernel<<<blocks_for(n_hypotheses), THREADS_PER_BLOCK, 0, stream>>>(
        points.data_ptr<float>(),
        normals.data_ptr<float>(),
        hypothesis_indices.data_ptr<int64_t>(),
        plane_normals.data_ptr<float>(),
        plane_offsets.data_ptr<float>(),
        normal_lengths.data_ptr<float>(),
        n_hypotheses,
        n_points);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    score_plane_hypotheses_batched_kernel<<<static_cast<int>(n_hypotheses), THREADS_PER_BLOCK, 0, stream>>>(
        points.data_ptr<float>(),
        normals.data_ptr<float>(),
        eval_indices.data_ptr<int64_t>(),
        pack_info.data_ptr<int64_t>(),
        plane_normals.data_ptr<float>(),
        plane_offsets.data_ptr<float>(),
        normal_lengths.data_ptr<float>(),
        scores.data_ptr<float>(),
        n_packs,
        n_points,
        static_cast<float>(distance_threshold),
        static_cast<float>(angle_threshold_deg));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    select_best_planes_batched_kernel<<<static_cast<int>(n_packs), 1, 0, stream>>>(
        pack_info.data_ptr<int64_t>(),
        plane_normals.data_ptr<float>(),
        plane_offsets.data_ptr<float>(),
        scores.data_ptr<float>(),
        best_normals.data_ptr<float>(),
        best_offsets.data_ptr<float>(),
        n_packs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    mark_plane_inliers_batched_kernel<<<blocks_for(n_points), THREADS_PER_BLOCK, 0, stream>>>(
        points.data_ptr<float>(),
        normals.data_ptr<float>(),
        pack_info.data_ptr<int64_t>(),
        best_normals.data_ptr<float>(),
        best_offsets.data_ptr<float>(),
        inlier_mask.data_ptr<bool>(),
        n_points,
        n_packs,
        static_cast<float>(distance_threshold),
        static_cast<float>(angle_threshold_deg));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {best_normals, best_offsets, inlier_mask};
}
