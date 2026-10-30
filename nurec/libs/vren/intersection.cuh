// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ku/helper_math.cuh>

__device__ __forceinline__ float2 ray_aabb_intersect(
    const float3 ray_o,
    const float3 inv_d,
    const float3 aabb_min,
    const float3 aabb_max) {
    const float3 t_min = (aabb_min - ray_o) * inv_d;
    const float3 t_max = (aabb_max - ray_o) * inv_d;

    const float3 _t1 = fminf(t_min, t_max);
    const float3 _t2 = fmaxf(t_min, t_max);
    const float t1   = fmaxf(fmaxf(_t1.x, _t1.y), _t1.z);
    const float t2   = fminf(fminf(_t2.x, _t2.y), _t2.z);

    if (t1 > t2)
        return make_float2(-1.0f); // no intersection
    return make_float2(t1, t2);
}
