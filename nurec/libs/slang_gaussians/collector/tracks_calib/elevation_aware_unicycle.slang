// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// sin(x/2) / (x/2). Near zero, the Taylor form keeps the autodiff derivative
// stable while its omitted fourth-order term is still below float32 precision.
[Differentiable]
float tracks_calib_sinc_half(const float x)
{
    const float h = 0.5f * x;
    return abs(h) < 1e-2f ? 1.0f - h * h / 6.0f : sin(h) / h;
}

// Compose yaw about +z and nose-up pitch about -y in xyzw order.
[Differentiable]
float4 tracks_calib_yaw_pitch_quat(const float theta, const float pitch)
{
    const float4 yaw_quat = quaternion::convertAngleToQuat(float3(0.0f, 0.0f, 1.0f), theta);
    const float4 pitch_quat = quaternion::convertAngleToQuat(float3(0.0f, -1.0f, 0.0f), pitch);
    return quaternion::normalizeSafe(quaternion::multiply(yaw_quat, pitch_quat));
}

// Compose T_world_ground * T_ground_cuboid_local * T_cuboid_local_cuboid per keyframe.
struct ElevationAwareUnicycleTracksCalib : TracksCalib
{
    DiffTensorView xy;
    DiffTensorView height;
    DiffTensorView theta;
    DiffTensorView pitch;
    no_diff TensorView<bool> gradient_mask;
    no_diff TensorView<int64_t> tracks_idx;
    no_diff TensorView<float> ground_t;
    no_diff TensorView<float> ground_q;
    no_diff TensorView<float> tclc;

    [Differentiable]
    transform::SE3 calib(no_diff uint pose_idx)
    {
        float2 p_xy = xy.loadVecOnce<2>(uint2(pose_idx, 0));
        float h = height.loadOnce(pose_idx);
        float th = theta.loadOnce(pose_idx);
        float pi_ = pitch.loadOnce(pose_idx);
        const bool use_gradient = gradient_mask[pose_idx];
        p_xy = use_gradient ? p_xy : detach(p_xy);
        h = use_gradient ? h : detach(h);
        th = use_gradient ? th : detach(th);
        pi_ = use_gradient ? pi_ : detach(pi_);

        const uint track_idx = uint(tracks_idx[pose_idx]);
        const float3 world_ground_t = float3(
            ground_t[uint2(track_idx, 0)], ground_t[uint2(track_idx, 1)], ground_t[uint2(track_idx, 2)]);
        const float4 world_ground_q = quaternion::normalizeSafe(float4(
            ground_q[uint2(track_idx, 0)], ground_q[uint2(track_idx, 1)], ground_q[uint2(track_idx, 2)], ground_q[uint2(track_idx, 3)]));
        const float3 tclc_t = float3(tclc[0], tclc[1], tclc[2]);
        const float4 tclc_q = quaternion::normalizeSafe(float4(tclc[3], tclc[4], tclc[5], tclc[6]));

        const float3 ground_cuboid_t = float3(p_xy.x, p_xy.y, h);
        const float4 ground_cuboid_q = tracks_calib_yaw_pitch_quat(th, pi_);

        const float3 world_local_t = world_ground_t + quaternion::rotateVector(world_ground_q, ground_cuboid_t);
        const float4 world_local_q = quaternion::normalizeSafe(
            quaternion::multiply(world_ground_q, ground_cuboid_q));
        const float3 world_cuboid_t = world_local_t + quaternion::rotateVector(world_local_q, tclc_t);
        const float4 world_cuboid_q = quaternion::normalizeSafe(quaternion::multiply(world_local_q, tclc_q));
        return transform::SE3.from_vec(world_cuboid_t, world_cuboid_q);
    }
};

// Materialize calibrated keyframes once because later intersection and interpolation reuse them.
[CUDAKernel]
[AutoPyBindCUDA]
[Differentiable]
void tracks_calib_unicycle_pose(
    no_diff uint count,
    ElevationAwareUnicycleTracksCalib calib,
    DiffTensorView output_poses)
{
    const uint tid = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (tid >= count)
    {
        return;
    }

    let pose = calib.calib(tid).to_vec();
    output_poses.storeVecOnce<3>(uint2(tid, 0), pose._0);
    output_poses.storeVecOnce<4>(uint2(tid, 3), pose._1);
}

// Compare adjacent pose changes with the learned motion state.
[CUDAKernel]
[AutoPyBindCUDA]
[Differentiable]
void tracks_calib_unicycle_consistency(
    DiffTensorView xy,                        // (n, 2)
    DiffTensorView height,                    // (n,)
    DiffTensorView theta,                     // (n,)
    DiffTensorView pitch,                     // (n,)
    DiffTensorView velocity,                  // (n_pairs,)
    DiffTensorView theta_velocity,            // (n_pairs,)
    DiffTensorView pitch_velocity,            // (n_pairs,)
    no_diff TensorView<bool> keep,            // (n,)
    no_diff TensorView<int64_t> pair_start,   // (n_pairs,)
    no_diff TensorView<float> time_delta_s,   // (n_pairs,)
    no_diff float pitch_slope_clamp_rad,
    DiffTensorView consistency)               // (n_pairs, 5) out
{
    const uint pair = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (pair >= velocity.size(0))
    {
        return;
    }

    const uint start = uint(pair_start[pair]);
    const uint end = start + 1;

    // Adjacent pairs share pose rows, so their backward gathers must accumulate.
    float2 xy_s = xy.loadVecEx<2>(uint2(start, 0));
    float2 xy_e = xy.loadVecEx<2>(uint2(end, 0));
    float h_s = height.loadEx(uint1(start));
    float h_e = height.loadEx(uint1(end));
    float th_s = theta.loadEx(uint1(start));
    float pi_s = pitch.loadEx(uint1(start));
    float pi_e = pitch.loadEx(uint1(end));
    float th_e = theta.loadEx(uint1(end));

    const bool k_s = keep[start];
    const bool k_e = keep[end];
    xy_s = k_s ? xy_s : detach(xy_s);
    h_s = k_s ? h_s : detach(h_s);
    th_s = k_s ? th_s : detach(th_s);
    pi_s = k_s ? pi_s : detach(pi_s);
    xy_e = k_e ? xy_e : detach(xy_e);
    h_e = k_e ? h_e : detach(h_e);
    th_e = k_e ? th_e : detach(th_e);
    pi_e = k_e ? pi_e : detach(pi_e);

    const float v = velocity.loadOnce(pair);
    const float tv = theta_velocity.loadOnce(pair);
    const float pv = pitch_velocity.loadOnce(pair);
    const float dt = time_delta_s[pair];

    const float dtheta = tv * dt;
    const float dpitch = pv * dt;
    const float theta_mid = th_s + 0.5f * dtheta;
    const float arc_length = v * dt * tracks_calib_sinc_half(dtheta);
    const float pitch_mid = clamp(0.5f * (pi_s + pi_e), -pitch_slope_clamp_rad, pitch_slope_clamp_rad);

    consistency.storeOnce(uint2(pair, 0), xy_e.x - xy_s.x - arc_length * cos(theta_mid));
    consistency.storeOnce(uint2(pair, 1), xy_e.y - xy_s.y - arc_length * sin(theta_mid));
    consistency.storeOnce(uint2(pair, 2), th_e - th_s - dtheta);
    consistency.storeOnce(uint2(pair, 3), pi_e - pi_s - dpitch);
    consistency.storeOnce(uint2(pair, 4), h_e - h_s - v * dt * tan(pitch_mid));
}

// Compute horizontal, angular, and vertical acceleration changes over three pairs.
[CUDAKernel]
[AutoPyBindCUDA]
[Differentiable]
void tracks_calib_unicycle_smoothness(
    DiffTensorView height,                     // (n,)
    DiffTensorView velocity,                   // (n_pairs,)
    DiffTensorView theta_velocity,             // (n_pairs,)
    DiffTensorView pitch_velocity,             // (n_pairs,)
    no_diff TensorView<bool> keep,             // (n,)
    no_diff TensorView<int64_t> pair_start,    // (n_pairs,)
    no_diff TensorView<int64_t> triplet_start, // (n_triplets,)
    no_diff TensorView<float> time_delta_s,    // (n_pairs,)
    no_diff TensorView<float> time_delta_v_s,  // (n_pairs,)
    DiffTensorView smoothness)                 // (n_triplets, 4) out
{
    const uint triplet = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (triplet >= smoothness.size(0))
    {
        return;
    }

    const uint t0 = uint(triplet_start[triplet]);
    const uint t1 = t0 + 1;
    const uint t2 = t0 + 2;
    const float dt1 = time_delta_v_s[t1];
    const float dt0 = time_delta_v_s[t0];

    // Neighbouring triplets share pair rows, so their backward gathers must accumulate.
    const float v0 = velocity.loadEx(uint1(t0));
    const float v1 = velocity.loadEx(uint1(t1));
    const float v2 = velocity.loadEx(uint1(t2));
    const float tv0 = theta_velocity.loadEx(uint1(t0));
    const float tv1 = theta_velocity.loadEx(uint1(t1));
    const float tv2 = theta_velocity.loadEx(uint1(t2));
    const float pv0 = pitch_velocity.loadEx(uint1(t0));
    const float pv1 = pitch_velocity.loadEx(uint1(t1));
    const float pv2 = pitch_velocity.loadEx(uint1(t2));

    float height_velocity[3];
    [ForceUnroll]
    for (int j = 0; j < 3; ++j)
    {
        const uint pair = t0 + j;
        const uint start = uint(pair_start[pair]);
        float h_s = height.loadEx(uint1(start));
        float h_e = height.loadEx(uint1(start + 1));
        h_s = keep[start] ? h_s : detach(h_s);
        h_e = keep[start + 1] ? h_e : detach(h_e);
        height_velocity[j] = (h_e - h_s) / time_delta_s[pair];
    }

    smoothness.storeOnce(uint2(triplet, 0), (v2 - v1) / dt1 - (v1 - v0) / dt0);
    smoothness.storeOnce(uint2(triplet, 1), (tv2 - tv1) / dt1 - (tv1 - tv0) / dt0);
    smoothness.storeOnce(uint2(triplet, 2), (pv2 - pv1) / dt1 - (pv1 - pv0) / dt0);
    smoothness.storeOnce(
        uint2(triplet, 3),
        (height_velocity[2] - height_velocity[1]) / dt1 - (height_velocity[1] - height_velocity[0]) / dt0);
}

// Keep learned positions close to their initialized positions.
[CUDAKernel]
[AutoPyBindCUDA]
[Differentiable]
void tracks_calib_unicycle_pose_regularization(
    DiffTensorView xy,                          // (n, 2)
    DiffTensorView height,                      // (n,)
    no_diff TensorView<bool> keep,              // (n,)
    no_diff TensorView<float> initial_xy,       // (n, 2)
    no_diff TensorView<float> initial_height,   // (n,)
    DiffTensorView regularization)              // (n,) out
{
    const uint tid = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (tid >= height.size(0))
    {
        return;
    }

    float2 p_xy = xy.loadVecOnce<2>(uint2(tid, 0));
    float h = height.loadOnce(tid);
    const bool k = keep[tid];
    p_xy = k ? p_xy : detach(p_xy);
    h = k ? h : detach(h);

    const float dx = p_xy.x - initial_xy[uint2(tid, 0)];
    const float dy = p_xy.y - initial_xy[uint2(tid, 1)];
    const float dh = h - initial_height[tid];
    regularization.storeOnce(tid, dx * dx + dy * dy + dh * dh);
}
