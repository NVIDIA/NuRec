// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

struct DirectTracksCalib : TracksCalib
{
    no_diff TensorView<bool> gradient_mask;
    no_diff TensorView<float> tracks_poses;
    DiffTensorView tracks_delta_q;
    DiffTensorView tracks_delta_t;
    no_diff bool left_multiply;

    [Differentiable]
    transform::SE3 calib(no_diff uint pose_idx)
    {
        const bool use_gradient = gradient_mask.load(pose_idx);
        const float3 current_tracks_delta_t = utils::pass(
            tracks_delta_t.loadVecOnce<3>(uint2(pose_idx, 0)), use_gradient
            );
        const float4 current_tracks_delta_q = quaternion::normalizeSafe(
            utils::pass(
                tracks_delta_q.loadVecOnce<4>(uint2(pose_idx, 0)), use_gradient
                )
            );

        const float3 tracks_t = no_diff tracks_poses.loadVec<3>(uint2(pose_idx, 0));
        float4 tracks_q = no_diff tracks_poses.loadVec<4>(uint2(pose_idx, 3));
        tracks_q = quaternion::normalizeSafe(tracks_q);

        // Track and calibration rotations use xyzw order.
        const var delta = transform::SE3.from_vec(current_tracks_delta_t, current_tracks_delta_q);
        const var pose = transform::SE3.from_vec(tracks_t, tracks_q);
        if (left_multiply)
        {
            return delta * pose;
        }
        return pose * delta;
    }
};

// Materialize calibrated keyframes once because later intersection and interpolation reuse them.
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void tracks_calib_direct(
    no_diff uint count,
    DirectTracksCalib calib,
    DiffTensorView output_poses,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    let pose = calib.calib(tid).to_vec();

    output_poses.storeVecOnce<3>(uint2(tid, 0), pose._0);
    output_poses.storeVecOnce<4>(uint2(tid, 3), pose._1);
}
