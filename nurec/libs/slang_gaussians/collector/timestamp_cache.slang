// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Included by collector.slang after common/ray_intersect.slang.

// Layout is a host-selected compile-time constant from explicit caller metadata: 0=rows, 1=columns, 2=global per image.
// Both readout directions on an axis use the same mapping. Random/unknown
// layouts bypass these kernels on the host. Images are flattened [B,H,W].
// Map a valid flattened [B,H,W] ray to its cache group. height and width must
// be positive; layout is 0 (rows), 1 (columns), or 2 (global per image).
uint timestamp_cache_group<let Layout : uint>(uint ray, uint height, uint width) {
    if (Layout == 0) return ray / width;
    let image = ray / (height * width);
    if (Layout == 1) return image * width + ray % width;
    return image;
}

// Return a valid representative ray for a group from timestamp_cache_group.
// The first column/row/pixel supplies the actual timestamp, including reverse readout.
uint timestamp_cache_representative_ray<let Layout : uint>(uint group, uint height, uint width) {
    if (Layout == 0) return group * width;
    if (Layout == 1) return (group / width) * height * width + group % width;
    return group * height * width;
}

// Refresh cached SE3 poses for the active tracks; this is non-differentiable.
// All tensors are contiguous on the launch CUDA device: float32 poses [P,7],
// int64 sorted per-track timestamps [P] and ray timestamps [B*H*W], int32
// packinfo [T,2], unique valid active indices [T], active_count [1] in [0,T].
// Outputs: float32 prepared_poses [T*groups,7], int32 prepared_valid [T*groups].
// groups = B*H (rows), B*W (columns), or B (global); T*groups must fit uint32.
// Cache entries use active_index*groups+group; only entries below count*groups
// are written. Invalid/out-of-range and <=2-pose tracks set valid=0; consumers
// must use original geometry for short tracks. No pose or active-list cache
// survives a call: run preparation again whenever inputs change. No barriers.
void prepare_track_timestamp_poses<let Layout : uint>(
    TensorView<float> prepared_poses,
    TensorView<int32_t> prepared_valid,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<int32_t> active_tracks_indices,
    TensorView<int32_t> active_count,
    uint height, uint width, uint groups
) {
    let count = uint(active_count.load(uint(0)));
    let stride = cuda_utils::cuda_block_count() * cudaBlockDim().x;
    for (uint task = cuda_utils::cuda_linear_block_idx() * cudaBlockDim().x + cudaThreadIdx().x;
         task < count * groups; task += stride) {
        // Track-major cache makes neighboring columns contiguous in memory.
        let track = uint(active_tracks_indices.load(task / groups));
        let start = tracks_packinfo.load(uint2(track, 0));
        let n = tracks_packinfo.load(uint2(track, 1));
        prepared_valid[task] = 0;
        // Short tracks retain the original geometry path; avoids preparing poses
        // for the common two-pose static representation.
        if (n <= 2) continue;
        let stamp = rays_timestamps.load(timestamp_cache_representative_ray<Layout>(task % groups, height, width));
        let times = ImmutablePtr<int64_t>(tracks_timestamps.data_ptr_at(start));
        if (stamp < times[0] || stamp > times[n - 1]) continue;
        typedef search::BinarySearch<int64_t, int32_t> Search;
        let ids = search::find_surrounding_indices<Search>(stamp, times, n);
        var pose = transform::SE3.from_vec(
            tracks_poses.loadVec<3>(uint2(start + ids._0, 0)),
            tracks_poses.loadVec<4>(uint2(start + ids._0, 3)));
        if (ids._0 != ids._1) {
            let right = transform::SE3.from_vec(
                tracks_poses.loadVec<3>(uint2(start + ids._1, 0)),
                tracks_poses.loadVec<4>(uint2(start + ids._1, 3)));
            let alpha = (stamp - times[ids._0]) / float(times[ids._1] - times[ids._0]);
            pose = transform::interpolate(pose, right, alpha);
        }
        let value = pose.to_vec();
        prepared_poses.storeVec<3>(uint2(task, 0), value._0);
        prepared_poses.storeVec<4>(uint2(task, 3), value._1);
        prepared_valid[task] = 1;
    }
}

// Cache-backed geometry uses the same dispatcher and handler contract as the
// uncached path. Inputs and scratch obey prepare_track_timestamp_poses contracts.
struct CachedTimestampGeometry<let Layout : uint> : ray_intersect::IActiveTrackGeometry {
    TensorView<float> rays;
    TensorView<int64_t> rays_timestamps;
    TensorView<float> tracks_poses;
    TensorView<int64_t> tracks_timestamps;
    TensorView<int32_t> tracks_packinfo;
    TensorView<float> cuboids_dimensions;
    TensorView<float> prepared_poses;
    TensorView<int32_t> prepared_valid;
    uint height, width, groups;

    Tuple<float2, int64_t> intersect(uint i, uint track, uint active_idx) {
        let stamp = rays_timestamps.load(i);
        let group = timestamp_cache_group<Layout>(i, height, width);
        let task = active_idx * groups + group;
        var hit = float2(-1.0f);
        if (tracks_packinfo.load(uint2(track, 1)) > 2 &&
            stamp == rays_timestamps.load(timestamp_cache_representative_ray<Layout>(group, height, width))) {
            if (prepared_valid.load(task) != 0) {
                let pose = transform::SE3.from_vec(
                    prepared_poses.loadVec<3>(uint2(task, 0)),
                    prepared_poses.loadVec<4>(uint2(task, 3)));
                hit = ray_intersect::LieIntersectionStrategy.intersect_world_ray_with_timed_cuboid(
                    rays.loadVec<3>(uint2(i, 0)), rays.loadVec<3>(uint2(i, 3)),
                    cuboids_dimensions.loadVec<3>(uint2(track, 0)), pose);
            }
        } else {
            hit = ray_intersect::intersect_one_ray_with_one_track_geometry<ray_intersect::LieIntersectionStrategy>(
                i, track, rays, rays_timestamps, tracks_poses, tracks_timestamps,
                tracks_packinfo, cuboids_dimensions)._0;
        }
        return makeTuple(hit, stamp);
    }
};

// Sum exact hit timestamp deltas and counts, using the freshly prepared cache.
// Tensor/layout contracts match prepare_track_timestamp_poses, plus float32
// rays [B*H*W,6], dimensions [T,3], and int64 output [T,2] initialized to zero.
// frame_timestamp is the int64 reference subtracted before integer accumulation.
// Run after preparation on the same stream with unchanged poses, timestamps and
// active list. Every ray must be valid for the supplied grid shape. Timestamp
// mismatches and short tracks use original geometry, then all lanes reconverge
// before sparse warp reduction. Launch with a positive block count and a block
// size divisible by 32. Output uses original track indices; untouched tracks
// retain zero and therefore the caller's existing default timestamp behavior.
void estimate_tracks_timestamps_masked_cached<let Layout : uint>(
    TensorView<int64_t> output,
    TensorView<float> rays,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<float> cuboids_dimensions,
    TensorView<int32_t> active_tracks_indices,
    TensorView<int32_t> active_count,
    no_diff int64_t frame_timestamp,
    TensorView<float> prepared_poses,
    TensorView<int32_t> prepared_valid,
    uint height, uint width, uint groups
) {
    typedef ray_intersect::timestamps_estimation::SparseWaveReduceIntersectionHandler Handler;
    let geometry = CachedTimestampGeometry<Layout>(
        rays, rays_timestamps, tracks_poses, tracks_timestamps, tracks_packinfo,
        cuboids_dimensions, prepared_poses, prepared_valid, height, width, groups);
    ray_intersect::dispatch_active_track_intersections(
        uint(rays.size(0)), active_tracks_indices, active_count,
        geometry, Handler(output, frame_timestamp));
}

// Host-selected rows specialization; see the shared implementation contract above.
[CUDAKernel]
[AutoPyBindCUDA]
void prepare_track_timestamp_poses_rows(
    TensorView<float> prepared_poses,
    TensorView<int32_t> prepared_valid,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<int32_t> active_tracks_indices,
    TensorView<int32_t> active_count,
    uint height, uint width, uint groups
) {
    prepare_track_timestamp_poses<0>(
        prepared_poses, prepared_valid, rays_timestamps, tracks_poses, tracks_timestamps,
        tracks_packinfo, active_tracks_indices, active_count, height, width, groups);
}

// Host-selected columns specialization; see the shared implementation contract above.
[CUDAKernel]
[AutoPyBindCUDA]
void prepare_track_timestamp_poses_columns(
    TensorView<float> prepared_poses,
    TensorView<int32_t> prepared_valid,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<int32_t> active_tracks_indices,
    TensorView<int32_t> active_count,
    uint height, uint width, uint groups
) {
    prepare_track_timestamp_poses<1>(
        prepared_poses, prepared_valid, rays_timestamps, tracks_poses, tracks_timestamps,
        tracks_packinfo, active_tracks_indices, active_count, height, width, groups);
}

// Host-selected global specialization; see the shared implementation contract above.
[CUDAKernel]
[AutoPyBindCUDA]
void prepare_track_timestamp_poses_global(
    TensorView<float> prepared_poses,
    TensorView<int32_t> prepared_valid,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<int32_t> active_tracks_indices,
    TensorView<int32_t> active_count,
    uint height, uint width, uint groups
) {
    prepare_track_timestamp_poses<2>(
        prepared_poses, prepared_valid, rays_timestamps, tracks_poses, tracks_timestamps,
        tracks_packinfo, active_tracks_indices, active_count, height, width, groups);
}

// Host-selected rows specialization; see the shared implementation contract above.
[CUDAKernel]
[AutoPyBindCUDA]
void estimate_tracks_timestamps_masked_cached_rows(
    TensorView<int64_t> output,
    TensorView<float> rays,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<float> cuboids_dimensions,
    TensorView<int32_t> active_tracks_indices,
    TensorView<int32_t> active_count,
    no_diff int64_t frame_timestamp,
    TensorView<float> prepared_poses,
    TensorView<int32_t> prepared_valid,
    uint height, uint width, uint groups
) {
    estimate_tracks_timestamps_masked_cached<0>(
        output, rays, rays_timestamps, tracks_poses, tracks_timestamps, tracks_packinfo,
        cuboids_dimensions, active_tracks_indices, active_count, frame_timestamp, prepared_poses,
        prepared_valid, height, width, groups);
}

// Host-selected columns specialization; see the shared implementation contract above.
[CUDAKernel]
[AutoPyBindCUDA]
void estimate_tracks_timestamps_masked_cached_columns(
    TensorView<int64_t> output,
    TensorView<float> rays,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<float> cuboids_dimensions,
    TensorView<int32_t> active_tracks_indices,
    TensorView<int32_t> active_count,
    no_diff int64_t frame_timestamp,
    TensorView<float> prepared_poses,
    TensorView<int32_t> prepared_valid,
    uint height, uint width, uint groups
) {
    estimate_tracks_timestamps_masked_cached<1>(
        output, rays, rays_timestamps, tracks_poses, tracks_timestamps, tracks_packinfo,
        cuboids_dimensions, active_tracks_indices, active_count, frame_timestamp, prepared_poses,
        prepared_valid, height, width, groups);
}

// Host-selected global specialization; see the shared implementation contract above.
[CUDAKernel]
[AutoPyBindCUDA]
void estimate_tracks_timestamps_masked_cached_global(
    TensorView<int64_t> output,
    TensorView<float> rays,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<float> cuboids_dimensions,
    TensorView<int32_t> active_tracks_indices,
    TensorView<int32_t> active_count,
    no_diff int64_t frame_timestamp,
    TensorView<float> prepared_poses,
    TensorView<int32_t> prepared_valid,
    uint height, uint width, uint groups
) {
    estimate_tracks_timestamps_masked_cached<2>(
        output, rays, rays_timestamps, tracks_poses, tracks_timestamps, tracks_packinfo,
        cuboids_dimensions, active_tracks_indices, active_count, frame_timestamp, prepared_poses,
        prepared_valid, height, width, groups);
}
