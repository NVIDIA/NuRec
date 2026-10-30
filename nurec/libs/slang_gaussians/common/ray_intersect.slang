// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <slang_utils/extensions.slang>
#include <slang_utils/cuda.slang>
#include <geometry/kernels/transform.slang>
#include <slang_gaussians/common/search.slang>

// This file contains the implementation of ray intersection with timed cuboids.
//
// It provides different intersection strategies with different benefits.
//
// It provides intersection handlers which compute the average timestamp of the
// rays intersecting with a cuboid track.

namespace ray_intersect {

    // This interface defines how to intersect a world ray with a timed cuboid.
interface IIntersectionStrategy {
    // Intersection when the pose needs to be interpolated.
    static float2 intersect_world_ray_with_timed_cuboid(
        float3 ray_origin_world,
        float3 ray_direction_world,
        float alpha,
        float3 cuboid_dimensions,
        transform::SE3 interpolation_start_pose,
        transform::SE3 interpolation_end_pose,
    );

    // Intersection when the pose is directly available.
    static float2 intersect_world_ray_with_timed_cuboid(
        float3 ray_origin_world,
        float3 ray_direction_world,
        float3 cuboid_dimensions,
        transform::SE3 interpolation_pose,
    );
};

// This interface defines what to do when a ray intersects with a cuboid track.
//
// For the time being, the only thing needed is the ray timestamp, but we can
// pass more of other ray intersecting algorithm need things like the ray id, etc.
interface IIntersectionHandler {
    void handle_intersection(
        float2 t1t2,
        uint track_idx,
        int64_t ray_timestamp,
    );
};

// Per-(ray, track) work shared by `intersect_rays_with_timed_cuboids` and
// `intersect_rays_with_active_timed_cuboids`.  Loads the track keyframes,
// finds the two surrounding poses for the ray's timestamp, runs the
// strategy-specific ray/cuboid intersection, and returns its interval and
// timestamp. Empty tracks and timestamps outside the keyframe range are misses.
// This function deliberately contains no warp operations.
//
// Inputs are CUDA tensor views: rays [R, 6] float32 (origin, direction),
// rays_timestamps [R] int64, tracks_poses [P, 7] float32 (translation, xyzw),
// tracks_timestamps [P] int64, tracks_packinfo [T, 2] int32 (start, length),
// and cuboids_dimensions [T, 3] float32. Vector components and each track's
// timestamp slice must be contiguous; timestamps must be sorted within tracks.
// ray_idx and track_idx must be valid indices, and packed slices must fit P.
// Returns the strategy's ray interval and input timestamp; misses use (-1, -1).
// No bounds validation or exceptions are provided by this device helper.
Tuple<float2, int64_t> intersect_one_ray_with_one_track_geometry<
    IntersectionStrategy : IIntersectionStrategy
    > (
    uint ray_idx,
    uint track_idx,
    TensorView<float> rays,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<float> cuboids_dimensions,
) {
    let track_start = tracks_packinfo.load(uint2(track_idx, 0));
    let track_length = tracks_packinfo.load(uint2(track_idx, 1));

    // The original CUDA implementation was using shared memory and copied the track
    // timestamps to it.  We couldn't see performance improvements by doing this and
    // it forced us to know the maximum track length in advance.  Therefore we just
    // read from global memory, which is likely to be cached in the most cases anyways.
    let track_timestamps = ImmutablePtr<int64_t>(tracks_timestamps.data_ptr_at(track_start));

    let ray_timestamp = rays_timestamps.load(ray_idx);

    if (track_length <= 0 || ray_timestamp < track_timestamps[0] || ray_timestamp > track_timestamps[track_length - 1]) {
        return makeTuple(float2(-1.0f), ray_timestamp);
    }

    typedef search::BinarySearch<int64_t, int32_t> SearchStrategy;
    let indices = search::find_surrounding_indices<SearchStrategy>(ray_timestamp, track_timestamps, track_length);

    let cuboid_dimensions = cuboids_dimensions.loadVec<3>(uint2(track_idx, 0));

    let ray_origin = rays.loadVec<3>(uint2(ray_idx, 0));
    let ray_direction = rays.loadVec<3>(uint2(ray_idx, 3));

    float2 t1t2;

    // We already checked that track is not empty, so we are guaranteed to have indices in range.
    if (indices._0 == indices._1) {
        let interpolation_pose = transform::SE3::from_vec(
            tracks_poses.loadVec<3>(uint2(track_start + indices._0, 0)),
            tracks_poses.loadVec<4>(uint2(track_start + indices._0, 3)),
        );
        t1t2 = IntersectionStrategy.intersect_world_ray_with_timed_cuboid(
            ray_origin,
            ray_direction,
            cuboid_dimensions,
            interpolation_pose,
        );
    }
    else {
        let interpolation_start_idx = indices._0;
        let interpolation_end_idx = indices._1;

        let interpolation_start_timestamp = track_timestamps[interpolation_start_idx];
        let interpolation_end_timestamp = track_timestamps[interpolation_end_idx];
        let interpolation_alpha = (ray_timestamp - interpolation_start_timestamp) / float(interpolation_end_timestamp - interpolation_start_timestamp);

        let interpolation_start_pose = transform::SE3::from_vec(
            tracks_poses.loadVec<3>(uint2(track_start + interpolation_start_idx, 0)),
            tracks_poses.loadVec<4>(uint2(track_start + interpolation_start_idx, 3)),
        );
        let interpolation_end_pose = transform::SE3::from_vec(
            tracks_poses.loadVec<3>(uint2(track_start + interpolation_end_idx, 0)),
            tracks_poses.loadVec<4>(uint2(track_start + interpolation_end_idx, 3)),
        );

        t1t2 = IntersectionStrategy.intersect_world_ray_with_timed_cuboid(
            ray_origin,
            ray_direction,
            interpolation_alpha,
            cuboid_dimensions,
            interpolation_start_pose,
            interpolation_end_pose
        );
    }

    return makeTuple(t1t2, ray_timestamp);
}


// Dispatch the geometry result to the handler, including misses. Tensor inputs
// match intersect_one_ray_with_one_track_geometry; ray_idx beyond R is a miss.
// Returns no value; side effects and lane participation requirements belong to
// the handler. All lanes participating in a wave must use the same track_idx.
void intersect_one_ray_with_one_track<
    IntersectionStrategy : IIntersectionStrategy,
    IntersectionHandler : IIntersectionHandler
    > (
    uint ray_idx,
    uint track_idx,
    TensorView<float> rays,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<float> cuboids_dimensions,
    IntersectionHandler intersection_handler,
) {
    // All lanes call the handler, including tail lanes and out-of-range times.
    // Keep geometry divergent and reconverge before any handler barriers.
    var result = makeTuple(float2(-1.0f), int64_t(0));
    if (ray_idx < uint(rays.size(0))) {
        result = intersect_one_ray_with_one_track_geometry<IntersectionStrategy>(
            ray_idx, track_idx, rays, rays_timestamps, tracks_poses, tracks_timestamps,
            tracks_packinfo, cuboids_dimensions,
        );
    }
    intersection_handler.handle_intersection(result._0, track_idx, result._1);
}


// This is the generic, non-differentiable implementation of the ray intersection algorithm.
//
// Grid axes are intentionally swapped from the natural (rays, tracks) layout.
// With blockIdx.x = track_idx (fastest-varying) and blockIdx.y = ray_idx_block,
// each SM gets a contiguous slab of blocks that share the same ray range and
// sweep across tracks.  That keeps `rays` / `rays_timestamps` warm in L1
// across the track dimension; per-track invariants are small enough to fit
// L1 even without cross-block reuse.  Callers must launch with grid =
// (nb_tracks, num_ray_blocks, 1).
void intersect_rays_with_timed_cuboids<
    IntersectionStrategy : IIntersectionStrategy,
    IntersectionHandler : IIntersectionHandler
    > (
    TensorView<float> rays,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<float> cuboids_dimensions,
    IntersectionHandler intersection_handler,
) {
    let nb_tracks = tracks_packinfo.size(0);
    let track_idx = cudaBlockIdx().x;
    if (track_idx >= nb_tracks) {
        return;
    }
    let ray_idx = cudaBlockIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

    intersect_one_ray_with_one_track<IntersectionStrategy, IntersectionHandler>(
        ray_idx,
        track_idx,
        rays,
        rays_timestamps,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        cuboids_dimensions,
        intersection_handler,
    );
}


// Geometry callbacks contain no wave operations or block barriers. The active
// slot selects optional per-active-track scratch; track_idx indexes real outputs.
interface IActiveTrackGeometry {
    Tuple<float2, int64_t> intersect(uint ray_idx, uint track_idx, uint active_idx);
};

// Uncached geometry adapter. Tensor contracts match the geometry helper above.
struct TimedTrackGeometry<IntersectionStrategy : IIntersectionStrategy> : IActiveTrackGeometry {
    TensorView<float> rays;
    TensorView<int64_t> rays_timestamps;
    TensorView<float> tracks_poses;
    TensorView<int64_t> tracks_timestamps;
    TensorView<int32_t> tracks_packinfo;
    TensorView<float> cuboids_dimensions;

    Tuple<float2, int64_t> intersect(uint ray_idx, uint track_idx, uint active_idx) {
        return intersect_one_ray_with_one_track_geometry<IntersectionStrategy>(
            ray_idx, track_idx, rays, rays_timestamps, tracks_poses, tracks_timestamps,
            tracks_packinfo, cuboids_dimensions);
    }
};

// Shared bounded dispatch for cached and uncached geometry. Active tracks vary
// fastest; device count avoids host synchronization. 1-D and flattened 2-D grids
// are supported. Every lane makes the same handler calls, with synthetic misses
// for tail rays, so both warp-only and block-barrier handlers are valid.
// active indices/count are CUDA int32 [T]/[1], count in [0,T], with valid unique
// track indices. Outputs use real track indices: unlisted tracks stay untouched;
// callers must zero-initialize accumulation buffers as appropriate.
void dispatch_active_track_intersections<
    Geometry : IActiveTrackGeometry,
    IntersectionHandler : IIntersectionHandler
    > (
    uint nb_rays,
    TensorView<int32_t> active_tracks_indices,
    TensorView<int32_t> active_count,
    Geometry geometry,
    IntersectionHandler intersection_handler,
) {
    let nb_active_tracks = uint(active_count.load(uint(0)));
    if (nb_active_tracks == 0) return;
    let ray_blocks = (nb_rays + cudaBlockDim().x - 1) / cudaBlockDim().x;
    let work_count = nb_active_tracks * ray_blocks;
    for (uint work = cuda_utils::cuda_linear_block_idx(); work < work_count; work += cuda_utils::cuda_block_count()) {
        let active_idx = work % nb_active_tracks;
        let signed_track_idx = active_tracks_indices.load(active_idx);
        cuda_utils::cuda_device_assert(signed_track_idx >= 0);
        let track_idx = uint(signed_track_idx);
        let ray_idx = (work / nb_active_tracks) * cudaBlockDim().x + cudaThreadIdx().x;
        var result = makeTuple(float2(-1.0f), int64_t(0));
        if (ray_idx < nb_rays) {
            result = geometry.intersect(ray_idx, track_idx, active_idx);
        }
        intersection_handler.handle_intersection(result._0, track_idx, result._1);
    }
}

// Uncached entry point; tensor contracts match intersect_one_ray_with_one_track_geometry.
void intersect_rays_with_active_timed_cuboids<
    IntersectionStrategy : IIntersectionStrategy,
    IntersectionHandler : IIntersectionHandler
    > (
    TensorView<float> rays,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<float> cuboids_dimensions,
    TensorView<int32_t> active_tracks_indices,
    TensorView<int32_t> active_count,
    IntersectionHandler intersection_handler,
) {
    let geometry = TimedTrackGeometry<IntersectionStrategy>(
        rays, rays_timestamps, tracks_poses, tracks_timestamps, tracks_packinfo, cuboids_dimensions);
    dispatch_active_track_intersections(
        uint(rays.size(0)), active_tracks_indices, active_count, geometry, intersection_handler);
}


// Intersect a ray with an axis-aligned bounding box.
// It takes the inverse of the ray direction to avoid computing it
// several times when intersecting the same ray with multiples objects.
static float2 ray_aabb_intersect(
    float3 ray_origin,
    float3 inv_ray_direction,
    float3 aabb_min,
    const float3 aabb_max
) {
    let t_min = (aabb_min - ray_origin) * inv_ray_direction;
    let t_max = (aabb_max - ray_origin) * inv_ray_direction;

    let _t1 = min(t_min, t_max);
    let _t2 = max(t_min, t_max);
    let t1  = max(max(_t1.x, _t1.y), _t1.z);
    let t2  = min(min(_t2.x, _t2.y), _t2.z);

    if (t1 > t2) {
        return float2(-1.0f);
    }
    return float2(t1, t2);
}


// This intersection strategy matches the original CUDA implementation.
// It uses quaternion slerp to interpolate the rotations and convert it
// to a rotation matrix before applying it to the ray.
//
// This approach is slightly faster than using the SE3 class.
struct QuatSlerpMatrixIntersectionStrategy : IIntersectionStrategy {
    static float2 intersect_world_ray_with_timed_cuboid(
        float3 ray_origin_world,
        float3 ray_direction_world,
        float alpha,
        float3 cuboid_dimensions,
        transform::SE3 interpolation_start_pose,
        transform::SE3 interpolation_end_pose,
    ) {
        let ray_o_world = ray_origin_world;
        let ray_d_world = ray_direction_world;
        let pose_start = interpolation_start_pose.to_vec();
        let c_start = pose_start._0;
        let q_start = pose_start._1;
        let pose_end = interpolation_end_pose.to_vec();
        let c_end = pose_end._0;
        let q_end = pose_end._1;

        let c_raytime = (1.0f - alpha) * c_start + alpha * c_end;
        let R_raytime = transpose(quaternion::convertNormalizedQuatToMatrix(quaternion::slerp(q_start, q_end, alpha)));

        let ray_o_box = mul(R_raytime, ray_o_world - c_raytime);
        let ray_d_box = mul(R_raytime, ray_d_world);

        let aabb_max = cuboid_dimensions * 0.5;
        let aabb_min = -aabb_max;
        let t1t2 = ray_aabb_intersect(ray_o_box, 1.0f / ray_d_box, aabb_min, aabb_max);

        return t1t2;
    }

    static float2 intersect_world_ray_with_timed_cuboid(
        float3 ray_origin_world,
        float3 ray_direction_world,
        float3 cuboid_dimensions,
        transform::SE3 interpolation_pose,
    ) {
        let ray_o_world = ray_origin_world;
        let ray_d_world = ray_direction_world;
        let pose_raytime = interpolation_pose.to_vec();
        let c_raytime = pose_raytime._0;
        let q_raytime = transpose(quaternion::convertNormalizedQuatToMatrix(pose_raytime._1));

        let ray_o_box = mul(q_raytime, ray_o_world - c_raytime);
        let ray_d_box = mul(q_raytime, ray_d_world);

        let aabb_max = cuboid_dimensions * 0.5;
        let aabb_min = -aabb_max;
        let t1t2 = ray_aabb_intersect(ray_o_box, 1.0f / ray_d_box, aabb_min, aabb_max);

        return t1t2;
    }
};

// This intersection strategy uses Lie algebra to interpolate the rotation.
// It matches the rest of the tracks interpolation approach.
struct LieIntersectionStrategy : IIntersectionStrategy {
    static float2 intersect_world_ray_with_timed_cuboid(
        float3 ray_origin_world,
        float3 ray_direction_world,
        float alpha,
        float3 cuboid_dimensions,
        transform::SE3 interpolation_start_pose,
        transform::SE3 interpolation_end_pose,
    ) {
        let ray_time_pose = transform::interpolate(interpolation_start_pose, interpolation_end_pose, alpha);
        return intersect_world_ray_with_timed_cuboid(
            ray_origin_world, ray_direction_world, cuboid_dimensions, ray_time_pose
        );
    }

    static float2 intersect_world_ray_with_timed_cuboid(
        float3 ray_origin_world,
        float3 ray_direction_world,
        float3 cuboid_dimensions,
        transform::SE3 interpolation_pose,
    ) {
        let ray_time_pose = interpolation_pose;
        let ray_time_translation = ray_time_pose.translation();
        let ray_time_rotation = ray_time_pose.so3.inv();

        let ray_origin_box = ray_time_rotation * (transform::Point(ray_origin_world) - ray_time_translation);
        let ray_direction_box = ray_time_rotation * transform::Point(ray_direction_world);

        let aabb_max = cuboid_dimensions * 0.5;
        let aabb_min = -aabb_max;

        let t1t2 = ray_aabb_intersect(ray_origin_box.p, 1.0f / ray_direction_box.p, aabb_min, aabb_max);

        return t1t2;
    }
};


namespace timestamps_estimation {

//
// The following intersection handlers are used to sum the timestamps of the rays
// intersecting with a cuboid track.
//
// They offer different performance trade-offs dependening on how much contention
// there is on the global memory, i.e. how many rays are intersecting with the same track.
//

// To avoid overflows, the intersection handlers add the difference between the ray timestamp
// and the frame timestamp.  By accumulating deltas instead of absolute values, we don't overflow
// in 64 bits.
//
// We could do addition on more than 64 bits or use min / max to be even more sure not to overflow,
// but this is enough for now.

// This handler simply atomically adds the delta to the result and increment the count
// using atomic additions to global memory.
struct AtomicAddIntersectionHandler : IIntersectionHandler {
    TensorView<int64_t> output_tracks_timestamps_sum_and_count;
    int64_t frame_timestamp;

    void handle_intersection(
        float2 t1t2,
        uint track_idx,
        int64_t ray_timestamp,
    ) {
        if (t1t2.y > 0) {
            int64_t old_value;
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 0), ray_timestamp - frame_timestamp, old_value);
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 1), 1, old_value);
        }
    }
}

// This handler performs a wave reduction to sum the deltas and the count
// without resorting to atomic operations, potentially dividing the number of
// atomic operations by 32.
//
// However, when there is very low contention, this approach actually adds overhead
// over the naive one, but because it always performs the (fast) reduction, which
// would be slower than doing almost no atomic operations.
struct WaveReduceIntersectionHandler : IIntersectionHandler {
    TensorView<int64_t> output_tracks_timestamps_sum_and_count;
    int64_t frame_timestamp;

    void handle_intersection(
        float2 t1t2,
        uint track_idx,
        int64_t ray_timestamp,
    ) {
        int64_t output_delta;
        int32_t count;
        if (t1t2.y > 0) {
            output_delta = ray_timestamp - frame_timestamp;
            count = 1;
        }
        else {
            output_delta = 0;
            count = 0;
        }
        output_delta = WaveActiveSum(output_delta);
        count = WaveActiveSum(count);
        if (WaveIsFirstLane() && count > 0) {
            int64_t old_value;
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 0), output_delta, old_value);
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 1), count, old_value);
        }
    }
}

// Empty warps skip the reduction entirely, while
// dense hits still produce at most two global atomics per warp. No block
// barriers are needed, including when a track's time range excludes lane zero.
struct SparseWaveReduceIntersectionHandler : IIntersectionHandler {
    TensorView<int64_t> output_tracks_timestamps_sum_and_count;
    int64_t frame_timestamp;

    void handle_intersection(
        float2 t1t2,
        uint track_idx,
        int64_t ray_timestamp,
    ) {
        let hit = t1t2.y > 0;
        if (!WaveActiveAnyTrue(hit)) {
            return;
        }
        let output_delta = WaveActiveSum(hit ? ray_timestamp - frame_timestamp : int64_t(0));
        let count = WaveActiveCountBits(hit);
        if (WaveIsFirstLane()) {
            int64_t old_value;
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 0), output_delta, old_value);
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 1), int64_t(count), old_value);
        }
    }
}

// This handler uses shared memory to accumulate the deltas and the count
// so the atomic operations are performed on the shared memory instead of the
// global memory.
//
// This approach can also be slower than the naive one when there is very low
// contention, because we go to shared memory before going to global memory,
// which is slower than going directly to global memory when there are near-zero
// operations to do.
struct SharedAccumulatorsIntersectionHandler : IIntersectionHandler {
    TensorView<int64_t> output_tracks_timestamps_sum_and_count;
    int64_t frame_timestamp;

    void handle_intersection(
        float2 t1t2,
        uint track_idx,
        int64_t ray_timestamp,
    ) {
        static groupshared int64_t shared_accumulators[2];
        if (cudaThreadIdx().x == 0) {
            shared_accumulators[0] = 0;
            shared_accumulators[1] = 0;
        }
        GroupMemoryBarrierWithGroupSync();

        if (t1t2.y > 0) {
            InterlockedAdd(shared_accumulators[0], ray_timestamp - frame_timestamp);
            InterlockedAdd(shared_accumulators[1], 1);
        }
        GroupMemoryBarrierWithGroupSync();
        if (cudaThreadIdx().x == 0 && shared_accumulators[1] > 0) {
            int64_t old_value;
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 0), shared_accumulators[0], old_value);
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 1), shared_accumulators[1], old_value);
        }
    }
}

// This handler combines the wave reduction and the shared memory approach,
// performing warp reduction to avoid atomic operations, then using shared
// memory to do the reduction at the block level using fewer atomic operations
// and doing them on shared memory, resulting in a single atomic operation per
// block on global memory.
struct WaveReduceSharedAccumulatorsIntersectionHandler : IIntersectionHandler {
    TensorView<int64_t> output_tracks_timestamps_sum_and_count;
    int64_t frame_timestamp;

    void handle_intersection(
        float2 t1t2,
        uint track_idx,
        int64_t ray_timestamp,
    ) {
        static groupshared int64_t shared_accumulators[2];
        if (cudaThreadIdx().x == 0) {
            shared_accumulators[0] = 0;
            shared_accumulators[1] = 0;
        }
        GroupMemoryBarrierWithGroupSync();

        int64_t output_delta;
        int32_t count;
        if (t1t2.y > 0) {
            output_delta = ray_timestamp - frame_timestamp;
            count = 1;
        }
        else {
            output_delta = 0;
            count = 0;
        }
        output_delta = WaveActiveSum(output_delta);
        count = WaveActiveSum(count);
        if (WaveIsFirstLane() && count > 0) {
            InterlockedAdd(shared_accumulators[0], output_delta);
            InterlockedAdd(shared_accumulators[1], count);
        }
        GroupMemoryBarrierWithGroupSync();
        if (cudaThreadIdx().x == 0 && shared_accumulators[1] > 0) {
            int64_t old_value;
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 0), shared_accumulators[0], old_value);
            output_tracks_timestamps_sum_and_count.InterlockedAdd(uint2(track_idx, 1), shared_accumulators[1], old_value);
        }
    }
}

}; // namespace timestamps_estimation

}; // namespace ray_intersect
