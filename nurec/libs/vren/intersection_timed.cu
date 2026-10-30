// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "intersection.cuh"

#include "utils.h"

#include <ku/helper_math.cuh>

#include <ku/binary_search.cuh>
#include <ku/common.cuh>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <tuple>

template <typename BBoxTimed>
inline __device__ auto worldray_bboxtimed_intersection(float3 const& ray_o_world, float3 const& ray_d_world,
                                                       float ray_t, BBoxTimed const& bboxtimed) {
    // grab timed bbox with start/end bbox->world pose
    auto const dim     = make_float3(bboxtimed[0], bboxtimed[1], bboxtimed[2]);
    auto const c_start = make_float3(bboxtimed[3], bboxtimed[4], bboxtimed[5]);
    auto const q_start = make_float4(bboxtimed[6], bboxtimed[7], bboxtimed[8], bboxtimed[9]);
    auto const c_end   = make_float3(bboxtimed[10], bboxtimed[11], bboxtimed[12]);
    auto const q_end   = make_float4(bboxtimed[13], bboxtimed[14], bboxtimed[15], bboxtimed[16]);

    // interpolate bbox to ray time and create world->bbox transformation
    auto const c_raytime = (1.f - ray_t) * c_start + ray_t * c_end;
    auto const R_raytime = transpose_matrix(unitquat_rotmatrix(unitquat_slerp(q_start, q_end, ray_t)));

    // transform world-ray to bbox-ray
    auto const ray_o_bbox = apply_matrix(R_raytime, ray_o_world - c_raytime);
    auto const ray_d_bbox = apply_matrix(R_raytime, ray_d_world);

    // perform intersection with aabb at origin
    auto const aabb_min = make_float3(-dim.x / 2.f, -dim.y / 2.f, -dim.z / 2.f);
    auto const aabb_max = make_float3(dim.x / 2.f, dim.y / 2.f, dim.z / 2.f);
    return ray_aabb_intersect(ray_o_bbox, 1.f / ray_d_bbox, aabb_min, aabb_max);
}

inline __device__ float rolling_shutter_time(int16_t const i, int16_t const j,
                                             int32_t const w, int32_t const h,
                                             int32_t const shutter_type) {
    // computes relative frame time [0,1] given pixel index, image resolution, and rolling-shutter type

    switch (shutter_type) {
    case 1: // ROLLING_TOP_TO_BOTTOM
        return float(j) / (h - 1);
    case 2: // ROLLING_LEFT_TO_RIGHT
        return float(i) / (w - 1);
    case 3: // ROLLING_BOTTOM_TO_TOP
        return (h - 1 - float(j)) / (h - 1);
    case 4: // ROLLING_RIGHT_TO_LEFT
        return (w - 1 - float(i)) / (w - 1);
    }

    return 0.f; // GLOBAL / fallback - stick to time of first pose
}

inline __device__ auto rolling_shutter_camera_ray_to_world(
    int16_t const pixel_i,
    int16_t const pixel_j,
    float3 const camera_ray, // 3d ray direction in camera frame
    float3 const t_start,    // camera-to-world at start of frame [t, q]
    float4 const q_start,
    float3 const t_end, // camera-to-world at end of frame [t, q]
    float4 const q_end,
    int32_t const w,
    int32_t const h,
    int32_t const shutter_type) {
    // determine relative frame time from shutter type and pixel location
    auto const relative_frame_time = rolling_shutter_time(pixel_i, pixel_j, w, h, shutter_type);

    // rolling-shutter sensor->world pose for the ray
    auto const t_raytime = (1.f - relative_frame_time) * t_start + relative_frame_time * t_end;
    auto const R_raytime = unitquat_rotmatrix(unitquat_slerp(q_start, q_end, relative_frame_time));

    float3 const ray_o_world = t_raytime;
    float3 const ray_d_world = apply_matrix(R_raytime, camera_ray);

    return std::make_tuple(ray_o_world, ray_d_world, relative_frame_time);
}

// Kernel / entry points for CuboidTracks APIs
template <typename scalar_t>
__global__ void ray_cuboidtracks_intersection_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const rays_o,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const rays_d,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const rays_timestamps_us,

    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> const tracks_packinfo,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const tracks_poses,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const tracks_timestamps_us,

    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const cuboids_dims,

    int32_t const max_intersections_per_ray,

    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> intersections_cnt,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> intersections_ts,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> intersections_tracks_idx,
    bool const with_intersections_ts) {

    auto const ray_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    auto const track_idx = blockIdx.y;

    if (track_idx >= tracks_packinfo.size(0))
        // track of block does not exist
        return;

    auto const track_packinfo  = tracks_packinfo[track_idx];
    auto const track_start_idx = track_packinfo[0];
    auto const N_track_poses   = track_packinfo[1];
    auto const N_total_poses   = tracks_poses.size(0);

    if (N_track_poses <= 1)
        // track has no poses
        return;

    // load the track's timestamps into shared memory to accelerate binary search
    // using all threads (irrespective of whether the local ray exists)
    extern __shared__ int64_t shared_track_timestamps_us[];

    for (auto i = threadIdx.x; i < N_track_poses; i += blockDim.x)
        shared_track_timestamps_us[i] = tracks_timestamps_us[track_start_idx + i];

    __syncthreads(); // all timestamps of the block-associated track are now loaded to shared memory

    if (ray_idx >= rays_o.size(0))
        // local ray doesn't exist
        return;

    // check if ray's timestamp is in the track's valid time-range and exit early if not
    auto const ray_timestamp_us = rays_timestamps_us[ray_idx];

    if (ray_timestamp_us < shared_track_timestamps_us[0] ||
        shared_track_timestamps_us[N_track_poses - 1] < ray_timestamp_us)
        return;

    // ray's time is in the tracks' valid time-range AND we have N_track_poses > 1 -> determine track poses for linear interpolation
    auto const interpolation_end_idx   = binary_search_interp(ray_timestamp_us, shared_track_timestamps_us, N_track_poses);
    auto const interpolation_start_idx = interpolation_end_idx - 1; // note: this is guaranteed to be in the valid range of poses due to the check above

    auto const local_interpolation_start_timestamp_us = shared_track_timestamps_us[interpolation_start_idx],
               local_interpolation_end_timestamp_us   = shared_track_timestamps_us[interpolation_end_idx];

    // linear interpolation parameter [0,1]
    auto const t_interp = scalar_t(ray_timestamp_us - local_interpolation_start_timestamp_us) / scalar_t(local_interpolation_end_timestamp_us - local_interpolation_start_timestamp_us);

    auto const local_interpolation_start_pose = tracks_poses[track_start_idx + interpolation_start_idx],
               local_interpolation_end_pose   = tracks_poses[track_start_idx + interpolation_end_idx];

    // collect timed-cuboid data for intersection computation
    auto const cuboidtimed = std::array<scalar_t, 17>{cuboids_dims[track_idx][0],
                                                      cuboids_dims[track_idx][1],
                                                      cuboids_dims[track_idx][2],

                                                      local_interpolation_start_pose[0],
                                                      local_interpolation_start_pose[1],
                                                      local_interpolation_start_pose[2],
                                                      local_interpolation_start_pose[3],
                                                      local_interpolation_start_pose[4],
                                                      local_interpolation_start_pose[5],
                                                      local_interpolation_start_pose[6],

                                                      local_interpolation_end_pose[0],
                                                      local_interpolation_end_pose[1],
                                                      local_interpolation_end_pose[2],
                                                      local_interpolation_end_pose[3],
                                                      local_interpolation_end_pose[4],
                                                      local_interpolation_end_pose[5],
                                                      local_interpolation_end_pose[6]};

    // grab timed ray
    auto const ray_o_world = make_float3(rays_o[ray_idx][0], rays_o[ray_idx][1], rays_o[ray_idx][2]);
    auto const ray_d_world = make_float3(rays_d[ray_idx][0], rays_d[ray_idx][1], rays_d[ray_idx][2]);

    // perform time-based cuboid intersection
    auto const t1t2 = worldray_bboxtimed_intersection(ray_o_world, ray_d_world, t_interp, cuboidtimed);

    if (t1t2.y > 0.f) { // if ray intersects the cuboid
        auto const cnt = atomicAdd(&intersections_cnt[ray_idx], 1);
        if (cnt < max_intersections_per_ray) {
            if (with_intersections_ts) {
                intersections_ts[ray_idx][cnt][0] = fmaxf(t1t2.x, 0.f);
                intersections_ts[ray_idx][cnt][1] = t1t2.y;
            }
            intersections_tracks_idx[ray_idx][cnt] = track_idx;
        }
    }
}

std::vector<torch::Tensor> ray_cuboidtracks_intersection_cu(
    torch::Tensor const rays_o,             // N_rays x 3 (3d world positions)
    torch::Tensor const rays_d,             // N_rays x 3 (normalized 3d world directions)
    torch::Tensor const rays_timestamps_us, // N_rays (per ray timestamp)

    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps

    torch::Tensor const cuboids_dims, // (N_tracks x 3) cuboid x/y/z extents (in local track frame)

    int32_t const max_track_n_poses,
    int32_t const max_intersections_per_ray,
    bool const with_intersections_ts) {

    auto rays_o_arg             = torch::TensorArg{rays_o, "rays_o", 1};
    auto rays_d_arg             = torch::TensorArg{rays_d, "rays_d", 2};
    auto rays_timestamps_us_arg = torch::TensorArg{rays_timestamps_us, "rays_timestamps_us", 3};

    auto tracks_packinfo_arg      = torch::TensorArg{tracks_packinfo, "tracks_packinfo", 4};
    auto tracks_poses_arg         = torch::TensorArg{tracks_poses, "tracks_poses", 5};
    auto tracks_timestamps_us_arg = torch::TensorArg{tracks_timestamps_us, "tracks_timestamps_us", 6};

    auto cuboids_dims_arg = torch::TensorArg{cuboids_dims, "cuboids_dims", 7};

    torch::checkAllSameType(__func__, {rays_o_arg, rays_d_arg, tracks_poses_arg, cuboids_dims_arg});
    torch::checkScalarType(__func__, rays_timestamps_us_arg, torch::kLong);
    torch::checkScalarType(__func__, tracks_packinfo_arg, torch::kInt32);
    torch::checkScalarType(__func__, tracks_timestamps_us_arg, torch::kLong);

    torch::checkAllSameGPU(__func__, {rays_o_arg, rays_d_arg, rays_timestamps_us_arg,
                                      tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg,
                                      cuboids_dims_arg});
    torch::checkAllContiguous(__func__, {rays_o_arg, rays_d_arg, rays_timestamps_us_arg,
                                         tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg,
                                         cuboids_dims_arg});

    auto const N_rays = rays_o.size(0), N_tracks = tracks_packinfo.size(0), N_total_poses = tracks_poses.size(0);

    torch::checkSize(__func__, rays_o_arg, {N_rays, 3});
    torch::checkSize(__func__, rays_d_arg, {N_rays, 3});
    torch::checkSize(__func__, rays_timestamps_us_arg, {N_rays});

    torch::checkSize(__func__, tracks_packinfo_arg, {N_tracks, 2});
    torch::checkSize(__func__, tracks_poses_arg, {N_total_poses, 7});
    torch::checkSize(__func__, tracks_timestamps_us_arg, {N_total_poses});

    torch::checkSize(__func__, cuboids_dims_arg, {N_tracks, 3});

    auto intersections_cnt = torch::zeros({N_rays}, torch::dtype(torch::kInt32).device(rays_o.device()));
    // If not needed put some arbitrary tensor instead
    auto intersections_ts         = with_intersections_ts ? torch::full({N_rays, max_intersections_per_ray, 2}, -1., rays_o.options()) : torch::empty({0, 0, 0}, rays_o.options());
    auto intersections_tracks_idx = torch::full({N_rays, max_intersections_per_ray}, -1, torch::dtype(torch::kInt32).device(rays_o.device()));

    if (N_rays == 0 || max_track_n_poses == 0) {
        if (with_intersections_ts) {
            return {intersections_cnt, intersections_tracks_idx, intersections_ts};
        } else {
            return {intersections_cnt, intersections_tracks_idx};
        }
    }

    auto const threads = 256l; // N threads cooperate in processing a single track within each block
    auto const blocks  = dim3((N_rays + threads - 1) / threads, N_tracks);

    // Ensure entire function runs on the same stream and device as data
    c10::cuda::CUDAGuard device_guard(rays_o.device());
    auto const current_stream = c10::cuda::getCurrentCUDAStream();
    c10::cuda::CUDAStreamGuard stream_guard(current_stream);
    auto const stream = current_stream.stream();

    auto const smem = max_track_n_poses * sizeof(int64_t); // allocate enought shared memory for all timestamps of the longest track

    AT_DISPATCH_FLOATING_TYPES(rays_o.scalar_type(), "ray_cuboidtracks_intersection_cu", ([&] {
                                   ray_cuboidtracks_intersection_kernel<<<blocks, threads, smem, stream>>>(
                                       rays_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       rays_d.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       rays_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),

                                       tracks_packinfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),

                                       cuboids_dims.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),

                                       max_intersections_per_ray,

                                       intersections_cnt.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                       intersections_ts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                       intersections_tracks_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(), with_intersections_ts);
                               }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (with_intersections_ts) {
        return {intersections_cnt, intersections_tracks_idx, intersections_ts};
    } else {
        return {intersections_cnt, intersections_tracks_idx};
    }
}

template <typename scalar_t>
__global__ void point_cuboidtracks_intersection_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const points,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const timestamps_us,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> const tracks_packinfo,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const tracks_poses,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const tracks_timestamps_us,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const cuboids_dims,
    torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> point_cuboidtracks_intersection_mask,
    bool const return_dense_mask) {

    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto const track_idx = blockIdx.y;

    if (track_idx >= tracks_packinfo.size(0)) {
        return;
    }

    auto const track_packinfo  = tracks_packinfo[track_idx];
    auto const track_start_idx = track_packinfo[0];
    auto const N_track_poses   = track_packinfo[1];
    auto const N_total_poses   = tracks_poses.size(0);

    if (N_track_poses <= 1) {
        // track has no poses
        return;
    }

    // Load track timestamps into shared memory
    extern __shared__ int64_t shared_track_timestamps_us[];

    for (auto i = threadIdx.x; i < N_track_poses; i += blockDim.x)
        shared_track_timestamps_us[i] = tracks_timestamps_us[track_start_idx + i];

    __syncthreads();

    // Once the loading is done, exit if point index is out of bounds
    if (point_idx >= points.size(0))
        return;

    // Get point timestamp (handle both per-point and global timestamp cases)
    auto const point_timestamp_us = timestamps_us.size(0) == 1 ? timestamps_us[0] : timestamps_us[point_idx];
    auto const point_world        = make_float3(points[point_idx][0], points[point_idx][1], points[point_idx][2]);

    // Check if point's timestamp is in track's time range and exit early if not
    if (point_timestamp_us < shared_track_timestamps_us[0] ||
        shared_track_timestamps_us[N_track_poses - 1] < point_timestamp_us)
        return;

    // Find interpolation poses
    auto const interpolation_end_idx   = binary_search_interp(point_timestamp_us, shared_track_timestamps_us, N_track_poses);
    auto const interpolation_start_idx = interpolation_end_idx - 1;

    // Calculate interpolation parameter
    auto const local_interpolation_start_timestamp_us = shared_track_timestamps_us[interpolation_start_idx];
    auto const local_interpolation_end_timestamp_us   = shared_track_timestamps_us[interpolation_end_idx];

    auto const t_interp = scalar_t(point_timestamp_us - local_interpolation_start_timestamp_us) /
                          scalar_t(local_interpolation_end_timestamp_us - local_interpolation_start_timestamp_us);

    // Get interpolated poses in tquat
    auto const start_pose = tracks_poses[track_start_idx + interpolation_start_idx];
    auto const end_pose   = tracks_poses[track_start_idx + interpolation_end_idx];

    // Interpolate pose
    auto const c_start = make_float3(start_pose[0], start_pose[1], start_pose[2]);
    auto const c_end   = make_float3(end_pose[0], end_pose[1], end_pose[2]);
    auto const q_start = make_float4(start_pose[3], start_pose[4], start_pose[5], start_pose[6]);
    auto const q_end   = make_float4(end_pose[3], end_pose[4], end_pose[5], end_pose[6]);

    auto const c_interp = (1.f - t_interp) * c_start + t_interp * c_end;
    auto const R_interp = transpose_matrix(unitquat_rotmatrix(unitquat_slerp(q_start, q_end, t_interp)));

    // Transform point to local space
    auto const point_local = apply_matrix(R_interp, point_world - c_interp);

    // Check if point is inside cuboid bounds
    auto const dims = make_float3(cuboids_dims[track_idx][0],
                                  cuboids_dims[track_idx][1],
                                  cuboids_dims[track_idx][2]);

    auto const half_dims = dims * 0.5f;

    bool const is_inside = point_local.x >= -half_dims.x && point_local.x <= half_dims.x &&
                           point_local.y >= -half_dims.y && point_local.y <= half_dims.y &&
                           point_local.z >= -half_dims.z && point_local.z <= half_dims.z;

    if (is_inside) {
        if (return_dense_mask) {
            point_cuboidtracks_intersection_mask[point_idx][track_idx] = true;
        } else {
            point_cuboidtracks_intersection_mask[point_idx][0] = true;
        }
    }
}

torch::Tensor point_cuboidtracks_intersection_cu(
    torch::Tensor const points,               // N_points x 3 (3d world positions)
    torch::Tensor const timestamps_us,        // N_points or 1 (per point or global timestamp)
    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps
    torch::Tensor const cuboids_dims,         // (N_tracks x 3) cuboid x/y/z extents (in local track frame)
    int32_t const max_track_n_poses,
    bool const return_dense_mask = true) { // If true (N_points x N_tracks) mask is returned else (N_points x 1) and true if point is inside any cuboidtrack

    auto points_arg        = torch::TensorArg{points, "points", 1};
    auto timestamps_us_arg = torch::TensorArg{timestamps_us, "timestamps_us", 2};

    auto tracks_packinfo_arg      = torch::TensorArg{tracks_packinfo, "tracks_packinfo", 3};
    auto tracks_poses_arg         = torch::TensorArg{tracks_poses, "tracks_poses", 4};
    auto tracks_timestamps_us_arg = torch::TensorArg{tracks_timestamps_us, "tracks_timestamps_us", 5};

    auto cuboids_dims_arg = torch::TensorArg{cuboids_dims, "cuboids_dims", 6};

    torch::checkAllSameType(__func__, {points_arg, tracks_poses_arg, cuboids_dims_arg});
    torch::checkScalarType(__func__, timestamps_us_arg, torch::kLong);
    torch::checkScalarType(__func__, tracks_packinfo_arg, torch::kInt32);
    torch::checkScalarType(__func__, tracks_timestamps_us_arg, torch::kLong);

    torch::checkAllSameGPU(__func__, {points_arg, timestamps_us_arg, tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg, cuboids_dims_arg});
    torch::checkAllContiguous(__func__, {points_arg, timestamps_us_arg, tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg, cuboids_dims_arg});

    auto const N_points = points.size(0), N_tracks = tracks_packinfo.size(0), N_total_poses = tracks_poses.size(0);

    torch::checkSize(__func__, points_arg, {N_points, 3});
    TORCH_CHECK(timestamps_us.size(0) == N_points || timestamps_us.size(0) == 1,
                "timestamps_us must have size N_points or 1, but got ", timestamps_us.size(0));

    torch::checkSize(__func__, tracks_packinfo_arg, {N_tracks, 2});
    torch::checkSize(__func__, tracks_poses_arg, {N_total_poses, 7});
    torch::checkSize(__func__, tracks_timestamps_us_arg, {N_total_poses});
    torch::checkSize(__func__, cuboids_dims_arg, {N_tracks, 3});

    // Create output tensor with appropriate dimensions based on return_dense_mask
    auto point_cuboidtracks_intersection_mask = torch::zeros(
        {N_points, return_dense_mask ? N_tracks : 1},
        torch::dtype(torch::kBool).device(points.device()));

    if (N_points == 0 || max_track_n_poses == 0) {
        return point_cuboidtracks_intersection_mask;
    }

    auto const threads = 256l; // N threads cooperate in processing a single track within each block
    auto const blocks  = dim3((N_points + threads - 1) / threads, N_tracks);
    auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
    auto const smem    = max_track_n_poses * sizeof(int64_t); // allocate enought shared memory for all timestamps of the longest track

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "point_cuboidtracks_intersection_cu", ([&] {
                                   point_cuboidtracks_intersection_kernel<<<blocks, threads, smem, stream>>>(
                                       points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                       tracks_packinfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                       cuboids_dims.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       point_cuboidtracks_intersection_mask.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
                                       return_dense_mask);
                               }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return point_cuboidtracks_intersection_mask;
}

template <typename scalar_t>
__global__ void point_cuboidtracks_intersection_interpolate_pose_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const points,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const points_timestamps_us,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> const tracks_packinfo,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const tracks_poses,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const tracks_timestamps_us,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const cuboids_dims,
    int32_t const max_intersections_per_point,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> intersections_cnt,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> interpolated_tracks_pose,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> interpolated_tracks_idx,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> interpolated_points_local,
    bool const return_local_points) {
    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto const track_idx = blockIdx.y;

    if (track_idx >= tracks_packinfo.size(0))
        // track of block does not exist
        return;

    auto const track_packinfo  = tracks_packinfo[track_idx];
    auto const track_start_idx = track_packinfo[0], N_track_poses = track_packinfo[1];

    if (N_track_poses <= 1)
        // track has no poses
        return;

    // load the track's timestamps into shared memory to accelerate binary search
    // using all threads (irrespective of whether the local ray exists)
    extern __shared__ int64_t shared_track_timestamps_us[];

    for (auto i = threadIdx.x; i < N_track_poses; i += blockDim.x)
        shared_track_timestamps_us[i] = tracks_timestamps_us[track_start_idx + i];

    __syncthreads(); // all timestamps of the block-associated track are now loaded to shared memory

    if (point_idx >= points.size(0))
        // local ray doesn't exist
        return;

    // check if point's timestamp is in the track's valid time-range and exit early if not
    auto const point_timestamp_us = points_timestamps_us[point_idx];

    if (point_timestamp_us < shared_track_timestamps_us[0] ||
        shared_track_timestamps_us[N_track_poses - 1] < point_timestamp_us)
        return;

    // point's time is in the tracks' valid time-range -> determine track poses for linear interpolation
    auto const interpolation_end_idx   = binary_search_interp(point_timestamp_us, shared_track_timestamps_us, N_track_poses);
    auto const interpolation_start_idx = interpolation_end_idx - 1; // note: this is guaranteed to be in the valid range of poses due to the check above

    auto const local_interpolation_start_timestamp_us = shared_track_timestamps_us[interpolation_start_idx],
               local_interpolation_end_timestamp_us   = shared_track_timestamps_us[interpolation_end_idx];

    // linear interpolation parameter [0,1]
    auto const t_interp = scalar_t(point_timestamp_us - local_interpolation_start_timestamp_us) / scalar_t(local_interpolation_end_timestamp_us - local_interpolation_start_timestamp_us);

    auto const local_interpolation_start_pose = tracks_poses[track_start_idx + interpolation_start_idx],
               local_interpolation_end_pose   = tracks_poses[track_start_idx + interpolation_end_idx];

    // grab timed bbox with start/end bbox->world pose
    auto const cuboid_dim = make_float3(cuboids_dims[track_idx][0], cuboids_dims[track_idx][1], cuboids_dims[track_idx][2]);
    auto const c_start    = make_float3(local_interpolation_start_pose[0], local_interpolation_start_pose[1], local_interpolation_start_pose[2]);
    auto const q_start    = make_float4(local_interpolation_start_pose[3], local_interpolation_start_pose[4], local_interpolation_start_pose[5], local_interpolation_start_pose[6]);
    auto const c_end      = make_float3(local_interpolation_end_pose[0], local_interpolation_end_pose[1], local_interpolation_end_pose[2]);
    auto const q_end      = make_float4(local_interpolation_end_pose[3], local_interpolation_end_pose[4], local_interpolation_end_pose[5], local_interpolation_end_pose[6]);

    // interpolate bbox to point time and create world->bbox transformation
    auto const c_pointtime = (1.f - t_interp) * c_start + t_interp * c_end;
    auto const q_pointtime = unitquat_slerp(q_start, q_end, t_interp);
    auto const R_pointtime = transpose_matrix(unitquat_rotmatrix(q_pointtime));

    // transform world-point to bbox-point
    auto const point_world = make_float3(points[point_idx][0], points[point_idx][1], points[point_idx][2]);
    auto const point_bbox  = apply_matrix(R_pointtime, point_world - c_pointtime);

    if (point_bbox > -cuboid_dim / 2.f && point_bbox < cuboid_dim / 2.f) {
        // point is inside the cuboid -> record the track pose and index in the next available slot
        auto const cnt = atomicAdd(&intersections_cnt[point_idx], 1);
        if (cnt < max_intersections_per_point) {
            interpolated_tracks_pose[point_idx][cnt][0] = c_pointtime.x;
            interpolated_tracks_pose[point_idx][cnt][1] = c_pointtime.y;
            interpolated_tracks_pose[point_idx][cnt][2] = c_pointtime.z;
            interpolated_tracks_pose[point_idx][cnt][3] = q_pointtime.x;
            interpolated_tracks_pose[point_idx][cnt][4] = q_pointtime.y;
            interpolated_tracks_pose[point_idx][cnt][5] = q_pointtime.z;
            interpolated_tracks_pose[point_idx][cnt][6] = q_pointtime.w;
            interpolated_tracks_idx[point_idx][cnt]     = track_idx;
            if (return_local_points) {
                interpolated_points_local[point_idx][cnt][0] = point_bbox.x;
                interpolated_points_local[point_idx][cnt][1] = point_bbox.y;
                interpolated_points_local[point_idx][cnt][2] = point_bbox.z;
            }
        }
    }
}

std::vector<torch::Tensor> point_cuboidtracks_intersection_interpolate_pose_cu(
    torch::Tensor const points,               // N_points x 3 (3d world positions)
    torch::Tensor const points_timestamps_us, // N_points (per point timestamp)
    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps
    torch::Tensor const cuboids_dims,         // (N_tracks x 3) cuboid x/y/z extents (in local track frame)
    int32_t const max_track_n_poses,
    int32_t const max_intersections_per_point,
    bool const return_local_points) {

    auto points_arg               = torch::TensorArg{points, "points", 1};
    auto points_timestamps_us_arg = torch::TensorArg{points_timestamps_us, "points_timestamps_us", 2};

    auto tracks_packinfo_arg      = torch::TensorArg{tracks_packinfo, "tracks_packinfo", 3};
    auto tracks_poses_arg         = torch::TensorArg{tracks_poses, "tracks_poses", 4};
    auto tracks_timestamps_us_arg = torch::TensorArg{tracks_timestamps_us, "tracks_timestamps_us", 5};

    auto cuboids_dims_arg = torch::TensorArg{cuboids_dims, "cuboids_dims", 6};

    torch::checkAllSameType(__func__, {points_arg, tracks_poses_arg, cuboids_dims_arg});
    torch::checkScalarType(__func__, points_timestamps_us_arg, torch::kLong);
    torch::checkScalarType(__func__, tracks_packinfo_arg, torch::kInt32);
    torch::checkScalarType(__func__, tracks_timestamps_us_arg, torch::kLong);

    torch::checkAllSameGPU(__func__, {points_arg, points_timestamps_us_arg,
                                      tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg,
                                      cuboids_dims_arg});
    torch::checkAllContiguous(__func__, {points_arg, points_timestamps_us_arg,
                                         tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg,
                                         cuboids_dims_arg});

    auto const N_points = points.size(0), N_tracks = tracks_packinfo.size(0), N_total_poses = tracks_poses.size(0);

    torch::checkSize(__func__, points_arg, {N_points, 3});
    torch::checkSize(__func__, points_timestamps_us_arg, {N_points});

    torch::checkSize(__func__, tracks_packinfo_arg, {N_tracks, 2});
    torch::checkSize(__func__, tracks_poses_arg, {N_total_poses, 7});
    torch::checkSize(__func__, tracks_timestamps_us_arg, {N_total_poses});

    torch::checkSize(__func__, cuboids_dims_arg, {N_tracks, 3});

    auto intersections_cnt        = torch::zeros({N_points}, torch::dtype(torch::kInt32).device(points.device()));
    auto interpolated_tracks_pose = torch::zeros({N_points, max_intersections_per_point, 7}, points.options());
    auto interpolated_tracks_idx  = torch::full({N_points, max_intersections_per_point}, -1, torch::dtype(torch::kInt32).device(points.device()));
    // If not needed put some arbitrary tensor instead
    auto interpolated_points_local = return_local_points ? torch::zeros({N_points, max_intersections_per_point, 3}, points.options())
                                                         : torch::empty({0, 0, 0}, points.options());

    if (N_points == 0 || max_track_n_poses == 0) {
        if (return_local_points) {
            return {intersections_cnt, interpolated_tracks_pose, interpolated_tracks_idx, interpolated_points_local};
        } else {
            return {intersections_cnt, interpolated_tracks_pose, interpolated_tracks_idx};
        }
    }

    auto const threads = 256l; // N threads cooperate in processing a single track within each block
    auto const blocks  = dim3((N_points + threads - 1) / threads, N_tracks);
    auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
    auto const smem    = max_track_n_poses * sizeof(int64_t); // allocate enought shared memory for all timestamps of the longest track

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "point_cuboidtracks_intersection_interpolate_pose_cu", ([&] {
                                   point_cuboidtracks_intersection_interpolate_pose_kernel<<<blocks, threads, smem, stream>>>(
                                       points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       points_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                       tracks_packinfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                       cuboids_dims.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       max_intersections_per_point,
                                       intersections_cnt.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                       interpolated_tracks_pose.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                       interpolated_tracks_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       interpolated_points_local.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                       return_local_points);
                               }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (return_local_points) {
        return {intersections_cnt, interpolated_tracks_pose, interpolated_tracks_idx, interpolated_points_local};
    } else {
        return {intersections_cnt, interpolated_tracks_pose, interpolated_tracks_idx};
    }
}

template <typename scalar_t>
__global__ void ray_cuboidtracks_rolling_shutter_intersection_kernel(
    torch::PackedTensorAccessor32<int16_t, 2, torch::RestrictPtrTraits> const pixel_idxs, // i in [0, width-1], j in [0, height-1]
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const camera_rays,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const camera_poses,
    int64_t const camera_timestamp_start_us,
    int64_t const camera_timestamp_end_us,
    int32_t const w,            // image resolution (width)
    int32_t const h,            // image resolution (height)
    int32_t const shutter_type, // ROLLING_TOP_TO_BOTTOM = 1, ROLLING_LEFT_TO_RIGHT = 2, ROLLING_BOTTOM_TO_TOP = 3,
                                // ROLLING_RIGHT_TO_LEFT = 4, GLOBAL = 5

    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> const tracks_packinfo,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const tracks_poses,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const tracks_timestamps_us,

    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const cuboids_dims,

    int32_t const max_intersections_per_ray,

    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> intersections_cnt,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> intersections_ts,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> intersections_tracks_idx,
    bool const with_intersections_ts) {
    auto const ray_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    auto const track_idx = blockIdx.y;

    if (track_idx >= tracks_packinfo.size(0))
        // track of block does not exist
        return;

    auto const track_packinfo  = tracks_packinfo[track_idx];
    auto const track_start_idx = track_packinfo[0];
    auto const N_track_poses   = track_packinfo[1];
    auto const N_total_poses   = tracks_poses.size(0);

    if (N_track_poses <= 1)
        // track has no poses
        return;

    // load the track's timestamps into shared memory to accelerate binary search
    // using all threads (irrespective of whether the local ray exists)
    extern __shared__ int64_t shared_track_timestamps_us[];

    for (auto i = threadIdx.x; i < N_track_poses; i += blockDim.x)
        shared_track_timestamps_us[i] = tracks_timestamps_us[track_start_idx + i];

    __syncthreads(); // all timestamps of the block-associated track are now loaded to shared memory

    if (ray_idx >= pixel_idxs.size(0))
        // local ray doesn't exist
        return;

    // grab and transform camera-ray to world-ray via rolling-shutter time
    auto const [ray_o_world, ray_d_world, relative_frame_time] = rolling_shutter_camera_ray_to_world(
        pixel_idxs[ray_idx][0],
        pixel_idxs[ray_idx][1],
        make_float3(camera_rays[ray_idx][0], camera_rays[ray_idx][1], camera_rays[ray_idx][2]),
        make_float3(camera_poses[0][0], camera_poses[0][1], camera_poses[0][2]),
        make_float4(camera_poses[0][3], camera_poses[0][4], camera_poses[0][5], camera_poses[0][6]),
        make_float3(camera_poses[1][0], camera_poses[1][1], camera_poses[1][2]),
        make_float4(camera_poses[1][3], camera_poses[1][4], camera_poses[1][5], camera_poses[1][6]),
        w, h, shutter_type);

    // map frame-relative time to absolute time
    auto const ray_timestamp_us = camera_timestamp_start_us + int64_t(relative_frame_time * (camera_timestamp_end_us - camera_timestamp_start_us));

    // check if ray's timestamp is in the track's valid time-range and exit early if not
    if (ray_timestamp_us < shared_track_timestamps_us[0] ||
        shared_track_timestamps_us[N_track_poses - 1] < ray_timestamp_us)
        return;

    // ray's time is in the tracks' valid time-range AND we have N_track_poses > 1 -> determine track poses for linear interpolation
    auto const interpolation_end_idx   = binary_search_interp(ray_timestamp_us, shared_track_timestamps_us, N_track_poses);
    auto const interpolation_start_idx = interpolation_end_idx - 1; // note: this is guaranteed to be in the valid range of poses due to the check above

    auto const local_interpolation_start_timestamp_us = shared_track_timestamps_us[interpolation_start_idx],
               local_interpolation_end_timestamp_us   = shared_track_timestamps_us[interpolation_end_idx];

    // linear interpolation parameter [0,1]
    auto const t_interp = scalar_t(ray_timestamp_us - local_interpolation_start_timestamp_us) / scalar_t(local_interpolation_end_timestamp_us - local_interpolation_start_timestamp_us);

    // load poses to interpolate from global memory
    auto const local_interpolation_start_pose = tracks_poses[track_start_idx + interpolation_start_idx],
               local_interpolation_end_pose   = tracks_poses[track_start_idx + interpolation_end_idx];

    // collect timed-cuboid data for intersection computation
    auto const cuboidtimed = std::array<scalar_t, 17>{cuboids_dims[track_idx][0],
                                                      cuboids_dims[track_idx][1],
                                                      cuboids_dims[track_idx][2],

                                                      local_interpolation_start_pose[0],
                                                      local_interpolation_start_pose[1],
                                                      local_interpolation_start_pose[2],
                                                      local_interpolation_start_pose[3],
                                                      local_interpolation_start_pose[4],
                                                      local_interpolation_start_pose[5],
                                                      local_interpolation_start_pose[6],

                                                      local_interpolation_end_pose[0],
                                                      local_interpolation_end_pose[1],
                                                      local_interpolation_end_pose[2],
                                                      local_interpolation_end_pose[3],
                                                      local_interpolation_end_pose[4],
                                                      local_interpolation_end_pose[5],
                                                      local_interpolation_end_pose[6]};

    // perform time-based cuboid intersection
    auto const t1t2 = worldray_bboxtimed_intersection(ray_o_world, ray_d_world, t_interp, cuboidtimed);

    if (t1t2.y > 0.f) { // if ray intersects the cuboid
        auto const cnt = atomicAdd(&intersections_cnt[ray_idx], 1);
        if (cnt < max_intersections_per_ray) {
            if (with_intersections_ts) {
                intersections_ts[ray_idx][cnt][0] = fmaxf(t1t2.x, 0.f);
                intersections_ts[ray_idx][cnt][1] = t1t2.y;
            }
            intersections_tracks_idx[ray_idx][cnt] = track_idx;
        }
    }
}

std::vector<torch::Tensor> ray_cuboidtracks_rolling_shutter_intersection_cu(
    torch::Tensor const pixel_idxs,   // N_rays x 2 (pixel indices of rays, i in [0, width-1] / j in [0, height-1])
    torch::Tensor const camera_rays,  // N_rays x 3 (camera-space rays)
    torch::Tensor const camera_poses, // 2 x 7  [[x, y, z], [quat_x, quat_y, quat_z, quat_w]] for start/end pose
    int64_t const camera_timestamp_start_us,
    int64_t const camera_timestamp_end_us,
    int32_t const w,            // image resolution (width)
    int32_t const h,            // image resolution (height)
    int32_t const shutter_type, // ROLLING_TOP_TO_BOTTOM = 1, ROLLING_LEFT_TO_RIGHT = 2, ROLLING_BOTTOM_TO_TOP = 3,
                                // ROLLING_RIGHT_TO_LEFT = 4, GLOBAL = 5

    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps

    torch::Tensor const cuboids_dims, // (N_tracks x 3) cuboid x/y/z extents (in local track frame)

    int32_t const max_track_n_poses,
    int32_t const max_intersections_per_ray,
    bool const with_intersections_ts) {

    auto pixel_idxs_arg   = torch::TensorArg{pixel_idxs, "pixel_idxs", 1};
    auto camera_rays_arg  = torch::TensorArg{camera_rays, "camera_rays", 2};
    auto camera_poses_arg = torch::TensorArg{camera_poses, "camera_poses", 3};

    auto tracks_packinfo_arg      = torch::TensorArg{tracks_packinfo, "tracks_packinfo", 4};
    auto tracks_poses_arg         = torch::TensorArg{tracks_poses, "tracks_poses", 5};
    auto tracks_timestamps_us_arg = torch::TensorArg{tracks_timestamps_us, "tracks_timestamps_us", 6};

    auto cuboids_dims_arg = torch::TensorArg{cuboids_dims, "cuboids_dims", 7};

    torch::checkAllSameType(__func__, {camera_rays_arg, camera_poses_arg, tracks_poses_arg, cuboids_dims_arg});
    torch::checkScalarType(__func__, pixel_idxs_arg, torch::kInt16);
    torch::checkScalarType(__func__, tracks_packinfo_arg, torch::kInt32);
    torch::checkScalarType(__func__, tracks_timestamps_us_arg, torch::kLong);

    torch::checkAllSameGPU(__func__, {pixel_idxs_arg, camera_rays_arg, camera_poses_arg,
                                      tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg,
                                      cuboids_dims_arg});
    torch::checkAllContiguous(__func__, {pixel_idxs_arg, camera_rays_arg, camera_poses_arg,
                                         tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg,
                                         cuboids_dims_arg});

    auto const N_rays = camera_rays.size(0), N_tracks = tracks_packinfo.size(0), N_total_poses = tracks_poses.size(0);

    torch::checkSize(__func__, pixel_idxs_arg, {N_rays, 2});
    torch::checkSize(__func__, camera_rays_arg, {N_rays, 3});
    torch::checkSize(__func__, camera_poses_arg, {2, 7});

    torch::checkSize(__func__, tracks_packinfo_arg, {N_tracks, 2});
    torch::checkSize(__func__, tracks_poses_arg, {N_total_poses, 7});
    torch::checkSize(__func__, tracks_timestamps_us_arg, {N_total_poses});

    torch::checkSize(__func__, cuboids_dims_arg, {N_tracks, 3});

    auto intersections_cnt = torch::zeros({N_rays}, torch::dtype(torch::kInt32).device(pixel_idxs.device()));
    // If not needed put some arbitrary tensor instead
    auto intersections_ts         = with_intersections_ts ? torch::full({N_rays, max_intersections_per_ray, 2}, -1., camera_rays.options()) : torch::empty({0, 0, 0}, camera_rays.options());
    auto intersections_tracks_idx = torch::full({N_rays, max_intersections_per_ray}, -1, torch::dtype(torch::kInt32).device(pixel_idxs.device()));

    if (N_rays == 0 || max_track_n_poses == 0) {
        if (with_intersections_ts) {
            return {intersections_cnt, intersections_tracks_idx, intersections_ts};
        } else {
            return {intersections_cnt, intersections_tracks_idx};
        }
    }

    auto const threads = 256l; // N threads cooperate in processing a single track within each block
    auto const blocks  = dim3((N_rays + threads - 1) / threads, N_tracks);
    auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
    auto const smem    = max_track_n_poses * sizeof(int64_t); // allocate enought shared memory for all timestamps of the longest track

    AT_DISPATCH_FLOATING_TYPES(camera_rays.scalar_type(), "ray_cuboidtracks_rolling_shutter_intersection_cu", ([&] {
                                   ray_cuboidtracks_rolling_shutter_intersection_kernel<<<blocks, threads, smem, stream>>>(
                                       pixel_idxs.packed_accessor32<int16_t, 2, torch::RestrictPtrTraits>(),
                                       camera_rays.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       camera_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       camera_timestamp_start_us, camera_timestamp_end_us, w, h, shutter_type,

                                       tracks_packinfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),

                                       cuboids_dims.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),

                                       max_intersections_per_ray,

                                       intersections_cnt.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                       intersections_ts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                       intersections_tracks_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       with_intersections_ts);
                               }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (with_intersections_ts) {
        return {intersections_cnt, intersections_tracks_idx, intersections_ts};
    } else {
        return {intersections_cnt, intersections_tracks_idx};
    }
}
