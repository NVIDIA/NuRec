// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vren/cameras.h>
#include <vren/lidars.h>
#include <vren/rolling_shutter.h>

#include <torch/extension.h>

#include <pybind11/numpy.h>

#include <cstdint>
#include <vector>

// Returns:
// - intersections_cnt, intersections_tracks_idx, intersections_ts if with_intersections_ts is True
// - intersections_cnt, intersections_tracks_idx if with_intersections_ts is False
std::vector<torch::Tensor> ray_cuboidtracks_intersection_cu(
    torch::Tensor const rays_o,               // N_rays x 3 (3d world positions)
    torch::Tensor const rays_d,               // N_rays x 3 (normalized 3d world directions)
    torch::Tensor const rays_timestamps_us,   // N_rays (per ray timestamp)
    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps
    torch::Tensor const cuboids_dims,         // (N_tracks x 3) cuboid x/y/z extents (in local track frame)
    int32_t const max_track_n_poses,
    int32_t const max_intersections_per_ray,
    bool const with_intersections_ts);

torch::Tensor point_cuboidtracks_intersection_cu(
    torch::Tensor const points,               // N_points x 3 (3d world positions)
    torch::Tensor const timestamps_us,        // N_points or 1 (per point or global timestamp)
    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps
    torch::Tensor const cuboids_dims,         // (N_tracks x 3) cuboid x/y/z extents (in local track frame)
    int32_t const max_track_n_poses,
    bool const return_dense_mask); // If true (N_points x N_tracks) mask is returned else (N_points x 1) and true if point is inside any cuboidtrack

// Returns:
// - intersections_cnt (N_points): number of cuboid-tracks each point is inside (may exceed max_intersections_per_point)
// - interpolated_tracks_pose (N_points x max_intersections_per_point x 7): per-intersection interpolated track pose [t, q]
// - interpolated_tracks_idx (N_points x max_intersections_per_point): per-intersection track index, -1 padded
// - interpolated_points_local (N_points x max_intersections_per_point x 3, optional): per-intersection point coords in bbox-local frame (if return_local_points)
std::vector<torch::Tensor> point_cuboidtracks_intersection_interpolate_pose_cu(
    torch::Tensor const points,               // N_points x 3 (3d world positions)
    torch::Tensor const points_timestamps_us, // N_points (per point timestamp)
    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps
    torch::Tensor const cuboids_dims,         // (N_tracks x 3) cuboid x/y/z extents (in local track frame)
    int32_t const max_track_n_poses,
    int32_t const max_intersections_per_point,
    bool const return_local_points);

// Returns:
// - intersections_cnt, intersections_tracks_idx, intersections_ts if with_intersections_ts is True
// - intersections_cnt, intersections_tracks_idx if with_intersections_ts is False
std::vector<torch::Tensor> ray_cuboidtracks_rolling_shutter_intersection_cu(
    torch::Tensor const pixel_idxs,   // N_rays x 2 (pixel indices of rays, i in [0, width-1] / j in [0, height-1])
    torch::Tensor const camera_rays,  // N_rays x 3 (camera-space rays)
    torch::Tensor const camera_poses, // 2 x 7  [[x, y, z], [quat_x, quat_y, quat_z, quat_w]] for start/end pose
    int64_t const camera_timestamp_start_us,
    int64_t const camera_timestamp_end_us,
    int32_t const w,                          // image resolution (width)
    int32_t const h,                          // image resolution (height)
    int32_t const shutter_type,               // ROLLING_TOP_TO_BOTTOM = 1, ROLLING_LEFT_TO_RIGHT = 2, ROLLING_BOTTOM_TO_TOP = 3,
                                              // ROLLING_RIGHT_TO_LEFT = 4, GLOBAL = 5
    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps
    torch::Tensor const cuboids_dims,         // (N_tracks x 3) cuboid x/y/z extents (in local track frame)
    int32_t const max_track_n_poses,
    int32_t const max_intersections_per_ray,
    bool const with_intersections_ts); // If true, sort intersections by depth (near to far); if False, keep unsorted (faster)

// Camera-specific types

std::tuple<torch::Tensor, torch::Tensor>
camera_rays_to_image_points_cu(
    const CameraModelParametersVariant camera_model_parameters,
    const torch::Tensor cam_rays);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
world_points_to_image_points_shutter_pose_cu(
    const CameraModelParametersVariant camera_model_parameters,
    const RollingShutterParameters rolling_shutter_parameters,
    const torch::Tensor world_points);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
image_points_to_world_points_shutter_pose_cu(
    const CameraModelParametersVariant camera_model_parameters,
    const torch::Tensor T_sensor_worlds, // (2, 7) tensor with tquat poses
    const torch::Tensor timestamps_us,   // (2,) tensor with int64 timestamps
    const torch::Tensor image_points);

// LiDAR specific types

torch::Tensor elements_to_sensor_rays_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters, const torch::Tensor& elements);

torch::Tensor elements_to_sensor_angles_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters, const torch::Tensor& elements);

std::tuple<torch::Tensor, torch::Tensor>
sensor_rays_to_sensor_angles_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters, const torch::Tensor& sensor_rays);

std::tuple<torch::Tensor, torch::Tensor>
sensor_angles_to_sensor_rays_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters, const torch::Tensor& sensor_angles);

torch::Tensor
sensor_angles_relative_frame_times_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters,
    const torch::Tensor& sensor_angles);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
world_points_to_sensor_angles_shutter_pose_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters,
    RollingShutterParameters const rolling_shutter_parameters,
    const torch::Tensor& world_points);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
elements_to_world_rays_shutter_pose_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters,
    const torch::Tensor T_sensor_worlds, // (2, 7) tensor with tquat poses
    const torch::Tensor timestamps_us,   // (2,) tensor with int64 timestamps
    const torch::Tensor& elements);

torch::Tensor sensor_angles_to_tile_indices_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters,
    const torch::Tensor& sensor_angles);

torch::Tensor lidar_seg_ensemble_cu(
    torch::Tensor points,
    torch::Tensor result,
    unsigned char ignore_label);
