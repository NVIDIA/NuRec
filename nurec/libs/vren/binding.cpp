// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "utils.h"

#include <vector>

#include <ku/common_host.h>

namespace LidarModelFunctions {

bool has_rolling_shutter_info(const RowOffsetStructuredSpinningLidarModelParameters& parameters) {
    return parameters._angles_to_columns_map != nullptr;
}

bool has_tiling_info(const RowOffsetStructuredSpinningLidarModelParameters& parameters) {
    return parameters._cdf_elevation != nullptr && parameters._tiles_pack_info != nullptr && parameters._tiles_to_elements_map != nullptr;
}

void set_row_column_angles_and_offsets(
    RowOffsetStructuredSpinningLidarModelParameters& parameters,
    torch::Tensor row_elevations_rad,
    torch::Tensor column_azimuths_rad,
    torch::Tensor row_azimuth_offsets_rad) {

    CHECK_INPUT(row_elevations_rad);
    CHECK_INPUT(column_azimuths_rad);
    CHECK_INPUT(row_azimuth_offsets_rad);

    auto const row_elevations_rad_arg      = torch::TensorArg{row_elevations_rad, "row_elevations_rad", 2};
    auto const column_azimuths_rad_arg     = torch::TensorArg{column_azimuths_rad, "column_azimuths_rad", 3};
    auto const row_azimuth_offsets_rad_arg = torch::TensorArg{row_azimuth_offsets_rad, "row_azimuth_offsets_rad", 4};

    auto const N_rows    = parameters.n_rows;
    auto const N_columns = parameters.n_columns;
    torch::checkSize(__func__, row_elevations_rad_arg, {N_rows});
    torch::checkSize(__func__, column_azimuths_rad_arg, {N_columns});
    torch::checkSize(__func__, row_azimuth_offsets_rad_arg, {N_rows});

    parameters._row_elevations_rad      = row_elevations_rad.data_ptr<float>();
    parameters._column_azimuths_rad     = column_azimuths_rad.data_ptr<float>();
    parameters._row_azimuth_offsets_rad = row_azimuth_offsets_rad.data_ptr<float>();
}

void set_angles_to_columns_map(
    RowOffsetStructuredSpinningLidarModelParameters& parameters,
    torch::Tensor angles_to_columns_map,
    float angles_to_columns_map_resolution_factor) {
    CHECK_INPUT(angles_to_columns_map);
    auto const angles_to_columns_map_arg = torch::TensorArg{angles_to_columns_map, "angles_to_columns_map", 2};

    auto const N_rows           = parameters.n_rows;
    auto const N_columns        = parameters.n_columns;
    auto const N_rows_factor    = static_cast<int>(N_rows * angles_to_columns_map_resolution_factor);
    auto const N_columns_factor = static_cast<int>(N_columns * angles_to_columns_map_resolution_factor);
    torch::checkSize(__func__, angles_to_columns_map_arg, {N_rows_factor, N_columns_factor});

    parameters._angles_to_columns_map                  = angles_to_columns_map.data_ptr<int>();
    parameters.angles_to_columns_map_resolution_factor = angles_to_columns_map_resolution_factor;
    parameters.map_resolution_horiz_rad                = parameters.fov_horiz_span_rad / (N_columns_factor - 1);
    parameters.map_resolution_vert_rad                 = parameters.fov_vert_span_rad / (N_rows_factor - 1);
}

void set_tiling_info(
    RowOffsetStructuredSpinningLidarModelParameters& parameters,
    int n_bins_azimuth,
    int n_bins_elevation,
    int densification_factor_azimuth,
    int max_pts_per_tile,
    torch::Tensor cdf_elevation,
    torch::Tensor cdf_dense_ray_mask,
    torch::Tensor tiles_to_elements_map,
    torch::Tensor tiles_pack_info) {
    CHECK_INPUT(cdf_elevation);
    CHECK_INPUT(tiles_to_elements_map);
    CHECK_INPUT(tiles_pack_info);

    auto const cdf_elevation_arg         = torch::TensorArg{cdf_elevation, "cdf_elevation", 6};
    auto const cdf_dense_ray_mask_arg    = torch::TensorArg{cdf_dense_ray_mask, "cdf_dense_ray_mask", 7};
    auto const tiles_to_elements_map_arg = torch::TensorArg{tiles_to_elements_map, "tiles_to_elements_map", 8};
    auto const tiles_pack_info_arg       = torch::TensorArg{tiles_pack_info, "tiles_pack_info", 9};

    auto const cdf_length = cdf_elevation.size(0);
    torch::checkSize(__func__, cdf_elevation_arg, {cdf_length});

    auto const n_points = parameters.n_rows * parameters.n_columns;
    torch::checkSize(__func__, tiles_to_elements_map_arg, {n_points, 2});

    auto const n_tiles = n_bins_azimuth * n_bins_elevation;
    torch::checkSize(__func__, tiles_pack_info_arg, {n_tiles, 2});

    auto const resolution_azimuth = n_bins_azimuth * densification_factor_azimuth;
    torch::checkSize(__func__, cdf_dense_ray_mask_arg, {resolution_azimuth + 1, cdf_length});

    const int last = cdf_elevation.index({cdf_length - 1}).item<int>();
    if (last != n_bins_elevation) {
        throw std::invalid_argument("Last element of cdf_elevation must be equal to n_bins_elevation");
    }

    parameters._cdf_elevation           = cdf_elevation.data_ptr<int>();
    parameters._cdf_dense_ray_mask      = cdf_dense_ray_mask.data_ptr<int>();
    parameters.elevation_cdf_resolution = cdf_length - 1;
    parameters.azimuth_cdf_resolution   = resolution_azimuth;

    parameters.n_bins_azimuth   = n_bins_azimuth;
    parameters.n_bins_elevation = n_bins_elevation;
    parameters.max_pts_per_tile = max_pts_per_tile;

    parameters._tiles_to_elements_map = tiles_to_elements_map.data_ptr<int>();
    parameters._tiles_pack_info       = tiles_pack_info.data_ptr<int>();
}

} // namespace LidarModelFunctions

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ray_cuboidtracks_intersection", &ray_cuboidtracks_intersection_cu,
          py::arg("rays_o"),
          py::arg("rays_d"),
          py::arg("rays_timestamps_us"),
          py::arg("tracks_packinfo"),
          py::arg("tracks_poses"),
          py::arg("tracks_timestamps_us"),
          py::arg("cuboids_dims"),
          py::arg("max_track_n_poses"),
          py::arg("max_intersections_per_ray"),
          py::arg("with_intersections_ts"));
    m.def("point_cuboidtracks_intersection", &point_cuboidtracks_intersection_cu);
    m.def("point_cuboidtracks_intersection_interpolate_pose", &point_cuboidtracks_intersection_interpolate_pose_cu,
          py::arg("points"),
          py::arg("points_timestamps_us"),
          py::arg("tracks_packinfo"),
          py::arg("tracks_poses"),
          py::arg("tracks_timestamps_us"),
          py::arg("cuboids_dims"),
          py::arg("max_track_n_poses"),
          py::arg("max_intersections_per_point"),
          py::arg("return_local_points"));
    m.def("ray_cuboidtracks_rolling_shutter_intersection", &ray_cuboidtracks_rolling_shutter_intersection_cu,
          py::arg("pixel_idxs"),
          py::arg("camera_rays"),
          py::arg("camera_poses"),
          py::arg("camera_timestamp_start_us"),
          py::arg("camera_timestamp_end_us"),
          py::arg("w"),
          py::arg("h"),
          py::arg("shutter_type"),
          py::arg("tracks_packinfo"),
          py::arg("tracks_poses"),
          py::arg("tracks_timestamps_us"),
          py::arg("cuboids_dims"),
          py::arg("max_track_n_poses"),
          py::arg("max_intersections_per_ray"),
          py::arg("with_intersections_ts"));

    // General types
    py::class_<RollingShutterParameters>(m, "RollingShutterParameters")
        .def(py::init<>())
        .def_readwrite("T_world_sensors", &RollingShutterParameters::T_world_sensors)
        .def_readwrite("timestamps_us", &RollingShutterParameters::timestamps_us);

    // Camera-specific types
    py::enum_<ShutterType>(m, "ShutterType")
        .value("ROLLING_TOP_TO_BOTTOM", ShutterType::ROLLING_TOP_TO_BOTTOM)
        .value("ROLLING_LEFT_TO_RIGHT", ShutterType::ROLLING_LEFT_TO_RIGHT)
        .value("ROLLING_BOTTOM_TO_TOP", ShutterType::ROLLING_BOTTOM_TO_TOP)
        .value("ROLLING_RIGHT_TO_LEFT", ShutterType::ROLLING_RIGHT_TO_LEFT)
        .value("GLOBAL", ShutterType::GLOBAL);

    py::enum_<ReferencePolynomial>(m, "ReferencePolynomial")
        .value("FORWARD", ReferencePolynomial::FORWARD)
        .value("BACKWARD", ReferencePolynomial::BACKWARD);

    py::class_<BivariateWindshieldModelParameters>(m, "BivariateWindshieldModelParameters")
        .def(py::init<>())
        .def_readwrite("reference_poly", &BivariateWindshieldModelParameters::reference_poly)
        .def_readwrite("horizontal_poly", &BivariateWindshieldModelParameters::horizontal_poly)
        .def_readwrite("vertical_poly", &BivariateWindshieldModelParameters::vertical_poly)
        .def_readwrite("horizontal_poly_inverse", &BivariateWindshieldModelParameters::horizontal_poly_inverse)
        .def_readwrite("vertical_poly_inverse", &BivariateWindshieldModelParameters::vertical_poly_inverse);

    py::class_<CameraModelParameters>(m, "CameraModelParameters")
        .def(py::init<>())
        .def_readwrite("resolution", &CameraModelParameters::resolution)
        .def_readwrite("shutter_type", &CameraModelParameters::shutter_type)
        .def_readwrite("external_distortion_parameters", &CameraModelParameters::external_distortion_parameters);

    py::class_<OpenCVPinholeCameraModelParameters, CameraModelParameters>(m, "OpenCVPinholeCameraModelParameters")
        .def(py::init<>())
        .def_readwrite("principal_point", &OpenCVPinholeCameraModelParameters::principal_point)
        .def_readwrite("focal_length", &OpenCVPinholeCameraModelParameters::focal_length)
        .def_readwrite("radial_coeffs", &OpenCVPinholeCameraModelParameters::radial_coeffs)
        .def_readwrite("tangential_coeffs", &OpenCVPinholeCameraModelParameters::tangential_coeffs)
        .def_readwrite("thin_prism_coeffs", &OpenCVPinholeCameraModelParameters::thin_prism_coeffs);

    py::class_<OpenCVFisheyeCameraModelParameters, CameraModelParameters>(m, "OpenCVFisheyeCameraModelParameters")
        .def(py::init<>())
        .def_readwrite("principal_point", &OpenCVFisheyeCameraModelParameters::principal_point)
        .def_readwrite("focal_length", &OpenCVFisheyeCameraModelParameters::focal_length)
        .def_readwrite("radial_coeffs", &OpenCVFisheyeCameraModelParameters::radial_coeffs)
        .def_readwrite("max_angle", &OpenCVFisheyeCameraModelParameters::max_angle);

    py::class_<FThetaCameraModelParameters, CameraModelParameters>(m, "FThetaCameraModelParameters")
        .def(py::init<>())
        .def_readwrite("principal_point", &FThetaCameraModelParameters::principal_point)
        .def_readwrite("reference_poly", &FThetaCameraModelParameters::reference_poly)
        .def_readwrite("pixeldist_to_angle_poly", &FThetaCameraModelParameters::pixeldist_to_angle_poly)
        .def_readwrite("angle_to_pixeldist_poly", &FThetaCameraModelParameters::angle_to_pixeldist_poly)
        .def_readwrite("max_angle", &FThetaCameraModelParameters::max_angle)
        .def_readwrite("linear_cde", &FThetaCameraModelParameters::linear_cde);

    py::enum_<FThetaCameraModelParameters::PolynomialType>(m, "PolynomialType")
        .value("PIXELDIST_TO_ANGLE", FThetaCameraModelParameters::PolynomialType::PIXELDIST_TO_ANGLE)
        .value("ANGLE_TO_PIXELDIST", FThetaCameraModelParameters::PolynomialType::ANGLE_TO_PIXELDIST);

    // LiDAR-specific types
    py::enum_<SpinningDirection>(m, "SpinningDirection")
        .value("CLOCK_WISE", SpinningDirection::CLOCK_WISE)
        .value("COUNTER_CLOCK_WISE", SpinningDirection::COUNTER_CLOCK_WISE);

    py::class_<BaseSpinningLidarModelParameters>(m, "BaseSpinningLidarModelParameters")
        .def(py::init<>())
        .def_readwrite("spinning_frequency_hz", &BaseSpinningLidarModelParameters::spinning_frequency_hz)
        .def_readwrite("spinning_direction", &BaseSpinningLidarModelParameters::spinning_direction);

    py::class_<BaseStructuredSpinningLidarModelParameters, BaseSpinningLidarModelParameters>(m, "BaseStructuredSpinningLidarModelParameters")
        .def(py::init<>())
        .def_readwrite("n_rows", &BaseStructuredSpinningLidarModelParameters::n_rows)
        .def_readwrite("n_columns", &BaseStructuredSpinningLidarModelParameters::n_columns)
        .def_readwrite("fov_horiz_start_rad", &BaseStructuredSpinningLidarModelParameters::fov_horiz_start_rad)
        .def_readwrite("fov_horiz_span_rad", &BaseStructuredSpinningLidarModelParameters::fov_horiz_span_rad)
        .def_readwrite("fov_vert_start_rad", &BaseStructuredSpinningLidarModelParameters::fov_vert_start_rad)
        .def_readwrite("fov_vert_span_rad", &BaseStructuredSpinningLidarModelParameters::fov_vert_span_rad);

    py::class_<RowOffsetStructuredSpinningLidarModelParameters, BaseStructuredSpinningLidarModelParameters>(m, "RowOffsetStructuredSpinningLidarModelParameters")
        .def(py::init<>())
        .def_readonly("angles_to_columns_map_resolution_factor", &RowOffsetStructuredSpinningLidarModelParameters::angles_to_columns_map_resolution_factor)
        .def_readonly("map_resolution_horiz_rad", &RowOffsetStructuredSpinningLidarModelParameters::map_resolution_horiz_rad)
        .def_readonly("map_resolution_vert_rad", &RowOffsetStructuredSpinningLidarModelParameters::map_resolution_vert_rad)
        .def_readonly("n_bins_azimuth", &RowOffsetStructuredSpinningLidarModelParameters::n_bins_azimuth)
        .def_readonly("n_bins_elevation", &RowOffsetStructuredSpinningLidarModelParameters::n_bins_elevation)
        .def("set_row_column_angles_and_offsets", &LidarModelFunctions::set_row_column_angles_and_offsets)
        .def("set_angles_to_columns_map", &LidarModelFunctions::set_angles_to_columns_map)
        .def("set_tiling_info", &LidarModelFunctions::set_tiling_info)
        .def("has_rolling_shutter_info", &LidarModelFunctions::has_rolling_shutter_info)
        .def("has_tiling_info", &LidarModelFunctions::has_tiling_info);

    // Camera-specific functions
    m.def("camera_rays_to_image_points",
          &camera_rays_to_image_points_cu);

    m.def("world_points_to_image_points_shutter_pose",
          &world_points_to_image_points_shutter_pose_cu);

    m.def("image_points_to_world_points_shutter_pose",
          &image_points_to_world_points_shutter_pose_cu);

    // LiDAR-specific functions
    m.def("elements_to_sensor_rays",
          &elements_to_sensor_rays_cu);

    m.def("elements_to_sensor_angles",
          &elements_to_sensor_angles_cu);

    m.def("sensor_rays_to_sensor_angles",
          &sensor_rays_to_sensor_angles_cu);

    m.def("sensor_angles_to_sensor_rays",
          &sensor_angles_to_sensor_rays_cu);

    m.def("sensor_angles_relative_frame_times",
          &sensor_angles_relative_frame_times_cu);

    m.def("world_points_to_sensor_angles_shutter_pose",
          &world_points_to_sensor_angles_shutter_pose_cu);

    m.def("elements_to_world_rays_shutter_pose",
          &elements_to_world_rays_shutter_pose_cu);

    m.def("sensor_angles_to_tile_indices",
          &sensor_angles_to_tile_indices_cu);

    // Ensemble functions
    m.def("lidar_seg_ensemble", &lidar_seg_ensemble_cu);
}
