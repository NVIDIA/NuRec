# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from pathlib import Path

import click
import numpy as np
import torch

from apps.av.road_mesh.ground_mesh import (
    get_nominal_ground_point_under_lidar,
    reconstruct_ground_mesh_from_points,
)
from nurec.config.parse import parse_untyped_config
from nurec.datasets import make as make_dataset
from nurec.datasets.ncore import NCOREDataSource
from nurec.utils.io.ply import save_ply
from nurec.utils.io.sequence_tracks import generate_track_colors, save_cuboids_ply
from nurec.utils.types import TrackFlags


logger = logging.getLogger(__name__)


@click.command("export-ground-mesh")
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    default="tests/ncore_ds",
    required=True,
)
@click.option(
    "--output-dir",
    type=str,
    help="Path to the output target directory",
    required=True,
)
@click.option(
    "--step-frame",
    type=click.IntRange(min=1, max_open=True),
    help="Frame step used to skip poses when fusing point clouds (>1 skips poses)",
    default=1,
)
@click.option(
    "--export-per-frame-diagnostics",
    is_flag=True,
    help="Export intermediate processing results per Lidar frame",
    default=False,
)
@click.option(
    "--export-meshing-diagnostics",
    is_flag=True,
    help="Export extra files (incl. input segmented points, mesh before smoothing) to PLY to help diagnose meshing",
    default=False,
)
@click.option(
    "--num-plane-hypotheses",
    type=int,
    help="Number of plane hypotheses to evaluate (>0)",
    default=100,
)
@click.option(
    "--min-ray-length",
    type=float,
    help="Minimum length of rays to filter unwanted points near the sensor (>0)",
    default=0.1,
)
@click.option(
    "--ground-compat-max-distance",
    type=float,
    help="Max distance between valid ground plane hypotheses and the ground control point under the ego vehicle",
    default=0.5,
)
@click.option(
    "--ground-compat-max-angle-deg",
    type=float,
    help="Max angle between valid ground plane hypotheses and the up direction",
    default=60.0,
)
@click.option(
    "--plane-max-distance",
    type=float,
    help="Max distance of a point from a plane hypothesis to be considered inlier",
    default=0.3,
)
@click.option(
    "--plane-max-angle-deg",
    type=float,
    help="Max angle between a point normal and the normal of a plane hypothesis for the point to be considered inlier",
    default=30.0,
)
@click.option(
    "--plane-min-eval-points",
    type=int,
    help="Min required number of points to evaluate plane hypotheses (>0), skipping segmentation if not reached",
    default=1000,
)
@click.option(
    "--plane-max-eval-points",
    type=int,
    help="Max number of points to evaluate plane hypotheses (>0), subsampling if exceeded",
    default=10000,
)
@click.option(
    "--plane-min-inliers",
    type=int,
    help="Min number of inlier points to accept a segmentation result (>0)",
    default=1000,
)
@click.option(
    "--enable-plane-extension",
    is_flag=True,
    help="Allow all points of a spin that are compatible with the dominant plane to be marked as ground point",
    default=False,
)
@click.option(
    "--voxel-size",
    type=float,
    help="Voxel size (in world units) for point cloud downsampling to be applied before meshing",
    default=0.1,
)
@click.option(
    "--min-points-per-voxel",
    type=int,
    help="Minimum number of points per voxel",
    default=1,
)
@click.option(
    "--grid-hole-fill-min-neighbors",
    type=click.IntRange(min=0, max=8),
    help="Fill empty grid cells that have at least this many occupied 8-neighbors; 0 disables filling",
    default=6,
)
@click.option(
    "--smoothing-passes",
    type=int,
    help="Number of mesh smoothing passes to apply (>=0)",
    default=10,
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Verbose mode",
    default=False,
)
@click.option(
    "--valid-points-only",
    is_flag=True,
    help="Only use valid points for ground mesh reconstruction",
    default=False,  # BUG:5195915 WAR
)
@click.option(
    "--colored",
    is_flag=True,
    help="Color mesh vertices by projecting camera RGB onto LiDAR points",
    default=False,
)
@click.option(
    "--z-offset",
    type=float,
    default=None,
    help="Ground height in ego/rig space for the ground control point; defaults to "
    "checkpoint.artifact.mesh.ground.z_offset from config, or 0.0",
)
@click.option(
    "--extend-to-cuboids",
    is_flag=True,
    help="Extend the ground mesh to cuboid track positions, to cover areas with no nearby scanned ground points.",
    default=False,
)
@click.option(
    "--cuboids-base-margin",
    type=click.FloatRange(min=0),
    help=(
        "Margin in meters added to each cuboid's base rectangle before sampling ground points from the cuboid. "
        "Only effective with --extend-to-cuboids."
    ),
    default=0.5,
)
@click.option(
    "--cuboids-exclude-points-radius",
    type=click.FloatRange(min=0),
    help=(
        "2D radius in meters to decide if a cuboid ground point is covered by scanned ground points. "
        "A point is covered if the k nearest scan points are all within this radius. "
        "Only effective with --extend-to-cuboids."
    ),
    default=1.0,
)
@click.option(
    "--cuboids-exclude-points-knns",
    type=click.IntRange(min=1),
    help=(
        "Nearest scan point neighbours queried per cuboid ground point for coverage check. "
        "Only effective with --extend-to-cuboids."
    ),
    default=1,
)
@click.option(
    "--cuboid-point-merge-cell-size",
    type=click.FloatRange(min=0, min_open=True),
    help=("XY grid cell size in meters used to merge cuboid ground samples. Only effective with --extend-to-cuboids."),
    default=0.5,
)
@click.argument("hydra-args", nargs=-1)
def export_ground_mesh(
    config_name: str,
    output_dir: str,
    step_frame: int,
    export_per_frame_diagnostics: bool,
    export_meshing_diagnostics: bool,
    num_plane_hypotheses: int,
    min_ray_length: float,
    ground_compat_max_distance: float,
    ground_compat_max_angle_deg: float,
    plane_max_distance: float,
    plane_max_angle_deg: float,
    plane_min_eval_points: int,
    plane_max_eval_points: int,
    plane_min_inliers: int,
    enable_plane_extension: bool,
    voxel_size: float,
    min_points_per_voxel: int,
    grid_hole_fill_min_neighbors: int,
    smoothing_passes: int,
    verbose: bool,
    valid_points_only: bool,
    colored: bool,
    z_offset: float | None,
    extend_to_cuboids: bool,
    cuboids_base_margin: float,
    cuboids_exclude_points_radius: float,
    cuboids_exclude_points_knns: int,
    cuboid_point_merge_cell_size: float,
    hydra_args: list[str],
):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
    config = parse_untyped_config(config_name=config_name, hydra_args=hydra_args)

    rig_z_offset = z_offset if z_offset is not None else config.checkpoint.artifact.mesh.ground.z_offset

    # Check parameters of the algorithm
    if num_plane_hypotheses < 1:
        raise ValueError("num_plane_hypotheses must be positive")
    if min_ray_length <= 0.0:
        raise ValueError("min_ray_length must be positive")
    if ground_compat_max_distance <= 0.0:
        raise ValueError("ground_compat_max_distance must be positive")
    if ground_compat_max_angle_deg <= 0.0:
        raise ValueError("ground_compat_max_angle_deg must be positive")
    if plane_max_distance <= 0.0:
        raise ValueError("plane_max_distance must be positive")
    if plane_max_angle_deg <= 0.0:
        raise ValueError("plane_max_angle_deg must be positive")
    if num_plane_hypotheses < 1:
        raise ValueError("num_plane_hypotheses must be positive")
    if num_plane_hypotheses < 1:
        raise ValueError("num_plane_hypotheses must be positive")
    if num_plane_hypotheses < 1:
        raise ValueError("num_plane_hypotheses must be positive")
    if plane_min_eval_points < 1:
        raise ValueError("plane_min_eval_points must be positive")
    if plane_max_eval_points < 1:
        raise ValueError("plane_max_eval_points must be positive")
    if plane_min_inliers < 1:
        raise ValueError("plane_min_inliers must be positive")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    if min_points_per_voxel < 1:
        raise ValueError("min_points_per_voxel must be positive")
    if not (0 <= grid_hole_fill_min_neighbors <= 8):
        raise ValueError("grid_hole_fill_min_neighbors must be in [0, 8]")
    if smoothing_passes <= 0.0:
        raise ValueError("voxel_size must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    datasource = make_dataset(config.dataset.name, config, split="train").get_datasource()
    if not isinstance(datasource, NCOREDataSource):
        raise NotImplementedError("Only NCORE datasets are supported")

    # Yields a point cloud per Lidar spin.
    point_cloud_generator = datasource.get_point_clouds(
        device=torch.device("cpu"),
        lidar_ids=None,
        camera_ids=None,
        valid_points_only=valid_points_only,
        non_dynamic_points_only=True,
        color_type="camera-rgb" if colored else None,
        step_frame=step_frame,
    )

    # Get the cuboid tracks in world frame in order to extend the ground mesh to the cuboids.
    # Areas far from the ego vehicle tend to not have scanned points on the ground but have detected cuboids.
    cuboid_tracks = datasource.get_cuboid_tracks(world_frame=True) if extend_to_cuboids else None

    (
        smooth_vertices,
        triangles,
        vertices,
        points,
        normals,
        road_mask,
        initial_road_mask,
        vertex_colors,
        cuboid_ground_points,
    ) = reconstruct_ground_mesh_from_points(
        point_cloud_generator=point_cloud_generator,
        min_ray_length=min_ray_length,
        ground_control_point=get_nominal_ground_point_under_lidar(datasource, rig_z_offset=rig_z_offset),
        ground_compat_max_distance=ground_compat_max_distance,
        ground_compat_max_angle_deg=ground_compat_max_angle_deg,
        num_plane_hypotheses=num_plane_hypotheses,
        plane_max_distance=plane_max_distance,
        plane_max_angle_deg=plane_max_angle_deg,
        plane_min_eval_points=plane_min_eval_points,
        plane_max_eval_points=plane_max_eval_points,
        plane_min_inliers=plane_min_inliers,
        enable_plane_extension=enable_plane_extension,
        voxel_size=voxel_size,
        min_points_per_voxel=min_points_per_voxel,
        grid_hole_fill_min_neighbors=grid_hole_fill_min_neighbors,
        smoothing_passes=smoothing_passes,
        export_per_frame_diagnostics=export_per_frame_diagnostics,
        output_path=output_path,
        points_to_world_transf=datasource.world_to_nurec.inverse(),
        verbose=verbose,
        cuboid_tracks=cuboid_tracks,
        cuboids_base_margin=cuboids_base_margin,
        cuboids_exclude_points_radius=cuboids_exclude_points_radius,
        cuboids_exclude_points_knns=cuboids_exclude_points_knns,
        cuboid_point_merge_cell_size=cuboid_point_merge_cell_size,
    )

    if len(smooth_vertices) < 3 or len(triangles) == 0:
        raise click.ClickException(
            f"Ground-mesh reconstruction produced a degenerate mesh "
            f"({len(smooth_vertices)} vertices, {len(triangles)} triangles)"
        )

    save_ply(str(output_path / "mesh_ground.ply"), smooth_vertices, triangles, colors=vertex_colors, logger=logger)

    if export_meshing_diagnostics:
        save_ply(str(output_path / "mesh_ground_before_smoothing.ply"), vertices, triangles, logger=logger)
        save_ply(
            str(output_path / "input_points_road_prior.ply"),
            vertices=points[initial_road_mask],
            normals=normals[initial_road_mask],
            logger=logger,
        )
        save_ply(
            str(output_path / "input_points_road.ply"),
            vertices=points[road_mask],
            normals=normals[road_mask],
            logger=logger,
        )
        nonroad_mask = np.logical_not(road_mask)
        save_ply(
            str(output_path / "input_points_nonroad.ply"),
            vertices=points[nonroad_mask],
            normals=normals[nonroad_mask],
            logger=logger,
        )

        if cuboid_ground_points is not None and len(cuboid_ground_points) > 0:
            save_ply(str(output_path / "cuboid_ground_points.ply"), cuboid_ground_points, logger=logger)

        if cuboid_tracks is not None:
            colors_per_track = generate_track_colors(cuboid_tracks.tracks_id)
            save_cuboids_ply(
                cuboid_tracks,
                str(output_path / "cuboids_dynamic.ply"),
                colors_per_track,
                require_flags=TrackFlags.DYNAMIC,
            )
            save_cuboids_ply(
                cuboid_tracks,
                str(output_path / "cuboids_static.ply"),
                colors_per_track,
                exclude_flags=TrackFlags.DYNAMIC,
            )
