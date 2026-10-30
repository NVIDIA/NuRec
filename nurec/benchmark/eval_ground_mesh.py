# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import io
import json
import logging
import os
import time

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import point_cloud_utils as pcu
import trimesh
import yaml

from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull, Delaunay, QhullError

from nurec.artifact import Artifact
from nurec.benchmark.mesh_processing import find_boundaries
from nurec.datasets.tracks import CuboidTracks, Tracks
from nurec.utils.cuboid import cuboid_base_grid_local, cuboid_corners_local
from nurec.utils.geometry import tquat_to_se3_matrix
from nurec.utils.io.ply import save_ply  # type: ignore
from nurec.utils.io.sequence_tracks import generate_track_colors
from nurec.utils.types import RigTrajectories, TrackFlags


log = logging.getLogger(__name__)


class Polygon:
    def __init__(self, vertices: NDArray[np.float32]):
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3 or vertices.shape[1] == 2
        self._vertices = vertices

    @property
    def vertices(self) -> NDArray[np.float32]:
        return self._vertices

    @property
    def num_vertices(self) -> int:
        return len(self._vertices)

    def total_length(self) -> float:
        return (
            np.sum(np.linalg.norm(self._vertices[1:] - self._vertices[:-1], axis=1)).item()
            if len(self._vertices) > 1
            else 0.0
        )

    def uniform_resampling(self, num_samples: int) -> NDArray[np.float32]:
        assert num_samples > 1, "Number of samples must be greater than 1"

        # Arc-length parameters per vertex, i.e. distance of each vertex along the polygon from the first vertex.
        vertex_arclen = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(self._vertices[1:] - self._vertices[:-1], axis=1))]
        )
        sample_arclen = np.linspace(0, vertex_arclen[-1], num_samples)

        # Linear interpolation of the vertices to the desired number of samples.
        xi = np.interp(sample_arclen, vertex_arclen, self._vertices[:, 0]).astype(np.float32)
        yi = np.interp(sample_arclen, vertex_arclen, self._vertices[:, 1]).astype(np.float32)
        zi = np.interp(sample_arclen, vertex_arclen, self._vertices[:, 2]).astype(np.float32)

        return np.stack([xi, yi, zi], axis=1).astype(np.float32)


def load_ground_mesh_from_artifact(artifact: Artifact) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
    mesh_bytes_io = io.BytesIO(artifact.ground_mesh_ply)
    mesh = trimesh.load(mesh_bytes_io, file_type="ply")
    return np.array(mesh.vertices).astype(np.float32), np.array(mesh.faces).astype(np.int32)


def load_ground_mesh_from_ply(ply_path: str) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
    mesh = trimesh.load(ply_path, file_type="ply")
    return np.array(mesh.vertices).astype(np.float32), np.array(mesh.faces).astype(np.int32)


def load_rig_trajectory_from_dict(rig_trajectories_dict: Dict[str, Any]) -> NDArray[np.float32]:
    """Import rig poses from a dictionary output by the NuRec training and return them as a (N, 4, 4) numpy array"""
    rig_trajectories = RigTrajectories.from_dict(rig_trajectories_dict)
    if len(rig_trajectories.rig_trajectories) == 0:
        raise ValueError("No rig trajectories found in the artifact")
    rig_trajectory = rig_trajectories.rig_trajectories[0]
    T_rig_worlds = rig_trajectory.T_rig_worlds
    return T_rig_worlds.cpu().numpy().astype(np.float32)


def load_rig_trajectory_from_artifact(artifact: Artifact) -> NDArray[np.float32]:
    """Import rig poses from an artifact and return them as a (N, 4, 4) numpy array"""
    return load_rig_trajectory_from_dict(artifact.rig_trajectories)


def load_rig_trajectory_from_json(json_path: str) -> NDArray[np.float32]:
    """Import rig poses from a JSON file output by the NuRec training and return them as a (N, 4, 4) numpy array"""
    with open(json_path, "r") as f:
        rig_trajectories_dict = json.load(f)
    return load_rig_trajectory_from_dict(rig_trajectories_dict)


def _load_cuboid_tracks_from_dict(sequence_tracks_dict: Dict[str, Any]) -> CuboidTracks:
    """Deserialize a sequence_tracks dict (from JSON) into a single CuboidTracks instance.

    Raises ValueError if the dict does not contain exactly one sequence.
    """
    if len(sequence_tracks_dict) != 1:
        raise ValueError(
            f"Expected exactly one sequence in sequence_tracks, got {len(sequence_tracks_dict)}: "
            f"{list(sequence_tracks_dict.keys())}"
        )
    return CuboidTracks.from_dict(next(iter(sequence_tracks_dict.values())))


def load_cuboid_tracks_from_artifact(artifact: Artifact) -> Optional[CuboidTracks]:
    """Load cuboid tracks from a USDZ artifact.

    Returns None if sequence_tracks.json is not present in the artifact.
    Raises ValueError if the artifact contains more than one sequence.
    """
    try:
        sequence_tracks = artifact.sequence_tracks
    except KeyError:
        return None
    return _load_cuboid_tracks_from_dict(sequence_tracks)


def load_cuboid_tracks_from_json(json_path: str) -> CuboidTracks:
    """Load cuboid tracks from a standalone sequence_tracks JSON file.

    Raises ValueError if the file contains more than one sequence.
    """
    with open(json_path, "r") as f:
        sequence_tracks_dict = json.load(f)
    return _load_cuboid_tracks_from_dict(sequence_tracks_dict)


def save_cuboids_ply(
    cuboids_per_track: Dict[str, List[NDArray[np.float32]]],
    colors_per_track: Dict[str, NDArray[np.uint8]],
    output_path: str,
    per_track: bool = False,
) -> None:
    """Save cuboid meshes as PLY.

    Always writes a single combined PLY at ``output_path``.  When *per_track*
    is True, also writes one PLY per track into a directory derived by stripping
    the ``.ply`` extension from *output_path*.

    Args:
        cuboids_per_track: Dict mapping track_id to per-pose corner arrays.
        colors_per_track: Per-track RGB color, e.g. from generate_track_colors().
        output_path: Path for the combined PLY file, ``/path/to/output.ply``.
            When *per_track* is True, per-track files are saved under ``/path/to/output/``.
        per_track: If True, also write one PLY per track inside the directory
            obtained by stripping the ``.ply`` extension from *output_path*.
    """
    # 12 triangles per cuboid with outward normals (CCW), shape (1, 12, 3) for broadcasting over poses.
    cuboid_faces = np.array(
        [
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 1],
                [1, 4, 5],
                [2, 6, 7],
                [2, 7, 3],
                [0, 3, 7],
                [0, 7, 4],
                [1, 5, 2],
                [2, 5, 6],
            ]
        ],
        dtype=np.int32,
    )

    all_vertices: List[NDArray[np.float32]] = []
    all_faces: List[NDArray[np.int32]] = []
    all_colors: List[NDArray[np.uint8]] = []
    vertex_offset = 0

    for track_id, corners_per_pose in cuboids_per_track.items():
        n_poses = len(corners_per_pose)
        if n_poses == 0:
            continue

        vertices = np.concatenate(corners_per_pose, axis=0)  # (n_poses * 8, 3)
        face_offset = np.arange(n_poses, dtype=np.int32)[:, None, None] * 8
        faces = (cuboid_faces + face_offset).reshape(-1, 3)  # (n_poses * 12, 3)
        colors = np.tile(colors_per_track[track_id], (len(vertices), 1)).astype(np.uint8)

        if per_track:
            per_track_dir = os.path.splitext(output_path)[0]
            os.makedirs(per_track_dir, exist_ok=True)
            save_ply(
                os.path.join(per_track_dir, f"{track_id}.ply"),
                vertices=vertices,
                triangles=faces,
                colors=colors,
                logger=log,
            )

        all_vertices.append(vertices)
        all_faces.append(faces + vertex_offset)
        all_colors.append(colors)
        vertex_offset += len(vertices)

    if all_vertices:
        save_ply(
            output_path,
            vertices=np.concatenate(all_vertices),
            triangles=np.concatenate(all_faces),
            colors=np.concatenate(all_colors),
            logger=log,
        )


def calculate_cuboid_coverage_by_ground_mesh(
    cuboid_tracks: CuboidTracks,
    mesh_vertices: NDArray[np.float32],
    mesh_faces: NDArray[np.int32],
    output_dir: Optional[str] = None,
    base_grid_n: int = 4,
) -> Dict[str, float]:
    """Calculate what fraction of cuboid track poses are covered by the ground mesh.

    For every pose of every track, rays are shot from an NxN grid of samples over
    the cuboid base (local z=-dz plane) in both the cuboid's local +Z and -Z
    world-frame directions.  A pose is considered covered when all N*N samples
    produce a valid mesh intersection in at least one direction.

    When output_dir is provided, exports:
      - cuboids_dynamic.ply: cuboid meshes for DYNAMIC+CONTROLLABLE tracks, colored per track
      - cuboids_static.ply: cuboid meshes for all other tracks, colored per track
      - cuboid_base_samples.ply: all base grid samples, colored per track
      - cuboid_projected_samples.ply: ray-mesh intersection points for all valid
        samples, colored per track

    Args:
        cuboid_tracks: Loaded CuboidTracks instance.
        mesh_vertices: (M, 3) array of mesh vertices.
        mesh_faces: (F, 3) array of mesh faces.
        output_dir: Optional directory for PLY exports.
        base_grid_n: Grid resolution N; shoots rays from an N×N sample grid.

    Returns:
        Dict with the following fields:

        - ``cuboid_tracks_count``: total number of cuboid tracks.
        - ``cuboid_poses_count``: total number of poses across all tracks.
        - ``cuboid_ground_coverage_percentage``: percentage of poses covered by
          the ground mesh, aggregated over all tracks.
        - ``cuboid_ground_distance_med_meters``: median ray-mesh hit distance
          across all samples of all covered poses (0.0 if no pose is covered).
        - ``cuboid_ground_distance_max_meters``: maximum ray-mesh hit distance
          across all samples of all covered poses (0.0 if no pose is covered).
        - ``dynamic_cuboid_tracks_count``: number of DYNAMIC+CONTROLLABLE tracks.
        - ``dynamic_cuboid_poses_count``: total poses for DYNAMIC+CONTROLLABLE tracks.
        - ``dynamic_cuboid_ground_coverage_percentage``: same as
          ``cuboid_ground_coverage_percentage`` but for DYNAMIC+CONTROLLABLE tracks.
        - ``dynamic_cuboid_ground_distance_med_meters``: same as
          ``cuboid_ground_distance_med_meters`` but for DYNAMIC+CONTROLLABLE tracks.
        - ``dynamic_cuboid_ground_distance_max_meters``: same as
          ``cuboid_ground_distance_max_meters`` but for DYNAMIC+CONTROLLABLE tracks.
    """
    unpacked = Tracks.Unpacked.from_data(cuboid_tracks.tracks_data)
    cuboids_dims = cuboid_tracks.cuboids_dims.cpu().numpy()  # (N_tracks, 3)
    colors_per_track = generate_track_colors(unpacked.tracks_id)

    n_samples = base_grid_n**2

    intersector = pcu.RayMeshIntersector(mesh_vertices, mesh_faces)

    total_poses = 0
    covered_poses = 0
    dynamic_total_tracks = 0
    dynamic_total_poses = 0
    dynamic_covered_poses = 0

    dynamic_cuboids_per_track: Dict[str, List[NDArray[np.float32]]] = {}
    static_cuboids_per_track: Dict[str, List[NDArray[np.float32]]] = {}

    # Needed for diagnostic PLY exports
    all_base_sample_vertices: List[NDArray[np.float32]] = []
    all_base_sample_colors: List[NDArray[np.uint8]] = []
    all_footprint_sample_vertices: List[NDArray[np.float32]] = []
    all_footprint_sample_colors: List[NDArray[np.uint8]] = []
    all_sample_distances: List[NDArray[np.float32]] = []
    all_dynamic_sample_distances: List[NDArray[np.float32]] = []

    # Per-track statistics for diagnostics
    cuboid_stats: Dict[str, List] = {
        "track_id": [],
        "category": [],
        "is_dynamic": [],
        "is_controllable": [],
        "trajectory_length": [],
        "travelled_distance": [],
        "num_poses": [],
        "num_poses_covered_by_ground": [],
        "num_samples": [],
        "num_samples_covered_by_ground": [],
        "max_ground_distance": [],
    }

    log.info(f"Calculating cuboid coverage metrics for {len(cuboid_tracks.tracks_id)} cuboid tracks")
    start_timestamp = time.perf_counter()

    # Cycle through each cuboid track
    for track_idx, track_id in enumerate(unpacked.tracks_id):
        cuboid_dims = cuboids_dims[track_idx]  # (3,)
        tquats = np.array(unpacked.tracks_poses[track_idx], dtype=np.float32)  # (n_poses, 7)
        T_track_worlds = tquat_to_se3_matrix(tquats, unbatch=False).cpu().numpy()  # (n_poses, 4, 4)

        n_poses = len(T_track_worlds)
        if n_poses == 0:
            continue
        total_poses += n_poses

        _dynamic_flags = TrackFlags.DYNAMIC | TrackFlags.CONTROLLABLE
        is_dynamic_and_controllable = (unpacked.tracks_flags[track_idx] & _dynamic_flags) == _dynamic_flags

        # List of length n_poses, where each element is a (8, 3) array — 8 world-frame corners per cuboid pose
        corners_local = cuboid_corners_local(cuboid_dims)  # (8, 3)
        # Point transform: R*p + t (not the normal/inverse-transpose form)
        corners_per_pose = [((T[:3, :3] @ corners_local.T).T + T[:3, 3]).astype(np.float32) for T in T_track_worlds]
        if is_dynamic_and_controllable:
            dynamic_cuboids_per_track[track_id] = corners_per_pose
        else:
            static_cuboids_per_track[track_id] = corners_per_pose

        # List of (n_samples, 3) - samples over the cuboid base in its local frame (n_samples = base_grid_n**2)
        base_samples_local = cuboid_base_grid_local(cuboid_dims, base_grid_n)  # (n_samples, 3)

        # List of length n_poses, each element is a (n_samples, 3) array of cuboid base samples in world frame
        base_samples_per_pose = [
            ((T[:3, :3] @ base_samples_local.T).T + T[:3, 3]).astype(np.float32) for T in T_track_worlds
        ]

        # Cuboid local +Z in world frame, one vector per pose
        up_dirs = T_track_worlds[:, :3, 2].astype(np.float32)  # (n_poses, 3)

        base_samples_flat = np.stack(base_samples_per_pose).reshape(-1, 3)  # (n_poses*n_samples, 3)
        up_dirs_tiled = np.repeat(up_dirs, n_samples, axis=0)  # (n_poses*n_samples, 3)

        fid_down, _, dist_down = intersector.intersect_rays(base_samples_flat, -up_dirs_tiled)
        fid_up, _, dist_up = intersector.intersect_rays(base_samples_flat, up_dirs_tiled)

        valid_down = (fid_down >= 0) & np.isfinite(dist_down)  # (n_poses*n_samples,)
        valid_up = (fid_up >= 0) & np.isfinite(dist_up)  # (n_poses*n_samples,)
        # A sample is covered if either direction hits the mesh.
        sample_valid = valid_down | valid_up  # (n_poses*n_samples,)
        # Prefer down-direction hit, fall back to up.
        ray_dist = np.where(valid_down, dist_down, dist_up)  # (n_poses*n_samples,)
        ray_dirs = np.where(valid_down[:, np.newaxis], -up_dirs_tiled, up_dirs_tiled)  # (n_poses*n_samples, 3)

        # A pose is covered if all n_samples of its grid points are valid.
        pose_covered = sample_valid.reshape(n_poses, n_samples).all(axis=1)  # (n_poses,)
        covered_poses += int(pose_covered.sum())

        if is_dynamic_and_controllable:
            dynamic_total_tracks += 1
            dynamic_total_poses += n_poses
            dynamic_covered_poses += int(pose_covered.sum())

        footprint_pts = base_samples_flat + ray_dirs * ray_dist[:, np.newaxis]  # (n_poses*n_samples, 3)

        # Accumulate base samples and valid footprint points for PLY export.
        track_color = colors_per_track[track_id]
        all_base_sample_vertices.append(base_samples_flat)
        all_base_sample_colors.append(np.tile(track_color, (len(base_samples_flat), 1)).astype(np.uint8))

        valid_footprints = footprint_pts[sample_valid]
        if len(valid_footprints) > 0:
            all_footprint_sample_vertices.append(valid_footprints.astype(np.float32))
            all_footprint_sample_colors.append(np.tile(track_color, (len(valid_footprints), 1)).astype(np.uint8))

        # Per-track statistics for diagnostics
        positions = T_track_worlds[:, :3, 3]  # (n_poses, 3)
        cuboid_stats["track_id"].append(track_id)
        cuboid_stats["category"].append(unpacked.tracks_label_class[track_idx])
        cuboid_stats["is_dynamic"].append(bool(unpacked.tracks_flags[track_idx] & TrackFlags.DYNAMIC))
        cuboid_stats["is_controllable"].append(bool(unpacked.tracks_flags[track_idx] & TrackFlags.CONTROLLABLE))
        cuboid_stats["trajectory_length"].append(float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()))
        cuboid_stats["travelled_distance"].append(float(np.linalg.norm(positions[-1] - positions[0])))
        cuboid_stats["num_poses"].append(n_poses)
        cuboid_stats["num_poses_covered_by_ground"].append(int(pose_covered.sum()))
        cuboid_stats["num_samples"].append(n_poses * n_samples)
        cuboid_stats["num_samples_covered_by_ground"].append(int(sample_valid.sum()))
        cuboid_stats["max_ground_distance"].append(float(ray_dist[sample_valid].max()) if sample_valid.any() else None)

        # Distances for covered poses only.
        covered_indices = np.where(pose_covered)[0]
        if len(covered_indices) == 0:
            continue

        covered_sample_dists = ray_dist.reshape(n_poses, n_samples)[covered_indices].reshape(-1)
        all_sample_distances.append(covered_sample_dists.astype(np.float32))
        if is_dynamic_and_controllable:
            all_dynamic_sample_distances.append(covered_sample_dists.astype(np.float32))

    log.info(f"Calculated cuboid coverage metrics in {time.perf_counter() - start_timestamp:.2f} seconds")

    if output_dir is not None:
        save_cuboids_ply(dynamic_cuboids_per_track, colors_per_track, os.path.join(output_dir, "cuboids_dynamic.ply"))
        save_cuboids_ply(static_cuboids_per_track, colors_per_track, os.path.join(output_dir, "cuboids_static.ply"))
        if all_base_sample_vertices:
            save_ply(
                os.path.join(output_dir, "cuboid_base_samples.ply"),
                vertices=np.concatenate(all_base_sample_vertices),
                colors=np.concatenate(all_base_sample_colors),
                logger=log,
            )
        if all_footprint_sample_vertices:
            save_ply(
                os.path.join(output_dir, "cuboid_projected_samples.ply"),
                vertices=np.concatenate(all_footprint_sample_vertices),
                colors=np.concatenate(all_footprint_sample_colors),
                logger=log,
            )
        cuboid_stats_path = os.path.join(output_dir, "cuboid_stats.csv")
        with open(cuboid_stats_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cuboid_stats.keys())
            writer.writerows([("" if v is None else v) for v in row] for row in zip(*cuboid_stats.values()))
        log.info(f"Saved cuboid statistics to {cuboid_stats_path}")

    coverage = covered_poses / total_poses if total_poses > 0 else 1.0
    log.info(
        f"Cuboid ground coverage (all cuboids): {100.0 * coverage:.2f}% ({covered_poses}/{total_poses} poses covered)"
    )
    dynamic_coverage = dynamic_covered_poses / dynamic_total_poses if dynamic_total_poses > 0 else 1.0
    log.info(
        f"Cuboid ground coverage (controllables): {100.0 * dynamic_coverage:.2f}%"
        f" ({dynamic_covered_poses}/{dynamic_total_poses} poses covered)"
    )

    if all_sample_distances:
        distances = np.concatenate(all_sample_distances)
        dist_med = float(np.median(distances))
        dist_max = float(np.max(distances))
    else:
        dist_med = 0.0
        dist_max = 0.0

    if all_dynamic_sample_distances:
        dynamic_distances = np.concatenate(all_dynamic_sample_distances)
        dynamic_dist_med = float(np.median(dynamic_distances))
        dynamic_dist_max = float(np.max(dynamic_distances))
    else:
        dynamic_dist_med = 0.0
        dynamic_dist_max = 0.0

    return {
        "cuboid_tracks_count": len(unpacked.tracks_id),
        "cuboid_poses_count": total_poses,
        "cuboid_ground_coverage_percentage": 100.0 * coverage,
        "cuboid_ground_distance_med_meters": dist_med,
        "cuboid_ground_distance_max_meters": dist_max,
        "dynamic_cuboid_tracks_count": dynamic_total_tracks,
        "dynamic_cuboid_poses_count": dynamic_total_poses,
        "dynamic_cuboid_ground_coverage_percentage": 100.0 * dynamic_coverage,
        "dynamic_cuboid_ground_distance_med_meters": dynamic_dist_med,
        "dynamic_cuboid_ground_distance_max_meters": dynamic_dist_max,
    }


def calculate_vehicle_trajectory_vs_ground_consistency_metrics(
    mesh_vertices: NDArray[np.float32],
    mesh_faces: NDArray[np.int32],
    T_rig_worlds: NDArray[np.float32],
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Compare the rig trajectory with the ground mesh and calculate coverage and distance metrics.

    Assumption: rig positions are on the nominal ground under the vehicle. A discrepancy between the rig positions
    and the ground mesh may only be due to vehicle suspension.

    Args:
        mesh_vertices: (M, 3) array of mesh vertices.
        mesh_faces: (F, 3) array of mesh faces.
        T_rig_worlds: (N, 4, 4) array of poses, as 4x4 transformations from the rig frame to the world frame.
        output_dir: Optional directory path for exporting diagnostic data.

    Returns:
        A dictionary of metrics that maps metric name to value.
    """

    # Defaulting to the worst possible values to make it possible to return from the function in case of an error.
    metrics = {
        "trajectory_sample_count": len(T_rig_worlds),
        "trajectory_length_meters": 0.0,
        "trajectory_ground_coverage_percentage": 0.0,
        "trajectory_ground_distance_max_meters": np.inf,
        "trajectory_ground_distance_med_meters": np.inf,
    }

    if len(mesh_vertices) < 3 or len(mesh_faces) == 0:
        log.warning("Mesh does not have a single valid face, skipping metrics")
        return metrics

    if len(T_rig_worlds) == 0:
        log.warning("Rig trajectory is empty, skipping trajectory-based metrics")
        return metrics

    # Extract rig positions from the rig trajectory.
    # Rig positions are expected in the world frame, on the nominal ground under the vehicle (NCore convention).
    # Spacing of adjacent positions is based on vehicle speed and is non-uniform.
    rig_positions = np.ascontiguousarray(T_rig_worlds[:, :3, 3]).astype(np.float32)

    # Add trajectory length to the metrics.
    rig_positions_polygon = Polygon(rig_positions)
    trajectory_length_meters = rig_positions_polygon.total_length()
    metrics["trajectory_length_meters"] = trajectory_length_meters
    log.info(f"Trajectory length: {trajectory_length_meters:.2f} meters")

    if trajectory_length_meters == 0.0:
        log.warning("Trajectory length is zero, skipping trajectory-based metrics")
        return metrics

    # Extract the per-sample vehicle up and lateral directions in the world frame.
    # These are the vehicle Z- and Y-axis directions in the world frame.
    vehicle_up_directions = np.ascontiguousarray(T_rig_worlds[:, :3, 2])

    # Note that any per-vertex shift of a curve may cause order reversion in space, so smoothness is not preserved,
    # i.e. do not calculate smoothness metrics on shifted curves in the future.
    rig_positions_up_shifted = rig_positions + vehicle_up_directions

    # Rig positions are in the nominal ground under the vehicle by NCore convention.
    # Project rig positions shifted in the vehicle up direction back onto the mesh in the vehicle down direction.
    # If there is no intersection in ray direction from the ray origin, the face index is -1 and the distance is inf.
    # The intersections are in the direction of the ray, not in the opposite direction, so distances are not signed.
    intersector = pcu.RayMeshIntersector(mesh_vertices, mesh_faces)
    ray_origins = rig_positions_up_shifted
    # ray_origins = rig_positions - vehicle_up_directions # For testing when there is no valid ground projection
    ray_directions = -vehicle_up_directions
    face_indices, bary_coords, ray_intersection_distances = intersector.intersect_rays(ray_origins, ray_directions)

    ground_projection_mask = (face_indices >= 0) & np.isfinite(ray_intersection_distances)
    ground_projections = ray_origins + ray_directions * ray_intersection_distances.reshape(-1, 1)
    ground_projections = ground_projections[ground_projection_mask]

    metrics["trajectory_ground_coverage_percentage"] = 100.0 * ground_projection_mask.sum().item() / len(ray_origins)
    log.info(f"Trajectory ground coverage: {metrics['trajectory_ground_coverage_percentage']:.2f}%")

    # Calculate trajectory (on nominal ground) vs. ground mesh distance on rig positions covered by the ground mesh.
    if len(ground_projections) > 0:
        trajectory_ground_distances = np.linalg.norm(ground_projections - rig_positions[ground_projection_mask], axis=1)
        metrics["trajectory_ground_distance_max_meters"] = trajectory_ground_distances.max().item()
        metrics["trajectory_ground_distance_med_meters"] = float(np.median(trajectory_ground_distances))
        log.info(f"Trajectory-ground distance max: {metrics['trajectory_ground_distance_max_meters']:.2f} meters")
        log.info(f"Trajectory-ground distance median: {metrics['trajectory_ground_distance_med_meters']:.2f} meters")
    else:
        log.warning("No ground projections found, skipping trajectory-ground distance metrics")

    if output_dir is not None:
        # Save rig positions to a PLY for visualization
        save_ply(os.path.join(output_dir, "rig_positions.ply"), vertices=rig_positions, logger=log)
        save_ply(
            os.path.join(output_dir, "rig_positions_up_shifted.ply"), vertices=rig_positions_up_shifted, logger=log
        )
        save_ply(os.path.join(output_dir, "trajectory_ground_projections.ply"), vertices=ground_projections, logger=log)

    return metrics


def _rasterize_mesh(
    mesh_vertices: NDArray[np.float32],
    mesh_faces: NDArray[np.int32],
    grid_points: NDArray[np.float32],
) -> NDArray[np.bool_]:
    """Return a boolean mask indicating which 2D grid points fall within the top-view projection of the mesh.

    Casts a vertical downward ray from above each grid point and marks those that hit a mesh face.
    This respects holes and non-convex boundaries; a Delaunay triangulation of vertex XY would fill the
    convex hull and overestimate coverage for typical ground meshes.

    Args:
        mesh_vertices: (M, 3) array of mesh vertex positions.
        mesh_faces: (F, 3) array of mesh face indices.
        grid_points: (N, 2) array of XY grid point positions to test.

    Returns:
        (N,) boolean array; True where a downward ray from the grid point hits the mesh.
    """
    if len(mesh_vertices) < 3 or len(mesh_faces) == 0:
        return np.zeros(len(grid_points), dtype=bool)
    intersector = pcu.RayMeshIntersector(mesh_vertices, mesh_faces)
    ray_origins_z = float(mesh_vertices[:, 2].max()) + 1.0
    n = len(grid_points)
    ray_origins = np.concatenate(
        [grid_points.astype(np.float32), np.full((n, 1), ray_origins_z, dtype=np.float32)], axis=1
    )
    ray_dirs = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (n, 1))
    face_idx, _, ray_dist = intersector.intersect_rays(ray_origins, ray_dirs)
    return (face_idx >= 0) & np.isfinite(ray_dist)


def _rasterize_points(
    points_xy: NDArray[np.float32],
    grid_points: NDArray[np.float32],
    grid_resolution_meters: float,
) -> NDArray[np.bool_]:
    """Return a boolean mask indicating which grid cells contain at least one point.

    Each input point is mapped to the nearest grid cell by snapping its XY coordinates to the grid.

    Args:
        points_xy: (N, 2) array of XY point positions to rasterize.
        grid_points: (M, 2) array of XY grid cell centre positions (as produced by meshgrid + ravel).
        grid_resolution_meters: grid cell size in metres; used to snap points to cell indices.

    Returns:
        (M,) boolean array; True where at least one point falls within the grid cell.
    """
    # Grid convention: columns increase with world X (left→right); rows increase as world Y decreases
    # (row 0 = max Y, last row = min Y). This matches standard image display with Y pointing up.
    if len(points_xy) == 0 or len(grid_points) == 0:
        return np.zeros(len(grid_points), dtype=bool)
    x_min = grid_points[:, 0].min()
    y_max = grid_points[:, 1].max()
    col_idx = np.round((points_xy[:, 0] - x_min) / grid_resolution_meters).astype(np.int32)
    row_idx = np.round((y_max - points_xy[:, 1]) / grid_resolution_meters).astype(np.int32)
    n_cols = int(np.round((grid_points[:, 0].max() - x_min) / grid_resolution_meters)) + 1
    n_rows = int(np.round((y_max - grid_points[:, 1].min()) / grid_resolution_meters)) + 1
    valid = (col_idx >= 0) & (col_idx < n_cols) & (row_idx >= 0) & (row_idx < n_rows)
    flat_idx = row_idx[valid] * n_cols + col_idx[valid]
    mask_flat = np.zeros(n_rows * n_cols, dtype=bool)
    mask_flat[flat_idx] = True
    if n_rows * n_cols != len(grid_points):
        raise ValueError(
            f"_rasterize_points: grid shape {n_rows}×{n_cols}={n_rows * n_cols} != len(grid_points)={len(grid_points)}"
        )
    return mask_flat


def blend_colorized_masks(
    masks: List[NDArray[np.bool_]],
    colors: List[Tuple[int, int, int]],
    alphas: List[float],
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> NDArray[np.uint8]:
    """Blend a list of boolean masks with their RGB colors for diagnostic image exports.

    Each layer is black where the mask is zero and the corresponding color where nonzero.
    Layers are accumulated with their unnormalized alphas and normalized per pixel. Pixels
    where no mask is nonzero (alpha_acc == 0) are filled with background_color.

    Args:
        masks: list of (H, W) boolean arrays.
        colors: list of (R, G, B) foreground color tuples with values in [0, 255].
        alphas: list of non-negative floats controlling the relative weight of each layer.
        background_color: (R, G, B) color applied to pixels not covered by any mask; defaults to white.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    assert len(masks) == len(colors) == len(alphas)
    h, w = masks[0].shape
    rgb_acc = np.zeros((h, w, 3), dtype=np.float32)
    alpha_acc = np.zeros((h, w), dtype=np.float32)
    for mask, (r, g, b), alpha in zip(masks, colors, alphas):
        layer = np.zeros((h, w, 3), dtype=np.float32)
        layer[mask] = (r, g, b)
        rgb_acc += alpha * layer
        alpha_acc += alpha * mask.astype(np.float32)
    covered = alpha_acc > 0
    rgb_acc[covered] /= alpha_acc[covered, np.newaxis]
    rgb_acc[~covered] = background_color
    return np.clip(rgb_acc, 0, 255).astype(np.uint8)


def calculate_point_cloud_hull_coverage_metrics(
    mesh_vertices: NDArray[np.float32],
    mesh_faces: NDArray[np.int32],
    input_points: NDArray[np.float32],
    ego_trajectory: NDArray[np.float32],
    hull_margins_meters: Optional[List[int]] = None,
    grid_resolution_meters: float = 0.5,
    max_range_meters: float = 200.0,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Compute point cloud hull coverage KPIs measuring ground mesh coverage of the convex hull of a point cloud.

    For each margin r, computes what fraction of the grid cells inside the r-expanded convex hull of
    the accumulated point cloud are covered by the ground mesh (i.e. have a downward ray hit).
    Also reports the extrapolation fraction: the share of mesh-covered area lying outside the point cloud hull.

    Args:
        mesh_vertices: (M, 3) mesh vertex positions.
        mesh_faces: (F, 3) mesh face indices.
        input_points: (N, 3) array of 3D input points in world coordinates.
        ego_trajectory: (T, 3) array of ego vehicle positions used as reference for point cloud trimming.
        hull_margins_meters: list of hull expansion radii in whole metres (duplicates are ignored); defaults to [0, 50].
        grid_resolution_meters: raster cell size in metres.
        output_dir: optional directory for diagnostic image exports (occupancy_mesh.png,
            occupancy_point_cloud_hull.png, occupancy_input_points.png, occupancy_input_points_vs_hull.png,
            occupancy_input_points_vs_mesh.png, occupancy_point_cloud_hull_extended_{r}m.png for each r).

    Returns:
        Dict with keys
        - ``point_cloud_hull_ground_coverage_percentage_{r}m`` for each r in hull_margins_meters
        - ``point_cloud_hull_area_m2_{r}m`` for each r in hull_margins_meters - area of the expanded hull in square metres
        - ``mesh_area_2d_m2`` - 2D projected mesh area on the XY plane in square metres
        - ``mesh_area_2d_inside_point_cloud_hull_m2`` - approximate 2D mesh area inside the hull in square metres
        - ``mesh_area_2d_outside_point_cloud_hull_percentage`` - approximate percentage of 2D mesh area outside the hull
          (not exactly 0% when the hull fully contains the mesh due to rasterization error).
    """
    if hull_margins_meters is None:
        hull_margins_meters = [0, 50]

    hull_margins_meters = sorted(set(hull_margins_meters))  # deduplicate and sort for deterministic metric key order

    metrics: Dict[str, float] = {}

    if len(ego_trajectory) == 0:
        log.warning("Empty ego trajectory, skipping point cloud hull coverage metrics")
        return metrics

    if len(input_points) < 3:
        log.warning("Too few input points for hull computation, skipping point cloud hull coverage metrics")
        return metrics

    # Trims the point cloud in order to keep the scanned domain limited for rasterization.
    # Trims points that are too far away from the ego trajectory - effectively an artificial scan range.
    # A bounding box approach is much faster than kNNs between points and ego trajectory (tested on ~100M points),
    # and much better than trimming a far percentile of points from the centroid, which would trim an arbitrary-size
    # outer chunk off a scene based on the arbitrary point density in the middle of the scene (tested).
    log.info(f"Point cloud trimming using a range limit of {max_range_meters:.2f} meters")
    start_timestamp = time.perf_counter()
    traj_xy_min = ego_trajectory[:, :2].min(axis=0) - max_range_meters
    traj_xy_max = ego_trajectory[:, :2].max(axis=0) + max_range_meters
    pts_xy = input_points[:, :2]
    points_mask = (pts_xy >= traj_xy_min).all(axis=1) & (pts_xy <= traj_xy_max).all(axis=1)
    points_filtered_xy = input_points[points_mask, :2]
    n_points_removed = int((~points_mask).sum())
    log.info(
        f"Point cloud trimmed in {time.perf_counter() - start_timestamp:.1f} seconds"
        f": {n_points_removed} of {len(input_points)} points "
        f"({100.0 * n_points_removed / len(input_points):.2f}%) removed"
    )

    if len(points_filtered_xy) < 3:
        log.warning("Too few filtered input points for hull computation, skipping point cloud hull coverage metrics")
        return metrics

    log.info(f"Computing convex hull over {len(points_filtered_xy)} input points")
    start_timestamp = time.perf_counter()
    try:
        hull = ConvexHull(points_filtered_xy)
    except QhullError as e:
        log.warning(f"ConvexHull failed: {e}, skipping point cloud hull coverage metrics")
        return metrics
    log.info(f"Convex hull computed in {time.perf_counter() - start_timestamp:.2f} seconds")

    hull_verts = points_filtered_xy[hull.vertices]
    max_margin = max(hull_margins_meters) if hull_margins_meters else 0.0

    # Build 2D grid covering hull bounding box + max margin.
    log.info("Building 2D rasterization grid")
    start_timestamp = time.perf_counter()
    xy_min = hull_verts.min(axis=0) - max_margin - grid_resolution_meters
    xy_max = hull_verts.max(axis=0) + max_margin + grid_resolution_meters
    n_xs = int(np.ceil((xy_max[0] - xy_min[0]) / grid_resolution_meters)) + 1
    n_ys = int(np.ceil((xy_max[1] - xy_min[1]) / grid_resolution_meters)) + 1
    # The world frame is right-handed and has Z pointing up to the sky (against gravity).
    # In Bird's-Eye View, the viewing direction is -Z, so if world X points to the right then world Y points up in BEV.
    # The image frame has x to the right (same as world X) but y down (opposite to world Y),
    # so xs samples go from Xmin to Xmax but ys samples go in reverse from Ymax to Ymin.
    xs = (xy_min[0] + np.arange(n_xs, dtype=np.float32) * grid_resolution_meters).astype(np.float32)
    ys = (xy_max[1] - np.arange(n_ys, dtype=np.float32) * grid_resolution_meters).astype(np.float32)
    gx, gy = np.meshgrid(xs, ys)
    grid_shape = gx.shape
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    log.info(f"Rasterization grid of {n_xs}×{n_ys} cells built in {time.perf_counter() - start_timestamp:.2f} seconds")

    log.info(f"Rasterizing mesh of {len(mesh_vertices)} vertices")
    start_timestamp = time.perf_counter()
    mesh_mask = _rasterize_mesh(mesh_vertices, mesh_faces, grid_pts).reshape(grid_shape)
    log.info(f"Mesh rasterized in {time.perf_counter() - start_timestamp:.2f} seconds")

    log.info(f"Rasterizing point cloud hull of {len(hull_verts)} vertices")
    start_timestamp = time.perf_counter()
    hull_delaunay = Delaunay(hull_verts)
    # Occupancy mask of the convex hull around the input points.
    hull_mask = (hull_delaunay.find_simplex(grid_pts) >= 0).reshape(grid_shape)
    log.info(f"point cloud hull rasterized in {time.perf_counter() - start_timestamp:.2f} seconds")

    log.info(f"Rasterizing {len(input_points)} input points")
    start_timestamp = time.perf_counter()
    points_grid_mask = _rasterize_points(input_points[:, :2], grid_pts, grid_resolution_meters).reshape(grid_shape)
    log.info(f"input points rasterized in {time.perf_counter() - start_timestamp:.2f} seconds")

    if output_dir is not None:
        Image.fromarray((mesh_mask * 255).astype(np.uint8), mode="L").save(
            os.path.join(output_dir, "occupancy_mesh.png")
        )
        Image.fromarray((hull_mask * 255).astype(np.uint8), mode="L").save(
            os.path.join(output_dir, "occupancy_point_cloud_hull.png")
        )
        Image.fromarray((points_grid_mask * 255).astype(np.uint8), mode="L").save(
            os.path.join(output_dir, "occupancy_input_points.png")
        )
        blended = blend_colorized_masks([points_grid_mask, hull_mask], [(0, 0, 0), (100, 255, 100)], [0.5, 0.5])
        Image.fromarray(blended, mode="RGB").save(os.path.join(output_dir, "occupancy_input_points_vs_hull.png"))
        blended = blend_colorized_masks([points_grid_mask, mesh_mask], [(0, 0, 0), (100, 100, 255)], [0.5, 0.5])
        Image.fromarray(blended, mode="RGB").save(os.path.join(output_dir, "occupancy_input_points_vs_mesh.png"))

    log.info(f"Calculating {len(hull_margins_meters)} point_cloud_hull_ground_coverage_percentage metrics")
    start_timestamp = time.perf_counter()
    for radius_meters in hull_margins_meters:
        if radius_meters <= 0.0:
            hull_extended_mask = hull_mask
        else:
            radius_pixels = int(np.ceil(radius_meters / grid_resolution_meters))
            yi, xi = np.ogrid[-radius_pixels : radius_pixels + 1, -radius_pixels : radius_pixels + 1]
            disk = (xi**2 + yi**2) <= radius_pixels**2
            hull_extended_mask = binary_dilation(hull_mask, structure=disk)
        total = int(hull_extended_mask.sum())
        covered = int((hull_extended_mask & mesh_mask).sum())
        metrics[f"point_cloud_hull_ground_coverage_percentage_{radius_meters}m"] = 100.0 * covered / max(total, 1)
        metrics[f"point_cloud_hull_area_m2_{radius_meters}m"] = total * grid_resolution_meters**2
        if output_dir is not None:
            Image.fromarray((hull_extended_mask * 255).astype(np.uint8), mode="L").save(
                os.path.join(output_dir, f"occupancy_point_cloud_hull_extended_{radius_meters}m.png")
            )
            blended = blend_colorized_masks(
                [points_grid_mask, hull_extended_mask, mesh_mask],
                [(0, 0, 0), (100, 255, 100), (100, 100, 255)],
                [0.5, 0.5, 0.5],
            )
            Image.fromarray(blended, mode="RGB").save(
                os.path.join(output_dir, f"occupancy_mesh_vs_point_cloud_hull_extended_{radius_meters}m.png")
            )

    log.info(
        f"{len(hull_margins_meters)} point_cloud_hull_ground_coverage_percentage metrics calculated"
        f" in {time.perf_counter() - start_timestamp:.2f} seconds"
    )

    log.info("Calculating mesh_area_2d_outside_point_cloud_hull_percentage")
    start_timestamp = time.perf_counter()
    # Flatten the mesh to the XY plane (z=0) so its area is comparable to the rasterized 2D hull intersection area.
    flat_vertices = mesh_vertices.copy()
    flat_vertices[:, 2] = 0.0
    mesh_area_2d = float(trimesh.Trimesh(flat_vertices, mesh_faces).area)
    # Approximate the area of the mesh inside the hull by rasterizing the mesh and counting the cells inside the hull.
    mesh_area_2d_in_hull = float((mesh_mask & hull_mask).sum()) * grid_resolution_meters**2
    # The inside hull area and therefore the outside-hull percentage is approximative due to the rasterization error,
    # The error should be negligible unless the flattened mesh collapsed and has comparable 2d area to the cell size.
    metrics["mesh_area_2d_m2"] = mesh_area_2d
    metrics["mesh_area_2d_inside_point_cloud_hull_m2"] = mesh_area_2d_in_hull
    metrics["mesh_area_2d_outside_point_cloud_hull_percentage"] = (
        100.0 * max(0.0, mesh_area_2d - mesh_area_2d_in_hull) / mesh_area_2d if mesh_area_2d > 0.0 else 0.0
    )
    log.info(
        f"mesh_area_2d_outside_point_cloud_hull_percentage calculated in {time.perf_counter() - start_timestamp:.2f} seconds"
    )

    return metrics


def calculate_ground_mesh_metrics(
    mesh_vertices: NDArray[np.float32],
    mesh_faces: NDArray[np.int32],
    T_rig_worlds: NDArray[np.float32],
    cuboid_tracks: Optional[CuboidTracks] = None,
    input_points: Optional[NDArray[np.float32]] = None,
    hull_margins_meters: Optional[List[int]] = None,
    max_range_meters: float = 200.0,
    grid_resolution_meters: float = 0.5,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Calculate quality metrics for the ground mesh and optionally export diagnostic data.

    Args:
        mesh_vertices: (M, 3) array of mesh vertices.
        mesh_faces: (F, 3) array of mesh faces.
        T_rig_worlds: (N, 4, 4) array of poses, as 4x4 transformations from the rig frame to the world frame.
        cuboid_tracks: Optional CuboidTracks for computing cuboid ground coverage KPIs.
        input_points: Optional (N, 3) array of 3D input points in world coordinates for hull coverage KPIs.
        hull_margins_meters: Hull expansion radii forwarded to calculate_point_cloud_hull_coverage_metrics.
        max_range_meters: Point cloud range limit forwarded to calculate_point_cloud_hull_coverage_metrics.
        grid_resolution_meters: Raster cell size forwarded to calculate_point_cloud_hull_coverage_metrics.
        output_dir: Optional directory path for exporting diagnostic data.

    Returns:
        A dict mapping metric name to float value. Always contains mesh geometry
        and trajectory ground-consistency metrics. When *cuboid_tracks* is provided,
        also contains ``cuboid_*`` and ``dynamic_cuboid_*`` keys. When *input_points*
        is provided and the rig trajectory is non-empty, also contains
        ``point_cloud_hull_ground_coverage_percentage_*`` and
        ``mesh_area_2d_outside_point_cloud_hull_percentage`` keys.
    """

    metrics = {
        "mesh_area_m2": 0.0,
        "mesh_vertex_count": len(mesh_vertices),
        "mesh_face_count": len(mesh_faces),
        "mesh_boundary_count": 0,
    }

    if len(mesh_vertices) < 3 or len(mesh_faces) == 0:
        log.warning("Mesh does not have a single valid face, skipping metrics")
        return metrics

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if output_dir is not None:
        # Export loaded mesh to a PLY file for verification.
        save_ply(os.path.join(output_dir, "mesh.ply"), vertices=mesh_vertices, triangles=mesh_faces, logger=log)

    # Add mesh area to the metrics.
    trimesh_mesh = trimesh.Trimesh(mesh_vertices, mesh_faces)
    metrics["mesh_area_m2"] = float(trimesh_mesh.area)
    log.info(f"Mesh area: {metrics['mesh_area_m2']:.2f} m2")

    log.info("Tracing mesh boundaries incl. hole boundaries")
    # Guard against bugs in find_boundaries crashing the whole evaluation; remove once mesh_processing_test
    # covers boundary tracing on degenerate inputs without raising.
    try:
        start_time = time.perf_counter()
        mesh_boundaries = find_boundaries(mesh_faces)
        metrics["mesh_boundary_count"] = len(mesh_boundaries)
        log.info(f"Found {len(mesh_boundaries)} boundaries in {time.perf_counter() - start_time:.1f} seconds")
    except Exception as e:
        log.warning(f"Mesh boundary tracing failed, skipping boundary metric: {e}")

    # Add mesh quality metrics based on a vehicle/rig trajectory on the nominal ground (treated as ground-truth).
    traj_ground_metrics = calculate_vehicle_trajectory_vs_ground_consistency_metrics(
        mesh_vertices, mesh_faces, T_rig_worlds, output_dir=output_dir
    )
    metrics.update(traj_ground_metrics)

    # Add cuboid ground coverage metrics.
    if cuboid_tracks is not None:
        cuboid_ground_metrics = calculate_cuboid_coverage_by_ground_mesh(
            cuboid_tracks, mesh_vertices, mesh_faces, output_dir=output_dir
        )
        metrics.update(cuboid_ground_metrics)
    else:
        log.warning("No cuboid tracks provided, skipping cuboid coverage metrics")

    # Add point-cloud-hull coverage KPIs.
    if input_points is not None and len(input_points) > 0:
        if len(T_rig_worlds) == 0:
            log.warning("Rig trajectory is empty, skipping point cloud hull coverage metrics")
        else:
            ego_trajectory = T_rig_worlds[:, :3, 3]
            hull_metrics = calculate_point_cloud_hull_coverage_metrics(
                mesh_vertices,
                mesh_faces,
                input_points,
                ego_trajectory,
                hull_margins_meters=hull_margins_meters,
                max_range_meters=max_range_meters,
                grid_resolution_meters=grid_resolution_meters,
                output_dir=output_dir,
            )
            metrics.update(hull_metrics)
    else:
        log.warning("No input points provided, skipping point cloud hull coverage metrics")

    return metrics


def log_metrics(metrics: Dict[str, float], title: Optional[str] = None) -> None:
    if title is not None:
        log.info(f"{title}:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            log.info(f"  {metric_name}: {metric_value:.2f}")
        else:
            log.info(f"  {metric_name}: {metric_value}")


def save_metrics_yaml(metrics: Dict[str, float], file_path: str) -> None:
    if not file_path.endswith(".yaml"):
        file_path += ".yaml"
    with open(file_path, "w") as f:
        yaml.dump(metrics, f)
    log.info(f"Saved metrics to {file_path}")


def save_metrics_json(metrics: Dict[str, float], file_path: str) -> None:
    if not file_path.endswith(".json"):
        file_path += ".json"
    with open(file_path, "w") as f:
        json.dump(metrics, f)
    log.info(f"Saved metrics to {file_path}")


def save_metrics_csv(metrics: Dict[str, float], file_path: str) -> None:
    if not file_path.endswith(".csv"):
        file_path += ".csv"
    with open(file_path, "w") as f:
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                f.write(f"{metric_name},{metric_value:.6f}\n")
            else:
                f.write(f"{metric_name},{metric_value}\n")
    log.info(f"Saved metrics to {file_path}")


def save_metrics_transposed_csv(metrics: Dict[str, float], file_path: str) -> None:
    if not file_path.endswith(".csv"):
        file_path += ".csv"
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    with open(file_path, "w") as f:
        f.write(",".join(metric_names) + "\n")
        f.write(",".join(str(round(v, 6)) if isinstance(v, float) else str(v) for v in metric_values))
        f.write("\n")
    log.info(f"Saved metrics to {file_path}")


@click.command("eval-ground-mesh")
@click.option(
    "--output-dir",
    type=str,
    help="Path to the output directory for metrics files and diagnostic exports",
    required=True,
    default=None,
)
@click.option(
    "--usdz-path",
    type=str,
    help=(
        "Path to an USDZ artifact file exported from training, e.g. last.usdz. "
        "If not provided, you must provide --ground-mesh-path and --rig-trajectory-path."
    ),
    required=False,
    default=None,
)
@click.option(
    "--ground-mesh-path",
    type=str,
    help="Path to a PLY file to load the ground mesh from, otherwise it will be loaded from the USDZ artifact.",
    required=False,
    default=None,
)
@click.option(
    "--rig-trajectory-path",
    type=str,
    help="Path to a JSON file to load the rig trajectory from, otherwise it will be loaded from the USDZ artifact.",
    required=False,
    default=None,
)
@click.option(
    "--cuboid-tracks-path",
    type=str,
    help=(
        "Path to a sequence_tracks JSON file to load object cuboid tracks from. "
        "If not provided, tracks will be loaded from the USDZ artifact if present."
    ),
    required=False,
    default=None,
)
@click.option(
    "--point-cloud-path",
    type=str,
    help=(
        "Path to a PLY point cloud file (e.g. from export-point-cloud --world-frame) "
        "used to compute point-cloud-hull coverage KPIs. Omit to skip hull KPIs."
    ),
    required=False,
    default=None,
)
@click.option(
    "--point-cloud-margin",
    type=int,
    multiple=True,
    help=(
        "Hull expansion radii in metres for point-cloud-hull coverage KPIs. "
        "Pass multiple times to compute at several radii (e.g. --point-cloud-margin 0 --point-cloud-margin 50). "
        "Defaults to [0, 50]."
    ),
    default=[0, 50],
    show_default=True,
)
@click.option(
    "--point-cloud-max-range",
    type=float,
    help="Maximum distance from the ego trajectory used to trim the input point cloud before hull computation (metres).",
    default=200.0,
    show_default=True,
)
@click.option(
    "--hull-grid-resolution",
    type=float,
    help="Raster cell size in metres for point-cloud-hull coverage computation.",
    default=0.5,
    show_default=True,
)
def eval_ground_mesh(
    output_dir: str,
    usdz_path: Optional[str],
    rig_trajectory_path: Optional[str],
    ground_mesh_path: Optional[str],
    cuboid_tracks_path: Optional[str],
    point_cloud_path: Optional[str],
    point_cloud_margin: Tuple[int, ...],
    point_cloud_max_range: float,
    hull_grid_resolution: float,
) -> None:
    """Evaluates a ground mesh stored in a USDZ artifact or in a PLY file."""

    artifact = None
    if usdz_path is not None:
        log.info(f"Loading USDZ artifact {usdz_path}")
        artifact = Artifact(Path(usdz_path))

    # Load the ground mesh from PLY if specified, or from the USDZ artifact otherwise.
    if ground_mesh_path is not None:
        log.info(f"Loading ground mesh from {ground_mesh_path}")
        mesh_vertices, mesh_faces = load_ground_mesh_from_ply(ground_mesh_path)
        if usdz_path is not None:
            log.warning("USDZ specified but ground mesh loaded from the provided PLY")
    elif artifact is not None:
        log.info("Loading ground mesh from the USDZ artifact")
        mesh_vertices, mesh_faces = load_ground_mesh_from_artifact(artifact)
    else:
        raise ValueError("Either --ground-mesh-path or --usdz-path must be provided")
    log.info(f"Loaded mesh with {len(mesh_vertices)} vertices, {len(mesh_faces)} faces")

    # Load the rig trajectories from a JSON if specified, or from the USDZ artifact otherwise.
    if rig_trajectory_path is not None:
        log.info(f"Loading rig trajectory from {rig_trajectory_path}")
        T_rig_worlds = load_rig_trajectory_from_json(rig_trajectory_path)
        if usdz_path is not None:
            log.warning("USDZ specified but rig trajectory loaded from the provided JSON")
    elif artifact is not None:
        log.info("Loading rig trajectory from the USDZ artifact")
        T_rig_worlds = load_rig_trajectory_from_artifact(artifact)
    else:
        raise ValueError("Either --rig-trajectory-path or --usdz-path must be provided")
    log.info(f"Loaded {len(T_rig_worlds)} rig poses")

    # Optionally load cuboid tracks from JSON if specified, or from sequence_tracks.json in the USDZ otherwise.
    cuboid_tracks: Optional[CuboidTracks] = None
    if cuboid_tracks_path is not None:
        log.info(f"Loading cuboid tracks from {cuboid_tracks_path}")
        if usdz_path is not None:
            log.warning("USDZ specified but cuboid tracks loaded from the JSON provided via --cuboid-tracks-path")
        cuboid_tracks = load_cuboid_tracks_from_json(cuboid_tracks_path)
    elif artifact is not None:
        cuboid_tracks = load_cuboid_tracks_from_artifact(artifact)
        if cuboid_tracks is None:
            log.warning("No sequence_tracks.json found in the USDZ artifact, skipping cuboid coverage metrics")

    # Optionally load 3D input points for hull coverage KPIs.
    input_points: Optional[NDArray[np.float32]] = None
    if point_cloud_path is not None:
        log.info(f"Loading point cloud from {point_cloud_path}")
        point_cloud_obj = trimesh.load(point_cloud_path)
        if isinstance(point_cloud_obj, trimesh.Scene):
            geometries = list(point_cloud_obj.geometry.values())
            point_cloud_obj = geometries[0] if geometries else None
        if point_cloud_obj is not None:
            input_points = np.asarray(point_cloud_obj.vertices, dtype=np.float32)
            log.info(f"Loaded {len(input_points)} input points")
        else:
            log.warning(
                f"Could not extract point cloud from {point_cloud_path}, skipping point cloud hull coverage metrics"
            )
    else:
        log.warning("No --point-cloud-path provided, skipping point cloud hull coverage metrics")

    log.info("Calculating ground mesh metrics")
    metrics = calculate_ground_mesh_metrics(
        mesh_vertices,
        mesh_faces,
        T_rig_worlds,
        cuboid_tracks=cuboid_tracks,
        input_points=input_points,
        hull_margins_meters=list(point_cloud_margin),
        max_range_meters=point_cloud_max_range,
        grid_resolution_meters=hull_grid_resolution,
        output_dir=os.path.join(output_dir, "diagnostics") if output_dir is not None else None,
    )

    log_metrics(metrics, title="Ground mesh metrics")

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "ground_mesh_quality_metrics")
    save_metrics_yaml(metrics, output_file_path)
    save_metrics_json(metrics, output_file_path)
    save_metrics_csv(metrics, output_file_path)
    save_metrics_transposed_csv(metrics, output_file_path + "_transposed.csv")
