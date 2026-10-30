# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build measured cone initialization points from native auxiliary masks and raw LiDAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ConeInitPoints:
    """Measured cone positions in world-space metres and row-aligned uint8 RGB."""

    xyz_world: np.ndarray
    rgb: np.ndarray


def components(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Find spatially connected point groups using a fixed metric radius."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
    _, labels = connected_components(graph, directed=False)
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]


def ground_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Deterministic local RANSAC; z is the NCore world up direction."""
    min_inliers = 20
    ransac_iterations = 128
    min_up_alignment = 0.97
    inlier_distance_m = 0.035
    min_inlier_fraction = 0.30
    max_slope = 0.25

    if len(points) < min_inliers:
        return None
    origin = np.median(points, axis=0)
    p = points - origin
    rng = np.random.default_rng(0)
    best = np.zeros(len(p), dtype=bool)
    for _ in range(ransac_iterations):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-8 or abs(normal[2]) / norm < min_up_alignment:
            continue
        normal /= norm
        inliers = np.abs((p - a) @ normal) < inlier_distance_m
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < min_inliers or best.mean() < min_inlier_fraction:
        return None
    design = np.column_stack((p[best, :2], np.ones(best.sum())))
    coeff = np.linalg.lstsq(design, p[best, 2], rcond=None)[0]
    if np.linalg.norm(coeff[:2]) > max_slope:
        return None
    return origin, coeff


def select_observation(
    world: np.ndarray,
    xy: np.ndarray,
    ranges: np.ndarray,
    mask: np.ndarray,
    tree: cKDTree,
) -> np.ndarray | None:
    """Select the nearest plausible surface within one instance mask."""
    min_points = 4
    cluster_radius_m = 0.18
    ground_radius_m = 1.25
    ground_search_below_m = 1.5
    ground_search_above_m = 0.1
    min_height_above_ground_m = 0.06
    max_height_above_ground_m = 1.3
    max_width_m = 0.8
    min_height_m = 0.12
    max_height_m = 1.3

    pixels = np.floor(xy).astype(int)
    hit = mask[pixels[:, 1], pixels[:, 0]]
    candidate = np.flatnonzero(hit)
    if len(candidate) < min_points:
        return None
    plausible = []
    for group in components(world[candidate], cluster_radius_m):
        indices = candidate[group]
        if len(indices) < min_points:
            continue
        center = np.median(world[indices], axis=0)
        # Restrict ground fitting to the immediate spatial neighborhood, not the distant road.
        near = np.asarray(tree.query_ball_point(center[:2], ground_radius_m), dtype=int)
        local = world[near]
        local = local[
            (local[:, 2] > center[2] - ground_search_below_m) & (local[:, 2] < center[2] + ground_search_above_m)
        ]
        plane = ground_plane(local)
        if plane is not None:
            origin, coeff = plane
            p = world[indices] - origin
            heights = p[:, 2] - (p[:, :2] @ coeff[:2] + coeff[2])
            indices = indices[(heights >= min_height_above_ground_m) & (heights <= max_height_above_ground_m)]
        if len(indices) < min_points:
            continue
        extent = np.ptp(world[indices], axis=0)
        if max(extent[:2]) > max_width_m or extent[2] < min_height_m or extent[2] > max_height_m:
            continue
        plausible.append((float(np.median(ranges[indices])), indices))
    if not plausible:
        return None
    # Choose the first plausible compact surface along the rays; never the densest background.
    return min(plausible, key=lambda item: item[0])[1]


def merge_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require static agreement in separate scans, then deduplicate and cap each object."""
    merge_radius_m = 0.4
    min_scans = 2
    max_width_m = 0.9
    support_radius_m = 0.12
    min_points = 8
    voxel_size_m = 0.02
    max_points = 512

    if not observations:
        return []
    centers = np.array([np.median(o["xyz"], axis=0) for o in observations])
    tracks = []
    for group in components(centers, merge_radius_m):
        obs = [observations[i] for i in group]
        if len({o["scan"] for o in obs}) < min_scans:
            continue
        xyz = np.concatenate([o["xyz"] for o in obs])
        rgb = np.concatenate([o["rgb"] for o in obs])
        scans = np.concatenate([np.full(len(o["xyz"]), o["scan"]) for o in obs])
        distance = np.concatenate([o["ranges"] for o in obs])
        if max(np.ptp(xyz[:, :2], axis=0)) > max_width_m:
            continue  # Reject merged neighboring objects or moving objects.
        tree = cKDTree(xyz)
        keep = np.array(
            [
                np.any(scans[neighbors] != scans[i])
                for i, neighbors in enumerate(tree.query_ball_point(xyz, support_radius_m))
            ]
        )
        if keep.sum() < min_points:
            continue
        # Prefer the closest observation for each voxel (including its RGB).
        order = np.flatnonzero(keep)[np.argsort(distance[keep], kind="stable")]
        voxels = np.floor(xyz[order] / voxel_size_m).astype(np.int64)
        _, first = np.unique(voxels, axis=0, return_index=True)
        selected = order[first]  # np.unique sorts the voxel coordinates deterministically.
        if len(selected) > max_points:
            selected = selected[np.linspace(0, len(selected) - 1, max_points, dtype=int)]
        if len(selected) < min_points:
            continue
        tracks.append({"xyz": xyz[selected], "rgb": rgb[selected]})
    return tracks


def select_frames(timestamps: np.ndarray, requested: list[int]) -> list[int]:
    """Look up exact exposure-end timestamps in chronological order."""
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Requested timestamps must be nonempty and unique")
    indices = []
    for timestamp in sorted(requested):
        matches = np.flatnonzero(timestamps == timestamp)
        if len(matches) != 1:
            raise ValueError(f"Expected one camera exposure ending at {timestamp}, found {len(matches)}")
        indices.append(int(matches[0]))
    return indices


def eligible_masks(masks: np.ndarray, valid_pixels: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Filter instance masks by image area, ego validity and pixel height."""
    min_area_px = 16
    max_area_fraction = 0.25
    min_valid_fraction = 0.95
    min_height_px = 12

    if masks.ndim != 3 or masks.shape[1:] != valid_pixels.shape or masks.dtype != np.bool_:
        raise ValueError("Expected boolean instance masks matching the image resolution")
    result = []
    for index, mask in enumerate(masks):
        area = int(np.count_nonzero(mask))
        if not min_area_px <= area <= max_area_fraction * mask.size:
            continue
        clipped = mask & valid_pixels
        if np.count_nonzero(clipped) / area < min_valid_fraction:
            continue
        y = np.nonzero(clipped)[0]
        if int(y.max() - y.min() + 1) >= min_height_px:
            result.append((index, clipped))
    return result


def project_scan(camera: Any, lidar: Any, frame_index: int, size: tuple[int, int], device: Any) -> tuple:
    """Project raw motion-compensated returns using distortion and both shutter poses."""
    import torch

    from ncore.impl.common.transformations import transform_point_cloud
    from ncore.sensors import CameraModel

    min_range_m = 1

    timestamp = int(camera.frames_timestamps_us[frame_index, 1])
    scan_index = int(lidar.get_closest_frame_index(timestamp))
    scan = lidar.get_frame_point_cloud(scan_index, motion_compensation=True, with_start_points=True, return_index=0)
    ranges = np.linalg.norm(scan.xyz_m_end - scan.xyz_m_start, axis=-1)
    valid = np.isfinite(scan.xyz_m_end).all(axis=-1) & np.isfinite(ranges) & (ranges >= min_range_m)
    world = transform_point_cloud(scan.xyz_m_end[valid], lidar.get_frames_T_sensor_target("world", scan_index))
    if not len(world):
        return np.empty((0, 3), dtype=world.dtype), np.empty((0, 2)), np.empty(0), scan_index
    poses = torch.as_tensor(
        camera.get_frames_T_source_sensor("world", frame_index, frame_timepoint=None),
        device=device,
        dtype=torch.float32,
    )
    model = CameraModel.from_parameters(camera.model_parameters, device=device, dtype=torch.float32)
    with torch.inference_mode():
        projection = model.world_points_to_image_points_shutter_pose(
            torch.as_tensor(world, device=device, dtype=torch.float32),
            poses[0],
            poses[1],
            return_valid_indices=True,
            return_all_projections=False,
        )
    xy = projection.image_points.cpu().numpy() * (np.asarray(size) / np.asarray(camera.model_parameters.resolution))
    assert projection.valid_indices is not None
    ids = projection.valid_indices.cpu().numpy().reshape(-1)
    world, ranges = world[ids], ranges[valid][ids]
    valid = np.isfinite(xy).all(axis=-1) & (xy >= 0).all(axis=-1) & (xy < size).all(axis=-1)
    return world[valid], xy[valid], ranges[valid], scan_index


def frame_observations(
    world: np.ndarray,
    xy: np.ndarray,
    ranges: np.ndarray,
    image: np.ndarray,
    masks: np.ndarray,
    scan_index: int,
) -> list[dict[str, Any]]:
    """Select and color each instance independently; road semantic labels are unused."""
    tree = cKDTree(world[:, :2])
    pixels = np.floor(xy).astype(int)
    observations = []
    for mask in masks:
        selected = select_observation(world, xy, ranges, mask, tree)
        if selected is None:
            continue
        observations.append(
            {
                "xyz": world[selected],
                "rgb": image[pixels[selected, 1], pixels[selected, 0]],
                "ranges": ranges[selected],
                "scan": scan_index,
            }
        )
    return observations


def combine_tracks(tracks: list[dict[str, Any]]) -> ConeInitPoints:
    """Preserve the accepted object tracks and their measured rows in merge order."""
    if not tracks:
        return ConeInitPoints(np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8))
    xyz = np.concatenate([track["xyz"] for track in tracks])
    rgb = np.concatenate([track["rgb"] for track in tracks])
    return ConeInitPoints(xyz, rgb)


def build_lidar_cone_init_points(source: Any, camera_id: str, lidar_id: str, device: Any) -> ConeInitPoints:
    """Build measured cone initialization points from native auxiliary instances and active raw LiDAR.

    No semantic road labels or dynamic-track filters are applied to raw returns.
    Auxiliary masks may cover a larger clip than the active training interval.
    """
    if camera_id not in source.train_camera_ids:
        raise ValueError("Object-mask camera must be a training camera")
    if lidar_id not in source.train_lidar_ids:
        raise ValueError("Object-initialization LiDAR must be a training sensor")
    aux = source.aux_loader
    if aux is None or not aux.has_object_instance_segmentation(camera_id):
        raise ValueError(
            "Missing object instances: enable dataset.aux_data and run ncore-aux-data with --sam3-objects-checkpoint"
        )
    meta = aux.get_object_instance_segmentation_meta(camera_id)
    if "resolution" not in meta:
        raise ValueError("Incomplete object auxiliary data: missing camera metadata")
    if meta.get("thing_classes") != ["cone"]:
        raise ValueError("LiDAR object initialization currently supports cone-only auxiliary instances")
    timestamps = aux.get_object_instance_segmentation_timestamps(camera_id)
    if not timestamps:
        raise ValueError("No object-mask frames found")
    camera = source.camera_sensors[camera_id]
    lidar = source.lidar_sensors[lidar_id]
    indices = select_frames(camera.frames_timestamps_us[:, 1], timestamps)
    observations: list[dict[str, Any]] = []
    for index in indices:
        timestamp = int(camera.frames_timestamps_us[index, 1])
        if index not in source.camera_frame_ranges[camera_id]:
            continue
        if int(lidar.get_closest_frame_index(timestamp)) not in source.lidar_frame_ranges[lidar_id]:
            continue
        image = np.asarray(camera.get_frame_image_array(index))
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected uint8 RGB source image")
        if list(image.shape[:2][::-1]) != meta["resolution"]:
            raise ValueError(f"Object-mask image mismatch at {timestamp}")
        data = aux.get_object_instance_segmentation(camera_id, timestamp)
        masks, classes, scores = data["instance_masks"], data["classes"], data["scores"]
        if (
            masks.shape != (len(scores), *image.shape[:2])
            or masks.dtype != np.uint8
            or np.any(masks > 1)
            or classes.shape != (len(masks),)
            or classes.dtype.kind not in "iu"
            or np.any(classes != 0)
            or scores.shape != (len(masks),)
            or not np.isfinite(scores).all()
        ):
            raise ValueError(f"Malformed object instances at {timestamp}")
        world, xy, ranges, scan = project_scan(camera, lidar, index, tuple(meta["resolution"]), device)
        observations.extend(frame_observations(world, xy, ranges, image, masks.astype(bool), scan))
    return combine_tracks(merge_observations(observations))
