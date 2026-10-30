# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict SAM3-mask-backed NCore point clouds for vehicle-light initialization."""

from __future__ import annotations

import hashlib
import json
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import cv2
import numpy as np
import torch

import ncore.sensors

from nurec.datasets.tracks import CuboidTracks
from nurec.libs.vren.interface import image_points_to_world_rays_shutter_pose
from nurec.utils.geometry import se3_matrix_to_tquat
from nurec.utils.misc import to_torch
from nurec.utils.types import PointCloud


if TYPE_CHECKING:
    from nurec.datasets.ncore import NCOREDataSource


logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class MaskedTrackPointCloud:
    track_id: str
    point_cloud: PointCloud
    track_pose_index: int


def _count_tracks_overlapping_interval(
    track_start_timestamps_us: torch.Tensor,
    track_end_timestamps_us: torch.Tensor,
    interval_start_us: int,
    interval_end_us: int,
) -> int:
    """Count tracks that could be active during an inclusive timestamp interval.

    Starts/ends are int64 ``[N_tracks]`` microsecond tensors on the same CPU
    or CUDA device. Returns a Python int (zero for empty inputs); CUDA inputs
    synchronize when reading the count. Raises ValueError for unequal shapes.
    """

    if track_start_timestamps_us.shape != track_end_timestamps_us.shape:
        raise ValueError(
            "track_start_timestamps_us and track_end_timestamps_us must have matching shapes, "
            f"got {tuple(track_start_timestamps_us.shape)} and {tuple(track_end_timestamps_us.shape)}"
        )
    overlaps = torch.logical_and(
        track_start_timestamps_us <= interval_end_us,
        track_end_timestamps_us >= interval_start_us,
    )
    return int(torch.count_nonzero(overlaps).item())


def _flatten_ray_intersections(
    intersections_cnt: torch.Tensor,
    intersections_tracks_idx: torch.Tensor,
    intersections_ts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten dense timed intersections into valid distance/index rows.

    Inputs share a device: int32 counts ``[N]``, int32 track indices ``[N, K]``,
    and floating entry/exit distances ``[N, K, 2]`` from the CUDA binding.
    Counts must not exceed capacity K. Returns distances ``[M, 2]`` and
    ``[ray, track]`` indices ``[M, 2]`` with input dtype/device preserved;
    M is the sum of counts. Raises ValueError for incompatible input shapes.
    """

    if intersections_cnt.ndim != 1:
        raise ValueError(f"intersections_cnt must have shape [N], got {tuple(intersections_cnt.shape)}")
    if intersections_tracks_idx.ndim != 2 or intersections_tracks_idx.shape[0] != intersections_cnt.shape[0]:
        raise ValueError(
            "intersections_tracks_idx must have shape [N, K] matching intersections_cnt, "
            f"got {tuple(intersections_tracks_idx.shape)}"
        )
    expected_ts_shape = (*intersections_tracks_idx.shape, 2)
    if intersections_ts.shape != expected_ts_shape:
        raise ValueError(f"intersections_ts must have shape {expected_ts_shape}, got {tuple(intersections_ts.shape)}")

    slot_idxs = torch.arange(intersections_tracks_idx.shape[1], device=intersections_cnt.device)
    valid_slots = slot_idxs.unsqueeze(0) < intersections_cnt.unsqueeze(1)
    ray_idxs, slot_idxs = valid_slots.nonzero(as_tuple=True)
    track_idxs = intersections_tracks_idx[ray_idxs, slot_idxs]
    intersection_idxs = torch.stack((ray_idxs, track_idxs.long()), dim=1).to(intersections_tracks_idx.dtype)
    return intersections_ts[ray_idxs, slot_idxs], intersection_idxs


def _select_frontmost_intersection_rows(
    intersection_ts: torch.Tensor,
    intersection_idxs: torch.Tensor,
) -> torch.Tensor:
    """Return one frontmost cuboid-intersection row for every intersecting ray.

    Inputs are floating entry/exit distances and integer ``[ray, track]``
    indices, both ``[M, 2]`` on the same device with one row per ray/track pair.
    Entry distances are clamped to zero; ties choose the lower track index.
    Returns int64 ``[N_hit_rays]`` row indices on the input device, including
    an empty tensor for no hits. Raises ValueError for incompatible shapes.
    """

    if intersection_ts.ndim != 2 or intersection_ts.shape[1] != 2:
        raise ValueError(f"intersection_ts must have shape [N, 2], got {tuple(intersection_ts.shape)}")
    if intersection_idxs.ndim != 2 or intersection_idxs.shape != intersection_ts.shape:
        raise ValueError(
            "intersection_idxs must have the same [N, 2] shape as intersection_ts, "
            f"got {tuple(intersection_idxs.shape)}"
        )
    if intersection_ts.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=intersection_ts.device)

    ray_idxs = intersection_idxs[:, 0].long()
    track_idxs = intersection_idxs[:, 1].long()
    entry_ts = torch.clamp_min(intersection_ts[:, 0], 0)
    n_hit_rays = int(ray_idxs.max().item()) + 1

    frontmost_ts = torch.full(
        (n_hit_rays,),
        float("inf"),
        dtype=entry_ts.dtype,
        device=entry_ts.device,
    )
    frontmost_ts.scatter_reduce_(0, ray_idxs, entry_ts, reduce="amin", include_self=True)
    is_frontmost = entry_ts == frontmost_ts[ray_idxs]

    # A boundary ray can enter two touching cuboids at the same distance. Keep
    # the lower track index so every ray still has exactly one deterministic owner.
    sentinel = torch.iinfo(track_idxs.dtype).max
    frontmost_track = torch.full((n_hit_rays,), sentinel, dtype=track_idxs.dtype, device=track_idxs.device)
    frontmost_track.scatter_reduce_(
        0,
        ray_idxs,
        torch.where(is_frontmost, track_idxs, sentinel),
        reduce="amin",
        include_self=True,
    )
    selected = torch.logical_and(is_frontmost, track_idxs == frontmost_track[ray_idxs])
    return selected.nonzero(as_tuple=False).flatten()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_identity_sha256(
    container: dict,
    *,
    field_owner: str,
    sampled_frames: list,
) -> str:
    identity_sha256 = container.get("sampled_frame_identity_sha256")
    if (
        not isinstance(identity_sha256, str)
        or len(identity_sha256) != 64
        or any(character not in "0123456789abcdef" for character in identity_sha256)
    ):
        raise ValueError(f"SAM3 mask manifest has invalid {field_owner} sampled_frame_identity_sha256")
    expected = _canonical_sha256(sampled_frames)
    if identity_sha256 != expected:
        raise ValueError(
            f"SAM3 mask manifest {field_owner} sampled_frame_identity_sha256 mismatch: "
            f"expected={expected} actual={identity_sha256}"
        )
    return identity_sha256


def _load_binary_pixel_mask(
    mask_path: Path,
    *,
    frame_width: int,
    frame_height: int,
    camera_id: str,
) -> np.ndarray:
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_image is None:
        raise FileNotFoundError(f"Required pixel mask is missing: {mask_path}")
    if mask_image.ndim != 2:
        raise ValueError(f"Mask {mask_path} must be single-channel, got shape {mask_image.shape}")
    if mask_image.dtype != np.uint8:
        raise ValueError(f"Mask {mask_path} must use uint8 storage, got {mask_image.dtype}")
    unique_values = np.unique(mask_image)
    if not np.isin(unique_values, np.array([0, 255], dtype=np.uint8)).all():
        raise ValueError(
            f"Mask {mask_path} must contain only binary values {{0, 255}}, got {unique_values[:8].tolist()}"
        )
    if mask_image.shape != (frame_height, frame_width):
        mask_height, mask_width = mask_image.shape
        if mask_width * frame_height != frame_width * mask_height:
            raise ValueError(
                f"Mask {mask_path} has incompatible aspect ratio {mask_image.shape}; "
                f"expected {(frame_height, frame_width)} for {camera_id}"
            )
        mask_image = cv2.resize(mask_image, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
    return mask_image == 255


def _camera_pixel_mask_root(mask_root: Path, camera_id: str, camera_count: int) -> Path:
    camera_root = mask_root / camera_id
    if camera_root.is_dir():
        return camera_root
    if camera_count > 1:
        raise FileNotFoundError(f"Multi-camera pixel masks require a subdirectory for {camera_id}: {camera_root}")
    return mask_root


def load_source_indexed_pixel_mask_paths(
    mask_root: Path,
    manifest_path: Path,
    camera_ids: list[str],
    *,
    source_dataset_path: Path | None = None,
) -> dict[str, dict[int, Path]]:
    """Load and validate the exact source-frame mask set published by SAM3."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"SAM3 mask manifest does not exist: {manifest_path}") from None
    if not isinstance(manifest, dict):
        raise ValueError(f"SAM3 mask manifest root must be an object: {manifest_path}")
    if manifest.get("schema_version") != 3:
        raise ValueError(f"SAM3 mask manifest schema_version must be 3, got {manifest.get('schema_version')!r}")
    source_provenance = manifest.get("source_provenance")
    if not isinstance(source_provenance, dict):
        raise ValueError(f"SAM3 mask manifest is missing source_provenance: {manifest_path}")
    sampled_frames = source_provenance.get("sampled_frames")
    if not isinstance(sampled_frames, list) or not sampled_frames:
        raise ValueError(f"SAM3 mask manifest has no sampled_frames: {manifest_path}")
    identity_count = manifest.get("sampled_frame_identity_count")
    if isinstance(identity_count, bool) or not isinstance(identity_count, int):
        raise ValueError(f"SAM3 mask manifest has invalid sampled_frame_identity_count: {identity_count!r}")
    if identity_count != len(sampled_frames):
        raise ValueError(
            "SAM3 mask manifest sampled-frame count mismatch: "
            f"identity_count={identity_count} sampled_frames={len(sampled_frames)}"
        )

    provenance_count = source_provenance.get("sampled_frame_identity_count")
    if (
        isinstance(provenance_count, bool)
        or not isinstance(provenance_count, int)
        or provenance_count != len(sampled_frames)
    ):
        raise ValueError(
            "SAM3 mask manifest source_provenance sampled-frame count mismatch: "
            f"identity_count={provenance_count!r} sampled_frames={len(sampled_frames)}"
        )
    top_level_sha256 = _validate_identity_sha256(
        manifest,
        field_owner="top-level",
        sampled_frames=sampled_frames,
    )
    provenance_sha256 = _validate_identity_sha256(
        source_provenance,
        field_owner="source_provenance",
        sampled_frames=sampled_frames,
    )
    if top_level_sha256 != provenance_sha256:
        raise ValueError("SAM3 mask manifest top-level/source_provenance sampled_frame_identity_sha256 mismatch")

    provenance_camera_ids = source_provenance.get("camera_ids")
    if (
        not isinstance(provenance_camera_ids, list)
        or any(not isinstance(camera_id, str) for camera_id in provenance_camera_ids)
        or len(set(provenance_camera_ids)) != len(provenance_camera_ids)
    ):
        raise ValueError("SAM3 mask manifest source_provenance.camera_ids must be a unique string list")
    missing_provenance_cameras = sorted(set(camera_ids) - set(provenance_camera_ids))
    if missing_provenance_cameras:
        raise ValueError(
            f"SAM3 mask manifest source_provenance is missing requested cameras: {missing_provenance_cameras}"
        )

    if source_dataset_path is not None:
        dataset_stat = source_dataset_path.stat()
        dataset_provenance = source_provenance.get("dataset")
        if not isinstance(dataset_provenance, dict):
            raise ValueError("SAM3 mask manifest source_provenance.dataset must be an object")
        expected_filename = source_dataset_path.name
        expected_size = dataset_stat.st_size
        expected_sha256 = _file_sha256(source_dataset_path)
        if (
            dataset_provenance.get("filename") != expected_filename
            or dataset_provenance.get("size_bytes") != expected_size
            or dataset_provenance.get("sha256") != expected_sha256
        ):
            raise ValueError(
                "SAM3 mask manifest dataset provenance mismatch: "
                f"expected filename={expected_filename!r} size_bytes={expected_size} "
                f"sha256={expected_sha256}, "
                f"got {dataset_provenance!r}"
            )

    requested = set(camera_ids)
    paths_by_camera: dict[str, dict[int, Path]] = {camera_id: {} for camera_id in camera_ids}
    for item in sampled_frames:
        if not isinstance(item, dict):
            raise ValueError(f"SAM3 sampled frame must be an object: {item!r}")
        camera_id = item.get("camera_id")
        if camera_id not in requested:
            continue
        source_index = item.get("source_frame_index")
        source_file = item.get("source_frame_file")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
            or not isinstance(source_file, str)
            or Path(source_file).name != source_file
        ):
            raise ValueError(f"Invalid SAM3 sampled-frame identity: {item!r}")
        source_stem = Path(source_file).stem
        if not source_stem.isdecimal() or int(source_stem) != source_index:
            raise ValueError(
                "SAM3 source-frame filename/index mismatch: "
                f"camera={camera_id} file={source_file!r} index={source_index}"
            )
        camera_paths = paths_by_camera[str(camera_id)]
        if source_index in camera_paths:
            raise ValueError(f"Duplicate SAM3 sampled frame for {camera_id}: {source_index}")
        camera_root = _camera_pixel_mask_root(mask_root, str(camera_id), len(camera_ids))
        camera_paths[source_index] = camera_root / f"{source_stem}.png"

    for camera_id, indexed_paths in paths_by_camera.items():
        if not indexed_paths:
            raise ValueError(f"SAM3 mask manifest has no sampled frames for requested camera {camera_id}")
        camera_root = _camera_pixel_mask_root(mask_root, camera_id, len(camera_ids))
        expected_names = {path.name for path in indexed_paths.values()}
        actual_names = {path.name for path in camera_root.glob("*.png") if path.is_file()}
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        if missing or extra:
            raise FileNotFoundError(
                f"SAM3 source-index mask set mismatch for {camera_id}: missing={missing[:5]} extra={extra[:5]}"
            )
    return paths_by_camera


def get_masked_camera_track_point_clouds(
    datasource: NCOREDataSource,
    camera_ids: list[str],
    cuboid_tracks: CuboidTracks,
    *,
    pixel_mask_dir: str,
    pixel_mask_manifest_path: str,
    cuboid_dim_scale_factor: float = 1.0,
    return_color: bool = True,
    keep_all_track_poses: bool = False,
    device: torch.device = torch.device("cuda"),
    cuboid_clamp_distance_m: float = 0.15,
) -> Generator[MaskedTrackPointCloud, None, None]:
    """Backproject SAM3 pixels and assign each point to one nearest eligible cuboid.

    The VREN cuboid-ray intersection used by this initializer requires CUDA.
    """

    if device.type != "cuda":
        raise RuntimeError(f"Vehicle-light mask backprojection requires a CUDA device, got {device}")
    if cuboid_clamp_distance_m < 0:
        raise ValueError("cuboid_clamp_distance_m must be non-negative")
    mask_root = Path(pixel_mask_dir)
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Pixel-mask directory does not exist: {mask_root}")
    indexed_masks = load_source_indexed_pixel_mask_paths(
        mask_root,
        Path(pixel_mask_manifest_path),
        camera_ids,
        source_dataset_path=Path(str(datasource.get_path())),
    )

    datasource._maybe_init_worker()
    if keep_all_track_poses:
        logger.warning("keep_all_track_poses has no effect for timestamp-interpolated vehicle-light ray assignment")

    cuboid_tracks = cuboid_tracks.to_device(device)
    cuboid_dims = cuboid_tracks.cuboids_dims.to(device)
    scaled_cuboid_dims = cuboid_dims * cuboid_dim_scale_factor
    # ray_intersection adds padding to the full cuboid dimensions, while
    # cuboid_clamp_distance_m is a per-side tolerance.
    cuboid_dims_padding = scaled_cuboid_dims - cuboid_dims + 2 * cuboid_clamp_distance_m
    track_pose_starts = cuboid_tracks.tracks_packinfo[:, 0].long()
    track_pose_ends = track_pose_starts + cuboid_tracks.tracks_packinfo[:, 1].long() - 1
    track_start_timestamps_us = cuboid_tracks.tracks_timestamps_us[track_pose_starts]
    track_end_timestamps_us = cuboid_tracks.tracks_timestamps_us[track_pose_ends]
    total_mask_pixels = 0
    total_assigned_pixels = 0
    assigned_track_ids: set[str] = set()
    requested_unique_ids = {camera_id: datasource.camera_unique_ids[camera_id].id for camera_id in camera_ids}
    for unique_camera_id in dict.fromkeys(requested_unique_ids.values()):
        for camera_id in camera_ids:
            if requested_unique_ids[camera_id] != unique_camera_id:
                continue
            camera_sensor = datasource.camera_sensors[camera_id]
            camera_params = camera_sensor.model_parameters
            camera_model = ncore.sensors.camera_model_from_parameters(camera_params, device=str(device))
            frame_width, frame_height = camera_params.resolution
            all_pixels = to_torch(datasource.cameras_all_pixels[camera_id], device=device)
            all_camera_footprints = to_torch(
                datasource.cameras_all_footprints[camera_id].reshape(-1),
                device=device,
            )

            camera_masks = indexed_masks[camera_id]
            available_indices = {int(index) for index in datasource.camera_frame_ranges[camera_id]}
            unavailable = sorted(set(camera_masks) - available_indices)
            if unavailable:
                raise ValueError(f"SAM3 sampled frames are unavailable in NCore camera {camera_id}: {unavailable[:5]}")

            for camera_frame_index in sorted(camera_masks):
                camera_frame_timestamp_start, camera_frame_timestamp_end = camera_sensor.frames_timestamps_us[
                    camera_frame_index
                ]
                max_intersections_per_ray = _count_tracks_overlapping_interval(
                    track_start_timestamps_us,
                    track_end_timestamps_us,
                    int(camera_frame_timestamp_start),
                    int(camera_frame_timestamp_end),
                )
                mask_path = camera_masks[camera_frame_index]
                pixel_mask = to_torch(
                    _load_binary_pixel_mask(
                        mask_path,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        camera_id=camera_id,
                    ).reshape(-1),
                    device=device,
                )
                if not pixel_mask.any():
                    continue

                total_mask_pixels += int(pixel_mask.sum().item())
                if max_intersections_per_ray == 0:
                    continue
                pixels = all_pixels[pixel_mask]
                footprints = all_camera_footprints[pixel_mask]
                if return_color:
                    color = to_torch(
                        camera_sensor.get_frame_image_array(camera_frame_index).reshape((-1, 3)),
                        device=device,
                    )[pixel_mask]

                sensor_to_nurec_startend = datasource.world_to_nurec.transform_poses(
                    camera_sensor.get_frames_T_sensor_target(
                        target_node="world",
                        frame_indices=camera_frame_index,
                        frame_timepoint=None,
                    )
                )
                image_points = camera_model.pixels_to_image_points(pixels)
                world_rays = image_points_to_world_rays_shutter_pose(
                    camera_params,
                    image_points,
                    se3_matrix_to_tquat(sensor_to_nurec_startend),
                    torch.LongTensor([camera_frame_timestamp_start, camera_frame_timestamp_end]),
                ).world_rays

                relative_frame_times = camera_model.image_points_relative_frame_times(image_points)
                rays_timestamps_us = int(camera_frame_timestamp_start) + (
                    relative_frame_times * float(int(camera_frame_timestamp_end) - int(camera_frame_timestamp_start))
                ).to(torch.int64)

                intersection_result = cuboid_tracks.ray_intersection(
                    world_rays[:, :3].contiguous(),
                    world_rays[:, 3:].contiguous(),
                    rays_timestamps_us,
                    cuboids_dims_padding=cuboid_dims_padding,
                    max_intersections_per_ray=max_intersections_per_ray,
                    with_intersections_ts=True,
                )
                assert intersection_result.intersections_ts is not None
                intersection_ts, intersection_idxs = _flatten_ray_intersections(
                    intersection_result.intersections_cnt,
                    intersection_result.intersections_tracks_idx,
                    intersection_result.intersections_ts,
                )
                # The flattened outputs own their storage; release dense buffers
                # before selecting hits and before the next frame's allocation.
                del intersection_result
                selected_rows = _select_frontmost_intersection_rows(intersection_ts, intersection_idxs)
                if selected_rows.numel() == 0:
                    continue

                selected_ray_idxs = intersection_idxs[selected_rows, 0].long()
                # The interpolation binding requires indices matching int32 packinfo.
                selected_track_idxs = intersection_idxs[selected_rows, 1]
                selected_entry_ts = torch.clamp_min(intersection_ts[selected_rows, 0], 0)
                selected_world_origins = world_rays[selected_ray_idxs, :3]
                selected_world_entries = selected_world_origins + world_rays[selected_ray_idxs, 3:] * (
                    selected_entry_ts + 0.01
                ).unsqueeze(-1)
                selected_track_poses = cuboid_tracks.interpolate_tracks_poses(
                    rays_timestamps_us[selected_ray_idxs],
                    selected_track_idxs,
                )
                selected_world_to_local = selected_track_poses.inv()
                selected_local_origins = selected_world_to_local * selected_world_origins
                selected_local_entries = selected_world_to_local * selected_world_entries
                total_assigned_pixels += int(selected_rows.numel())

                for track_index_tensor in torch.unique(selected_track_idxs):
                    track_index = int(track_index_tensor.item())
                    point_mask = selected_track_idxs == track_index
                    ray_idxs = selected_ray_idxs[point_mask]
                    local_start = selected_local_origins[point_mask]
                    entry_ts = selected_entry_ts[point_mask]
                    local_end = selected_local_entries[point_mask]

                    half_extent = scaled_cuboid_dims[track_index, :3] / 2
                    inward_epsilon = torch.minimum(torch.full_like(half_extent, 0.01), half_extent * 0.01)
                    inner_half_extent = torch.clamp(half_extent - inward_epsilon, min=1.0e-4)
                    local_end = torch.maximum(torch.minimum(local_end, inner_half_extent), -inner_half_extent)

                    track_start = int(cuboid_tracks.tracks_packinfo[track_index, 0].item())
                    track_count = int(cuboid_tracks.tracks_packinfo[track_index, 1].item())
                    track_timestamps_us = cuboid_tracks.tracks_timestamps_us[track_start : track_start + track_count]
                    representative_timestamp_us = rays_timestamps_us[ray_idxs].median()
                    track_pose_index = int(
                        torch.argmin(torch.abs(track_timestamps_us - representative_timestamp_us)).item()
                    )
                    track_id = cuboid_tracks.tracks_id[track_index]
                    assigned_track_ids.add(track_id)
                    yield MaskedTrackPointCloud(
                        track_id=track_id,
                        point_cloud=PointCloud(
                            xyz_start=local_start,
                            xyz_end=local_end,
                            flags=None,
                            color=color[ray_idxs] if return_color else None,
                            semantic_class_id=None,
                            camera_footprint_scale=footprints[ray_idxs] * entry_ts,
                        ),
                        track_pose_index=track_pose_index,
                    )

    if total_mask_pixels > 0:
        logger.info(
            "Assigned %d/%d SAM3 vehicle-light pixels to %d timestamp-interpolated cuboid tracks",
            total_assigned_pixels,
            total_mask_pixels,
            len(assigned_track_ids),
        )
    if total_mask_pixels > 0 and total_assigned_pixels == 0:
        logger.warning("SAM3 vehicle-light masks contained pixels, but none intersected a cuboid track")
