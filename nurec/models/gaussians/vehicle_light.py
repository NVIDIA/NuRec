# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mask-guided vehicle-light Gaussian initialization and temporal appearance."""

from __future__ import annotations

import math

from collections import OrderedDict, defaultdict
from pathlib import Path
from threading import Lock
from typing import Optional

import torch

from omegaconf import DictConfig
from torch import nn

import nurec.models.gaussians.collect as collect

from nurec.config.model import SHGaussiansLayerConfig, VehicleLightRigidGaussiansLayerConfig
from nurec.config.trainer import TrainerConfig
from nurec.datasets.ncore import NCOREDataSource
from nurec.datasets.summary import DataSourceSummary
from nurec.datasets.tracks import CuboidTracks
from nurec.datasets.vehicle_light import get_masked_camera_track_point_clouds
from nurec.libs.packed_ops.interface import packed_ops
from nurec.models.base import BaseModel
from nurec.models.gaussians.gaussians_model import (
    BaseGaussianModel,
    GaussianExportFormat,
    RigidGaussianModel,
)
from nurec.models.gaussians.initializations import BaseInitialization, NoPointsFoundException
from nurec.models.gaussians.utils import RGB2SH, sh_degree_to_specular_dim, write_ply_3dgs
from nurec.utils.misc import unpack_optional
from nurec.utils.optim import OptimizerLRSchedulerConfig
from nurec.utils.trainer import adjust_learning_rate_for_batch_size


def _resolve_vehicle_light_mask_inputs(config: DictConfig) -> tuple[str, str]:
    """Use explicit mask paths or the resolved aux root; skip if both are absent."""

    mask_dir = config.get("mask_dir")
    manifest_path = config.get("mask_manifest_path")
    if mask_dir is None and manifest_path is None:
        if aux_dir := config.get("sam3_aux_dir"):
            return str(Path(aux_dir) / "masks"), str(Path(aux_dir) / "manifest.json")
        raise NoPointsFoundException("SAM3 mask inputs are not configured; skipping the optional vehicle-light layer")
    if not mask_dir or not manifest_path:
        raise ValueError(
            "vehicle-light initialization requires both mask_dir and mask_manifest_path when either is configured"
        )
    return str(mask_dir), str(manifest_path)


class _TrackFrameSlotCache:
    """Cache CUDA appearance-frame metadata independently of mutable Gaussian rows."""

    def __init__(self, max_entries: int = 4096) -> None:
        self._initialize_empty_state(max_entries)

    def _initialize_empty_state(self, max_entries: int) -> None:
        """Initialize an empty cache with no transient stream or lock state.

        Args:
            max_entries: Maximum number of timestamp rows retained by the LRU.

        Raises:
            ValueError: If ``max_entries`` is not positive.
        """

        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._timestamps: torch.Tensor | None = None
        self._packinfo: torch.Tensor | None = None
        self._versions: tuple[int, int] | None = None
        self._device: torch.device | None = None
        self._track_timestamps: torch.Tensor | None = None
        self._track_packinfo: torch.Tensor | None = None
        self._rows: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._ready_event: torch.cuda.Event | None = None
        self._ready_stream_id: int | None = None
        self._lock = Lock()

    def __getstate__(self) -> dict[str, int]:
        """Return durable configuration without transient CUDA or lock state.

        Returns:
            State needed to recreate an empty cache with the configured capacity.
        """

        return {"max_entries": self._max_entries}

    def __setstate__(self, state: dict[str, int]) -> None:
        """Restore an empty runtime cache after unpickling.

        Args:
            state: Durable state produced by :meth:`__getstate__`.
        """

        self._initialize_empty_state(state["max_entries"])

    def get(
        self,
        timestamp_us: int,
        gaussian_track_ids: torch.Tensor,
        track_timestamps_us: torch.Tensor,
        track_packinfo: torch.Tensor,
    ) -> torch.Tensor:
        """Return cached nearest-frame slots for the requested Gaussian rows.

        Args:
            timestamp_us: Render timestamp in microseconds.
            gaussian_track_ids: CUDA track indices of any shape; int32/int64 avoid a cast.
            track_timestamps_us: CUDA ``[N]`` timestamps, sorted within each track, converted to int64.
            track_packinfo: CUDA ``[T, 2]`` start/count pairs, converted to int64; ranges must fit timestamps.

        Returns:
            Int64 track-relative slots with the same shape and device as ``gaussian_track_ids``.

        Raises:
            RuntimeError: If any input tensor is not on CUDA, including empty inputs.
        """

        if not (gaussian_track_ids.is_cuda and track_timestamps_us.is_cuda and track_packinfo.is_cuda):
            raise RuntimeError("Track-frame slot inputs must be CUDA tensors")
        if gaussian_track_ids.numel() == 0:
            return torch.empty_like(gaussian_track_ids, dtype=torch.long)

        # PyTorch operators release the GIL. Protect the mutable Python LRU and
        # event bookkeeping while concurrent serving threads enqueue their work.
        with self._lock:
            return self._get_locked(timestamp_us, gaussian_track_ids, track_timestamps_us, track_packinfo)

    def _get_locked(
        self,
        timestamp_us: int,
        gaussian_track_ids: torch.Tensor,
        track_timestamps_us: torch.Tensor,
        track_packinfo: torch.Tensor,
    ) -> torch.Tensor:
        """Implement :meth:`get` while the cache mutation lock is held.

        Args:
            timestamp_us: Render timestamp in microseconds.
            gaussian_track_ids: Track index for every requested Gaussian.
            track_timestamps_us: Packed source-frame timestamps for all tracks.
            track_packinfo: Per-track ``[start, count]`` entries into the packed timestamps.

        Returns:
            Track-relative frame slots with the same shape as ``gaussian_track_ids``.
        """

        current_stream = torch.cuda.current_stream(gaussian_track_ids.device)
        current_stream_id = current_stream.cuda_stream

        # Tensor identity catches track replacement; versions catch normal in-place
        # edits. Inference tensors have no version counter, so do not reuse their
        # snapshots. Neither version nor device inspection reads GPU data.
        versions = (
            None
            if track_timestamps_us.is_inference() or track_packinfo.is_inference()
            else (track_timestamps_us._version, track_packinfo._version)
        )
        if (
            versions is None
            or self._timestamps is not track_timestamps_us
            or self._packinfo is not track_packinfo
            or self._versions != versions
            or self._device != gaussian_track_ids.device
        ):
            # Own snapshots on this stream; later invalidation must not recycle
            # metadata still used by an asynchronous lookup on another stream.
            with torch.inference_mode(False):
                self._track_timestamps = track_timestamps_us.detach().to(
                    device=gaussian_track_ids.device, dtype=torch.int64, copy=True
                )
                self._track_packinfo = track_packinfo.detach().to(
                    device=gaussian_track_ids.device, dtype=torch.int64, copy=True
                )
            self._timestamps = track_timestamps_us
            self._packinfo = track_packinfo
            self._versions = versions
            self._device = gaussian_track_ids.device
            self._rows.clear()
            self._ready_event = torch.cuda.Event()
            self._ready_event.record(current_stream)
            self._ready_stream_id = current_stream_id

        crossed_stream = self._ready_event is not None and self._ready_stream_id != current_stream_id
        if crossed_stream:
            assert self._ready_event is not None
            current_stream.wait_event(self._ready_event)

        per_track_slots = self._rows.get(timestamp_us)
        row_was_missing = per_track_slots is None
        if row_was_missing:
            assert self._track_timestamps is not None
            assert self._track_packinfo is not None

            # A row first requested during inference may later be used in training.
            # Keep cached indices ordinary tensors so autograd can save them.
            with torch.inference_mode(False):
                per_track_slots = self._track_packinfo.new_empty(self._track_packinfo.shape[:1])
                packed_ops.nearest_track_frame_slots(
                    timestamp_us, self._track_timestamps, self._track_packinfo, per_track_slots
                )
            self._rows[timestamp_us] = per_track_slots
            if len(self._rows) > self._max_entries:
                self._rows.popitem(last=False)
            assert self._ready_event is not None
            self._ready_event.record(current_stream)
            self._ready_stream_id = current_stream_id
        else:
            self._rows.move_to_end(timestamp_us)

        assert per_track_slots is not None
        if crossed_stream:
            # The cache can evict or invalidate these tensors as soon as this
            # method returns. Tell the allocator about every cross-stream use.
            if row_was_missing:
                assert self._track_timestamps is not None
                assert self._track_packinfo is not None
                self._track_timestamps.record_stream(current_stream)
                self._track_packinfo.record_stream(current_stream)
            else:
                per_track_slots.record_stream(current_stream)

        # index_select accepts the model's int32 IDs directly; advanced indexing
        # converts them to int64 and adds a per-Gaussian allocation/kernel.
        if gaussian_track_ids.dtype in (torch.int32, torch.int64):
            return per_track_slots.index_select(0, gaussian_track_ids.reshape(-1)).reshape(gaussian_track_ids.shape)
        return per_track_slots[gaussian_track_ids.long()]


def nearest_track_frame_slots(
    timestamp_us: int,
    gaussian_track_ids: torch.Tensor,
    track_timestamps_us: torch.Tensor,
    track_packinfo: torch.Tensor,
    *,
    cache: _TrackFrameSlotCache | None = None,
) -> torch.Tensor:
    """Return int64 track-relative frame slots with the shape of ``gaussian_track_ids``.

    All inputs must be CUDA tensors. ``track_packinfo`` contains valid [start, count]
    ranges into ``track_timestamps_us``, sorted within each track; IDs index these
    ranges. Empty/singleton tracks return zero, endpoints clamp, and ties prefer
    the earlier neighbor. Timestamp differences must fit int64.

    Reuse ``cache`` to retain timestamp rows across calls. Metadata changes invalidate
    cached rows; Gaussian IDs may change independently without rebuilding metadata.
    """

    if cache is None:
        cache = _TrackFrameSlotCache()
    return cache.get(timestamp_us, gaussian_track_ids, track_timestamps_us, track_packinfo)


def _knn_indices(positions: torch.Tensor, k: int, chunk_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    n_points = positions.shape[0]
    if n_points <= 1 or k <= 0:
        empty_idx = torch.empty((n_points, 0), dtype=torch.long, device=positions.device)
        empty_dist = torch.empty((n_points, 0), dtype=positions.dtype, device=positions.device)
        return empty_idx, empty_dist

    k = min(k, n_points - 1)
    all_indices = []
    all_distances = []
    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        distances = torch.cdist(positions[start:end], positions)
        local_rows = torch.arange(end - start, device=positions.device)
        global_cols = torch.arange(start, end, device=positions.device)
        distances[local_rows, global_cols] = torch.inf
        nearest_distances, nearest_indices = torch.topk(distances, k=k, dim=1, largest=False)
        all_indices.append(nearest_indices)
        all_distances.append(nearest_distances)
    return torch.cat(all_indices), torch.cat(all_distances)


def propagate_temporal_colors(
    colors: torch.Tensor,
    observed: torch.Tensor,
    positions: torch.Tensor,
    spatial_k: int = 4,
    spatial_iterations: int = 2,
    temporal_chunk_size: int = 64,
) -> torch.Tensor:
    """Fill missing ``[point, frame]`` RGB entries spatially, then temporally."""

    if colors.ndim != 3 or colors.shape[-1] != 3:
        raise ValueError(f"Expected colors [G,N,3], got {tuple(colors.shape)}")
    if observed.shape != colors.shape[:2]:
        raise ValueError(f"Observed shape {tuple(observed.shape)} does not match {tuple(colors.shape[:2])}")
    if positions.shape != (colors.shape[0], 3):
        raise ValueError(f"Positions shape {tuple(positions.shape)} does not match color points")

    values = colors.clone()
    valid = observed.clone()
    neighbor_indices, neighbor_distances = _knn_indices(positions, spatial_k)
    if neighbor_indices.shape[1] > 0:
        base_weights = 1.0 / torch.clamp(neighbor_distances, min=1.0e-4)
        for _ in range(max(0, spatial_iterations)):
            any_change = False
            for frame_start in range(0, colors.shape[1], temporal_chunk_size):
                frame_end = min(frame_start + temporal_chunk_size, colors.shape[1])
                current_valid = valid[:, frame_start:frame_end]
                neighbor_valid = valid[neighbor_indices, frame_start:frame_end]
                weights = base_weights.unsqueeze(-1) * neighbor_valid.to(base_weights.dtype)
                denominator = weights.sum(dim=1)
                can_fill = torch.logical_and(~current_valid, denominator > 0)
                if not can_fill.any():
                    continue
                neighbor_values = values[neighbor_indices, frame_start:frame_end, :]
                propagated = (neighbor_values * weights.unsqueeze(-1)).sum(dim=1)
                propagated /= torch.clamp(denominator.unsqueeze(-1), min=1.0e-8)
                values[:, frame_start:frame_end, :] = torch.where(
                    can_fill.unsqueeze(-1), propagated, values[:, frame_start:frame_end, :]
                )
                valid[:, frame_start:frame_end] = torch.logical_or(current_valid, can_fill)
                any_change = True
            if not any_change:
                break

    n_points, n_frames = valid.shape
    large_distance = n_frames + 1
    forward_values = torch.zeros_like(values)
    forward_distance = torch.full((n_points, n_frames), large_distance, dtype=torch.long, device=values.device)
    last_values = torch.zeros((n_points, 3), dtype=values.dtype, device=values.device)
    last_distance = torch.full((n_points,), large_distance, dtype=torch.long, device=values.device)
    for frame_index in range(n_frames):
        is_valid = valid[:, frame_index]
        last_values = torch.where(is_valid.unsqueeze(-1), values[:, frame_index], last_values)
        last_distance = torch.where(is_valid, torch.zeros_like(last_distance), last_distance + 1)
        forward_values[:, frame_index] = last_values
        forward_distance[:, frame_index] = last_distance

    backward_values = torch.zeros_like(values)
    backward_distance = torch.full_like(forward_distance, large_distance)
    next_values = torch.zeros((n_points, 3), dtype=values.dtype, device=values.device)
    next_distance = torch.full((n_points,), large_distance, dtype=torch.long, device=values.device)
    for frame_index in range(n_frames - 1, -1, -1):
        is_valid = valid[:, frame_index]
        next_values = torch.where(is_valid.unsqueeze(-1), values[:, frame_index], next_values)
        next_distance = torch.where(is_valid, torch.zeros_like(next_distance), next_distance + 1)
        backward_values[:, frame_index] = next_values
        backward_distance[:, frame_index] = next_distance

    use_forward = forward_distance <= backward_distance
    filled = torch.where(use_forward.unsqueeze(-1), forward_values, backward_values)
    unresolved = torch.logical_and(forward_distance > n_frames, backward_distance > n_frames)
    if unresolved.any():
        fallback = torch.where(observed.unsqueeze(-1), colors, torch.zeros_like(colors)).sum(dim=(0, 1))
        fallback /= torch.clamp(observed.sum().to(colors.dtype), min=1.0)
        filled = torch.where(unresolved.unsqueeze(-1), fallback.view(1, 1, 3), filled)
    return torch.where(valid.unsqueeze(-1), values, filled)


def _aggregate_track_observations(
    positions: torch.Tensor,
    colors_uint8: torch.Tensor,
    scales: torch.Tensor,
    frame_slots: torch.Tensor,
    n_frames: int,
    voxel_size_m: float,
    max_points: int,
    spatial_k: int,
    spatial_iterations: int,
    min_scale_voxel_ratio: float,
    scale_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be finite and positive")
    if not math.isfinite(min_scale_voxel_ratio) or min_scale_voxel_ratio <= 0:
        raise ValueError("min_scale_voxel_ratio must be finite and positive")
    if not math.isfinite(scale_multiplier) or scale_multiplier <= 0:
        raise ValueError("scale_multiplier must be finite and positive")
    if max_points <= 0 or n_frames <= 0:
        raise ValueError("max_points and n_frames must be positive")

    current_voxel_size = voxel_size_m
    for iteration in range(16):
        voxel_coords = torch.floor(positions / current_voxel_size).to(torch.int32)
        _, inverse, counts = torch.unique(voxel_coords, dim=0, return_inverse=True, return_counts=True)
        if counts.numel() <= max_points or iteration == 15:
            break
        current_voxel_size *= 1.25

    n_voxels = int(counts.numel())
    if n_voxels > max_points:
        kept_voxels = torch.topk(counts, k=max_points, largest=True).indices
        remap = torch.full((n_voxels,), -1, dtype=torch.long, device=positions.device)
        remap[kept_voxels] = torch.arange(max_points, device=positions.device)
        remapped_inverse = remap[inverse]
        keep = remapped_inverse >= 0
        positions = positions[keep]
        colors_uint8 = colors_uint8[keep]
        scales = scales[keep]
        frame_slots = frame_slots[keep]
        inverse = remapped_inverse[keep]
        n_voxels = max_points
        counts = torch.bincount(inverse, minlength=n_voxels)

    counts_float = counts.to(positions.dtype)
    aggregated_positions = torch.zeros((n_voxels, 3), dtype=positions.dtype, device=positions.device)
    aggregated_positions.index_add_(0, inverse, positions)
    aggregated_positions /= counts_float.unsqueeze(-1)

    aggregated_scales = torch.zeros((n_voxels,), dtype=scales.dtype, device=scales.device)
    aggregated_scales.index_add_(0, inverse, scales)
    aggregated_scales /= counts_float.to(scales.dtype)
    minimum_observation_scale = current_voxel_size * min_scale_voxel_ratio / scale_multiplier
    aggregated_scales = torch.clamp(aggregated_scales, min=minimum_observation_scale)

    flat_indices = inverse.long() * n_frames + frame_slots.long()
    flat_color_sums = torch.zeros((n_voxels * n_frames, 3), dtype=torch.float32, device=positions.device)
    flat_counts = torch.zeros((n_voxels * n_frames,), dtype=torch.float32, device=positions.device)
    flat_color_sums.index_add_(0, flat_indices, colors_uint8.to(torch.float32) / 255.0)
    flat_counts.index_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32))

    observed = flat_counts.reshape(n_voxels, n_frames) > 0
    temporal_rgb = flat_color_sums.reshape(n_voxels, n_frames, 3)
    temporal_rgb /= torch.clamp(flat_counts.reshape(n_voxels, n_frames, 1), min=1.0)
    temporal_rgb = propagate_temporal_colors(
        temporal_rgb,
        observed,
        aggregated_positions,
        spatial_k=spatial_k,
        spatial_iterations=spatial_iterations,
    )
    return aggregated_positions, aggregated_scales, temporal_rgb, observed


class VehicleLightTracksInitialization(BaseInitialization):
    """Initialize vehicle-light Gaussians from source-indexed SAM3 masks."""

    features_albedo_temporal: torch.Tensor

    def __init__(
        self,
        config: DictConfig,
        gaussian_config: SHGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        labels_to_ignore: list[str],
        labels_to_use: list[str],
    ) -> None:
        # Validate before BaseInitialization mutates state so a fully absent mask
        # pair uses the composite's standard optional-layer skip path during both
        # training construction and checkpoint reload.
        _resolve_vehicle_light_mask_inputs(config)
        super().__init__(config, gaussian_config, trainer_config, labels_to_ignore, labels_to_use)

    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        cuboid_tracks = kwargs.get("cuboid_tracks")
        if not isinstance(cuboid_tracks, CuboidTracks):
            raise TypeError("VehicleLightTracksInitialization requires CuboidTracks")
        ncore_datasource = datasource.datasource
        if not isinstance(ncore_datasource, NCOREDataSource):
            raise TypeError("VehicleLightTracksInitialization currently supports NCOREDataSource only")

        mask_dir, manifest_path = _resolve_vehicle_light_mask_inputs(self.config)
        camera_ids = self.config.get("camera_ids")
        if not camera_ids:
            raise ValueError("vehicle-light initialization requires at least one camera_id")

        track_observations: dict[str, list] = defaultdict(list)
        observations = get_masked_camera_track_point_clouds(
            ncore_datasource,
            camera_ids=list(camera_ids),
            cuboid_tracks=cuboid_tracks,
            pixel_mask_dir=str(mask_dir),
            pixel_mask_manifest_path=str(manifest_path),
            cuboid_dim_scale_factor=1.0,
            return_color=True,
            keep_all_track_poses=False,
            device=self.device,
            cuboid_clamp_distance_m=float(self.config.get("cuboid_clamp_distance_m", 0.15)),
        )
        for observation in observations:
            if observation.point_cloud.n_points == 0:
                continue
            if observation.track_pose_index is None:
                raise ValueError("Masked camera observation is missing track_pose_index")
            track_observations[observation.track_id].append(observation)

        active_track_indices = [
            track_index
            for track_index, track_id in enumerate(cuboid_tracks.tracks_id)
            if track_observations.get(track_id)
        ]
        max_track_frames = max(
            (int(cuboid_tracks.tracks_packinfo[track_index, 1].item()) for track_index in active_track_indices),
            default=1,
        )
        all_positions = []
        all_scales = []
        all_temporal_albedo = []
        all_observed = []
        gaussian_cuboid_ids = []

        for track_index, track_id in enumerate(cuboid_tracks.tracks_id):
            per_track = track_observations.get(track_id, [])
            if not per_track:
                continue
            n_track_frames = int(cuboid_tracks.tracks_packinfo[track_index, 1].item())
            positions = torch.cat([item.point_cloud.xyz_end for item in per_track])
            colors = torch.cat([unpack_optional(item.point_cloud.color) for item in per_track])
            scales = torch.cat([unpack_optional(item.point_cloud.camera_footprint_scale) for item in per_track])
            slots = torch.cat(
                [
                    torch.full(
                        (item.point_cloud.n_points,),
                        int(unpack_optional(item.track_pose_index)),
                        dtype=torch.long,
                        device=self.device,
                    )
                    for item in per_track
                ]
            )
            valid_slot = torch.logical_and(slots >= 0, slots < n_track_frames)
            positions, colors, scales, slots = (
                positions[valid_slot],
                colors[valid_slot],
                scales[valid_slot],
                slots[valid_slot],
            )
            if positions.numel() == 0:
                continue

            track_positions, track_scales, track_rgb, track_observed = _aggregate_track_observations(
                positions=positions,
                colors_uint8=colors,
                scales=scales,
                frame_slots=slots,
                n_frames=n_track_frames,
                voxel_size_m=float(self.config.get("voxel_size_m", 0.02)),
                max_points=int(self.config.get("num_point_cloud_points_per_track", 2048)),
                spatial_k=int(self.config.get("spatial_propagation_k", 4)),
                spatial_iterations=int(self.config.get("spatial_propagation_iterations", 2)),
                min_scale_voxel_ratio=float(self.config.get("min_scale_voxel_ratio", 0.2)),
                scale_multiplier=float(self.config.get("scale_multiplier", 1.0)),
            )
            if n_track_frames < max_track_frames:
                padding = max_track_frames - n_track_frames
                track_rgb = torch.cat([track_rgb, track_rgb[:, -1:, :].expand(-1, padding, -1)], dim=1)
                track_observed = torch.cat(
                    [
                        track_observed,
                        torch.zeros(
                            (track_observed.shape[0], padding),
                            dtype=torch.bool,
                            device=track_observed.device,
                        ),
                    ],
                    dim=1,
                )

            all_positions.append(track_positions)
            all_scales.append(track_scales.unsqueeze(-1))
            all_temporal_albedo.append(RGB2SH(track_rgb))
            all_observed.append(track_observed)
            gaussian_cuboid_ids.append(
                torch.full(
                    (track_positions.shape[0], 1),
                    track_index,
                    dtype=torch.int32,
                    device=self.device,
                )
            )

        if not all_positions:
            raise NoPointsFoundException(f"No vehicle-light points were produced from masks in {mask_dir}")

        positions = torch.cat(all_positions)
        scales = torch.cat(all_scales)
        temporal_albedo = torch.cat(all_temporal_albedo)
        observed = torch.cat(all_observed)
        layer_limit = self.config.get("num_point_cloud_points_in_layer")
        if layer_limit is not None and positions.shape[0] > int(layer_limit):
            keep = torch.topk(observed.sum(dim=1), k=int(layer_limit), largest=True).indices
            positions = positions[keep]
            scales = scales[keep]
            temporal_albedo = temporal_albedo[keep]
            flat_ids = torch.cat(gaussian_cuboid_ids).squeeze(-1)[keep]
            gaussian_cuboid_ids = [flat_ids.unsqueeze(-1)]

        self.init_from_positions(positions, observation_scale=scales)
        self.features_albedo = nn.Parameter(temporal_albedo[:, 0].contiguous())
        self.features_albedo_temporal = temporal_albedo.contiguous()
        self.features_specular = nn.Parameter(
            torch.zeros(
                positions.shape[0],
                sh_degree_to_specular_dim(self.gaussian_config.progressive_training.max_n_features),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.gaussian_cuboid_ids = gaussian_cuboid_ids


class VehicleLightRigidGaussianModel(RigidGaussianModel):
    """Rigid vehicle-light Gaussians whose DC color is indexed by track frame."""

    config: VehicleLightRigidGaussiansLayerConfig  # type: ignore[assignment]
    mcmc_position_anchor: nn.Buffer | nn.UninitializedBuffer
    _frame_slot_cache: _TrackFrameSlotCache | None = None

    def __init__(
        self,
        config: VehicleLightRigidGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: Optional[BaseInitialization],
        precision: Optional[int] = None,
        cuboid_tracks: Optional[CuboidTracks] = None,
        start_end_timestamp_us: Optional[tuple[int, int]] = None,
    ) -> None:
        if unpack_optional(config.fourier_features_dim) != 1:
            raise ValueError("vehicle-light frame tables require fourier_features_dim=1")
        super().__init__(
            config=config,
            trainer_config=trainer_config,
            datasource=datasource,
            init_from_datasource=init_from_datasource,
            initializer=initializer,
            precision=precision,
            cuboid_tracks=cuboid_tracks,
            start_end_timestamp_us=start_end_timestamp_us,
        )
        self.mcmc_position_anchor = nn.UninitializedBuffer()

    def has_temporal_appearance_for_export(self) -> bool:
        """Opt the explicit per-track-frame albedo table into animated SH export."""

        return True

    def initialize_gaussians_from_datasource(
        self,
        initializer: BaseInitialization,
        datasource: DataSourceSummary,
        cuboid_tracks: Optional[CuboidTracks] = None,
    ) -> None:
        if not isinstance(initializer, VehicleLightTracksInitialization):
            raise TypeError("VehicleLightRigidGaussianModel requires VehicleLightTracksInitialization")
        super().initialize_gaussians_from_datasource(initializer, datasource, cuboid_tracks)
        world_rank, world_size = self._get_slice_boundaries()
        self.features_albedo = nn.Parameter(initializer.features_albedo_temporal[world_rank::world_size].contiguous())
        self.mcmc_position_anchor = nn.Buffer(self.positions.detach().clone())

    def _validate_fields(self) -> None:
        BaseGaussianModel._validate_fields(self)
        n_gaussians = self.get_num_gaussians()
        if self.features_albedo.ndim != 3 or self.features_albedo.shape[0] != n_gaussians:
            raise ValueError(f"Invalid temporal albedo shape: {tuple(self.features_albedo.shape)}")
        if self.features_albedo.shape[-1] != 3:
            raise ValueError(f"Temporal albedo must end in RGB/SH DC: {tuple(self.features_albedo.shape)}")
        if self.features_specular.shape != (
            n_gaussians,
            sh_degree_to_specular_dim(self.config.progressive_training.max_n_features),
        ):
            raise ValueError(f"Invalid shared specular shape: {tuple(self.features_specular.shape)}")
        if self.gaussian_cuboid_ids.shape != (n_gaussians,):
            raise ValueError(f"Invalid Gaussian track IDs: {tuple(self.gaussian_cuboid_ids.shape)}")
        if self.mcmc_position_anchor.shape != (n_gaussians, 3):
            raise ValueError(f"Invalid MCMC position anchor: {tuple(self.mcmc_position_anchor.shape)}")

    def configure_sharded_params_and_buffers(self) -> list[str]:
        configured = super().configure_sharded_params_and_buffers()
        BaseModel.mark_as_sharded(self.mcmc_position_anchor)
        return configured + ["mcmc_position_anchor"]

    def get_additional_buffers(self) -> dict[str, torch.Tensor]:
        buffers = super().get_additional_buffers()
        buffers["mcmc_position_anchor"] = self.mcmc_position_anchor
        return buffers

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        configured = super().configure_optimizers(name_prefix)
        features_albedo_lr = adjust_learning_rate_for_batch_size(self.trainer_config, self.config.features_albedo_lr)
        updated_groups = 0
        for optimizer_config in configured:
            for param_group in optimizer_config["optimizer"].param_groups:
                if param_group.get("name", "").rsplit(".", 1)[-1] == "features_albedo":
                    param_group["lr"] = features_albedo_lr
                    if "initial_lr" in param_group:
                        param_group["initial_lr"] = features_albedo_lr
                    updated_groups += 1
        if updated_groups != 1:
            raise RuntimeError(f"Expected one features_albedo optimizer group, found {updated_groups}")
        return configured

    def get_layer_data(
        self,
        context: BaseGaussianModel.CollectionContext,
        gathered_parameters: dict[str, torch.Tensor],
    ) -> collect.LayerDataRigid:
        layer_data = super().get_layer_data(context, gathered_parameters)
        if layer_data.features_albedo.ndim != 3:
            raise ValueError(
                f"Expected gathered vehicle-light albedo [G,N,3], got {tuple(layer_data.features_albedo.shape)}"
            )
        # Temporal albedo columns are keyed to the original datasource track frames.
        # Calibration and actor edits alter motion only and carry no source-frame-slot provenance.
        if self._frame_slot_cache is None:
            self._frame_slot_cache = _TrackFrameSlotCache()
        frame_slots = nearest_track_frame_slots(
            self.get_frame_timestamp(context.rendering_data),
            layer_data.tracks_ids,
            self.cuboid_tracks.tracks_timestamps_us,
            self.cuboid_tracks.tracks_packinfo,
            cache=self._frame_slot_cache,
        )
        row_indices = torch.arange(frame_slots.shape[0], device=frame_slots.device)
        layer_data.features_albedo = layer_data.features_albedo[row_indices, frame_slots]
        layer_data.embed_data = None
        return layer_data

    def export_ply(
        self,
        export_dir: Path,
        format: GaussianExportFormat = GaussianExportFormat._3DGS,
        percentage_gaussians: float = 100,
    ) -> None:
        if format != GaussianExportFormat._3DGS:
            super().export_ply(export_dir, format=format, percentage_gaussians=percentage_gaussians)
            return

        for track_index, track_id in enumerate(self.cuboid_tracks.tracks_id):
            track_mask = self.gaussian_cuboid_ids == track_index
            if not track_mask.any():
                continue
            n_track_frames = int(self.cuboid_tracks.tracks_packinfo[track_index, 1].item())
            export_albedo = self.features_albedo[track_mask, :n_track_frames].mean(dim=1)
            write_ply_3dgs(
                export_dir / f"{track_id}.ply",
                self.positions[track_mask],
                self.rotations[track_mask],
                self.scales[track_mask],
                self.densities[track_mask],
                export_albedo,
                self.features_specular[track_mask],
            )


BaseInitialization.register_to_gaussian_initialization_factory(
    "camera-vehicle-light-tracks", VehicleLightTracksInitialization
)
BaseGaussianModel.register_to_gaussians_factory("vehicle-light-rigid-gaussians", VehicleLightRigidGaussianModel)
