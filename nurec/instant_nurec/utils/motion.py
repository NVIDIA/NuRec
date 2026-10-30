# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from nurec.utils.misc import unpack_optional


if TYPE_CHECKING:
    from nurec.datasets.tracks import CuboidTracks


@dataclass(kw_only=True, slots=True)
class TimeRemapping:
    """
    Map from start_timestamp_us to end_timestamp_us to 0-1.
    """

    start_timestamp_us: int
    end_timestamp_us: int
    frame_gap_timestamps_us: torch.Tensor

    @classmethod
    def from_timestamps_startend_us(
        cls, timestamps_startend_us_cpu: torch.Tensor, frames_camera_idxs: torch.Tensor
    ) -> TimeRemapping:
        assert timestamps_startend_us_cpu.shape[1] == 2, "Timestamps must be (V, 2)"
        return cls(
            # Already a CPU copy; use directly
            start_timestamp_us=int(timestamps_startend_us_cpu[:, 0].min().item()),
            end_timestamp_us=int(timestamps_startend_us_cpu[:, 1].max().item()),
            frame_gap_timestamps_us=cls._compute_frame_gap(timestamps_startend_us_cpu, frames_camera_idxs),
        )

    @staticmethod
    def _compute_frame_gap(frames_timestamps_us: torch.Tensor, frames_camera_idxs: torch.Tensor) -> torch.Tensor:
        """
        Compute the timestamp gap from each frame to its nearest prev/next neighbor (at the same camera).
        Input:
            frames_timestamps_us: (V, 2) of start/end timestamps
            frames_camera_idxs: (V,)
        Output:
            gap_timestamps_us: (V, 2). If either prev/next is missing, it's set to the existing one.
        will set to 0.5s if neither exists.
        """
        # Compute median timestamp for each frame
        frames_timestamps_us = (
            frames_timestamps_us[:, 0] + (frames_timestamps_us[:, 1] - frames_timestamps_us[:, 0]) // 2
        )

        prev_gap_timestamps_us = torch.zeros_like(frames_timestamps_us)
        next_gap_timestamps_us = torch.zeros_like(frames_timestamps_us)

        # Process for each camera
        num_cameras: int = frames_camera_idxs.unique().shape[0]
        for camera_idx in range(num_cameras):
            camera_mask = torch.where(frames_camera_idxs == camera_idx)[0]
            sorted_time, sorted_idx = torch.sort(frames_timestamps_us[camera_mask])
            sorted_time_gap = sorted_time[1:] - sorted_time[:-1]
            # For prev we miss the first one, for next we miss the last one
            prev_gap_timestamps_us[camera_mask[sorted_idx[1:]]] = sorted_time_gap
            next_gap_timestamps_us[camera_mask[sorted_idx[:-1]]] = sorted_time_gap

        # Fill in missing values
        prev_gap_timestamps_us = torch.where(
            prev_gap_timestamps_us == 0, next_gap_timestamps_us, prev_gap_timestamps_us
        )
        next_gap_timestamps_us = torch.where(
            next_gap_timestamps_us == 0, prev_gap_timestamps_us, next_gap_timestamps_us
        )
        gap_timestamps_us = torch.stack([prev_gap_timestamps_us, next_gap_timestamps_us], dim=-1)
        gap_timestamps_us[gap_timestamps_us == 0] = 500000

        return gap_timestamps_us

    def timestamps_us_to_continuous_times(self, timestamps_us: torch.Tensor) -> torch.Tensor:
        span = self.end_timestamp_us - self.start_timestamp_us
        if span == 0:
            return torch.zeros_like(timestamps_us, dtype=torch.float32)
        return (timestamps_us - self.start_timestamp_us) / span

    def continuous_times_to_timestamps_us(self, continuous_times: torch.Tensor) -> torch.Tensor:
        return (continuous_times * (self.end_timestamp_us - self.start_timestamp_us)).long() + self.start_timestamp_us

    def time_span_s(self) -> float:
        return (self.end_timestamp_us - self.start_timestamp_us) / 1e6


def associate_points_with_cuboid_tracks(
    points: torch.Tensor,
    points_timestamps_us: torch.Tensor,
    points_dynamic_mask: torch.Tensor | None,
    cuboid_tracks: CuboidTracks,
    cuboids_dims_padding: torch.Tensor | None,
    second_pass_cuboids_dims_padding_scale: float = 6.0,
    second_pass_chunk_size: int = 65536,
) -> torch.Tensor:
    """
    Associate each point with a cuboid track at its timestamp.

    Association runs a standard point-cuboid pass first. Movable points that remain
    unassigned get a second pass with wider padding; if multiple widened boxes
    contain the point, the closest original bbox wins.

    Convention: per-point scalars (timestamps, dynamic mask, returned track index) all carry
    a trailing singleton dim, mirroring ``points``' trailing 3-vector dim.

    Args:
        points: [..., 3] world points at the source timestamps.
        points_timestamps_us: [..., 1] source timestamps per point.
        points_dynamic_mask: [..., 1] bool, True for points to associate.
        cuboid_tracks: CuboidTracks to associate against. It's recommended to pass in all tracks
            (including static ones).
        cuboids_dims_padding: optional 3D padding broadcastable to N_tracks x 3.
        second_pass_cuboids_dims_padding_scale: scale factor applied to ``cuboids_dims_padding``
            (or to a 1.0 padding when it is None) for the second-pass widened bbox check.
        second_pass_chunk_size: number of points processed at once in the second pass.
            Lower it if the second-pass kernel OOMs.

    Returns:
        - tracks_idx: [..., 1] int, the cuboid-track index each point was associated with
          (indexes into ``cuboid_tracks.tracks_id``). ``-1`` for unassociated points.
    """
    data_shape = points.shape[:-1]
    expected_scalar_shape = data_shape + (1,)
    assert points_timestamps_us.shape == expected_scalar_shape
    if points_dynamic_mask is not None:
        assert points_dynamic_mask.shape == expected_scalar_shape

    if cuboid_tracks.n_tracks == 0:
        return torch.full(expected_scalar_shape, -1, device=points.device, dtype=cuboid_tracks.tracks_packinfo.dtype)

    points_ts = points_timestamps_us.squeeze(-1)

    # First pass: standard point-cuboid intersection with the configured padding.
    first_pass = cuboid_tracks.point_intersection_interpolate_pose(
        points, points_ts, cuboids_dims_padding, max_intersections_per_point=1
    )
    tracks_idx = first_pass.intersections_tracks_idx  # [..., 1]

    if points_dynamic_mask is None:
        return tracks_idx

    # Second pass: only movable points that missed the first pass. Use a wider box,
    # then select the hit whose original bbox is closest to the point.
    second_pass_mask = (points_dynamic_mask & (tracks_idx == -1)).squeeze(-1)
    n_second_pass = int(second_pass_mask.sum().item())
    if n_second_pass == 0:
        return tracks_idx

    if cuboids_dims_padding is None:
        wide_cuboids_dims_padding = torch.full(
            (3,),
            second_pass_cuboids_dims_padding_scale,
            device=points.device,
            dtype=points.dtype,
        )
    else:
        wide_cuboids_dims_padding = (
            cuboids_dims_padding.to(device=points.device, dtype=points.dtype) * second_pass_cuboids_dims_padding_scale
        )

    cuboids_dims = cuboid_tracks.cuboids_dims.to(device=points.device, dtype=points.dtype)
    sp_points = points[second_pass_mask]
    sp_points_ts = points_ts[second_pass_mask]

    # Process in chunks to bound peak GPU memory; the wide pass intermediates would
    # otherwise be multi-GiB for full-frame inputs.
    best_tracks_idx_sp = torch.full((n_second_pass, 1), -1, device=points.device, dtype=tracks_idx.dtype)
    has_any_sp = torch.zeros((n_second_pass, 1), device=points.device, dtype=torch.bool)
    chunk_size = max(1, second_pass_chunk_size)
    for start in range(0, n_second_pass, chunk_size):
        end = min(start + chunk_size, n_second_pass)
        second_pass = cuboid_tracks.point_intersection_interpolate_pose(
            points=sp_points[start:end],
            points_timestamps_us=sp_points_ts[start:end],
            cuboids_dims_padding=wide_cuboids_dims_padding,
            max_intersections_per_point=64,
            with_local_points=True,
        )
        wide_tracks_idx = second_pass.intersections_tracks_idx  # [N_sp_chunk, 64]
        wide_points_local = unpack_optional(second_pass.intersections_points_local)  # [N_sp_chunk, 64, 3]
        wide_valid_mask = wide_tracks_idx != -1
        if not torch.any(wide_valid_mask):
            continue

        # Compute point to box distance: point to the box closest corner/edge/face.
        # If the point is inside then distance is 0.
        matched_dims = cuboids_dims[wide_tracks_idx.clamp_min(0).to(torch.long)]
        half_dims = matched_dims * 0.5
        distance_to_bbox = torch.linalg.norm(
            (wide_points_local.abs() - half_dims).clamp_min(0.0),
            dim=-1,
        )
        distance_to_bbox = torch.where(
            wide_valid_mask,
            distance_to_bbox,
            torch.full_like(distance_to_bbox, torch.inf),
        )

        # Assign points to nearest bbox.
        best_hit_idx = torch.argmin(distance_to_bbox, dim=-1, keepdim=True)
        best_tracks_idx_sp[start:end] = torch.gather(wide_tracks_idx, dim=-1, index=best_hit_idx)
        has_any_sp[start:end] = wide_valid_mask.any(dim=-1, keepdim=True)

    tracks_idx[second_pass_mask] = torch.where(
        has_any_sp,
        best_tracks_idx_sp,
        tracks_idx[second_pass_mask],
    )

    return tracks_idx


def warp_points_with_cuboid_tracks(
    points: torch.Tensor,
    source_timestamps_us: torch.Tensor,
    target_timestamps_us_list: list[torch.Tensor],
    cuboid_tracks: CuboidTracks,
    tracks_idx: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Warp points assigned to cuboid tracks to each given target timestamp using the
    assigned track's interpolated pose.

    Args:
        points: [..., 3] world points at the source timestamps.
        source_timestamps_us: [..., 1] source timestamps per point.
        target_timestamps_us_list: list of target timestamp tensors, each [..., 1].
        cuboid_tracks: CuboidTracks to warp against.
        tracks_idx: [..., 1] track indices assigned by ``associate_points_with_cuboid_tracks``.

    Returns:
        - dynamic_mask: [..., 1] bool, True for points associated with a track.
        - warped_points_list: per target, a [..., 3] tensor where associated points are
          warped to the corresponding target timestamp and unassociated points are unchanged.
    """
    data_shape = points.shape[:-1]
    expected_scalar_shape = data_shape + (1,)
    assert source_timestamps_us.shape == expected_scalar_shape
    for i, t in enumerate(target_timestamps_us_list):
        assert t.shape == expected_scalar_shape
    assert tracks_idx.shape == expected_scalar_shape

    src_ts = source_timestamps_us.squeeze(-1)
    tgt_ts_list = [t.squeeze(-1) for t in target_timestamps_us_list]
    tracks_idx_flat = tracks_idx.squeeze(-1)

    dynamic_mask = tracks_idx_flat != -1
    sel = torch.where(dynamic_mask)
    sel_tracks_idx = tracks_idx_flat[sel].to(dtype=cuboid_tracks.tracks_packinfo.dtype)

    warped_points_list: list[torch.Tensor]

    # Shortcut if no dynamic points are found.
    if sel_tracks_idx.numel() == 0:
        warped_points_list = [points.clone() for _ in tgt_ts_list]
        return dynamic_mask.unsqueeze(-1), warped_points_list

    inv_current_pose = cuboid_tracks.interpolate_tracks_poses(
        timestamps_us=src_ts[sel],
        tracks_idx=sel_tracks_idx,
    ).inv()

    warped_points_list = []
    for tgt_ts in tgt_ts_list:
        target_pose = cuboid_tracks.interpolate_tracks_poses(
            timestamps_us=tgt_ts[sel],
            tracks_idx=sel_tracks_idx,
        )
        new_points = points.clone()
        new_points[sel] = (target_pose * inv_current_pose) * new_points[sel]  # type: ignore[operator]
        warped_points_list.append(new_points)

    return dynamic_mask.unsqueeze(-1), warped_points_list
