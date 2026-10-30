# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ncore.impl.common.transformations import PoseInterpolator
from nurec.datasets.tracks import CuboidTracks, TrackFlags, Tracks
from nurec.utils.geometry import se3_matrix_to_tquat
from nurec.utils.types import CuboidTracksData


def clip_window_to_cuboid_track(
    cuboid_track: CuboidTracks,
    frame_start_us: int,
    frame_end_us: int,
) -> Optional[Tuple[int, int]]:
    """Clip [frame_start_us, frame_end_us] to a cuboid track's keyframe timestamp span.

    Returns (ts_start, ts_end) with ts_start <= ts_end, or None if there is no overlap.

    A zero-length window (``frame_start_us == frame_end_us``) is treated as a point query at that
    instant (instant shutter / missing frame-start timestamps / ``--disable-rolling-shutter``): the
    track overlaps when the instant lies within ``[track_min, track_max]`` inclusive. The half-open
    ``frame_end_us <= track_min`` test used for positive-length windows would otherwise drop a track
    whose first keyframe is exactly ``frame_end_us``.
    """
    track_ts = cuboid_track.tracks_timestamps_us
    if track_ts.numel() == 0:
        return None
    track_min = int(track_ts.min().item())
    track_max = int(track_ts.max().item())
    # Boundary convention (customer-observable, shared by ALL callers — MODEL/ARTIFACT bbox overlays,
    # the render CLI, and the actor-editing export path): the capture window is half-open
    # [frame_start_us, frame_end_us), so a positive-length window uses `<=` and EXCLUDES a track whose
    # first keyframe is exactly frame_end_us. This is a deliberate change from the pre-refactor
    # inclusive `frame_end < track_min` check that lived inline in get_snapshot_at_frame (a track
    # starting exactly at frame_end used to be included). Zero-length point queries instead use an
    # inclusive `<` bound so an instant-shutter frame at exactly track_min is not dropped (see docstring).
    is_point_query = frame_start_us == frame_end_us
    starts_before_track = frame_end_us < track_min if is_point_query else frame_end_us <= track_min
    if starts_before_track or frame_start_us > track_max:
        return None
    ts_start = max(frame_start_us, track_min)
    ts_end = min(frame_end_us, track_max)
    return ts_start, ts_end


# TODO: Test on static scenes whether this works with zero tracks
@dataclass(frozen=True)  # Frozen to guarantee integrity via the __post_init_() check at all times
class ActorsSnapshot:
    """Class to represent the poses of each controllable actors present at the start and end of a single frame capture

    Each actor has two poses because of rolling shutter support.
    The contents are specific to the time range used to obtain this "snapshot", e.g. from ActorTracks.
    """

    # TODO: Switch to tensor of repeatable indices (~repeated actor instances) for a full vectorization.
    actor_ids: List[str]  # list of length M, Track ID of each actor present within the frame capture time interval
    actor_poses: torch.Tensor  # [M, 2, 7], start and end tquat pose of each actor's cuboid, [tx,ty,tz,qx,qy,qz,qw]
    # poses[i,0,:] is the start pose and poses[i,1,:] is the end pose of the actor with ID ids[i].
    cuboid_dims: Optional[torch.Tensor] = None  # Per-actor full edge lengths (shape (M,3), order matching actor_ids)

    @classmethod
    def empty(cls):
        return cls(actor_ids=[], actor_poses=torch.empty((0, 2, 7), dtype=torch.float32), cuboid_dims=None)

    def __post_init__(self):
        assert self.actor_poses.shape == (len(self.actor_ids), 2, 7), (
            f"actor_poses shape {tuple(self.actor_poses.shape)} != ({len(self.actor_ids)}, 2, 7)"
        )
        if self.cuboid_dims is not None:
            assert self.cuboid_dims.shape == (len(self.actor_ids), 3), (
                f"cuboid_dims {tuple(self.cuboid_dims.shape)} does not match {len(self.actor_ids)} actor_ids"
            )

    def num_actors(self) -> int:
        """Number of actors contained in the snapshot"""
        return len(self.actor_ids)


@dataclass(frozen=True)
class ActorTracks:
    """Class representing controllable actors in the scene and their trajectories"""

    # These are not part of the public API, the user is not meant to access these directly.
    # TODO: vectorize these
    _cuboid_tracks_list: List[CuboidTracks] = field(default_factory=list)
    _interpolator_list: List[PoseInterpolator] = field(default_factory=list)
    # Cached at construction; avoids rebuilding O(M) dict on every get_snapshot_at_frame call.
    _id_to_cuboid_dims: Dict[str, torch.Tensor] = field(default_factory=dict, init=False, compare=False, repr=False)

    def __post_init__(self):
        id_to_dims: Dict[str, torch.Tensor] = {}
        for cuboid_track in self._cuboid_tracks_list:
            assert len(cuboid_track.tracks_id) == 1, (
                f"Expected single-ID CuboidTrack, got {list(cuboid_track.tracks_id)}"
            )
            tid = cuboid_track.tracks_id[0]
            id_to_dims[tid] = cuboid_track.cuboids_dims[0]
        object.__setattr__(self, "_id_to_cuboid_dims", id_to_dims)

    # TODO: Test on static scenes whether this works with zero tracks
    @staticmethod
    def _from_cuboid_tracks(cuboid_tracks: CuboidTracks, controllable_only: bool = True):
        """Private factory method, only use internally.

        Args:
            controllable_only: When True (default), only include tracks flagged CONTROLLABLE.
                Pass False to include all tracks regardless of flag (used for annotation overlays).
        """
        if controllable_only:
            source_tracks = CuboidTracks.Ops.subset_from_mask(
                cuboid_tracks, cuboid_tracks.get_mask_flags_all(TrackFlags.CONTROLLABLE)
            )
        else:
            source_tracks = cuboid_tracks
        interpolator_list: List[PoseInterpolator] = []
        cuboid_tracks_list: List[CuboidTracks] = []
        for track_idx in range(source_tracks.n_tracks):
            cuboid_track = CuboidTracks.Ops.subset_from_indices(source_tracks, [track_idx])
            cuboid_track = CuboidTracks.Ops.clone(cuboid_track)
            cuboid_tracks_list.append(cuboid_track)
            interpolator_list.append(
                PoseInterpolator(
                    poses=cuboid_track.tracks_poses.matrix().cpu(), timestamps=cuboid_track.tracks_timestamps_us.cpu()
                )
            )
        return ActorTracks(_cuboid_tracks_list=cuboid_tracks_list, _interpolator_list=interpolator_list)

    @staticmethod
    def from_dict_cpu(cuboid_tracks_dict: dict) -> ActorTracks:
        """Build actor tracks on CPU from serialized cuboid tracks.

        For code that must not use the GPU: ``CuboidTracks.from_dict`` decodes onto CUDA and
        ``CuboidTracks.Ops`` subsets run CUDA kernels.

        Args:
            cuboid_tracks_dict: ``CuboidTracks.to_dict()`` output, with float32 tquat poses and
                int64 microsecond timestamps per track.

        Returns:
            The CONTROLLABLE tracks in source order, each a single-track ``CuboidTracks`` with CPU
            tensors, with their pose interpolators. Empty when no track is controllable.

        Raises:
            KeyError: if ``tracks_data`` or ``cuboidtracks_data`` is missing.
            AssertionError: if a track's poses or timestamps are malformed.
        """
        cpu = torch.device("cpu")
        unpacked = Tracks.Unpacked.from_dict(cuboid_tracks_dict["tracks_data"])
        cuboids_dims = CuboidTracks.Unpacked.from_dict(cuboid_tracks_dict["cuboidtracks_data"]).cuboids_dims
        cuboid_tracks_list: List[CuboidTracks] = []
        interpolator_list: List[PoseInterpolator] = []
        for idx, flags in enumerate(unpacked.tracks_flags):
            if flags & TrackFlags.CONTROLLABLE != TrackFlags.CONTROLLABLE:
                continue
            track = Tracks.Factory.from_numpy(
                [unpacked.tracks_id[idx]],
                [np.array(unpacked.tracks_poses[idx], dtype=np.float32)],
                [np.array(unpacked.tracks_timestamps_us[idx], dtype=np.int64)],
                [unpacked.tracks_label_class[idx]],
                [flags],
                "tquat",
                device=cpu,
            )
            cuboid_track = CuboidTracks(
                tracks_data=track.tracks_data,
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=torch.tensor(cuboids_dims[idx], dtype=torch.float32, device=cpu).reshape(1, 3)
                ),
            )
            cuboid_tracks_list.append(cuboid_track)
            interpolator_list.append(
                PoseInterpolator(poses=cuboid_track.tracks_poses.matrix(), timestamps=cuboid_track.tracks_timestamps_us)
            )
        return ActorTracks(_cuboid_tracks_list=cuboid_tracks_list, _interpolator_list=interpolator_list)

    def num_tracks(self) -> int:
        """Get the number of tracks"""
        return len(self._cuboid_tracks_list)

    def known_track_ids(self) -> List[str]:
        """Return the track ids known to this instance (the keys used by :meth:`get_cuboid_dims`)."""
        return list(self._id_to_cuboid_dims.keys())

    # TODO: Test on static scenes whether this works with zero tracks
    def get_snapshot_at_frame(self, frame_start_timestamp_us: int, frame_end_timestamp_us: int) -> ActorsSnapshot:
        """Return the interpolated poses of all actors whose keyframe span overlaps the capture window.

        A track is included when its keyframe timestamps overlap the half-open window
        [frame_start_timestamp_us, frame_end_timestamp_us) (i.e. not strictly contained), except that a track
        starting exactly at frame_end_timestamp_us is excluded while one ending exactly at frame_start_timestamp_us
        is still included (the track span is effectively [track_min, track_max + 1) for integer timestamps).
        A zero-length window (frame_start_timestamp_us == frame_end_timestamp_us, e.g. instant shutter or
        --disable-rolling-shutter) is treated as a point query: a track starting exactly at that instant is
        included rather than dropped (see clip_window_to_cuboid_track).
        The interpolation timestamps are clamped to [track_min, track_max] so
        the interpolator never extrapolates. This is more permissive than strict containment and is correct for
        actors whose cuboid span covers only a short visibility window (e.g. generated static rigids in NCORE).
        """
        assert len(self._cuboid_tracks_list) == len(self._interpolator_list)
        if frame_start_timestamp_us > frame_end_timestamp_us:
            raise ValueError(
                f"invalid timestamps, start time is after end time "
                f"({frame_start_timestamp_us}>{frame_end_timestamp_us})"
            )

        if len(self._cuboid_tracks_list) == 0:
            return ActorsSnapshot.empty()

        track_ids: List[str] = []
        poses_list = []  # List of [2,7] tensors containing start, end pose at start, end frame capture timestamp
        for cuboid_track, interpolator in zip(self._cuboid_tracks_list, self._interpolator_list):
            clipped = clip_window_to_cuboid_track(cuboid_track, frame_start_timestamp_us, frame_end_timestamp_us)
            if clipped is None:
                continue
            ts_start, ts_end = clipped
            world_poses = interpolator.interpolate_to_timestamps([ts_start, ts_end])
            assert isinstance(world_poses, np.ndarray)
            assert world_poses.dtype == np.float32
            assert world_poses.shape == (2, 4, 4)
            world_poses_tquat = se3_matrix_to_tquat(world_poses)
            assert world_poses_tquat.shape == (2, 7)

            assert len(cuboid_track.tracks_id) == 1  # guaranteed by __post_init__, kept for safety
            track_ids.append(cuboid_track.tracks_id[0])
            poses_list.append(world_poses_tquat)

        if len(poses_list) == 0:
            return ActorsSnapshot.empty()
        stacked_poses = torch.stack(poses_list, dim=0)
        return ActorsSnapshot(
            actor_ids=track_ids,
            actor_poses=stacked_poses,
            cuboid_dims=self.get_cuboid_dims(track_ids),
        )

    def get_cuboid_dims(self, actor_ids: List[str]) -> torch.Tensor:
        """
        Per-actor cuboid size parameters in the same order as ``actor_ids``.

        Returns a tensor of shape ``(M, 3)`` where each row is the **full edge length** (meters) along
        the three cuboid axes in that track's local frame—half-extents are ``0.5 * dim`` per axis.
        Values are in the same coordinate frame and scaling as poses from :meth:`get_snapshot_at_frame`.

        Returns:
            Shape ``(M, 3)`` tensor, or shape ``(0, 3)`` if ``actor_ids`` is empty.

        Raises:
            ValueError: if any id is not a track in this :class:`ActorTracks` instance.
        """
        if not actor_ids:
            return torch.empty((0, 3), dtype=torch.float32)
        rows: List[torch.Tensor] = []
        for aid in actor_ids:
            if aid not in self._id_to_cuboid_dims:
                raise ValueError(f"No cuboid dimensions for actor track id {aid!r}")
            rows.append(self._id_to_cuboid_dims[aid])
        return torch.stack(rows, dim=0)
