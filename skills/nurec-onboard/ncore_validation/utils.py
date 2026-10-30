# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for NCore V4 validation and evaluation scripts.

Provides :class:`NcoreV4Context` for holding open V4 readers,
:func:`open_context` for constructing one from component-group paths,
and reader-lookup helpers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from ncore.data import ConcreteLidarModelParametersUnion, CuboidTrackObservation
from ncore.data.v4 import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
)
from ncore.impl.common.transformations import (
    HalfClosedInterval,  # internal API; no public equivalent yet
    se3_inverse,
    transform_bbox,
)
from ncore.impl.data.v4.components import (
    SequenceMeta,
)  # internal API; no public equivalent yet


MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG = 0.5
"""NuRec viability threshold: stored ray directions must match the LiDAR model
within this angular tolerance (degrees)."""

WORLD_ANCHORED_CUBOID_FRAMES = ("world", "world_global")
"""Cuboid ``reference_frame_id`` values that reach the ``world`` frame without
crossing the dynamic ``rig`` -> ``world`` edge: ``world`` is that frame, and
``world_global`` connects to it by a static edge.  Neither evaluates a pose at
the reference timestamp, and neither reaches ``rig`` by a static edge -- the
two properties ``check_cuboid_pose_gap`` and ``cuboid_bbox_to_rig``
respectively branch on."""


@dataclass
class NcoreV4Context:
    """Holds open readers for one NCore V4 sequence.

    Constructed via :func:`open_context` after opening
    ``SequenceComponentGroupsReader`` and each component reader.

    Attributes:
        reader: Open ``SequenceComponentGroupsReader`` for the sequence.
        interval: Sequence time span (``sequence_timestamp_interval_us``); half-open
            ``[start, stop)`` in microseconds.
        meta: Sequence metadata from ``reader.get_sequence_meta()``.
        poses_readers: Pose component readers keyed by instance name.
        camera_readers: Camera sensor readers keyed by camera id.
        intrinsics_readers: Intrinsics readers keyed by instance name.
        masks_readers: Masks readers keyed by instance name.
        lidar_readers: LiDAR sensor readers keyed by lidar id.
        radar_readers: Radar sensor readers keyed by radar id.
        cuboid_readers: Cuboid track readers keyed by instance name (empty if none).
        errors: Human-readable failure messages; used by the validation script.
    """

    reader: SequenceComponentGroupsReader
    interval: HalfClosedInterval
    meta: SequenceMeta
    poses_readers: dict[str, PosesComponent.Reader]
    camera_readers: dict[str, CameraSensorComponent.Reader]
    intrinsics_readers: dict[str, IntrinsicsComponent.Reader]
    masks_readers: dict[str, MasksComponent.Reader]
    lidar_readers: dict[str, LidarSensorComponent.Reader]
    radar_readers: dict[str, RadarSensorComponent.Reader]
    cuboid_readers: dict[str, CuboidsComponent.Reader]
    errors: list[str] = field(default_factory=list)


def open_context(
    component_groups: list[Path] | tuple[Path, ...],
    open_consolidated: bool = True,
) -> NcoreV4Context:
    """Open a V4 sequence and return a populated :class:`NcoreV4Context`.

    Args:
        component_groups: Path(s) to V4 component group directories, zarr roots, or
            a sequence meta ``*.json`` listing stores.
        open_consolidated: Whether to pre-load consolidated zarr metadata.

    Returns:
        A fully populated context with all component readers opened.

    Raises:
        Exception: If the sequence cannot be opened.
    """
    reader = SequenceComponentGroupsReader(
        list(component_groups),
        open_consolidated=open_consolidated,
    )
    return NcoreV4Context(
        reader=reader,
        interval=reader.sequence_timestamp_interval_us,
        meta=reader.get_sequence_meta(),
        poses_readers=reader.open_component_readers(PosesComponent.Reader),
        camera_readers=reader.open_component_readers(CameraSensorComponent.Reader),
        intrinsics_readers=reader.open_component_readers(IntrinsicsComponent.Reader),
        masks_readers=reader.open_component_readers(MasksComponent.Reader),
        lidar_readers=reader.open_component_readers(LidarSensorComponent.Reader),
        radar_readers=reader.open_component_readers(RadarSensorComponent.Reader),
        cuboid_readers=reader.open_component_readers(CuboidsComponent.Reader),
    )


def any_reader_has(readers: dict, lookup: Callable[[Any, str], object], instance_id: str) -> bool:
    """Return whether ``lookup`` succeeds for ``instance_id`` on any reader.

    Typical use: intrinsics or static poses keyed by camera/lidar id across
    multiple pose or intrinsics reader instances.

    Only ``KeyError`` is caught — other exceptions raised by ``lookup``
    propagate to the caller.

    Args:
        readers: Mapping of instance name to an NCore component reader.
        lookup: Callable ``(reader, instance_id) -> None``; success means the id exists.
        instance_id: Camera id, lidar id, or other key passed to ``lookup``.

    Returns:
        ``True`` if at least one reader's ``lookup`` completes without ``KeyError``.
    """
    for reader in readers.values():
        try:
            lookup(reader, instance_id)
            return True
        except KeyError:
            continue
    return False


def any_intrinsics_reader_has_lidar_model(intrinsics_readers: dict, lidar_id: str) -> bool:
    """Return whether ``lidar_id`` has stored LiDAR model parameters on any intrinsics reader.

    Uses :meth:`IntrinsicsComponent.Reader.get_lidar_model_parameters`, which returns
    ``None`` when the id is missing (unlike camera parameters, which may raise).

    Args:
        intrinsics_readers: Mapping of instance name to ``IntrinsicsComponent.Reader``.
        lidar_id: LiDAR sensor instance id.

    Returns:
        ``True`` if at least one reader returns a non-``None`` model for ``lidar_id``.
    """
    for reader in intrinsics_readers.values():
        if reader.get_lidar_model_parameters(lidar_id) is not None:
            return True
    return False


def get_lidar_model_parameters_first(
    intrinsics_readers: dict[str, IntrinsicsComponent.Reader],
    lidar_id: str,
) -> ConcreteLidarModelParametersUnion | None:
    """Return LiDAR intrinsic model parameters for ``lidar_id`` from the first reader that has them.

    Args:
        intrinsics_readers: Mapping of instance name to ``IntrinsicsComponent.Reader``.
        lidar_id: LiDAR sensor instance id.

    Returns:
        The model parameters, or ``None`` if no reader has them for ``lidar_id``.
    """
    for reader in intrinsics_readers.values():
        params = reader.get_lidar_model_parameters(lidar_id)
        if params is not None:
            return cast(ConcreteLidarModelParametersUnion, params)
    return None


def _get_world_to_world_global(ctx: NcoreV4Context) -> np.ndarray | None:
    """Return the static ``world -> world_global`` SE3 if any reader has it."""
    for pose_reader in ctx.poses_readers.values():
        try:
            return pose_reader.get_static_pose("world", "world_global").astype(np.float64)
        except KeyError:
            continue
    return None


def cuboid_bbox_to_rig(
    obs: CuboidTrackObservation,
    ctx: NcoreV4Context,
    T_rig_worlds: np.ndarray | None = None,
    rig_ts: np.ndarray | None = None,
) -> np.ndarray | None:
    """Transform a cuboid observation's BBox3 into the rig frame.

    Handles these ``reference_frame_id`` cases:

    * ``"rig"`` -- returned as-is.
    * A sensor id (e.g. ``"lidar_top"``) -- looked up via sensor->rig
      static pose.
    * ``"world"`` -- rig->world dynamic pose at the observation's
      ``reference_frame_timestamp_us`` is inverted.
    * ``"world_global"`` -- chained through the static
      ``world -> world_global`` edge and the rig->world dynamic pose.

    Args:
        obs: A ``CuboidTrackObservation`` instance.
        ctx: Open NCore V4 sequence context (for pose lookups).
        T_rig_worlds: ``[N, 4, 4]`` rig->world dynamic poses, or ``None``.
        rig_ts: ``[N]`` timestamps for *T_rig_worlds*, or ``None``.

    Returns:
        9-vector ``[cx,cy,cz, L,W,H, rx,ry,rz]`` in rig frame, or
        ``None`` if the transform is unavailable.
    """
    bbox3 = getattr(obs, "bbox3", None)
    if bbox3 is None:
        return None

    bbox_arr = bbox3.to_array().astype(np.float64)
    ref_frame = getattr(obs, "reference_frame_id", "rig")

    if ref_frame == "rig":
        return bbox_arr

    if ref_frame not in WORLD_ANCHORED_CUBOID_FRAMES:
        for pose_reader in ctx.poses_readers.values():
            try:
                T_ref_rig = pose_reader.get_static_pose(ref_frame, "rig")
                return transform_bbox(bbox_arr, T_ref_rig)
            except KeyError:
                continue
        return None

    if T_rig_worlds is None or rig_ts is None or rig_ts.size == 0:
        return None
    if T_rig_worlds.shape[0] != rig_ts.size:
        return None
    ref_ts = getattr(obs, "reference_frame_timestamp_us", None)
    if ref_ts is None:
        return None
    # Find the pose waypoint closest to the cuboid's reference timestamp.
    # searchsorted gives the first index >= ref_ts; check the left neighbor
    # in case it's actually closer (e.g. ref_ts sits just below rig_ts[idx]).
    idx = int(np.clip(np.searchsorted(rig_ts, ref_ts), 0, len(rig_ts) - 1))
    if idx > 0 and abs(int(rig_ts[idx - 1]) - int(ref_ts)) < abs(int(rig_ts[idx]) - int(ref_ts)):
        idx -= 1
    T_world_rig = se3_inverse(T_rig_worlds[idx].astype(np.float64))

    if ref_frame == "world":
        return transform_bbox(bbox_arr, T_world_rig)

    # reference_frame_id == "world_global": compose world_global -> world -> rig.
    T_world_world_global = _get_world_to_world_global(ctx)
    if T_world_world_global is None:
        return None
    T_world_global_rig = T_world_rig @ se3_inverse(T_world_world_global)
    return transform_bbox(bbox_arr, T_world_global_rig)


class LidarCoverageIndex:
    """Union rig-frame LiDAR points from frames containing an instant."""

    def __init__(self, ctx: NcoreV4Context, lidar_pts_rig: dict[str, dict[int, np.ndarray]]) -> None:
        """Index the stored point clouds by sensor.

        Args:
            ctx: Open NCore V4 sequence context, read for frame windows.
            lidar_pts_rig: Rig-frame point clouds, keyed by LiDAR id then frame
                end timestamp, as returned by ``check_lidar_frame_data``.
        """
        self._sensors: list[tuple[str, np.ndarray, np.ndarray, dict[int, np.ndarray]]] = []
        for lidar_id, per_sensor in lidar_pts_rig.items():
            if not per_sensor:
                continue
            reader = ctx.lidar_readers.get(lidar_id)
            if reader is None:
                continue
            timestamps = np.asarray(reader.frames_timestamps_us, dtype=np.int64)
            if timestamps.ndim != 2 or timestamps.shape[1] != 2:
                continue
            windows_by_end = {int(end): (int(start), int(end)) for start, end in timestamps}
            keys = np.array(sorted(per_sensor), dtype=np.int64)
            if any(int(key) not in windows_by_end for key in keys):
                continue
            starts = np.array([windows_by_end[int(key)][0] for key in keys], dtype=np.int64)
            self._sensors.append((lidar_id, keys, starts, per_sensor))
        self._last_keys: tuple[tuple[str, int], ...] | None = None
        self._last_points: np.ndarray | None = None

    def points_near(self, timestamp_us: int) -> np.ndarray | None:
        """Return the union of every sensor's frame covering ``timestamp_us``.

        Args:
            timestamp_us: The instant to gather coverage for.

        Returns:
            Concatenated rig-frame points, or ``None`` when no sensor stores a
            frame window containing that instant.
        """
        chosen: list[tuple[str, int, np.ndarray]] = []
        for lidar_id, keys, starts, per_sensor in self._sensors:
            covering = np.flatnonzero((starts <= timestamp_us) & (timestamp_us <= keys))
            if covering.size == 0:
                continue
            nearest_idx = covering[np.argmin(np.abs(keys[covering] - np.int64(timestamp_us)))]
            nearest_key = int(keys[nearest_idx])
            pts = per_sensor[nearest_key]
            if pts.size:
                chosen.append((lidar_id, nearest_key, pts))
        if not chosen:
            return None

        cache_key = tuple((lidar_id, timestamp) for lidar_id, timestamp, _ in chosen)
        if cache_key == self._last_keys and self._last_points is not None:
            return self._last_points
        self._last_keys = cache_key
        self._last_points = np.concatenate([pts for _, _, pts in chosen], axis=0)
        return self._last_points


def sample_evenly(items: list, cap: int, *, key: Callable[[Any], Any] | None = None) -> list:
    """Return at most ``cap`` items spread across an optional sort key.

    Args:
        items: The full list to sample from.
        cap: Maximum number of items to return.
        key: Optional ordering key, such as an observation timestamp.

    Returns:
        The ordered input when it is within ``cap``; otherwise an evenly spaced
        subset of that ordering.
    """
    if cap <= 0:
        return []
    ordered = sorted(items, key=key) if key is not None else items
    if len(ordered) <= cap:
        return ordered
    idx = np.linspace(0, len(ordered) - 1, cap).astype(np.int64)
    return [ordered[int(i)] for i in np.unique(idx)]
