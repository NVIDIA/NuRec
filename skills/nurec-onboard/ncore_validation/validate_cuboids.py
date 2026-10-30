# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cuboid annotation checks."""

from __future__ import annotations

import logging

import numpy as np

from scipy.spatial.transform import Rotation

from ncore.impl.common.transformations import is_within_3d_bboxes
from ncore_validation.utils import (
    WORLD_ANCHORED_CUBOID_FRAMES,
    LidarCoverageIndex,
    NcoreV4Context,
    cuboid_bbox_to_rig,
    sample_evenly,
)
from ncore_validation.validate_helpers import (
    CUBOID_FLOATING_MEDIAN_RATIO,
    CUBOID_GROUND_CLEARANCE_PROBE_M,
    CUBOID_NEAR_RANGE_M,
    CUBOID_POSE_GAP_MAX_US,
    MAX_CUBOID_OBSERVATIONS,
    MAX_CUBOID_RIG_DISTANCE_P95_M,
    MIN_CUBOID_OBB_INLIER_RATIO,
    MIN_CUBOID_RIG_PROXIMITY_OBSERVATIONS,
    MIN_CUBOIDS_FOR_OBB_CHECK,
    MIN_NEAR_RANGE_CUBOID_HIT_RATE,
    MIN_NEAR_RANGE_CUBOIDS_FOR_CHECK,
    _check,
    _warn,
)


log = logging.getLogger("validate")


def nearest_idx(sorted_ts: np.ndarray, target: int) -> int:
    """Return the index into *sorted_ts* whose value is closest to *target*."""
    idx = int(np.clip(np.searchsorted(sorted_ts, target), 0, len(sorted_ts) - 1))
    if idx > 0 and abs(int(sorted_ts[idx - 1]) - target) < abs(int(sorted_ts[idx]) - target):
        idx -= 1
    return idx


def check_cuboid_ts_overlaps_sensor_range(ctx: NcoreV4Context) -> None:
    """Require at least one cuboid observation timestamp inside the sensor time hull."""
    if not (ctx.cuboid_readers and (ctx.camera_readers or ctx.lidar_readers)):
        return
    sensor_min_ts = ctx.interval.stop
    sensor_max_ts = ctx.interval.start
    for cam_reader in ctx.camera_readers.values():
        ts = cam_reader.frames_timestamps_us
        if ts.size > 0:
            sensor_min_ts = min(sensor_min_ts, int(ts[:, 0].min()))
            sensor_max_ts = max(sensor_max_ts, int(ts[:, 1].max()))
    for lidar_reader in ctx.lidar_readers.values():
        ts = lidar_reader.frames_timestamps_us
        if ts.size > 0:
            sensor_min_ts = min(sensor_min_ts, int(ts[:, 0].min()))
            sensor_max_ts = max(sensor_max_ts, int(ts[:, 1].max()))

    if sensor_max_ts <= sensor_min_ts:
        return

    has_any_observation = False
    any_cuboid_in_range = False
    for cuboid_reader in ctx.cuboid_readers.values():
        try:
            for obs in cuboid_reader.get_observations():
                has_any_observation = True
                if sensor_min_ts <= obs.timestamp_us <= sensor_max_ts:
                    any_cuboid_in_range = True
                    break
        except Exception as e:
            log.debug(
                "check_cuboid_ts_overlaps_sensor_range: failed reading observations: %s",
                e,
            )
            continue
        if any_cuboid_in_range:
            break

    if not has_any_observation:
        return

    _check(
        "cuboid timestamps overlap with sensor range",
        any_cuboid_in_range,
        ctx.errors,
        f"no cuboid timestamps fall within sensor range [{sensor_min_ts}, {sensor_max_ts}]",
    )


def check_cuboid_timestamps_in_interval(ctx: NcoreV4Context) -> None:
    """Fail when any cuboid timestamp falls outside the sequence interval.

    NCore's writer asserts that every ``obs.timestamp_us`` and
    ``obs.reference_frame_timestamp_us`` is contained in
    ``sequence_timestamp_interval_us``.  Reports up to 10 violations per
    field to avoid flooding the error output.
    """
    if not ctx.cuboid_readers:
        return
    bad_ts: list[str] = []
    bad_ref: list[str] = []
    for component_instance_name, cuboid_reader in ctx.cuboid_readers.items():
        try:
            observations = list(cuboid_reader.get_observations())
        except Exception as e:
            log.debug(
                "check_cuboid_timestamps_in_interval: failed reading %s: %s",
                component_instance_name,
                e,
            )
            continue
        for obs in observations:
            if obs.timestamp_us not in ctx.interval:
                bad_ts.append(f"{component_instance_name}:{obs.track_id}@{obs.timestamp_us}")
                if len(bad_ts) >= 10:
                    break
            if obs.reference_frame_timestamp_us not in ctx.interval:
                bad_ref.append(f"{component_instance_name}:{obs.track_id}@ref={obs.reference_frame_timestamp_us}")
                if len(bad_ref) >= 10:
                    break
        if len(bad_ts) >= 10 and len(bad_ref) >= 10:
            break
    _check(
        "cuboid observation timestamps in sequence interval",
        not bad_ts,
        ctx.errors,
        f"cuboid timestamp_us outside interval: {'; '.join(bad_ts[:5])}"
        + (f" (and {len(bad_ts) - 5} more)" if len(bad_ts) > 5 else ""),
    )
    _check(
        "cuboid reference_frame timestamps in sequence interval",
        not bad_ref,
        ctx.errors,
        f"cuboid reference_frame_timestamp_us outside interval: {'; '.join(bad_ref[:5])}"
        + (f" (and {len(bad_ref) - 5} more)" if len(bad_ref) > 5 else ""),
    )


def _label_source(obs: object) -> str:
    """Return an observation's label source as ``name@version``, or ``name@any`` when unversioned."""
    source = getattr(obs, "source", None)
    version = getattr(obs, "source_version", None)
    return f"{getattr(source, 'name', source)}@{'any' if version is None else version}"


def _observations_by_component(ctx: NcoreV4Context, caller: str) -> dict[str, list]:
    """Read every cuboid component's observations, skipping unreadable components."""
    observations: dict[str, list] = {}
    for component_instance_name, cuboid_reader in ctx.cuboid_readers.items():
        try:
            observations[component_instance_name] = list(cuboid_reader.get_observations())
        except Exception as e:
            log.debug("%s: failed reading %s: %s", caller, component_instance_name, e)
    return observations


def _preview(items: list[str]) -> str:
    preview = ", ".join(items[:5])
    if len(items) > 5:
        preview += f" (and {len(items) - 5} more)"
    return preview


def check_cuboid_duplicate_observation_timestamps(ctx: NcoreV4Context) -> None:
    """Fail when one label source repeats a track observation timestamp.

    NuRec consolidates observations into one trajectory per ``track_id`` and
    silently drops repeated timestamps after label-source filtering. Rejecting
    duplicates within one source prevents output from depending on which
    observation NuRec sees first.

    Args:
        ctx: Open NCore V4 sequence context.
    """
    if not ctx.cuboid_readers:
        return

    duplicates: list[str] = []
    for component, observations in _observations_by_component(
        ctx, "check_cuboid_duplicate_observation_timestamps"
    ).items():
        seen: set[tuple[str, int, str]] = set()
        for obs in observations:
            key = (str(obs.track_id), int(obs.timestamp_us), _label_source(obs))
            if key in seen:
                duplicates.append(f"{component}:{key[0]}@{key[1]} ({key[2]})")
            else:
                seen.add(key)

    _check(
        "cuboid track timestamps unique within each label source",
        not duplicates,
        ctx.errors,
        f"duplicate cuboid (track_id, timestamp_us, label source) observations: {_preview(duplicates)}. "
        "Store one source observation at each track timestamp; NuRec otherwise "
        "silently drops repeated timestamps after label-source filtering.",
    )


def warn_cuboid_cross_source_timestamp_collisions(ctx: NcoreV4Context) -> None:
    """Warn when label sources or versions share a track id and timestamp.

    NuRec keys tracks by ``track_id`` alone once label sources are filtered,
    and a selector without a version admits every version of that source. When
    the colliding sources are selected together NuRec keeps one observation
    and silently drops the other; selected apart, they do not interact.

    Args:
        ctx: Open NCore V4 sequence context.
    """
    if not ctx.cuboid_readers:
        return

    collisions: list[str] = []
    for component, observations in _observations_by_component(
        ctx, "warn_cuboid_cross_source_timestamp_collisions"
    ).items():
        sources: dict[tuple[str, int], set[str]] = {}
        for obs in observations:
            sources.setdefault((str(obs.track_id), int(obs.timestamp_us)), set()).add(_label_source(obs))
        collisions.extend(
            f"{component}:{track_id}@{ts} ({', '.join(sorted(labels))})"
            for (track_id, ts), labels in sorted(sources.items())
            if len(labels) > 1
        )

    if collisions:
        _warn(
            f"{len(collisions)} cuboid (track_id, timestamp_us) pair(s) appear under more than one "
            f"label source or version: {_preview(collisions)}. Selecting those sources together "
            "makes NuRec drop all but one observation at each shared timestamp."
        )


def warn_cuboid_tracks_below_two_observations(ctx: NcoreV4Context) -> None:
    """Warn about label-source tracks with fewer than two observations.

    NCore accepts a single-observation track, but NuRec needs at least two
    distinct timestamps to build a trajectory and skips the rest, so the
    object is baked into the static scene. Counts are per component and exact
    label source; NuRec merges the label sources it is configured to select,
    so such a track is omitted when its source is selected alone but may
    survive when another selected source adds a distinct timestamp.

    Args:
        ctx: Open NCore V4 sequence context.
    """
    if not ctx.cuboid_readers:
        return

    short: list[str] = []
    for component, observations in _observations_by_component(ctx, "warn_cuboid_tracks_below_two_observations").items():
        counts: dict[tuple[str, str], int] = {}
        for obs in observations:
            key = (str(obs.track_id), _label_source(obs))
            counts[key] = counts.get(key, 0) + 1
        short.extend(
            f"{component}:{track_id} ({label}, {count} observation)"
            for (track_id, label), count in sorted(counts.items())
            if count < 2
        )

    if short:
        _warn(
            f"{len(short)} cuboid track(s) have fewer than two observations in their label source; NuRec "
            f"omits them when that source is selected alone: {_preview(short)}"
        )


def check_cuboid_pose_gap(ctx: NcoreV4Context) -> None:
    """Fail when a cuboid reference timestamp lacks a nearby pose waypoint.

    Placing a cuboid in the world evaluates the ``reference_frame_id`` ->
    ``world`` pose at ``reference_frame_timestamp_us``.  From ``"rig"`` and
    from any sensor frame that path crosses the dynamic ``rig`` -> ``world``
    edge and is interpolated there, so a trajectory with no waypoint near that
    timestamp yields a pose differing from the one the labels were defined
    relative to, offsetting the object by roughly vehicle speed x time gap.
    This makes dynamic objects blurry in renders.  A sensor frame's static
    sensor -> rig pose prefixes that traversal rather than replacing it, so
    sensor-anchored cuboids carry the same requirement as rig-anchored ones.

    Cuboids anchored in ``"world"`` or ``"world_global"`` are exempt: they
    reach the world frame through an identity or a static edge, leaving their
    placement independent of the reference timestamp.
    """
    if not (ctx.cuboid_readers and ctx.poses_readers):
        return

    pose_ts: np.ndarray | None = None
    for pose_reader in ctx.poses_readers.values():
        try:
            _, pose_ts = pose_reader.get_dynamic_pose("rig", "world")
            break
        except KeyError:
            continue
    if pose_ts is None or pose_ts.size == 0:
        return

    pose_ts_sorted = np.sort(pose_ts.astype(np.int64))

    ref_timestamps: set[int] = set()
    for cuboid_reader in ctx.cuboid_readers.values():
        try:
            for obs in cuboid_reader.get_observations():
                ref_frame = getattr(obs, "reference_frame_id", None)
                if ref_frame is None or ref_frame in WORLD_ANCHORED_CUBOID_FRAMES:
                    continue
                ref_ts = getattr(obs, "reference_frame_timestamp_us", None)
                if ref_ts is not None:
                    ref_timestamps.add(int(ref_ts))
        except Exception as e:
            log.debug("check_cuboid_pose_gap: failed reading observations: %s", e)
            continue

    if not ref_timestamps:
        return

    max_gap_us = 0
    worst_ts = 0
    for ts in ref_timestamps:
        idx = nearest_idx(pose_ts_sorted, ts)
        gap = abs(int(pose_ts_sorted[idx]) - ts)
        if gap > max_gap_us:
            max_gap_us = gap
            worst_ts = ts

    _check(
        f"cuboid reference timestamp pose gap <= {CUBOID_POSE_GAP_MAX_US / 1e3:.1f} ms "
        f"(max {max_gap_us / 1e3:.1f} ms across {len(ref_timestamps)} unique timestamps)",
        max_gap_us <= CUBOID_POSE_GAP_MAX_US,
        ctx.errors,
        f"cuboid reference timestamp pose gap: {len(ref_timestamps)} unique "
        f"pose-dependent reference timestamps, max gap {max_gap_us / 1e3:.1f} ms "
        f"(at t={worst_ts}). Labels defined relative to a frame-level ego pose "
        f"require that pose in the trajectory; interpolation causes spatial "
        f"offset on dynamic objects.",
    )


def check_cuboid_lidar_overlap(ctx: NcoreV4Context, *, lidar_pts_rig: dict[str, dict[int, np.ndarray]]) -> None:
    """Require that a minimum fraction of near-range cuboids contain LiDAR points.

    For each cuboid observation whose centroid is within
    ``CUBOID_NEAR_RANGE_M`` of the rig origin, counts how many LiDAR
    points from the closest-in-time frame fall inside the oriented
    bounding box.  If the hit rate (fraction with >= 1 point) is below
    ``MIN_NEAR_RANGE_CUBOID_HIT_RATE``, this is a hard failure -- it
    typically indicates a ``T_lidar_rig`` or model-frame error.

    Skipped when fewer than ``MIN_NEAR_RANGE_CUBOIDS_FOR_CHECK``
    near-range cuboids are available (sample too small).

    Cuboid observations are capped at 200 for runtime.

    Args:
        ctx: Open NCore V4 sequence context.
        lidar_pts_rig: Rig-frame point clouds keyed by LiDAR id then frame end
            timestamp, as returned by :func:`check_lidar_frame_data`.
    """
    if not (ctx.cuboid_readers and lidar_pts_rig and ctx.poses_readers):
        return

    T_rig_worlds: np.ndarray | None = None
    rig_ts: np.ndarray | None = None
    for pose_reader in ctx.poses_readers.values():
        try:
            T_rig_worlds, rig_ts = pose_reader.get_dynamic_pose("rig", "world")
            break
        except KeyError:
            continue

    observations: list = []
    for cuboid_reader in ctx.cuboid_readers.values():
        try:
            observations.extend(cuboid_reader.get_observations())
        except Exception as e:
            log.debug("check_cuboid_lidar_overlap: failed reading observations: %s", e)
            continue

    coverage = LidarCoverageIndex(ctx, lidar_pts_rig)
    n_near = 0
    n_hit = 0

    for obs in sample_evenly(
        observations,
        MAX_CUBOID_OBSERVATIONS,
        key=lambda observation: int(getattr(observation, "timestamp_us", 0)),
    ):
        obs_ts = getattr(obs, "timestamp_us", None)
        if obs_ts is None:
            continue

        bbox_rig = cuboid_bbox_to_rig(obs, ctx, T_rig_worlds, rig_ts)
        if bbox_rig is None:
            continue

        if float(np.linalg.norm(bbox_rig[:3])) > CUBOID_NEAR_RANGE_M:
            continue

        # Count only cuboids the LiDAR could have seen. An observation with no
        # sensor covering its instant can never contribute a hit, so counting it
        # here would depress the hit rate toward a failure that says nothing
        # about T_lidar_rig.
        pts_rig = coverage.points_near(int(obs_ts))
        if pts_rig is None:
            continue

        n_near += 1

        try:
            inside = is_within_3d_bboxes(pts_rig.astype(np.float64), bbox_rig.reshape(1, 9))[:, 0]
            if inside.sum() > 0:
                n_hit += 1
        except Exception as e:
            log.debug("check_cuboid_lidar_overlap: is_within_3d_bboxes failed: %s", e)
            continue

    if n_near < MIN_NEAR_RANGE_CUBOIDS_FOR_CHECK:
        _warn(
            f"only {n_near} near-range cuboids (< {CUBOID_NEAR_RANGE_M} m); "
            f"need >= {MIN_NEAR_RANGE_CUBOIDS_FOR_CHECK} for overlap check"
        )
        return

    hit_rate = n_hit / n_near
    _check(
        f"near-range cuboid LiDAR overlap ({n_hit}/{n_near} = {hit_rate:.0%}; "
        f"threshold >= {MIN_NEAR_RANGE_CUBOID_HIT_RATE:.0%})",
        hit_rate >= MIN_NEAR_RANGE_CUBOID_HIT_RATE,
        ctx.errors,
        f"only {hit_rate:.0%} of {n_near} near-range cuboids (< {CUBOID_NEAR_RANGE_M} m) "
        f"contain LiDAR points (minimum: {MIN_NEAR_RANGE_CUBOID_HIT_RATE:.0%}). "
        f"This usually indicates a T_lidar_rig or LiDAR model-frame rotation error.",
    )


def check_cuboid_obb_inlier_ratio(ctx: NcoreV4Context, *, lidar_pts_rig: dict[str, dict[int, np.ndarray]]) -> None:
    """Require a minimum fraction of in-AABB LiDAR points to fall inside the OBB.

    For each cuboid observation, compute the axis-aligned bounding box
    (AABB) of the OBB in rig frame, select LiDAR points inside that
    AABB, and require that at least ``MIN_CUBOID_OBB_INLIER_RATIO`` of
    those also fall inside the actual OBB.

    Catches geometry bugs that :func:`check_cuboid_lidar_overlap` misses
    (its >=1-point-hit threshold is satisfied even by heavily-offset
    OBBs when the sample is generous):

    * dim.x / dim.y swap without a compensating ``rot`` change
    * wrong rotation axis
    * wrong centroid convention
    * wrong reference frame
    * wrong dim magnitude

    Dataset-agnostic: no class-label filter and no geometric filter.
    Cubic cuboids trivially pass because their AABB equals their OBB at
    rotation 0 -- correct behavior, not a false positive.
    """
    if not (ctx.cuboid_readers and lidar_pts_rig and ctx.poses_readers):
        return

    T_rig_worlds: np.ndarray | None = None
    rig_ts: np.ndarray | None = None
    for pose_reader in ctx.poses_readers.values():
        try:
            T_rig_worlds, rig_ts = pose_reader.get_dynamic_pose("rig", "world")
            break
        except KeyError:
            continue

    observations: list = []
    for cuboid_reader in ctx.cuboid_readers.values():
        try:
            observations.extend(cuboid_reader.get_observations())
        except Exception as e:
            log.debug("check_cuboid_obb_inlier_ratio: failed reading observations: %s", e)
            continue

    # OBB corner offsets (unit cube), [8, 3].
    signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=np.float64,
    )

    coverage = LidarCoverageIndex(ctx, lidar_pts_rig)
    ratios: list[float] = []
    for obs in sample_evenly(
        observations,
        MAX_CUBOID_OBSERVATIONS,
        key=lambda observation: int(getattr(observation, "timestamp_us", 0)),
    ):
        obs_ts = getattr(obs, "timestamp_us", None)
        if obs_ts is None:
            continue

        bbox_rig = cuboid_bbox_to_rig(obs, ctx, T_rig_worlds, rig_ts)
        if bbox_rig is None:
            continue

        centroid = bbox_rig[:3].astype(np.float64)
        extents = bbox_rig[3:6].astype(np.float64)  # (L, W, H)
        eulers = bbox_rig[6:9].astype(np.float64)
        if not np.all(np.isfinite(extents)) or not np.all(extents > 0):
            continue

        R_obb = Rotation.from_euler("xyz", eulers).as_matrix()
        corners_local = signs * (extents / 2.0)
        corners_rig = corners_local @ R_obb.T + centroid
        aabb_min = corners_rig.min(axis=0)
        aabb_max = corners_rig.max(axis=0)

        pts_rig = coverage.points_near(int(obs_ts))
        if pts_rig is None or pts_rig.size == 0:
            continue

        in_aabb = np.all((pts_rig >= aabb_min) & (pts_rig <= aabb_max), axis=1)
        n_aabb = int(in_aabb.sum())
        if n_aabb == 0:
            continue

        try:
            in_obb = is_within_3d_bboxes(pts_rig[in_aabb].astype(np.float64), bbox_rig.reshape(1, 9))[:, 0]
        except Exception as e:
            log.debug("check_cuboid_obb_inlier_ratio: is_within_3d_bboxes failed: %s", e)
            continue

        ratios.append(float(in_obb.sum()) / float(n_aabb))

    if len(ratios) < MIN_CUBOIDS_FOR_OBB_CHECK:
        return

    median_ratio = float(np.median(ratios))
    _check(
        f"cuboid OBB inlier ratio (median {median_ratio:.0%} across "
        f"{len(ratios)} cuboids; threshold >= {MIN_CUBOID_OBB_INLIER_RATIO:.0%})",
        median_ratio >= MIN_CUBOID_OBB_INLIER_RATIO,
        ctx.errors,
        f"median OBB inlier ratio {median_ratio:.0%} across {len(ratios)} "
        f"cuboids is below the {MIN_CUBOID_OBB_INLIER_RATIO:.0%} threshold. "
        f"LiDAR points inside each cuboid's AABB are largely NOT inside the "
        f"actual OBB, which indicates a rotation / dim-axis-ordering / "
        f"centroid / reference-frame bug (e.g. dim.x/dim.y swap without a "
        f"compensating change to BBox3.rot).",
    )


def check_cuboid_rig_proximity(ctx: NcoreV4Context) -> None:
    """Hard-error when the p95 of cuboid-centroid rig distance exceeds
    :data:`MAX_CUBOID_RIG_DISTANCE_P95_M`.

    Detects world-anchored cuboids left in the original global frame after
    the rig trajectory was re-referenced -- centroids then appear far from
    the rig.  Uses ``cuboid_bbox_to_rig`` (handles ``rig`` / sensor id /
    ``world`` / ``world_global``).  Skipped below
    :data:`MIN_CUBOID_RIG_PROXIMITY_OBSERVATIONS` observations.
    """
    if not (ctx.cuboid_readers and ctx.poses_readers):
        return

    T_rig_worlds: np.ndarray | None = None
    rig_ts: np.ndarray | None = None
    for pose_reader in ctx.poses_readers.values():
        try:
            T_rig_worlds, rig_ts = pose_reader.get_dynamic_pose("rig", "world")
            break
        except KeyError:
            continue

    distances: list[float] = []
    for cuboid_reader in ctx.cuboid_readers.values():
        try:
            observations = list(cuboid_reader.get_observations())
        except Exception as e:
            log.debug("check_cuboid_rig_proximity: failed reading observations: %s", e)
            continue
        for obs in observations:
            bbox_rig = cuboid_bbox_to_rig(obs, ctx, T_rig_worlds, rig_ts)
            if bbox_rig is None:
                continue
            centroid = bbox_rig[:3].astype(np.float64)
            if not np.all(np.isfinite(centroid)):
                continue
            distances.append(float(np.linalg.norm(centroid)))

    if len(distances) < MIN_CUBOID_RIG_PROXIMITY_OBSERVATIONS:
        return

    arr = np.array(distances, dtype=np.float64)
    p95 = float(np.percentile(arr, 95))
    p50 = float(np.percentile(arr, 50))
    _check(
        f"cuboid rig-frame centroid 95th-percentile distance "
        f"(p95={p95:.1f} m, median={p50:.1f} m across {arr.size} observations; "
        f"threshold <= {MAX_CUBOID_RIG_DISTANCE_P95_M:.0f} m)",
        p95 <= MAX_CUBOID_RIG_DISTANCE_P95_M,
        ctx.errors,
        f"cuboid rig-frame centroid p95 is {p95:.1f} m, above "
        f"{MAX_CUBOID_RIG_DISTANCE_P95_M:.0f} m across {arr.size} "
        f"observations. Cuboids are likely in a different world frame "
        f"than the re-referenced rig trajectory -- either pre-transform "
        f"centroids by se3_inverse(T_world_world_global) or switch "
        f"reference_frame_id to 'world_global'.",
    )


def warn_cuboid_ground_clearance(ctx: NcoreV4Context, *, lidar_pts_rig: dict[str, dict[int, np.ndarray]]) -> None:
    """Warn when near-range cuboids appear to float above their LiDAR returns.

    For each near-range cuboid, constructs a probe volume with the same
    oriented XY footprint extending ``CUBOID_GROUND_CLEARANCE_PROBE_M``
    below the cuboid bottom face.  Counts LiDAR points *inside* the cuboid
    versus *below* it.  If the median below-ratio across cuboids exceeds
    ``CUBOID_FLOATING_MEDIAN_RATIO``, the cuboids are likely shifted upward
    (e.g. treating a centroid *z* as bottom-center and adding ``height/2``).

    Skipped when fewer than ``MIN_NEAR_RANGE_CUBOIDS_FOR_CHECK``
    near-range cuboids with LiDAR coverage are available.
    """
    if not (ctx.cuboid_readers and lidar_pts_rig and ctx.poses_readers):
        return

    T_rig_worlds: np.ndarray | None = None
    rig_ts: np.ndarray | None = None
    for pose_reader in ctx.poses_readers.values():
        try:
            T_rig_worlds, rig_ts = pose_reader.get_dynamic_pose("rig", "world")
            break
        except KeyError:
            continue

    observations: list = []
    for cuboid_reader in ctx.cuboid_readers.values():
        try:
            observations.extend(cuboid_reader.get_observations())
        except Exception as e:
            log.debug("warn_cuboid_ground_clearance: failed reading observations: %s", e)
            continue

    coverage = LidarCoverageIndex(ctx, lidar_pts_rig)
    below_ratios: list[float] = []

    for obs in sample_evenly(
        observations,
        MAX_CUBOID_OBSERVATIONS,
        key=lambda observation: int(getattr(observation, "timestamp_us", 0)),
    ):
        obs_ts = getattr(obs, "timestamp_us", None)
        if obs_ts is None:
            continue

        bbox_rig = cuboid_bbox_to_rig(obs, ctx, T_rig_worlds, rig_ts)
        if bbox_rig is None:
            continue

        if float(np.linalg.norm(bbox_rig[:3])) > CUBOID_NEAR_RANGE_M:
            continue

        pts_rig = coverage.points_near(int(obs_ts))
        if pts_rig is None:
            continue

        cz = float(bbox_rig[2])
        H = float(bbox_rig[5])
        probe_h = CUBOID_GROUND_CLEARANCE_PROBE_M
        below_cz = cz - H / 2 - probe_h / 2
        below_bbox = bbox_rig.copy()
        below_bbox[2] = below_cz
        below_bbox[5] = probe_h

        try:
            check_bboxes = np.stack([bbox_rig, below_bbox]).reshape(2, 9)
            inside = is_within_3d_bboxes(pts_rig.astype(np.float64), check_bboxes)
            n_inside = int(inside[:, 0].sum())
            n_below = int(inside[:, 1].sum())
            total = n_inside + n_below
            if total > 0:
                below_ratios.append(n_below / total)
        except Exception as e:
            log.debug("warn_cuboid_ground_clearance: is_within_3d_bboxes failed: %s", e)
            continue

    if len(below_ratios) < MIN_NEAR_RANGE_CUBOIDS_FOR_CHECK:
        return

    median_ratio = float(np.median(below_ratios))
    if median_ratio > CUBOID_FLOATING_MEDIAN_RATIO:
        _warn(
            f"cuboid ground clearance: median below-cuboid point ratio "
            f"{median_ratio:.0%} across {len(below_ratios)} near-range cuboids "
            f"(threshold {CUBOID_FLOATING_MEDIAN_RATIO:.0%}) — cuboids may be "
            f"shifted upward (e.g. bottom-center vs centroid z convention)"
        )
    else:
        _check(
            f"cuboid ground clearance OK (median below-ratio {median_ratio:.0%}, {len(below_ratios)} cuboids)",
            True,
            ctx.errors,
        )
