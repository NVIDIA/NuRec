# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-sensor consistency checks."""

from __future__ import annotations

import logging

import numpy as np

from ncore_validation.utils import NcoreV4Context
from ncore_validation.validate_cuboids import nearest_idx
from ncore_validation.validate_helpers import (
    LIDAR_CAMERA_POSITIVE_OFFSET_RATIO,
    LIDAR_CAMERA_SWEEP_OFFSET_RATIO,
    MAX_POSE_GAP_SWEEP_RATIO,
    MIN_CENTROID_DRIFT_EGO_RATIO,
    MIN_EGO_DISPLACEMENT_FOR_CLOCK_CHECK_M,
    MIN_LIDAR_CENTROID_DRIFT_M,
    SWEEP_CENTER_OFFSET_RATIO,
    _check,
    _warn,
)


log = logging.getLogger("validate")


def check_cameras_inside_lidar_bbox(ctx: NcoreV4Context, *, lidar_pts_rig: dict[str, dict[int, np.ndarray]]) -> None:
    """Check that every camera's static mounting position falls inside the LiDAR point cloud.

    Uses pre-computed rig-frame LiDAR points (from :func:`check_lidar_frame_data`)
    to build a padded axis-aligned bounding box.  Each camera's fixed
    sensor-to-rig translation is tested against that box.  A camera outside
    the box usually indicates a sign error or convention mismatch in the
    extrinsic transform.

    Args:
        ctx: Open NCore V4 sequence context.
        lidar_pts_rig: Rig-frame point clouds keyed by LiDAR id then frame end
            timestamp, as returned by :func:`check_lidar_frame_data`.
    """
    if not (ctx.camera_readers and ctx.poses_readers and lidar_pts_rig):
        return

    all_pts = np.concatenate([pts for per_sensor in lidar_pts_rig.values() for pts in per_sensor.values()], axis=0)
    bbox_min = all_pts.min(axis=0)
    bbox_max = all_pts.max(axis=0)

    # Pad the AABB to tolerate typical sensor mounting offsets (e.g.,
    # cameras on a rooftop rack above the LiDAR scan plane).  Padding
    # is applied in all directions except negative-Z: keeping the Z
    # floor tight catches cameras that are incorrectly flipped in Z
    # (sign error in the extrinsic transform).
    _BBOX_PAD_M = 2.0
    bbox_min[:2] -= _BBOX_PAD_M  # pad X, Y lower bounds
    bbox_max += _BBOX_PAD_M  # pad X, Y, Z upper bounds

    cameras_outside_bbox: list[str] = []
    for camera_id in ctx.camera_readers:
        for pose_reader in ctx.poses_readers.values():
            try:
                cam_to_rig = pose_reader.get_static_pose(camera_id, "rig")
                cam_pos = cam_to_rig[:3, 3]
                if np.any(cam_pos < bbox_min) or np.any(cam_pos > bbox_max):
                    cameras_outside_bbox.append(
                        f"{camera_id} (pos=[{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}], "
                        f"bbox=[{bbox_min[0]:.1f}..{bbox_max[0]:.1f}, "
                        f"{bbox_min[1]:.1f}..{bbox_max[1]:.1f}, "
                        f"{bbox_min[2]:.1f}..{bbox_max[2]:.1f}])"
                    )
                break
            except KeyError:
                continue
    _check(
        "camera positions within LiDAR point cloud bounds",
        len(cameras_outside_bbox) == 0,
        ctx.errors,
        f"cameras outside LiDAR bounding box in rig frame: {'; '.join(cameras_outside_bbox)}. "
        "This may indicate a coordinate frame convention mismatch in sensor\u2192rig transforms.",
    )


def warn_camera_follows_ego_if_moving(ctx: NcoreV4Context) -> None:
    """Warn when ego moves but camera world displacement stays near zero (clock sync)."""
    if not (ctx.camera_readers and ctx.poses_readers):
        return
    T_rig_worlds = None
    rig_ts = None
    for pose_reader in ctx.poses_readers.values():
        try:
            T_rig_worlds, rig_ts = pose_reader.get_dynamic_pose("rig", "world")
            break
        except KeyError:
            continue

    if T_rig_worlds is not None and rig_ts is not None and rig_ts.size >= 2:
        ego_displacement = float(np.linalg.norm(T_rig_worlds[-1, :3, 3] - T_rig_worlds[0, :3, 3]))

        if ego_displacement > MIN_EGO_DISPLACEMENT_FOR_CLOCK_CHECK_M:
            for camera_id, cam_reader in ctx.camera_readers.items():
                cam_ts = cam_reader.frames_timestamps_us
                if cam_ts.size == 0:
                    continue

                cam_first_ts = int(cam_ts[0, 0])
                cam_last_ts = int(cam_ts[-1, 0])

                if cam_first_ts == cam_last_ts:
                    _warn(
                        f"{camera_id} has all frames at a single timestamp \u2014 "
                        f"will not follow ego trajectory ({ego_displacement:.1f} m displacement). "
                        "Possible clock sync issue."
                    )
                    continue

                idx_first = nearest_idx(rig_ts, cam_first_ts)
                idx_last = nearest_idx(rig_ts, cam_last_ts)

                cam_to_rig = None
                for pose_reader in ctx.poses_readers.values():
                    try:
                        cam_to_rig = pose_reader.get_static_pose(camera_id, "rig")
                        break
                    except KeyError:
                        continue
                if cam_to_rig is None:
                    continue

                cam_world_first = (T_rig_worlds[idx_first] @ cam_to_rig)[:3, 3]
                cam_world_last = (T_rig_worlds[idx_last] @ cam_to_rig)[:3, 3]
                cam_displacement = float(np.linalg.norm(cam_world_last - cam_world_first))

                if cam_displacement < 0.1:
                    _warn(
                        f"{camera_id} displacement ({cam_displacement:.2f} m) does not follow "
                        f"ego trajectory ({ego_displacement:.1f} m) \u2014 possible clock sync issue"
                    )


def check_camera_lidar_time_overlap(ctx: NcoreV4Context) -> None:
    """Require global camera and LiDAR frame time ranges to intersect."""
    if not (ctx.camera_readers and ctx.lidar_readers):
        return
    cam_starts = []
    cam_ends = []
    for cam_reader in ctx.camera_readers.values():
        ts = cam_reader.frames_timestamps_us
        if ts.size > 0:
            cam_starts.append(int(ts[:, 0].min()))
            cam_ends.append(int(ts[:, 1].max()))
    lidar_starts = []
    lidar_ends = []
    for lidar_reader in ctx.lidar_readers.values():
        ts = lidar_reader.frames_timestamps_us
        if ts.size > 0:
            lidar_starts.append(int(ts[:, 0].min()))
            lidar_ends.append(int(ts[:, 1].max()))

    if cam_starts and lidar_starts:
        cam_range = (min(cam_starts), max(cam_ends))
        lidar_range = (min(lidar_starts), max(lidar_ends))
        overlap = cam_range[0] <= lidar_range[1] and lidar_range[0] <= cam_range[1]
        _check(
            "camera and LiDAR timestamp ranges overlap",
            overlap,
            ctx.errors,
            f"camera range [{cam_range[0]}, {cam_range[1]}] and "
            f"LiDAR range [{lidar_range[0]}, {lidar_range[1]}] are disjoint",
        )


def warn_sequence_envelope_coverage(ctx: NcoreV4Context) -> None:
    """Warn when sensor envelopes don't span the sequence interval.

    All sensor envelope checks are non-blocking ``WARNING`` s.  NuRec
    restricts its working window to where rig poses exist and filters
    each sensor's frames to the overlapping subset -- sensors with
    fewer frames simply contribute fewer
    training samples rather than causing errors.  Per-sensor gaps
    are expected on multi-camera rigs with staggered
    (non-synchronized) triggers.

    **Cuboid** component coverage is advisory for a different reason:
    the NCore writer requires cuboid timestamps to fall *inside* the
    interval (enforced by :func:`check_cuboid_timestamps_in_interval`)
    but does **not** require the component's envelope to span it.
    Per-track filtering downstream operates on each track's own
    ``[min, max]`` envelope, so a narrower component envelope is a
    legitimate representation of "objects entered/left mid-sequence."

    Pose envelope coverage is enforced by the NCore writer's
    interval / endpoint assertions and is not re-checked here.
    """
    interval_start = int(ctx.interval.start)
    last_ts = int(ctx.interval.stop) - 1
    sensor_bad: list[str] = []
    cuboid_bad: list[str] = []

    def _format_gap(component_label: str, min_ts: int, max_ts: int) -> str | None:
        gap_start = min_ts - interval_start
        gap_end = last_ts - max_ts
        if gap_start <= 0 and gap_end <= 0:
            return None
        parts = []
        if gap_start > 0:
            parts.append(f"starts {gap_start} us after interval start")
        if gap_end > 0:
            parts.append(f"ends {gap_end} us before interval end")
        return (
            f"{component_label} envelope [{min_ts}, {max_ts}] does not contain "
            f"sequence interval [{interval_start}, {last_ts}]: " + ", ".join(parts)
        )

    for cam_id, cam_reader in ctx.camera_readers.items():
        ts = cam_reader.frames_timestamps_us
        if ts.size > 0:
            msg = _format_gap(f"camera '{cam_id}'", int(ts[:, 0].min()), int(ts[:, 1].max()))
            if msg is not None:
                sensor_bad.append(msg)

    for lidar_id, lidar_reader in ctx.lidar_readers.items():
        ts = lidar_reader.frames_timestamps_us
        if ts.size > 0:
            msg = _format_gap(f"lidar '{lidar_id}'", int(ts[:, 0].min()), int(ts[:, 1].max()))
            if msg is not None:
                sensor_bad.append(msg)

    for component_instance_name, cuboid_reader in ctx.cuboid_readers.items():
        try:
            obs_list = list(cuboid_reader.get_observations())
        except (OSError, ValueError, KeyError) as e:
            log.debug(f"Could not read cuboid observations from {component_instance_name}: {e}")
            continue
        if not obs_list:
            continue
        ts_min = int(min(o.timestamp_us for o in obs_list))
        ts_max = int(max(o.timestamp_us for o in obs_list))
        msg = _format_gap(f"cuboids '{component_instance_name}'", ts_min, ts_max)
        if msg is not None:
            cuboid_bad.append(msg)

    if sensor_bad:
        _warn(
            "sensor envelope(s) narrower than sequence interval: "
            + "; ".join(sensor_bad[:5])
            + (f" (and {len(sensor_bad) - 5} more)" if len(sensor_bad) > 5 else "")
            + ". NuRec restricts its working window to where rig poses "
            "exist and filters per sensor, so partial coverage "
            "is handled gracefully. Verify the gap reflects real trigger "
            "offsets rather than dropped frames."
        )

    if cuboid_bad:
        _warn(
            "cuboid component envelope narrower than sequence interval: "
            + "; ".join(cuboid_bad[:5])
            + (f" (and {len(cuboid_bad) - 5} more)" if len(cuboid_bad) > 5 else "")
            + ". NCore only requires cuboid timestamps to fall inside the "
            "interval, not to span it; per-track filtering downstream handles "
            "tracks that come and go mid-sequence. Confirm this gap reflects "
            "real label coverage rather than a converter bug."
        )


def warn_sparse_pose_trajectory(ctx: NcoreV4Context) -> None:
    """Warn when rig->world pose waypoint gaps exceed the LiDAR sweep duration.

    Sparse trajectories cause staircase interpolation during NuRec motion
    compensation, producing rowing (horizontal stripe) artifacts.
    """
    if not (ctx.poses_readers and ctx.lidar_readers):
        return

    pose_ts = None
    for pose_reader in ctx.poses_readers.values():
        try:
            _, pose_ts = pose_reader.get_dynamic_pose("rig", "world")
            break
        except KeyError:
            continue
    if pose_ts is None or pose_ts.size < 2:
        return

    all_sweep_durations: list[float] = []
    for lidar_reader in ctx.lidar_readers.values():
        ts = lidar_reader.frames_timestamps_us
        if ts.size == 0:
            continue
        durations = (ts[:, 1] - ts[:, 0]).astype(np.float64)
        all_sweep_durations.extend(durations[durations > 0].tolist())
    if not all_sweep_durations:
        return

    median_sweep_us = float(np.median(all_sweep_durations))
    pose_gaps = np.diff(pose_ts.astype(np.float64))
    max_gap_us = float(pose_gaps.max())
    median_gap_us = float(np.median(pose_gaps))

    if max_gap_us >= median_sweep_us * MAX_POSE_GAP_SWEEP_RATIO:
        _warn(
            f"Sparse pose trajectory: {len(pose_ts)} waypoints, "
            f"max gap {max_gap_us / 1e3:.1f} ms, median gap {median_gap_us / 1e3:.1f} ms, "
            f"but median LiDAR sweep is {median_sweep_us / 1e3:.1f} ms. "
            f"NuRec interpolates at every per-ray timestamp; gaps >= sweep duration "
            f"cause rowing artifacts. Use the densest pose source available "
            f"(e.g. per-camera-image poses from all cameras)."
        )


def warn_lidar_centroid_drift(ctx: NcoreV4Context) -> None:
    """Warn when LiDAR point cloud centroids drift with ego motion.

    In true sensor-frame data, the centroid of ``direction * distance`` stays
    near the sensor origin across all frames.  If world-frame or ego-compensated
    data was stored as sensor-frame, the centroid tracks the vehicle's world
    position and drifts by roughly the ego displacement.
    """
    if not (ctx.lidar_readers and ctx.poses_readers):
        return

    T_rig_worlds = None
    for pose_reader in ctx.poses_readers.values():
        try:
            T_rig_worlds, _ = pose_reader.get_dynamic_pose("rig", "world")
            break
        except KeyError:
            continue
    if T_rig_worlds is None or len(T_rig_worlds) < 2:
        return

    ego_displacement = float(np.linalg.norm(T_rig_worlds[-1, :3, 3] - T_rig_worlds[0, :3, 3]))
    if ego_displacement < MIN_LIDAR_CENTROID_DRIFT_M:
        return

    for lidar_id, lidar_reader in ctx.lidar_readers.items():
        ts = lidar_reader.frames_timestamps_us
        if ts.shape[0] < 2:
            continue

        sample_indices = [0, ts.shape[0] - 1]
        centroids = []
        for idx in sample_indices:
            end_ts = int(ts[idx, 1])
            try:
                direction = lidar_reader.get_frame_ray_bundle_data(end_ts, "direction")
                distance = lidar_reader.get_frame_ray_bundle_return_data(end_ts, "distance_m", return_index=0)
            except KeyError:
                break
            if direction is None or distance is None or direction.shape[0] == 0:
                break
            xyz_sensor = direction * distance[:, np.newaxis]
            valid = np.isfinite(xyz_sensor).all(axis=1) & (distance > 0)
            if not valid.any():
                break
            centroids.append(xyz_sensor[valid].mean(axis=0))

        if len(centroids) == 2:
            drift = float(np.linalg.norm(centroids[1] - centroids[0]))
            ratio = drift / ego_displacement
            if drift > MIN_LIDAR_CENTROID_DRIFT_M and ratio > MIN_CENTROID_DRIFT_EGO_RATIO:
                _warn(
                    f"{lidar_id}: sensor-frame point cloud centroid drifts {drift:.1f} m "
                    f"across the sequence (ego displacement {ego_displacement:.1f} m, "
                    f"ratio {ratio:.2f}). This suggests world-frame or ego-compensated "
                    f"data was stored as sensor-frame. NCore V4 requires uncompensated "
                    f"sensor-frame directions; use MotionCompensator.motion_decompensate_points "
                    f"to recover sensor-frame points before deriving directions."
                )


def warn_sweep_timestamps_centered(ctx: NcoreV4Context) -> None:
    """Warn when LiDAR sweep start timestamps appear centered rather than at sweep start.

    Compares each LiDAR frame's start timestamp against the nearest dynamic
    pose waypoint.  If the median offset is a large fraction of the sweep
    duration, the converter may be centering sweeps at the frame timestamp
    (``frame_ts - half_sweep`` to ``frame_ts + half_sweep``) instead of
    using it as the start (``frame_ts`` to ``frame_ts + sweep_duration``).

    Note: this heuristic assumes pose waypoints are roughly aligned with
    LiDAR sweep boundaries.  Datasets where poses originate from a
    different source (e.g. per-camera-image poses only) may trigger this
    warning spuriously because the pose sample times have no relationship
    to LiDAR sweep starts.
    """
    if not (ctx.lidar_readers and ctx.poses_readers):
        return

    pose_ts = None
    for pose_reader in ctx.poses_readers.values():
        try:
            _, pose_ts = pose_reader.get_dynamic_pose("rig", "world")
            break
        except KeyError:
            continue
    if pose_ts is None or pose_ts.size < 2:
        return

    for lidar_id, lidar_reader in ctx.lidar_readers.items():
        ts = lidar_reader.frames_timestamps_us
        if ts.shape[0] < 3:
            continue

        sweep_durations = (ts[:, 1] - ts[:, 0]).astype(np.float64)
        positive_durations = sweep_durations[sweep_durations > 0]
        if positive_durations.size == 0:
            continue
        median_sweep = float(np.median(positive_durations))
        if median_sweep <= 0:
            continue

        offsets: list[float] = []
        for idx in range(ts.shape[0]):
            start_ts = int(ts[idx, 0])
            pidx = int(np.clip(np.searchsorted(pose_ts, start_ts), 0, len(pose_ts) - 1))
            nearest_pose_ts = int(pose_ts[pidx])
            if pidx > 0:
                alt = int(pose_ts[pidx - 1])
                if abs(start_ts - alt) < abs(start_ts - nearest_pose_ts):
                    nearest_pose_ts = alt
            offsets.append(abs(start_ts - nearest_pose_ts))

        median_offset = float(np.median(offsets))
        ratio = median_offset / median_sweep

        if ratio >= SWEEP_CENTER_OFFSET_RATIO:
            _warn(
                f"{lidar_id}: LiDAR sweep start timestamps are offset from the nearest "
                f"pose waypoint by a median of {median_offset / 1e3:.1f} ms "
                f"({ratio:.0%} of the {median_sweep / 1e3:.1f} ms sweep duration). "
                f"This may indicate sweeps are centered at the frame timestamp "
                f"instead of starting there. Per-point timestamps should span "
                f"[frame_ts, frame_ts + sweep_duration], not "
                f"[frame_ts - half_sweep, frame_ts + half_sweep]."
            )


def check_lidar_camera_sweep_alignment(ctx: NcoreV4Context) -> None:
    """Fail when LiDAR sweep midpoints are systematically offset from camera midpoints.

    Compares per-frame LiDAR and camera exposure midpoints, pairing frame
    ``i`` with frame ``i``.  Cameras are filtered by rate and temporal guards
    before comparison.  Two branches, both scaled by the sweep duration:

    * A **negative** median offset past ``LIDAR_CAMERA_SWEEP_OFFSET_RATIO``
      means the converter centered sweeps on the frame timestamp instead of
      starting them there.
    * A **positive** median offset past
      ``LIDAR_CAMERA_POSITIVE_OFFSET_RATIO`` is larger than a
      trigger-at-sweep-start setup explains, and points at either a camera
      timestamp misinterpretation or a constant index shift between the two
      streams' retained frames.
    """
    if not (ctx.lidar_readers and ctx.camera_readers):
        return

    cam_ts_arrays: list[np.ndarray] = []
    for cam_reader in ctx.camera_readers.values():
        ts = cam_reader.frames_timestamps_us
        if ts.size > 0:
            cam_ts_arrays.append(ts.astype(np.float64))
    if not cam_ts_arrays:
        return

    for lidar_id, lidar_reader in ctx.lidar_readers.items():
        ts = lidar_reader.frames_timestamps_us
        n_lidar = ts.shape[0]
        if n_lidar < 3:
            continue

        sweep_durations = (ts[:, 1] - ts[:, 0]).astype(np.float64)
        positive_durations = sweep_durations[sweep_durations > 0]
        if positive_durations.size == 0:
            continue
        median_sweep = float(np.median(positive_durations))
        if median_sweep <= 0:
            continue

        lidar_start_0 = float(ts[0, 0])

        # Filter cameras: frame count within 20% (or 3 frames) AND first
        # frame starts within one sweep duration of the LiDAR's first frame.
        # The rate guard avoids nonsense when sensors run at different Hz;
        # the temporal guard avoids nonsense when same-rate sensors have a
        # staggered start (frame 0 of camera != frame 0 of LiDAR).
        matched_cam_ts: list[np.ndarray] = []
        for c in cam_ts_arrays:
            if abs(c.shape[0] - n_lidar) > max(3, n_lidar * 0.2):
                continue
            if abs(float(c[0, 0]) - lidar_start_0) > median_sweep:
                continue
            matched_cam_ts.append(c)
        if not matched_cam_ts:
            continue

        lidar_mids = (ts[:, 0].astype(np.float64) + ts[:, 1].astype(np.float64)) / 2.0

        offsets: list[float] = []
        for i in range(n_lidar):
            cam_mids_this_frame: list[float] = []
            for cam_ts in matched_cam_ts:
                if i < cam_ts.shape[0]:
                    cam_mids_this_frame.append((cam_ts[i, 0] + cam_ts[i, 1]) / 2.0)
            if not cam_mids_this_frame:
                continue
            offsets.append(float(lidar_mids[i]) - float(np.median(cam_mids_this_frame)))

        if len(offsets) < 3:
            continue

        median_offset = float(np.median(offsets))
        ratio = abs(median_offset) / median_sweep

        if median_offset < 0 and ratio >= LIDAR_CAMERA_SWEEP_OFFSET_RATIO:
            _check(
                f"{lidar_id}: LiDAR-camera sweep midpoint alignment "
                f"(median offset {median_offset / 1e3:+.1f} ms, "
                f"{ratio:.0%} of {median_sweep / 1e3:.1f} ms sweep)",
                False,
                ctx.errors,
                f"{lidar_id}: LiDAR sweep midpoints are systematically "
                f"{abs(median_offset) / 1e3:.1f} ms EARLIER than camera exposure "
                f"midpoints ({ratio:.0%} of the {median_sweep / 1e3:.1f} ms "
                f"sweep duration). This indicates the converter treated the frame "
                f"timestamp as the sweep center rather than the sweep start, "
                f"shifting every per-ray timestamp by ~{abs(median_offset) / 1e3:.0f} ms. "
                f"Per-point timestamps should span [frame_ts, frame_ts + sweep_duration], "
                f"not [frame_ts - half_sweep, frame_ts + half_sweep].",
            )
        elif median_offset > 0 and ratio >= LIDAR_CAMERA_POSITIVE_OFFSET_RATIO:
            _check(
                f"{lidar_id}: LiDAR-camera sweep midpoint alignment "
                f"(median offset {median_offset / 1e3:+.1f} ms, "
                f"{ratio:.0%} of {median_sweep / 1e3:.1f} ms sweep)",
                False,
                ctx.errors,
                f"{lidar_id}: LiDAR sweep midpoints are systematically "
                f"{median_offset / 1e3:.1f} ms LATER than camera exposure "
                f"midpoints ({ratio:.0%} of the {median_sweep / 1e3:.1f} ms "
                f"sweep duration). A positive offset up to ~50% is normal for "
                f"global-shutter cameras triggered at sweep start, but >= 75% "
                f"indicates either a timestamp interpretation error or frames "
                f"paired across a constant index shift. Check the pairing against "
                f"source frame keys before revisiting the camera timestamp meaning "
                f"(exposure start, center, or end) -- see conventions.md "
                f'"Sequence frame-range intersection".',
            )
        else:
            _check(
                f"{lidar_id}: LiDAR-camera sweep midpoint alignment "
                f"(median offset {median_offset / 1e3:+.1f} ms, "
                f"{ratio:.0%} of {median_sweep / 1e3:.1f} ms sweep)",
                True,
                ctx.errors,
            )
