# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose data quality checks."""

from __future__ import annotations

import logging

import numpy as np

from scipy.spatial.transform import Rotation

from ncore_validation.utils import NcoreV4Context
from ncore_validation.validate_helpers import (
    MAX_POSE_ANGULAR_RATE_DEG_S,
    MAX_STATIC_POSE_TRANSLATION_M,
    MIN_RIG_FORWARD_ALIGNMENT,
    MIN_RIG_FORWARD_ALIGNMENT_ERROR,
    MIN_RIG_FORWARD_ARCLENGTH_M,
    MIN_RIG_FORWARD_DISPLACEMENT_M,
    MIN_RIG_FORWARD_SAMPLES,
    NEAR_ZERO_SENSOR_TRANSLATION_M,
    POSE_TRANSLATION_MAX_M,
    _check,
    _group,
    _warn,
)


log = logging.getLogger("validate")


def check_static_sensor_rig_translations(ctx: NcoreV4Context) -> None:
    """Bound static sensor->rig translation norms (calibration / frame sanity)."""
    if ctx.poses_readers:
        _group("Poses")
        bad_poses: list[str] = []
        sensor_ids = list(ctx.camera_readers.keys()) + list(ctx.lidar_readers.keys())
        for sensor_id in sensor_ids:
            for pose_reader in ctx.poses_readers.values():
                try:
                    # Static pose is a 4x4 SE(3) matrix; [:3, 3] extracts the
                    # translation vector (sensor origin in rig coordinates).
                    pose = pose_reader.get_static_pose(sensor_id, "rig")
                    t_norm = float(np.linalg.norm(pose[:3, 3]))
                    if t_norm > MAX_STATIC_POSE_TRANSLATION_M:
                        bad_poses.append(f"{sensor_id}→rig ({t_norm:.1f} m)")
                    break
                except KeyError:
                    continue
        _check(
            f"static sensor→rig translations < {MAX_STATIC_POSE_TRANSLATION_M} m",
            len(bad_poses) == 0,
            ctx.errors,
            f"sensor→rig translations unexpectedly large: {'; '.join(bad_poses)}. "
            "This may indicate a coordinate frame or calibration error.",
        )


def warn_identity_sensor_translations(ctx: NcoreV4Context) -> None:
    """Warn when all sensor-to-rig static translations are near-zero (< 1 cm).

    A converter that missed the dataset's calibration file typically leaves
    every sensor pose as identity (zero translation).  Finding all sensors at
    (0, 0, 0) simultaneously is almost always a calibration-load bug rather
    than a legitimate configuration.

    Skips silently when fewer than two sensors are present (a single mount
    at the rig origin is plausible) or when any sensor's static pose cannot
    be resolved from the available readers.
    """
    sensor_ids = list(ctx.camera_readers) + list(ctx.lidar_readers)
    if len(sensor_ids) < 2:
        return

    magnitudes: list[float] = []
    for sensor_id in sensor_ids:
        found = False
        for pr in ctx.poses_readers.values():
            try:
                T = np.asarray(pr.get_static_pose(sensor_id, "rig"), dtype=np.float64)
                magnitudes.append(float(np.linalg.norm(T[:3, 3])))
                found = True
                break
            except (KeyError, IndexError):
                continue
        if not found:
            return

    if all(m < NEAR_ZERO_SENSOR_TRANSLATION_M for m in magnitudes):
        _warn(
            "all sensor-to-rig translations are near-zero — confirm the "
            "calibration file was loaded; sensor mount heights are rarely zero."
        )


def check_pose_dtypes(ctx: NcoreV4Context) -> None:
    """Require stored pose dtypes that match NuRec's matmul expectations.

    NuRec's pose processing does not promote input dtypes, so a ``float64``
    sensor-to-rig or rig-to-world pose collides with ``float32``
    intermediates, raising ``RuntimeError: expected mat1 and mat2 to have
    the same dtype, but got: double != float``.  The only exception is
    ``world -> world_global``, which must stay ``float64`` for geo-
    referenced precision.

    NCore's singular accessors (``get_static_pose``, ``get_dynamic_pose``)
    force ``float64`` regardless of stored dtype, so this check uses the
    plural accessors (``get_static_poses``, ``get_dynamic_poses``) which
    preserve the dtype actually written to disk.
    """
    if not ctx.poses_readers:
        return

    bad_static: list[str] = []
    bad_dynamic: list[str] = []
    for component_instance_name, pose_reader in ctx.poses_readers.items():
        try:
            static_poses = list(pose_reader.get_static_poses())
        except (OSError, ValueError, KeyError) as e:
            log.debug("Could not read static poses from %s: %s", component_instance_name, e)
            static_poses = []
        for (source, target), pose in static_poses:
            expected = np.float64 if (source == "world" and target == "world_global") else np.float32
            if pose.dtype != np.dtype(expected):
                bad_static.append(
                    f"{component_instance_name}:{source}→{target} (dtype={pose.dtype}, expected={np.dtype(expected)})"
                )

        try:
            dynamic_poses = list(pose_reader.get_dynamic_poses())
        except (OSError, ValueError, KeyError) as e:
            log.debug("Could not read dynamic poses from %s: %s", component_instance_name, e)
            dynamic_poses = []
        for (source, target), (poses, _ts) in dynamic_poses:
            if poses.dtype != np.dtype(np.float32):
                bad_dynamic.append(
                    f"{component_instance_name}:{source}→{target} (dtype={poses.dtype}, expected=float32)"
                )

    _check(
        "static pose dtypes (float32 except world→world_global which is float64)",
        not bad_static,
        ctx.errors,
        f"static pose dtype mismatch -- NuRec will raise 'double != float': {'; '.join(bad_static)}",
    )
    _check(
        "dynamic pose dtypes (float32)",
        not bad_dynamic,
        ctx.errors,
        f"dynamic pose dtype mismatch -- NuRec will raise 'double != float': {'; '.join(bad_dynamic)}",
    )


def check_pose_se3_valid(ctx: NcoreV4Context) -> None:
    """Fail when any stored pose does not have a valid SE3 last row ``[0,0,0,1]``.

    NCore's writer asserts ``pose[3,:] == [0,0,0,1]`` for every static and
    dynamic pose.  Malformed last rows (e.g. all-zeros from uninitialized
    memory) produce silently wrong transforms in NuRec.
    """
    if not ctx.poses_readers:
        return
    expected = np.array([0.0, 0.0, 0.0, 1.0])
    bad: list[str] = []
    for component_instance_name, pose_reader in ctx.poses_readers.items():
        try:
            static_poses = list(pose_reader.get_static_poses())
        except Exception:
            static_poses = []
        for (source, target), pose in static_poses:
            if not np.all(pose[3, :] == expected):
                bad.append(f"{component_instance_name}:{source}→{target} (static, last row={pose[3, :].tolist()})")

        try:
            dynamic_poses = list(pose_reader.get_dynamic_poses())
        except Exception:
            dynamic_poses = []
        for (source, target), (poses, _ts) in dynamic_poses:
            bad_mask = ~np.all(poses[:, 3, :] == expected, axis=1)
            if np.any(bad_mask):
                idx = int(np.argmax(bad_mask))
                bad.append(
                    f"{component_instance_name}:{source}→{target} "
                    f"(dynamic, {int(bad_mask.sum())}/{len(poses)} bad, "
                    f"first at index {idx}: last row={poses[idx, 3, :].tolist()})"
                )

    _check(
        "poses are valid SE3 (last row = [0, 0, 0, 1])",
        not bad,
        ctx.errors,
        f"invalid SE3 poses: {'; '.join(bad)}",
    )


def check_pose_translation_magnitude(ctx: NcoreV4Context) -> None:
    """Fail when rig->world translations are far from the origin.

    NCore requires ``T_rig_world`` poses stored as ``float32``.  If the
    converter stores raw global-frame translations (GPS / UTM / ENU) without
    re-referencing to the first pose, float32 quantization noise causes
    motion-compensation artifacts (diagonal streaking).  At 10 km, float32
    precision is ~1 mm; at 1000 km it degrades to ~0.1 m.
    """
    if not ctx.poses_readers:
        return
    for _name, pose_reader in ctx.poses_readers.items():
        try:
            poses, _ts = pose_reader.get_dynamic_pose("rig", "world")
        except KeyError:
            continue
        trans = poses[:, :3, 3].astype(np.float64)
        max_abs = float(np.abs(trans).max())
        eps = float(np.finfo(np.float32).eps * max_abs)
        ok = max_abs <= POSE_TRANSLATION_MAX_M
        _check(
            f"rig→world translations near origin "
            f"(max |component| = {max_abs:.0f} m, float32 precision "
            f"≈ {eps * 1000:.2f} mm)",
            ok,
            ctx.errors,
            f"rig→world translations are far from the origin "
            f"(max |component| = {max_abs:.0f} m, float32 precision "
            f"≈ {eps * 1000:.2f} mm). Re-reference poses to the "
            f"first ego pose before float32 cast to avoid "
            f"motion-compensation artifacts.",
        )


def _rig_forward_alignment(
    ctx: NcoreV4Context,
    min_displacement_m: float,
    min_samples: int,
    min_arclength_m: float,
) -> tuple[float, int, np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute the median rig +X alignment with travel direction.

    Returns ``(alignment, n_moving, T_rig_worlds, moving_mask, seg_dirs)``
    or ``None`` when the trajectory is unavailable or the scope guards
    (arclength, sample count) are not met.
    """
    T_rig_worlds = None
    for pose_reader in ctx.poses_readers.values():
        try:
            T_rig_worlds, _ = pose_reader.get_dynamic_pose("rig", "world")
            break
        except KeyError:
            continue
    if T_rig_worlds is None or len(T_rig_worlds) < 2:
        return None

    positions = T_rig_worlds[:, :3, 3].astype(np.float64)
    segments = positions[1:] - positions[:-1]
    norms = np.linalg.norm(segments, axis=1)

    if float(np.sum(norms)) < min_arclength_m:
        return None

    moving = norms > min_displacement_m
    n_moving = int(np.sum(moving))
    if n_moving < min_samples:
        return None

    rig_x_axes = T_rig_worlds[:-1, :3, 0].astype(np.float64)[moving]
    seg_dirs = segments[moving] / norms[moving, np.newaxis]
    dots = np.sum(rig_x_axes * seg_dirs, axis=1)
    alignment = float(np.median(dots))
    return alignment, n_moving, T_rig_worlds, moving, seg_dirs


def check_rig_forward_alignment(ctx: NcoreV4Context) -> None:
    """Hard-error when rig +X is severely misaligned with travel direction.

    Computes ``median(dot(rig_x_world, travel_dir))`` over moving segments
    and fails if below ``MIN_RIG_FORWARD_ALIGNMENT_ERROR``.  Skips when
    there is no rig trajectory, insufficient moving samples, or too
    little arclength.
    """
    result = _rig_forward_alignment(
        ctx,
        min_displacement_m=MIN_RIG_FORWARD_DISPLACEMENT_M,
        min_samples=MIN_RIG_FORWARD_SAMPLES,
        min_arclength_m=MIN_RIG_FORWARD_ARCLENGTH_M,
    )
    if result is None:
        return
    alignment, n_moving, _, _, _ = result

    _check(
        f"rig +X alignment with travel direction "
        f"(median dot={alignment:.3f} over {n_moving} moving "
        f"segments; threshold >= {MIN_RIG_FORWARD_ALIGNMENT_ERROR})",
        alignment >= MIN_RIG_FORWARD_ALIGNMENT_ERROR,
        ctx.errors,
        f"median rig-+X alignment with travel direction is "
        f"{alignment:.3f}, below the {MIN_RIG_FORWARD_ALIGNMENT_ERROR} "
        f"hard-error threshold. Per NCore's rig-frame convention "
        f"(docs/data/conventions.rst), rig +X is forward; this "
        f"indicates the rig trajectory was derived in a sensor frame "
        f"(e.g. LiDAR) without rotating into the rig frame, or that "
        f"T_lidar_rig / T_camera_rig have a rotation axis swap.",
    )


def warn_rig_forward_not_x(ctx: NcoreV4Context) -> None:
    """Warn when the rig's +x axis does not align with the travel direction.

    For each consecutive pose pair with sufficient displacement, computes the
    dot product of the rig +x axis with the instantaneous travel direction.
    The median of these per-segment values is compared against the threshold.
    This is robust to sequences where the vehicle turns significantly.
    """
    result = _rig_forward_alignment(ctx, min_displacement_m=0.01, min_samples=1, min_arclength_m=0.0)
    if result is None:
        return
    alignment, _, T_rig_worlds, moving, seg_dirs = result

    if alignment < MIN_RIG_FORWARD_ALIGNMENT:
        rig_y_axes = T_rig_worlds[:-1, :3, 1].astype(np.float64)[moving]
        median_y = float(np.median(np.sum(rig_y_axes * seg_dirs, axis=1)))
        candidates = {
            "+x": alignment,
            "-x": -alignment,
            "+y": median_y,
            "-y": -median_y,
        }
        actual_forward = max(candidates, key=lambda axis: candidates[axis])
        _warn(
            f"Rig +x axis does not align with travel direction "
            f"(median dot={alignment:.3f}, threshold={MIN_RIG_FORWARD_ALIGNMENT}). "
            f"Axis most aligned with forward: {actual_forward}. "
            f"This may affect ncore_vis arrows and NuRec directional priors."
        )


def warn_pose_angular_rate(ctx: NcoreV4Context) -> None:
    """Warn when consecutive rig->world poses have physically implausible angular rates.

    Computes the angular rate (deg/s) between every consecutive pair of
    dynamic pose waypoints.  A rate exceeding ``MAX_POSE_ANGULAR_RATE_DEG_S``
    indicates corrupted pose entries, often from sampling a per-pixel or
    per-column pose grid that contains invalid/zero rows.
    """
    if not ctx.poses_readers:
        return

    T_rig_worlds = None
    pose_ts = None
    for pose_reader in ctx.poses_readers.values():
        try:
            T_rig_worlds, pose_ts = pose_reader.get_dynamic_pose("rig", "world")
            break
        except KeyError:
            continue
    if T_rig_worlds is None or pose_ts is None or pose_ts.size < 2:
        return

    R = T_rig_worlds[:, :3, :3].astype(np.float64)
    r_rel = R[1:] @ R[:-1].transpose(0, 2, 1)
    angles_deg = np.degrees(Rotation.from_matrix(r_rel).magnitude())
    dt = np.diff(pose_ts).astype(np.float64) / 1e6
    valid = dt > 0
    rates = np.where(valid, angles_deg / np.where(valid, dt, 1.0), 0.0)
    max_rate_idx = int(np.argmax(rates))
    max_rate = float(rates[max_rate_idx])

    if max_rate > MAX_POSE_ANGULAR_RATE_DEG_S:
        _warn(
            f"Pose trajectory has physically implausible angular rate: "
            f"{max_rate:.0f} deg/s between waypoints {max_rate_idx} and "
            f"{max_rate_idx + 1} (timestamps {int(pose_ts[max_rate_idx])} -> "
            f"{int(pose_ts[max_rate_idx + 1])}). "
            f"Threshold: {MAX_POSE_ANGULAR_RATE_DEG_S} deg/s. "
            f"This usually indicates corrupted pose entries from sampling "
            f"a per-pixel or per-column pose grid. Use frame-level and "
            f"per-camera-image poses as the primary trajectory source."
        )
