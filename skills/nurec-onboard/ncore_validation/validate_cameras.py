# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera data quality checks."""

from __future__ import annotations

import logging
import re

import numpy as np

from ncore.data import OpenCVFisheyeCameraModelParameters, ShutterType
from ncore_validation.utils import NcoreV4Context
from ncore_validation.validate_helpers import (
    MAX_CAMERA_OPTICAL_AXIS_AZIMUTH_DEG,
    MAX_CAMERA_OPTICAL_AXIS_TILT_Z,
    MAX_FISHEYE_MAX_ANGLE_DEVIATION_DEG,
    _check_sensor_frame_ts,
    _group,
    _warn,
)


log = logging.getLogger("validate")


def check_camera_frame_ts_order(ctx: NcoreV4Context) -> None:
    """Check camera frame timestamp ordering, monotonicity, and end-ts uniqueness."""
    if not ctx.camera_readers:
        return
    _group("Cameras")
    _check_sensor_frame_ts(ctx.camera_readers, "camera", ctx.errors)


def warn_camera_timestamps_all_identical(ctx: NcoreV4Context) -> None:
    """Warn when all cameras share byte-identical frame timestamps.

    If a converter uses a single global frame timestamp for every camera,
    the per-camera exposure timing is lost and NuRec rolling-shutter
    compensation will be incorrect.

    Note: hardware-synchronized global-shutter cameras may legitimately
    share timestamps.  This warning is most relevant for rolling-shutter
    setups where each camera has distinct exposure start/end times.
    """
    if not ctx.camera_readers or len(ctx.camera_readers) < 2:
        return

    ts_arrays = []
    for cam_reader in ctx.camera_readers.values():
        ts = cam_reader.frames_timestamps_us
        if ts.size == 0:
            continue
        ts_arrays.append(ts)

    if not ts_arrays:
        return

    ref = ts_arrays[0]
    all_identical = all(arr.shape == ref.shape and np.array_equal(arr, ref) for arr in ts_arrays[1:])
    if all_identical:
        cam_list = ", ".join(ctx.camera_readers.keys())
        _warn(
            f"All {len(ctx.camera_readers)} cameras share identical frame timestamps "
            f"({cam_list}). If cameras use rolling shutter, each should have its "
            f"own exposure start/end timestamps — using a global frame timestamp "
            f"causes incorrect rolling-shutter compensation in NuRec."
        )


def warn_all_cameras_global_shutter(ctx: NcoreV4Context) -> None:
    """Warn when every camera in a multi-camera setup has ShutterType.GLOBAL.

    Most automotive datasets use rolling-shutter cameras. If all cameras are
    marked GLOBAL, the converter likely has a rolling-shutter direction enum
    mismap (e.g. mapping by integer value instead of by name), which silently
    disables NuRec's per-scanline pose interpolation.
    """
    if not ctx.camera_readers or len(ctx.camera_readers) < 2 or not ctx.intrinsics_readers:
        return

    shutter_types: dict[str, ShutterType | None] = {}
    for camera_id in ctx.camera_readers:
        for intrinsics_reader in ctx.intrinsics_readers.values():
            try:
                cam_params = intrinsics_reader.get_camera_model_parameters(camera_id)
                shutter_types[camera_id] = getattr(cam_params, "shutter_type", None)
                break
            except KeyError:
                continue

    known = {cid: st for cid, st in shutter_types.items() if st is not None}
    if len(known) < 2:
        return

    if all(st == ShutterType.GLOBAL for st in known.values()):
        _warn(
            f"All {len(known)} cameras have ShutterType.GLOBAL. "
            f"If the source dataset uses rolling-shutter cameras, this likely "
            f"indicates a rolling-shutter direction enum mismap. "
            f"Verify the mapping from source-dataset enums to NCore ShutterType "
            f"(map by name, not by integer value)."
        )


def warn_camera_frame_count_mismatch_if_multi_cam(ctx: NcoreV4Context) -> None:
    """Warn when multi-camera sequences have unequal ``frames_timestamps_us`` row counts."""
    if ctx.camera_readers and len(ctx.camera_readers) > 1:
        cam_frame_counts = {
            camera_id: cam_reader.frames_timestamps_us.shape[0] for camera_id, cam_reader in ctx.camera_readers.items()
        }
        unique_counts = set(cam_frame_counts.values())
        if len(unique_counts) > 1:
            detail = ", ".join(f"{camera_id}={n}" for camera_id, n in cam_frame_counts.items())
            _warn(f"camera frame counts differ: {detail}")


def warn_camera_fisheye_max_angle_deviation(ctx: NcoreV4Context) -> None:
    """Warn when an OpenCV-fisheye camera's stored ``max_angle`` deviates from the
    calibration-derived value by more than ``MAX_FISHEYE_MAX_ANGLE_DEVIATION_DEG``.

    ``max_angle`` is the Kannala-Brandt polynomial's valid domain boundary.
    Setting it too large forces NuRec's rasterizer to evaluate the KB polynomial
    outside its calibration range, producing geometric distortion and blur at image
    edges.  Setting it too small silently drops valid image-edge pixels from
    reconstruction.

    Uses ``OpenCVFisheyeCameraModelParameters.compute_max_angle()`` -- a
    NCore built-in that finds the valid angle that maps to the farthest
    image corner. Only applicable to
    ``OpenCVFisheyeCameraModelParameters``; FTheta cameras are skipped (their
    ``max_angle`` is a direct calibration output from the vendor toolchain and has no
    equivalent NCore built-in for comparison).

    """
    if not ctx.camera_readers or not ctx.intrinsics_readers:
        return

    issues: list[str] = []
    for camera_id in ctx.camera_readers:
        params = None
        for reader in ctx.intrinsics_readers.values():
            try:
                params = reader.get_camera_model_parameters(camera_id)
                break
            except KeyError:
                continue
        if not isinstance(params, OpenCVFisheyeCameraModelParameters):
            continue

        computed = OpenCVFisheyeCameraModelParameters.compute_max_angle(
            resolution=params.resolution,
            focal_length=params.focal_length,
            principal_point=params.principal_point,
            radial_coeffs=params.radial_coeffs,
        )
        stored = float(params.max_angle)
        delta_deg = abs(np.degrees(stored) - np.degrees(computed))

        if delta_deg > MAX_FISHEYE_MAX_ANGLE_DEVIATION_DEG:
            direction = "larger" if stored > computed else "smaller"
            issues.append(
                f"{camera_id}: stored max_angle={np.degrees(stored):.1f} deg, "
                f"calibration-derived={np.degrees(computed):.1f} deg "
                f"(deviation {delta_deg:.1f} deg, stored is {direction}). "
                + (
                    "Too large: NuRec evaluates the KB polynomial outside its calibration "
                    "domain, causing geometric distortion and blur at image edges. "
                    "Fix: use OpenCVFisheyeCameraModelParameters.compute_max_angle() "
                    "to derive max_angle from the calibration parameters."
                    if stored > computed
                    else "Too small: image-edge pixels produce rays beyond max_angle and are "
                    "silently dropped from NuRec reconstruction. "
                    "Fix: use OpenCVFisheyeCameraModelParameters.compute_max_angle() "
                    "to derive max_angle from the calibration parameters."
                )
            )

    for issue in issues:
        _warn(f"OpenCV-fisheye max_angle deviation: {issue}")


def warn_camera_optical_axis_tilt(ctx: NcoreV4Context) -> None:
    """Warn when a camera's optical axis sits more than ~45 deg off horizontal in the rig frame.

    NCore camera convention has +Z as the optical axis, so the optical
    axis expressed in the rig frame is the third column of
    ``T_camera_rig.R``.  AV cameras are mounted within ~20 deg of
    horizontal in practice; an optical axis with ``|z_rig|`` above
    ``MAX_CAMERA_OPTICAL_AXIS_TILT_Z`` (~45 deg from horizontal) almost
    always indicates a missing source-frame to NCore-camera-frame
    basis change in ``T_camera_rig``: when the source dataset's camera
    frame uses +X or +Y (rather than +Z) as the principal axis, the
    converter must right-multiply ``T_camera_rig.R`` by a static rotation
    that re-bases the source camera axes onto NCore's (+X right, +Y down,
    +Z forward) so the resulting camera +Z lands along the physical
    forward direction in rig coords.  Without that change the camera
    ends up looking straight up or straight down.

    Pure rig-Z (yaw) rotations of ``T_camera_rig`` are *not* caught
    here: those leave the optical axis horizontal, just pointing the
    wrong horizontal direction.  See
    :func:`warn_camera_optical_axis_azimuth` for the companion check
    that catches yaw bugs when the camera id encodes a mounting
    direction; for opaque ids use NCore's
    ``//tools:ncore_project_pc_to_img`` and ``ncore_vis`` for
    pixel-level visual confirmation.
    """
    if not (ctx.camera_readers and ctx.poses_readers):
        return

    issues: list[str] = []
    for camera_id in ctx.camera_readers:
        T_camera_rig: np.ndarray | None = None
        for pose_reader in ctx.poses_readers.values():
            try:
                T_camera_rig = pose_reader.get_static_pose(camera_id, "rig").astype(np.float64)
                break
            except KeyError:
                continue
        if T_camera_rig is None:
            continue

        optical_axis_rig = T_camera_rig[:3, 2]
        norm = float(np.linalg.norm(optical_axis_rig))
        if norm < 1e-9:
            continue
        z_component = float(optical_axis_rig[2] / norm)
        if abs(z_component) > MAX_CAMERA_OPTICAL_AXIS_TILT_Z:
            tilt_deg = float(np.degrees(np.arcsin(min(abs(z_component), 1.0))))
            direction = "up" if z_component > 0 else "down"
            issues.append(
                f"{camera_id}: optical axis tilted {tilt_deg:.0f} deg from horizontal "
                f"(pointing {direction}; z_rig={z_component:+.2f}, threshold "
                f"|z_rig| <= {MAX_CAMERA_OPTICAL_AXIS_TILT_Z:.2f})"
            )

    for issue in issues:
        _warn(
            f"camera optical axis far from horizontal: {issue}. "
            f"Likely a missing source-frame to NCore-camera-frame basis change "
            f"in T_camera_rig: if the source camera frame is +X- or +Y-principal "
            f"(rather than NCore's +Z-principal, +X right, +Y down), "
            f"T_camera_rig.R needs to be right-multiplied by a static rotation "
            f"that re-bases the source axes onto NCore's. "
            f"Pure rig-Z yaw rotation bugs are caught by "
            f"warn_camera_optical_axis_azimuth when the camera id encodes a "
            f"mounting direction; otherwise use ncore_project_pc_to_img for "
            f"pixel-level visual confirmation."
        )


_CAMERA_DIRECTION_TOKENS_FORWARD: frozenset[str] = frozenset({"front", "forward", "fwd"})
_CAMERA_DIRECTION_TOKENS_BACKWARD: frozenset[str] = frozenset({"back", "rear", "bwd"})
_CAMERA_DIRECTION_TOKENS_LEFT: frozenset[str] = frozenset({"left", "port"})
_CAMERA_DIRECTION_TOKENS_RIGHT: frozenset[str] = frozenset({"right", "starboard"})


def _parse_camera_direction_azimuth_rad(camera_id: str) -> float | None:
    """Return the rig-frame azimuth (radians, ``atan2(rig_+Y, rig_+X)``) implied by
    direction tokens in ``camera_id``, or ``None`` when no unambiguous direction
    is encoded.

    Tokens are matched case-insensitively against alphanumeric components of
    the id (split on any non-alphanumeric character).  Forward / backward and
    left / right axes are independent: their sum gives the four cardinal and
    four diagonal directions.  Conflicting tokens (e.g. ``front_back_camera``)
    return ``None`` so the caller stays silent rather than guess.

    Mapping (NCore convention: rig +X = forward, rig +Y = left):
        ``front`` / ``forward`` / ``fwd``  -> rig +X (azimuth   0 deg)
        ``back``  / ``rear``    / ``bwd``  -> rig -X (azimuth 180 deg)
        ``left``  / ``port``               -> rig +Y (azimuth +90 deg)
        ``right`` / ``starboard``          -> rig -Y (azimuth -90 deg)
        ``front_left``                     -> rig +X+Y (azimuth +45 deg)
        etc.
    """
    tokens = {t for t in re.split(r"[^a-z0-9]+", camera_id.lower()) if t}
    has_front = bool(tokens & _CAMERA_DIRECTION_TOKENS_FORWARD)
    has_back = bool(tokens & _CAMERA_DIRECTION_TOKENS_BACKWARD)
    has_left = bool(tokens & _CAMERA_DIRECTION_TOKENS_LEFT)
    has_right = bool(tokens & _CAMERA_DIRECTION_TOKENS_RIGHT)

    if has_front and has_back:
        return None
    if has_left and has_right:
        return None

    fwd = 1.0 if has_front else (-1.0 if has_back else 0.0)
    side = 1.0 if has_left else (-1.0 if has_right else 0.0)
    if fwd == 0.0 and side == 0.0:
        return None

    return float(np.arctan2(side, fwd))


def warn_camera_optical_axis_azimuth(ctx: NcoreV4Context) -> None:
    """Warn when a camera's id encodes a mounting direction (front / back / left /
    right or any of the four diagonals) but the actual optical-axis azimuth in
    the rig frame disagrees with that direction by more than
    ``MAX_CAMERA_OPTICAL_AXIS_AZIMUTH_DEG``.

    NCore convention puts rig +X along the travel direction (validated
    independently by ``check_rig_forward_alignment``), so e.g. a ``front_*``
    camera's optical axis (camera +Z, i.e. ``T_camera_rig.R[:, 2]``) projected
    into the rig XY plane should sit close to rig +X.  This is the only
    structural check that catches whole-rig yaw bugs in ``T_camera_rig`` --
    bugs the optical-axis-tilt warning fundamentally cannot see because they
    leave the optical axis horizontal.

    Silent in any of:
      * the camera id contains no recognized direction token;
      * the camera id contains conflicting tokens (e.g. ``front_back_cam``);
      * the optical axis is too close to vertical (``|z_rig| > MAX_CAMERA_OPTICAL_AXIS_TILT_Z``)
        for azimuth to be meaningful -- that case is the optical-axis-tilt
        warning's responsibility.
    """
    if not (ctx.camera_readers and ctx.poses_readers):
        return

    issues: list[str] = []
    for camera_id in ctx.camera_readers:
        expected_az = _parse_camera_direction_azimuth_rad(camera_id)
        if expected_az is None:
            continue

        T_camera_rig: np.ndarray | None = None
        for pose_reader in ctx.poses_readers.values():
            try:
                T_camera_rig = pose_reader.get_static_pose(camera_id, "rig").astype(np.float64)
                break
            except KeyError:
                continue
        if T_camera_rig is None:
            continue

        optical_axis = T_camera_rig[:3, 2]
        norm = float(np.linalg.norm(optical_axis))
        if norm < 1e-9:
            continue
        oa = optical_axis / norm
        if abs(float(oa[2])) > MAX_CAMERA_OPTICAL_AXIS_TILT_Z:
            continue

        actual_az = float(np.arctan2(oa[1], oa[0]))
        signed_delta = float(np.arctan2(np.sin(actual_az - expected_az), np.cos(actual_az - expected_az)))
        delta_deg = float(np.degrees(abs(signed_delta)))
        if delta_deg > MAX_CAMERA_OPTICAL_AXIS_AZIMUTH_DEG:
            expected_deg = float(np.degrees(expected_az))
            actual_deg = float(np.degrees(actual_az))
            issues.append(
                f"{camera_id}: id implies azimuth {expected_deg:+.0f} deg but "
                f"optical axis azimuth in rig is {actual_deg:+.0f} deg "
                f"(delta {delta_deg:.0f} deg, threshold "
                f"{MAX_CAMERA_OPTICAL_AXIS_AZIMUTH_DEG:.0f} deg)"
            )

    for issue in issues:
        _warn(
            f"camera optical-axis azimuth disagrees with id-implied direction: "
            f"{issue}. Likely a yaw (rig-Z) rotation bug in T_camera_rig -- the "
            f"converter built the static extrinsic in a frame whose +X is not "
            f"aligned with rig +X (forward / travel direction). NCore convention: "
            f"rig +X = forward, rig +Y = left. Cross-confirm with NCore's "
            f"ncore_project_pc_to_img / ncore_vis at pixel level."
        )
