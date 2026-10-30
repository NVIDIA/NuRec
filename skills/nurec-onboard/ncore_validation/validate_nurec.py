# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NuRec Readiness tier checks."""

from __future__ import annotations

import logging

import numpy as np

from ncore.data import RowOffsetStructuredSpinningLidarModelParameters
from ncore_validation.utils import (
    NcoreV4Context,
    any_intrinsics_reader_has_lidar_model,
    any_reader_has,
    get_lidar_model_parameters_first,
)
from ncore_validation.validate_cuboids import warn_cuboid_tracks_below_two_observations
from ncore_validation.validate_helpers import EGO_MASK_KEY_PATTERN, _check, _group, _section, _warn


log = logging.getLogger("validate")


def check_sequence_id_nurec_safe(ctx: NcoreV4Context) -> None:
    """Require a sequence ID that is safe as a store filename and discovery prefix.

    Converters pass ``store_base_name=sequence_id``, so this field names the
    data shard, the sequence meta JSON, and the auxiliary stores.  NuRec finds
    auxiliary data by truncating the *data shard* filename at its first ``"."``
    and keeping neighboring files whose names begin with that prefix plus
    ``".aux."``, so a dotted sequence ID truncates mid-name and matches
    nothing.  A converter that names its shard independently of
    ``sequence_id`` is outside this check.

    Args:
        ctx: Open NCore V4 sequence context.
    """
    _group("Sequence ID")
    sequence_id = ctx.reader.sequence_id or ""
    invalid_characters = sorted({char for char in sequence_id if char in "./" or char.isspace()})
    _check(
        "sequence_id safe as a store filename and aux-discovery prefix",
        not invalid_characters,
        ctx.errors,
        (
            f"sequence_id {sequence_id!r} contains unsafe character(s): "
            f"{', '.join(repr(char) for char in invalid_characters)}. "
            "A '.' truncates the prefix NuRec matches auxiliary stores against, so the "
            "reconstruction silently runs without them; this bites when dataset.aux_data "
            "is enabled, which is the default, and nurec-onboard always generates "
            "auxiliary data. '/' and whitespace do not break the match -- they are "
            "rejected because the id also names an output directory and a store filename."
        ),
    )


def check_cameras_nurec(ctx: NcoreV4Context) -> None:
    """NuRec checklist: cameras exist, intrinsics, camera->rig poses, frame times in interval."""
    _group("Cameras")
    _check(
        "at least one present",
        len(ctx.camera_readers) >= 1,
        ctx.errors,
        "at least one camera required",
    )

    if ctx.camera_readers:
        missing_intrinsics = [
            camera_id
            for camera_id in ctx.camera_readers
            if not any_reader_has(
                ctx.intrinsics_readers,
                lambda reader, sensor_id: reader.get_camera_model_parameters(sensor_id),
                camera_id,
            )
        ]
        _check(
            "every camera has intrinsics",
            len(missing_intrinsics) == 0,
            ctx.errors,
            f"cameras missing intrinsics: {', '.join(missing_intrinsics)}",
        )

        missing_extrinsics = [
            camera_id
            for camera_id in ctx.camera_readers
            if not any_reader_has(
                ctx.poses_readers,
                lambda reader, sensor_id: reader.get_static_pose(sensor_id, "rig"),
                camera_id,
            )
        ]
        _check(
            "every camera has camera→rig pose",
            len(missing_extrinsics) == 0,
            ctx.errors,
            f"cameras missing camera→rig static pose: {', '.join(missing_extrinsics)}",
        )

        cam_frames_out_of_range: list[str] = []
        for camera_id, cam_reader in ctx.camera_readers.items():
            # frames_timestamps_us is [N, 2]: columns are (exposure_start, exposure_end).
            ts = cam_reader.frames_timestamps_us
            if ts.size == 0:
                continue
            in_range = (
                (ts[:, 0] >= ctx.interval.start)
                & (ts[:, 0] < ctx.interval.stop)
                & (ts[:, 1] >= ctx.interval.start)
                & (ts[:, 1] < ctx.interval.stop)
            )
            if not in_range.all():
                cam_frames_out_of_range.append(camera_id)
        _check(
            "frame timestamps in interval",
            len(cam_frames_out_of_range) == 0,
            ctx.errors,
            f"camera frame timestamps outside interval: {', '.join(cam_frames_out_of_range)}",
        )


def check_masks_nurec(ctx: NcoreV4Context) -> None:
    """Require at least one masks component (NuRec expects it)."""
    _group("Masks")
    _check(
        "component present",
        len(ctx.masks_readers) >= 1,
        ctx.errors,
        "MasksComponent not found — NuRec requires it. "
        "Register MasksComponent.Writer and call store_camera_masks(camera_id, {}) "
        "for each camera, even if no masks are available.",
    )


def warn_masks_ego_key(ctx: NcoreV4Context) -> None:
    """Warn when cameras have masks but are missing the exact ``"ego"`` key.

    NuRec expects the ego-vehicle mask to be stored under the key ``"ego"``; any
    other name is silently ignored.  Two tiers:

    - Ego-like key present (e.g. ``"ego_vehicle"``) — likely a naming mistake.
    - No ego-related key at all — informational note that NuRec will skip
      ego-vehicle masking for those cameras.
    """
    if not ctx.masks_readers:
        return
    cameras_ego_misspelled: dict[str, list[str]] = {}
    cameras_no_ego: list[str] = []
    for camera_id in ctx.camera_readers:
        for masks_reader in ctx.masks_readers.values():
            try:
                names = masks_reader.get_camera_mask_names(camera_id)
            except KeyError:
                continue
            if names and "ego" not in names:
                ego_like = [n for n in names if EGO_MASK_KEY_PATTERN.search(n)]
                if ego_like:
                    cameras_ego_misspelled[camera_id] = ego_like
                else:
                    cameras_no_ego.append(camera_id)
            break
    if cameras_ego_misspelled:
        details = ", ".join(f"{cam}: {keys}" for cam, keys in sorted(cameras_ego_misspelled.items()))
        _warn(
            'ego-like mask keys found but not the exact "ego" key — '
            "NuRec will silently ignore them. "
            f'Rename to "ego" in store_camera_masks(): {details}'
        )
    if cameras_no_ego:
        _warn(
            'cameras have masks but no "ego" key — NuRec will not apply '
            "ego-vehicle masking for: "
            f"{', '.join(sorted(cameras_no_ego))}"
        )


def check_lidar_nurec(ctx: NcoreV4Context) -> None:
    """NuRec checklist: LiDAR count, intrinsics, lidar->rig poses, frame times in interval."""
    _group("LiDAR")
    _check(
        "at least one sensor present",
        len(ctx.lidar_readers) >= 1,
        ctx.errors,
        "at least one LiDAR sensor required",
    )

    if ctx.lidar_readers:
        missing_lidar_intrinsics = [
            lidar_id
            for lidar_id in ctx.lidar_readers
            if not any_intrinsics_reader_has_lidar_model(ctx.intrinsics_readers, lidar_id)
        ]
        _check(
            "every LiDAR has intrinsics",
            len(missing_lidar_intrinsics) == 0,
            ctx.errors,
            f"LiDARs missing intrinsics: {', '.join(missing_lidar_intrinsics)}. "
            "NuRec expects spinning LiDAR models: register "
            "IntrinsicsComponent.Writer.store_lidar_intrinsics per NCore data formats.",
        )

        missing_lidar_poses = [
            lidar_id
            for lidar_id in ctx.lidar_readers
            if not any_reader_has(
                ctx.poses_readers,
                lambda reader, sensor_id: reader.get_static_pose(sensor_id, "rig"),
                lidar_id,
            )
        ]
        _check(
            "every sensor has lidar→rig pose",
            len(missing_lidar_poses) == 0,
            ctx.errors,
            f"LiDARs missing lidar→rig static pose: {', '.join(missing_lidar_poses)}. "
            "ncore_vis needs this for motion compensation (use identity if LiDAR defines the rig frame).",
        )

        lidar_frames_out_of_range: list[str] = []
        for lidar_id, lidar_reader in ctx.lidar_readers.items():
            ts = lidar_reader.frames_timestamps_us
            if ts.size == 0:
                continue
            in_range = (
                (ts[:, 0] >= ctx.interval.start)
                & (ts[:, 0] < ctx.interval.stop)
                & (ts[:, 1] >= ctx.interval.start)
                & (ts[:, 1] < ctx.interval.stop)
            )
            if not in_range.all():
                lidar_frames_out_of_range.append(lidar_id)
        _check(
            "frame timestamps in interval",
            len(lidar_frames_out_of_range) == 0,
            ctx.errors,
            f"LiDAR frame timestamps outside interval: {', '.join(lidar_frames_out_of_range)}",
        )

        # model_element maps each ray to a (row, col) in the LiDAR's structured grid.
        # NuRec uses this to evaluate the intrinsic model for each ray independently of
        # the stored direction, so it must be present, correctly typed, and within bounds.
        model_element_issues: list[str] = []
        for lidar_id, lidar_reader in ctx.lidar_readers.items():
            ts = lidar_reader.frames_timestamps_us
            if ts.size == 0:
                model_element_issues.append(f"{lidar_id} (no frames; cannot verify model_element)")
                continue

            examined_frame = False
            for frame_idx in range(ts.shape[0]):
                frame_end_ts = int(ts[frame_idx, 1])
                try:
                    direction = lidar_reader.get_frame_ray_bundle_data(frame_end_ts, "direction")
                except (KeyError, ValueError, OSError) as e:
                    log.debug(
                        "check_lidar_nurec: failed to read direction at ts=%d for %s: %s",
                        frame_end_ts,
                        lidar_id,
                        e,
                    )
                    continue
                n_rays = int(direction.shape[0])
                if n_rays == 0:
                    continue
                examined_frame = True
                has_model_element = lidar_reader.has_frame_ray_bundle_data(frame_end_ts, "model_element")
                if not has_model_element:
                    model_element_issues.append(f"{lidar_id} (missing model_element; required [N,2] uint16 row/col)")
                    break

                model_element = lidar_reader.get_frame_ray_bundle_data(frame_end_ts, "model_element")
                model_element_errors: list[str] = []
                if model_element.dtype != np.dtype(np.uint16):
                    model_element_errors.append(f"model_element dtype {model_element.dtype}, expected uint16")
                if model_element.ndim != 2 or model_element.shape[1] != 2:
                    model_element_errors.append(f"model_element shape {model_element.shape}, expected (N, 2)")
                elif model_element.shape[0] != n_rays:
                    model_element_errors.append(
                        f"model_element length {model_element.shape[0]} != direction length {n_rays}"
                    )
                if model_element_errors:
                    model_element_issues.append(f"{lidar_id} ({'; '.join(model_element_errors)})")
                    break

                # Verify (row, col) indices are within the grid dimensions defined
                # by the spinning LiDAR intrinsics (n_rows x n_columns).
                params = get_lidar_model_parameters_first(ctx.intrinsics_readers, lidar_id)
                if isinstance(params, RowOffsetStructuredSpinningLidarModelParameters):
                    rows = model_element[:, 0].astype(np.int64, copy=False)
                    cols = model_element[:, 1].astype(np.int64, copy=False)
                    if rows.max() >= params.n_rows or cols.max() >= params.n_columns:
                        model_element_issues.append(
                            f"{lidar_id} (model_element outside grid: need rows in [0, {params.n_rows}), "
                            f"cols in [0, {params.n_columns}))"
                        )
                break

            if not examined_frame:
                model_element_issues.append(f"{lidar_id} (no non-empty ray bundle; cannot verify model_element)")

        _check(
            "per-ray model_element present and consistent",
            len(model_element_issues) == 0,
            ctx.errors,
            "; ".join(model_element_issues),
        )


def check_cuboids_when_present(ctx: NcoreV4Context) -> None:
    """If cuboid readers exist, verify observations are readable and tracks usable."""
    if ctx.cuboid_readers:
        _group("Cuboids")
        unreadable_cuboids: list[str] = []
        for component_instance_name, cuboid_reader in ctx.cuboid_readers.items():
            try:
                list(cuboid_reader.get_observations())
            except (KeyError, ValueError, OSError) as exc:
                unreadable_cuboids.append(f"{component_instance_name} ({exc})")
        _check(
            "observations readable",
            len(unreadable_cuboids) == 0,
            ctx.errors,
            f"cuboid read errors: {'; '.join(unreadable_cuboids)}",
        )
        warn_cuboid_tracks_below_two_observations(ctx)


def run_nurec_readiness_checks(ctx: NcoreV4Context) -> None:
    """Run the **NuRec Readiness** tier (sequence id, cameras, masks, lidar, optional cuboids)."""
    _section("NuRec Readiness")
    check_sequence_id_nurec_safe(ctx)
    check_cameras_nurec(ctx)
    check_masks_nurec(ctx)
    warn_masks_ego_key(ctx)
    check_lidar_nurec(ctx)
    check_cuboids_when_present(ctx)
