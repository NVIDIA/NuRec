# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiDAR data quality checks."""

from __future__ import annotations

import logging

import numpy as np
import torch

from ncore.data import RowOffsetStructuredSpinningLidarModelParameters
from ncore.data.v4 import LidarSensorComponent
from ncore.impl.common.transformations import transform_point_cloud
from ncore.sensors import RowOffsetStructuredSpinningLidarModel
from ncore_validation.utils import (
    NcoreV4Context,
    get_lidar_model_parameters_first,
)
from ncore_validation.validate_helpers import (
    COLUMN_AZIMUTHS_UNIFORM_STRIDE_RAD,
    FULL_ROTATION_COVERAGE_MIN_RAD,
    LIDAR_DIRECTION_TAUTOLOGY_THRESHOLD_DEG,
    MAX_LIDAR_DISTANCE_M,
    MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG,
    MIN_LIDAR_AZIMUTH_COVERAGE_RAD,
    MIN_LIDAR_FORWARD_RAY_ALIGNMENT,
    MIN_LIDAR_FORWARD_RAY_AXIS_RESIDUAL,
    MODEL_ELEMENT_ZERO_FRACTION_WARN,
    ROW_ELEVATIONS_UNIFORM_STRIDE_RAD,
    SPIN_FREQUENCY_MISMATCH_RATIO,
    _check,
    _check_sensor_frame_ts,
    _group,
    _sample_frame_indices,
    _warn,
)


log = logging.getLogger("validate")


def _check_direction_unit_norm(lidar_id: str, direction: np.ndarray) -> str | None:
    """Return an error string if direction vectors are not unit-norm, else ``None``.

    Uses the same squared-norm check as NCore's writer:
    ``np.abs(np.sum(direction**2, axis=1) - 1.0) < 1e-4``.

    Args:
        lidar_id: Sensor identifier used in the error message.
        direction: ``(N, 3)`` array of ray direction vectors.

    Returns:
        A human-readable error string when any direction fails the
        squared-norm tolerance, or ``None`` when all pass.
    """
    sq_norm_err = np.abs(np.sum(direction.astype(np.float64) ** 2, axis=1) - 1.0)
    if not np.all(sq_norm_err < 1e-4):
        worst = float(sq_norm_err.max())
        return f"{lidar_id} (max |sum(d²)-1| = {worst:.6f}, threshold 1e-4)"
    return None


def _check_distance_range(lidar_id: str, distance: np.ndarray) -> str | None:
    """Return an error string if distances are negative or exceed the max, else ``None``."""
    valid = distance[~np.isnan(distance)]
    if valid.size == 0:
        return None
    if valid.min() < 0:
        return f"{lidar_id} (negative distances)"
    if valid.max() > MAX_LIDAR_DISTANCE_M:
        return f"{lidar_id} (max {valid.max():.1f} m > {MAX_LIDAR_DISTANCE_M} m)"
    return None


def _check_intensity_range(
    lidar_id: str,
    lidar_reader: LidarSensorComponent.Reader,
    frame_end_ts: int,
) -> str | None:
    """Return an error string if intensities fall outside ``[0, 1]``, else ``None``."""
    try:
        intensity = lidar_reader.get_frame_ray_bundle_return_data(frame_end_ts, "intensity", return_index=0)
    except KeyError:
        return None
    if intensity.size == 0:
        return None
    if not np.all(np.isfinite(intensity)):
        return f"{lidar_id} (non-finite intensity values)"
    if intensity.min() < 0.0 or intensity.max() > 1.0:
        return f"{lidar_id} (range [{intensity.min():.4f}, {intensity.max():.4f}])"
    return None


def _check_point_timestamps(
    lidar_id: str,
    lidar_reader: LidarSensorComponent.Reader,
    frame_start_ts: int,
    frame_end_ts: int,
) -> str | None:
    """Return an error string if per-ray timestamps fall outside the frame window, else ``None``."""
    try:
        pt_timestamps = lidar_reader.get_frame_ray_bundle_data(frame_end_ts, "timestamp_us")
    except KeyError:
        return None
    if pt_timestamps.size == 0:
        return None
    if pt_timestamps.min() < frame_start_ts or pt_timestamps.max() > frame_end_ts:
        return (
            f"{lidar_id} (point ts [{pt_timestamps.min()}, {pt_timestamps.max()}] "
            f"vs frame [{frame_start_ts}, {frame_end_ts}])"
        )
    return None


def check_lidar_frame_data(
    ctx: NcoreV4Context,
    sample_frames: int | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Check direction unit-norm, distance range, intensity range, and point timestamps.

    Reads ``direction`` and ``distance_m`` once per frame and reuses the arrays
    for both data-quality checks and rig-frame point accumulation (returned for
    downstream use, e.g. camera-in-bbox and cuboid-LiDAR overlap).

    Stops checking a given sensor for a given property on first failure (but
    continues other properties, other sensors, and point accumulation).

    Args:
        ctx: NCore V4 context.
        sample_frames: When set, evaluate only ``sample_frames`` evenly-spaced
            frames per LiDAR (intended for fast iteration cycles).  ``None``
            (the default) iterates every frame; full coverage is required
            before treating a sequence as converged.

    Returns:
        Rig-frame point clouds keyed by LiDAR id, then by frame end
        timestamp (microseconds). Keyed per sensor so cross-sensor consumers
        can union coverage across sensors at an instant rather than picking
        whichever sensor's frame key lands nearest; see
        ``utils.LidarCoverageIndex``.
    """
    bad_norm: list[str] = []
    bad_dist: list[str] = []
    bad_intensity: list[str] = []
    bad_pt_ts: list[str] = []
    bad_shape: list[str] = []
    lidar_pts_rig: dict[str, dict[int, list[np.ndarray]]] = {}

    for lidar_id, lidar_reader in ctx.lidar_readers.items():
        ts = lidar_reader.frames_timestamps_us
        if ts.size == 0:
            continue
        norm_ok = dist_ok = intensity_ok = pt_ts_ok = shape_ok = True

        lidar_to_rig = None
        for pose_reader in ctx.poses_readers.values():
            try:
                lidar_to_rig = pose_reader.get_static_pose(lidar_id, "rig")
                break
            except KeyError:
                continue

        frame_indices = _sample_frame_indices(ts.shape[0], sample_frames)
        if sample_frames is not None and frame_indices.size < ts.shape[0]:
            _warn(
                f"{lidar_id}: sampled mode -- evaluating "
                f"{frame_indices.size}/{ts.shape[0]} frames in "
                f"check_lidar_frame_data; rerun without --sample-frames "
                f"before treating the sequence as converged."
            )
        for idx in frame_indices:
            start_ts, end_ts = int(ts[idx, 0]), int(ts[idx, 1])

            try:
                direction = lidar_reader.get_frame_ray_bundle_data(end_ts, "direction")
            except KeyError:
                direction = None
            try:
                distance = lidar_reader.get_frame_ray_bundle_return_data(end_ts, "distance_m", return_index=0)
            except KeyError:
                distance = None

            if norm_ok and direction is not None and direction.shape[0] > 0:
                if err := _check_direction_unit_norm(lidar_id, direction):
                    bad_norm.append(err)
                    norm_ok = False
            if dist_ok and distance is not None and distance.size > 0:
                if err := _check_distance_range(lidar_id, distance):
                    bad_dist.append(err)
                    dist_ok = False
            if intensity_ok and (err := _check_intensity_range(lidar_id, lidar_reader, end_ts)):
                bad_intensity.append(err)
                intensity_ok = False
            if pt_ts_ok and (err := _check_point_timestamps(lidar_id, lidar_reader, start_ts, end_ts)):
                bad_pt_ts.append(err)
                pt_ts_ok = False

            # A direction/distance_m ray-count mismatch is malformed data. Check
            # it independently of the rig-frame projection below so frames with
            # an empty distance_m or a sensor lacking a rig pose are still caught.
            # Record only the first mismatch per sensor (like bad_norm/bad_dist),
            # but skip projection for every mismatched frame.
            if direction is not None and distance is not None and direction.shape[0] != distance.size:
                if shape_ok:
                    bad_shape.append(
                        f"{lidar_id} frame {end_ts} direction has {direction.shape[0]} rays "
                        f"but distance_m has {distance.size}"
                    )
                    shape_ok = False
                continue

            if lidar_to_rig is not None and direction is not None and distance is not None and direction.shape[0] > 0:
                n_rays = direction.shape[0]
                xyz_sensor = direction[:n_rays] * distance[:n_rays, np.newaxis]
                xyz_rig = transform_point_cloud(xyz_sensor, lidar_to_rig)
                finite = np.isfinite(xyz_rig).all(axis=1)
                if finite.any():
                    lidar_pts_rig.setdefault(lidar_id, {}).setdefault(end_ts, []).append(xyz_rig[finite])

    _check(
        "direction vectors are unit-norm",
        not bad_norm,
        ctx.errors,
        f"LiDAR direction vectors not unit-norm: {'; '.join(bad_norm)}",
    )
    _check(
        f"distances non-negative and < {MAX_LIDAR_DISTANCE_M} m",
        not bad_dist,
        ctx.errors,
        f"LiDAR distance issues: {'; '.join(bad_dist)}",
    )
    _check(
        "intensity values in [0, 1]",
        not bad_intensity,
        ctx.errors,
        f"LiDAR intensity out of range: {'; '.join(bad_intensity)}",
    )
    _check(
        "per-point timestamps within frame bounds",
        not bad_pt_ts,
        ctx.errors,
        f"LiDAR per-point timestamps outside frame bounds: {'; '.join(bad_pt_ts)}",
    )
    _check(
        "ray bundle return shapes match ray count",
        not bad_shape,
        ctx.errors,
        f"LiDAR ray bundle shape issues: {'; '.join(bad_shape)}",
    )
    return {
        lidar_id: {ts: np.concatenate(arrs) for ts, arrs in per_sensor.items()}
        for lidar_id, per_sensor in lidar_pts_rig.items()
    }


def check_lidar_model_accuracy(
    ctx: NcoreV4Context,
    sample_frames: int | None = None,
) -> None:
    """Validate stored ray directions against the spinning LiDAR model.

    For each LiDAR with spinning intrinsics and ``model_element``, iterates
    frames, evaluates the model for the stored ``(row, col)`` indices,
    and computes the angular error between stored and model-predicted directions.
    Fails if the maximum error exceeds :data:`MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG`.

    Args:
        ctx: NCore V4 context.
        sample_frames: When set, evaluate only ``sample_frames`` evenly-spaced
            frames per LiDAR.  Intended for fast iteration cycles; rerun
            without it before treating a sequence as converged.  ``None``
            (the default) iterates every frame.
    """
    bad_accuracy: list[str] = []
    bad_model_eval: list[str] = []
    has_spinning_lidar = False
    for lidar_id, lidar_reader in ctx.lidar_readers.items():
        params = get_lidar_model_parameters_first(ctx.intrinsics_readers, lidar_id)
        if not isinstance(params, RowOffsetStructuredSpinningLidarModelParameters):
            continue
        has_spinning_lidar = True

        ts = lidar_reader.frames_timestamps_us
        if ts.size == 0:
            continue

        lidar_model = RowOffsetStructuredSpinningLidarModel(params, device="cpu")
        all_errors_deg: list[np.ndarray] = []
        unique_elements: set[tuple[int, int]] = set()
        frames_checked = 0
        model_eval_ok = True

        frame_indices = _sample_frame_indices(ts.shape[0], sample_frames)
        if sample_frames is not None and frame_indices.size < ts.shape[0]:
            _warn(
                f"{lidar_id}: sampled mode -- evaluating "
                f"{frame_indices.size}/{ts.shape[0]} frames in "
                f"check_lidar_model_accuracy; rerun without "
                f"--sample-frames before treating the sequence as converged."
            )
        for frame_idx in frame_indices:
            frame_end_ts = int(ts[frame_idx, 1])
            try:
                if not lidar_reader.has_frame_ray_bundle_data(frame_end_ts, "model_element"):
                    continue
                direction = lidar_reader.get_frame_ray_bundle_data(frame_end_ts, "direction")
                model_element = lidar_reader.get_frame_ray_bundle_data(frame_end_ts, "model_element")
            except Exception as e:
                log.debug(
                    "check_lidar_model_accuracy: failed frame %d for %s: %s",
                    frame_end_ts,
                    lidar_id,
                    e,
                )
                continue
            if direction.shape[0] == 0 or model_element.shape[0] == 0:
                continue
            n = min(direction.shape[0], model_element.shape[0])
            # Normalize stored directions so the dot product yields a true cosine.
            direction = direction[:n].astype(np.float64)
            norms = np.linalg.norm(direction, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            direction = direction / norms
            model_element = model_element[:n]

            # Evaluate the intrinsic model at each ray's (row, col) to get the
            # model-predicted direction.
            # Cast to int64: elements_to_sensor_rays internally converts to
            # torch.long but its uint16→int16 guard rejects values > 32767.
            try:
                model_dir = (
                    lidar_model.elements_to_sensor_rays(model_element.astype(np.int64)).cpu().numpy().astype(np.float64)
                )
            except Exception as e:
                # Record only the first eval failure per sensor (like bad_shape),
                # so one malformed sensor can't flood the report per-frame.
                if model_eval_ok:
                    bad_model_eval.append(
                        f"{lidar_id} frame {frame_end_ts} model_element cannot be evaluated "
                        f"(shape={model_element.shape}, error={type(e).__name__}: {e})"
                    )
                    model_eval_ok = False
                continue

            # Angular error = arccos(dot(stored, model)).
            dot = np.clip(np.einsum("ij,ij->i", direction, model_dir), -1.0, 1.0)
            errors_rad = np.arccos(dot)
            all_errors_deg.append(np.degrees(errors_rad))

            # Track which (row, col) elements were observed across all frames.
            for row, col in model_element:
                unique_elements.add((int(row), int(col)))
            frames_checked += 1

        if frames_checked == 0:
            continue

        errors = np.concatenate(all_errors_deg)
        max_err = float(errors.max())
        mean_err = float(errors.mean())
        coverage = len(unique_elements) / (params.n_rows * params.n_columns) * 100.0

        if max_err > MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG:
            bad_accuracy.append(
                f"{lidar_id} (max={max_err:.3f}\u00b0, mean={mean_err:.3f}\u00b0, "
                f"coverage={coverage:.1f}%, frames={frames_checked})"
            )

    if bad_accuracy or has_spinning_lidar:
        _check(
            f"model accuracy <= {MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG}\u00b0 (stored vs model directions)",
            len(bad_accuracy) == 0 and len(bad_model_eval) == 0,
            ctx.errors,
            "LiDAR model accuracy/evaluation issues: " + "; ".join(bad_accuracy + bad_model_eval),
        )


def check_lidar_model_fov_consistency(ctx: NcoreV4Context) -> None:
    """Validate that stored LiDAR model parameters pass NuRec's FOV self-consistency check.

    Instantiates :class:`RowOffsetStructuredSpinningLidarModel` with
    ``angles_to_columns_map_init=True`` -- the same code path NuRec's
    rendering pipeline executes.  If the model's beam angles fall outside
    the FOV rectangle computed by ``get_horizontal_fov``, NCore raises an
    ``AssertionError``.  Catching it here avoids a training-time crash.

    This is lightweight: one model instantiation per LiDAR sensor (not per
    frame), delegating entirely to NCore's own assertion logic.
    """
    bad_fov: list[str] = []
    has_spinning = False
    for lidar_id in ctx.lidar_readers:
        params = get_lidar_model_parameters_first(ctx.intrinsics_readers, lidar_id)
        if not isinstance(params, RowOffsetStructuredSpinningLidarModelParameters):
            continue
        has_spinning = True
        try:
            RowOffsetStructuredSpinningLidarModel(params, angles_to_columns_map_init=True, device="cpu")
        except (AssertionError, ValueError) as exc:
            bad_fov.append(
                f"LiDAR model FOV self-consistency check failed for '{lidar_id}': {exc}. "
                "NuRec's angles-to-columns inverse lookup map cannot be built for these "
                "model parameters. This typically occurs when row_azimuth_offsets_rad has "
                "large per-beam offsets (|peak-to-peak| exceeding the native column step), "
                "causing beam angles to fall outside the model's computed FOV rectangle. "
                "Fix: switch to a synthetic-uniform column_azimuths_rad schedule with "
                "per-beam offsets folded into the column index and "
                "row_azimuth_offsets_rad = 0 (the 'encode-in-column' strategy)."
            )
    if has_spinning:
        _check(
            "LiDAR model FOV self-consistency",
            len(bad_fov) == 0,
            ctx.errors,
            "; ".join(bad_fov) if bad_fov else None,
        )


def warn_lidar_direction_collapsed_to_model(
    ctx: NcoreV4Context,
    sample_frames: int | None = None,
) -> None:
    """Warn when stored direction is tautologically identical to model-derived direction.

    Fires when median and P99 angular error are both below
    ``LIDAR_DIRECTION_TAUTOLOGY_THRESHOLD_DEG`` *and*
    ``row_azimuth_offsets_rad`` is all-zero -- indicating that
    per-beam azimuth offsets were silently dropped during conversion.
    """
    if not (ctx.lidar_readers and ctx.intrinsics_readers):
        return

    threshold_rad = float(np.deg2rad(LIDAR_DIRECTION_TAUTOLOGY_THRESHOLD_DEG))
    for lidar_id, lidar_reader in ctx.lidar_readers.items():
        params = get_lidar_model_parameters_first(ctx.intrinsics_readers, lidar_id)
        if not isinstance(params, RowOffsetStructuredSpinningLidarModelParameters):
            continue
        if not np.allclose(np.asarray(params.row_azimuth_offsets_rad, dtype=np.float64), 0.0):
            continue

        ts = lidar_reader.frames_timestamps_us
        if ts.size == 0:
            continue

        # Use float64: the model's default float32 trig yields ~0.02 deg of ray
        # error, coarser than this check's ~0.01 deg detection threshold.
        lidar_model = RowOffsetStructuredSpinningLidarModel(params, device="cpu", dtype=torch.float64)

        all_errors_rad: list[np.ndarray] = []
        frames_checked = 0
        frame_indices = _sample_frame_indices(ts.shape[0], sample_frames)
        for frame_idx in frame_indices:
            frame_end_ts = int(ts[frame_idx, 1])
            try:
                if not lidar_reader.has_frame_ray_bundle_data(frame_end_ts, "model_element"):
                    continue
                direction = lidar_reader.get_frame_ray_bundle_data(frame_end_ts, "direction")
                model_element = lidar_reader.get_frame_ray_bundle_data(frame_end_ts, "model_element")
            except Exception as e:
                log.debug(
                    "warn_lidar_direction_collapsed: failed frame %d for %s: %s",
                    frame_end_ts,
                    lidar_id,
                    e,
                )
                continue
            if direction.shape[0] == 0 or model_element.shape[0] == 0:
                continue
            n = min(direction.shape[0], model_element.shape[0])
            stored = direction[:n].astype(np.float64)
            norms = np.linalg.norm(stored, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            stored = stored / norms

            # int64 sidesteps the model's uint16->int16 range guard.
            # elements_to_sensor_rays raises on malformed shape or out-of-range
            # (row, col) indices; skip the frame so one bad frame can't abort
            # this non-blocking warning.
            try:
                model_dir = (
                    lidar_model.elements_to_sensor_rays(model_element[:n].astype(np.int64))
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            except Exception as e:
                log.debug(
                    "warn_lidar_direction_collapsed: model eval failed frame %d for %s (model_element shape %s): %s",
                    frame_end_ts,
                    lidar_id,
                    model_element.shape,
                    e,
                )
                continue

            dot = np.clip(np.einsum("ij,ij->i", stored, model_dir), -1.0, 1.0)
            errors_rad = np.arccos(dot)
            all_errors_rad.append(errors_rad)
            frames_checked += 1

        if frames_checked == 0 or not all_errors_rad:
            continue

        errors = np.concatenate(all_errors_rad)
        median_rad = float(np.median(errors))
        p99_rad = float(np.quantile(errors, 0.99))
        max_rad = float(errors.max())

        if (
            median_rad < threshold_rad
            and p99_rad < threshold_rad
            and max_rad <= np.deg2rad(MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG)
        ):
            _warn(
                f"{lidar_id} stored direction matches model-derived direction to "
                f"median={np.degrees(median_rad):.2e}\u00b0, "
                f"p99={np.degrees(p99_rad):.2e}\u00b0 (< "
                f"{LIDAR_DIRECTION_TAUTOLOGY_THRESHOLD_DEG}\u00b0) with "
                f"row_azimuth_offsets_rad = 0 across {frames_checked} frames. "
                "Likely the converter stored model-derived (Case 1) directions "
                "on a rotor-phase-only column_azimuths_rad with zero per-beam "
                "offsets -- the LiDAR-smearing failure mode where per-beam "
                "azimuth offsets \u0394 are silently dropped. "
                "Switch the converter to XYZ-derived direction "
                "(direction = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)) "
                "and choose a row_azimuth_offsets_rad strategy: "
                "sensor-native column schedule + preserve \u0394 on "
                "row_azimuth_offsets_rad (when manufacturer offsets exist), "
                "sensor-native column schedule with \u0394 encoded in the column "
                "index (when the rotor schedule is dense enough to resolve "
                "\u0394), or synthetic-uniform column schedule with \u0394 encoded "
                "in the column index (fallback when the native rotor schedule "
                "is unrecoverable: rotor-phase-only stream, dual-return run "
                "patterns, or |\u0394| exceeds the native column step). "
                "This is a converter-quality heuristic; NCore allows any "
                "unit-norm stored direction."
            )


def _lidar_column_rays_rig(
    lidar_id: str,
    params: RowOffsetStructuredSpinningLidarModelParameters,
    T_lidar_rig: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(d_model, d_rig)`` for the central elevation row across every column.

    ``d_model[c]`` is the unit-norm ray direction predicted by the spinning
    intrinsic model at ``(row=n_rows//2, col=c)`` in the LiDAR's model frame.
    ``d_rig[c] = T_lidar_rig.R @ d_model[c]`` is the same ray in the rig
    frame.  Returns ``None`` on model-evaluation failure.
    """
    n_rows = int(params.n_rows)
    n_columns = int(params.n_columns)
    if n_rows <= 0 or n_columns <= 0:
        return None
    e0 = n_rows // 2
    rows = np.full(n_columns, e0, dtype=np.int64)
    cols = np.arange(n_columns, dtype=np.int64)
    elements = np.stack([rows, cols], axis=-1)
    try:
        lidar_model = RowOffsetStructuredSpinningLidarModel(params, device="cpu")
        d_model = lidar_model.elements_to_sensor_rays(elements).cpu().numpy().astype(np.float64)
    except Exception as e:  # noqa: BLE001 - model evaluation failure is logged below.
        log.debug(f"{lidar_id}: elements_to_sensor_rays failed: {e}")
        return None
    # Defensive normalization; model should already return unit rays.
    norms = np.linalg.norm(d_model, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    d_model = d_model / norms
    d_rig = d_model @ T_lidar_rig[:3, :3].astype(np.float64).T
    return d_model, d_rig


def _lidar_azimuth_coverage_rad(
    params: RowOffsetStructuredSpinningLidarModelParameters,
) -> float:
    """Return the angular span of ``column_azimuths_rad`` (radians, wrap-aware)."""
    az = np.mod(params.column_azimuths_rad.astype(np.float64), 2.0 * np.pi)
    if az.size < 2:
        return 0.0
    az_sorted = np.sort(az)
    wraps = np.concatenate([az_sorted, az_sorted[:1] + 2.0 * np.pi])
    max_gap = float(np.max(np.diff(wraps)))
    return 2.0 * np.pi - max_gap


# Cardinal model-frame azimuths sampled by ``check_lidar_forward_ray_in_rig``.
# Under the canonical model-frame choice (column-0 ray aligned with rig +X), each
# cardinal model azimuth alpha must project to the rig-frame direction
# ``[cos(alpha), sin(alpha), 0]``.  Spacing the samples ~90 deg apart around the
# rotor catches a wholesale residual yaw between ``T_lidar_rig.R`` and
# ``column_azimuths_rad`` -- the single-column ``best_dot`` check is necessary
# but not sufficient (a 360-deg spinner always has *some* column landing near
# rig +X regardless of how the cloud is rotated).
_CARDINAL_MODEL_AZIMUTHS_RAD = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], dtype=np.float64)


def _spinning_lidar_forward_ray_summary(ctx: NcoreV4Context, lidar_id: str) -> dict | None:
    """Compute the rig-frame forward-ray alignment summary for one LiDAR.

    Returns a dict with ``best_dot``, ``best_col``, ``best_d_rig``,
    ``n_columns``, ``azimuth_coverage_rad``, and ``cardinals`` (a list of
    per-cardinal-azimuth records, see below) on success, or ``None`` when
    the LiDAR has no spinning model, no rig extrinsic, or insufficient
    azimuth coverage.

    Each ``cardinals`` entry is::

        {
            "target_model_az_rad": <float>,       # one of 0, pi/2, pi, 3*pi/2
            "col": <int>,                         # column whose model azimuth is
                                                  # closest to target_model_az_rad
            "model_az_rad": <float>,              # actual model azimuth at col
            "d_rig": <ndarray shape=(3,)>,        # T_lidar_rig.R @ d_model[col]
            "expected_rig_dir": <ndarray (3,)>,   # [cos(target), sin(target), 0]
            "dot": <float>,                       # dot(d_rig, expected_rig_dir)
        }
    """
    params = get_lidar_model_parameters_first(ctx.intrinsics_readers, lidar_id)
    if not isinstance(params, RowOffsetStructuredSpinningLidarModelParameters):
        return None

    coverage = _lidar_azimuth_coverage_rad(params)
    if coverage < MIN_LIDAR_AZIMUTH_COVERAGE_RAD:
        return None

    T_lidar_rig: np.ndarray | None = None
    for pose_reader in ctx.poses_readers.values():
        try:
            T_lidar_rig = pose_reader.get_static_pose(lidar_id, "rig").astype(np.float64)
            break
        except KeyError:
            continue
    if T_lidar_rig is None:
        return None

    rays = _lidar_column_rays_rig(lidar_id, params, T_lidar_rig)
    if rays is None:
        return None
    _d_model, d_rig = rays
    dots = d_rig @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    best_idx = int(np.argmax(dots))

    column_az = np.mod(params.column_azimuths_rad.astype(np.float64), 2.0 * np.pi)
    cardinals: list[dict] = []
    for target in _CARDINAL_MODEL_AZIMUTHS_RAD:
        delta = np.mod(column_az - target + np.pi, 2.0 * np.pi) - np.pi
        col = int(np.argmin(np.abs(delta)))
        expected = np.array([np.cos(float(target)), np.sin(float(target)), 0.0], dtype=np.float64)
        d = d_rig[col]
        cardinals.append(
            {
                "target_model_az_rad": float(target),
                "col": col,
                "model_az_rad": float(column_az[col]),
                "d_rig": d,
                "expected_rig_dir": expected,
                "dot": float(np.dot(d, expected)),
            }
        )

    return {
        "best_dot": float(dots[best_idx]),
        "best_col": best_idx,
        "best_d_rig": d_rig[best_idx],
        "n_columns": int(params.n_columns),
        "azimuth_coverage_rad": coverage,
        "cardinals": cardinals,
    }


def check_lidar_forward_ray_in_rig(ctx: NcoreV4Context) -> None:
    """Hard-error when a spinning LiDAR's best column ray does not point near +X_rig.

    For every spinning LiDAR with azimuth coverage >=
    ``MIN_LIDAR_AZIMUTH_COVERAGE_RAD`` and a ``<lidar>->rig`` static pose,
    ``max_c dot(T_lidar_rig.R @ d_model(e0, c), [1, 0, 0])`` must be >=
    ``MIN_LIDAR_FORWARD_RAY_ALIGNMENT``.  This catches the extreme case
    where no column anywhere lands near rig +X (e.g.
    ``T_lidar_rig`` set up as a near-90 deg rotation, or a totally-wrong
    extrinsic).

    A wholesale residual yaw between ``T_lidar_rig.R`` and
    ``column_azimuths_rad`` (some column always lands near rig +X but the
    cloud as a whole is rotated by a constant offset) is **not** caught
    here -- the single-column ``best_dot`` is high regardless.  See
    :func:`warn_lidar_forward_ray_rig_axis_residual` for the cardinal-spread
    sub-hard warning that surfaces this case at the LiDAR-only stage, and
    ``check_cuboid_lidar_overlap`` for the authoritative cross-sensor check.

    Other LiDARs are skipped.
    """
    if not (ctx.lidar_readers and ctx.poses_readers and ctx.intrinsics_readers):
        return

    bad: list[str] = []
    evaluated_any = False
    for lidar_id in ctx.lidar_readers:
        summary = _spinning_lidar_forward_ray_summary(ctx, lidar_id)
        if summary is None:
            continue
        evaluated_any = True
        if summary["best_dot"] < MIN_LIDAR_FORWARD_RAY_ALIGNMENT:
            d = summary["best_d_rig"]
            bad.append(
                f"{lidar_id} (best dot={summary['best_dot']:.3f}, "
                f"col={summary['best_col']}/{summary['n_columns']}, "
                f"d_rig=[{d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f}])"
            )

    if not evaluated_any:
        return

    _check(
        f"spinning LiDAR best forward ray in rig (dot >= {MIN_LIDAR_FORWARD_RAY_ALIGNMENT:.2f})",
        not bad,
        ctx.errors,
        "spinning LiDAR(s) have no column ray aligned with rig +X: "
        f"{'; '.join(bad)}. Native-to-rig basis change is likely "
        "double-counted across T_lidar_rig.R and column_azimuths_rad. "
        "Pick ONE place to carry the native-to-rig yaw.",
    )


def warn_lidar_forward_ray_rig_axis_residual(ctx: NcoreV4Context) -> None:
    """Warn when the cardinal model azimuths don't project to the corresponding
    rig-frame axes -- companion to :func:`check_lidar_forward_ray_in_rig`.

    For every spinning LiDAR with sufficient azimuth coverage, sample four
    columns whose ``column_azimuths_rad`` are closest to the cardinal model
    azimuths ``{0, pi/2, pi, 3*pi/2}`` (~90 deg apart around the rotor) and
    compute ``dot(T_lidar_rig.R @ d_model[col], [cos(target), sin(target), 0])``
    for each.  Warn when the worst dot is below
    ``MIN_LIDAR_FORWARD_RAY_AXIS_RESIDUAL`` (~14 deg).  Two failure modes
    surface here:

    * **Small residual yaw / mount-bracket double-count**: worst cardinal in
      ``[MIN_LIDAR_FORWARD_RAY_ALIGNMENT, MIN_LIDAR_FORWARD_RAY_AXIS_RESIDUAL)``.
    * **Wholesale residual yaw** (model frame is rotated away from the rig as
      a whole; ``check_lidar_forward_ray_in_rig`` still passes because *some*
      column always lands near rig +X on a 360 deg spinner): worst cardinal
      far below ``MIN_LIDAR_FORWARD_RAY_ALIGNMENT``.

    The warning also fires for legitimate Option-B converters (column-0 ray
    aligned with sensor-native +X rather than rig +X) -- the cross-sensor
    ``check_cuboid_lidar_overlap`` is the authoritative arbiter; this check
    is informational on top of the LiDAR-only data.
    """
    if not (ctx.lidar_readers and ctx.poses_readers and ctx.intrinsics_readers):
        return

    for lidar_id in ctx.lidar_readers:
        summary = _spinning_lidar_forward_ray_summary(ctx, lidar_id)
        if summary is None:
            continue
        cardinals = summary["cardinals"]
        worst = min(cardinals, key=lambda c: c["dot"])
        if worst["dot"] >= MIN_LIDAR_FORWARD_RAY_AXIS_RESIDUAL:
            continue
        d = worst["d_rig"]
        _warn(
            f"{lidar_id} worst cardinal forward-ray dot is "
            f"{worst['dot']:+.3f} (< {MIN_LIDAR_FORWARD_RAY_AXIS_RESIDUAL:.2f}) "
            f"at target model azimuth "
            f"{np.rad2deg(worst['target_model_az_rad']):+.0f} deg "
            f"(col={worst['col']}/{summary['n_columns']}, "
            f"d_rig=[{d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f}]). "
            "Either (a) a residual rotation between T_lidar_rig.R and "
            "column_azimuths_rad (mount-bracket double-count), (b) a "
            "wholesale yaw between model and rig frames, OR (c) a legitimate "
            "Option-B converter (spinning_lidar_model_frame_choice: "
            "option_b_native_aligned) where column-0 ray points along "
            "sensor-native +X rather than rig +X. The cross-sensor "
            "check_cuboid_lidar_overlap is the authoritative arbiter."
        )


def warn_lidar_model_element_all_zero(ctx: NcoreV4Context) -> None:
    """Warn when stored ``model_element`` is (near-)uniformly ``(0, 0)``.

    A common converter mistake is leaving ``model_element`` as an
    ``np.zeros((N, 2), uint16)`` placeholder.  Downstream consumers that
    reconstruct rays from ``(row, col)`` indices (including NuRec) then
    collapse every point to the first grid cell.  :func:`check_lidar_model_accuracy`
    can still pass if every stored direction happens to align with cell
    ``(0, 0)``, so this warning catches the case explicitly.
    """
    if not (ctx.lidar_readers and ctx.intrinsics_readers):
        return

    for lidar_id, lidar_reader in ctx.lidar_readers.items():
        params = get_lidar_model_parameters_first(ctx.intrinsics_readers, lidar_id)
        if not isinstance(params, RowOffsetStructuredSpinningLidarModelParameters):
            continue

        ts = lidar_reader.frames_timestamps_us
        if ts.size == 0:
            continue

        total_points = 0
        total_zeros = 0
        frames_checked = 0
        for frame_idx in range(ts.shape[0]):
            frame_end_ts = int(ts[frame_idx, 1])
            try:
                if not lidar_reader.has_frame_ray_bundle_data(frame_end_ts, "model_element"):
                    continue
                model_element = lidar_reader.get_frame_ray_bundle_data(frame_end_ts, "model_element")
            except Exception as e:
                log.debug(
                    "warn_lidar_model_element_all_zero: failed frame %d for %s: %s",
                    frame_end_ts,
                    lidar_id,
                    e,
                )
                continue
            if model_element.shape[0] == 0:
                continue
            total_points += model_element.shape[0]
            total_zeros += int(np.sum((model_element[:, 0] == 0) & (model_element[:, 1] == 0)))
            frames_checked += 1

        if frames_checked == 0 or total_points == 0:
            continue

        zero_fraction = total_zeros / total_points
        if zero_fraction >= MODEL_ELEMENT_ZERO_FRACTION_WARN:
            _warn(
                f"{lidar_id} model_element is (0, 0) for "
                f"{zero_fraction * 100:.1f}% of rays across {frames_checked} frames "
                f"(>= {MODEL_ELEMENT_ZERO_FRACTION_WARN * 100:.0f}%). "
                "Likely an uninitialized np.zeros placeholder; populate "
                "real (row, col) indices from calibration or per-ray metadata."
            )


def warn_lidar_row_elevations_uniform(ctx: NcoreV4Context) -> None:
    """Warn when ``row_elevations_rad`` is (nearly) uniformly spaced.

    Real spinning LiDARs have per-beam elevation tables with non-uniform
    spacing; a uniform grid (``np.linspace(vfov_min, vfov_max, n_rows)``)
    almost always means the per-beam calibration was not consulted and
    the converter fell back to synthesis.  Synthesis of
    ``row_elevations_rad`` silently discards per-beam firing-time
    offsets and elevation accuracy -- re-read the raw calibration and
    replace the grid with the real per-beam table.
    """
    if not (ctx.lidar_readers and ctx.intrinsics_readers):
        return

    for lidar_id in ctx.lidar_readers:
        params = get_lidar_model_parameters_first(ctx.intrinsics_readers, lidar_id)
        if not isinstance(params, RowOffsetStructuredSpinningLidarModelParameters):
            continue
        elev = np.asarray(params.row_elevations_rad, dtype=np.float64)
        if elev.size < 3:
            continue
        stride_std = float(np.std(np.diff(elev)))
        if stride_std < ROW_ELEVATIONS_UNIFORM_STRIDE_RAD:
            _warn(
                f"{lidar_id} row_elevations_rad look uniformly spaced "
                f"(std of diffs = {stride_std:.2e} rad "
                f"< {ROW_ELEVATIONS_UNIFORM_STRIDE_RAD:.0e}). "
                "Real spinning LiDARs have non-uniform per-beam "
                "elevations; re-read the raw calibration and replace "
                "the synthetic grid with the real per-beam table."
            )


def warn_lidar_column_azimuths_uniform(ctx: NcoreV4Context) -> None:
    """Warn when ``column_azimuths_rad`` is (nearly) uniformly spaced.

    Real free-spinning LiDARs ship per-frame azimuth schedules whose
    unique-azimuth set has a non-uniform per-column stride (dual-return
    run patterns, near-field channel bursts, non-uniform schedules); a
    uniform grid (``np.linspace(0, 2*pi, n_columns, endpoint=False)``)
    almost always means the per-frame azimuth stream was not consulted
    and the converter fell back to synthesis.  Synthesis of
    ``column_azimuths_rad`` is a synthesis artifact in exactly the same
    sense as a synthetic ``row_elevations_rad``: it discards the
    sensor's actual per-column firing structure and replaces it with a
    fiction that happens to share an average angular resolution.
    Re-read the per-frame azimuth stream and extract the real schedule
    (see ``build_lidar_model_from_per_frame_azimuth`` in
    ``ncore_template/ncore_example/utils.py``).

    Phase-locked rotors with a near-uniform native schedule will trigger
    this warning; that is acceptable -- the warning is advisory, and a
    converter that derived the schedule from real per-frame data can
    safely ignore it.
    """
    if not (ctx.lidar_readers and ctx.intrinsics_readers):
        return

    for lidar_id in ctx.lidar_readers:
        params = get_lidar_model_parameters_first(ctx.intrinsics_readers, lidar_id)
        if not isinstance(params, RowOffsetStructuredSpinningLidarModelParameters):
            continue
        az = np.asarray(params.column_azimuths_rad, dtype=np.float64)
        if az.size < 3:
            continue
        diffs = np.diff(az)
        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi)
        stride_std = float(np.std(diffs))
        if stride_std < COLUMN_AZIMUTHS_UNIFORM_STRIDE_RAD:
            _warn(
                f"{lidar_id} column_azimuths_rad look uniformly spaced "
                f"(std of diffs = {stride_std:.2e} rad "
                f"< {COLUMN_AZIMUTHS_UNIFORM_STRIDE_RAD:.0e}). "
                "Real free-spinning LiDARs have non-uniform per-column "
                "schedules; re-read the per-frame azimuth stream and "
                "extract the real schedule via "
                "build_lidar_model_from_per_frame_azimuth (in the "
                "converter's utils.py) or an equivalent reference-frame "
                "extraction."
            )


def warn_lidar_spin_direction_data_mismatch(ctx: NcoreV4Context) -> None:
    """Warn when ``spinning_direction`` disagrees with the per-column data.

    NCore's writer asserts that ``relative_angle(azim[0], azim,
    spinning_direction)`` is strictly increasing, so a hard mismatch
    crashes at write time with a generic assertion. This warning runs
    the same check after the fact and produces an actionable, direction-
    specific message: increasing ``column_azimuths_rad`` (mod 2 pi)
    pairs with ``"ccw"``; decreasing pairs with ``"cw"``. The check is
    a defense-in-depth signal -- it should never fire on valid stored
    data, but if it does the message tells the converter which value to
    write rather than leaving it to interpret a verbal label.
    """
    if not ctx.intrinsics_readers:
        return

    two_pi = 2.0 * np.pi
    for lidar_id in ctx.lidar_readers or []:
        params = get_lidar_model_parameters_first(ctx.intrinsics_readers, lidar_id)
        if not isinstance(params, RowOffsetStructuredSpinningLidarModelParameters):
            continue
        az = np.mod(np.asarray(params.column_azimuths_rad, dtype=np.float64), two_pi)
        if az.size < 2:
            continue
        # NCore's relative_angle semantics, copied here to avoid an NCore import.
        rel_ccw = np.mod(az - az[0], two_pi)
        rel_cw = np.mod(az[0] - az, two_pi)
        ccw_ok = bool(np.all(np.diff(rel_ccw) > 0))
        cw_ok = bool(np.all(np.diff(rel_cw) > 0))
        implied: str | None = None
        if ccw_ok and not cw_ok:
            implied = "ccw"
        elif cw_ok and not ccw_ok:
            implied = "cw"
        # If neither, the schedule is not a single continuous rotation;
        # NCore's own assertion will surface that and there is no single
        # correct direction to recommend, so skip.
        if implied is None:
            continue
        if implied != params.spinning_direction:
            _warn(
                f"{lidar_id} spinning_direction='{params.spinning_direction}' "
                f"disagrees with column_azimuths_rad monotonicity. "
                f"The stored schedule is monotonically "
                f"{'increasing' if implied == 'ccw' else 'decreasing'} "
                f"(mod 2 pi), which pairs with spinning_direction='{implied}'. "
                "Set spinning_direction from the data, not from a verbal "
                "'CW'/'CCW' label."
            )


def warn_lidar_spin_frequency_mismatch(ctx: NcoreV4Context) -> None:
    """Warn when ``spinning_frequency_hz`` disagrees with the frame timestamps.

    When ``column_azimuths_rad`` spans a full rotation, one stored frame is
    one revolution, so the frame window (``frame_end - frame_start``) fixes
    the rotor rate at ``1 / frame duration``.  Nothing else in the shard
    constrains the field, so a frequency derived any other way reaches NuRec
    unchallenged.

    Only the frame window gates the warning.  Frame-to-frame cadence is
    reported alongside it but cannot be a gate: a dataset that publishes
    every other revolution has a cadence of half the rotor rate while both
    the declared frequency and the frame window are correct.

    Skipped below ``FULL_ROTATION_COVERAGE_MIN_RAD`` of azimuth coverage,
    where a frame is a fraction of a revolution and the frame window is
    legitimately shorter than one turn.
    """
    if not (ctx.lidar_readers and ctx.intrinsics_readers):
        return

    for lidar_id, lidar_reader in ctx.lidar_readers.items():
        params = get_lidar_model_parameters_first(ctx.intrinsics_readers, lidar_id)
        if not isinstance(params, RowOffsetStructuredSpinningLidarModelParameters):
            continue
        if _lidar_azimuth_coverage_rad(params) < FULL_ROTATION_COVERAGE_MIN_RAD:
            continue
        declared_hz = float(params.spinning_frequency_hz)
        if declared_hz <= 0.0:
            continue
        ts = np.asarray(lidar_reader.frames_timestamps_us, dtype=np.float64)
        if ts.ndim != 2 or ts.shape[0] == 0:
            continue

        sweep_us = float(np.median(ts[:, 1] - ts[:, 0]))
        if sweep_us <= 0:
            continue
        window_hz = 1e6 / sweep_us
        if abs(window_hz - declared_hz) / window_hz < SPIN_FREQUENCY_MISMATCH_RATIO:
            continue

        cadence_note = ""
        if ts.shape[0] >= 2:
            cadence_us = float(np.median(np.diff(ts[:, 1])))
            if cadence_us > 0:
                cadence_note = f" Frames arrive at {1e6 / cadence_us:.2f} Hz."
        _warn(
            f"{lidar_id} spinning_frequency_hz={declared_hz:.2f} disagrees with the stored "
            f"frame windows (median {sweep_us / 1000.0:.1f} ms implies {window_hz:.2f} Hz; "
            f"tolerance {SPIN_FREQUENCY_MISMATCH_RATIO:.0%}). A full-rotation frame spans one "
            f"revolution, so either the declared rate is wrong or the window is.{cadence_note} "
            "Derive the rate from the frame window rather than from a fit over per-point "
            "timestamps, whose epoch may be frame-relative."
        )


def check_lidar_data_quality_group(
    ctx: NcoreV4Context,
    sample_frames: int | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Run all LiDAR data-quality checks under one ``LiDAR`` sub-group.

    Args:
        ctx: NCore V4 context.
        sample_frames: When set, the iteratively-expensive sub-checks
            (:func:`check_lidar_frame_data`,
            :func:`check_lidar_model_accuracy`) evaluate only
            ``sample_frames`` evenly-spaced frames per LiDAR.  Intended
            for fast iteration cycles; cheap structural / parameter
            checks (uniform-grid warnings, forward-ray axis check, etc.)
            always run on the full set and are unaffected.  Default
            ``None`` runs every frame.

    Returns:
        Rig-frame point clouds keyed by frame end timestamp (microseconds).
    """
    if not ctx.lidar_readers:
        return {}
    _group("LiDAR")
    _check_sensor_frame_ts(ctx.lidar_readers, "LiDAR", ctx.errors)
    lidar_pts_rig = check_lidar_frame_data(ctx, sample_frames=sample_frames)
    check_lidar_model_accuracy(ctx, sample_frames=sample_frames)
    check_lidar_model_fov_consistency(ctx)
    warn_lidar_direction_collapsed_to_model(ctx, sample_frames=sample_frames)
    warn_lidar_model_element_all_zero(ctx)
    warn_lidar_row_elevations_uniform(ctx)
    warn_lidar_column_azimuths_uniform(ctx)
    warn_lidar_spin_direction_data_mismatch(ctx)
    warn_lidar_spin_frequency_mismatch(ctx)
    check_lidar_forward_ray_in_rig(ctx)
    warn_lidar_forward_ray_rig_axis_residual(ctx)
    return lidar_pts_rig


def warn_low_intensity_dynamic_range(ctx: NcoreV4Context) -> None:
    """Warn when LiDAR intensity values have very low dynamic range.

    If the 99th-percentile intensity across sampled LiDAR frames is below
    0.01, the converter likely used linear normalization (raw / large_constant)
    instead of a range-preserving method like ``np.tanh`` for raw intensities
    that are small positive floats.
    """
    if not ctx.lidar_readers:
        return

    _INTENSITY_P99_THRESHOLD = 0.01
    _MAX_FRAMES_TO_SAMPLE = 10

    all_intensities: list[np.ndarray] = []
    for _lidar_id, lidar_reader in ctx.lidar_readers.items():
        ts = lidar_reader.frames_timestamps_us
        if ts.size == 0:
            continue
        n_frames = min(ts.shape[0], _MAX_FRAMES_TO_SAMPLE)
        indices = np.linspace(0, ts.shape[0] - 1, n_frames, dtype=int)
        for idx in indices:
            frame_end_ts = int(ts[idx, 1])
            try:
                intensity = lidar_reader.get_frame_ray_bundle_return_data(
                    frame_end_ts,
                    "intensity",
                    return_index=0,
                )
            except Exception as e:
                log.debug(
                    "warn_low_intensity_dynamic_range: failed frame %d: %s",
                    frame_end_ts,
                    e,
                )
                continue
            if intensity is not None and intensity.size > 0:
                finite = intensity[np.isfinite(intensity)]
                if finite.size > 0:
                    all_intensities.append(finite)

    if not all_intensities:
        return

    combined = np.concatenate(all_intensities)
    p99 = float(np.percentile(combined, 99))

    if p99 < _INTENSITY_P99_THRESHOLD:
        _warn(
            f"LiDAR intensity values have very low dynamic range "
            f"(p99 = {p99:.6f}). Raw intensity may have been divided by an "
            f"overly large constant instead of using a range-preserving "
            f"normalization like np.tanh. "
            f"NCore intensity should use the full [0, 1] range."
        )
