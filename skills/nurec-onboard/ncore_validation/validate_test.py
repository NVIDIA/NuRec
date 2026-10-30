# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for selected validation checks.

Coverage:

* **warn_lidar_direction_collapsed_to_model** -- tautology detection
  for model-derived vs XYZ-derived LiDAR directions.
* **check_rig_forward_alignment** -- hard-error when rig +X doesn't
  align with travel direction.
* **warn_sequence_envelope_coverage** -- sensor/cuboid envelope
  coverage relative to the sequence interval.
* **check_cuboid_lidar_overlap** -- near-range cuboid hit rate.
* **check_cuboid_obb_inlier_ratio** -- AABB-to-OBB inlier ratio.
* **check_cuboid_pose_gap** -- pose waypoint coverage at cuboid reference
  timestamps, per reference frame.
* **warn_cuboid_ground_clearance** -- floating cuboid detection.
* **check_lidar_model_fov_consistency** -- NuRec FOV self-consistency
  via angles-to-columns map initialization.
* **warn_lidar_spin_frequency_mismatch** -- declared ``spinning_frequency_hz``
  against the rate implied by the stored frame windows, including the
  sector-coverage skip and the decimated-cadence false positive.
* **warn_camera_fisheye_max_angle_deviation** -- stored vs computed
  max_angle for OpenCV-fisheye cameras.
* **nearest_idx** -- nearest-neighbor index lookup helper.
* **_check_sensor_frame_ts** -- frame timestamp invariant validation.
* **warn_camera_timestamps_all_identical** -- identical-timestamp
  warning with empty-camera guard.
* **warn_pose_coverage** -- pose coverage warning across multiple readers.
* **check_sequence_id_nurec_safe** -- sequence id usable as a store
  filename and auxiliary-data discovery prefix.
* **check_masks_nurec** -- masks component presence (hard error).
* **warn_masks_ego_key** -- ego mask key naming warnings.
* **warn_identity_sensor_translations** -- all-near-zero sensor-to-rig
  translation soft warning.
* **warn_camera_optical_axis_tilt** -- camera optical axis far from
  horizontal in rig coords (missing source-frame basis change).
* **warn_camera_optical_axis_azimuth** -- camera optical axis azimuth in
  rig coords disagrees with id-implied mounting direction (whole-rig yaw
  bug in T_camera_rig).
* **ResultCollector lifecycle** -- ``start_collection`` /
  ``stop_collection`` activation, replacement, and post-stop helper
  no-op behavior.
* **Collector wiring** -- ``_section`` / ``_group`` / ``_check`` /
  ``_warn`` recording under the active tier/group context, including
  default-message fallback and cross-tier emission order.
* **dump_validate_result round-trip** -- on-disk YAML field order
  matches ``ValidateResult`` declaration order; nested dataclasses
  convert recursively; empty collections survive.
* **check_poses_rig_world_in_interval** -- hard error when rig->world
  dynamic poses are absent or fall outside the sequence interval.
* **LiDAR malformed-data hard errors** -- invalid model elements and
  ray/return shape mismatches are reported instead of silently skipped.
"""

from __future__ import annotations

import logging
import tempfile
import types
import unittest

from pathlib import Path
from typing import Any, cast
from unittest import mock

import numpy as np
import yaml

from scipy.spatial.transform import Rotation
from validators._validation_base import VALIDATE_RESULT_PROGRAM, VALIDATE_RESULT_SCHEMA_VERSION

from ncore.data import (
    BBox3,
    CuboidTrackObservation,
    LabelSource,
    OpenCVFisheyeCameraModelParameters,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)
from ncore.impl.common.transformations import (
    HalfClosedInterval,
)  # no public re-export yet
from ncore_validation.utils import WORLD_ANCHORED_CUBOID_FRAMES, LidarCoverageIndex, NcoreV4Context, sample_evenly
from ncore_validation.validate_cameras import (
    _parse_camera_direction_azimuth_rad,
    warn_camera_fisheye_max_angle_deviation,
    warn_camera_optical_axis_azimuth,
    warn_camera_optical_axis_tilt,
    warn_camera_timestamps_all_identical,
)
from ncore_validation.validate_cross_sensor import warn_sequence_envelope_coverage
from ncore_validation.validate_cuboids import (
    check_cuboid_duplicate_observation_timestamps,
    check_cuboid_lidar_overlap,
    check_cuboid_obb_inlier_ratio,
    check_cuboid_pose_gap,
    nearest_idx,
    warn_cuboid_cross_source_timestamp_collisions,
    warn_cuboid_ground_clearance,
    warn_cuboid_tracks_below_two_observations,
)
from ncore_validation.validate_helpers import (
    FailedCheck,
    ResultCollector,
    ValidateResult,
    WarningResult,
    _check,
    _check_sensor_frame_ts,
    _group,
    _section,
    _warn,
    dump_validate_result,
    start_collection,
    stop_collection,
)
from ncore_validation.validate_lidar import (
    check_lidar_frame_data,
    check_lidar_model_accuracy,
    check_lidar_model_fov_consistency,
    warn_lidar_direction_collapsed_to_model,
    warn_lidar_spin_frequency_mismatch,
)
from ncore_validation.validate_nurec import check_masks_nurec, check_sequence_id_nurec_safe, warn_masks_ego_key
from ncore_validation.validate_poses import (
    check_rig_forward_alignment,
    warn_identity_sensor_translations,
)
from ncore_validation.validate_structure import (
    check_poses_rig_world_in_interval,
    warn_pose_coverage,
)


_validate_log = logging.getLogger("validate")


def _fake_ctx(**fields: Any) -> NcoreV4Context:
    """Build a stand-in context carrying only the fields the check under test reads."""
    return cast(NcoreV4Context, types.SimpleNamespace(**fields))


class _capture_logs:
    """Context manager that captures INFO+ records from ``_validate_log``.

    Usage::

        with _capture_logs() as records:
            some_check(ctx)
        msgs = [r.getMessage() for r in records]
    """

    def __init__(self) -> None:
        self.records: list[logging.LogRecord] = []

    def __enter__(self) -> list[logging.LogRecord]:
        self._prev_level = _validate_log.level
        _validate_log.setLevel(logging.INFO)

        class _Handler(logging.Handler):
            def emit(handler_self, record: logging.LogRecord) -> None:  # noqa: N805
                self.records.append(record)

        self._handler = _Handler(level=logging.INFO)
        _validate_log.addHandler(self._handler)
        return self.records

    def __exit__(self, *exc: object) -> None:
        _validate_log.removeHandler(self._handler)
        _validate_log.setLevel(self._prev_level)


_N_ROWS = 32
_N_COLS = 900


def _make_params(
    row_azimuth_offsets_rad: np.ndarray | None = None,
    spinning_frequency_hz: float = 10.0,
    coverage_rad: float = 2.0 * np.pi,
) -> RowOffsetStructuredSpinningLidarModelParameters:
    """Build spinning-LiDAR parameters spanning *coverage_rad* of azimuth.

    Args:
        row_azimuth_offsets_rad: Per-row offsets; defaults to zeros.
        spinning_frequency_hz: Declared rotor rate.
        coverage_rad: Azimuth span of ``column_azimuths_rad``.  Values below a
            full turn model sector-coverage sensors.

    Returns:
        Parameters with ``_N_ROWS`` rows and ``_N_COLS`` columns.
    """
    if row_azimuth_offsets_rad is None:
        row_azimuth_offsets_rad = np.zeros(_N_ROWS, dtype=np.float64)
    return RowOffsetStructuredSpinningLidarModelParameters(
        spinning_frequency_hz=spinning_frequency_hz,
        spinning_direction="ccw",
        n_rows=_N_ROWS,
        n_columns=_N_COLS,
        row_elevations_rad=np.deg2rad(np.linspace(15.0, -15.0, _N_ROWS, dtype=np.float64)).astype(np.float32),
        column_azimuths_rad=np.linspace(0.0, coverage_rad, _N_COLS, endpoint=False, dtype=np.float32),
        row_azimuth_offsets_rad=row_azimuth_offsets_rad.astype(np.float32),
    )


def _model_direction(
    params: RowOffsetStructuredSpinningLidarModelParameters,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Compute model-derived direction from intrinsic parameters."""
    col_az = np.asarray(params.column_azimuths_rad, dtype=np.float64)
    row_az = np.asarray(params.row_azimuth_offsets_rad, dtype=np.float64)
    row_el = np.asarray(params.row_elevations_rad, dtype=np.float64)
    az = col_az[cols] + row_az[rows]
    el = row_el[rows]
    cos_e = np.cos(el)
    return np.stack([cos_e * np.cos(az), cos_e * np.sin(az), np.sin(el)], axis=-1).astype(np.float32)


class _FakeLidarReader:
    """Minimal mock returning per-frame direction + model_element arrays.

    Accepts parallel lists of per-frame ``(direction, model_element)``
    pairs so multi-frame aggregation paths can be exercised.
    """

    def __init__(
        self,
        directions: list[np.ndarray],
        model_elements: list[np.ndarray],
        frame_end_ts_us: list[int] | None = None,
        distances: list[np.ndarray] | None = None,
    ) -> None:
        if len(directions) != len(model_elements):
            raise ValueError("directions and model_elements must be the same length")
        if distances is None:
            distances = [np.ones(direction.shape[0], dtype=np.float32) for direction in directions]
        if len(distances) != len(directions):
            raise ValueError("distances length must match directions/model_elements")
        if frame_end_ts_us is None:
            frame_end_ts_us = [(i + 1) * 100_000 for i in range(len(directions))]
        if len(frame_end_ts_us) != len(directions):
            raise ValueError("frame_end_ts_us length must match directions/model_elements")
        ends = np.asarray(frame_end_ts_us, dtype=np.int64)
        starts = ends - 100_000
        self.frames_timestamps_us = np.stack([starts, ends], axis=-1)
        self._direction_by_ts = {int(ts): directions[i] for i, ts in enumerate(ends)}
        self._element_by_ts = {int(ts): model_elements[i] for i, ts in enumerate(ends)}
        self._distance_by_ts = {int(ts): distances[i] for i, ts in enumerate(ends)}

    def has_frame_ray_bundle_data(self, ts: int, key: str) -> bool:
        return int(ts) in self._direction_by_ts and key in {
            "direction",
            "model_element",
        }

    def get_frame_ray_bundle_data(self, ts: int, key: str) -> np.ndarray:
        ts = int(ts)
        if key == "direction":
            return self._direction_by_ts[ts]
        if key == "model_element":
            return self._element_by_ts[ts]
        raise KeyError(key)

    def get_frame_ray_bundle_return_data(self, ts: int, key: str, return_index: int | None) -> np.ndarray:
        if key == "distance_m" and return_index == 0:
            return self._distance_by_ts[int(ts)]
        raise KeyError(key)


class _FakeIntrinsicsReader:
    """Intrinsics reader supporting both LiDAR and camera model lookups."""

    def __init__(
        self,
        lidar_params_by_id: dict | None = None,
        camera_params_by_id: dict | None = None,
    ) -> None:
        self._lidar = lidar_params_by_id or {}
        self._camera = camera_params_by_id or {}

    def get_lidar_model_parameters(self, lidar_id: str):
        return self._lidar.get(lidar_id)

    def get_camera_model_parameters(self, camera_id: str):
        if camera_id in self._camera:
            return self._camera[camera_id]
        raise KeyError(camera_id)


_TAUTOLOGY_MARKER = "stored direction matches model-derived direction"


def _random_model_element(rng: np.random.Generator, n_points: int) -> np.ndarray:
    rows = rng.integers(0, _N_ROWS, size=n_points).astype(np.uint16)
    cols = rng.integers(0, _N_COLS, size=n_points).astype(np.uint16)
    return np.stack([rows, cols], axis=-1)


def _make_ctx(
    lidar_id: str,
    params: RowOffsetStructuredSpinningLidarModelParameters,
    directions: list[np.ndarray],
    model_elements: list[np.ndarray],
) -> NcoreV4Context:
    return _fake_ctx(
        lidar_readers={lidar_id: _FakeLidarReader(directions, model_elements)},
        intrinsics_readers={"intrinsics": _FakeIntrinsicsReader(lidar_params_by_id={lidar_id: params})},
    )


class TestWarnLidarDirectionCollapsedToModel(unittest.TestCase):
    LIDAR_ID = "lidar_top"

    def _assert_warning_fires(self, ctx: NcoreV4Context) -> None:
        with self.assertLogs(_validate_log, level=logging.INFO) as cm:
            warn_lidar_direction_collapsed_to_model(ctx)
        msgs = [r.getMessage() for r in cm.records if r.levelno == logging.INFO]
        hits = [m for m in msgs if _TAUTOLOGY_MARKER in m]
        self.assertEqual(len(hits), 1, msg=f"expected exactly one warning; got: {msgs}")

    def _assert_warning_silent(self, ctx: NcoreV4Context) -> None:
        with _capture_logs() as records:
            warn_lidar_direction_collapsed_to_model(ctx)
        hits = [r.getMessage() for r in records if _TAUTOLOGY_MARKER in r.getMessage()]
        self.assertEqual(hits, [], msg=f"unexpected warning; got: {hits}")

    def test_tautological_model_derived_direction_fires(self) -> None:
        """Model-derived direction with zero offsets must fire across multiple frames."""
        rng = np.random.default_rng(42)
        params = _make_params()
        # Two frames with independent ray samples so the multi-frame
        # error-aggregation path (``np.concatenate(all_errors_rad)``) runs.
        me_a = _random_model_element(rng, 5000)
        me_b = _random_model_element(rng, 5000)
        dir_a = _model_direction(params, me_a[:, 0], me_a[:, 1])
        dir_b = _model_direction(params, me_b[:, 0], me_b[:, 1])
        ctx = _make_ctx(self.LIDAR_ID, params, [dir_a, dir_b], [me_a, me_b])
        self._assert_warning_fires(ctx)

    def test_xyz_derived_direction_silent(self) -> None:
        """XYZ-derived direction (with noise) must not fire even with zero offsets."""
        rng = np.random.default_rng(42)
        params = _make_params()
        me = _random_model_element(rng, 5000)
        base = _model_direction(params, me[:, 0], me[:, 1]).astype(np.float64)
        # Noise at 0.5 deg scale: well above the 0.01 deg tautology threshold,
        # at the float32 quantization noise floor of XYZ-derived directions.
        noise = rng.normal(0.0, np.deg2rad(0.5), size=base.shape)
        perturbed = base + noise
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        direction = (perturbed / norms).astype(np.float32)
        ctx = _make_ctx(self.LIDAR_ID, params, [direction], [me])
        self._assert_warning_silent(ctx)

    def test_nonzero_offsets_skips(self) -> None:
        """Non-zero row_azimuth_offsets_rad => early skip, no warning."""
        rng = np.random.default_rng(42)
        # Calibration-style offsets at +/- 0.1 deg: physically plausible
        # per-beam offsets that NCore preserves on the parameters.
        offsets = np.deg2rad(np.linspace(-0.1, 0.1, _N_ROWS, dtype=np.float64))
        params = _make_params(row_azimuth_offsets_rad=offsets)
        me = _random_model_element(rng, 5000)
        direction = _model_direction(params, me[:, 0], me[:, 1])
        ctx = _make_ctx(self.LIDAR_ID, params, [direction], [me])
        self._assert_warning_silent(ctx)

    def test_no_op_when_no_lidar_readers(self) -> None:
        """No LiDARs => early return, no warning."""
        ctx = _fake_ctx(lidar_readers={}, intrinsics_readers={})
        self._assert_warning_silent(ctx)


class TestLidarMalformedDataHardErrors(unittest.TestCase):
    """Tests for malformed LiDAR data that must not be silently skipped."""

    LIDAR_ID = "lidar_top"

    def test_model_element_eval_failure_is_hard_error(self) -> None:
        """Out-of-range model_element indices must fail model accuracy."""
        params = _make_params()
        model_element = np.array([[_N_ROWS, 0]], dtype=np.uint16)
        direction = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        ctx = _fake_ctx(
            lidar_readers={self.LIDAR_ID: _FakeLidarReader([direction], [model_element])},
            intrinsics_readers={"intrinsics": _FakeIntrinsicsReader(lidar_params_by_id={self.LIDAR_ID: params})},
            errors=[],
        )

        check_lidar_model_accuracy(ctx)

        self.assertEqual(len(ctx.errors), 1)
        self.assertIn("model_element cannot be evaluated", ctx.errors[0])

    def test_direction_distance_length_mismatch_is_hard_error(self) -> None:
        """distance_m length must match the direction ray count."""

        class _StaticPoseReader:
            def get_static_pose(self, source: str, target: str) -> np.ndarray:
                if source == TestLidarMalformedDataHardErrors.LIDAR_ID and target == "rig":
                    return np.eye(4, dtype=np.float32)
                raise KeyError(f"{source}->{target}")

        params = _make_params()
        model_element = np.array([[0, 0], [0, 1]], dtype=np.uint16)
        direction = _model_direction(params, model_element[:, 0], model_element[:, 1])
        ctx = _fake_ctx(
            lidar_readers={
                self.LIDAR_ID: _FakeLidarReader(
                    [direction],
                    [model_element],
                    distances=[np.array([5.0], dtype=np.float32)],
                )
            },
            poses_readers={"default": _StaticPoseReader()},
            errors=[],
        )

        points = check_lidar_frame_data(ctx)

        self.assertEqual(points, {})
        self.assertEqual(len(ctx.errors), 1)
        self.assertIn("ray bundle shape issues", ctx.errors[0])


# ---------------------------------------------------------------------------
# Shared mock helpers for rig-alignment and cuboid tests
# ---------------------------------------------------------------------------


class _FakePoseReader:
    """Minimal pose reader returning pre-built dynamic poses."""

    def __init__(self, T_rig_worlds: np.ndarray, timestamps_us: np.ndarray) -> None:
        self._T = T_rig_worlds
        self._ts = timestamps_us

    def get_dynamic_pose(self, source: str, target: str) -> tuple[np.ndarray, np.ndarray]:
        if source == "rig" and target == "world":
            return self._T, self._ts
        raise KeyError(f"{source}->{target}")

    def get_static_pose(self, source: str, target: str) -> np.ndarray:
        raise KeyError(f"{source}->{target}")


def _make_straight_trajectory(
    n_poses: int,
    heading_deg: float = 0.0,
    step_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a straight-line rig trajectory along heading_deg in the XY plane.

    Returns (T_rig_worlds [n, 4, 4], timestamps_us [n]).
    heading_deg=0 means travel along +X (aligned with NCore rig convention).
    """
    rad = np.deg2rad(heading_deg)
    cos_h, sin_h = np.cos(rad), np.sin(rad)
    R = np.eye(3, dtype=np.float64)
    R[0, 0], R[0, 1] = cos_h, -sin_h
    R[1, 0], R[1, 1] = sin_h, cos_h

    poses = np.zeros((n_poses, 4, 4), dtype=np.float32)
    ts = np.arange(n_poses, dtype=np.uint64) * 100_000
    for i in range(n_poses):
        poses[i, :3, :3] = R.astype(np.float32)
        poses[i, 0, 3] = float(i * step_m * cos_h)
        poses[i, 1, 3] = float(i * step_m * sin_h)
        poses[i, 3, 3] = 1.0
    return poses, ts


# ---------------------------------------------------------------------------
# check_rig_forward_alignment tests
# ---------------------------------------------------------------------------


class TestCheckRigForwardAlignment(unittest.TestCase):
    """Tests for the hard-error rig +X alignment check."""

    def _make_ctx(self, T_rig_worlds: np.ndarray, timestamps_us: np.ndarray) -> NcoreV4Context:
        return _fake_ctx(
            poses_readers={"default": _FakePoseReader(T_rig_worlds, timestamps_us)},
            errors=[],
        )

    def test_passes_on_straight_forward_trajectory(self) -> None:
        """Perfectly +X-aligned trajectory must pass."""
        T, ts = _make_straight_trajectory(20, heading_deg=0.0)
        ctx = self._make_ctx(T, ts)
        check_rig_forward_alignment(ctx)
        self.assertEqual(ctx.errors, [])

    def test_fails_on_90deg_rotated_rig(self) -> None:
        """Rig faces +X but travels along +Y => hard-error must fire."""
        n = 20
        T = np.zeros((n, 4, 4), dtype=np.float32)
        ts = np.arange(n, dtype=np.uint64) * 100_000
        for i in range(n):
            T[i, :3, :3] = np.eye(3, dtype=np.float32)
            T[i, 1, 3] = float(i)  # translate along +Y
            T[i, 3, 3] = 1.0
        ctx = self._make_ctx(T, ts)
        check_rig_forward_alignment(ctx)
        self.assertGreater(len(ctx.errors), 0)
        self.assertIn("alignment", ctx.errors[0].lower())

    def test_skips_on_too_few_poses(self) -> None:
        """Single pose => no trajectory to evaluate, should skip gracefully."""
        T, ts = _make_straight_trajectory(1)
        ctx = self._make_ctx(T, ts)
        check_rig_forward_alignment(ctx)
        self.assertEqual(ctx.errors, [])

    def test_skips_on_stationary_sequence(self) -> None:
        """All poses at origin => zero displacement, should skip."""
        n = 20
        T = np.zeros((n, 4, 4), dtype=np.float32)
        for i in range(n):
            T[i] = np.eye(4, dtype=np.float32)
        ts = np.arange(n, dtype=np.uint64) * 100_000
        ctx = self._make_ctx(T, ts)
        check_rig_forward_alignment(ctx)
        self.assertEqual(ctx.errors, [])


# ---------------------------------------------------------------------------
# warn_sequence_envelope_coverage tests
# ---------------------------------------------------------------------------


class _FakeCameraReader:
    """Minimal camera reader with configurable frame timestamps.

    ``min_ts`` / ``max_ts`` define the envelope: first frame start = min_ts,
    last frame end = max_ts.
    """

    def __init__(self, min_ts: int, max_ts: int, n_frames: int = 10) -> None:
        starts = np.linspace(min_ts, max_ts - 1000, n_frames, dtype=np.int64)
        ends = np.copy(starts) + 1000
        ends[-1] = max(ends[-1], max_ts)
        self.frames_timestamps_us = np.stack([starts, ends], axis=-1)


class _FakeLidarReaderEnvelope:
    """Minimal LiDAR reader with configurable frame timestamps."""

    def __init__(self, min_ts: int, max_ts: int, n_frames: int = 10) -> None:
        starts = np.linspace(min_ts, max_ts - 10_000, n_frames, dtype=np.int64)
        ends = np.copy(starts) + 10_000
        ends[-1] = max(ends[-1], max_ts)
        self.frames_timestamps_us = np.stack([starts, ends], axis=-1)


class _FakeCuboidObservation:
    def __init__(self, ts: int) -> None:
        self.timestamp_us = ts


class _FakeCuboidReader:
    def __init__(self, timestamps: list[int]) -> None:
        self._obs = [_FakeCuboidObservation(t) for t in timestamps]

    def get_observations(self):
        return iter(self._obs)


class TestWarnSequenceEnvelopeCoverage(unittest.TestCase):
    """Tests for the sensor envelope coverage warning."""

    def _make_ctx(
        self,
        interval_start: int,
        interval_stop: int,
        *,
        camera_ranges: dict[str, tuple[int, int]] | None = None,
        lidar_ranges: dict[str, tuple[int, int]] | None = None,
        cuboid_timestamps: dict[str, list[int]] | None = None,
    ) -> NcoreV4Context:
        camera_readers = {}
        if camera_ranges:
            for cam_id, (s, e) in camera_ranges.items():
                camera_readers[cam_id] = _FakeCameraReader(s, e)

        lidar_readers = {}
        if lidar_ranges:
            for lid_id, (s, e) in lidar_ranges.items():
                lidar_readers[lid_id] = _FakeLidarReaderEnvelope(s, e)

        cuboid_readers = {}
        if cuboid_timestamps:
            for cub_id, ts_list in cuboid_timestamps.items():
                cuboid_readers[cub_id] = _FakeCuboidReader(ts_list)

        return _fake_ctx(
            interval=HalfClosedInterval.from_start_end(interval_start, interval_stop),
            camera_readers=camera_readers,
            lidar_readers=lidar_readers,
            cuboid_readers=cuboid_readers,
        )

    def _assert_no_warning(self, ctx: NcoreV4Context, marker: str) -> None:
        with _capture_logs() as records:
            warn_sequence_envelope_coverage(ctx)
        hits = [r.getMessage() for r in records if marker in r.getMessage()]
        self.assertEqual(hits, [], msg=f"unexpected warning: {hits}")

    def _assert_warning(self, ctx: NcoreV4Context, marker: str) -> None:
        with self.assertLogs(_validate_log, level=logging.INFO) as cm:
            warn_sequence_envelope_coverage(ctx)
        msgs = [r.getMessage() for r in cm.records]
        hits = [m for m in msgs if marker in m]
        self.assertGreater(len(hits), 0, msg=f"expected warning containing '{marker}'; got: {msgs}")

    def test_no_warning_when_sensors_span_interval(self) -> None:
        """Full coverage => no warning."""
        ctx = self._make_ctx(
            0,
            1_000_000,
            camera_ranges={"cam": (0, 1_000_000)},
            lidar_ranges={"lidar": (0, 1_000_000)},
        )
        self._assert_no_warning(ctx, "narrower than sequence interval")

    def test_warns_on_camera_gap(self) -> None:
        """Camera envelope starts late => sensor envelope warning."""
        ctx = self._make_ctx(
            0,
            1_000_000,
            camera_ranges={"cam": (100_000, 1_000_000)},
            lidar_ranges={"lidar": (0, 1_000_000)},
        )
        self._assert_warning(ctx, "sensor envelope")

    def test_warns_on_cuboid_gap(self) -> None:
        """Cuboid timestamps start late => cuboid envelope warning."""
        ctx = self._make_ctx(
            0,
            1_000_000,
            camera_ranges={"cam": (0, 1_000_000)},
            cuboid_timestamps={"default": [200_000, 500_000, 800_000]},
        )
        self._assert_warning(ctx, "cuboid component envelope")

    def test_no_cuboid_warning_when_spanning(self) -> None:
        """Cuboid timestamps span the interval => no cuboid warning."""
        ctx = self._make_ctx(
            0,
            1_000_000,
            camera_ranges={"cam": (0, 1_000_000)},
            cuboid_timestamps={"default": [0, 500_000, 1_000_000]},
        )
        self._assert_no_warning(ctx, "cuboid component envelope")

    def test_negative_gap_suppressed(self) -> None:
        """Sensor envelope wider than interval should not produce a warning."""
        ctx = self._make_ctx(
            100_000,
            900_000,
            camera_ranges={"cam": (0, 1_000_000)},
        )
        self._assert_no_warning(ctx, "sensor envelope")


# ---------------------------------------------------------------------------
# check_cuboid_lidar_overlap tests (integration-style with is_within_3d_bboxes)
# ---------------------------------------------------------------------------


class _FakeCuboidReaderFull:
    """Cuboid reader returning pre-built CuboidTrackObservation objects."""

    def __init__(self, observations: list[CuboidTrackObservation]) -> None:
        self._obs = observations

    def get_observations(self):
        return iter(self._obs)


def _make_cuboid_obs(
    centroid: tuple[float, float, float],
    dim: tuple[float, float, float],
    ts: int = 100_000,
    ref_frame: str = "rig",
) -> CuboidTrackObservation:
    return CuboidTrackObservation(
        track_id="t1",
        class_id="car",
        timestamp_us=ts,
        reference_frame_id=ref_frame,
        reference_frame_timestamp_us=ts,
        bbox3=BBox3(centroid=centroid, dim=dim, rot=(0.0, 0.0, 0.0)),
        source=LabelSource.EXTERNAL,
    )


def _identity_pose_ctx(
    observations: list[CuboidTrackObservation],
    interval_start: int = 0,
    interval_stop: int = 200_000,
) -> NcoreV4Context:
    """Build a ctx with identity rig->world pose at a single timestamp."""
    T = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
    ts = np.array([100_000], dtype=np.uint64)
    return _fake_ctx(
        interval=HalfClosedInterval.from_start_end(interval_start, interval_stop),
        poses_readers={"default": _FakePoseReader(T, ts)},
        cuboid_readers={"default": _FakeCuboidReaderFull(observations)},
        camera_readers={},
        lidar_readers={
            "lidar_top": types.SimpleNamespace(
                frames_timestamps_us=np.array([[interval_start, 100_000]], dtype=np.int64)
            )
        },
        errors=[],
    )


class TestCheckCuboidLidarOverlap(unittest.TestCase):
    """Tests for the near-range cuboid LiDAR overlap hard error."""

    def test_passes_when_points_inside_cuboid(self) -> None:
        """Points at centroid => 100% hit rate, must pass."""
        obs = [_make_cuboid_obs((5.0, 0.0, 0.0), (4.0, 2.0, 2.0)) for _ in range(10)]
        pts = np.tile(np.array([[5.0, 0.0, 0.0]], dtype=np.float64), (100, 1))
        lidar_pts = {"lidar_top": {100_000: pts}}
        ctx = _identity_pose_ctx(obs)
        check_cuboid_lidar_overlap(ctx, lidar_pts_rig=lidar_pts)
        self.assertEqual(ctx.errors, [])

    def test_fails_when_points_far_from_cuboid(self) -> None:
        """All points far from any cuboid => 0% hit rate, must fail."""
        obs = [_make_cuboid_obs((5.0, 0.0, 0.0), (1.0, 1.0, 1.0)) for _ in range(10)]
        pts = np.tile(np.array([[50.0, 50.0, 50.0]], dtype=np.float64), (100, 1))
        lidar_pts = {"lidar_top": {100_000: pts}}
        ctx = _identity_pose_ctx(obs)
        check_cuboid_lidar_overlap(ctx, lidar_pts_rig=lidar_pts)
        self.assertGreater(len(ctx.errors), 0)

    def test_skips_far_range_cuboids(self) -> None:
        """Cuboids beyond CUBOID_NEAR_RANGE_M are not checked."""
        obs = [_make_cuboid_obs((200.0, 0.0, 0.0), (4.0, 2.0, 2.0)) for _ in range(10)]
        pts = np.zeros((100, 3), dtype=np.float64)
        lidar_pts = {"lidar_top": {100_000: pts}}
        ctx = _identity_pose_ctx(obs)
        check_cuboid_lidar_overlap(ctx, lidar_pts_rig=lidar_pts)
        self.assertEqual(ctx.errors, [])


# ---------------------------------------------------------------------------
# check_cuboid_obb_inlier_ratio tests
# ---------------------------------------------------------------------------


class TestCheckCuboidObbInlierRatio(unittest.TestCase):
    """Tests for the OBB inlier ratio hard error."""

    def test_passes_when_points_inside_obb(self) -> None:
        """Points inside the OBB => high inlier ratio, must pass."""
        obs = [_make_cuboid_obs((5.0, 0.0, 0.0), (4.0, 2.0, 2.0)) for _ in range(10)]
        pts = np.array([[5.0, 0.0, 0.0]] * 200, dtype=np.float64)
        lidar_pts = {"lidar_top": {100_000: pts}}
        ctx = _identity_pose_ctx(obs)
        check_cuboid_obb_inlier_ratio(ctx, lidar_pts_rig=lidar_pts)
        self.assertEqual(ctx.errors, [])

    def test_fails_on_axis_swap(self) -> None:
        """Points in AABB but outside OBB (rotated box) => low inlier ratio.

        Create a long thin box rotated 45 deg: AABB is much larger than OBB,
        so points at AABB corners are outside the OBB.
        """
        obs_list = []
        for _ in range(10):
            obs_list.append(
                CuboidTrackObservation(
                    track_id="t1",
                    class_id="car",
                    timestamp_us=100_000,
                    reference_frame_id="rig",
                    reference_frame_timestamp_us=100_000,
                    bbox3=BBox3(
                        centroid=(5.0, 0.0, 0.0),
                        dim=(10.0, 1.0, 1.0),
                        rot=(0.0, 0.0, np.pi / 4),
                    ),
                    source=LabelSource.EXTERNAL,
                )
            )
        R45 = Rotation.from_euler("xyz", [0, 0, np.pi / 4]).as_matrix()
        extents = np.array([10.0, 1.0, 1.0])
        corners_local = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=np.float64,
        ) * (extents / 2)
        corners_rig = corners_local @ R45.T + np.array([5.0, 0.0, 0.0])
        aabb_min = corners_rig.min(axis=0)
        aabb_max = corners_rig.max(axis=0)
        rng = np.random.default_rng(42)
        pts = rng.uniform(aabb_min, aabb_max, size=(500, 3))
        lidar_pts = {"lidar_top": {100_000: pts}}
        ctx = _identity_pose_ctx(obs_list)
        check_cuboid_obb_inlier_ratio(ctx, lidar_pts_rig=lidar_pts)
        self.assertGreater(len(ctx.errors), 0)


# ---------------------------------------------------------------------------
# check_cuboid_pose_gap tests
# ---------------------------------------------------------------------------


def _pose_gap_ctx(ref_frame: str, ref_ts: int, pose_ts_us: list[int]) -> NcoreV4Context:
    """Build a ctx with one cuboid in *ref_frame* and waypoints at *pose_ts_us*."""
    obs = [_make_cuboid_obs((5.0, 0.0, 0.0), (4.0, 2.0, 2.0), ts=ref_ts, ref_frame=ref_frame)]
    ts = np.array(pose_ts_us, dtype=np.uint64)
    T = np.tile(np.eye(4, dtype=np.float32), (ts.size, 1, 1))
    return _fake_ctx(
        poses_readers={"default": _FakePoseReader(T, ts)},
        cuboid_readers={"default": _FakeCuboidReaderFull(obs)},
        errors=[],
    )


class TestCheckCuboidPoseGap(unittest.TestCase):
    """Tests for the cuboid reference-timestamp pose gap hard error.

    Placement crosses the dynamic ``rig`` -> ``world`` edge for ``rig`` and
    sensor reference frames, so those need a waypoint at the reference
    timestamp; ``world`` and ``world_global`` reach the world frame without
    it.
    """

    def test_passes_when_rig_frame_waypoint_present(self) -> None:
        """Waypoint exactly at the reference timestamp => no error."""
        ctx = _pose_gap_ctx("rig", 100_000, [0, 100_000, 200_000])
        check_cuboid_pose_gap(ctx)
        self.assertEqual(ctx.errors, [])

    def test_fails_when_rig_frame_waypoint_missing(self) -> None:
        """Nearest waypoint 50 ms away => error."""
        ctx = _pose_gap_ctx("rig", 100_000, [0, 50_000, 200_000])
        check_cuboid_pose_gap(ctx)
        self.assertEqual(len(ctx.errors), 1)

    def test_fails_when_sensor_frame_waypoint_missing(self) -> None:
        """Sensor-anchored cuboids compose onto the same dynamic edge."""
        ctx = _pose_gap_ctx("lidar_top", 100_000, [0, 50_000, 200_000])
        check_cuboid_pose_gap(ctx)
        self.assertEqual(len(ctx.errors), 1)

    def test_exempts_world_anchored_frames(self) -> None:
        """Neither world frame evaluates a pose at the reference timestamp."""
        for ref_frame in WORLD_ANCHORED_CUBOID_FRAMES:
            with self.subTest(ref_frame=ref_frame):
                ctx = _pose_gap_ctx(ref_frame, 100_000, [0, 50_000, 200_000])
                check_cuboid_pose_gap(ctx)
                self.assertEqual(ctx.errors, [])

    def test_no_op_without_pose_readers(self) -> None:
        """Nothing to compare against => check stays silent."""
        ctx = _pose_gap_ctx("rig", 100_000, [0, 50_000, 200_000])
        ctx.poses_readers = {}
        check_cuboid_pose_gap(ctx)
        self.assertEqual(ctx.errors, [])


# ---------------------------------------------------------------------------
# warn_cuboid_ground_clearance tests
# ---------------------------------------------------------------------------

_GROUND_CLEARANCE_MARKER = "cuboid ground clearance"


class TestWarnCuboidGroundClearance(unittest.TestCase):
    """Tests for the floating-cuboid ground clearance warning."""

    def _assert_warning(self, ctx: NcoreV4Context, lidar_pts: dict) -> None:
        with self.assertLogs(_validate_log, level=logging.INFO) as cm:
            warn_cuboid_ground_clearance(ctx, lidar_pts_rig=lidar_pts)
        msgs = [r.getMessage() for r in cm.records]
        hits = [m for m in msgs if _GROUND_CLEARANCE_MARKER in m and "below-cuboid" in m]
        self.assertGreater(len(hits), 0, msg=f"expected ground clearance warning; got: {msgs}")

    def _assert_no_warning(self, ctx: NcoreV4Context, lidar_pts: dict) -> None:
        with _capture_logs() as records:
            warn_cuboid_ground_clearance(ctx, lidar_pts_rig=lidar_pts)
        hits = [
            r.getMessage()
            for r in records
            if _GROUND_CLEARANCE_MARKER in r.getMessage() and "below-cuboid" in r.getMessage()
        ]
        self.assertEqual(hits, [], msg=f"unexpected ground clearance warning: {hits}")

    def test_warns_on_floating_cuboids(self) -> None:
        """Points mostly below cuboid bottom => warning fires."""
        obs = [_make_cuboid_obs((5.0, 0.0, 2.0), (4.0, 2.0, 2.0)) for _ in range(10)]
        pts = np.tile(np.array([[5.0, 0.0, -0.5]], dtype=np.float64), (200, 1))
        rng = np.random.default_rng(42)
        pts += rng.normal(0, 0.2, size=pts.shape)
        lidar_pts = {"lidar_top": {100_000: pts}}
        ctx = _identity_pose_ctx(obs)
        self._assert_warning(ctx, lidar_pts)

    def test_silent_when_points_inside_cuboid(self) -> None:
        """Points at cuboid centroid => no below-ratio, no warning."""
        obs = [_make_cuboid_obs((5.0, 0.0, 0.0), (4.0, 2.0, 2.0)) for _ in range(10)]
        pts = np.tile(np.array([[5.0, 0.0, 0.0]], dtype=np.float64), (200, 1))
        lidar_pts = {"lidar_top": {100_000: pts}}
        ctx = _identity_pose_ctx(obs)
        self._assert_no_warning(ctx, lidar_pts)


# ---------------------------------------------------------------------------
# check_lidar_model_fov_consistency tests
# ---------------------------------------------------------------------------


class TestCheckLidarModelFovConsistency(unittest.TestCase):
    """Tests for the NuRec FOV self-consistency hard error."""

    LIDAR_ID = "lidar_top"

    def _make_ctx(
        self,
        params: RowOffsetStructuredSpinningLidarModelParameters,
    ) -> NcoreV4Context:
        return _fake_ctx(
            lidar_readers={self.LIDAR_ID: types.SimpleNamespace()},
            intrinsics_readers={"intrinsics": _FakeIntrinsicsReader(lidar_params_by_id={self.LIDAR_ID: params})},
            errors=[],
        )

    def test_passes_with_zero_offsets(self) -> None:
        """Standard parameters with zero row_azimuth_offsets_rad must pass."""
        params = _make_params()
        ctx = self._make_ctx(params)
        check_lidar_model_fov_consistency(ctx)
        self.assertEqual(ctx.errors, [])

    def test_passes_with_small_offsets(self) -> None:
        """Small calibration-scale offsets (< column step) must pass."""
        col_step = 2.0 * np.pi / _N_COLS
        offsets = np.linspace(-0.3 * col_step, 0.3 * col_step, _N_ROWS, dtype=np.float64)
        params = _make_params(row_azimuth_offsets_rad=offsets)
        ctx = self._make_ctx(params)
        check_lidar_model_fov_consistency(ctx)
        self.assertEqual(ctx.errors, [])

    def test_fails_when_map_init_raises(self) -> None:
        """Parameters that cause NCore's map init to assert must trigger a hard error.

        n_columns > 32767 overflows the default int16 angles-to-columns map
        dtype, triggering a real AssertionError inside NCore's
        _init_angles_to_columns_map -- the same code path NuRec exercises.
        """
        n_rows, n_cols = 4, 40_000
        params = RowOffsetStructuredSpinningLidarModelParameters(
            spinning_frequency_hz=10.0,
            spinning_direction="ccw",
            n_rows=n_rows,
            n_columns=n_cols,
            row_elevations_rad=np.deg2rad(np.linspace(15.0, -15.0, n_rows, dtype=np.float64)).astype(np.float32),
            column_azimuths_rad=np.linspace(0.0, 2.0 * np.pi, n_cols, endpoint=False, dtype=np.float32),
            row_azimuth_offsets_rad=np.zeros(n_rows, dtype=np.float32),
        )
        ctx = self._make_ctx(params)
        check_lidar_model_fov_consistency(ctx)
        self.assertGreater(len(ctx.errors), 0)
        self.assertIn("FOV self-consistency", ctx.errors[0])

    def test_skips_non_spinning_lidar(self) -> None:
        """Non-spinning LiDAR (no RowOffset params) => no error, no pass line."""
        ctx = _fake_ctx(
            lidar_readers={"lidar_solid_state": types.SimpleNamespace()},
            intrinsics_readers={"intrinsics": _FakeIntrinsicsReader()},
            errors=[],
        )
        check_lidar_model_fov_consistency(ctx)
        self.assertEqual(ctx.errors, [])


# ---------------------------------------------------------------------------
# warn_lidar_spin_frequency_mismatch tests
# ---------------------------------------------------------------------------

_SPIN_FREQUENCY_MARKER = "disagrees with the stored frame windows"


def _frames_at_hz(rate_hz: float, n_frames: int = 10, cadence_hz: float | None = None) -> np.ndarray:
    """Return ``[start, end]`` frame windows of one revolution each.

    Args:
        rate_hz: Rotor rate; sets each window's duration.
        n_frames: Number of frames to emit.
        cadence_hz: Frame publication rate.  Defaults to ``rate_hz``
            (contiguous frames); a lower value models a dataset that stores
            only every Nth revolution.

    Returns:
        ``(n_frames, 2)`` array of start/end timestamps in microseconds.
    """
    window_us = 1e6 / rate_hz
    stride_us = 1e6 / (cadence_hz if cadence_hz is not None else rate_hz)
    starts = np.arange(n_frames, dtype=np.float64) * stride_us
    return np.stack([starts, starts + window_us], axis=-1).astype(np.int64)


class TestWarnLidarSpinFrequencyMismatch(unittest.TestCase):
    """Tests for the declared-versus-measured spin frequency warning."""

    LIDAR_ID = "lidar_top"

    def _make_ctx(
        self,
        params: RowOffsetStructuredSpinningLidarModelParameters,
        frames: np.ndarray,
    ) -> NcoreV4Context:
        return _fake_ctx(
            lidar_readers={self.LIDAR_ID: types.SimpleNamespace(frames_timestamps_us=frames)},
            intrinsics_readers={"intrinsics": _FakeIntrinsicsReader(lidar_params_by_id={self.LIDAR_ID: params})},
        )

    def _warnings(self, ctx: NcoreV4Context) -> list[str]:
        with _capture_logs() as records:
            warn_lidar_spin_frequency_mismatch(ctx)
        return [r.getMessage() for r in records if _SPIN_FREQUENCY_MARKER in r.getMessage()]

    def test_declared_matches_frame_timing_silent(self) -> None:
        """10 Hz declared against 100 ms frames must not fire."""
        ctx = self._make_ctx(_make_params(), _frames_at_hz(10.0))
        self.assertEqual(self._warnings(ctx), [])

    def test_declared_disagrees_with_window_fires(self) -> None:
        """8.6 Hz against 100 ms windows is a 14% error the tolerance must catch."""
        ctx = self._make_ctx(_make_params(spinning_frequency_hz=8.6), _frames_at_hz(10.0))
        hits = self._warnings(ctx)
        self.assertEqual(len(hits), 1, msg=f"expected exactly one warning; got: {hits}")
        self.assertIn("8.60", hits[0])
        self.assertIn("10.00 Hz", hits[0])

    def test_tolerance_is_relative_to_the_window_rate(self) -> None:
        """11 Hz against 100 ms windows is 10% off the window rate, though only 9.1% off the declared one."""
        ctx = self._make_ctx(_make_params(spinning_frequency_hz=11.0), _frames_at_hz(10.0))
        self.assertEqual(len(self._warnings(ctx)), 1)

    def test_jitter_within_tolerance_silent(self) -> None:
        """A 4% deviation stays inside the tolerance band."""
        ctx = self._make_ctx(_make_params(spinning_frequency_hz=10.4), _frames_at_hz(10.0))
        self.assertEqual(self._warnings(ctx), [])

    def test_decimated_frame_publication_silent(self) -> None:
        """A 20 Hz rotor stored every other revolution must not fire on cadence."""
        frames = _frames_at_hz(20.0, cadence_hz=10.0)
        ctx = self._make_ctx(_make_params(spinning_frequency_hz=20.0), frames)
        self.assertEqual(self._warnings(ctx), [])

    def test_sector_coverage_skipped(self) -> None:
        """A sector frame is a fraction of a turn, so the window estimator does not apply.

        The 270 deg case is the one a 180 deg gate would let through: its
        window implies a rate 33% high, well past the mismatch tolerance.
        """
        for coverage_deg in (90.0, 180.0, 270.0):
            with self.subTest(coverage_deg=coverage_deg):
                params = _make_params(spinning_frequency_hz=10.0, coverage_rad=np.deg2rad(coverage_deg))
                ctx = self._make_ctx(params, _frames_at_hz(10.0 * 360.0 / coverage_deg))
                self.assertEqual(self._warnings(ctx), [])

    def test_single_frame_uses_window_only(self) -> None:
        """With one frame the cadence is undefined; the window still applies."""
        ctx = self._make_ctx(_make_params(spinning_frequency_hz=20.0), _frames_at_hz(10.0, n_frames=1))
        self.assertEqual(len(self._warnings(ctx)), 1)

    def test_no_lidar_readers_silent(self) -> None:
        """No LiDARs => early return, no warning."""
        ctx = _fake_ctx(lidar_readers={}, intrinsics_readers={})
        self.assertEqual(self._warnings(ctx), [])


# ---------------------------------------------------------------------------
# warn_camera_fisheye_max_angle_deviation tests
# ---------------------------------------------------------------------------

_FISHEYE_MARKER = "OpenCV-fisheye max_angle deviation"


_FISHEYE_RESOLUTION = np.array([640, 480], dtype=np.uint64)
_FISHEYE_FOCAL = np.array([300.0, 300.0], dtype=np.float32)
_FISHEYE_PP = np.array([320.0, 240.0], dtype=np.float32)
_FISHEYE_RADIAL = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _make_fisheye_params(max_angle_rad: float) -> OpenCVFisheyeCameraModelParameters:
    """Build minimal OpenCVFisheyeCameraModelParameters with a given max_angle."""
    return OpenCVFisheyeCameraModelParameters(
        resolution=_FISHEYE_RESOLUTION.copy(),
        shutter_type=ShutterType.GLOBAL,
        focal_length=_FISHEYE_FOCAL.copy(),
        principal_point=_FISHEYE_PP.copy(),
        radial_coeffs=_FISHEYE_RADIAL.copy(),
        max_angle=float(max_angle_rad),
    )


class TestWarnCameraFisheyeMaxAngleDeviation(unittest.TestCase):
    """Tests for the fisheye max_angle deviation warning."""

    def _make_ctx(self, camera_params: dict) -> NcoreV4Context:
        camera_readers = {cam_id: types.SimpleNamespace() for cam_id in camera_params}
        intrinsics_readers = {"intrinsics": _FakeIntrinsicsReader(camera_params_by_id=camera_params)}
        return _fake_ctx(
            camera_readers=camera_readers,
            intrinsics_readers=intrinsics_readers,
        )

    def _assert_warning(self, ctx: NcoreV4Context) -> None:
        with self.assertLogs(_validate_log, level=logging.INFO) as cm:
            warn_camera_fisheye_max_angle_deviation(ctx)
        msgs = [r.getMessage() for r in cm.records]
        hits = [m for m in msgs if _FISHEYE_MARKER in m]
        self.assertGreater(len(hits), 0, msg=f"expected fisheye warning; got: {msgs}")

    def _assert_no_warning(self, ctx: NcoreV4Context) -> None:
        with _capture_logs() as records:
            warn_camera_fisheye_max_angle_deviation(ctx)
        hits = [r.getMessage() for r in records if _FISHEYE_MARKER in r.getMessage()]
        self.assertEqual(hits, [], msg=f"unexpected fisheye warning: {hits}")

    def _compute_correct_max_angle(self) -> float:
        return OpenCVFisheyeCameraModelParameters.compute_max_angle(
            resolution=_FISHEYE_RESOLUTION,
            focal_length=_FISHEYE_FOCAL,
            principal_point=_FISHEYE_PP,
            radial_coeffs=_FISHEYE_RADIAL,
        )

    def test_warns_when_stored_too_small(self) -> None:
        """Stored max_angle much smaller than computed => warning fires."""
        correct = self._compute_correct_max_angle()
        params = _make_fisheye_params(correct - np.radians(15.0))
        ctx = self._make_ctx({"cam_front": params})
        self._assert_warning(ctx)

    def test_warns_when_stored_too_large(self) -> None:
        """Stored max_angle much larger than computed => warning fires."""
        correct = self._compute_correct_max_angle()
        params = _make_fisheye_params(correct + np.radians(15.0))
        ctx = self._make_ctx({"cam_front": params})
        self._assert_warning(ctx)

    def test_silent_when_max_angle_correct(self) -> None:
        """Stored max_angle matches computed => no warning."""
        correct = self._compute_correct_max_angle()
        params = _make_fisheye_params(correct)
        ctx = self._make_ctx({"cam_front": params})
        self._assert_no_warning(ctx)

    def test_skips_when_no_cameras(self) -> None:
        """No cameras => early return, no warning."""
        ctx = _fake_ctx(camera_readers={}, intrinsics_readers={})
        self._assert_no_warning(ctx)


# ---------------------------------------------------------------------------
# nearest_idx tests
# ---------------------------------------------------------------------------


class TestNearestIdx(unittest.TestCase):
    """Tests for the ``nearest_idx`` helper in ``validate_cuboids``."""

    def test_exact_match(self) -> None:
        arr = np.array([10, 20, 30, 40, 50])
        self.assertEqual(nearest_idx(arr, 30), 2)

    def test_nearest_left(self) -> None:
        arr = np.array([10, 20, 30, 40, 50])
        self.assertEqual(nearest_idx(arr, 29), 2)

    def test_nearest_right(self) -> None:
        arr = np.array([10, 20, 30, 40, 50])
        self.assertEqual(nearest_idx(arr, 31), 2)

    def test_below_range(self) -> None:
        arr = np.array([10, 20, 30])
        self.assertEqual(nearest_idx(arr, 5), 0)

    def test_above_range(self) -> None:
        arr = np.array([10, 20, 30])
        self.assertEqual(nearest_idx(arr, 100), 2)

    def test_single_element(self) -> None:
        arr = np.array([42])
        self.assertEqual(nearest_idx(arr, 0), 0)
        self.assertEqual(nearest_idx(arr, 100), 0)

    def test_midpoint_keeps_right(self) -> None:
        """When equidistant, searchsorted picks right; left-check requires strictly-less so right stays."""
        arr = np.array([10, 20])
        self.assertEqual(nearest_idx(arr, 15), 1)


# ---------------------------------------------------------------------------
# _check_sensor_frame_ts tests
# ---------------------------------------------------------------------------


class _FakeSensorReader:
    """Minimal reader with configurable frames_timestamps_us."""

    def __init__(self, timestamps: np.ndarray) -> None:
        self.frames_timestamps_us = timestamps


class TestCheckSensorFrameTs(unittest.TestCase):
    """Tests for the generic frame timestamp invariant checker."""

    def test_passes_well_ordered(self) -> None:
        """Properly ordered, monotonic, unique-end timestamps pass all checks."""
        ts = np.array([[0, 100], [200, 300], [400, 500]], dtype=np.int64)
        errors: list[str] = []
        _check_sensor_frame_ts({"sensor_a": _FakeSensorReader(ts)}, "test", errors)
        self.assertEqual(errors, [])

    def test_fails_start_after_end(self) -> None:
        """Frame with start > end must fail the ordering check."""
        ts = np.array([[100, 50], [200, 300]], dtype=np.int64)
        errors: list[str] = []
        _check_sensor_frame_ts({"sensor_a": _FakeSensorReader(ts)}, "test", errors)
        self.assertTrue(any("ordering" in e for e in errors))

    def test_fails_non_monotonic(self) -> None:
        """Non-strictly-increasing ends must fail monotonicity check."""
        ts = np.array([[0, 300], [200, 250]], dtype=np.int64)
        errors: list[str] = []
        _check_sensor_frame_ts({"sensor_a": _FakeSensorReader(ts)}, "test", errors)
        self.assertTrue(any("increasing" in e for e in errors))

    def test_fails_duplicate_end(self) -> None:
        """Duplicate end timestamps must fail uniqueness check."""
        ts = np.array([[0, 100], [200, 100]], dtype=np.int64)
        errors: list[str] = []
        _check_sensor_frame_ts({"sensor_a": _FakeSensorReader(ts)}, "test", errors)
        self.assertTrue(any("duplicate" in e or "increasing" in e for e in errors))

    def test_skips_empty(self) -> None:
        """Empty timestamps array is silently skipped."""
        ts = np.empty((0, 2), dtype=np.int64)
        errors: list[str] = []
        _check_sensor_frame_ts({"sensor_a": _FakeSensorReader(ts)}, "test", errors)
        self.assertEqual(errors, [])

    def test_multiple_sensors(self) -> None:
        """Error from one sensor doesn't mask the other; both names appear."""
        good = np.array([[0, 100], [200, 300]], dtype=np.int64)
        bad = np.array([[100, 50]], dtype=np.int64)
        errors: list[str] = []
        _check_sensor_frame_ts(
            {
                "good_sensor": _FakeSensorReader(good),
                "bad_sensor": _FakeSensorReader(bad),
            },
            "test",
            errors,
        )
        self.assertTrue(any("bad_sensor" in e for e in errors))
        self.assertFalse(any("good_sensor" in e for e in errors))


# ---------------------------------------------------------------------------
# warn_camera_timestamps_all_identical tests (empty guard)
# ---------------------------------------------------------------------------

_IDENTICAL_TS_MARKER = "identical frame timestamps"


class TestWarnCameraTimestampsAllIdentical(unittest.TestCase):
    """Tests for the camera identical-timestamps warning."""

    def _make_ctx(self, camera_ts: dict[str, np.ndarray]) -> NcoreV4Context:
        readers = {}
        for cam_id, ts in camera_ts.items():
            readers[cam_id] = types.SimpleNamespace(frames_timestamps_us=ts)
        return _fake_ctx(camera_readers=readers)

    def _assert_warning(self, ctx: NcoreV4Context) -> None:
        with self.assertLogs(_validate_log, level=logging.INFO) as cm:
            warn_camera_timestamps_all_identical(ctx)
        msgs = [r.getMessage() for r in cm.records]
        hits = [m for m in msgs if _IDENTICAL_TS_MARKER in m]
        self.assertGreater(len(hits), 0, msg=f"expected identical-ts warning; got: {msgs}")

    def _assert_no_warning(self, ctx: NcoreV4Context) -> None:
        with _capture_logs() as records:
            warn_camera_timestamps_all_identical(ctx)
        hits = [r.getMessage() for r in records if _IDENTICAL_TS_MARKER in r.getMessage()]
        self.assertEqual(hits, [], msg=f"unexpected warning: {hits}")

    def test_warns_when_all_cameras_identical(self) -> None:
        """Two cameras with byte-identical timestamps => warning."""
        ts = np.array([[0, 100], [200, 300]], dtype=np.int64)
        ctx = self._make_ctx({"cam_a": ts.copy(), "cam_b": ts.copy()})
        self._assert_warning(ctx)

    def test_silent_when_timestamps_differ(self) -> None:
        """Different timestamps per camera => no warning."""
        ts_a = np.array([[0, 100], [200, 300]], dtype=np.int64)
        ts_b = np.array([[10, 110], [210, 310]], dtype=np.int64)
        ctx = self._make_ctx({"cam_a": ts_a, "cam_b": ts_b})
        self._assert_no_warning(ctx)

    def test_no_crash_when_all_cameras_empty(self) -> None:
        """All cameras with empty timestamps => no crash, no warning."""
        empty = np.empty((0, 2), dtype=np.int64)
        ctx = self._make_ctx({"cam_a": empty, "cam_b": empty})
        self._assert_no_warning(ctx)

    def test_skips_single_camera(self) -> None:
        """Single camera => early return, no warning."""
        ts = np.array([[0, 100]], dtype=np.int64)
        ctx = self._make_ctx({"cam_a": ts})
        self._assert_no_warning(ctx)

    def test_skips_empty_cameras_checks_remaining(self) -> None:
        """One empty + two identical => warning fires on the non-empty cameras."""
        ts = np.array([[0, 100], [200, 300]], dtype=np.int64)
        empty = np.empty((0, 2), dtype=np.int64)
        ctx = self._make_ctx({"cam_a": empty, "cam_b": ts.copy(), "cam_c": ts.copy()})
        self._assert_warning(ctx)


# ---------------------------------------------------------------------------
# warn_pose_coverage tests (multi-reader)
# ---------------------------------------------------------------------------

_POSE_COVERAGE_MARKER = "do not cover the full sequence interval"


class TestWarnPoseCoverage(unittest.TestCase):
    """Tests for the pose coverage warning checking all readers."""

    def _assert_warning(self, ctx: NcoreV4Context, expected_instance: str) -> None:
        with self.assertLogs(_validate_log, level=logging.INFO) as cm:
            warn_pose_coverage(ctx)
        msgs = [r.getMessage() for r in cm.records]
        hits = [m for m in msgs if _POSE_COVERAGE_MARKER in m and expected_instance in m]
        self.assertGreater(len(hits), 0, msg=f"expected warning for {expected_instance}; got: {msgs}")

    def _assert_no_warning(self, ctx: NcoreV4Context) -> None:
        with _capture_logs() as records:
            warn_pose_coverage(ctx)
        hits = [r.getMessage() for r in records if _POSE_COVERAGE_MARKER in r.getMessage()]
        self.assertEqual(hits, [], msg=f"unexpected warning: {hits}")

    def test_warns_on_mismatched_coverage(self) -> None:
        """Timestamps not spanning interval => warning."""
        T = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
        ts = np.array([500], dtype=np.uint64)
        ctx = _fake_ctx(
            interval=HalfClosedInterval.from_start_end(0, 1000),
            poses_readers={"default": _FakePoseReader(T, ts)},
        )
        self._assert_warning(ctx, "default")

    def test_silent_when_coverage_matches(self) -> None:
        """Timestamps spanning interval exactly => no warning."""
        T = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        ts = np.array([0, 1000], dtype=np.uint64)
        ctx = _fake_ctx(
            interval=HalfClosedInterval.from_start_end(0, 1000),
            poses_readers={"default": _FakePoseReader(T, ts)},
        )
        self._assert_no_warning(ctx)

    def test_checks_all_readers(self) -> None:
        """Both pose readers are checked, not just the first."""
        T_good = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        ts_good = np.array([0, 1000], dtype=np.uint64)
        T_bad = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
        ts_bad = np.array([500], dtype=np.uint64)
        ctx = _fake_ctx(
            interval=HalfClosedInterval.from_start_end(0, 1000),
            poses_readers={
                "reader_good": _FakePoseReader(T_good, ts_good),
                "reader_bad": _FakePoseReader(T_bad, ts_bad),
            },
        )
        self._assert_warning(ctx, "reader_bad")


# ---------------------------------------------------------------------------
# check_poses_rig_world_in_interval tests
# ---------------------------------------------------------------------------


class TestCheckPosesRigWorldInInterval(unittest.TestCase):
    """Tests for the hard check requiring in-interval rig->world poses."""

    def _make_ctx(self, poses_readers: dict[str, _FakePoseReader]) -> NcoreV4Context:
        return _fake_ctx(
            interval=HalfClosedInterval.from_start_end(0, 1000),
            poses_readers=poses_readers,
            errors=[],
        )

    def test_fails_when_no_pose_readers(self) -> None:
        """Absent pose readers must fail, not silently pass."""
        ctx = self._make_ctx({})
        check_poses_rig_world_in_interval(ctx)
        self.assertEqual(len(ctx.errors), 1)
        self.assertIn("no rig→world dynamic pose found", ctx.errors[0])

    def test_fails_when_poses_outside_interval(self) -> None:
        """Readers present but timestamps outside the interval => fail."""
        T = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        ts = np.array([0, 5000], dtype=np.uint64)  # 5000 >= stop (1000)
        ctx = self._make_ctx({"default": _FakePoseReader(T, ts)})
        check_poses_rig_world_in_interval(ctx)
        self.assertEqual(len(ctx.errors), 1)
        self.assertIn("timestamps outside interval", ctx.errors[0])

    def test_passes_when_poses_in_interval(self) -> None:
        """Readers with all timestamps inside the interval => pass."""
        T = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        ts = np.array([0, 500], dtype=np.uint64)
        ctx = self._make_ctx({"default": _FakePoseReader(T, ts)})
        check_poses_rig_world_in_interval(ctx)
        self.assertEqual(ctx.errors, [])

    def test_fails_when_any_reader_outside_interval(self) -> None:
        """One valid reader does not mask another reader's out-of-interval trajectory."""
        T = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        bad = _FakePoseReader(T, np.array([0, 5000], dtype=np.uint64))
        good = _FakePoseReader(T, np.array([0, 500], dtype=np.uint64))
        ctx = self._make_ctx({"bad": bad, "good": good})
        check_poses_rig_world_in_interval(ctx)
        self.assertEqual(len(ctx.errors), 1)
        self.assertIn("'bad': timestamps outside interval", ctx.errors[0])


# ---------------------------------------------------------------------------
# check_sequence_id_nurec_safe tests
# ---------------------------------------------------------------------------


class TestCheckSequenceIdNurecSafe(unittest.TestCase):
    """Tests for ``check_sequence_id_nurec_safe``."""

    @staticmethod
    def _make_ctx(sequence_id: str) -> NcoreV4Context:
        """Create a minimal context containing a sequence ID.

        Args:
            sequence_id: Sequence ID exposed by the fake reader.

        Returns:
            A context accepted by ``check_sequence_id_nurec_safe``.
        """
        return _fake_ctx(
            reader=types.SimpleNamespace(sequence_id=sequence_id),
            errors=[],
        )

    def test_accepts_filename_safe_sequence_id(self) -> None:
        """Alphanumeric, underscore, and hyphen characters pass."""
        ctx = self._make_ctx("nuro_20250103-00171")

        check_sequence_id_nurec_safe(ctx)

        self.assertEqual(ctx.errors, [])

    def test_rejects_dotted_sequence_id_with_aux_data_context(self) -> None:
        """A dot fails with an explanation of filename-prefix discovery."""
        ctx = self._make_ctx("18544.0_18574.0")

        check_sequence_id_nurec_safe(ctx)

        self.assertEqual(len(ctx.errors), 1)
        self.assertIn("truncates the prefix", ctx.errors[0])
        self.assertIn("dataset.aux_data", ctx.errors[0])

    def test_rejects_path_separator_and_whitespace(self) -> None:
        """Path separators and whitespace fail."""
        for sequence_id in ("route/segment", "route segment", "route\tsegment"):
            with self.subTest(sequence_id=sequence_id):
                ctx = self._make_ctx(sequence_id)

                check_sequence_id_nurec_safe(ctx)

                self.assertEqual(len(ctx.errors), 1)
                self.assertIn("unsafe character", ctx.errors[0])


# ---------------------------------------------------------------------------
# check_masks_nurec / warn_masks_ego_key tests
# ---------------------------------------------------------------------------


class _FakeMasksReader:
    """Minimal MasksComponent.Reader stand-in for mask-key tests."""

    def __init__(self, camera_mask_names: dict[str, list[str]]) -> None:
        self._camera_mask_names = camera_mask_names

    def get_camera_mask_names(self, camera_id: str) -> list[str]:
        if camera_id not in self._camera_mask_names:
            raise KeyError(camera_id)
        return self._camera_mask_names[camera_id]


def _make_masks_ctx(
    masks_readers: dict,
    camera_ids: list[str] | None = None,
) -> NcoreV4Context:
    cameras = {c: object() for c in (camera_ids or ["cam_front", "cam_rear"])}
    return _fake_ctx(
        masks_readers=masks_readers,
        camera_readers=cameras,
        errors=[],
    )


class TestCheckMasksNurec(unittest.TestCase):
    """Tests for ``check_masks_nurec`` (hard error for missing component)."""

    def test_fails_when_no_masks_component(self) -> None:
        """No masks readers at all — the 'component present' check fires."""
        ctx = _make_masks_ctx({})
        check_masks_nurec(ctx)
        self.assertEqual(len(ctx.errors), 1)
        self.assertIn("MasksComponent not found", ctx.errors[0])

    def test_passes_when_masks_component_present(self) -> None:
        """Masks reader exists — no error."""
        reader = _FakeMasksReader({"cam_front": ["ego"]})
        ctx = _make_masks_ctx({"default": reader})
        check_masks_nurec(ctx)
        self.assertEqual(ctx.errors, [])


class TestWarnMasksEgoKey(unittest.TestCase):
    """Tests for ``warn_masks_ego_key`` (soft warnings for ego key naming)."""

    def test_passes_when_ego_key_present(self) -> None:
        """All cameras have ego key — no warning."""
        reader = _FakeMasksReader({"cam_front": ["ego", "sky"], "cam_rear": ["ego"]})
        ctx = _make_masks_ctx({"default": reader})
        with _capture_logs() as records:
            warn_masks_ego_key(ctx)
        msgs = [r.getMessage() for r in records]
        self.assertFalse(any("ego" in m for m in msgs))

    def test_warns_when_ego_like_key_misspelled(self) -> None:
        """Camera has 'ego_vehicle' but not 'ego' — warning is logged."""
        reader = _FakeMasksReader({"cam_front": ["ego_vehicle", "sky"], "cam_rear": ["ego"]})
        ctx = _make_masks_ctx({"default": reader})
        with _capture_logs() as records:
            warn_masks_ego_key(ctx)
        msgs = [r.getMessage() for r in records]
        self.assertTrue(any("ego-like" in m and "cam_front" in m for m in msgs))

    def test_info_when_masks_have_no_ego_key(self) -> None:
        """Camera has masks but none are ego-related — informational note."""
        reader = _FakeMasksReader({"cam_front": ["sky"], "cam_rear": ["sky"]})
        ctx = _make_masks_ctx({"default": reader})
        with _capture_logs() as records:
            warn_masks_ego_key(ctx)
        msgs = [r.getMessage() for r in records]
        self.assertFalse(any("ego-like" in m for m in msgs))
        self.assertTrue(any('no "ego" key' in m for m in msgs))

    def test_no_warning_when_masks_empty_for_camera(self) -> None:
        """Camera has an empty mask list (no masks written) — no warning."""
        reader = _FakeMasksReader({"cam_front": [], "cam_rear": []})
        ctx = _make_masks_ctx({"default": reader})
        with _capture_logs() as records:
            warn_masks_ego_key(ctx)
        msgs = [r.getMessage() for r in records]
        self.assertFalse(any("ego" in m for m in msgs))

    def test_no_warning_when_camera_not_in_masks_component(self) -> None:
        """Camera not present in the masks component at all — no warning."""
        reader = _FakeMasksReader({})
        ctx = _make_masks_ctx({"default": reader})
        with _capture_logs() as records:
            warn_masks_ego_key(ctx)
        msgs = [r.getMessage() for r in records]
        self.assertFalse(any("ego" in m for m in msgs))


# ---------------------------------------------------------------------------
# warn_identity_sensor_translations tests
# ---------------------------------------------------------------------------

_IDENTITY_TRANSLATION_MARKER = "all sensor-to-rig translations are near-zero"


class _FakePoseReaderStatic:
    """Minimal pose reader returning pre-built static sensor-to-rig poses.

    ``static_poses`` maps ``sensor_id`` to a 4x4 float32 SE(3) matrix.
    ``get_static_pose(source, "rig")`` returns the matrix for *source* or
    raises ``KeyError`` when the sensor is absent.
    """

    def __init__(self, static_poses: dict[str, np.ndarray]) -> None:
        self._static = static_poses

    def get_static_pose(self, source: str, target: str) -> np.ndarray:
        if target == "rig" and source in self._static:
            return self._static[source]
        raise KeyError(f"{source}->{target}")

    def get_dynamic_pose(self, source: str, target: str):
        raise KeyError(f"{source}->{target}")


def _T_with_translation(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    """Build a 4x4 identity-rotation pose with the given translation."""
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


class TestWarnIdentitySensorTranslations(unittest.TestCase):
    """Tests for ``warn_identity_sensor_translations``."""

    def _make_ctx(
        self,
        camera_ids: list[str],
        lidar_ids: list[str],
        static_poses: dict[str, np.ndarray],
    ) -> NcoreV4Context:
        return _fake_ctx(
            camera_readers={c: object() for c in camera_ids},
            lidar_readers={ld: object() for ld in lidar_ids},
            poses_readers={"default": _FakePoseReaderStatic(static_poses)},
        )

    def _assert_warning(self, ctx: NcoreV4Context) -> None:
        with self.assertLogs(_validate_log, level=logging.INFO) as cm:
            warn_identity_sensor_translations(ctx)
        msgs = [r.getMessage() for r in cm.records]
        hits = [m for m in msgs if _IDENTITY_TRANSLATION_MARKER in m]
        self.assertGreater(len(hits), 0, msg=f"expected warning; got: {msgs}")

    def _assert_no_warning(self, ctx: NcoreV4Context) -> None:
        with _capture_logs() as records:
            warn_identity_sensor_translations(ctx)
        hits = [r.getMessage() for r in records if _IDENTITY_TRANSLATION_MARKER in r.getMessage()]
        self.assertEqual(hits, [], msg=f"unexpected warning: {hits}")

    def test_warns_when_all_translations_zero(self) -> None:
        """All sensors have zero translation => warning fires."""
        static = {
            "cam_front": _T_with_translation(),
            "cam_rear": _T_with_translation(),
            "lidar_top": _T_with_translation(),
        }
        ctx = self._make_ctx(["cam_front", "cam_rear"], ["lidar_top"], static)
        self._assert_warning(ctx)

    def test_silent_when_lidar_has_mount_height(self) -> None:
        """LiDAR has a 1.8 m mount height => no warning."""
        static = {
            "cam_front": _T_with_translation(),
            "cam_rear": _T_with_translation(),
            "lidar_top": _T_with_translation(tz=1.8),
        }
        ctx = self._make_ctx(["cam_front", "cam_rear"], ["lidar_top"], static)
        self._assert_no_warning(ctx)

    def test_warns_when_all_below_threshold(self) -> None:
        """Translations of 5 mm (< 1 cm threshold) => warning fires."""
        static = {
            "cam_front": _T_with_translation(tx=0.005),
            "lidar_top": _T_with_translation(ty=0.005),
        }
        ctx = self._make_ctx(["cam_front"], ["lidar_top"], static)
        self._assert_warning(ctx)

    def test_silent_when_translation_above_threshold(self) -> None:
        """Translations of 2 cm (above the 1 cm threshold) => no warning.

        Stored as float32 in the SE(3) pose matrices; values right at the
        threshold (0.01) hit float32-representation rounding (the nearest
        representable value is ~0.00999999978, slightly below 0.01) and
        would spuriously fire the warning.  Picking 0.02 keeps the test
        well clear of float32 rounding while still covering the
        "translation above threshold => no warning" intent.
        """
        static = {
            "cam_front": _T_with_translation(tx=0.02),
            "lidar_top": _T_with_translation(ty=0.02),
        }
        ctx = self._make_ctx(["cam_front"], ["lidar_top"], static)
        self._assert_no_warning(ctx)

    def test_skips_single_sensor(self) -> None:
        """Only one sensor total => skip (single mount at origin is plausible)."""
        static = {"cam_front": _T_with_translation()}
        ctx = self._make_ctx(["cam_front"], [], static)
        self._assert_no_warning(ctx)

    def test_skips_no_sensors(self) -> None:
        """No sensors at all => early return, no warning."""
        ctx = self._make_ctx([], [], {})
        self._assert_no_warning(ctx)

    def test_skips_when_pose_absent_for_sensor(self) -> None:
        """One sensor has no static pose in any reader => suppress warning."""
        static = {
            "cam_front": _T_with_translation(),
            # lidar_top intentionally absent
        }
        ctx = self._make_ctx(["cam_front"], ["lidar_top"], static)
        self._assert_no_warning(ctx)

    def test_silent_when_only_cameras_zero_but_lidar_present(self) -> None:
        """Cameras at origin but LiDAR has a real mount offset => no warning."""
        static = {
            "cam_front": _T_with_translation(),
            "lidar_top": _T_with_translation(tx=0.5, tz=1.8),
        }
        ctx = self._make_ctx(["cam_front"], ["lidar_top"], static)
        self._assert_no_warning(ctx)


# ---------------------------------------------------------------------------
# warn_camera_optical_axis_tilt tests
# ---------------------------------------------------------------------------

_CAMERA_TILT_MARKER = "camera optical axis far from horizontal"


def _T_canonical_camera_to_rig(tx: float = 0.0, ty: float = 0.0, tz: float = 1.5) -> np.ndarray:
    """Return ``T_camera_rig`` for an NCore-canonical front-mounted camera.

    OpenCV camera convention is +X right / +Y down / +Z forward; NCore rig
    convention is +X forward / +Y left / +Z up.  ``T_camera_rig`` maps a
    point in the camera frame to a point in the rig frame:

      rig +X (forward)  <-  camera +Z
      rig +Y (left)     <- -camera +X (camera +X is rig -Y)
      rig +Z (up)       <- -camera +Y (camera +Y is rig -Z)

    so the rotation matrix has columns equal to where each camera axis
    lands in the rig frame.  In particular the third column (``camera +Z``
    in rig coords) is rig ``+X`` -- the optical axis points forward, with
    zero ``z_rig`` component.
    """
    R = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = (tx, ty, tz)
    return T


def _T_rotated_axis(T: np.ndarray, axis: str, angle_rad: float) -> np.ndarray:
    """Apply a rig-frame rotation about ``axis`` to ``T_camera_rig``."""
    R = Rotation.from_euler(axis, angle_rad).as_matrix()
    T_out = T.copy()
    T_out[:3, :3] = R @ T[:3, :3]
    T_out[:3, 3] = R @ T[:3, 3]
    return T_out


class _FakeCameraReaderTilt:
    """Minimal camera reader stand-in (only the camera-id key is read)."""


def _make_tilt_ctx(
    T_camera_rig_by_cam: dict[str, np.ndarray],
) -> NcoreV4Context:
    """Build a minimal ctx wired for ``warn_camera_optical_axis_tilt``."""
    cam_readers = {cam_id: _FakeCameraReaderTilt() for cam_id in T_camera_rig_by_cam}
    static_poses = {cam_id: T.astype(np.float32) for cam_id, T in T_camera_rig_by_cam.items()}
    return _fake_ctx(
        camera_readers=cam_readers,
        poses_readers={"default": _FakePoseReaderStatic(static_poses)},
    )


class TestWarnCameraOpticalAxisTilt(unittest.TestCase):
    """Tests for ``warn_camera_optical_axis_tilt``."""

    def _assert_warning(
        self,
        ctx: NcoreV4Context,
        camera_id: str,
        containing: str | None = None,
    ) -> None:
        with self.assertLogs(_validate_log, level=logging.INFO) as cm:
            warn_camera_optical_axis_tilt(ctx)
        msgs = [r.getMessage() for r in cm.records]
        hits = [m for m in msgs if _CAMERA_TILT_MARKER in m and camera_id in m]
        self.assertGreater(len(hits), 0, msg=f"expected warning for {camera_id}; got: {msgs}")
        if containing is not None:
            self.assertTrue(
                any(containing in m for m in hits),
                msg=f"expected substring {containing!r} in warning for {camera_id}; got: {hits}",
            )

    def _assert_no_warning(self, ctx: NcoreV4Context) -> None:
        with _capture_logs() as records:
            warn_camera_optical_axis_tilt(ctx)
        hits = [r.getMessage() for r in records if _CAMERA_TILT_MARKER in r.getMessage()]
        self.assertEqual(hits, [], msg=f"unexpected warning: {hits}")

    def test_silent_on_canonical_extrinsic(self) -> None:
        """NCore-canonical front camera => optical axis = rig +X (z=0) => silent."""
        ctx = _make_tilt_ctx({"cam_front": _T_canonical_camera_to_rig()})
        self._assert_no_warning(ctx)

    def test_warns_on_identity_extrinsic(self) -> None:
        """T_camera_rig = identity (missed source-to-NCore basis change for a
        +X-principal source camera) => optical axis = rig +Z (camera looks up)
        => warning fires with ``up`` direction."""
        T_bad = np.eye(4, dtype=np.float64)
        T_bad[:3, 3] = (0.0, 0.0, 1.5)
        ctx = _make_tilt_ctx({"cam_front": T_bad})
        self._assert_warning(ctx, "cam_front", containing="up")

    def test_warns_on_90deg_y_rotation(self) -> None:
        """Canonical front camera + 90 deg rotation about rig Y => camera now
        looks down (rig -Z) => warning fires with ``down`` direction."""
        T_bad = _T_rotated_axis(_T_canonical_camera_to_rig(), "y", np.pi / 2.0)
        ctx = _make_tilt_ctx({"cam_front": T_bad})
        self._assert_warning(ctx, "cam_front", containing="down")

    def test_silent_on_90deg_z_rotation(self) -> None:
        """Canonical front camera + 90 deg rotation about rig Z => camera now
        looks left, but the optical axis is still horizontal (z=0).  This is a
        documented limitation: pure rig-Z (yaw) rotation bugs are not caught
        by the tilt check; ncore_project_pc_to_img is the canonical visual
        check for that case."""
        T_bad = _T_rotated_axis(_T_canonical_camera_to_rig(), "z", np.pi / 2.0)
        ctx = _make_tilt_ctx({"cam_front": T_bad})
        self._assert_no_warning(ctx)

    def test_warns_only_for_tilted_camera_in_multi_cam_setup(self) -> None:
        """Mixed multi-camera setup: one canonical front, one tilted-up rear =>
        only the tilted camera warns."""
        T_good_front = _T_canonical_camera_to_rig(tx=2.0)
        T_bad_rear = np.eye(4, dtype=np.float64)
        T_bad_rear[:3, 3] = (-2.0, 0.0, 1.5)
        ctx = _make_tilt_ctx({"cam_front": T_good_front, "cam_rear": T_bad_rear})

        with self.assertLogs(_validate_log, level=logging.INFO) as cm:
            warn_camera_optical_axis_tilt(ctx)
        msgs = [r.getMessage() for r in cm.records]
        rear_hits = [m for m in msgs if _CAMERA_TILT_MARKER in m and "cam_rear" in m]
        front_hits = [m for m in msgs if _CAMERA_TILT_MARKER in m and "cam_front" in m]
        self.assertGreater(len(rear_hits), 0, msg=f"expected warning for cam_rear; got: {msgs}")
        self.assertEqual(front_hits, [], msg=f"unexpected warning for cam_front: {front_hits}")

    def test_skips_when_no_cameras(self) -> None:
        """No cameras => early return, no warning."""
        ctx = _make_tilt_ctx({})
        self._assert_no_warning(ctx)

    def test_skips_when_pose_absent_for_camera(self) -> None:
        """Camera registered but no static pose for it => silently skip."""
        ctx = _fake_ctx(
            camera_readers={"cam_front": _FakeCameraReaderTilt()},
            poses_readers={"default": _FakePoseReaderStatic({})},
        )
        self._assert_no_warning(ctx)


# ---------------------------------------------------------------------------
# warn_camera_optical_axis_azimuth tests
# ---------------------------------------------------------------------------

_CAMERA_AZIMUTH_MARKER = "camera optical-axis azimuth disagrees with id-implied direction"


def _make_azimuth_ctx(
    T_camera_rig_by_cam: dict[str, np.ndarray],
) -> NcoreV4Context:
    """Build a minimal ctx wired for ``warn_camera_optical_axis_azimuth``.

    Reuses the tilt fakes since both checks read the same fields
    (``camera_readers`` keys + ``poses_readers[*].get_static_pose``).
    """
    cam_readers = {cam_id: _FakeCameraReaderTilt() for cam_id in T_camera_rig_by_cam}
    static_poses = {cam_id: T.astype(np.float32) for cam_id, T in T_camera_rig_by_cam.items()}
    return _fake_ctx(
        camera_readers=cam_readers,
        poses_readers={"default": _FakePoseReaderStatic(static_poses)},
    )


class TestParseCameraDirectionAzimuthRad(unittest.TestCase):
    """Tests for the ``_parse_camera_direction_azimuth_rad`` id-token parser."""

    def test_cardinal_directions(self) -> None:
        """Cardinal direction tokens map to the expected NCore-rig azimuths."""
        cases = {
            "front_camera": 0.0,
            "back_camera": np.pi,
            "left_camera": np.pi / 2.0,
            "right_camera": -np.pi / 2.0,
        }
        for camera_id, expected in cases.items():
            with self.subTest(camera_id=camera_id):
                got = _parse_camera_direction_azimuth_rad(camera_id)
                assert got is not None
                self.assertAlmostEqual(got, expected, places=6)

    def test_diagonal_directions(self) -> None:
        """Forward/backward and left/right tokens combine to four diagonal azimuths."""
        cases = {
            "front_left_camera": np.pi / 4.0,
            "front_right_camera": -np.pi / 4.0,
            "back_left_camera": 3.0 * np.pi / 4.0,
            "back_right_camera": -3.0 * np.pi / 4.0,
        }
        for camera_id, expected in cases.items():
            with self.subTest(camera_id=camera_id):
                got = _parse_camera_direction_azimuth_rad(camera_id)
                assert got is not None
                self.assertAlmostEqual(got, expected, places=6)

    def test_synonyms_recognized(self) -> None:
        """``forward`` / ``rear`` / ``port`` / ``starboard`` are recognized synonyms."""
        cases = {
            "forward_cam": 0.0,
            "rear_cam": np.pi,
            "port_cam": np.pi / 2.0,
            "starboard_cam": -np.pi / 2.0,
        }
        for camera_id, expected in cases.items():
            with self.subTest(camera_id=camera_id):
                got = _parse_camera_direction_azimuth_rad(camera_id)
                assert got is not None
                self.assertAlmostEqual(got, expected, places=6)

    def test_case_insensitive_and_separator_agnostic(self) -> None:
        """Token matching is case-insensitive and works across ``_``/``-``/``.`` separators."""
        for camera_id in ("FRONT_CAM", "Front-Cam", "front.cam", "Cam-front-1"):
            with self.subTest(camera_id=camera_id):
                got = _parse_camera_direction_azimuth_rad(camera_id)
                assert got is not None
                self.assertAlmostEqual(got, 0.0, places=6)

    def test_no_token_returns_none(self) -> None:
        """Numeric / opaque ids carry no direction information => ``None``."""
        for camera_id in ("cam_0", "cam0", "sensor_a", "rgb1"):
            with self.subTest(camera_id=camera_id):
                self.assertIsNone(_parse_camera_direction_azimuth_rad(camera_id))

    def test_conflicting_tokens_returns_none(self) -> None:
        """``front_back_*`` / ``left_right_*`` => ambiguous => silent (return None)."""
        self.assertIsNone(_parse_camera_direction_azimuth_rad("front_back_camera"))
        self.assertIsNone(_parse_camera_direction_azimuth_rad("left_right_camera"))


class TestWarnCameraOpticalAxisAzimuth(unittest.TestCase):
    """Tests for ``warn_camera_optical_axis_azimuth``."""

    def _assert_warning(self, ctx: NcoreV4Context, camera_id: str) -> None:
        with self.assertLogs(_validate_log, level=logging.INFO) as cm:
            warn_camera_optical_axis_azimuth(ctx)
        msgs = [r.getMessage() for r in cm.records]
        hits = [m for m in msgs if _CAMERA_AZIMUTH_MARKER in m and camera_id in m]
        self.assertGreater(len(hits), 0, msg=f"expected warning for {camera_id}; got: {msgs}")

    def _assert_no_warning(self, ctx: NcoreV4Context) -> None:
        with _capture_logs() as records:
            warn_camera_optical_axis_azimuth(ctx)
        hits = [r.getMessage() for r in records if _CAMERA_AZIMUTH_MARKER in r.getMessage()]
        self.assertEqual(hits, [], msg=f"unexpected warning: {hits}")

    def test_silent_on_canonical_front_camera(self) -> None:
        """``front_camera`` with NCore-canonical extrinsic => optical axis = rig +X
        (azimuth 0 deg) matches id-implied 0 deg => silent."""
        ctx = _make_azimuth_ctx({"front_camera": _T_canonical_camera_to_rig()})
        self._assert_no_warning(ctx)

    def test_warns_on_90deg_yaw(self) -> None:
        """``front_camera`` rotated 90 deg about rig Z => optical axis lands on
        rig -Y (azimuth -90 deg) but id implies azimuth 0 deg => warning fires.
        This is exactly the whole-rig yaw bug the tilt check cannot see (the
        converter built ``T_camera_rig`` in a frame whose +X is not aligned
        with rig +X)."""
        T_bad = _T_rotated_axis(_T_canonical_camera_to_rig(), "z", -np.pi / 2.0)
        ctx = _make_azimuth_ctx({"front_camera": T_bad})
        self._assert_warning(ctx, "front_camera")

    def test_warns_on_180deg_yaw(self) -> None:
        """``front_camera`` rotated 180 deg about rig Z => optical axis lands on
        rig -X (azimuth 180 deg) but id implies 0 deg => warning fires."""
        T_bad = _T_rotated_axis(_T_canonical_camera_to_rig(), "z", np.pi)
        ctx = _make_azimuth_ctx({"front_camera": T_bad})
        self._assert_warning(ctx, "front_camera")

    def test_silent_on_30deg_yaw_within_tolerance(self) -> None:
        """``front_camera`` rotated 30 deg about rig Z => delta 30 deg < 60 deg
        threshold => silent.  Real AV cameras are routinely yawed 20-40 deg
        from their nominal direction (overlapping FOV, fish-eye corners)."""
        T_ok = _T_rotated_axis(_T_canonical_camera_to_rig(), "z", np.pi / 6.0)
        ctx = _make_azimuth_ctx({"front_camera": T_ok})
        self._assert_no_warning(ctx)

    def test_silent_on_numeric_id(self) -> None:
        """Camera id carries no direction token => silent regardless of extrinsic."""
        T_bad = _T_rotated_axis(_T_canonical_camera_to_rig(), "z", np.pi)
        ctx = _make_azimuth_ctx({"cam_0": T_bad})
        self._assert_no_warning(ctx)

    def test_silent_on_conflicting_tokens(self) -> None:
        """``front_back_camera`` is ambiguous => silent regardless of extrinsic."""
        T_bad = _T_rotated_axis(_T_canonical_camera_to_rig(), "z", np.pi)
        ctx = _make_azimuth_ctx({"front_back_camera": T_bad})
        self._assert_no_warning(ctx)

    def test_silent_when_optical_axis_near_vertical(self) -> None:
        """Identity extrinsic (camera looks straight up) is the tilt check's
        responsibility; the azimuth check stays silent so we don't double-warn."""
        T_bad = np.eye(4, dtype=np.float64)
        T_bad[:3, 3] = (0.0, 0.0, 1.5)
        ctx = _make_azimuth_ctx({"front_camera": T_bad})
        self._assert_no_warning(ctx)

    def test_warns_on_diagonal_camera_swap(self) -> None:
        """``front_left_camera`` whose optical axis lands at +X-Y (forward-right
        diagonal, the wrong corner) => 90 deg delta from id-implied +45 deg =>
        warning fires.  This is the front-left/front-right swap pattern from
        a whole-rig yaw bug."""
        T_canonical_fl = _T_rotated_axis(_T_canonical_camera_to_rig(), "z", np.pi / 4.0)
        T_bad_fl = _T_rotated_axis(T_canonical_fl, "z", -np.pi / 2.0)
        ctx = _make_azimuth_ctx({"front_left_camera": T_bad_fl})
        self._assert_warning(ctx, "front_left_camera")

    def test_warns_only_for_misaligned_camera_in_multi_cam_setup(self) -> None:
        """Mixed setup: well-aligned ``front_camera`` + 90-deg-yawed
        ``back_camera`` => only the back camera warns."""
        T_good = _T_canonical_camera_to_rig()
        T_bad = _T_rotated_axis(_T_canonical_camera_to_rig(), "z", np.pi / 2.0)
        ctx = _make_azimuth_ctx({"front_camera": T_good, "back_camera": T_bad})

        with self.assertLogs(_validate_log, level=logging.INFO) as cm:
            warn_camera_optical_axis_azimuth(ctx)
        msgs = [r.getMessage() for r in cm.records]
        back_hits = [m for m in msgs if _CAMERA_AZIMUTH_MARKER in m and "back_camera" in m]
        front_hits = [m for m in msgs if _CAMERA_AZIMUTH_MARKER in m and "front_camera" in m]
        self.assertGreater(len(back_hits), 0, msg=f"expected warning for back_camera; got: {msgs}")
        self.assertEqual(front_hits, [], msg=f"unexpected warning for front_camera: {front_hits}")

    def test_skips_when_no_cameras(self) -> None:
        """No cameras => early return, no warning."""
        ctx = _make_azimuth_ctx({})
        self._assert_no_warning(ctx)

    def test_skips_when_pose_absent_for_camera(self) -> None:
        """Camera registered but no static pose for it => silently skip."""
        ctx = _fake_ctx(
            camera_readers={"front_camera": _FakeCameraReaderTilt()},
            poses_readers={"default": _FakePoseReaderStatic({})},
        )
        self._assert_no_warning(ctx)


# ---------------------------------------------------------------------------
# validate_result.yaml producer-side coverage
# ---------------------------------------------------------------------------
#
# These tests pair with the consumer-side schema tests in
# ``validators/validation_base_test.py`` to pin the
# ``validate_result.yaml`` contract from both ends. Producer tests live
# here (next to ``validate_helpers.py``) so a regression in the
# collector wiring or YAML serialization surfaces as
# ``//skills/nurec-onboard/ncore_validation:validate_test FAILED`` rather
# than only via the end-to-end template test in
# ``//ncore_template/tests:ncore_example_test``.


class TestStartStopCollection(unittest.TestCase):
    """``start_collection`` / ``stop_collection`` lifecycle invariants."""

    def tearDown(self) -> None:
        """Reset the module-global collector between tests."""
        stop_collection()

    def test_start_returns_fresh_collector(self) -> None:
        """``start_collection`` returns a new empty collector each call."""
        collector = start_collection()
        self.assertIsInstance(collector, ResultCollector)
        self.assertEqual(collector.failed_checks, [])
        self.assertEqual(collector.warnings, [])
        self.assertEqual(collector.current_tier, "")
        self.assertIsNone(collector.current_group)

    def test_second_start_replaces_first(self) -> None:
        """A second ``start_collection`` overwrites the first (no merge).

        Documented contract: the module global is single-shot; nested
        runs would silently drop entries from the earlier collector.
        Pinning this here so any future change to a stack-style
        collector triggers a test refresh rather than a silent
        behavior change.
        """
        first = start_collection()
        first.failed_checks.append(FailedCheck("V4 Structure", None, "x", "msg"))
        second = start_collection()
        self.assertIsNot(first, second)
        self.assertEqual(second.failed_checks, [], "second start must not inherit first's entries")

    def test_stop_makes_helpers_no_op(self) -> None:
        """After ``stop_collection`` the helpers fall back to log-only."""
        collector = start_collection()
        stop_collection()
        _section("V4 Structure")
        _check("z", False, [], "msg")
        _warn("w")
        # The local reference is still valid; it just shouldn't have grown.
        self.assertEqual(collector.failed_checks, [])
        self.assertEqual(collector.warnings, [])


class TestCollectorWiring(unittest.TestCase):
    """``_section`` / ``_group`` / ``_check`` / ``_warn`` -> collector recording."""

    def setUp(self) -> None:
        """Activate a fresh collector for each test."""
        self.collector = start_collection()

    def tearDown(self) -> None:
        """Drop the active collector so cross-test state does not leak."""
        stop_collection()

    def test_section_resets_group_and_sets_tier(self) -> None:
        """``_section`` sets ``current_tier`` and clears any prior group."""
        _section("V4 Structure")
        _group("Cameras")
        self.assertEqual(self.collector.current_tier, "V4 Structure")
        self.assertEqual(self.collector.current_group, "Cameras")
        _section("Data Quality")
        self.assertEqual(self.collector.current_tier, "Data Quality")
        self.assertIsNone(self.collector.current_group, "_section must reset current_group")

    def test_check_failure_records_with_active_context(self) -> None:
        """A failed ``_check`` records tier/group/name/message from active context."""
        _section("V4 Structure")
        _group("Cameras")
        errors: list[str] = []
        _check("frame ts ordered", False, errors, "camera frame ts ordering violated: cam0")
        self.assertEqual(errors, ["camera frame ts ordering violated: cam0"])
        self.assertEqual(len(self.collector.failed_checks), 1)
        entry = self.collector.failed_checks[0]
        self.assertEqual(entry.tier, "V4 Structure")
        self.assertEqual(entry.group, "Cameras")
        self.assertEqual(entry.name, "frame ts ordered")
        self.assertEqual(entry.message, "camera frame ts ordering violated: cam0")

    def test_check_passing_does_not_record(self) -> None:
        """A passing ``_check`` neither appends to ``errors`` nor to the collector."""
        _section("V4 Structure")
        errors: list[str] = []
        _check("ok", True, errors)
        self.assertEqual(errors, [])
        self.assertEqual(self.collector.failed_checks, [])

    def test_check_defaults_message_to_label(self) -> None:
        """When ``error_msg`` is omitted, the failure message defaults to ``label``."""
        _section("V4 Structure")
        errors: list[str] = []
        _check("frame ts ordered", False, errors)
        self.assertEqual(self.collector.failed_checks[0].message, "frame ts ordered")
        self.assertEqual(errors, ["frame ts ordered"])

    def test_warn_records_with_active_context(self) -> None:
        """``_warn`` records tier/group/message and never touches the error list."""
        _section("Data Quality")
        _group("LiDAR")
        _warn("model_element mostly (0, 0)")
        self.assertEqual(len(self.collector.warnings), 1)
        warning = self.collector.warnings[0]
        self.assertIsInstance(warning, WarningResult)
        self.assertEqual(warning.tier, "Data Quality")
        self.assertEqual(warning.group, "LiDAR")
        self.assertEqual(warning.message, "model_element mostly (0, 0)")

    def test_check_outside_any_group_records_none(self) -> None:
        """A ``_check`` emitted after ``_section`` but before any ``_group`` records ``group=None``."""
        _section("V4 Structure")
        errors: list[str] = []
        _check("structural check", False, errors, "missing field")
        self.assertIsNone(self.collector.failed_checks[0].group)

    def test_emission_order_preserved(self) -> None:
        """Multiple records preserve their emission order across tiers and groups."""
        _section("V4 Structure")
        errors: list[str] = []
        _check("a", False, errors, "first failure")
        _warn("first warning")
        _section("Data Quality")
        _group("LiDAR")
        _check("b", False, errors, "second failure")
        _warn("second warning")
        self.assertEqual(
            [c.message for c in self.collector.failed_checks],
            ["first failure", "second failure"],
        )
        self.assertEqual(
            [w.message for w in self.collector.warnings],
            ["first warning", "second warning"],
        )
        self.assertEqual(self.collector.failed_checks[1].tier, "Data Quality")
        self.assertEqual(self.collector.failed_checks[1].group, "LiDAR")


class TestDumpValidateResult(unittest.TestCase):
    """``dump_validate_result`` YAML round-trip and field-order contract."""

    def _build_result(self) -> ValidateResult:
        """Return a representative :class:`ValidateResult` with mixed fields."""
        return ValidateResult(
            schema_version=VALIDATE_RESULT_SCHEMA_VERSION,
            program=VALIDATE_RESULT_PROGRAM,
            program_version="19.0.0",
            started_at="2026-05-06T22:00:00Z",
            ended_at="2026-05-06T22:00:01Z",
            duration_sec=1.25,
            exit_code=1,
            passed=False,
            component_groups=["/tmp/a.json", "/tmp/b.json"],
            sample_frames=16,
            failed_checks=[
                FailedCheck(
                    tier="V4 Structure",
                    group=None,
                    name="open sequence",
                    message="could not open sequence: ValueError: bad",
                ),
                FailedCheck(
                    tier="Data Quality",
                    group="LiDAR",
                    name="model accuracy",
                    message="LiDAR model angular error > 0.5 deg",
                ),
            ],
            warnings=[WarningResult(tier="Data Quality", group="Cameras", message="optical axis tilt")],
        )

    def test_round_trip_preserves_all_fields(self) -> None:
        """The YAML can be reloaded and every field round-trips through ``yaml.safe_load``."""
        result = self._build_result()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "validate_result.yaml"
            dump_validate_result(path, result)
            self.assertTrue(path.is_file())
            loaded = yaml.safe_load(path.read_text())

        self.assertEqual(loaded["schema_version"], VALIDATE_RESULT_SCHEMA_VERSION)
        self.assertEqual(loaded["program"], VALIDATE_RESULT_PROGRAM)
        self.assertEqual(loaded["program_version"], "19.0.0")
        self.assertEqual(loaded["exit_code"], 1)
        self.assertIs(loaded["passed"], False)
        self.assertEqual(loaded["component_groups"], ["/tmp/a.json", "/tmp/b.json"])
        self.assertEqual(loaded["sample_frames"], 16)
        self.assertEqual(len(loaded["failed_checks"]), 2)
        self.assertEqual(loaded["failed_checks"][0]["name"], "open sequence")
        self.assertIsNone(loaded["failed_checks"][0]["group"])
        self.assertEqual(loaded["failed_checks"][1]["group"], "LiDAR")
        self.assertEqual(len(loaded["warnings"]), 1)
        self.assertEqual(loaded["warnings"][0]["tier"], "Data Quality")

    def test_top_level_field_order_matches_dataclass(self) -> None:
        """The on-disk top-level field order matches :class:`ValidateResult` declaration order.

        ``dump_validate_result`` writes with ``sort_keys=False`` so
        operators reading the sidecar always see the schema receipt
        (``schema_version`` / ``program``) before the body.
        """
        result = self._build_result()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "validate_result.yaml"
            dump_validate_result(path, result)
            text = path.read_text()
        keys_in_order = [line.split(":", 1)[0] for line in text.splitlines() if line and line[0].isalpha()]
        expected_prefix = [
            "schema_version",
            "program",
            "program_version",
            "started_at",
            "ended_at",
            "duration_sec",
            "exit_code",
            "passed",
            "component_groups",
            "sample_frames",
            "failed_checks",
        ]
        self.assertEqual(keys_in_order[: len(expected_prefix)], expected_prefix)

    def test_empty_collections_serialize_as_empty_lists(self) -> None:
        """``failed_checks: []`` / ``warnings: []`` survive the round-trip as empty lists."""
        result = ValidateResult(
            schema_version=VALIDATE_RESULT_SCHEMA_VERSION,
            program=VALIDATE_RESULT_PROGRAM,
            program_version="19.0.0",
            started_at="2026-05-06T22:00:00Z",
            ended_at="2026-05-06T22:00:01Z",
            duration_sec=0.0,
            exit_code=0,
            passed=True,
            component_groups=[],
            sample_frames=None,
            failed_checks=[],
            warnings=[],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "validate_result.yaml"
            dump_validate_result(path, result)
            loaded = yaml.safe_load(path.read_text())
        self.assertEqual(loaded["failed_checks"], [])
        self.assertEqual(loaded["warnings"], [])
        self.assertEqual(loaded["component_groups"], [])
        self.assertIsNone(loaded["sample_frames"])

    def test_replaces_existing_file_atomically(self) -> None:
        """A second dump overwrites the first without leaving stale content."""
        result_a = self._build_result()
        result_b = ValidateResult(
            schema_version=VALIDATE_RESULT_SCHEMA_VERSION,
            program=VALIDATE_RESULT_PROGRAM,
            program_version="20.0.0",
            started_at="2026-05-07T00:00:00Z",
            ended_at="2026-05-07T00:00:02Z",
            duration_sec=2.0,
            exit_code=0,
            passed=True,
            component_groups=[],
            sample_frames=None,
            failed_checks=[],
            warnings=[],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "validate_result.yaml"
            dump_validate_result(path, result_a)
            dump_validate_result(path, result_b)
            loaded = yaml.safe_load(path.read_text())
        self.assertIs(loaded["passed"], True)
        self.assertEqual(loaded["program_version"], "20.0.0")
        self.assertEqual(loaded["failed_checks"], [])

    def test_no_tempfile_residue_on_success(self) -> None:
        """The temp file used by the atomic write is gone after a successful dump."""
        result = self._build_result()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "validate_result.yaml"
            dump_validate_result(path, result)
            siblings = sorted(p.name for p in Path(td).iterdir())
        self.assertEqual(siblings, ["validate_result.yaml"])

    def test_failed_dump_does_not_overwrite_existing_file(self) -> None:
        """A serialization failure leaves the prior sidecar intact (atomic write).

        Simulates a serialization-time crash by passing a value
        ``yaml.safe_dump`` cannot represent (an arbitrary Python object).
        The previous sidecar must still be readable, and no temp residue
        should remain.
        """

        class _Unserializable:
            pass

        result = self._build_result()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "validate_result.yaml"
            dump_validate_result(path, result)
            original = path.read_text()

            bad = self._build_result()
            bad.component_groups = [_Unserializable()]  # type: ignore[list-item]
            with self.assertRaises(yaml.YAMLError):
                dump_validate_result(path, bad)

            self.assertEqual(path.read_text(), original)
            siblings = sorted(p.name for p in Path(td).iterdir())
        self.assertEqual(siblings, ["validate_result.yaml"])


class TestValidateMainStaleSidecarOnDumpFailure(unittest.TestCase):
    """``main`` removes a pre-existing sidecar when the fresh dump fails.

    Otherwise a downstream reader would see the prior run's ``passed:
    true`` content and treat it as the current run's outcome. The
    atomic-write contract of :func:`dump_validate_result` guarantees
    the old file is still in place after a dump failure, so ``main``
    must explicitly clear it.

    Also pins the success-banner ordering: ``"Validation passed."`` is
    emitted only when the sidecar write completes, never before, so
    stdout and ``exit_code`` cannot disagree on a dump-failure path.
    """

    def test_stale_sidecar_removed_when_dump_fails(self) -> None:
        """A prior sidecar is unlinked and the success banner is suppressed when ``dump_validate_result`` raises."""
        from ncore_validation import validate

        callback = validate.main.callback
        assert callback is not None
        with tempfile.TemporaryDirectory() as td:
            stale = Path(td) / "validate_result.yaml"
            stale.write_text("schema_version: 1\npassed: true\nexit_code: 0\n")

            with _capture_logs() as records:
                with mock.patch.object(validate, "dump_validate_result", side_effect=OSError("disk full")):
                    with self.assertRaises(SystemExit) as cm:
                        callback(
                            component_groups=(),
                            open_consolidated=False,
                            verbose=False,
                            sample_frames=None,
                            output_result=stale,
                        )
            self.assertEqual(cm.exception.code, 1)
            self.assertFalse(stale.exists(), "stale sidecar must be removed when fresh dump fails")
            messages = [r.getMessage() for r in records]
            self.assertFalse(
                any("Validation passed" in m for m in messages),
                msg=f"success banner must not be emitted when the sidecar write fails; got: {messages}",
            )


class _FakeTrackedCuboidObservation:
    def __init__(self, ts: int, track_id: str, source: str = "EXTERNAL", source_version: str | None = None) -> None:
        self.timestamp_us = ts
        self.track_id = track_id
        self.source = types.SimpleNamespace(name=source)
        self.source_version = source_version


class _FakeTrackedCuboidReader:
    def __init__(self, observations: list[tuple]) -> None:
        self._obs = [_FakeTrackedCuboidObservation(*observation) for observation in observations]

    def get_observations(self):
        return iter(self._obs)


def _tracked_cuboid_ctx(*components: list[tuple]) -> NcoreV4Context:
    """Build a context with one fake cuboid component per observation list."""
    return _fake_ctx(
        cuboid_readers={f"component_{i}": _FakeTrackedCuboidReader(obs) for i, obs in enumerate(components)},
        errors=[],
    )


class TestCuboidObservationTimestampCollisions(unittest.TestCase):
    """Tests for the exact-source duplicate check and the cross-source collision warning."""

    TS = 1_300_000
    _WARNING = "more than one label source or version"

    def _run(self, *components: list[tuple]) -> tuple[list[str], list[str]]:
        ctx = _tracked_cuboid_ctx(*components)
        with _capture_logs() as records:
            check_cuboid_duplicate_observation_timestamps(ctx)
            warn_cuboid_cross_source_timestamp_collisions(ctx)
        return ctx.errors, [r.getMessage() for r in records if self._WARNING in r.getMessage()]

    def test_same_exact_source_and_version_fails(self):
        errors, warnings = self._run([(self.TS, "track-a", "HUMAN", "v1"), (self.TS, "track-a", "HUMAN", "v1")])
        self.assertEqual(len(errors), 1)
        self.assertIn("component_0:track-a@1300000 (HUMAN@v1)", errors[0])
        self.assertEqual(warnings, [])

    def test_different_versions_of_one_source_warn(self):
        errors, warnings = self._run([(self.TS, "track-a", "HUMAN", "v1"), (self.TS, "track-a", "HUMAN", "v2")])
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("HUMAN@v1, HUMAN@v2", warnings[0])

    def test_different_source_types_warn(self):
        errors, warnings = self._run([(self.TS, "track-a", "HUMAN"), (self.TS, "track-a", "AUTOLABEL")])
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("AUTOLABEL@any, HUMAN@any", warnings[0])

    def test_distinct_tracks_at_one_timestamp_pass(self):
        errors, warnings = self._run([(self.TS, "track-a"), (self.TS, "track-b")])
        self.assertEqual((errors, warnings), ([], []))

    def test_same_track_timestamp_in_separate_components_passes(self):
        errors, warnings = self._run([(self.TS, "track-a", "HUMAN")], [(self.TS, "track-a", "AUTOLABEL")])
        self.assertEqual((errors, warnings), ([], []))

    def test_skips_without_cuboids(self):
        ctx = _tracked_cuboid_ctx()
        check_cuboid_duplicate_observation_timestamps(ctx)
        warn_cuboid_cross_source_timestamp_collisions(ctx)
        self.assertEqual(ctx.errors, [])


class TestWarnCuboidTracksBelowTwoObservations(unittest.TestCase):
    """Tests for ``warn_cuboid_tracks_below_two_observations``."""

    _WARNING = "fewer than two observations"

    def _warnings(self, *components: list[tuple]) -> list[str]:
        ctx = _tracked_cuboid_ctx(*components)
        with _capture_logs() as records:
            warn_cuboid_tracks_below_two_observations(ctx)
        self.assertEqual(ctx.errors, [])
        return [r.getMessage() for r in records if self._WARNING in r.getMessage()]

    def test_component_without_observations_is_silent(self):
        self.assertEqual(self._warnings([]), [])

    def test_single_observation_track_warns_with_its_identity(self):
        warnings = self._warnings([(100, "track-a", "HUMAN", "v1"), (100, "track-b"), (200, "track-b")])
        self.assertEqual(len(warnings), 1)
        self.assertIn("component_0:track-a (HUMAN@v1, 1 observation)", warnings[0])
        self.assertNotIn("track-b", warnings[0])

    def test_two_observation_track_is_silent(self):
        self.assertEqual(self._warnings([(100, "track-a"), (200, "track-a")]), [])

    def test_label_sources_are_counted_independently(self):
        warnings = self._warnings([(100, "track-a", "HUMAN"), (200, "track-a", "AUTOLABEL")])
        self.assertEqual(len(warnings), 1)
        self.assertIn("2 cuboid track(s)", warnings[0])
        self.assertIn("track-a (AUTOLABEL@any", warnings[0])
        self.assertIn("track-a (HUMAN@any", warnings[0])


class TestLidarFramesSubsampled(unittest.TestCase):
    """Tests for the gate that skips the cuboid-LiDAR checks under --sample-frames."""

    @staticmethod
    def _ctx(frame_counts: list[int]) -> NcoreV4Context:
        readers = {
            f"lidar_{i}": types.SimpleNamespace(frames_timestamps_us=np.zeros((n, 2), dtype=np.int64))
            for i, n in enumerate(frame_counts)
        }
        return _fake_ctx(lidar_readers=readers)

    def test_flag_absent_or_non_positive_is_a_full_sweep(self):
        from ncore_validation import validate

        ctx = self._ctx([300])
        self.assertFalse(validate._lidar_frames_subsampled(ctx, None))
        self.assertFalse(validate._lidar_frames_subsampled(ctx, 0))

    def test_request_larger_than_the_shard_is_a_full_sweep(self):
        """`_sample_frame_indices` keeps every frame here, so the checks must run."""
        from ncore_validation import validate

        self.assertFalse(validate._lidar_frames_subsampled(self._ctx([300]), 10_000))
        self.assertFalse(validate._lidar_frames_subsampled(self._ctx([300]), 300))

    def test_any_thinned_sensor_gates_the_checks(self):
        from ncore_validation import validate

        self.assertTrue(validate._lidar_frames_subsampled(self._ctx([300]), 16))
        self.assertTrue(validate._lidar_frames_subsampled(self._ctx([8, 300]), 16))


class TestLidarCoverageIndex(unittest.TestCase):
    """Tests for the multi-LiDAR coverage union used by the cuboid checks."""

    @staticmethod
    def _ctx(frame_ends_by_sensor: dict[str, list[int]]) -> NcoreV4Context:
        """Build a ctx exposing each sensor's full frame cadence."""
        readers = {}
        for sensor_id, ends in frame_ends_by_sensor.items():
            ts = np.array([[e - 50_000, e] for e in ends], dtype=np.int64)
            readers[sensor_id] = types.SimpleNamespace(frames_timestamps_us=ts)
        return _fake_ctx(lidar_readers=readers)

    def test_unions_every_sensors_nearest_frame(self):
        """A sector unit's closer frame key must not displace the roof unit's points.

        The regression: on a rig with one roof spinner and four short-range
        sector units, picking the single globally-nearest frame key reduces the
        shard to whichever sensor's key interleaves closest to the cuboid
        timestamp -- usually a sector unit that cannot see the object.
        """
        roof = np.array([[10.0, 0.0, 0.0]])
        sector = np.array([[1.0, 0.0, 0.0]])
        ctx = self._ctx({"roof": [100_000, 200_000], "sector": [99_000, 199_000]})
        index = LidarCoverageIndex(ctx, {"roof": {100_000: roof}, "sector": {99_000: sector}})
        pts = index.points_near(99_000)
        self.assertEqual(pts.shape[0], 2, msg="both sensors must contribute despite the closer sector key")
        self.assertEqual(index._last_keys, (("roof", 100_000), ("sector", 99_000)))

    def test_picks_the_nearest_frame_within_each_sensor(self):
        near = np.array([[1.0, 2.0, 3.0]])
        far = np.array([[9.0, 9.0, 9.0]])
        ctx = self._ctx({"roof": [100_000, 200_000, 300_000, 400_000, 500_000]})
        index = LidarCoverageIndex(ctx, {"roof": {100_000: near, 500_000: far}})
        np.testing.assert_allclose(index.points_near(90_000), near)

    def test_excludes_a_sensor_whose_nearest_frame_is_stale(self):
        """A sensor that stopped recording must not contribute an old cloud.

        Its nearest frame is still nearest, but it describes the scene seconds
        earlier, which reads to the cuboid checks as the object having no LiDAR
        support where another sensor covers it perfectly.
        """
        roof = np.array([[10.0, 0.0, 0.0]])
        stopped = np.array([[1.0, 0.0, 0.0]])
        ctx = self._ctx({"roof": [1_000_000, 1_100_000], "stopped": [100_000, 200_000]})
        index = LidarCoverageIndex(ctx, {"roof": {1_100_000: roof}, "stopped": {200_000: stopped}})
        np.testing.assert_allclose(index.points_near(1_100_000), roof)

    def test_returns_none_when_every_sensor_is_stale(self):
        ctx = self._ctx({"roof": [100_000, 200_000]})
        index = LidarCoverageIndex(ctx, {"roof": {200_000: np.array([[1.0, 0.0, 0.0]])}})
        self.assertIsNone(index.points_near(9_000_000))

    def test_keeps_a_sensor_within_its_own_cadence(self):
        """An observation inside the stored frame window receives its points."""
        pts = np.array([[1.0, 0.0, 0.0]])
        ctx = self._ctx({"slow": [0, 1_000_000, 2_000_000]})
        index = LidarCoverageIndex(ctx, {"slow": {1_000_000: pts}})
        np.testing.assert_allclose(index.points_near(975_000), pts)

    def test_returns_none_without_points(self):
        self.assertIsNone(LidarCoverageIndex(self._ctx({}), {}).points_near(0))
        self.assertIsNone(LidarCoverageIndex(self._ctx({"roof": [0, 1]}), {"roof": {}}).points_near(0))


class TestSampleEvenly(unittest.TestCase):
    """Tests for the representative-sample helper."""

    def test_returns_everything_under_the_cap(self):
        self.assertEqual(sample_evenly([1, 2, 3], 10), [1, 2, 3])

    def test_non_positive_cap_returns_nothing(self):
        self.assertEqual(sample_evenly([1, 2, 3], 0), [])
        self.assertEqual(sample_evenly([1, 2, 3], -1), [])

    def test_spreads_across_the_list_rather_than_taking_the_head(self):
        items = list(range(1000))
        sampled = sample_evenly(items, 10)
        self.assertLessEqual(len(sampled), 10)
        self.assertEqual(sampled[0], 0)
        self.assertEqual(sampled[-1], 999)
        self.assertGreater(max(sampled) - min(sampled), 900)

    def test_sorts_by_key_before_sampling(self):
        items = [{"timestamp": timestamp} for timestamp in (900, 100, 700, 300, 500)]
        sampled = sample_evenly(items, 3, key=lambda item: item["timestamp"])
        self.assertEqual([item["timestamp"] for item in sampled], [100, 500, 900])


if __name__ == "__main__":
    unittest.main()
