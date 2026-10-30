# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic-data unit tests for the public LiDAR and camera helpers in ``ncore_example.utils``.

The following helpers are covered:

* ``sanitize_sequence_id`` -- replace dots in source names and reject
  characters that are unsafe in sequence store filenames and paths.
* ``cuboid_track_sample_timestamps`` -- keep in-coverage source samples, add
  a sample at each coverage boundary the track crosses, and reject repeats.
* ``check_cuboid_track_timestamps`` -- reject stored track timestamps that
  are resampled, dropped, or repeated relative to that selection.
* ``build_lidar_model_from_per_frame_azimuth`` -- per-frame azimuth
  stream → ``(column_azimuths_rad, per_frame_phase_offsets_rad,
  model_element_col_indices)``.  Exercised across phase-locked rotor,
  free-spinning rotor with dual-return run patterns, single-return,
  CW / CCW direction, and per-beam-offset extraction regimes.
* ``bin_per_ray_to_columns`` -- per-frame circular nearest-neighbor
  binning of per-ray azimuths to a known column schedule, used by all
  three ``row_azimuth_offsets_rad`` strategies (sensor-native +
  preserve-Δ, sensor-native + Δ encoded in column index,
  synthetic-uniform + Δ encoded in column index).
* ``build_pinhole_model_from_opencv_K_D`` -- OpenCV pinhole ``K``/``D``
  → NCore camera model. Shape, resolution, and distortion-vector
  length validation paths.
* ``verify_camera_model_projection`` -- fail-closed reprojection check against
  source-derived image-point references.
* ``solve_static_extrinsic_mean`` -- average a sequence of per-frame
  rigid transforms into a single static extrinsic. Edge cases such as
  empty timestamp lists.

The tests are dataset-agnostic: they synthesize inputs from first
principles and verify the helpers' contracts.
"""

from __future__ import annotations

import unittest

import numpy as np

from ncore.data import FThetaCameraModelParameters, OpenCVFisheyeCameraModelParameters, ShutterType

from ncore_example.utils import (
    MAX_CAMERA_MODEL_REPROJ_PX,
    MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG,
    bin_per_ray_to_columns,
    build_lidar_model_from_per_frame_azimuth,
    build_pinhole_model_from_opencv_K_D,
    check_cuboid_track_timestamps,
    cuboid_track_sample_timestamps,
    sanitize_sequence_id,
    solve_static_extrinsic_mean,
    verify_camera_model_projection,
)


class SanitizeSequenceIdTest(unittest.TestCase):
    """Tests for ``sanitize_sequence_id``."""

    def test_replaces_dots_and_preserves_other_safe_characters(self) -> None:
        """Dots become hyphens while safe characters remain unchanged."""
        self.assertEqual(
            sanitize_sequence_id("20250103_18544.0-18574.0"),
            "20250103_18544-0-18574-0",
        )

    def test_rejects_path_separator_and_whitespace(self) -> None:
        """Path separators and whitespace raise ``ValueError``."""
        for name in ("route/segment", "route segment", "route\tsegment"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "unsafe character"):
                    sanitize_sequence_id(name)


class CuboidTrackSampleTimestampsTest(unittest.TestCase):
    """Tests for ``cuboid_track_sample_timestamps`` over coverage ``[1000, 2000]``."""

    LO = 1000
    HI = 2000

    def _samples(self, source: list[int]) -> list[int]:
        return cuboid_track_sample_timestamps(np.array(source, dtype=np.int64), self.LO, self.HI).tolist()

    def test_crossed_boundaries_are_added_even_with_samples_inside(self) -> None:
        """Inside samples are kept and both crossed boundaries are still added."""
        self.assertEqual(self._samples([500, 1200, 1800, 2500]), [self.LO, 1200, 1800, self.HI])

    def test_uneven_cadence_keeps_coverage_to_the_far_boundary(self) -> None:
        """One-step extrapolation from 1100/1200 would not reach 2000, so the boundary is stored."""
        self.assertEqual(self._samples([500, 1100, 1200, 2500]), [self.LO, 1100, 1200, self.HI])

    def test_track_entirely_inside_gains_nothing(self) -> None:
        """A track that crosses no boundary stores exactly its source samples."""
        self.assertEqual(self._samples([1100, 1400, 1900]), [1100, 1400, 1900])

    def test_native_sample_on_a_boundary_is_not_duplicated(self) -> None:
        """A source sample already at a crossed boundary is stored once."""
        self.assertEqual(self._samples([500, 1000, 1500]), [self.LO, 1500])

    def test_unsorted_input_is_returned_sorted_int64(self) -> None:
        """Valid unsorted input is sorted without dropping or adding samples."""
        result = cuboid_track_sample_timestamps(np.array([1800, 1200, 1500]), self.LO, self.HI)
        self.assertEqual(result.dtype, np.int64)
        self.assertEqual(result.tolist(), [1200, 1500, 1800])

    def test_rejects_repeated_source_timestamps(self) -> None:
        """Repeated source timestamps are ambiguous and raise instead of collapsing."""
        with self.assertRaisesRegex(ValueError, r"repeats timestamp\(s\) \[1500\]"):
            cuboid_track_sample_timestamps(np.array([1500, 1200, 1500]), self.LO, self.HI)

    def test_bracketing_samples_outside_become_the_two_boundaries(self) -> None:
        """A stationary track stored as two endpoints outside the coverage keeps a span over every frame."""
        self.assertEqual(self._samples([500, 2500]), [self.LO, self.HI])

    def test_single_inside_sample_gains_both_crossed_boundaries(self) -> None:
        """One inside sample alone would be discarded, so both crossed boundaries are added."""
        self.assertEqual(self._samples([500, 1500, 2500]), [self.LO, 1500, self.HI])

    def test_single_inside_sample_gains_only_the_crossed_boundary(self) -> None:
        """A track starting inside and ending after the coverage gains only the end boundary."""
        self.assertEqual(self._samples([1500, 2500]), [1500, self.HI])

    def test_track_entirely_outside_on_one_side_stores_nothing(self) -> None:
        """A track that never overlaps the coverage is not recovered."""
        self.assertEqual(self._samples([100, 500]), [])

    def test_lone_source_sample_is_not_padded(self) -> None:
        """A single-sample source track inside the coverage crosses no boundary and gains nothing."""
        self.assertEqual(self._samples([1500]), [1500])

    def test_rejects_non_1d_input_and_reversed_coverage(self) -> None:
        """Malformed inputs raise ``ValueError``."""
        with self.assertRaisesRegex(ValueError, "1-D"):
            cuboid_track_sample_timestamps(np.zeros((2, 2), dtype=np.int64), self.LO, self.HI)
        with self.assertRaisesRegex(ValueError, "reversed"):
            cuboid_track_sample_timestamps(np.array([1500], dtype=np.int64), self.HI, self.LO)


class CheckCuboidTrackTimestampsTest(unittest.TestCase):
    """Tests for ``check_cuboid_track_timestamps`` over coverage ``[1000, 2000]``."""

    LO = 1000
    HI = 2000
    SOURCE = np.array([500, 1200, 1800, 2500], dtype=np.int64)

    def _check(self, stored: list[int]) -> None:
        check_cuboid_track_timestamps("track-a", np.array(stored, dtype=np.int64), self.SOURCE, self.LO, self.HI)

    def test_selected_timestamps_in_any_order_pass(self) -> None:
        """Exactly the selection passes regardless of order."""
        self._check([2000, 1200, 1000, 1800])

    def test_resampled_timestamps_fail(self) -> None:
        """A per-frame grid adds timestamps the source never had."""
        with self.assertRaisesRegex(ValueError, r"track-a: .*\(2 not in the source selection, 0 missing"):
            self._check([1000, 1100, 1200, 1500, 1800, 2000])

    def test_dropped_native_timestamp_fails(self) -> None:
        """Every in-coverage source observation must be stored."""
        with self.assertRaisesRegex(ValueError, r"0 not in the source selection, 1 missing"):
            self._check([1000, 1200, 2000])

    def test_repeated_timestamp_fails(self) -> None:
        """Storing one timestamp twice is rejected."""
        with self.assertRaisesRegex(ValueError, r"1 repeated"):
            self._check([1000, 1200, 1200, 1800, 2000])


def _emit_frame(
    column_azimuths_rad: np.ndarray,
    run_lengths: np.ndarray,
    n_rows: int,
    cyclic_shift: int = 0,
    fractional_phase_rad: float = 0.0,
    per_beam_offset_rad: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    noise_rad: float = 0.0,
) -> np.ndarray:
    """Synthesize a per-ray (row-varies-fastest) azimuth stream for one frame.

    For each unique azimuth ``column_azimuths_rad[i]``, emits
    ``run_lengths[i]`` raw columns of ``n_rows`` rays each.  Cyclically
    rotates the schedule by ``cyclic_shift`` columns and adds a global
    fractional phase drift, optionally with per-beam azimuth offsets and
    Gaussian noise.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    two_pi = 2.0 * np.pi
    n_unique = column_azimuths_rad.size
    rotated = np.roll(column_azimuths_rad, -cyclic_shift)
    per_col = np.repeat(rotated, run_lengths)
    n_raw_cols = per_col.size
    per_col = (per_col + fractional_phase_rad) % two_pi
    rays = np.tile(per_col, (n_rows, 1)).T.reshape(-1)  # row-varies-fastest
    if per_beam_offset_rad is not None:
        beam_idx = np.arange(rays.size) % n_rows
        rays = (rays + per_beam_offset_rad[beam_idx]) % two_pi
    if noise_rad > 0.0:
        rays = (rays + rng.normal(0.0, noise_rad, rays.size)) % two_pi
    assert rays.size == n_rows * n_raw_cols, (rays.size, n_rows, n_raw_cols, n_unique)
    return rays


class BuildLidarModelFromPerFrameAzimuthTest(unittest.TestCase):
    def setUp(self) -> None:
        # A non-uniform 8-azimuth reference schedule, CCW order in [0, 2*pi).
        # Mixed gaps (4 deg, 8 deg, 12 deg) so the schedule is not a linspace.
        deg = np.array([0.0, 4.0, 12.0, 16.0, 28.0, 40.0, 60.0, 80.0])
        self.col_az_ccw = np.deg2rad(deg)
        self.n_unique = self.col_az_ccw.size
        # Mixed 2x / 4x dual-return run pattern (sums to 20 raw columns).
        self.run_lengths = np.array([2, 2, 4, 2, 2, 4, 2, 2], dtype=np.int64)
        self.n_rows = 8

    def test_phase_locked_rotor_recovers_reference(self) -> None:
        """Every frame identical → phase ~ 0, indices recover the rotated schedule."""
        rng = np.random.default_rng(42)
        frames = [_emit_frame(self.col_az_ccw, self.run_lengths, self.n_rows, rng=rng) for _ in range(5)]
        col_az, phase_off, col_idx = build_lidar_model_from_per_frame_azimuth(frames, spinning_direction="ccw")
        np.testing.assert_allclose(col_az, self.col_az_ccw.astype(np.float32), atol=1e-6)
        # All phases ~ 0 (or 2*pi) since all frames identical.
        wrapped = np.minimum(phase_off, 2.0 * np.pi - phase_off)
        self.assertTrue(np.all(wrapped < 1e-3), f"phase_off={phase_off}")
        # Column indices must reproduce the run pattern: each unique azimuth at
        # firing-order position i appears in (run_lengths[i] * n_rows) consecutive rays.
        expected = np.repeat(np.arange(self.n_unique), self.run_lengths * self.n_rows)
        for f, idx in enumerate(col_idx):
            np.testing.assert_array_equal(idx, expected, err_msg=f"frame {f}")

    def test_free_spinning_rotor_recovers_cyclic_shift(self) -> None:
        """Per-frame integer cyclic shift → phase_off matches; bin indices follow rotation."""
        rng = np.random.default_rng(7)
        shifts = [0, 1, 3, 5, 7]
        frames = [
            _emit_frame(
                self.col_az_ccw,
                self.run_lengths,
                self.n_rows,
                cyclic_shift=k,
                rng=rng,
            )
            for k in shifts
        ]
        col_az, phase_off, col_idx = build_lidar_model_from_per_frame_azimuth(frames, spinning_direction="ccw")
        # Reference recovered.
        np.testing.assert_allclose(col_az, self.col_az_ccw.astype(np.float32), atol=1e-6)
        # Phase offset for shift k matches col_az[k] - col_az[0] (CCW).
        for f, k in enumerate(shifts):
            expected_phase = float((self.col_az_ccw[k] - self.col_az_ccw[0]) % (2.0 * np.pi))
            self.assertAlmostEqual(
                float(phase_off[f]),
                expected_phase,
                places=4,
                msg=f"frame {f} shift {k}",
            )
        # Column indices for frame f's first ray of each unique-azimuth run point at
        # cyclically rotated positions.
        run_starts = np.concatenate(([0], np.cumsum(self.run_lengths * self.n_rows)[:-1]))
        for f, k in enumerate(shifts):
            for i, rs in enumerate(run_starts):
                expected_col = (i + k) % self.n_unique
                self.assertEqual(
                    int(col_idx[f][rs]),
                    expected_col,
                    msg=f"frame {f} shift {k} run {i}",
                )

    def test_fractional_phase_within_half_step(self) -> None:
        """Small fractional drift → recovered, NN-bin still correct."""
        rng = np.random.default_rng(11)
        # Half a column step is ~2 deg (smallest gap in our schedule is 4 deg).
        # Frame 0 is the clean reference; later frames carry per-frame fractional drift.
        per_frame = [(0, 0.0), (2, np.deg2rad(0.4)), (4, np.deg2rad(-0.3))]
        frames = [
            _emit_frame(
                self.col_az_ccw,
                self.run_lengths,
                self.n_rows,
                cyclic_shift=k,
                fractional_phase_rad=eps,
                rng=rng,
            )
            for k, eps in per_frame
        ]
        col_az, phase_off, col_idx = build_lidar_model_from_per_frame_azimuth(frames, spinning_direction="ccw")
        # Reference recovered from frame 0 (eps=0) matches col_az_ccw.
        np.testing.assert_allclose(col_az, self.col_az_ccw.astype(np.float32), atol=1e-6)
        # Each phase should equal cyclic part + eps (CCW), mod 2*pi.
        two_pi = 2.0 * np.pi
        for f, (k, eps) in enumerate(per_frame):
            cyclic_part = float((self.col_az_ccw[k] - self.col_az_ccw[0]) % two_pi)
            expected = (cyclic_part + eps) % two_pi
            # Wrap symmetrically so 359.7 deg ~ -0.3 deg compares correctly.
            diff = ((float(phase_off[f]) - expected + np.pi) % two_pi) - np.pi
            self.assertLess(abs(diff), 1e-3, msg=f"frame {f} (k={k}, eps={eps})")
        # Bin indices unchanged from the no-drift case (drift below half-column step).
        run_starts = np.concatenate(([0], np.cumsum(self.run_lengths * self.n_rows)[:-1]))
        for f, (k, _eps) in enumerate(per_frame):
            for i, rs in enumerate(run_starts):
                self.assertEqual(int(col_idx[f][rs]), (i + k) % self.n_unique)

    def test_single_return_no_run_collapse(self) -> None:
        """Schedule without dual-return runs: dedup is a no-op, n_columns == n_raw_columns."""
        rng = np.random.default_rng(3)
        run_lengths = np.ones(self.n_unique, dtype=np.int64)
        frames = [_emit_frame(self.col_az_ccw, run_lengths, self.n_rows, rng=rng) for _ in range(3)]
        col_az, phase_off, col_idx = build_lidar_model_from_per_frame_azimuth(frames, spinning_direction="ccw")
        self.assertEqual(col_az.size, self.n_unique)
        np.testing.assert_allclose(col_az, self.col_az_ccw.astype(np.float32), atol=1e-6)
        # Each frame's per-ray indices: each unique col appears in n_rows consecutive rays.
        expected = np.repeat(np.arange(self.n_unique), self.n_rows)
        for f, idx in enumerate(col_idx):
            np.testing.assert_array_equal(idx, expected, err_msg=f"frame {f}")
        # Phase ~ 0 since no shift was applied.
        wrapped = np.minimum(phase_off, 2.0 * np.pi - phase_off)
        self.assertTrue(np.all(wrapped < 1e-3))

    def test_per_beam_offset_recovers_rotor_only_schedule(self) -> None:
        """`per_beam_azimuth_offset=Δ` -> helper subtracts Δ and recovers rotor-only schedule."""
        rng = np.random.default_rng(99)
        per_beam = np.deg2rad(np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]))
        # Smallest gap is 4 deg; per-beam max is 2 deg, so within half-step.
        frames = [
            _emit_frame(
                self.col_az_ccw,
                self.run_lengths,
                self.n_rows,
                cyclic_shift=k,
                per_beam_offset_rad=per_beam,
                rng=rng,
            )
            for k in [0, 1, 2]
        ]
        col_az, phase_off, col_idx = build_lidar_model_from_per_frame_azimuth(
            frames,
            spinning_direction="ccw",
            per_beam_azimuth_offset=per_beam,
        )
        # Reference is the run-pattern dedup of frame 0's per-column azimuths.  Frame 0
        # had cyclic_shift=0, so the schedule should match self.col_az_ccw.
        np.testing.assert_allclose(col_az, self.col_az_ccw.astype(np.float32), atol=1e-6)
        # Bin indices follow the cyclic shift, irrespective of the per-beam injection.
        run_starts = np.concatenate(([0], np.cumsum(self.run_lengths * self.n_rows)[:-1]))
        for f, k in enumerate([0, 1, 2]):
            for i, rs in enumerate(run_starts):
                self.assertEqual(int(col_idx[f][rs]), (i + k) % self.n_unique)
        self.assertEqual(phase_off.shape, (3,))

    def test_cw_monotonicity(self) -> None:
        """CW direction: schedule must be monotonic in CW (descending mod 2*pi)."""
        # CW order: reverse the CCW schedule and offset by an arbitrary anchor so
        # values aren't monotonically decreasing in raw float space (forces the
        # relative_angle wrap-around to kick in).
        col_az_cw = np.deg2rad(np.array([350.0, 340.0, 320.0, 280.0, 240.0, 200.0, 100.0, 50.0]))
        run_lengths = np.array([2, 2, 4, 2, 2, 4, 2, 2], dtype=np.int64)
        rng = np.random.default_rng(13)
        frames = [_emit_frame(col_az_cw, run_lengths, self.n_rows, cyclic_shift=k, rng=rng) for k in [0, 2, 4]]
        col_az, phase_off, _col_idx = build_lidar_model_from_per_frame_azimuth(frames, spinning_direction="cw")
        np.testing.assert_allclose(col_az, col_az_cw.astype(np.float32), atol=1e-6)
        # CW phase: relative_angle in CW direction is (ref - angle) mod 2*pi.
        for f, k in enumerate([0, 2, 4]):
            expected = float((col_az_cw[0] - col_az_cw[k]) % (2.0 * np.pi))
            self.assertAlmostEqual(float(phase_off[f]), expected, places=4)

    def test_violates_monotonicity_raises(self) -> None:
        """A schedule that wraps incorrectly for the declared spin direction is rejected."""
        # CCW schedule passed with CW direction: the relative_angle diff goes the wrong way.
        rng = np.random.default_rng(0)
        frames = [_emit_frame(self.col_az_ccw, self.run_lengths, self.n_rows, rng=rng)]
        with self.assertRaisesRegex(ValueError, "not strictly monotonic"):
            build_lidar_model_from_per_frame_azimuth(frames, spinning_direction="cw")

    def test_residual_threshold_violation_raises(self) -> None:
        """NN-bin residual above MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG is reported."""
        rng = np.random.default_rng(0)
        # Frame 0 defines a clean reference.  Frame 1 cyclically rotates frame 0 then
        # nudges all rays of beam 0 by 0.7 deg -- larger than the 0.5 deg quality
        # gate but smaller than the smallest column gap (4 deg), so dedup and
        # monotonicity remain intact and the failure surfaces from NN-bin only.
        frame0 = _emit_frame(self.col_az_ccw, self.run_lengths, self.n_rows, rng=rng)
        frame1 = _emit_frame(self.col_az_ccw, self.run_lengths, self.n_rows, cyclic_shift=2, rng=rng)
        beam0_mask = (np.arange(frame1.size) % self.n_rows) == 0
        frame1[beam0_mask] = (frame1[beam0_mask] + np.deg2rad(MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG + 0.2)) % (2.0 * np.pi)
        with self.assertRaisesRegex(ValueError, "MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG"):
            build_lidar_model_from_per_frame_azimuth([frame0, frame1], spinning_direction="ccw")


class BinPerRayToColumnsTest(unittest.TestCase):
    """Tests for the public per-frame circular NN-bin helper."""

    def setUp(self) -> None:
        # Same non-uniform 8-column schedule as the build-lidar-model tests.
        self.col_az = np.deg2rad(np.array([0.0, 4.0, 12.0, 16.0, 28.0, 40.0, 60.0, 80.0]))
        self.n_rows = 4

    def test_binning_is_identity_on_exact_match(self) -> None:
        """Per-ray azimuths exactly at column azimuths bin to identity column indices."""
        per_ray = np.tile(self.col_az, 3)  # 3 frames worth, 8 rays each
        out = bin_per_ray_to_columns(per_ray, self.col_az)
        expected = np.tile(np.arange(self.col_az.size, dtype=np.uint16), 3)
        np.testing.assert_array_equal(out, expected)
        self.assertEqual(out.dtype, np.uint16)

    def test_wrap_around_near_zero(self) -> None:
        """A per-ray azimuth at 359.9 deg bins to column 0 (closer than the 80-deg last col)."""
        per_ray = np.array([np.deg2rad(359.9), np.deg2rad(0.1)])
        out = bin_per_ray_to_columns(per_ray, self.col_az)
        np.testing.assert_array_equal(out, np.array([0, 0], dtype=np.uint16))

    def test_jitter_within_half_step_bins_to_nearest(self) -> None:
        """Jitter under the half-column-step still bins to the nearest column."""
        rng = np.random.default_rng(17)
        # Smallest gap is 4 deg → half-step = 2 deg.  Use 1.5-deg jitter to stay safe.
        jitter = np.deg2rad(rng.uniform(-1.5, 1.5, size=self.col_az.size))
        per_ray = (self.col_az + jitter) % (2.0 * np.pi)
        out = bin_per_ray_to_columns(per_ray, self.col_az)
        np.testing.assert_array_equal(out, np.arange(self.col_az.size, dtype=np.uint16))

    def test_preserve_subtracts_per_beam_offset(self) -> None:
        """Preserve pattern: ray az = col_az + per_beam_offset[row] bins to col_az's row."""
        per_beam = np.deg2rad(np.array([-1.5, -0.5, 0.5, 1.5]))  # one offset per row
        # Build a row-varies-fastest stream: for each col c, rays = col_az[c] + per_beam[r].
        n_cols = self.col_az.size
        n_rows = per_beam.size
        col_grid = np.repeat(self.col_az, n_rows)
        row_grid = np.tile(np.arange(n_rows), n_cols)
        per_ray = (col_grid + per_beam[row_grid]) % (2.0 * np.pi)
        out = bin_per_ray_to_columns(
            per_ray,
            self.col_az,
            per_beam_az_offset_rad=per_beam,
            per_ray_beam_idx=row_grid,
        )
        expected = np.repeat(np.arange(n_cols, dtype=np.uint16), n_rows)
        np.testing.assert_array_equal(out, expected)

    def test_unsorted_column_schedule(self) -> None:
        """Reference column azimuths may be in firing order, not numerically sorted."""
        # Arbitrary firing order (non-monotonic).
        firing_order = np.array([3, 0, 5, 1, 7, 2, 4, 6])
        col_az_firing = self.col_az[firing_order]
        # Generate per-ray azimuths sweeping the schedule in firing order.
        per_ray = col_az_firing.copy()
        out = bin_per_ray_to_columns(per_ray, col_az_firing)
        np.testing.assert_array_equal(out, np.arange(col_az_firing.size, dtype=np.uint16))

    def test_offsets_without_beam_idx_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "per_ray_beam_idx is required"):
            bin_per_ray_to_columns(
                np.array([0.0]),
                self.col_az,
                per_beam_az_offset_rad=np.array([0.0]),
            )

    def test_beam_idx_without_offsets_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "per_ray_beam_idx provided but"):
            bin_per_ray_to_columns(
                np.array([0.0]),
                self.col_az,
                per_ray_beam_idx=np.array([0]),
            )

    def test_beam_idx_length_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "per_ray_beam_idx length"):
            bin_per_ray_to_columns(
                np.array([0.0, 0.0, 0.0]),
                self.col_az,
                per_beam_az_offset_rad=np.array([0.0, 0.0]),
                per_ray_beam_idx=np.array([0, 1]),  # length 2, per_ray length 3
            )

    def test_negative_beam_idx_rejected(self) -> None:
        """Silent corruption: ``offsets[-1]`` would subtract the last beam's offset.

        Mirrors the guard in ``_bare_frame`` of
        ``build_lidar_model_from_per_frame_azimuth``: a negative entry
        in ``per_ray_beam_idx`` wraps around via numpy negative indexing
        and silently subtracts the wrong beam's offset.
        """
        with self.assertRaisesRegex(ValueError, "non-negative"):
            bin_per_ray_to_columns(
                np.array([0.0, 0.0]),
                self.col_az,
                per_beam_az_offset_rad=np.array([0.0, 0.1]),
                per_ray_beam_idx=np.array([0, -1], dtype=np.int64),
            )

    def test_non_1d_inputs_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "per_ray_az_rad must be 1-D"):
            bin_per_ray_to_columns(np.zeros((2, 3)), self.col_az)
        with self.assertRaisesRegex(ValueError, "column_azimuths_rad must be 1-D"):
            bin_per_ray_to_columns(np.zeros(4), self.col_az.reshape(2, -1))

    def test_too_few_columns_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2 elements"):
            bin_per_ray_to_columns(np.array([0.0]), np.array([0.0]))


class BareFramePerRayBeamIndexValidationTest(unittest.TestCase):
    """Silent-corruption guard for ``per_ray_beam_index`` in ``_bare_frame``.

    The index is fed to ``per_beam_offset_arr[beam_idx]``. numpy negative
    indexing wraps around silently -- a stray ``-1`` would subtract the
    *last* beam's offset and the helper would return wrong-but-plausible
    bare azimuths with no exception raised. Exercise via the public API
    by passing a single frame whose ray count matches the bad index.
    """

    def test_negative_per_ray_beam_index_rejected(self) -> None:
        per_beam = np.deg2rad(np.array([-1.0, 0.0, 1.0, 2.0]))
        beam_idx = np.array([0, -1, 2, 3], dtype=np.int64)
        frame = np.zeros(beam_idx.size, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_lidar_model_from_per_frame_azimuth(
                [frame],
                spinning_direction="ccw",
                per_beam_azimuth_offset=per_beam,
                per_ray_beam_index=beam_idx,
            )


class BuildLidarModelPerFrameAzimuthShapeValidationTest(unittest.TestCase):
    """Silent-corruption guard for ``per_frame_azimuth`` input shape.

    The helper accepts either a stacked ``(F, N)`` ndarray or a sequence
    of 1-D per-frame arrays. Without a shape guard, a 1-D ndarray (or a
    sequence of scalars / 0-D arrays) iterates as scalars and produces a
    list of 0-D "frames" with ``a.size == 1``; downstream stride
    detection then silently collapses the column schedule to length 1.
    """

    def test_ndarray_must_be_2d(self) -> None:
        for bad_shape in ((10,), (3, 4, 5)):
            with self.subTest(bad_shape=bad_shape):
                bad_input = np.zeros(bad_shape, dtype=np.float64)
                with self.assertRaisesRegex(ValueError, "ndarray must be 2-D"):
                    build_lidar_model_from_per_frame_azimuth(bad_input, spinning_direction="ccw")

    def test_sequence_entries_must_be_1d(self) -> None:
        bad_inputs = (
            ([np.array(0.0), np.array(1.0)], "sequence of 0-D arrays"),
            ([np.zeros((2, 3)), np.zeros((2, 3))], "sequence of 2-D arrays"),
        )
        for entries, label in bad_inputs:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "sequence entries must be 1-D"):
                    build_lidar_model_from_per_frame_azimuth(entries, spinning_direction="ccw")


class BuildPinholeModelKShapeValidationTest(unittest.TestCase):
    """Silent-corruption guard for ``K`` shape in ``build_pinhole_model_from_opencv_K_D``.

    The function indexes ``K[0, 0]`` / ``K[1, 1]`` / ``K[0, 2]`` /
    ``K[1, 2]`` directly. A transposed ``(3, 3)`` K still satisfies all
    four index accesses but returns the wrong scalars (``K[0, 2]`` is
    the bottom-left, not ``cx``); a ``(3, 4)`` projection matrix passed
    by mistake also satisfies the indices but ``[1, 2]`` is a
    translation, not ``cy``. Reject anything that isn't exactly ``(3, 3)``.
    """

    def setUp(self) -> None:
        self.D = np.zeros(5, dtype=np.float64)

    def test_non_3x3_K_rejected(self) -> None:
        for bad_shape in ((3, 4), (4, 3), (4, 4), (2, 2), (9,), (1, 3, 3)):
            with self.subTest(bad_shape=bad_shape):
                bad_K = np.zeros(bad_shape, dtype=np.float64)
                bad_K.flat[0] = 1.0  # avoid all-zeros pathologies hitting other guards first
                with self.assertRaisesRegex(ValueError, r"K must have shape \(3, 3\)"):
                    build_pinhole_model_from_opencv_K_D(bad_K, self.D, (1280, 720))


class BuildPinholeModelResolutionValidationTest(unittest.TestCase):
    """Silent-corruption guard for ``resolution_wh`` in ``build_pinhole_model_from_opencv_K_D``.

    The resolution flows into ``np.array([w, h], dtype=np.uint64)``,
    which silently wraps negative integers (``-1`` -> ``2**64 - 1``)
    without warning, and accepts ``0`` as a "valid" but nonsensical
    dimension. Reject non-positive elements at the boundary.
    """

    def setUp(self) -> None:
        self.K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D = np.zeros(5, dtype=np.float64)

    def test_non_positive_resolution_rejected(self) -> None:
        for bad_res in ((-1, 720), (1280, 0), (0, 0), (-1, -1)):
            with self.subTest(bad_res=bad_res):
                with self.assertRaisesRegex(ValueError, "elements > 0"):
                    build_pinhole_model_from_opencv_K_D(self.K, self.D, bad_res)


class BuildPinholeModelDLengthValidationTest(unittest.TestCase):
    """Silent-corruption guard for ``D`` length in ``build_pinhole_model_from_opencv_K_D``.

    The function unpacks the 12 OpenCV distortion coefficients
    (k1..k6, p1..p2, s1..s4) from ``D`` index-by-index and substitutes
    ``0.0`` for any index past the end. A non-canonical ``D`` length
    (one ``cv2.calibrateCamera`` never emits) silently builds a
    distortion model with the unsupplied trailing coefficients set to
    zero. The result *looks* valid but is wrong; reject non-canonical
    lengths at the boundary instead.
    """

    def setUp(self) -> None:
        self.K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def test_D_non_canonical_length_rejected(self) -> None:
        for bad_len in (0, 3, 6, 9, 13, 15):
            with self.subTest(bad_len=bad_len):
                bad_D = np.zeros(bad_len, dtype=np.float64)
                with self.assertRaisesRegex(ValueError, "canonical OpenCV length"):
                    build_pinhole_model_from_opencv_K_D(self.K, bad_D, (1280, 720))

    def test_D_canonical_length_accepted(self) -> None:
        """Smoke: every canonical OpenCV length builds without raising."""
        for ok_len in (4, 5, 8, 12, 14):
            with self.subTest(ok_len=ok_len):
                build_pinhole_model_from_opencv_K_D(self.K, np.zeros(ok_len, dtype=np.float64), (1280, 720))


def _pinhole_probe_rays() -> np.ndarray:
    """Return center-to-corner camera rays for pinhole distortion tests."""
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [-0.95, -0.52, 1.0],
            [0.95, -0.52, 1.0],
            [-0.95, 0.52, 1.0],
            [0.95, 0.52, 1.0],
        ],
        dtype=np.float32,
    )


def _source_project_rational(
    K: np.ndarray,
    rays: np.ndarray,
    radial_coeffs: np.ndarray,
) -> np.ndarray:
    """Project rays to image points with NCore's rational Brown-Conrady radial form."""
    xy = rays[:, :2].astype(np.float64) / rays[:, 2:3].astype(np.float64)
    r2 = np.sum(xy * xy, axis=1)
    numerator = 1.0 + r2 * (radial_coeffs[0] + r2 * (radial_coeffs[1] + r2 * radial_coeffs[2]))
    denominator = 1.0 + r2 * (radial_coeffs[3] + r2 * (radial_coeffs[4] + r2 * radial_coeffs[5]))
    distorted = xy * (numerator / denominator)[:, None]
    image_points = np.empty((rays.shape[0], 2), dtype=np.float32)
    image_points[:, 0] = distorted[:, 0] * float(K[0, 0]) + float(K[0, 2])
    image_points[:, 1] = distorted[:, 1] * float(K[1, 1]) + float(K[1, 2])
    return image_points


def _source_project_four_term_numerator(
    K: np.ndarray,
    rays: np.ndarray,
    radial_coeffs: np.ndarray,
) -> np.ndarray:
    """Project rays to image points with a four-term numerator-only radial polynomial."""
    xy = rays[:, :2].astype(np.float64) / rays[:, 2:3].astype(np.float64)
    r2 = np.sum(xy * xy, axis=1)
    factor = 1.0 + r2 * (radial_coeffs[0] + r2 * (radial_coeffs[1] + r2 * (radial_coeffs[2] + r2 * radial_coeffs[3])))
    distorted = xy * factor[:, None]
    image_points = np.empty((rays.shape[0], 2), dtype=np.float32)
    image_points[:, 0] = distorted[:, 0] * float(K[0, 0]) + float(K[0, 2])
    image_points[:, 1] = distorted[:, 1] * float(K[1, 1]) + float(K[1, 2])
    return image_points


def _angle_probe_terms(rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(theta, unit_xy)`` terms used by angle-domain camera models."""
    ray_xy = rays[:, :2].astype(np.float64)
    ray_xy_norm = np.linalg.norm(ray_xy, axis=1)
    theta = np.arctan2(ray_xy_norm, rays[:, 2].astype(np.float64))
    unit_xy = np.divide(ray_xy, ray_xy_norm[:, None], out=np.zeros_like(ray_xy), where=ray_xy_norm[:, None] > 0.0)
    return theta, unit_xy


def _source_project_fisheye_zero_distortion(
    params: OpenCVFisheyeCameraModelParameters,
    rays: np.ndarray,
) -> np.ndarray:
    """Project rays to image points with the zero-distortion OpenCV-fisheye source model."""
    theta, unit_xy = _angle_probe_terms(rays)
    radius = theta
    image_points = np.column_stack(
        [
            float(params.focal_length[0]) * radius * unit_xy[:, 0] + float(params.principal_point[0]),
            float(params.focal_length[1]) * radius * unit_xy[:, 1] + float(params.principal_point[1]),
        ]
    )
    return image_points.astype(np.float32)


def _source_project_ftheta_linear(
    params: FThetaCameraModelParameters,
    rays: np.ndarray,
) -> np.ndarray:
    """Project rays to image points with the skeleton's linear FTheta source model.

    NCore's FThetaCameraModel shifts the stored principal_point by +0.5, so its
    image points are expressed about ``principal_point + 0.5``; the +0.5 here
    matches that convention.
    """
    theta, unit_xy = _angle_probe_terms(rays)
    scale = float(params.angle_to_pixeldist_poly[1])
    radius = scale * theta
    image_points = np.column_stack(
        [
            radius * unit_xy[:, 0] + float(params.principal_point[0]) + 0.5,
            radius * unit_xy[:, 1] + float(params.principal_point[1]) + 0.5,
        ]
    )
    return image_points.astype(np.float32)


class VerifyCameraModelProjectionTest(unittest.TestCase):
    """Tests for the source-reference camera projection guard."""

    def setUp(self) -> None:
        self.K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.resolution = (1920, 1080)
        self.rays = _pinhole_probe_rays()

    def test_matching_rational_mapping_passes(self) -> None:
        """A source reference matching NCore's rational form stays within tolerance."""
        radial = np.array([0.01, -0.005, 0.002, 0.003, -0.001, 0.0005], dtype=np.float32)
        D = np.array([radial[0], radial[1], 0.0, 0.0, radial[2], radial[3], radial[4], radial[5]], dtype=np.float32)
        params = build_pinhole_model_from_opencv_K_D(self.K, D, self.resolution)
        expected_image_points = _source_project_rational(self.K, self.rays, radial)

        max_err = verify_camera_model_projection(
            params,
            self.rays,
            expected_image_points,
            tol_px=MAX_CAMERA_MODEL_REPROJ_PX,
        )

        self.assertLessEqual(max_err, MAX_CAMERA_MODEL_REPROJ_PX)

    def test_fourth_numerator_term_misplaced_in_denominator_raises(self) -> None:
        """A source r^8 numerator term copied into NCore's denominator slot is rejected."""
        source_radial = np.array([0.02, -0.01, 0.005, 0.04], dtype=np.float32)
        wrong_D = np.array(
            [
                source_radial[0],
                source_radial[1],
                0.0,
                0.0,
                source_radial[2],
                source_radial[3],
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        params = build_pinhole_model_from_opencv_K_D(self.K, wrong_D, self.resolution)
        expected_image_points = _source_project_four_term_numerator(self.K, self.rays, source_radial)

        with self.assertRaisesRegex(ValueError, "NCore camera projection mismatch"):
            verify_camera_model_projection(
                params,
                self.rays,
                expected_image_points,
                tol_px=MAX_CAMERA_MODEL_REPROJ_PX,
            )

    def test_matching_fisheye_mapping_passes(self) -> None:
        """A source reference matching OpenCV-fisheye zero distortion passes."""
        params = OpenCVFisheyeCameraModelParameters(
            resolution=np.array(self.resolution, dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([960.0, 540.0], dtype=np.float32),
            focal_length=np.array([500.0, 500.0], dtype=np.float32),
            radial_coeffs=np.zeros(4, dtype=np.float32),
            max_angle=np.pi / 2.0,
        )
        expected_image_points = _source_project_fisheye_zero_distortion(params, self.rays)

        max_err = verify_camera_model_projection(
            params,
            self.rays,
            expected_image_points,
            tol_px=MAX_CAMERA_MODEL_REPROJ_PX,
        )

        self.assertLessEqual(max_err, MAX_CAMERA_MODEL_REPROJ_PX)

    def test_matching_ftheta_mapping_passes(self) -> None:
        """A source reference matching the skeleton's linear FTheta model passes."""
        params = FThetaCameraModelParameters(
            resolution=np.array(self.resolution, dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([960.0, 540.0], dtype=np.float32),
            reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
            pixeldist_to_angle_poly=np.array([0.0, 1.0 / 500.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            angle_to_pixeldist_poly=np.array([0.0, 500.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            max_angle=np.pi / 2.0,
        )
        expected_image_points = _source_project_ftheta_linear(params, self.rays)

        max_err = verify_camera_model_projection(
            params,
            self.rays,
            expected_image_points,
            tol_px=MAX_CAMERA_MODEL_REPROJ_PX,
        )

        self.assertLessEqual(max_err, MAX_CAMERA_MODEL_REPROJ_PX)


class VerifyCameraModelProjectionValidationTest(unittest.TestCase):
    """Input-validation guards for ``verify_camera_model_projection``.

    These exercise the shape / emptiness / finiteness / tolerance checks that run
    before NCore projection, so they need only a valid baseline to perturb.
    """

    def setUp(self) -> None:
        K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        resolution = (1920, 1080)
        self.rays = _pinhole_probe_rays()
        radial = np.array([0.01, -0.005, 0.002, 0.003, -0.001, 0.0005], dtype=np.float32)
        D = np.array([radial[0], radial[1], 0.0, 0.0, radial[2], radial[3], radial[4], radial[5]], dtype=np.float32)
        self.params = build_pinhole_model_from_opencv_K_D(K, D, resolution)
        self.expected = _source_project_rational(K, self.rays, radial)

    def test_non_2d_cam_rays_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"cam_rays must have shape \(N, 3\)"):
            verify_camera_model_projection(self.params, self.rays.reshape(-1), self.expected)

    def test_cam_rays_wrong_width_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"cam_rays must have shape \(N, 3\)"):
            verify_camera_model_projection(self.params, self.rays[:, :2], self.expected)

    def test_expected_image_points_wrong_shape_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected_image_points must have shape"):
            verify_camera_model_projection(self.params, self.rays, self.expected[:, :1])

    def test_empty_inputs_rejected(self) -> None:
        empty_rays = np.zeros((0, 3), dtype=np.float32)
        empty_expected = np.zeros((0, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            verify_camera_model_projection(self.params, empty_rays, empty_expected)

    def test_non_finite_inputs_rejected(self) -> None:
        bad_rays = self.rays.copy()
        bad_rays[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "only finite values"):
            verify_camera_model_projection(self.params, bad_rays, self.expected)

    def test_negative_tol_px_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "tol_px must be finite and non-negative"):
            verify_camera_model_projection(self.params, self.rays, self.expected, tol_px=-1.0)


class SolveStaticExtrinsicMeanEmptyTimestampsTest(unittest.TestCase):
    """Silent-corruption guard for empty timestamps in ``solve_static_extrinsic_mean``.

    ``np.mean(axis=0)`` over an empty per-frame translation stack emits
    a ``RuntimeWarning`` and returns ``NaN``; the SVD over the empty
    rotation sum returns the identity. Net effect: the helper returns
    a 4x4 with identity rotation and NaN translation *without raising*,
    pushing the corruption downstream where the NaN may or may not be
    caught. Validation runs before the ``rig_interpolator`` is touched,
    so the test passes a sentinel for it.
    """

    def test_empty_timestamps_rejected(self) -> None:
        poses = np.empty((0, 4, 4), dtype=np.float64)
        ts = np.empty(0, dtype=np.uint64)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            solve_static_extrinsic_mean(object(), poses, ts)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
