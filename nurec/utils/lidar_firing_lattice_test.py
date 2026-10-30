# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest
import torch

from nurec.utils.lidar_firing_lattice import FiringLatticeEstimator


def _angles_to_dirs(az: torch.Tensor, el: torch.Tensor) -> torch.Tensor:
    ce = torch.cos(el)
    return torch.stack([torch.cos(az) * ce, torch.sin(az) * ce, torch.sin(el)], dim=-1)


def _make_zigzag_spinning_frame(
    frame_idx: int,
    n_cols: int = 900,
    n_rows: int = 32,
    el_lo: float = -0.2,
    el_hi: float = 0.35,
    frame_dur_us: float = 100_000.0,
):
    """One frame of a spinning sensor with a per-ROW azimuth zigzag (P128-like).

    At column ``c`` all rows fire; row ``r`` has a fixed azimuth offset. The
    timestamp is a function of the COLUMN only (rel_t = c/(n_cols-1)), NOT of the
    raw azimuth -- so azimuth alone cannot order time. A small per-frame azimuth
    phase makes frames differ slightly (as a real spin start differs).
    """
    cols = torch.arange(n_cols)
    rows = torch.arange(n_rows)
    col_az = torch.linspace(-math.pi, math.pi, n_cols)  # base azimuth per column
    row_off = 0.02 * torch.sin(rows.float() / n_rows * 2 * math.pi)  # zigzag offset per row
    row_el = torch.linspace(el_lo, el_hi, n_rows)

    cc, rr = torch.meshgrid(cols, rows, indexing="ij")  # [n_cols, n_rows]
    az = (col_az[cc] + row_off[rr]).reshape(-1)
    el = row_el[rr].reshape(-1)
    dirs = _angles_to_dirs(az, el)
    # timestamp: column phase -> rel_t, plus a tiny stable per-frame start jitter
    rel_t = (cc.reshape(-1).float()) / (n_cols - 1)
    t0 = frame_idx * frame_dur_us
    ts = t0 + rel_t * frame_dur_us
    return dirs, ts, az, el, rel_t


class TestFiringLatticeEstimator:
    def _build_clip(self, n_frames=8, drop_seed=0):
        frames = []
        gen = torch.Generator().manual_seed(drop_seed)
        for f in range(n_frames):
            dirs, ts, az, el, rel = _make_zigzag_spinning_frame(f)
            # simulate raydrops: each frame drops a DIFFERENT random 15%
            keep = torch.rand(dirs.shape[0], generator=gen) > 0.15
            frames.append((dirs[keep].numpy(), ts[keep].numpy()))
        return frames

    def test_recovers_missing_direction_timestamps(self):
        """A direction dropped in a test frame gets an accurate rel_t from the
        lattice aggregated over the other frames."""
        est = FiringLatticeEstimator.from_frames(
            self._build_clip(n_frames=10),
            n_bins_azimuth=1800,
            n_bins_elevation=64,
            az_lo=-math.pi,
            az_hi=math.pi,
            el_lo=-0.3,
            el_hi=0.45,
        )
        # a fresh, complete frame as ground truth
        dirs, ts, az, el, rel_gt = _make_zigzag_spinning_frame(99)
        rel_est = est.estimate_rel_t(dirs)
        valid = ~torch.isnan(rel_est)
        assert float(valid.float().mean()) > 0.95, "lattice should cover most directions"
        err = (rel_est[valid] - rel_gt[valid]).abs()
        # rel_t error should be a small fraction of the frame (< ~1 column width)
        assert float(err.median()) < 2.0 / 900, (float(err.median()), 1.0 / 900)
        assert float(err.quantile(0.95)) < 5.0 / 900

    def test_robust_to_whole_channel_dropout(self):
        """Even if a test frame is MISSING an entire elevation band, the lattice
        (built from frames that saw it) recovers those directions' rel_t."""
        est = FiringLatticeEstimator.from_frames(
            self._build_clip(n_frames=10),
            n_bins_azimuth=1800,
            n_bins_elevation=64,
            az_lo=-math.pi,
            az_hi=math.pi,
            el_lo=-0.3,
            el_hi=0.45,
        )
        dirs, ts, az, el, rel_gt = _make_zigzag_spinning_frame(99)
        band = (el > 0.0) & (el < 0.1)  # a whole elevation band "dropped" in this frame
        rel_est = est.estimate_rel_t(dirs[band])
        valid = ~torch.isnan(rel_est)
        assert float(valid.float().mean()) > 0.9
        err = (rel_est[valid] - rel_gt[band][valid]).abs()
        assert float(err.median()) < 2.0 / 900

    def test_consistency_diagnostic_small(self):
        """The cross-frame rel_t std must be tiny -- confirming the firing pattern
        is stable enough for aggregation (the load-bearing assumption)."""
        est = FiringLatticeEstimator.from_frames(
            self._build_clip(n_frames=10),
            n_bins_azimuth=1800,
            n_bins_elevation=64,
            az_lo=-math.pi,
            az_hi=math.pi,
            el_lo=-0.3,
            el_hi=0.45,
        )
        cons = est.consistency_us(frame_duration_us=100_000.0)
        # stable lattice: median cross-frame std well under one column (~111us)
        assert cons["median_us"] < 111.0, cons

    def test_empty_cells_return_nan(self):
        """Directions the lattice never observed (far outside the FOV) return NaN
        so the caller can skip them rather than fabricate a timestamp."""
        est = FiringLatticeEstimator.from_frames(
            self._build_clip(n_frames=4),
            n_bins_azimuth=1800,
            n_bins_elevation=64,
            az_lo=-math.pi,
            az_hi=math.pi,
            el_lo=-0.3,
            el_hi=0.45,
        )
        # elevation far below the sensor FOV -> no evidence
        far = _angles_to_dirs(torch.tensor([0.0]), torch.tensor([-0.29]))
        rel = est.estimate_rel_t(far, max_search_radius=1)
        assert torch.isnan(rel).all()

    def test_raises_on_empty_input(self):
        bounds = dict(az_lo=-math.pi, az_hi=math.pi, el_lo=-math.pi / 2, el_hi=math.pi / 2)
        with pytest.raises(ValueError):
            FiringLatticeEstimator.from_frames([], **bounds)
        with pytest.raises(ValueError):
            FiringLatticeEstimator.from_frames([(np.zeros((0, 3), np.float32), np.zeros((0,), np.int64))], **bounds)

    def test_from_sensor_matches_from_frames(self):
        """from_sensor gathers the sensor's own frames and must produce the same
        lattice as feeding those frames to from_frames directly."""

        class _FakeSensor:
            def __init__(self, frames):
                self._frames = frames

            @property
            def frames_count(self):
                return len(self._frames)

            def get_frame_ray_bundle_direction(self, i):
                return self._frames[i][0]

            def get_frame_ray_bundle_timestamp_us(self, i):
                return self._frames[i][1]

        frames = self._build_clip(n_frames=8)
        sensor = _FakeSensor(frames)

        bounds = dict(az_lo=-math.pi, az_hi=math.pi, el_lo=-0.3, el_hi=0.4)
        via_sensor = FiringLatticeEstimator.from_sensor(
            sensor, max_frames=64, n_bins_azimuth=1800, n_bins_elevation=64, **bounds
        )
        via_frames = FiringLatticeEstimator.from_frames(frames, n_bins_azimuth=1800, n_bins_elevation=64, **bounds)
        # identical grids (from_sensor with max_frames >= n_frames gathers all)
        assert torch.equal(via_sensor.count_grid, via_frames.count_grid)
        m1, m2 = via_sensor.rel_t_grid, via_frames.rel_t_grid
        both = ~(torch.isnan(m1) | torch.isnan(m2))
        assert torch.allclose(m1[both], m2[both])
        assert torch.equal(torch.isnan(m1), torch.isnan(m2))

    def test_from_sensor_raises_on_no_frames(self):
        class _EmptySensor:
            frames_count = 0

            def get_frame_ray_bundle_direction(self, i):  # pragma: no cover
                raise AssertionError

            def get_frame_ray_bundle_timestamp_us(self, i):  # pragma: no cover
                raise AssertionError

        with pytest.raises(ValueError, match="no frames"):
            FiringLatticeEstimator.from_sensor(
                _EmptySensor(), az_lo=-math.pi, az_hi=math.pi, el_lo=-math.pi / 2, el_hi=math.pi / 2
            )


def _make_multicover_frame(frame_idx, n_dirs=64, visits=(0.25, 0.75), frame_dur_us=100_000.0):
    """A sensor that fires the SAME set of directions at TWO distinct times per
    frame (a self-crossing / multi-cover pattern). Each direction thus has two
    true visit times; a single mean would be ~0.5 (never actually fired).

    Anchor rays at rel_t 0 and 1 (in far-away directions) fix the per-frame time
    span so from_frames' per-frame normalization keeps the two visits at their
    intended rel_t (0.25, 0.75) rather than renormalizing them to (0, 1)."""
    az = torch.linspace(-0.4, 0.4, n_dirs)
    el = torch.linspace(-0.1, 0.1, n_dirs)
    dirs_one = _angles_to_dirs(az, el)
    dirs = torch.cat([dirs_one, dirs_one], dim=0)  # same dirs, two visits
    rel = torch.cat([torch.full((n_dirs,), visits[0]), torch.full((n_dirs,), visits[1])])
    # anchors at rel 0 and 1 in distinct directions (they occupy their own cells)
    anchor_dirs = _angles_to_dirs(torch.tensor([-1.2, 1.2]), torch.tensor([0.15, 0.15]))
    dirs = torch.cat([dirs, anchor_dirs], dim=0)
    rel = torch.cat([rel, torch.tensor([0.0, 1.0])])
    ts = frame_idx * frame_dur_us + rel.double() * frame_dur_us
    return dirs, ts, dirs_one


class TestMultiTimeModes:
    def test_default_build_is_single_mode_and_unchanged(self):
        """Default (n_time_modes=1) build sets no modes grid and estimate_rel_t is
        byte-for-byte the original single-cover behaviour."""
        frames = TestFiringLatticeEstimator()._build_clip(n_frames=6)
        est = FiringLatticeEstimator.from_frames(
            frames, n_bins_azimuth=1800, n_bins_elevation=64, az_lo=-math.pi, az_hi=math.pi, el_lo=-0.3, el_hi=0.45
        )
        assert est.rel_t_modes_grid is None
        assert est.n_time_modes == 1
        # estimate_rel_t_modes degrades to [M,1] == estimate_rel_t
        dirs, _, _, _, _ = _make_zigzag_spinning_frame(99)
        single = est.estimate_rel_t(dirs)
        modes = est.estimate_rel_t_modes(dirs)
        assert modes.shape == (dirs.shape[0], 1)
        valid = ~torch.isnan(single)
        assert torch.allclose(modes[valid, 0], single[valid], equal_nan=False)

    def test_single_cover_multimode_yields_one_mode(self):
        """Enabling multi-mode on SINGLE-cover data still yields exactly one real
        mode per cell, equal to the single mean -> clean degradation."""
        frames = TestFiringLatticeEstimator()._build_clip(n_frames=8)
        est = FiringLatticeEstimator.from_frames(
            frames,
            n_bins_azimuth=1800,
            n_bins_elevation=64,
            az_lo=-math.pi,
            az_hi=math.pi,
            el_lo=-0.3,
            el_hi=0.45,
            n_time_modes=3,
            time_hist_bins=256,
        )
        assert est.rel_t_modes_grid is not None
        dirs, _, _, rel_gt, _ = _make_zigzag_spinning_frame(99)
        modes = est.estimate_rel_t_modes(dirs)
        # per direction: exactly one non-NaN mode, and it matches the mean
        n_modes = (~torch.isnan(modes)).sum(dim=1)
        covered = n_modes > 0
        assert torch.all(n_modes[covered] == 1), "single-cover must give exactly one mode"
        single = est.estimate_rel_t(dirs)
        m0 = modes[:, 0]
        both = covered & ~torch.isnan(single)
        # Exclude azimuth-wrap cells: there rel_t straddles the 0/1 boundary, so the
        # multi-mode path (correctly) collapses to ~0 while the naive mean sits mid-
        # range -- they legitimately differ. Compare only non-wrap cells, and assert
        # on a robust quantile (a handful of boundary cells near the wrap heuristic
        # can still diverge; the bulk must agree to sub-bin precision).
        non_wrap = both & (single > 0.02) & (single < 0.98)
        diff = (m0[non_wrap] - single[non_wrap]).abs()
        assert float(diff.median()) < 2.0 / 256, float(diff.median())
        assert float(diff.quantile(0.99)) < 4.0 / 256, float(diff.quantile(0.99))

    def test_multicover_recovers_two_modes(self):
        """On genuine multi-cover data, the estimator recovers BOTH visit times
        per direction (where a single mean would be wrong)."""
        frames = [_make_multicover_frame(f)[:2] for f in range(8)]
        est = FiringLatticeEstimator.from_frames(
            frames,
            n_bins_azimuth=512,
            n_bins_elevation=64,
            az_lo=-math.pi,
            az_hi=math.pi,
            el_lo=-0.2,
            el_hi=0.2,
            n_time_modes=2,
            time_hist_bins=256,
            mode_merge_rel=4.0,
        )
        _, _, dirs_one = _make_multicover_frame(99)
        modes = est.estimate_rel_t_modes(dirs_one)
        n_modes = (~torch.isnan(modes)).sum(dim=1)
        covered = n_modes > 0
        # most covered directions should recover two modes near 0.25 and 0.75
        assert float((n_modes[covered] == 2).float().mean()) > 0.8, n_modes[covered].float().mean()
        two = modes[n_modes == 2]
        lo = two.min(dim=1).values
        hi = two.max(dim=1).values
        assert float((lo - 0.25).abs().median()) < 0.02, float((lo - 0.25).abs().median())
        assert float((hi - 0.75).abs().median()) < 0.02, float((hi - 0.75).abs().median())
        # the naive single mean would sit near 0.5 -- confirm modes are NOT that
        assert float(lo.median()) < 0.4 and float(hi.median()) > 0.6

    def test_limited_azimuth_fov_bounds_sharpen_rel_t(self):
        """A limited-azimuth-FOV sensor (e.g. front-facing, ~[-30, 30] deg) is
        aggregated far more accurately when az_lo/az_hi match its real FOV than
        with the full-circle [-pi, pi] default: the same n_bins_azimuth then spans
        only the observed wedge, so distinct columns land in distinct cells instead
        of being averaged together (which blurs the column->time mapping)."""
        az_lo, az_hi = math.radians(-30.0), math.radians(30.0)
        el_lo, el_hi = -0.1, 0.1

        def _make_front_frame(frame_idx: int, n_cols: int = 300, n_rows: int = 16):
            cols = torch.arange(n_cols)
            rows = torch.arange(n_rows)
            col_az = torch.linspace(az_lo, az_hi, n_cols)
            row_el = torch.linspace(el_lo, el_hi, n_rows)
            cc, rr = torch.meshgrid(cols, rows, indexing="ij")
            az = col_az[cc].reshape(-1)
            el = row_el[rr].reshape(-1)
            dirs = _angles_to_dirs(az, el)
            rel_t = cc.reshape(-1).float() / (n_cols - 1)
            ts = frame_idx * 1e5 + rel_t * 1e5
            return dirs.numpy(), ts.numpy(), az, rel_t

        frames = [_make_front_frame(f)[:2] for f in range(8)]
        _, _, probe_az, probe_rel = _make_front_frame(99)
        probe_dirs = _angles_to_dirs(probe_az, torch.zeros_like(probe_az))

        # A coarse azimuth grid that would be starved on the full circle (most bins
        # empty, many columns colliding into one wedge cell) but is ample on the
        # real FOV: 30 deg / 256 bins ~ 0.12 deg/bin vs 360 deg / 256 ~ 1.4 deg/bin.
        n_az = 256
        est_fov = FiringLatticeEstimator.from_frames(
            frames, n_bins_azimuth=n_az, n_bins_elevation=32, az_lo=az_lo, az_hi=az_hi, el_lo=el_lo, el_hi=el_hi
        )
        est_full = FiringLatticeEstimator.from_frames(
            frames, n_bins_azimuth=n_az, n_bins_elevation=32, az_lo=-math.pi, az_hi=math.pi, el_lo=el_lo, el_hi=el_hi
        )
        rel_fov = est_fov.estimate_rel_t(probe_dirs, max_search_radius=1)
        rel_full = est_full.estimate_rel_t(probe_dirs, max_search_radius=1)

        ok_fov = ~torch.isnan(rel_fov)
        ok_full = ~torch.isnan(rel_full)
        err_fov = float((rel_fov[ok_fov] - probe_rel[ok_fov]).abs().median())
        err_full = float((rel_full[ok_full] - probe_rel[ok_full]).abs().median())
        # FOV-matched bounds recover rel_t accurately...
        assert err_fov < 0.02, err_fov
        # ...and materially better than the full-circle default (which collapses the
        # 60 deg wedge into ~256*60/360 ~ 43 usable bins, averaging distinct times).
        assert err_fov < err_full, (err_fov, err_full)
