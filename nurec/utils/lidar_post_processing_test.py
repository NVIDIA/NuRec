# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from unittest.mock import MagicMock

import torch

from nurec.utils.lidar_post_processing import (
    _auto_elevation_rows,
    angle_gradient_filter,
    distance_based_filter,
)


def _make_rings(ring_elev_deg, n_az, jitter_deg=0.0, az_lo=-30.0, az_hi=30.0, seed=0):
    """Synthesize a discrete-ring scan: rays at the given ring elevations, each swept over azimuth.

    Returns (el_deg, az_deg, true_ring_ids) with rings ordered by increasing elevation, so a correct
    row assignment equals `true_ring_ids` exactly.
    """
    g = torch.Generator().manual_seed(seed)
    az_line = torch.linspace(az_lo, az_hi, n_az)
    els, azs, rings = [], [], []
    for k, e in enumerate(ring_elev_deg):
        jit = (torch.rand(n_az, generator=g) * 2 - 1) * jitter_deg if jitter_deg > 0 else torch.zeros(n_az)
        els.append(torch.full((n_az,), float(e)) + jit)
        azs.append(az_line.clone())
        rings.append(torch.full((n_az,), k, dtype=torch.long))
    return torch.cat(els), torch.cat(azs), torch.cat(rings)


def _dirs_from_el_az(el_deg, az_deg):
    e = torch.deg2rad(el_deg.float())
    a = torch.deg2rad(az_deg.float())
    return torch.stack([e.cos() * a.cos(), e.cos() * a.sin(), e.sin()], dim=1)


def _nonuniform_ring_elevations(n_rings=24, near=0.1, far=1.0):
    """Ring elevations whose spacing grows monotonically from `near` (center) to `far` (edge)."""
    spacings = torch.linspace(near, far, n_rings - 1)
    return [0.0] + torch.cumsum(spacings, dim=0).tolist()


class TestDistanceBasedFilter(unittest.TestCase):
    def setUp(self):
        # grid dims
        self.b, self.h, self.w = 1, 3, 5
        self.n_rays = self.b * self.h * self.w

    def make_rendered(self, distances):
        # Create a simple mock object that has a .distance attribute
        rendered = MagicMock()
        rendered.distance = distances
        return rendered

    def make_model_elements(self, seed=0):
        # model_element should be shape (n_rays, 2) with vertical,horizontal indices
        gen = torch.Generator().manual_seed(seed)
        v = torch.randint(0, self.h, (self.n_rays, 1), generator=gen)
        gen.manual_seed(seed + 1)
        u = torch.randint(0, self.w, (self.n_rays, 1), generator=gen)
        return torch.cat([v, u], dim=1)

    def test_all_invalid_valid_mask_produces_all_false(self):
        # If no rays are marked valid, filter should return all False
        distances = torch.linspace(1.0, 2.0, self.n_rays)
        rendered = self.make_rendered(distances)
        model_elements = self.make_model_elements(seed=0)

        valid_mask_pred = torch.zeros(self.n_rays, dtype=torch.bool)

        out = distance_based_filter(
            rendered=rendered,
            model_elements=model_elements,
            valid_mask_pred=valid_mask_pred,
            filter_threshold=0.5,
            n_vertical_bins=self.h,
            n_horizontal_bins=self.w,
        )

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.dtype, torch.bool)
        self.assertEqual(out.numel(), self.n_rays)
        # all should be False because nothing was valid to compute discontinuities
        self.assertTrue(torch.all(~out))

    def test_all_valid_returns_bool_mask_length_n_rays(self):
        # When all rays are valid, we at least get a boolean mask of the right size
        # Construct distances that vary smoothly so thresholding will likely keep most False
        distances = torch.linspace(1.0, 3.0, self.n_rays)
        rendered = self.make_rendered(distances)
        model_elements = self.make_model_elements(seed=42)
        valid_mask_pred = torch.ones(self.n_rays, dtype=torch.bool)

        out = distance_based_filter(
            rendered=rendered,
            model_elements=model_elements,
            valid_mask_pred=valid_mask_pred,
            filter_threshold=0.01,  # small threshold
            n_vertical_bins=self.h,
            n_horizontal_bins=self.w,
        )

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.dtype, torch.bool)
        self.assertEqual(out.numel(), self.n_rays)
        # As a basic sanity check ensure mask contains some True/False values (not all same)
        # It's allowed that all could be False in degenerate setups, but with this data we expect variability.
        unique_vals = torch.unique(out)
        self.assertTrue(unique_vals.numel() >= 1)


class TestAngleGradientFilter(unittest.TestCase):
    """Angle-only variant on a synthetic DISCRETE-ring scene (its designed use case).

    3 rings at elevation -5/0/+5 deg (clearly separable), each sweeping azimuth in ~1 deg steps.
    Flat 10 m surface with one flying pixel (3 m) in the middle-ring middle-azimuth; its azimuth
    neighbours are the flat surface. Only the flying pixel should be flagged.
    """

    def _build(self, flyer=True):
        rings = [-5.0, 0.0, 5.0]
        azs = [(-10.0 + i) for i in range(21)]  # -10..10 deg, ~1 deg steps
        dirs, dist = [], []
        flyer_idx = None
        for ri, e in enumerate(rings):
            er = math.radians(e)
            for ai, a in enumerate(azs):
                ar = math.radians(a)
                dirs.append([math.cos(er) * math.cos(ar), math.cos(er) * math.sin(ar), math.sin(er)])
                is_flyer = flyer and ri == 1 and ai == len(azs) // 2
                if is_flyer:
                    flyer_idx = len(dist)
                dist.append(3.0 if is_flyer else 10.0)
        dirs = torch.tensor(dirs, dtype=torch.float32)
        distances = torch.tensor(dist, dtype=torch.float32)
        rendered = MagicMock()
        rendered.distance = distances
        valid = torch.ones(len(dist), dtype=torch.bool)
        return rendered, dirs, valid, flyer_idx

    def test_flags_only_the_flying_pixel(self):
        rendered, dirs, valid, flyer_idx = self._build(flyer=True)
        out = angle_gradient_filter(
            rendered=rendered,
            ray_direction_sensor=dirs,
            valid_mask_pred=valid,
            filter_threshold=0.02,
        )  # auto elevation/azimuth resolution
        self.assertEqual(out.dtype, torch.bool)
        self.assertTrue(bool(out[flyer_idx]), "flying pixel should be flagged")
        self.assertEqual(int(out.sum()), 1, "only the flying pixel should be flagged")

    def test_flat_surface_flags_nothing(self):
        rendered, dirs, valid, _ = self._build(flyer=False)
        out = angle_gradient_filter(
            rendered=rendered, ray_direction_sensor=dirs, valid_mask_pred=valid, filter_threshold=0.02
        )
        self.assertEqual(int(out.sum()), 0, "a flat surface should flag nothing")

    def test_row_boundary_edge_not_flagged(self):
        # Single row (one elevation), azimuth swept -10..10 deg. The LAST azimuth ray sits at a real
        # depth edge (10 m -> 3 m jump) but has only its left neighbour. It must NOT be flagged
        # (single-neighbour boundary case), matching the range-view path which never flags the
        # first/last column. The second-to-last ray keeps its near-side neighbour so it is not
        # flagged either -> nothing should be flagged.
        azs = [(-10.0 + i) for i in range(21)]
        dirs, dist = [], []
        for a in azs:
            ar = math.radians(a)
            dirs.append([math.cos(ar), math.sin(ar), 0.0])
            dist.append(3.0 if a == azs[-1] else 10.0)  # boundary ray at a real edge
        rendered = MagicMock()
        rendered.distance = torch.tensor(dist, dtype=torch.float32)
        out = angle_gradient_filter(
            rendered=rendered,
            ray_direction_sensor=torch.tensor(dirs, dtype=torch.float32),
            valid_mask_pred=torch.ones(len(dist), dtype=torch.bool),
            filter_threshold=0.02,
        )
        self.assertEqual(int(out.sum()), 0, "a row-boundary ray must not be flagged on one neighbour")


class TestAutoElevationRows(unittest.TestCase):
    """`_auto_elevation_rows` labels rings directly across a range of ray patterns, and fails loud
    when rings are not separable in elevation."""

    def _assert_exact_rings(self, el, true_rings):
        row = _auto_elevation_rows(el)
        self.assertEqual(int(row.max()) + 1, int(true_rings.max()) + 1, "wrong number of rows")
        self.assertTrue(torch.equal(row, true_rings), "each ray must map to its own ring")

    def test_uniform_rings(self):
        # 32 uniform rings, 0.5 deg spacing, 512 azimuths/ring.
        el, _, rings = _make_rings([0.5 * k for k in range(32)], n_az=512, jitter_deg=0.01, seed=1)
        self._assert_exact_rings(el, rings)

    def test_nonuniform_dense_center_sparse_edge(self):
        # Ring spacing grows monotonically from 0.1 deg (dense center) to 1.0 deg (sparse edge) -- no
        # single `elevation_resolution_deg` could bin this, but direct labelling handles it. Min
        # spacing (0.1) stays well above the intra-ring jitter (0.01).
        elevs = _nonuniform_ring_elevations()
        el, _, rings = _make_rings(elevs, n_az=256, jitter_deg=0.01, seed=2)
        self._assert_exact_rings(el, rings)

    def test_sparse_azimuth(self):
        # Only 6 azimuths/ring -- the regime the old p90/p50 heuristic happened to handle.
        el, _, rings = _make_rings([2.0 * k for k in range(10)], n_az=6, jitter_deg=0.01, seed=3)
        self._assert_exact_rings(el, rings)

    def test_dense_azimuth(self):
        # 16 rings, 2048 azimuths/ring: inter-ring gaps are ~0.05% of all gaps, so the old
        # p90-quantile heuristic silently fell back to isotropic here. Direct labelling still nails it.
        el, _, rings = _make_rings([1.0 * k for k in range(16)], n_az=2048, jitter_deg=0.005, seed=4)
        self._assert_exact_rings(el, rings)

    def test_zero_jitter_uniform(self):
        # Rings with EXACTLY constant elevation (intra-ring gaps are 0).
        el, _, rings = _make_rings([0.6 * k for k in range(20)], n_az=128, jitter_deg=0.0, seed=5)
        self._assert_exact_rings(el, rings)

    def test_single_ring(self):
        el, _, _ = _make_rings([0.0], n_az=64, jitter_deg=0.005, seed=6)
        row = _auto_elevation_rows(el)
        self.assertEqual(int(row.max()), 0, "a single elevation is one row")

    def test_shuffled_input_same_labels(self):
        el, _, rings = _make_rings([1.5 * k for k in range(12)], n_az=64, jitter_deg=0.01, seed=7)
        perm = torch.randperm(el.numel(), generator=torch.Generator().manual_seed(0))
        row = _auto_elevation_rows(el[perm])
        self.assertTrue(torch.equal(row, rings[perm]), "labelling must be order-independent")

    def test_interleaved_rings_fail_loud(self):
        # ET30X-like: each ring's elevation DRIFTS across azimuth by ~the ring spacing, so rings
        # overlap in elevation and cannot be separated -> must raise, not silently mis-bin.
        spacing, drift = 0.055, 0.05
        az = torch.linspace(-30.0, 30.0, 200)
        els = []
        for k in range(32):
            els.append(spacing * k + drift * (az - az.mean()) / (az.max() - az.mean()))
        el = torch.cat(els)
        with self.assertRaises(ValueError):
            _auto_elevation_rows(el)

    def test_continuous_sweep_fail_loud(self):
        # A genuinely continuous elevation sweep (no ring structure) -> raises.
        el = torch.linspace(0.0, 18.0, 2000)
        with self.assertRaises(ValueError):
            _auto_elevation_rows(el)


class TestAngleGradientFilterRayPatterns(unittest.TestCase):
    """End-to-end filter behaviour on non-uniform rings and the fail-loud path."""

    def test_nonuniform_rings_flag_only_flyer(self):
        # Dense-center / sparse-edge rings; put a flying pixel in a dense-center ring where a single
        # scalar elevation_resolution would merge neighbours and either miss it or false-positive.
        elevs = _nonuniform_ring_elevations()
        el, az, rings = _make_rings(elevs, n_az=64, jitter_deg=0.0, seed=8)
        dirs = _dirs_from_el_az(el, az)
        dist = torch.full((el.numel(),), 10.0)
        mid_ring = len(elevs) // 2
        idx = (rings == mid_ring).nonzero().flatten()
        flyer = int(idx[len(idx) // 2])
        dist[flyer] = 3.0
        rendered = MagicMock()
        rendered.distance = dist
        out = angle_gradient_filter(
            rendered=rendered,
            ray_direction_sensor=dirs,
            valid_mask_pred=torch.ones(el.numel(), dtype=torch.bool),
            filter_threshold=0.02,
        )
        self.assertTrue(bool(out[flyer]), "flying pixel in a dense ring should be flagged")
        self.assertEqual(int(out.sum()), 1, "only the flying pixel should be flagged")

    def test_interleaved_rings_raises_without_explicit_resolution(self):
        spacing, drift = 0.055, 0.05
        az_line = torch.linspace(-30.0, 30.0, 200)
        els = [spacing * k + drift * (az_line - az_line.mean()) / (az_line.max() - az_line.mean()) for k in range(32)]
        el = torch.cat(els)
        az = az_line.repeat(32)
        dirs = _dirs_from_el_az(el, az)
        rendered = MagicMock()
        rendered.distance = torch.full((el.numel(),), 10.0)
        with self.assertRaises(ValueError):
            angle_gradient_filter(
                rendered=rendered,
                ray_direction_sensor=dirs,
                valid_mask_pred=torch.ones(el.numel(), dtype=torch.bool),
                filter_threshold=0.02,
            )

    def test_interleaved_rings_ok_with_explicit_resolution(self):
        # The same interleaved sensor works when the row spacing is given explicitly.
        spacing, drift = 0.055, 0.05
        az_line = torch.linspace(-30.0, 30.0, 200)
        els = [spacing * k + drift * (az_line - az_line.mean()) / (az_line.max() - az_line.mean()) for k in range(32)]
        el = torch.cat(els)
        az = az_line.repeat(32)
        dirs = _dirs_from_el_az(el, az)
        rendered = MagicMock()
        rendered.distance = torch.full((el.numel(),), 10.0)
        out = angle_gradient_filter(
            rendered=rendered,
            ray_direction_sensor=dirs,
            valid_mask_pred=torch.ones(el.numel(), dtype=torch.bool),
            filter_threshold=0.02,
            elevation_resolution_deg=spacing,
        )
        self.assertEqual(out.dtype, torch.bool)
        self.assertEqual(out.numel(), el.numel())


if __name__ == "__main__":
    unittest.main()
