# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-free reconstruction of a spinning LiDAR's fixed firing lattice.

Motivation
----------
The model-free unstructured LiDAR path needs per-ray relative frame times for
"virtual" (dropped) rays -- directions the sensor fired but got no return -- so
their rolling-shutter pose is physically consistent. When a sensor model
(``RowOffsetStructuredSpinningLidarModel``) is available, the exact per-element
sweep time is used (the ``model_elements`` raydrop mode). This type provides the
MODEL-FREE equivalent (the ``aggregate_lattice`` raydrop mode).

Approach
--------
A spinning / repeating-lattice sensor fires the SAME intrinsic set of sensor-frame
directions every revolution, each at a fixed relative frame time (its column
phase). Any single frame is missing rays (raydrops), but the UNION over all frames
recovers the full lattice + per-direction relative time. We aggregate every
frame's ``(azimuth, elevation) -> rel_t`` into a fine 2D lookup once per sensor,
then query it to assign ``rel_t`` to any direction.

Validated on the real A/B P128 clip (``RowOffsetStructuredSpinningLidarModel``,
n_rows=128, n_cols=3600, 10Hz): reconstructs per-ray relative time to ~2-3us
median / ~8us p95 vs measured timestamps, robust to whole-channel and
contiguous-arc dropout (the cases where single-frame angle-nearest-neighbour
collapses to 1-3ms). The load-bearing assumption -- that the firing pattern is
stable across frames -- is verified there: per-direction rel_t std across frames
is ~8us (median). This is why the lattice is a valid PER-SENSOR invariant (cached
once per sensor), unlike the per-frame tiling/pose which must never be cached per
sensor.

The reconstruction makes no P128-specific assumption; it works for any sensor
whose firing lattice repeats across frames (spinning, elevation-swept, repeating
MEMS).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class FiringLatticeSensor(Protocol):
    """Minimal per-frame ray-bundle interface needed to build a firing lattice.

    Structurally satisfied by the ncore lidar sensor (``ncore.sensors`` /
    ``NCOREDataSource.lidar_sensors[sensor_id]``); declared here so the estimator
    can gather its own frames without importing ncore. Only the members used by
    :meth:`FiringLatticeEstimator.from_sensor` are listed.
    """

    @property
    def frames_count(self) -> int: ...

    def get_frame_ray_bundle_direction(self, frame_index: int) -> np.ndarray:
        """``[N, 3]`` sensor-frame unit ray directions for a frame (may be empty)."""
        ...

    def get_frame_ray_bundle_timestamp_us(self, frame_index: int) -> np.ndarray:
        """``[N]`` per-ray absolute timestamps (microseconds) for a frame."""
        ...


def _dirs_to_az_el(dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sensor-frame unit directions ``[N, 3]`` -> ``(azimuth, elevation)`` radians.

    Matches the convention in ``gsplat`` / the ncore lidar model:
    ``azimuth = atan2(y, x)`` in ``(-pi, pi]``, ``elevation = asin(z)``.
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    az = torch.atan2(y, x)
    el = torch.asin(z.clamp(-1.0, 1.0))
    return az, el


@dataclass
class FiringLatticeEstimator:
    """A model-free estimate of a sensor's fixed firing lattice.

    Reconstructed by aggregating ``(azimuth, elevation) -> relative frame time``
    over all frames of one sensor. Use :meth:`from_frames` to build, then
    :meth:`estimate_rel_t` to look up relative frame times for arbitrary
    sensor-frame directions.

    Attributes:
        n_bins_azimuth, n_bins_elevation: lookup-grid resolution.
        az_lo, az_hi, el_lo, el_hi: grid angular bounds (radians).
        rel_t_grid: ``[n_bins_elevation, n_bins_azimuth]`` mean relative frame
            time in ``[0, 1]`` per cell; ``NaN`` where no frame ever fired.
        count_grid: ``[n_bins_elevation, n_bins_azimuth]`` int32 number of
            (frame, ray) samples that landed in each cell (occupancy evidence).
        rel_t_std_grid: ``[n_bins_elevation, n_bins_azimuth]`` cross-frame std of
            rel_t per cell (NaN where <2 samples); the consistency diagnostic.
        rel_t_modes_grid: optional ``[n_bins_elevation, n_bins_azimuth, K]`` of up
            to ``K = n_time_modes`` distinct per-cell visit times (NaN-padded),
            for multi-cover sensors. ``None`` in the default single-mode build.
        n_time_modes: ``K`` (1 = single-mode, the default single-cover behaviour).
    """

    n_bins_azimuth: int
    n_bins_elevation: int
    az_lo: float
    az_hi: float
    el_lo: float
    el_hi: float
    rel_t_grid: torch.Tensor
    count_grid: torch.Tensor
    rel_t_std_grid: torch.Tensor

    # Optional multi-time-mode representation (opt-in via n_time_modes>1). For a
    # MULTI-COVER sensor (e.g. a self-crossing MEMS Lissajous), one (az, el)
    # direction can be fired at several distinct times per frame, so a single mean
    # rel_t is wrong for those cells. When enabled, ``rel_t_modes_grid`` holds up to
    # ``n_time_modes`` distinct per-cell visit times (NaN-padded), recovered by
    # clustering each cell's cross-frame rel_t histogram. ``None`` (default) means
    # single-mode -- identical to the original single-cover behaviour.
    rel_t_modes_grid: Optional[torch.Tensor] = None  # [n_el, n_az, K] float32, NaN-padded
    n_time_modes: int = 1

    # ---- build ---------------------------------------------------------------
    @classmethod
    def from_frames(
        cls,
        frames: Sequence[tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]],
        *,
        az_lo: float,
        az_hi: float,
        el_lo: float,
        el_hi: float,
        n_bins_azimuth: int = 7200,  # single cover defaults if not overwritten
        n_bins_elevation: int = 512,
        device: str | torch.device = "cpu",
        n_time_modes: int = 1,
        time_hist_bins: int = 256,
        mode_merge_rel: float = 4.0,
        mode_min_mass_frac: float = 0.05,
    ) -> "FiringLatticeEstimator":
        """Aggregate a sensor's firing lattice from all its frames.

        Args:
            frames: list of ``(directions[N, 3], timestamps_us[N])`` per frame,
                the sensor-frame unit ray directions and per-ray absolute
                timestamps (microseconds) of the RETURNING rays. Each frame's
                timestamps are normalized to a per-frame ``rel_t`` in ``[0, 1]``.
            az_lo/az_hi/el_lo/el_hi: grid bounds (radians). REQUIRED -- pass the
                sensor's exact resolved FOV. There is deliberately no default: a
                silent full-circle / full-hemisphere fallback would mis-bin a
                limited-FOV sensor and waste bins, so the caller must be explicit.
            n_bins_azimuth, n_bins_elevation: lookup resolution. Defaults resolve
                a P128 (3600 columns, 128 rows) comfortably.
            device: device to hold the accumulation grids on.
            n_time_modes: ``K``. ``1`` (default) builds the single-mean lattice --
                byte-for-byte the original single-cover behaviour, no extra work or
                memory. ``>1`` additionally builds a per-cell time histogram and
                clusters it into up to ``K`` distinct visit times (for multi-cover
                sensors like a self-crossing MEMS). Single-cover cells still yield
                exactly one mode equal to the mean, so it degrades cleanly.
            time_hist_bins: resolution of the per-cell rel_t histogram used for
                mode clustering (only when ``n_time_modes>1``).
            mode_merge_rel: histogram bins within ``mode_merge_rel`` bins of each
                other are merged into one mode (a gap-based 1D clustering scale).
            mode_min_mass_frac: modes holding less than this fraction of the
                cell's total sample mass are discarded as noise (so cross-frame
                jitter in a single-cover cell collapses to one mode).

        Returns:
            A populated :class:`FiringLatticeEstimator`.

        Raises:
            ValueError: if ``frames`` is empty or no frame has any ray.
        """
        if len(frames) == 0:
            raise ValueError("FiringLatticeEstimator.from_frames: no frames provided")
        if n_time_modes < 1:
            raise ValueError(f"n_time_modes must be >= 1, got {n_time_modes}")

        # We accumulate, over ALL frames, into a flat [ncell] grid (ncell =
        # n_az * n_el angular cells). For each cell we keep running sums of the
        # per-ray relative frame time rel_t so we can recover, at the end, the mean
        # rel_t (the cell's characteristic firing phase) and its cross-frame std
        # (the consistency diagnostic). We use float64 sums because rel_t in [0, 1]
        # is added millions of times and we want the mean/variance to stay accurate.
        #   rel_sum[c]   = sum   of rel_t over samples in cell c   -> mean = rel_sum/cnt
        #   rel_sqsum[c] = sum   of rel_t^2                        -> var  = E[t^2]-E[t]^2
        #   cnt[c]       = number of (frame, ray) samples in cell c
        dev = torch.device(device)
        ncell = n_bins_azimuth * n_bins_elevation
        rel_sum = torch.zeros(ncell, dtype=torch.float64, device=dev)
        rel_sqsum = torch.zeros(ncell, dtype=torch.float64, device=dev)
        cnt = torch.zeros(ncell, dtype=torch.float64, device=dev)

        # Multi-mode only: to recover MULTIPLE distinct visit times per cell (for a
        # multi-cover sensor) we need the full per-cell distribution of rel_t, not
        # just its mean. A dense [ncell, time_hist_bins] histogram would be many GB
        # at fine angular resolution, so instead we record each hit SPARSELY as a
        # single flat index (cell * time_hist_bins + time_bin), append per frame,
        # and coalesce (unique + counts) once at the very end -- see
        # _cluster_time_modes_sparse. Skipped entirely when n_time_modes == 1.
        multi_mode = n_time_modes > 1
        hist_index_chunks: list[torch.Tensor] = []

        any_ray = False
        for dirs, ts in frames:
            # Directions -> (az, el); absolute timestamps -> per-frame rel_t in [0, 1].
            if (dirs_t := torch.as_tensor(np.asarray(dirs).astype(np.float32), device=dev)).numel() == 0:
                continue
            ts_t = torch.as_tensor(np.asarray(ts).astype(np.int64), device=dev)
            any_ray = True
            az, el = _dirs_to_az_el(dirs_t)
            # Normalize this frame's absolute us timestamps to a per-frame phase in
            # [0, 1] (0 = first ray of the frame, 1 = last). max(span, 1.0) guards a
            # degenerate single-timestamp frame. rel_t is what the render uses to
            # interpolate the rolling-shutter pose, so it is the quantity we want a
            # per-direction estimate of.
            span = float(ts_t.max() - ts_t.min())
            rel = (ts_t - ts_t.min()).to(torch.float64) / max(span, 1.0)
            # Scatter every ray's rel_t into its (az, el) cell's running sums.
            lin = cls._cell_index(az, el, n_bins_azimuth, n_bins_elevation, az_lo, az_hi, el_lo, el_hi)
            rel_sum.scatter_add_(0, lin, rel)
            rel_sqsum.scatter_add_(0, lin, rel * rel)
            cnt.scatter_add_(0, lin, torch.ones_like(rel))

            if multi_mode:
                # Record (cell, time_bin) hits for later per-cell clustering.
                tb = torch.clamp((rel * time_hist_bins).long(), 0, time_hist_bins - 1)
                hist_index_chunks.append(lin * time_hist_bins + tb)

        if not any_ray:
            raise ValueError("FiringLatticeEstimator.from_frames: all frames empty")

        # Finalize per-cell statistics. Cells no frame ever hit stay NaN (the query
        # methods treat NaN as "no evidence" and either neighbour-search or skip).
        # mean = characteristic firing phase; std = cross-frame jitter (needs >=2
        # samples to be defined) which the consistency_us() diagnostic reports.
        mean = torch.where(cnt > 0, rel_sum / cnt.clamp(min=1.0), torch.full_like(rel_sum, float("nan")))
        var = (rel_sqsum / cnt.clamp(min=1.0)) - mean * mean
        std = torch.where(cnt >= 2, var.clamp(min=0.0).sqrt(), torch.full_like(var, float("nan")))

        rel_t_modes_grid = None

        if multi_mode:
            hist_index = (
                torch.cat(hist_index_chunks) if hist_index_chunks else torch.zeros(0, dtype=torch.long, device=dev)
            )
            rel_t_modes_grid = cls._cluster_time_modes_sparse(
                hist_index, ncell, n_time_modes, time_hist_bins, mode_merge_rel, mode_min_mass_frac, dev
            ).reshape(n_bins_elevation, n_bins_azimuth, n_time_modes)

        return cls(
            n_bins_azimuth=n_bins_azimuth,
            n_bins_elevation=n_bins_elevation,
            az_lo=az_lo,
            az_hi=az_hi,
            el_lo=el_lo,
            el_hi=el_hi,
            rel_t_grid=mean.reshape(n_bins_elevation, n_bins_azimuth).to(torch.float32),
            count_grid=cnt.reshape(n_bins_elevation, n_bins_azimuth).to(torch.int32),
            rel_t_std_grid=std.reshape(n_bins_elevation, n_bins_azimuth).to(torch.float32),
            rel_t_modes_grid=rel_t_modes_grid,
            n_time_modes=n_time_modes,
        )

    @classmethod
    def from_sensor(
        cls,
        sensor: FiringLatticeSensor,
        *,
        az_lo: float,
        az_hi: float,
        el_lo: float,
        el_hi: float,
        max_frames: int = 64,
        n_time_modes: int = 1,
        n_bins_azimuth: Optional[int] = None,
        n_bins_elevation: Optional[int] = None,
    ) -> "FiringLatticeEstimator":
        """Build the lattice directly from a lidar sensor (gathers its own frames).

        Samples up to ``max_frames`` frames uniformly across the sensor's sequence
        (the union over frames fills single-frame dropouts) and aggregates their
        ``(direction, timestamp)`` ray bundles via :meth:`from_frames`. Empty frames
        (no returns) are skipped.

        Grid resolution: an explicit ``n_bins_azimuth`` / ``n_bins_elevation`` is
        used as-is; otherwise a mode-appropriate default is chosen -- a fine grid
        (7200 x 512, the ``from_frames`` default) for the single-mode lattice, which
        resolves a P128 (3600 cols x 128 rows) comfortably, or a coarser grid
        (1024 x 128) for the multi-mode (``n_time_modes>1``) build, whose per-cell
        clustering loops over occupied cells and does not need P128-fine azimuth.

        ``az_lo/az_hi/el_lo/el_hi`` are the grid bounds (radians). REQUIRED -- pass
        the sensor's exact resolved FOV; there is no default (see :meth:`from_frames`).

        Raises:
            ValueError: if the sensor has no frames.
        """
        if (nf := sensor.frames_count) == 0:
            raise ValueError("FiringLatticeEstimator.from_sensor: sensor has no frames to build a lattice from")

        frames: list[tuple[np.ndarray, np.ndarray]] = []
        # Ceiling division: floor overshoots the cap (nf=127, max_frames=8 gives
        # step 15 -> 9 frames), so the knob would be silently exceeded and the
        # prebuild would read more shards than asked for.
        step = max(1, -(-nf // max_frames))
        for fi in range(0, nf, step):
            dirs = sensor.get_frame_ray_bundle_direction(fi)
            ts = sensor.get_frame_ray_bundle_timestamp_us(fi)
            if len(dirs) == 0:
                continue
            frames.append((dirs, ts))

        if n_bins_azimuth is not None and n_bins_elevation is not None:
            n_az, n_el = n_bins_azimuth, n_bins_elevation
        elif n_time_modes > 1:
            n_az, n_el = 1024, 128
        else:
            # from_frames defaults (7200 x 512) -- the fine single-mode grid.
            n_az, n_el = 7200, 512
        return cls.from_frames(
            frames,
            n_bins_azimuth=n_az,
            n_bins_elevation=n_el,
            n_time_modes=n_time_modes,
            az_lo=az_lo,
            az_hi=az_hi,
            el_lo=el_lo,
            el_hi=el_hi,
        )

    @staticmethod
    def _cluster_time_modes_sparse(
        hist_index: torch.Tensor,
        ncell: int,
        n_time_modes: int,
        time_hist_bins: int,
        mode_merge_rel: float,
        mode_min_mass_frac: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Cluster each occupied cell's rel_t histogram into up to ``n_time_modes``
        visit times (NaN-padded), from a SPARSE list of flat ``cell*time_hist_bins
        + tbin`` hit indices (avoids a dense [ncell, bins] grid that is many GB at
        fine resolutions).

        Gap-based 1D clustering: consecutive occupied bins within ``mode_merge_rel``
        bins form one mode (count-weighted centroid); modes with <
        ``mode_min_mass_frac`` of the cell mass are dropped as noise; circular wrap
        (0 == 1) is merged; the ``n_time_modes`` largest-mass survivors are kept,
        sorted by time. Returns ``[ncell, n_time_modes]`` float32 rel_t.
        """

        out = torch.full((ncell, n_time_modes), float("nan"), dtype=torch.float32, device=device)
        if hist_index.numel() == 0:
            return out

        # Step 1 -- coalesce the sparse hit list. hist_index holds one entry per
        # (frame, ray) sample, flat-encoded as cell*time_hist_bins + tbin. unique()
        # with counts collapses repeats into (unique flat index -> hit count). We
        # then decode each unique index back into its (cell, tbin) pair.
        uniq, counts = torch.unique(hist_index, return_counts=True)
        cell_of = torch.div(uniq, time_hist_bins, rounding_mode="floor")
        bin_of = uniq - cell_of * time_hist_bins

        # Step 2 -- group by cell. Sorting by cell id makes each cell's (tbin, count)
        # entries a contiguous slice [starts[ci] : starts[ci] + first_idx[ci]], so we
        # can iterate one cell at a time without a scatter.
        order = torch.argsort(cell_of)
        cell_s = cell_of[order]
        bin_s = bin_of[order].to(torch.float32)
        cnt_s = counts[order].to(torch.float32)

        # boundaries between cells: unique_consecutive gives, per occupied cell, the
        # number of distinct time bins it has; a running cumsum gives each cell's
        # start offset into the sorted arrays.
        uniq_cells, first_idx = torch.unique_consecutive(cell_s, return_counts=True)
        starts = torch.cumsum(torch.cat([torch.zeros(1, dtype=first_idx.dtype, device=device), first_idx[:-1]]), 0)
        merge_gap = float(mode_merge_rel)  # max bin gap that still counts as one mode
        centers_scale = 1.0 / time_hist_bins  # bin index -> rel_t in [0, 1]
        # Step 3 -- cluster each cell's time histogram into distinct visit times.
        for ci in range(uniq_cells.numel()):
            c = int(uniq_cells[ci].item())
            s = int(starts[ci].item())
            e = s + int(first_idx[ci].item())
            bins = bin_s[s:e]  # occupied time-bin indices for this cell
            w = cnt_s[s:e]  # their hit counts (mass)
            # sort the occupied bins along the time axis so gaps between consecutive
            # bins reveal separate visit clusters.
            border, _ = torch.sort(bins)
            wb = w[torch.argsort(bins)]
            total_mass = float(w.sum())

            # A "mode" is a run of occupied bins with no internal gap larger than
            # merge_gap bins. Split points are where consecutive occupied bins jump
            # by more than merge_gap; each resulting segment is one candidate visit.
            gaps = torch.diff(border)
            split_pts = torch.nonzero(gaps > merge_gap, as_tuple=False).reshape(-1)
            seg_starts = [0] + (split_pts + 1).tolist()
            seg_ends = split_pts.tolist() + [border.numel() - 1]

            modes = []  # (mass, centroid_rel_t) per surviving segment
            for ss, ee in zip(seg_starts, seg_ends):
                b = border[ss : ee + 1]
                ww = wb[ss : ee + 1]
                mass = float(ww.sum())
                # Drop low-mass segments: cross-frame jitter in a genuinely
                # single-visit cell shows up as a few stray bins; discarding them
                # keeps a single-cover cell from being reported as several visits.
                if mass < mode_min_mass_frac * total_mass:
                    continue
                # mass-weighted centroid of the segment, in rel_t units.
                centroid = float(((b + 0.5) * centers_scale * ww).sum() / ww.sum())
                modes.append((mass, centroid))

            # Keep the K highest-mass visit times, then report them in time order.
            modes.sort(key=lambda m: m[0], reverse=True)
            kept = sorted(t for _, t in modes[:n_time_modes])

            # circular wrap: rel_t is periodic (0 == 1). If the earliest and latest
            # surviving modes are both near the [0,1] boundary they are the SAME
            # sweep instant split across the wrap -- merge to one so a single-cover
            # sweep straddling the frame boundary is not counted as two visits.
            if len(kept) >= 2:
                edge = merge_gap / time_hist_bins
                if kept[0] <= edge and kept[-1] >= 1.0 - edge:
                    kept = sorted(kept[1:-1] + [0.0])

            for i, tval in enumerate(kept[:n_time_modes]):
                out[c, i] = tval
        return out

    # ---- query ---------------------------------------------------------------
    @staticmethod
    def _cell_index(
        az: torch.Tensor,
        el: torch.Tensor,
        n_az: int,
        n_el: int,
        az_lo: float,
        az_hi: float,
        el_lo: float,
        el_hi: float,
    ) -> torch.Tensor:
        """Map ``(az, el)`` angles (radians) to a flat cell index ``ei * n_az + ai``.

        Uniform binning of each axis over its ``[lo, hi]`` bound, clamped into range
        (out-of-FOV angles fall into the edge cell rather than wrapping). The flat
        layout is row-major in (elevation, azimuth) so it matches ``.reshape(n_el,
        n_az)`` of the grids.
        """
        ai = torch.floor((az - az_lo) / (az_hi - az_lo) * n_az).long().clamp(0, n_az - 1)
        ei = torch.floor((el - el_lo) / (el_hi - el_lo) * n_el).long().clamp(0, n_el - 1)
        return ei * n_az + ai

    def estimate_rel_t(self, dirs: torch.Tensor, *, max_search_radius: int = 5) -> torch.Tensor:
        """Estimate relative frame time ``rel_t in [0, 1]`` for directions.

        Looks up each direction's ``(az, el)`` cell in the reconstructed lattice.
        For cells the lattice never observed, expands a small square neighbour
        search (up to ``max_search_radius`` cells) to borrow the nearest observed
        rel_t. Cells still unresolved return ``NaN`` -- the caller must skip those
        virtual rays rather than fabricate a timestamp (fail-safe).

        Args:
            dirs: ``[M, 3]`` sensor-frame unit directions.
            max_search_radius: neighbour search radius (in cells) for empty cells.

        Returns:
            ``[M]`` float32 relative frame times in ``[0, 1]`` (``NaN`` where the
            lattice has no evidence even after the neighbour search).
        """
        dev = self.rel_t_grid.device
        dirs = dirs.to(dev)
        az, el = _dirs_to_az_el(dirs)
        flat = self.rel_t_grid.reshape(-1)
        ncell = flat.numel()
        lin = self._cell_index(
            az,
            el,
            self.n_bins_azimuth,
            self.n_bins_elevation,
            self.az_lo,
            self.az_hi,
            self.el_lo,
            self.el_hi,
        )
        # Direct cell lookup. NaN = the lattice never observed that cell (a hole in
        # the reconstructed pattern, e.g. a direction dropped in every sampled frame).
        out = flat[lin].clone()
        miss = torch.isnan(out)
        # Fill holes by an expanding square neighbour search: for radius r = 1, 2,
        # ... up to max_search_radius, scan the (2r+1)^2 cells around each still-
        # missing query and borrow the first observed rel_t found. Neighbouring
        # cells fire at nearly the same phase, so this is a good estimate for a
        # sparse hole; a hole too large to fill within max_search_radius stays NaN
        # (the caller must skip that virtual ray rather than fabricate a timestamp).
        # gid is the flat index; +/- n_bins_azimuth steps one elevation row, +/- 1
        # steps one azimuth column. clamp keeps candidate indices in range.
        r = 1
        while bool(miss.any()) and r <= max_search_radius:
            gid = lin[miss]
            best = torch.full((int(miss.sum()),), float("nan"), device=dev, dtype=out.dtype)
            for dei in range(-r, r + 1):
                for dai in range(-r, r + 1):
                    cand = (gid + dei * self.n_bins_azimuth + dai).clamp(0, ncell - 1)
                    val = flat[cand]
                    take = torch.isnan(best) & ~torch.isnan(val)
                    best[take] = val[take]
            out[miss] = best
            miss = torch.isnan(out)
            r += 1

        return out

    def estimate_rel_t_modes(self, dirs: torch.Tensor) -> torch.Tensor:
        """Per-direction distinct visit times ``rel_t`` for MULTI-COVER cells.

        Returns ``[M, K]`` (``K = n_time_modes``) of the distinct relative frame
        times each direction's cell was fired at, NaN-padded (fewer than ``K``
        real modes -> trailing NaNs; a cell the lattice never observed -> all NaN).
        The caller can emit one virtual ray per non-NaN mode (each is a genuine
        firing event / raydrop negative), or pick the nearest mode.

        DEGRADES TO SINGLE-COVER: when the estimator was built single-mode
        (``n_time_modes == 1``, the default), this returns the single mean per
        direction as ``[M, 1]`` -- identical information to :meth:`estimate_rel_t`
        (just shaped ``[M, 1]``). So callers can always use this method; it is a
        strict superset of the single-mode behaviour.
        """

        dev = self.rel_t_grid.device
        dirs = dirs.to(dev)
        az, el = _dirs_to_az_el(dirs)
        lin = self._cell_index(
            az,
            el,
            self.n_bins_azimuth,
            self.n_bins_elevation,
            self.az_lo,
            self.az_hi,
            self.el_lo,
            self.el_hi,
        )

        if self.rel_t_modes_grid is None:
            # single-mode build: there is only the mean grid, so every direction has
            # at most one visit time. Return it shaped [M, 1] (NaN where unobserved)
            # so callers get the same interface as the multi-mode path.
            return self.rel_t_grid.reshape(-1)[lin].reshape(-1, 1).clone()

        # multi-mode build: gather each direction's cell row of up to K distinct
        # visit times (NaN-padded). No neighbour search here -- multi-cover queries
        # come from sampled unoccupied tiles whose cell was observed by construction.
        modes_flat = self.rel_t_modes_grid.reshape(-1, self.n_time_modes)

        return modes_flat[lin].clone()

    # ---- diagnostics ---------------------------------------------------------
    def coverage(self) -> float:
        """Fraction of lattice cells that observed at least one ray."""
        return float((self.count_grid > 0).float().mean().item())

    def consistency_us(self, frame_duration_us: float) -> dict[str, float]:
        """Cross-frame rel_t stability (the aggregation-validity diagnostic).

        Returns median / p95 / max of per-cell cross-frame rel_t std, converted to
        microseconds via ``frame_duration_us``. Small values (<< shot spacing)
        confirm the firing pattern is stable enough for aggregation.
        """
        std = self.rel_t_std_grid.reshape(-1)
        std = std[~torch.isnan(std)]
        if std.numel() == 0:
            return {"median_us": float("nan"), "p95_us": float("nan"), "max_us": float("nan")}

        return {
            "median_us": float(std.median().item()) * frame_duration_us,
            "p95_us": float(std.quantile(0.95).item()) * frame_duration_us,
            "max_us": float(std.max().item()) * frame_duration_us,
        }
