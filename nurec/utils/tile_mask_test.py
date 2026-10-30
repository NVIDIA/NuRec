# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from nurec.utils.batch import LidarFrameLabels
from nurec.utils.tile_mask import (
    CameraToLidarTileMapping,
    _sample_jittered_world_cam_poses,
    _stamp_inactive_lidar_tile_rays_inplace,
    camera_tile_mask_to_lidar_mask,
    compute_active_tile_mask,
    mark_inactive_lidar_tile_rays_inplace,
    precompute_from_element_to_tile,
)
from nurec.utils.types import RayFlags


# Representative training camera geometry: a 1088x1920 pixel frame arranged
# as 68x120 tiles with 16 pixels per tile dimension.
CAMERA_TILES_Y = 68
CAMERA_TILES_X = 120
CAMERA_TILE_PX = 16
CAMERA_FRAME_H = CAMERA_TILES_Y * CAMERA_TILE_PX  # 1088
CAMERA_FRAME_W = CAMERA_TILES_X * CAMERA_TILE_PX  # 1920


def _make_flags(h: int, w: int, invalid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Helper: build an int32 ray_flags tensor. Pixels in *invalid_mask* get RayFlags.INVALID set."""
    flags = torch.full((h, w), RayFlags.RGB_LABEL.value, dtype=torch.int32)
    if invalid_mask is not None:
        flags[invalid_mask] = flags[invalid_mask] | RayFlags.INVALID.value
    return flags


class TestComputeActiveTileMask:
    """The function always returns ``(B, tiles_y, tiles_x)``. B is 1 for 2-D input, and
    equals the input batch for 3-D/4-D input."""

    def test_all_valid(self):
        """All pixels valid -> all tiles active."""
        flags = _make_flags(32, 48)
        mask = compute_active_tile_mask(flags, tile_width=16, tile_height=16)
        assert mask.shape == (1, 2, 3)
        assert mask.dtype == torch.bool
        assert mask.all()

    def test_all_invalid(self):
        """All pixels INVALID -> all tiles inactive."""
        flags = torch.full((32, 48), RayFlags.INVALID.value, dtype=torch.int32)
        mask = compute_active_tile_mask(flags, tile_width=16, tile_height=16)
        assert mask.shape == (1, 2, 3)
        assert not mask.any()

    def test_single_valid_pixel_activates_tile(self):
        """One valid pixel in an otherwise-invalid tile makes it active."""
        h, w, tw, th = 32, 32, 16, 16
        flags = torch.full((h, w), RayFlags.INVALID.value, dtype=torch.int32)
        # Place one valid pixel at (3, 5) -> tile (0, 0)
        flags[3, 5] = RayFlags.RGB_LABEL.value
        mask = compute_active_tile_mask(flags, tile_width=tw, tile_height=th)
        assert mask.shape == (1, 2, 2)
        assert mask[0, 0, 0].item() is True
        assert not mask[0, 0, 1]
        assert not mask[0, 1, 0]
        assert not mask[0, 1, 1]

    def test_checkerboard_pattern(self):
        """Alternating valid/invalid tiles."""
        h, w, tw, th = 32, 32, 16, 16
        flags = torch.full((h, w), RayFlags.INVALID.value, dtype=torch.int32)
        # Top-left tile (0,0): make valid
        flags[:16, :16] = RayFlags.RGB_LABEL.value
        # Bottom-right tile (1,1): make valid
        flags[16:, 16:] = RayFlags.RGB_LABEL.value
        mask = compute_active_tile_mask(flags, tile_width=tw, tile_height=th)
        expected = torch.tensor([[[True, False], [False, True]]])
        assert mask.shape == (1, 2, 2)
        assert torch.equal(mask, expected)

    def test_non_divisible_dimensions(self):
        """Frame dimensions not evenly divisible by tile size."""
        h, w = 30, 50
        flags = _make_flags(h, w)
        mask = compute_active_tile_mask(flags, tile_width=16, tile_height=16)
        # ceil(30/16)=2, ceil(50/16)=4
        assert mask.shape == (1, 2, 4)
        assert mask.all()

    def test_non_divisible_partial_tile_invalid(self):
        """Partial tile at edge is inactive when its pixels are all INVALID."""
        h, w, tw, th = 20, 20, 16, 16
        # tiles_y=2, tiles_x=2. Tile (1,1) covers rows 16-19, cols 16-19 (4x4 pixels).
        flags = _make_flags(h, w)
        # Make the bottom-right partial tile fully INVALID
        flags[16:, 16:] = flags[16:, 16:] | RayFlags.INVALID.value
        mask = compute_active_tile_mask(flags, tile_width=tw, tile_height=th)
        assert mask.shape == (1, 2, 2)
        assert mask[0, 0, 0]
        assert mask[0, 0, 1]
        assert mask[0, 1, 0]
        assert not mask[0, 1, 1]

    def test_input_shape_bhw1(self):
        """Accepts (B, H, W, 1) input shape and preserves B."""
        flags_2d = _make_flags(32, 32)
        flags_4d = flags_2d.unsqueeze(0).unsqueeze(-1)
        assert flags_4d.shape == (1, 32, 32, 1)
        mask = compute_active_tile_mask(flags_4d, tile_width=16, tile_height=16)
        assert mask.shape == (1, 2, 2)
        assert mask.all()

    def test_input_shape_bhw(self):
        """Accepts (B, H, W) input shape and preserves B."""
        flags_2d = _make_flags(32, 32)
        flags_3d = flags_2d.unsqueeze(0)
        assert flags_3d.shape == (1, 32, 32)
        mask = compute_active_tile_mask(flags_3d, tile_width=16, tile_height=16)
        assert mask.shape == (1, 2, 2)
        assert mask.all()

    def test_batched_input_produces_per_frame_mask(self):
        """(B>1, H, W, 1) returns (B, tiles_y, tiles_x) with per-frame validity."""
        # Frame 0: all valid. Frame 1: all invalid except one pixel.
        frame0 = _make_flags(32, 32)
        frame1 = torch.full((32, 32), RayFlags.INVALID.value, dtype=torch.int32)
        frame1[3, 5] = RayFlags.RGB_LABEL.value  # tile (0, 0)
        flags = torch.stack([frame0, frame1], dim=0).unsqueeze(-1)  # (2, 32, 32, 1)

        mask = compute_active_tile_mask(flags, tile_width=16, tile_height=16)
        assert mask.shape == (2, 2, 2)
        # Frame 0: all active.
        assert mask[0].all()
        # Frame 1: only tile (0, 0) active.
        assert mask[1, 0, 0]
        assert not mask[1, 0, 1]
        assert not mask[1, 1, 0]
        assert not mask[1, 1, 1]

    def test_non_square_tiles(self):
        """Non-square tile dimensions."""
        h, w = 32, 64
        flags = torch.full((h, w), RayFlags.INVALID.value, dtype=torch.int32)
        # Make first tile (rows 0-7, cols 0-31) valid
        flags[:8, :32] = RayFlags.RGB_LABEL.value
        mask = compute_active_tile_mask(flags, tile_width=32, tile_height=8)
        # tiles_y=4, tiles_x=2
        assert mask.shape == (1, 4, 2)
        assert mask[0, 0, 0]
        assert not mask[0, 0, 1]
        assert not mask[0, 1, 0]

    def test_other_flags_dont_matter(self):
        """Pixels with other flags set (but not INVALID) are still valid."""
        flags = torch.full(
            (16, 16),
            (RayFlags.RGB_LABEL | RayFlags.SKY_SEMANTIC | RayFlags.DROPPED).value,
            dtype=torch.int32,
        )
        mask = compute_active_tile_mask(flags, tile_width=16, tile_height=16)
        assert mask.shape == (1, 1, 1)
        assert mask[0, 0, 0]

    def test_invalid_combined_with_other_flags(self):
        """INVALID bit set alongside other flags still marks pixel as invalid."""
        flags = torch.full(
            (16, 16),
            (RayFlags.RGB_LABEL | RayFlags.INVALID).value,
            dtype=torch.int32,
        )
        mask = compute_active_tile_mask(flags, tile_width=16, tile_height=16)
        assert mask.shape == (1, 1, 1)
        assert not mask[0, 0, 0]

    def test_ego_mask_pattern(self):
        """Simulate a typical ego-mask: bottom rows of the image are INVALID."""
        h, w, tw, th = 64, 96, 16, 16
        flags = _make_flags(h, w)
        # Bottom 16 rows fully invalid (simulating ego/car hood)
        flags[48:, :] = flags[48:, :] | RayFlags.INVALID.value
        mask = compute_active_tile_mask(flags, tile_width=tw, tile_height=th)
        # tiles_y=4, tiles_x=6. Last row of tiles (row 3) should be inactive.
        assert mask.shape == (1, 4, 6)
        assert mask[0, :3, :].all()
        assert not mask[0, 3, :].any()

    def test_small_frame_single_tile(self):
        """Frame smaller than one tile."""
        flags = _make_flags(5, 7)
        mask = compute_active_tile_mask(flags, tile_width=16, tile_height=16)
        assert mask.shape == (1, 1, 1)
        assert mask[0, 0, 0]

    def test_dtype_assertion(self):
        """Non-int32 input raises a TypeError."""
        flags = torch.zeros(16, 16, dtype=torch.float32)
        with pytest.raises(TypeError, match="ray_flags must be int32"):
            compute_active_tile_mask(flags, tile_width=16, tile_height=16)

    def test_trailing_channel_c_gt_1_rejected(self):
        """(B, H, W, C>1) must raise — squeeze(-1) would silently leave a 4-D shape."""
        flags = torch.zeros((1, 32, 32, 3), dtype=torch.int32)
        with pytest.raises(ValueError, match="trailing channel dim of size 1"):
            compute_active_tile_mask(flags, tile_width=16, tile_height=16)

    def test_output_device_matches_input(self):
        """Output is on the same device as input (CPU test)."""
        flags = _make_flags(32, 32)
        mask = compute_active_tile_mask(flags, tile_width=16, tile_height=16)
        assert mask.device == flags.device

    def test_production_camera_size_all_valid(self):
        """Production camera geometry (1088x1920, 16px tiles -> 68x120 grid), all valid."""
        flags = _make_flags(CAMERA_FRAME_H, CAMERA_FRAME_W)
        mask = compute_active_tile_mask(flags, tile_width=CAMERA_TILE_PX, tile_height=CAMERA_TILE_PX)
        assert mask.shape == (1, CAMERA_TILES_Y, CAMERA_TILES_X)
        assert mask.sum().item() == CAMERA_TILES_Y * CAMERA_TILES_X

    def test_production_camera_size_ego_hood(self):
        """Production camera geometry with a realistic hood-mask on bottom ~12 rows of tiles."""
        flags = _make_flags(CAMERA_FRAME_H, CAMERA_FRAME_W)
        hood_start_row = 56 * CAMERA_TILE_PX  # last 12 tile rows
        flags[hood_start_row:, :] = flags[hood_start_row:, :] | RayFlags.INVALID.value

        mask = compute_active_tile_mask(flags, tile_width=CAMERA_TILE_PX, tile_height=CAMERA_TILE_PX)
        assert mask.shape == (1, CAMERA_TILES_Y, CAMERA_TILES_X)
        assert mask[0, :56, :].all()
        assert not mask[0, 56:, :].any()


def _make_mapping(
    element_to_tile: torch.Tensor,
    n_bins_azimuth: int,
    n_bins_elevation: int,
    cam_tiles_y: int = 1,
    cam_tiles_x: int = 1,
    mapping_matrix: torch.Tensor | None = None,
) -> CameraToLidarTileMapping:
    """Build a minimal ``CameraToLidarTileMapping`` for unit-testing downstream helpers."""
    n_lidar_tiles = n_bins_azimuth * n_bins_elevation
    n_camera_tiles = cam_tiles_y * cam_tiles_x
    if mapping_matrix is None:
        mapping_matrix = torch.zeros((n_lidar_tiles, n_camera_tiles), dtype=torch.bool)
    return CameraToLidarTileMapping(
        mapping=mapping_matrix,
        n_lidar_tiles=n_lidar_tiles,
        n_camera_tiles=n_camera_tiles,
        cam_tiles_y=cam_tiles_y,
        cam_tiles_x=cam_tiles_x,
        n_bins_azimuth=n_bins_azimuth,
        n_bins_elevation=n_bins_elevation,
        lidar_tile_grid_shape=(n_bins_elevation, n_bins_azimuth),
        element_to_tile=element_to_tile,
    )


class TestMarkInactiveLidarTileRays:
    """Coverage for the 2-D and 4-D ``ray_flags`` branches of ``mark_inactive_lidar_tile_rays_inplace``.

    Layout used in these tests::

        element_to_tile = [[ 0,  1, -1],     n_bins_azimuth   = 2
                           [ 2,  3, -1]]     n_bins_elevation = 2
                                             n_lidar_tiles    = 4

    i.e. pixel (r, c) maps to lidar tile-id ``r*2 + c`` for c in {0,1};
    the third column is unassigned and must never be stamped.
    """

    @staticmethod
    def _layout():
        element_to_tile = torch.tensor(
            [[0, 1, -1], [2, 3, -1]],
            dtype=torch.int64,
        )
        return element_to_tile, 2, 2  # element_to_tile, n_bins_azim, n_bins_elev

    # -- 4-D ray_flags path --------------------------------------------------

    def test_4d_batched_per_frame_invalidation(self):
        """Each frame in the batch gets its own pattern — no cross-contamination."""
        element_to_tile, n_azim, n_elev = self._layout()
        mapping = _make_mapping(element_to_tile, n_azim, n_elev)

        B, H, W = 2, element_to_tile.shape[0], element_to_tile.shape[1]
        # Frame 0: tile 0 active, others inactive -> rays at (0,0) stay valid, (0,1)/(1,0)/(1,1) flipped.
        # Frame 1: tile 3 active, others inactive -> only (1,1) stays valid.
        mask = torch.zeros((B, n_azim, n_elev), dtype=torch.bool)
        mask[0, 0, 0] = True  # lidar tile 0 (= 0*2+0) active for frame 0
        mask[1, 1, 1] = True  # lidar tile 3 (= 1*2+1) active for frame 1

        ray_flags = torch.full((B, H, W, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32)
        _stamp_inactive_lidar_tile_rays_inplace(ray_flags, mask, mapping)

        def is_invalid(b, r, c):
            return bool((ray_flags[b, r, c, 0] & RayFlags.INVALID.value) != 0)

        # Frame 0: only (0,0) stays valid; (0,1)/(1,0)/(1,1) are assigned + inactive -> INVALID.
        assert not is_invalid(0, 0, 0)
        assert is_invalid(0, 0, 1)
        assert is_invalid(0, 1, 0)
        assert is_invalid(0, 1, 1)
        # Frame 1: only (1,1) stays valid; (0,0)/(0,1)/(1,0) INVALID.
        assert is_invalid(1, 0, 0)
        assert is_invalid(1, 0, 1)
        assert is_invalid(1, 1, 0)
        assert not is_invalid(1, 1, 1)
        # Unassigned column (c=2) is never stamped, for either frame.
        for b in range(B):
            for r in range(H):
                assert not is_invalid(b, r, 2), f"unassigned pixel ({b}, {r}, 2) must not be INVALID"

    def test_4d_modifies_in_place_and_or_preserves_existing_bits(self):
        """INVALID is OR'd into pre-existing flags; other bits are preserved."""
        element_to_tile, n_azim, n_elev = self._layout()
        mapping = _make_mapping(element_to_tile, n_azim, n_elev)

        B, H, W = 1, 2, 3
        base = (RayFlags.RGB_LABEL | RayFlags.SKY_SEMANTIC).value
        ray_flags = torch.full((B, H, W, 1), base, dtype=torch.int32)
        ray_flags_id = id(ray_flags)

        mask = torch.zeros((B, n_azim, n_elev), dtype=torch.bool)  # all inactive
        _stamp_inactive_lidar_tile_rays_inplace(ray_flags, mask, mapping)
        assert id(ray_flags) == ray_flags_id, "function must mutate in place, not reassign"

        for r in range(H):
            # Assigned columns: INVALID set on top of the original bits.
            for c in (0, 1):
                assert (ray_flags[0, r, c, 0] & RayFlags.INVALID.value) != 0
                assert (ray_flags[0, r, c, 0] & RayFlags.RGB_LABEL.value) != 0
                assert (ray_flags[0, r, c, 0] & RayFlags.SKY_SEMANTIC.value) != 0
            # Unassigned column: untouched.
            assert ray_flags[0, r, 2, 0].item() == base

    def test_4d_all_tiles_active_leaves_ray_flags_unchanged(self):
        """When every tile is active the function is a no-op (bitwise)."""
        element_to_tile, n_azim, n_elev = self._layout()
        mapping = _make_mapping(element_to_tile, n_azim, n_elev)
        B, H, W = 2, 2, 3
        mask = torch.ones((B, n_azim, n_elev), dtype=torch.bool)
        ray_flags = torch.full((B, H, W, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32)
        expected = ray_flags.clone()
        _stamp_inactive_lidar_tile_rays_inplace(ray_flags, mask, mapping)
        assert torch.equal(ray_flags, expected)

    # -- 2-D ray_flags path --------------------------------------------------

    def test_2d_ray_flags_with_b1_mask(self):
        """2-D ray_flags are supported for the B==1 special case."""
        element_to_tile, n_azim, n_elev = self._layout()
        mapping = _make_mapping(element_to_tile, n_azim, n_elev)

        mask = torch.zeros((1, n_azim, n_elev), dtype=torch.bool)
        mask[0, 0, 0] = True  # only lidar tile 0 active

        ray_flags = torch.full((2, 3), RayFlags.RGB_LABEL.value, dtype=torch.int32)
        _stamp_inactive_lidar_tile_rays_inplace(ray_flags, mask, mapping)

        invalid = (ray_flags & RayFlags.INVALID.value) != 0
        expected_invalid = torch.tensor(
            [[False, True, False], [True, True, False]],
        )
        assert torch.equal(invalid, expected_invalid)

    def test_2d_ray_flags_rejected_when_b_gt_1(self):
        """2-D ray_flags cannot express per-frame information; reject B>1 masks."""
        element_to_tile, n_azim, n_elev = self._layout()
        mapping = _make_mapping(element_to_tile, n_azim, n_elev)
        mask = torch.zeros((2, n_azim, n_elev), dtype=torch.bool)
        ray_flags = torch.zeros((2, 3), dtype=torch.int32)
        with pytest.raises(ValueError, match="2D ray_flags require B == 1"):
            _stamp_inactive_lidar_tile_rays_inplace(ray_flags, mask, mapping)

    # -- Input validation ----------------------------------------------------

    def test_rejects_2d_active_tile_mask(self):
        """Un-batched ``(n_azim, n_elev)`` mask is rejected."""
        element_to_tile, n_azim, n_elev = self._layout()
        mapping = _make_mapping(element_to_tile, n_azim, n_elev)
        mask_2d = torch.zeros((n_azim, n_elev), dtype=torch.bool)  # missing B dim
        ray_flags = torch.zeros((1, 2, 3, 1), dtype=torch.int32)
        with pytest.raises(ValueError, match="batched active_tile_mask"):
            _stamp_inactive_lidar_tile_rays_inplace(ray_flags, mask_2d, mapping)

    def test_rejects_non_int32_ray_flags(self):
        element_to_tile, n_azim, n_elev = self._layout()
        mapping = _make_mapping(element_to_tile, n_azim, n_elev)
        mask = torch.zeros((1, n_azim, n_elev), dtype=torch.bool)
        ray_flags = torch.zeros((1, 2, 3, 1), dtype=torch.int64)  # wrong dtype
        with pytest.raises(TypeError):
            _stamp_inactive_lidar_tile_rays_inplace(ray_flags, mask, mapping)

    def test_rejects_unsupported_ray_flags_ndim(self):
        element_to_tile, n_azim, n_elev = self._layout()
        mapping = _make_mapping(element_to_tile, n_azim, n_elev)
        mask = torch.zeros((1, n_azim, n_elev), dtype=torch.bool)
        ray_flags_3d = torch.zeros((1, 2, 3), dtype=torch.int32)  # neither 2-D nor 4-D
        with pytest.raises(ValueError, match="Unexpected ray_flags shape"):
            _stamp_inactive_lidar_tile_rays_inplace(ray_flags_3d, mask, mapping)

    def test_rejects_element_to_tile_shape_mismatch_4d(self):
        """A wrong-sized ``element_to_tile`` raises a named error, not a bare broadcast error."""
        element_to_tile, n_azim, n_elev = self._layout()  # (2, 3)
        mapping = _make_mapping(element_to_tile, n_azim, n_elev)
        mask = torch.zeros((1, n_azim, n_elev), dtype=torch.bool)
        ray_flags = torch.zeros((1, 4, 5, 1), dtype=torch.int32)  # (n_rows, n_cols) = (4, 5) != (2, 3)
        with pytest.raises(ValueError, match=r"element_to_tile shape \(2, 3\).*\(4, 5\)"):
            _stamp_inactive_lidar_tile_rays_inplace(ray_flags, mask, mapping)

    def test_rejects_element_to_tile_shape_mismatch_2d(self):
        """Same named error on the 2-D ``ray_flags`` branch."""
        element_to_tile, n_azim, n_elev = self._layout()  # (2, 3)
        mapping = _make_mapping(element_to_tile, n_azim, n_elev)
        mask = torch.zeros((1, n_azim, n_elev), dtype=torch.bool)
        ray_flags = torch.zeros((4, 5), dtype=torch.int32)
        with pytest.raises(ValueError, match=r"element_to_tile shape \(2, 3\).*\(4, 5\)"):
            _stamp_inactive_lidar_tile_rays_inplace(ray_flags, mask, mapping)


class TestMarkInactiveLidarTileRaysInvalidatesCaches:
    """``mark_inactive_lidar_tile_rays_inplace`` must drop the cached valid-ray counters.

    The lidar losses normalize by ``1 / n_valid_lidar`` and ``1 / n_valid_raydrop``; those
    counters are warmed up before ``forward()`` (dataset load + ``collate_fn``). Stamping
    INVALID without invalidating them silently scales every lidar loss down.
    """

    @staticmethod
    def _mapping_and_mask():
        # pixel (r, c) -> lidar tile r*2 + c for c in {0, 1}; column 2 is unassigned.
        element_to_tile = torch.tensor([[0, 1, -1], [2, 3, -1]], dtype=torch.int64)
        mapping = _make_mapping(element_to_tile, 2, 2)
        mask = torch.zeros((1, 2, 2), dtype=torch.bool)
        mask[0, 0, 0] = True  # only lidar tile 0 active
        return mapping, mask

    @staticmethod
    def _make_labels() -> LidarFrameLabels:
        # (1, 2, 3, 1) flags: all rays valid to start with, one already DROPPED.
        flags = torch.full((1, 2, 3, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32)
        flags[0, 1, 2, 0] |= RayFlags.DROPPED.value
        return LidarFrameLabels(flags=flags)

    def test_stamping_refreshes_warm_n_valid_caches(self):
        mapping, mask = self._mapping_and_mask()
        labels = self._make_labels()

        # Warm the caches, exactly as the dataset / collate path does before forward().
        warm_n_valid_lidar = labels.n_valid_lidar
        warm_n_valid_raydrop = labels.n_valid_raydrop
        assert labels._n_valid_lidar is not None and labels._n_valid_raydrop is not None

        mark_inactive_lidar_tile_rays_inplace(labels, mask, mapping)

        # Freshly recomputed reference counts from the post-stamp flags.
        assert labels.flags is not None
        invalid = (labels.flags & RayFlags.INVALID.value) != 0
        dropped = (labels.flags & RayFlags.DROPPED.value) != 0
        expected_n_valid_lidar = int((~invalid & ~dropped).sum().item())
        expected_n_valid_raydrop = int((~invalid).sum().item())

        assert labels.n_valid_lidar == expected_n_valid_lidar
        assert labels.n_valid_raydrop == expected_n_valid_raydrop
        # And the stamp really did shrink the counts, i.e. the test would fail on a stale cache.
        assert expected_n_valid_lidar < warm_n_valid_lidar
        assert expected_n_valid_raydrop < warm_n_valid_raydrop

    def test_stamping_marks_the_expected_rays(self):
        mapping, mask = self._mapping_and_mask()
        labels = self._make_labels()
        mark_inactive_lidar_tile_rays_inplace(labels, mask, mapping)

        assert labels.flags is not None
        invalid = ((labels.flags[..., 0] & RayFlags.INVALID.value) != 0)[0]
        expected = torch.tensor([[False, True, False], [True, True, False]])
        assert torch.equal(invalid, expected)

    def test_cold_caches_stay_cold(self):
        """Stamping on labels whose caches were never warmed leaves them unpopulated."""
        mapping, mask = self._mapping_and_mask()
        labels = self._make_labels()
        assert labels._n_valid_lidar is None and labels._n_valid_raydrop is None
        mark_inactive_lidar_tile_rays_inplace(labels, mask, mapping)
        assert labels._n_valid_lidar is None and labels._n_valid_raydrop is None

    def test_rejects_missing_flags(self):
        mapping, mask = self._mapping_and_mask()
        labels = LidarFrameLabels()
        with pytest.raises(ValueError, match="labels.flags is None"):
            mark_inactive_lidar_tile_rays_inplace(labels, mask, mapping)

    def test_element_to_tile_mismatch_raises_named_error(self):
        """Public entry point surfaces the named shape error rather than a broadcast failure."""
        mapping, mask = self._mapping_and_mask()  # element_to_tile is (2, 3)
        labels = LidarFrameLabels(flags=torch.zeros((1, 4, 5, 1), dtype=torch.int32))
        with pytest.raises(ValueError, match=r"element_to_tile shape \(2, 3\).*\(4, 5\)"):
            mark_inactive_lidar_tile_rays_inplace(labels, mask, mapping)


class TestCameraTileMaskToLidarMask:
    """Coverage for the batched ``camera_tile_mask_to_lidar_mask``."""

    def test_per_frame_reduction_any_over_camera_tiles(self):
        """A lidar tile is active in frame b iff any of its mapped camera tiles are active in b."""
        # 3 lidar tiles, 4 camera tiles. Lidar 0 <- cam {0,1}; lidar 1 <- cam {2}; lidar 2 <- cam {3}.
        mapping_matrix = torch.tensor(
            [
                [True, True, False, False],  # lidar 0
                [False, False, True, False],  # lidar 1
                [False, False, False, True],
            ],  # lidar 2
            dtype=torch.bool,
        )
        mapping = _make_mapping(
            element_to_tile=torch.empty((0, 0), dtype=torch.int64),  # unused here
            n_bins_azimuth=3,
            n_bins_elevation=1,
            cam_tiles_y=2,
            cam_tiles_x=2,
            mapping_matrix=mapping_matrix,
        )

        # Frame 0: cam tiles {0, 3} active. Frame 1: cam tiles {2} only.
        cam_mask = torch.tensor(
            [
                [[True, False], [False, True]],  # frame 0: cam 0 and 3 active
                [[False, False], [True, False]],
            ],  # frame 1: cam 2 active
            dtype=torch.bool,
        )
        assert cam_mask.shape == (2, 2, 2)

        lidar_mask = camera_tile_mask_to_lidar_mask(cam_mask, mapping)
        # Frame 0: lidar 0 (cam 0 or 1) -> True, lidar 1 (cam 2) -> False, lidar 2 (cam 3) -> True.
        # Frame 1: lidar 0 -> False, lidar 1 -> True, lidar 2 -> False.
        expected = torch.tensor(
            [[True, False, True], [False, True, False]],
            dtype=torch.bool,
        )
        assert lidar_mask.shape == (2, 3)
        assert torch.equal(lidar_mask, expected)

    def test_rejects_2d_camera_mask(self):
        """2-D camera mask (missing batch dim) is rejected."""
        mapping = _make_mapping(
            element_to_tile=torch.empty((0, 0), dtype=torch.int64),
            n_bins_azimuth=1,
            n_bins_elevation=1,
            cam_tiles_y=1,
            cam_tiles_x=1,
            mapping_matrix=torch.ones((1, 1), dtype=torch.bool),
        )
        with pytest.raises(ValueError, match="batched camera_active_tile_mask"):
            camera_tile_mask_to_lidar_mask(torch.ones((1, 1), dtype=torch.bool), mapping)

    @staticmethod
    def _mapping_for_grid(cam_tiles_y: int, cam_tiles_x: int) -> CameraToLidarTileMapping:
        n_camera_tiles = cam_tiles_y * cam_tiles_x
        return _make_mapping(
            element_to_tile=torch.empty((0, 0), dtype=torch.int64),  # unused here
            n_bins_azimuth=3,
            n_bins_elevation=1,
            cam_tiles_y=cam_tiles_y,
            cam_tiles_x=cam_tiles_x,
            mapping_matrix=torch.zeros((3, n_camera_tiles), dtype=torch.bool),
        )

    def test_rejects_camera_mask_with_mismatched_tile_grid(self):
        """A mask on a different camera tile grid than the mapping was built for is rejected."""
        mapping = self._mapping_for_grid(cam_tiles_y=2, cam_tiles_x=8)
        cam_mask = torch.zeros((1, 3, 3), dtype=torch.bool)  # 9 tiles vs the mapping's 16
        with pytest.raises(ValueError, match=r"3x3.*does not match.*2x8"):
            camera_tile_mask_to_lidar_mask(cam_mask, mapping)

    def test_rejects_transposed_camera_grid_with_equal_tile_count(self):
        """The silent failure mode: same numel, transposed grid -> wrong camera->lidar adjacency."""
        mapping = self._mapping_for_grid(cam_tiles_y=2, cam_tiles_x=8)
        cam_mask = torch.zeros((1, 8, 2), dtype=torch.bool)  # identical tile *count*, wrong layout
        assert cam_mask[0].numel() == mapping.n_camera_tiles
        with pytest.raises(ValueError, match=r"8x2.*does not match.*2x8"):
            camera_tile_mask_to_lidar_mask(cam_mask, mapping)

    def test_matching_camera_grid_is_accepted(self):
        """The correctly-shaped mask still goes through."""
        mapping = self._mapping_for_grid(cam_tiles_y=2, cam_tiles_x=8)
        cam_mask = torch.zeros((1, 2, 8), dtype=torch.bool)
        lidar_mask = camera_tile_mask_to_lidar_mask(cam_mask, mapping)
        assert lidar_mask.shape == (1, mapping.n_lidar_tiles)


def _rotation_angle_rad(R: torch.Tensor) -> torch.Tensor:
    """Axis-angle magnitude of a (batch of) 3x3 rotation matrix(es).

    For a rotation ``R`` with axis-angle vector ``rvec``, ``angle = ||rvec||_2``
    and ``trace(R) = 1 + 2 cos(angle)``, so ``angle = acos((trace - 1) / 2)``.
    """
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((trace - 1) / 2).clamp(-1.0, 1.0))


def _make_non_trivial_pose(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Build a non-identity SE(3) pose (rotation + translation) for bound-check tests.

    Uses ``float64`` by default to keep the recovered ``D_inv`` close enough to
    the analytical bound that we can assert without padding the tolerance.
    """
    angle = 0.5
    c, s = math.cos(angle), math.sin(angle)
    T = torch.eye(4, dtype=dtype)
    T[:3, :3] = torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    T[:3, 3] = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    return T


class TestSampleJitteredWorldCamPoses:
    """Coverage for ``_sample_jittered_world_cam_poses``.

    The helper builds the SE(3) jitter envelope used by
    :func:`precompute_from_element_to_tile`; a sign-flip in the Rodrigues
    block, an axis swap in the skew matrix, or a flipped ``D^-1 @ T`` jitter-frame
    convention would silently shrink (or distort) the cached camera→lidar mapping
    and kill gradient flow on rays that should have stayed live.
    """

    def test_n_samples_one_with_identity_returns_identity(self):
        """``n_samples == 1`` with ``T = I`` returns a single-row stack equal to identity."""
        T = torch.eye(4)
        out = _sample_jittered_world_cam_poses(T, n_samples=1, translation_bound_m=0.3, rotation_bound_rad=0.05)
        assert out.shape == (1, 4, 4)
        assert torch.equal(out[0], T)

    def test_first_row_is_input_pose(self):
        """For any ``n_samples >= 1``, ``out[0]`` is the un-jittered input pose."""
        T = _make_non_trivial_pose(dtype=torch.float32)
        for n in (1, 2, 8, 32):
            out = _sample_jittered_world_cam_poses(T, n_samples=n, translation_bound_m=0.3, rotation_bound_rad=0.05)
            assert out.shape == (n, 4, 4)
            assert torch.equal(out[0], T), f"first row not preserved for n_samples={n}"

    def test_n_samples_eq_0_returns_empty_stack(self):
        """``n_samples == 0`` returns an empty ``(0, 4, 4)`` stack, honoring the documented shape."""
        T = _make_non_trivial_pose(dtype=torch.float32)
        out = _sample_jittered_world_cam_poses(T, n_samples=0, translation_bound_m=0.3, rotation_bound_rad=0.05)
        assert out.shape == (0, 4, 4)

    def test_n_samples_eq_1_returns_identity_only(self):
        """``n_samples == 1`` short-circuits to identity (jitter disabled)."""
        T = _make_non_trivial_pose(dtype=torch.float32)
        out = _sample_jittered_world_cam_poses(T, n_samples=1, translation_bound_m=0.3, rotation_bound_rad=0.05)
        assert out.shape == (1, 4, 4)
        assert torch.equal(out[0], T)

    def test_n_samples_negative_raises(self):
        """Negative ``n_samples`` has no use case — raises rather than silently returning identity."""
        T = _make_non_trivial_pose(dtype=torch.float32)
        with pytest.raises(ValueError, match="n_samples must be"):
            _sample_jittered_world_cam_poses(T, n_samples=-1, translation_bound_m=0.3, rotation_bound_rad=0.05)

    def test_translation_jitter_within_envelope(self):
        """``(jittered_i @ T^{-1})[:3, 3]`` magnitude is bounded by the per-axis envelope.

        The math:
          ``jittered_i = D_inv_i @ T``  ⇒  ``jittered_i @ T^{-1} = D_inv_i``.
          For ``D_i = [[R_delta_i, trans_i], [0, 1]]``, the translation block of
          ``D_inv_i`` is ``-R_delta_i^T @ trans_i``. Rotation is norm-preserving, so
          ``||relative_translation||_2 == ||trans_i||_2 <= sqrt(3) * translation_bound_m``,
          and recovering ``trans_i = -R_delta_i @ relative_translation`` lets us assert
          the per-axis bound exactly.
        """
        torch.manual_seed(0)
        T = _make_non_trivial_pose(dtype=torch.float64)
        bound = 0.3
        # Use many samples to exercise the envelope upper extremes.
        out = _sample_jittered_world_cam_poses(T, n_samples=1024, translation_bound_m=bound, rotation_bound_rad=0.05)
        T_inv = torch.linalg.inv(T)
        relative = out @ T_inv  # (n_samples, 4, 4) — D_inv stack; first sample is identity
        rel_t = relative[:, :3, 3]
        rel_R = relative[:, :3, :3]

        # L2-magnitude bound: ||relative_translation|| == ||trans|| <= sqrt(3) * bound.
        l2_bound = math.sqrt(3) * bound
        assert rel_t.norm(dim=-1).max().item() <= l2_bound + 1e-9
        # Per-axis bound: recover trans = -R_delta @ relative_translation and check L_inf.
        # rel_R = R_delta^T, so R_delta = rel_R.transpose(-1, -2).
        R_delta = rel_R.transpose(-1, -2)
        recovered_trans = -torch.einsum("nij,nj->ni", R_delta, rel_t)
        assert recovered_trans.abs().max().item() <= bound + 1e-9
        # Sanity: 1023 jittered samples should at least once approach the envelope.
        assert recovered_trans.abs().max().item() > 0.5 * bound

    def test_rotation_jitter_within_envelope(self):
        """Axis-angle of the jitter rotation block is bounded by ``sqrt(3) * rotation_bound_rad``.

        ``rvec_i`` is uniformly sampled per-axis in ``[-bound, +bound]``, so its L2 norm
        — which equals the rotation angle of ``R_delta_i = exp(skew(rvec_i))`` — is
        bounded by ``sqrt(3) * bound``. The angle of ``R_delta_i^T`` (the rotation
        block of ``D_inv_i``) is identical to that of ``R_delta_i``.

        Also asserts ``rel_R`` is a true rotation matrix (orthogonal, det == 1).
        A sign error or axis-ordering flip in the Rodrigues skew matrix would
        make ``skew`` non-antisymmetric, and ``matrix_exp(skew)`` would no
        longer be orthogonal — the angle-from-trace check above can still come
        out within bound for many such bugs, so this catches the long tail.
        """
        torch.manual_seed(0)
        T = _make_non_trivial_pose(dtype=torch.float64)
        bound = 0.05
        out = _sample_jittered_world_cam_poses(T, n_samples=1024, translation_bound_m=0.3, rotation_bound_rad=bound)
        T_inv = torch.linalg.inv(T)
        rel_R = (out @ T_inv)[:, :3, :3]
        angles = _rotation_angle_rad(rel_R)
        assert angles.max().item() <= math.sqrt(3) * bound + 1e-9
        # Sanity: at least one sample exercises a non-trivial fraction of the envelope.
        assert angles.max().item() > 0.5 * bound

        # Orthogonality + unit determinant — catches non-antisymmetric `skew` bugs.
        eye_batched = torch.eye(3, dtype=rel_R.dtype).expand_as(rel_R)
        torch.testing.assert_close(rel_R @ rel_R.transpose(-1, -2), eye_batched, atol=1e-9, rtol=0.0)
        torch.testing.assert_close(
            torch.linalg.det(rel_R),
            torch.ones(rel_R.shape[0], dtype=rel_R.dtype),
            atol=1e-9,
            rtol=0.0,
        )

    def test_seeded_determinism(self):
        """Same ``torch.manual_seed`` -> bit-identical jittered samples."""
        T = _make_non_trivial_pose(dtype=torch.float32)
        torch.manual_seed(42)
        out1 = _sample_jittered_world_cam_poses(T, n_samples=8, translation_bound_m=0.3, rotation_bound_rad=0.05)
        torch.manual_seed(42)
        out2 = _sample_jittered_world_cam_poses(T, n_samples=8, translation_bound_m=0.3, rotation_bound_rad=0.05)
        assert torch.equal(out1, out2)
        # And a different seed actually shifts the jittered rows (row 0 is always ==).
        torch.manual_seed(43)
        out3 = _sample_jittered_world_cam_poses(T, n_samples=8, translation_bound_m=0.3, rotation_bound_rad=0.05)
        assert torch.equal(out1[0], out3[0])
        assert not torch.equal(out1[1:], out3[1:])

    def test_zero_envelope_collapses_to_identity_jitter(self):
        """Zero translation/rotation envelope -> every sample equals the input pose."""
        T = _make_non_trivial_pose(dtype=torch.float32)
        out = _sample_jittered_world_cam_poses(T, n_samples=8, translation_bound_m=0.0, rotation_bound_rad=0.0)
        assert out.shape == (8, 4, 4)
        torch.testing.assert_close(out, T.unsqueeze(0).expand(8, 4, 4), atol=1e-7, rtol=0.0)


@dataclass
class _FakePinholeResult:
    """Duck-typed stand-in for :class:`nurec.libs.sensors.models.common.PixelsReturn`."""

    pixels: torch.Tensor
    valid_flag: torch.Tensor


@dataclass
class _FakePinholeCamera:
    """Tiny duck-typed pinhole camera implementing ``camera_rays_to_pixels``.

    Returns rounded integer pixel coordinates so the test is robust against
    sub-ulp noise from ray normalization (the function under test floor-divides
    pixels by tile size, which would otherwise be flaky on tile boundaries).
    """

    fx: float
    fy: float
    cx: float
    cy: float

    def camera_rays_to_pixels(self, cam_rays: torch.Tensor) -> _FakePinholeResult:
        z = cam_rays[:, 2]
        valid = z > 0
        # Avoid div-by-zero on rays that the caller is supposed to filter via valid_flag.
        z_safe = torch.where(valid, z, torch.ones_like(z))
        u = self.fx * cam_rays[:, 0] / z_safe + self.cx
        v = self.fy * cam_rays[:, 1] / z_safe + self.cy
        pixels = torch.stack([u, v], dim=-1).round().to(torch.int64)
        return _FakePinholeResult(pixels=pixels, valid_flag=valid)


class TestPrecomputeCameraToLidarTileMapping:
    """Coverage for ``precompute_from_element_to_tile``.

    The synthetic fixture is a ``2 x 4`` lidar raster with 4 lidar tiles
    (each owning two adjacent (row, col) positions) and a centered pinhole
    camera at identity pose. Ray directions are chosen so each of the 8 rays
    projects to a distinct, mid-tile camera pixel, making the no-jitter mapping
    trivially hand-computable. ``fx == 100`` is large enough that the production
    rotation envelope (0.05 rad) shifts pixels by ~5 px = >1 tile, so additional
    jittered pose samples observably grow the OR-union mapping.
    """

    @staticmethod
    def _build_setup() -> dict:
        """Return kwargs for ``precompute_from_element_to_tile``.

        Lidar layout (``n_bins_azimuth = 4``, ``n_bins_elevation = 1``)::

            position (row, col)  ->  cam pixel  ->  cam tile (col, row)  -> linear
            (0, 0)  dir=( 0.00,  0.00, 1)  -> (10, 10) -> (col=2, row=2) -> 12
            (0, 1)  dir=( 0.04,  0.00, 1)  -> (14, 10) -> (col=3, row=2) -> 13
            (0, 2)  dir=(-0.04,  0.00, 1)  -> ( 6, 10) -> (col=1, row=2) -> 11
            (0, 3)  dir=( 0.00,  0.04, 1)  -> (10, 14) -> (col=2, row=3) -> 17
            (1, 0)  dir=( 0.00, -0.04, 1)  -> (10,  6) -> (col=2, row=1) ->  7
            (1, 1)  dir=( 0.04,  0.04, 1)  -> (14, 14) -> (col=3, row=3) -> 18
            (1, 2)  dir=( 0.04, -0.04, 1)  -> (14,  6) -> (col=3, row=1) ->  8
            (1, 3)  dir=(-0.04,  0.04, 1)  -> ( 6, 14) -> (col=1, row=3) -> 16

        Lidar tile ownership (each tile owns two raster positions)::

            tile 0: (0, 0), (0, 1)  ->  cam tiles {12, 13}
            tile 1: (0, 2), (0, 3)  ->  cam tiles {11, 17}
            tile 2: (1, 0), (1, 1)  ->  cam tiles { 7, 18}
            tile 3: (1, 2), (1, 3)  ->  cam tiles { 8, 16}
        """
        directions = torch.tensor(
            [
                [
                    [0.00, 0.00, 1.0],
                    [0.04, 0.00, 1.0],
                    [-0.04, 0.00, 1.0],
                    [0.00, 0.04, 1.0],
                ],
                [
                    [0.00, -0.04, 1.0],
                    [0.04, 0.04, 1.0],
                    [0.04, -0.04, 1.0],
                    [-0.04, 0.04, 1.0],
                ],
            ],
            dtype=torch.float32,
        )
        directions = directions / directions.norm(dim=-1, keepdim=True)

        H, W = 2, 4
        lidar_rays_world = torch.zeros(H, W, 6, dtype=torch.float32)
        # Origins stay at the world origin; only the direction half is populated.
        lidar_rays_world[..., 3:6] = directions

        camera = _FakePinholeCamera(fx=100.0, fy=100.0, cx=10.0, cy=10.0)

        # 4 lidar tiles, two adjacent raster positions each; every position is assigned.
        element_to_tile = torch.tensor([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=torch.int64)

        return dict(
            camera_model=camera,
            lidar_rays_world=lidar_rays_world,
            T_cam_world=torch.eye(4, dtype=torch.float32),
            element_to_tile=element_to_tile,
            n_bins_azimuth=4,
            n_bins_elevation=1,
            cam_tile_width=4,
            cam_tile_height=4,
            cam_frame_width=20,
            cam_frame_height=20,
        )

    def test_no_jitter_mapping_matches_hand_computed_reference(self):
        """``n_pose_samples == 1`` produces exactly the hand-computed adjacency."""
        setup = self._build_setup()
        out = precompute_from_element_to_tile(
            **setup,
            depth_shells=(10.0,),
            ray_subsample=1,
            n_pose_samples=1,
        )

        # Layout sanity.
        assert out.n_lidar_tiles == 4
        assert out.n_camera_tiles == 5 * 5
        assert out.cam_tiles_y == 5 and out.cam_tiles_x == 5
        assert out.n_bins_azimuth == 4 and out.n_bins_elevation == 1

        # The caller-supplied per-element tile assignment is carried through unchanged.
        # Every (row, col) in the raster is owned by exactly one tile in this fixture.
        assert out.element_to_tile.shape == (2, 4)
        expected_e2t = torch.tensor(
            [[0, 0, 1, 1], [2, 2, 3, 3]],
            dtype=torch.int64,
        )
        assert torch.equal(out.element_to_tile, expected_e2t)

        # Hand-computed (lidar_tile, cam_tile_linear) entries from the docstring.
        expected_pairs = {
            (0, 12),
            (0, 13),
            (1, 11),
            (1, 17),
            (2, 7),
            (2, 18),
            (3, 8),
            (3, 16),
        }
        actual_pairs = {(int(r.item()), int(c.item())) for r, c in out.mapping.nonzero()}
        assert actual_pairs == expected_pairs, (
            f"no-jitter mapping mismatch: got {actual_pairs}, expected {expected_pairs}"
        )
        assert out.mapping.sum().item() == len(expected_pairs)

    def test_depth_shells_invariant_for_origin_rays(self):
        """Rays anchored at the world origin project to the same pixel at every depth.

        Adding extra depth shells therefore must not change the resulting mapping —
        a useful invariant since a too-tight or mis-ordered depth shell list would
        otherwise silently degrade coverage for rays that aren't depth-varying.
        """
        setup = self._build_setup()
        single_depth = precompute_from_element_to_tile(
            **setup,
            depth_shells=(10.0,),
            ray_subsample=1,
            n_pose_samples=1,
        )
        many_depths = precompute_from_element_to_tile(
            **setup,
            depth_shells=(5.0, 15.0, 30.0, 60.0, 100.0, 200.0),
            ray_subsample=1,
            n_pose_samples=1,
        )
        assert torch.equal(single_depth.mapping, many_depths.mapping)

    def test_mapping_grows_monotonically_with_n_pose_samples(self):
        """Increasing ``n_pose_samples`` only ever adds (lidar, cam) pairs.

        For any ``n_pose_samples > 1``, sample 0 is the un-jittered input pose and
        samples 1..n-1 are independent jittered draws; the mapping is the OR-union
        of every (lidar_tile, cam_tile) hit, so the mapping at ``n>1`` is element-wise
        ``>=`` the no-jitter mapping. With the production envelope (0.3 m / 0.05 rad)
        and our ``fx = 100`` pinhole, the 0.05 rad rotation shifts pixels by ~5 px
        (>1 tile), so 64 jittered samples observably grow the mapping vs. baseline.
        """
        setup = self._build_setup()
        no_jitter = precompute_from_element_to_tile(
            **setup,
            depth_shells=(10.0,),
            ray_subsample=1,
            n_pose_samples=1,
        )
        no_jitter_count = int(no_jitter.mapping.sum().item())

        counts_by_n = {}
        for n in (8, 32, 64):
            torch.manual_seed(0)
            out = precompute_from_element_to_tile(
                **setup,
                depth_shells=(10.0,),
                ray_subsample=1,
                n_pose_samples=n,
                pose_jitter_translation_m=0.3,
                pose_jitter_rotation_rad=0.05,
            )
            counts_by_n[n] = int(out.mapping.sum().item())
            # OR-superset guarantee: jittered mapping never drops a no-jitter pair.
            assert (out.mapping >= no_jitter.mapping).all(), (
                f"n_pose_samples={n} dropped (lidar, cam) pairs present in the no-jitter mapping"
            )

        # The full envelope at 64 samples should observably grow coverage.
        assert counts_by_n[64] > no_jitter_count, (
            f"64 jittered samples did not grow the mapping over the no-jitter baseline "
            f"({counts_by_n[64]} vs {no_jitter_count})"
        )

    def test_mapping_unchanged_when_jitter_envelope_is_zero(self):
        """Zero jitter envelopes degenerate to the no-jitter mapping regardless of ``n_pose_samples``."""
        setup = self._build_setup()
        no_jitter = precompute_from_element_to_tile(
            **setup,
            depth_shells=(10.0,),
            ray_subsample=1,
            n_pose_samples=1,
        )
        torch.manual_seed(0)
        out = precompute_from_element_to_tile(
            **setup,
            depth_shells=(10.0,),
            ray_subsample=1,
            n_pose_samples=64,
            pose_jitter_translation_m=0.0,
            pose_jitter_rotation_rad=0.0,
        )
        assert torch.equal(out.mapping, no_jitter.mapping)

    def test_envelope_knobs_independently_grow_mapping(self):
        """Both ``pose_jitter_translation_m`` and ``pose_jitter_rotation_rad`` independently grow coverage.

        Catches:
        - An implementation that ignores either kwarg (e.g., hardcoded constants
          inside ``_sample_jittered_world_cam_poses``).
        - An implementation that swaps the two kwargs: a translation 'value' of
          0.05 m gives a sub-pixel projection shift (``fx * 0.05 / depth ≈ 0.5 px``)
          that cannot cross a 4 px tile boundary in this fixture, so the rotation
          knob check fails under a swap.
        """
        setup = self._build_setup()

        def _coverage_with(t_bound: float, r_bound: float) -> int:
            torch.manual_seed(0)
            out = precompute_from_element_to_tile(
                **setup,
                depth_shells=(10.0,),
                ray_subsample=1,
                n_pose_samples=64,
                pose_jitter_translation_m=t_bound,
                pose_jitter_rotation_rad=r_bound,
            )
            return int(out.mapping.sum().item())

        no_jitter = _coverage_with(0.0, 0.0)
        trans_only = _coverage_with(0.5, 0.0)
        rot_only = _coverage_with(0.0, 0.05)

        assert trans_only > no_jitter, (
            f"pose_jitter_translation_m=0.5 had no effect on coverage ({trans_only} vs {no_jitter})"
        )
        assert rot_only > no_jitter, (
            f"pose_jitter_rotation_rad=0.05 had no effect on coverage ({rot_only} vs {no_jitter})"
        )

    def test_depth_shells_affect_coverage_for_non_origin_rays(self):
        """Depth shell coverage matters for rays whose pixel projection depends on depth.

        The rays in :meth:`_build_setup` are anchored at the world origin, so depth
        does not change their projection (covered by
        :meth:`test_depth_shells_invariant_for_origin_rays`). For a ray with a
        non-zero origin under the same identity pose, the pixel
        ``(fx / depth + cx, cy)`` is depth-dependent, and a too-tight depth shell
        list would silently miss the camera tiles those depths land in.

        For ray ``origin = (1, 0, 0)``, ``direction = (0, 0, 1)`` and the same
        pinhole camera as :meth:`_build_setup` (``fx=100``, ``cx=cy=10``,
        frame ``20x20``, tile ``4``)::

            depth=100   -> pixel=(11, 10) -> tile (col=2, row=2) -> linear=12
            depth= 25   -> pixel=(14, 10) -> tile (col=3, row=2) -> linear=13
            depth= 12.5 -> pixel=(18, 10) -> tile (col=4, row=2) -> linear=14
        """
        H, W = 1, 1
        lidar_rays_world = torch.zeros(H, W, 6, dtype=torch.float32)
        lidar_rays_world[0, 0, :3] = torch.tensor([1.0, 0.0, 0.0])
        lidar_rays_world[0, 0, 3:6] = torch.tensor([0.0, 0.0, 1.0])

        common = dict(
            camera_model=_FakePinholeCamera(fx=100.0, fy=100.0, cx=10.0, cy=10.0),
            lidar_rays_world=lidar_rays_world,
            T_cam_world=torch.eye(4, dtype=torch.float32),
            element_to_tile=torch.zeros((H, W), dtype=torch.int64),
            n_bins_azimuth=1,
            n_bins_elevation=1,
            cam_tile_width=4,
            cam_tile_height=4,
            cam_frame_width=20,
            cam_frame_height=20,
            ray_subsample=1,
            n_pose_samples=1,
        )

        def _pairs(out):
            return {(int(r.item()), int(c.item())) for r, c in out.mapping.nonzero()}

        far_only = precompute_from_element_to_tile(**common, depth_shells=(100.0,))
        far_mid = precompute_from_element_to_tile(**common, depth_shells=(100.0, 25.0))
        far_mid_near = precompute_from_element_to_tile(**common, depth_shells=(100.0, 25.0, 12.5))

        assert _pairs(far_only) == {(0, 12)}, f"far-only: got {_pairs(far_only)}"
        assert _pairs(far_mid) == {(0, 12), (0, 13)}, f"far+mid: got {_pairs(far_mid)}"
        assert _pairs(far_mid_near) == {(0, 12), (0, 13), (0, 14)}, f"all shells: got {_pairs(far_mid_near)}"


class TestPrecomputeLidarTileGridShape:
    """``lidar_tile_grid_shape`` is the only record of the lidar tile layout the mapping was
    built in, and it is the reshape target for the flat lidar tile mask.

    Getting it wrong is invisible to every shape check (``n_az * n_el == n_el * n_az``), so
    the elevation-major default and the explicit override are pinned here.
    """

    def test_defaults_to_elevation_major(self):
        """gsplat tile id = elevation_idx * n_bins_azimuth + azimuth_idx -> (n_el, n_az)."""
        setup = TestPrecomputeCameraToLidarTileMapping._build_setup()
        assert setup["n_bins_azimuth"] != setup["n_bins_elevation"], "fixture must have asymmetric bin counts"
        out = precompute_from_element_to_tile(**setup, depth_shells=(10.0,), ray_subsample=1, n_pose_samples=1)
        assert out.lidar_tile_grid_shape == (setup["n_bins_elevation"], setup["n_bins_azimuth"])

    def test_explicit_lidar_tile_grid_shape_overrides_the_default(self):
        """An explicit ``lidar_tile_grid_shape`` wins over the elevation-major default."""
        setup = TestPrecomputeCameraToLidarTileMapping._build_setup()
        override = (2, 2)
        out = precompute_from_element_to_tile(
            **setup,
            depth_shells=(10.0,),
            ray_subsample=1,
            n_pose_samples=1,
            lidar_tile_grid_shape=override,
        )
        assert out.lidar_tile_grid_shape == override


class TestPrecomputeRunsWithoutAutograd:
    """The precompute produces a bool adjacency matrix; it must never build an autograd graph.

    Before the fix, a grad-requiring ``T_cam_world`` made every projection pass save tensors
    for backward, so the retained pack count scaled linearly with ``n_pose_samples`` (1->68,
    8->531, 64->4227 on the review fixture) for a result that carries no gradient at all.
    """

    @staticmethod
    def _run_with_pack_counter(n_pose_samples: int) -> tuple:
        setup = TestPrecomputeCameraToLidarTileMapping._build_setup()
        setup["T_cam_world"] = torch.eye(4, dtype=torch.float32).requires_grad_(True)

        packed: list[int] = []

        def _pack(tensor: torch.Tensor) -> torch.Tensor:
            packed.append(1)
            return tensor

        def _unpack(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        shared: dict[str, Any] = dict(depth_shells=(5.0, 15.0, 30.0), ray_subsample=1, n_pose_samples=n_pose_samples)
        torch.manual_seed(0)
        with torch.autograd.graph.saved_tensors_hooks(_pack, _unpack):
            out = precompute_from_element_to_tile(**setup, **shared)
        return out, len(packed)

    @pytest.mark.parametrize("n_pose_samples", [1, 8])
    def test_no_saved_tensors_with_grad_requiring_pose(self, n_pose_samples: int):
        """Zero saved-tensor packs, at any ``n_pose_samples``."""
        out, n_packed = self._run_with_pack_counter(n_pose_samples)
        assert n_packed == 0, (
            f"precompute retained {n_packed} autograd saved-tensor packs at "
            f"n_pose_samples={n_pose_samples}; the precompute must run under no_grad"
        )
        assert not out.mapping.requires_grad
        assert out.mapping.grad_fn is None
        assert not out.element_to_tile.requires_grad

    def test_pack_count_is_flat_in_n_pose_samples(self):
        """Saved-tensor pack count must not scale with n_pose_samples."""
        _, packs_1 = self._run_with_pack_counter(1)
        _, packs_8 = self._run_with_pack_counter(8)
        assert packs_1 == packs_8 == 0

    def test_result_is_still_correct_under_grad_requiring_pose(self):
        """no_grad must not change the numerics: same mapping as the detached-pose run."""
        setup = TestPrecomputeCameraToLidarTileMapping._build_setup()
        shared: dict[str, Any] = dict(depth_shells=(10.0,), ray_subsample=1, n_pose_samples=1)
        detached = precompute_from_element_to_tile(**setup, **shared)

        grad_setup = TestPrecomputeCameraToLidarTileMapping._build_setup()
        grad_setup["T_cam_world"] = grad_setup["T_cam_world"].clone().requires_grad_(True)
        with_grad = precompute_from_element_to_tile(**grad_setup, **shared)

        assert torch.equal(detached.mapping, with_grad.mapping)


def _reference_active_tile_mask_via_maxpool(ray_flags: torch.Tensor, tile_width: int, tile_height: int) -> torch.Tensor:
    """max_pool2d formulation of the tile reduction: bool -> float32 -> ``max_pool2d`` -> bool,
    plus a trailing slice.

    Kept only as a differential-testing oracle for
    :class:`TestComputeActiveTileMaskMatchesMaxPoolReference`.
    """
    if ray_flags.ndim == 4:
        ray_flags = ray_flags.squeeze(-1)
    elif ray_flags.ndim == 2:
        ray_flags = ray_flags.unsqueeze(0)
    _, h, w = ray_flags.shape

    invalid_bits = torch.bitwise_and(ray_flags, RayFlags.INVALID.value)
    valid_mask = (invalid_bits == 0).to(dtype=torch.float32)

    pad_h = (tile_height - h % tile_height) % tile_height
    pad_w = (tile_width - w % tile_width) % tile_width
    if pad_h > 0 or pad_w > 0:
        valid_mask = torch.nn.functional.pad(valid_mask, (0, pad_w, 0, pad_h), value=0.0)

    pooled = torch.nn.functional.max_pool2d(
        valid_mask.unsqueeze(1),
        kernel_size=(tile_height, tile_width),
        stride=(tile_height, tile_width),
    )
    tiles_y = math.ceil(h / tile_height)
    tiles_x = math.ceil(w / tile_width)
    return pooled.squeeze(1)[:, :tiles_y, :tiles_x].bool()


class TestComputeActiveTileMaskMatchesMaxPoolReference:
    """Differential test: the bool ``any`` reduction must be bitwise-identical to the
    float32 ``max_pool2d`` round trip it replaced, including on non-divisible frame sizes where
    the (now-removed) trailing slice used to run.
    """

    @pytest.mark.parametrize(
        "b, h, w, tile_h, tile_w",
        [
            (1, 16, 16, 16, 16),  # exact single tile
            (1, 17, 33, 16, 16),  # both axes non-divisible
            (3, 48, 64, 16, 16),  # batched, exact
            (2, 45, 61, 8, 4),  # batched, non-square tiles, non-divisible
            (1, 1, 1, 16, 16),  # frame smaller than one tile
            (4, 7, 9, 3, 5),  # odd everything
            (1, 100, 100, 1, 1),  # degenerate 1x1 tiles
        ],
    )
    def test_matches_reference_on_random_flags(self, b: int, h: int, w: int, tile_h: int, tile_w: int):
        torch.manual_seed(b * 1000 + h * 100 + w)
        for density in (0.0, 0.05, 0.5, 0.95, 1.0):
            invalid = torch.rand(b, h, w) < density
            flags = torch.full((b, h, w), RayFlags.RGB_LABEL.value, dtype=torch.int32)
            flags[invalid] |= RayFlags.INVALID.value

            new = compute_active_tile_mask(flags, tile_width=tile_w, tile_height=tile_h)
            old = _reference_active_tile_mask_via_maxpool(flags, tile_width=tile_w, tile_height=tile_h)

            assert new.shape == old.shape, f"shape mismatch at density={density}: {new.shape} vs {old.shape}"
            assert new.dtype == torch.bool
            assert torch.equal(new, old), f"mask mismatch at density={density} for {(b, h, w, tile_h, tile_w)}"

    def test_matches_reference_on_2d_and_4d_inputs(self):
        torch.manual_seed(7)
        invalid = torch.rand(23, 31) < 0.4
        flags_2d = torch.full((23, 31), RayFlags.RGB_LABEL.value, dtype=torch.int32)
        flags_2d[invalid] |= RayFlags.INVALID.value

        assert torch.equal(
            compute_active_tile_mask(flags_2d, tile_width=8, tile_height=8),
            _reference_active_tile_mask_via_maxpool(flags_2d, tile_width=8, tile_height=8),
        )

        flags_4d = flags_2d.unsqueeze(0).unsqueeze(-1)  # (1, 23, 31, 1)
        assert torch.equal(
            compute_active_tile_mask(flags_4d, tile_width=8, tile_height=8),
            _reference_active_tile_mask_via_maxpool(flags_4d, tile_width=8, tile_height=8),
        )
