# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from nurec.utils.geometry import se3_matrix_inverse
from nurec.utils.types import RayFlags


if TYPE_CHECKING:
    from ncore.sensors import CameraModel
    from nurec.utils.batch import LidarFrameLabels

log = logging.getLogger(__name__)


def compute_active_tile_mask(
    ray_flags: torch.Tensor,
    tile_width: int = 16,
    tile_height: int = 16,
) -> torch.Tensor:
    """Compute a boolean per-frame mask indicating which tiles contain at least one valid pixel.

    A tile is "active" if any pixel within it does NOT have the RayFlags.INVALID bit set.
    The returned mask can be handed to the renderer to skip projection, sorting, and
    rendering for entirely-invalid tiles.

    Args:
        ray_flags: Per-pixel flag bitmask. Shape ``(H, W)``, ``(B, H, W)``, or ``(B, H, W, 1)``.
                   dtype ``torch.int32``.
        tile_width: Width of a tile in pixels (must match the renderer's camera tile config).
        tile_height: Height of a tile in pixels (must match the renderer's camera tile config).

    Returns:
        Bool tensor of shape ``(B, tiles_y, tiles_x)`` on the same device as *ray_flags*,
        where ``tiles_y = ceil(H / tile_height)`` and ``tiles_x = ceil(W / tile_width)``.
        ``B`` is 1 when ray_flags was 2-D. True means the tile has at least one valid pixel
        in the corresponding frame.
    """
    if ray_flags.dtype != torch.int32:
        raise TypeError(f"ray_flags must be int32, got {ray_flags.dtype}")
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError(f"tile_width and tile_height must be positive, got {tile_width}x{tile_height}")

    # Normalize to (B, H, W).
    if ray_flags.ndim == 4:
        if ray_flags.shape[-1] != 1:
            raise ValueError(f"Expected trailing channel dim of size 1, got shape {tuple(ray_flags.shape)}")
        ray_flags = ray_flags.squeeze(-1)
    elif ray_flags.ndim == 2:
        ray_flags = ray_flags.unsqueeze(0)
    if ray_flags.ndim != 3:
        raise ValueError(f"Expected (B, H, W) ray_flags, got shape {ray_flags.shape}")

    b, h, w = ray_flags.shape

    # valid: True where the INVALID bit is NOT set.
    invalid_bits = torch.bitwise_and(ray_flags, RayFlags.INVALID.value)
    valid = invalid_bits == 0  # (B, H, W) bool

    # Pad (with False) to an exact multiple of the tile size so the reshape below is a clean
    # tiling of the frame. After padding, H/W are exact multiples, so tiles_y/tiles_x computed
    # from the padded extents are exactly ceil(h / tile_height) / ceil(w / tile_width).
    pad_h = (tile_height - h % tile_height) % tile_height
    pad_w = (tile_width - w % tile_width) % tile_width
    if pad_h > 0 or pad_w > 0:
        valid = F.pad(valid, (0, pad_w, 0, pad_h), value=False)

    tiles_y = (h + pad_h) // tile_height
    tiles_x = (w + pad_w) // tile_width

    # Reduce each tile with a pure-bool `any`. Reshaping to
    # (B, tiles_y, tile_height, tiles_x, tile_width) puts the two intra-tile axes at dims 2
    # and 4; `any` takes a single dim, so reduce the inner (width) axis first and then the
    # height axis. This stays in bool end-to-end — no float32 round trip and no max_pool2d
    # temporary (which would be needed only because max_pool2d has no bool kernel).
    tiled = valid.reshape(b, tiles_y, tile_height, tiles_x, tile_width)
    return tiled.any(dim=4).any(dim=2)  # (B, tiles_y, tiles_x)


@dataclass
class CameraToLidarTileMapping:
    """Precomputed mapping from camera tiles to lidar tiles.

    Stores a boolean matrix of shape ``(n_lidar_tiles, n_camera_tiles)`` where
    entry ``[i, j]`` is True if lidar tile ``i`` overlaps with camera tile ``j``
    at any of the sampled depth shells.

    **Lidar tile index space.** The flat lidar tile index ``i`` means whatever
    ``element_to_tile`` says it means — it is the consuming renderer's own tile numbering,
    carried through unchanged. This class does not impose a layout; ``lidar_tile_grid_shape``
    records the 2-D shape that flat space reshapes to. In tree that is always gsplat's
    elevation-major grid (see that field).

    Use :func:`camera_tile_mask_to_lidar_mask` at runtime for a fast lookup.
    """

    mapping: torch.Tensor  # (n_lidar_tiles, n_camera_tiles) bool
    n_lidar_tiles: int
    n_camera_tiles: int
    cam_tiles_y: int
    cam_tiles_x: int
    # Bin counts only. These do NOT by themselves fix the flat lidar tile index layout —
    # that is defined by ``element_to_tile`` and described by ``lidar_tile_grid_shape``.
    n_bins_azimuth: int
    n_bins_elevation: int
    # (dim0, dim1) to reshape the flat lidar tile mask into so it matches the consuming
    # renderer's tile grid. The lidar layout is gsplat's elevation-major
    # (n_bins_elevation, n_bins_azimuth), matching its tile id = el * n_bins_azimuth + az;
    # that is what `precompute_from_element_to_tile` records here unless a caller passes an
    # explicit override.
    lidar_tile_grid_shape: tuple[int, int]
    # Per-element lidar tile assignment: (n_rows, n_cols) int64, -1 for unassigned
    element_to_tile: torch.Tensor


def _sample_jittered_world_cam_poses(
    T_world_cam: torch.Tensor,
    n_samples: int,
    translation_bound_m: float,
    rotation_bound_rad: float,
) -> torch.Tensor:
    """Return a ``(n_samples, 4, 4)`` stack of jittered world→camera transforms.

    Behavior by ``n_samples``:

    - ``n_samples < 0`` — raises ``ValueError``.
    - ``n_samples == 0`` — returns shape ``(0, 4, 4)`` (empty stack).
    - ``n_samples == 1`` — returns shape ``(1, 4, 4)`` equal to
      ``T_world_cam.unsqueeze(0)``.
    - ``n_samples > 1`` — returns shape ``(n_samples, 4, 4)`` where row 0 is
      ``T_world_cam`` unchanged and rows ``1..n-1`` apply independent SE(3)
      perturbations **in the camera-local frame** (uniform translation in the
      box ``±translation_bound_m`` along each camera axis; uniform axis-angle
      rotation with per-axis angle bounded by ``±rotation_bound_rad``).

    Purpose: the mapping is built once and reused across frames. Between
    frames, ego motion (translation + rotation over the ~few-ms gap between
    camera frame-start time and lidar pixel emission time) perturbs the
    effective camera→lidar rigid transform by ~10–30 cm of translation and up
    to ~50 mrad per axis of rotation with the default bound. The axis-angle
    magnitude can reach ~87 mrad because each component is sampled
    independently and uniformly in ``[-rotation_bound_rad, +rotation_bound_rad]``.
    OR-ing all sampled tile pairs into the mapping yields an envelope-covering
    cache that stays accurate across frames without needing to rebuild.
    """
    device = T_world_cam.device
    dtype = T_world_cam.dtype
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0; got {n_samples}")
    if n_samples <= 1:
        return T_world_cam.unsqueeze(0)[:n_samples]

    n_extra = n_samples - 1
    trans = (torch.rand(n_extra, 3, device=device, dtype=dtype) * 2 - 1) * translation_bound_m
    rvec = (torch.rand(n_extra, 3, device=device, dtype=dtype) * 2 - 1) * rotation_bound_rad

    # Rodrigues via matrix exponential of the skew-symmetric cross-product matrix.
    zeros = torch.zeros(n_extra, device=device, dtype=dtype)
    skew = torch.stack(
        [
            torch.stack([zeros, -rvec[:, 2], rvec[:, 1]], dim=-1),
            torch.stack([rvec[:, 2], zeros, -rvec[:, 0]], dim=-1),
            torch.stack([-rvec[:, 1], rvec[:, 0], zeros], dim=-1),
        ],
        dim=1,
    )  # (n_extra, 3, 3)
    R_delta = torch.linalg.matrix_exp(skew)

    D = torch.eye(4, device=device, dtype=dtype).expand(n_extra, 4, 4).clone()
    D[:, :3, :3] = R_delta
    D[:, :3, 3] = trans

    # Jitter in the camera-local frame corresponds to T_world_cam_new = D^-1 @ T_world_cam.
    # unbatch=False keeps the (n_extra, 4, 4) shape even when n_extra == 1.
    D_inv = se3_matrix_inverse(D, unbatch=False)
    jittered = D_inv @ T_world_cam.unsqueeze(0)  # (n_extra, 4, 4)
    return torch.cat([T_world_cam.unsqueeze(0), jittered], dim=0)


# Historical hard-coded precompute defaults, applied by `precompute_from_element_to_tile`;
# `nurec.config.model.CameraToLidarTileMappingConfig` repeats the same values as its field
# defaults.
DEFAULT_DEPTH_SHELLS: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0, 100.0, 200.0)
DEFAULT_RAY_SUBSAMPLE = 2
DEFAULT_N_POSE_SAMPLES = 64
DEFAULT_POSE_JITTER_TRANSLATION_M = 0.3
DEFAULT_POSE_JITTER_ROTATION_RAD = 0.05


def precompute_from_element_to_tile(
    camera_model: CameraModel,
    lidar_rays_world: torch.Tensor,
    T_cam_world: torch.Tensor,
    cam_tile_width: int,
    cam_tile_height: int,
    cam_frame_width: int,
    cam_frame_height: int,
    element_to_tile: torch.Tensor,
    n_bins_azimuth: int,
    n_bins_elevation: int,
    lidar_tile_grid_shape: Optional[tuple[int, int]] = None,
    depth_shells: tuple[float, ...] = DEFAULT_DEPTH_SHELLS,
    ray_subsample: int = DEFAULT_RAY_SUBSAMPLE,
    n_pose_samples: int = DEFAULT_N_POSE_SAMPLES,
    pose_jitter_translation_m: float = DEFAULT_POSE_JITTER_TRANSLATION_M,
    pose_jitter_rotation_rad: float = DEFAULT_POSE_JITTER_ROTATION_RAD,
) -> CameraToLidarTileMapping:
    """Build a camera→lidar tile mapping from a **caller-supplied per-element tile map**.

    Used by the gsplat path, where the consuming renderer already knows its own lidar tile
    assignment (histogram-equalized azimuth bins) and hands it over directly.

    Layout: gsplat lidar tile ids are **elevation-major** — linear id
    ``elevation_idx * n_bins_azimuth + azimuth_idx`` — so the flat lidar tile mask reshapes to
    ``(n_bins_elevation, n_bins_azimuth)``. That is the default for ``lidar_tile_grid_shape``
    here; see :attr:`CameraToLidarTileMapping.lidar_tile_grid_shape`.

    Args:
        camera_model: ncore CameraModel for the camera sensor.
        lidar_rays_world: World-space lidar rays ``(n_rows, n_cols, 6)``, last dim
            ``(origin_xyz, direction_xyz)``.
        T_cam_world: ``(4, 4)`` camera-sensor-to-world transform.
        cam_tile_width: Camera tile width in pixels.
        cam_tile_height: Camera tile height in pixels.
        cam_frame_width: Camera frame width in pixels.
        cam_frame_height: Camera frame height in pixels.
        element_to_tile: ``(n_rows, n_cols)`` int64 per-lidar-pixel tile id in the consuming
            renderer's own flat tile-index space, ``-1`` for unassigned pixels.
        n_bins_azimuth: Number of azimuth bins in that tile space.
        n_bins_elevation: Number of elevation bins in that tile space.
        lidar_tile_grid_shape: Override for the reshape target. Defaults to the
            elevation-major ``(n_bins_elevation, n_bins_azimuth)``.
        depth_shells: Distances (metres) along each ray at which to sample.
        ray_subsample: Stride applied to both lidar ray axes before projection.
        n_pose_samples: Number of camera poses (including the unjittered input pose) OR-ed
            into the mapping. 1 disables jitter entirely.
        pose_jitter_translation_m: Per-axis uniform translation bound (metres, camera-local).
        pose_jitter_rotation_rad: Per-axis uniform rotation bound (radians, camera-local).

    Returns:
        A :class:`CameraToLidarTileMapping` for :func:`camera_tile_mask_to_lidar_mask`.
    """
    return _precompute(
        camera_model=camera_model,
        lidar_rays_world=lidar_rays_world,
        T_cam_world=T_cam_world,
        cam_tile_width=cam_tile_width,
        cam_tile_height=cam_tile_height,
        cam_frame_width=cam_frame_width,
        cam_frame_height=cam_frame_height,
        element_to_tile=element_to_tile,
        n_bins_azimuth=n_bins_azimuth,
        n_bins_elevation=n_bins_elevation,
        # gsplat tile id = elevation_idx * n_bins_azimuth + azimuth_idx → elevation-major.
        lidar_tile_grid_shape=(
            lidar_tile_grid_shape if lidar_tile_grid_shape is not None else (n_bins_elevation, n_bins_azimuth)
        ),
        depth_shells=depth_shells,
        ray_subsample=ray_subsample,
        n_pose_samples=n_pose_samples,
        pose_jitter_translation_m=pose_jitter_translation_m,
        pose_jitter_rotation_rad=pose_jitter_rotation_rad,
    )


@torch.no_grad()
def _precompute(
    camera_model: CameraModel,
    lidar_rays_world: torch.Tensor,
    T_cam_world: torch.Tensor,
    cam_tile_width: int,
    cam_tile_height: int,
    cam_frame_width: int,
    cam_frame_height: int,
    element_to_tile: torch.Tensor,
    n_bins_azimuth: int,
    n_bins_elevation: int,
    lidar_tile_grid_shape: tuple[int, int],
    depth_shells: tuple[float, ...],
    ray_subsample: int,
    n_pose_samples: int,
    pose_jitter_translation_m: float,
    pose_jitter_rotation_rad: float,
) -> CameraToLidarTileMapping:
    """Projection worker behind :func:`precompute_from_element_to_tile`.

    Uses actual lidar ray directions from the rendering data to avoid any lidar
    coordinate-frame convention issues. Each lidar ray's tile assignment is already known
    (``element_to_tile``); the ray is projected at several depths into the camera and the
    camera tile it lands in is OR-ed into the mapping.

    To make the cached mapping robust against ego-motion-induced staleness (the camera pose
    at frame-start differs slightly from the pose at each lidar pixel's emission time, and
    this residual varies frame-to-frame), ``n_pose_samples`` jittered copies of the camera
    pose are sampled and every resulting (lidar_tile, camera_tile) hit is OR-ed in.

    The whole body runs under ``torch.no_grad()``: the result is a bool adjacency matrix, but
    without it autograd retains a saved-tensor pack per projection pass, scaling linearly in
    ``n_pose_samples`` for no benefit.

    Callers must have already resolved the lidar tile layout: ``element_to_tile`` and
    ``lidar_tile_grid_shape`` are taken as given, never defaulted here. That choice belongs to
    the entry point, which documents the elevation-major layout it fills in.
    """
    device = lidar_rays_world.device
    n_rows, n_cols = lidar_rays_world.shape[:2]
    n_lidar_tiles = n_bins_azimuth * n_bins_elevation
    element_to_tile = element_to_tile.to(device)

    cam_tiles_x = math.ceil(cam_frame_width / cam_tile_width)
    cam_tiles_y = math.ceil(cam_frame_height / cam_tile_height)
    n_cam_tiles = cam_tiles_y * cam_tiles_x

    # --- Subsample rays for efficiency ---
    row_idx = torch.arange(0, n_rows, ray_subsample, device=device)
    col_idx = torch.arange(0, n_cols, ray_subsample, device=device)
    grid_r, grid_c = torch.meshgrid(row_idx, col_idx, indexing="ij")
    grid_r, grid_c = grid_r.reshape(-1), grid_c.reshape(-1)

    tile_ids = element_to_tile[grid_r, grid_c]  # (M,)
    valid_rays = tile_ids >= 0
    grid_r, grid_c, tile_ids = grid_r[valid_rays], grid_c[valid_rays], tile_ids[valid_rays]

    origins = lidar_rays_world[grid_r, grid_c, :3]  # (M, 3)
    directions = lidar_rays_world[grid_r, grid_c, 3:]  # (M, 3)

    # --- Sample jittered world→camera poses ---
    T_world_cam = se3_matrix_inverse(T_cam_world.to(device))  # camera-to-world → world-to-camera
    T_world_cam_samples = _sample_jittered_world_cam_poses(
        T_world_cam,
        n_samples=max(1, n_pose_samples),
        translation_bound_m=pose_jitter_translation_m,
        rotation_bound_rad=pose_jitter_rotation_rad,
    )  # (n_pose_samples, 4, 4)

    n_poses = int(T_world_cam_samples.shape[0])
    log.info(
        f"[tile_mask] Building camera→lidar tile mapping: "
        f"lidar raster {n_rows}x{n_cols} → {n_lidar_tiles} tiles "
        f"({n_bins_azimuth} az x {n_bins_elevation} el, grid shape {tuple(lidar_tile_grid_shape)}), "
        f"camera {cam_frame_height}x{cam_frame_width}px → {cam_tiles_y}x{cam_tiles_x} tiles "
        f"@ {cam_tile_height}x{cam_tile_width}px; "
        f"{int(tile_ids.numel())} subsampled rays (stride={ray_subsample}), "
        f"{n_poses} poses x {len(depth_shells)} depth shells = "
        f"{n_poses * len(depth_shells)} projection passes"
    )

    mapping = torch.zeros(n_lidar_tiles, n_cam_tiles, dtype=torch.bool, device=device)

    # Loop invariant: the world-space sample points depend only on the depth shell, not on the
    # camera pose, so hoist them out of the pose loop. (The projection itself stays sequential:
    # ``camera_rays_to_pixels`` takes a single (K, 3) ray batch, so batching over poses would
    # mean flattening pose x ray into one call — kept out of scope to avoid changing numerics.)
    pts_world_per_shell = [origins + directions * depth for depth in depth_shells]

    for T_wc in T_world_cam_samples:
        R_T = T_wc[:3, :3].T
        t = T_wc[:3, 3].unsqueeze(0)

        for pts_world in pts_world_per_shell:
            pts_cam = pts_world @ R_T + t  # (M, 3)

            in_front = pts_cam[:, 2] > 0
            if not in_front.any():
                continue

            cam_rays = pts_cam[in_front]
            cam_rays = cam_rays / cam_rays.norm(dim=-1, keepdim=True)

            result = camera_model.camera_rays_to_pixels(cam_rays)
            pixels = result.pixels  # (K, 2) — (x, y)
            valid = result.valid_flag  # (K,)

            px, py = pixels[:, 0], pixels[:, 1]
            in_bounds = valid & (px >= 0) & (px < cam_frame_width) & (py >= 0) & (py < cam_frame_height)

            if not in_bounds.any():
                continue

            cam_tile_col = px[in_bounds] // cam_tile_width
            cam_tile_row = py[in_bounds] // cam_tile_height
            cam_tile_linear = (cam_tile_row * cam_tiles_x + cam_tile_col).long()

            lidar_tile_linear = tile_ids[in_front][in_bounds]

            mapping[lidar_tile_linear, cam_tile_linear] = True

    n_mapped = mapping.any(dim=1).sum().item()
    log.info(
        f"[tile_mask] Precomputed camera→lidar tile mapping: "
        f"{n_mapped}/{n_lidar_tiles} lidar tiles map to camera tiles "
        f"({len(depth_shells)} depth shells, subsample={ray_subsample}, "
        f"pose_samples={n_poses}, "
        f"trans_jit={pose_jitter_translation_m:.2f}m, rot_jit={pose_jitter_rotation_rad:.3f}rad)"
    )

    return CameraToLidarTileMapping(
        mapping=mapping,
        n_lidar_tiles=n_lidar_tiles,
        n_camera_tiles=n_cam_tiles,
        cam_tiles_y=cam_tiles_y,
        cam_tiles_x=cam_tiles_x,
        n_bins_azimuth=n_bins_azimuth,
        n_bins_elevation=n_bins_elevation,
        lidar_tile_grid_shape=lidar_tile_grid_shape,
        element_to_tile=element_to_tile,
    )


def camera_tile_mask_to_lidar_mask(
    camera_active_tile_mask: torch.Tensor,
    mapping: CameraToLidarTileMapping,
) -> torch.Tensor:
    """Convert a per-frame camera active tile mask to a per-frame lidar active tile mask.

    Args:
        camera_active_tile_mask: Bool tensor of shape ``(B, cam_tiles_y, cam_tiles_x)``.
        mapping: Precomputed :class:`CameraToLidarTileMapping`.

    Returns:
        Bool tensor of shape ``(B, n_lidar_tiles)`` — a **flat** lidar tile mask indexed in
        exactly the tile-index space of ``mapping.element_to_tile``, which is gsplat's
        elevation-major lidar tile numbering (see
        :attr:`CameraToLidarTileMapping.lidar_tile_grid_shape`). To hand the mask to a
        renderer, reshape it to ``(B, *mapping.lidar_tile_grid_shape)`` — never to a
        hard-coded ``(n_bins_azimuth, n_bins_elevation)``. Because
        ``n_az * n_el == n_el * n_az``, a transposed reshape passes every shape check and
        silently addresses the wrong tiles.
    """
    if camera_active_tile_mask.ndim != 3:
        raise ValueError(
            f"Expected batched camera_active_tile_mask (B, cam_tiles_y, cam_tiles_x), "
            f"got shape {tuple(camera_active_tile_mask.shape)}"
        )
    # The flat reshape below only addresses the right camera tiles if the incoming grid is the
    # grid the mapping was built against. A different tile count would blow up far away in the
    # broadcast; a transposed grid with the same numel (e.g. 2x8 vs 8x2) would silently pass and
    # map every camera tile to the wrong lidar tiles. Both become one clear error here, which
    # also covers externally-supplied masks the composite's cache revalidation never sees.
    if tuple(camera_active_tile_mask.shape[1:]) != (mapping.cam_tiles_y, mapping.cam_tiles_x):
        raise ValueError(
            f"camera_active_tile_mask tile grid "
            f"{camera_active_tile_mask.shape[1]}x{camera_active_tile_mask.shape[2]} "
            f"(cam_tiles_y x cam_tiles_x) does not match the grid this mapping was built for, "
            f"{mapping.cam_tiles_y}x{mapping.cam_tiles_x}; got mask shape "
            f"{tuple(camera_active_tile_mask.shape)}"
        )
    B = camera_active_tile_mask.shape[0]
    cam_flat = camera_active_tile_mask.reshape(B, -1).to(mapping.mapping.device)  # (B, n_cam)
    # mapping.mapping is (n_lidar_tiles, n_camera_tiles) bool. Broadcast it across B and
    # reduce over the camera axis: a lidar tile is active in frame b iff any of its mapped
    # camera tiles is active in frame b.
    return (mapping.mapping.unsqueeze(0) & cam_flat.unsqueeze(1)).any(dim=2)  # (B, n_lidar)


def mark_inactive_lidar_tile_rays_inplace(
    labels: "LidarFrameLabels",
    active_tile_mask: torch.Tensor,
    mapping: CameraToLidarTileMapping,
) -> None:
    """Set ``RayFlags.INVALID`` on lidar rays whose tiles are inactive.

    This ensures that loss functions (which filter by ``INVALID``) correctly
    skip rays that the renderer also skipped via ``active_tile_mask``.

    Modifies ``labels.flags`` **in-place** and invalidates the lazily cached
    valid-ray counters (``_n_valid_lidar`` / ``_n_valid_raydrop``) so the loss
    normalization factors are recomputed from the freshly-stamped flags. Taking
    the whole :class:`~nurec.utils.batch.LidarFrameLabels` (rather than the raw
    flags tensor) is deliberate: it makes the cache invalidation impossible to
    forget at the call site.

    Args:
        labels: Lidar frame labels whose ``flags`` tensor is stamped in place.
            ``flags`` must be ``int32`` with shape ``(B, H, W, 1)`` or ``(H, W)``.
        active_tile_mask: Per-frame bool tensor ``(B, *mapping.lidar_tile_grid_shape)``, as
            set on ``RenderingData.active_tile_mask``. It is flattened to ``(B, -1)`` and
            indexed by ``mapping.element_to_tile``, so the only contract that matters is that
            the mask's trailing dims are laid out in **that** renderer's tile-index space —
            gsplat's elevation-major grid (see
            :attr:`CameraToLidarTileMapping.lidar_tile_grid_shape`). A transposed mask has the
            same numel and will be accepted, stamping an entirely different set of rays.
        mapping: The :class:`CameraToLidarTileMapping` that produced the mask
            (carries ``element_to_tile``).
    """
    if labels.flags is None:
        raise ValueError("labels.flags is None; cannot stamp INVALID on inactive lidar tile rays")

    _stamp_inactive_lidar_tile_rays_inplace(labels.flags, active_tile_mask, mapping)

    # The flags just changed, so any cached valid-ray count is stale. Drop it so the loss
    # normalization factors (1 / n_valid_*) see the masked counts.
    labels._n_valid_lidar = None
    labels._n_valid_raydrop = None


def _stamp_inactive_lidar_tile_rays_inplace(
    ray_flags: torch.Tensor,
    active_tile_mask: torch.Tensor,
    mapping: CameraToLidarTileMapping,
) -> None:
    """Tensor-level worker for :func:`mark_inactive_lidar_tile_rays_inplace`.

    Prefer the public function: it also invalidates the cached valid-ray counters.

    ``active_tile_mask`` must be ``(B, *mapping.lidar_tile_grid_shape)``; it is flattened and
    indexed by ``mapping.element_to_tile``, whose tile-index space is the renderer's own
    (gsplat's elevation-major grid — see
    :attr:`CameraToLidarTileMapping.lidar_tile_grid_shape`). Only the *rank* and the total
    tile count can be validated here; a transposed mask is indistinguishable by shape.
    """
    if ray_flags.dtype != torch.int32:
        raise TypeError(f"ray_flags must be int32, got {ray_flags.dtype}")
    if active_tile_mask.ndim != 3:
        raise ValueError(
            f"Expected batched active_tile_mask (B, *mapping.lidar_tile_grid_shape), "
            f"got shape {tuple(active_tile_mask.shape)}"
        )

    # Validate the per-element tile assignment against the flag raster before indexing with
    # it — a mismatch would otherwise surface as a bare torch broadcast error naming neither
    # tensor (or, worse, silently broadcast).
    if ray_flags.ndim == 4:
        flags_rows_cols = tuple(ray_flags.shape[1:3])
    elif ray_flags.ndim == 2:
        flags_rows_cols = tuple(ray_flags.shape)
    else:
        raise ValueError(f"Unexpected ray_flags shape {ray_flags.shape}")
    if tuple(mapping.element_to_tile.shape) != flags_rows_cols:
        raise ValueError(
            f"mapping.element_to_tile shape {tuple(mapping.element_to_tile.shape)} does not match the lidar "
            f"ray_flags (n_rows, n_cols) {flags_rows_cols} (full ray_flags shape {tuple(ray_flags.shape)})"
        )

    B = active_tile_mask.shape[0]
    element_to_tile = mapping.element_to_tile  # (n_rows, n_cols) int64, -1 = unassigned
    tile_mask_flat = active_tile_mask.reshape(B, -1)  # (B, n_lidar_tiles)

    assigned = element_to_tile >= 0  # (n_rows, n_cols)
    # Gather per-pixel tile-active status for every frame in one shot. Clamp to a valid
    # index for unassigned pixels; the `assigned` mask below zeroes them back out.
    safe_ids = element_to_tile.clamp(min=0)  # (n_rows, n_cols)
    per_pixel_active = tile_mask_flat[:, safe_ids]  # (B, n_rows, n_cols)
    inactive_rays = assigned.unsqueeze(0) & ~per_pixel_active  # (B, n_rows, n_cols)

    if ray_flags.ndim == 4:
        ray_flags[inactive_rays.unsqueeze(-1)] |= RayFlags.INVALID.value
    else:  # 2-D; ndim already validated above.
        if B != 1:
            raise ValueError(f"2D ray_flags require B == 1, got active_tile_mask with B={B}")
        ray_flags[inactive_rays.squeeze(0)] |= RayFlags.INVALID.value
