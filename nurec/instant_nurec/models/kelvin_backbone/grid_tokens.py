# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grid-tokenization helpers for the Kelvin point-query decoder.

Splits the per-pixel tensors produced by the depth/encoder stack into per-token,
per-Gaussian groups for the cross-attention point-query head. Main entry points:

- build_grid_tokens(...): depth-strided + (optional) road-area oversampled tokens.
- build_voxel_dedup_tokens(...): static metric-voxel representatives plus
  untruncated dynamic per-frame grid tokens under a static-aware token budget.
- concat_grid_tokens([...]): field-wise dim=1 concat of multiple GridTokens.
"""

from dataclasses import dataclass, fields, replace

import torch

from einops import rearrange

from nurec.utils.morton import _encode_morton_3d


_MORTON_AXIS_LIMIT = 1 << 21


@dataclass(kw_only=True, slots=True)
class GridTokens:
    """Per-token / per-Gaussian tensors produced by `build_grid_tokens`.

    Shapes (with `M` = number of tokens, `K` = `cs**2` sub-points per token):
      - `query_xyz`: `(B, M, K, 3)` -- input to `xyz_embed`.
      - all `gs_*`: `(B, M*K, *)` -- per-Gaussian tensors for downstream activation / split.
      - `point_valid_mask`: `(B, M*K, 1)` -- already folds in per-token validity (padded tokens -> 0).
    """

    query_xyz: torch.Tensor
    gs_xyz: torch.Tensor
    pixel_scale: torch.Tensor
    point_valid_mask: torch.Tensor
    gs_ctx_rgb: torch.Tensor
    gs_sem_class: torch.Tensor
    gs_prev_flow: torch.Tensor
    gs_next_flow: torch.Tensor
    gs_dyn_mask: torch.Tensor
    gs_ts_src: torch.Tensor
    gs_ts_prev: torch.Tensor
    gs_ts_next: torch.Tensor
    gs_normals: torch.Tensor
    gs_ray_origins: torch.Tensor
    gs_ray_directions: torch.Tensor


@dataclass(kw_only=True, slots=True)
class FullResInputs:
    """Bundle of full-resolution per-pixel tensors consumed by `build_grid_tokens`.

    All tensors are shaped ``(B, V, H, W[, C])`` at the encoder's full resolution.
    The decoder builds this once and feeds the same instance to every call.
    """

    xyz: torch.Tensor
    prev_flow: torch.Tensor
    next_flow: torch.Tensor
    dyn_mask: torch.Tensor
    ts_src: torch.Tensor
    ts_prev: torch.Tensor
    ts_next: torch.Tensor
    footprints: torch.Tensor
    distance: torch.Tensor
    opacity_mask: torch.Tensor
    ctx_rgb: torch.Tensor
    sem_class: torch.Tensor
    world_normal: torch.Tensor
    ray_origins: torch.Tensor
    ray_directions: torch.Tensor


@dataclass(frozen=True, kw_only=True, slots=True)
class VoxelDedupDiagnostics:
    """Per-sample point/token counts emitted by ``build_voxel_dedup_tokens``."""

    raw_candidates: int
    valid_static_candidates: int
    valid_dynamic_candidates: int
    occupied_static_voxels: int
    static_tokens_before_cap: int
    dynamic_tokens: int
    dynamic_overflow_tokens: int
    kept_static_tokens: int
    kept_static_points: int
    padded_tokens: int
    static_overflow_tokens: int
    kept_static_source_counts: tuple[int, ...]


def concat_grid_tokens(tokens: list[GridTokens]) -> GridTokens:
    """Concatenate per-view-token lists field-wise along dim=1.

    Each input is a GridTokens; the result holds the union of tokens, in input order.
    For a single-element list the input is returned unchanged.
    """
    if len(tokens) == 1:
        return tokens[0]
    return GridTokens(**{f.name: torch.cat([getattr(t, f.name) for t in tokens], dim=1) for f in fields(GridTokens)})


def _flatten_mk(x: torch.Tensor) -> torch.Tensor:
    """Collapse the per-token / per-Gaussian dims: ``(B, M, K[, C]) -> (B, M*K[, C])``."""
    if x.dim() == 4:
        return rearrange(x, "B M K C -> B (M K) C")
    return rearrange(x, "B M K -> B (M K)")


def _gather_strided_candidates(
    tensor: torch.Tensor,
    *,
    batch_idx: int,
    flat_indices: torch.Tensor,
    stride: int,
    downsampled_height: int,
    downsampled_width: int,
) -> torch.Tensor:
    """Gather only requested entries from a virtual ``tensor[:, :, ::stride, ::stride]`` flattening."""
    pixels_per_view = downsampled_height * downsampled_width
    view_idx = flat_indices // pixels_per_view
    pixel_idx = flat_indices % pixels_per_view
    y_idx = (pixel_idx // downsampled_width) * stride
    x_idx = (pixel_idx % downsampled_width) * stride
    return tensor[batch_idx, view_idx, y_idx, x_idx]


def _zero_where_invalid(tensor: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Replace invalid/padded entries with finite zero without propagating NaNs through multiplication."""
    mask = valid
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, tensor, torch.zeros_like(tensor))


def _zero_grid_tokens(*, full: FullResInputs, num_tokens: int, points_per_token: int) -> GridTokens:
    """Allocate finite, fully invalid token padding with source-compatible dtypes/devices."""
    num_points = num_tokens * points_per_token
    return GridTokens(
        query_xyz=full.xyz.new_zeros((1, num_tokens, points_per_token, 3)),
        gs_xyz=full.xyz.new_zeros((1, num_points, 3)),
        pixel_scale=full.footprints.new_zeros((1, num_points, 1)),
        point_valid_mask=full.opacity_mask.new_zeros((1, num_points, 1)),
        gs_ctx_rgb=full.ctx_rgb.new_zeros((1, num_points, full.ctx_rgb.shape[-1])),
        gs_sem_class=full.sem_class.new_zeros((1, num_points)),
        gs_prev_flow=full.prev_flow.new_zeros((1, num_points, full.prev_flow.shape[-1])),
        gs_next_flow=full.next_flow.new_zeros((1, num_points, full.next_flow.shape[-1])),
        gs_dyn_mask=full.dyn_mask.new_zeros((1, num_points), dtype=torch.bool),
        gs_ts_src=full.ts_src.new_zeros((1, num_points)),
        gs_ts_prev=full.ts_prev.new_zeros((1, num_points)),
        gs_ts_next=full.ts_next.new_zeros((1, num_points)),
        gs_normals=full.world_normal.new_zeros((1, num_points, full.world_normal.shape[-1])),
        gs_ray_origins=full.ray_origins.new_zeros((1, num_points, 3)),
        gs_ray_directions=full.ray_directions.new_zeros((1, num_points, 3)),
    )


def _build_indexed_grid_tokens(
    *,
    full: FullResInputs,
    batch_idx: int,
    flat_indices: torch.Tensor,
    valid: torch.Tensor,
    stride: int,
    downsampled_height: int,
    downsampled_width: int,
    dynamic: bool,
) -> GridTokens:
    """Gather selected source points into ``[M,K]`` token groups and zero their padding."""
    assert flat_indices.ndim == 2 and valid.shape == flat_indices.shape
    num_tokens, points_per_token = flat_indices.shape
    if num_tokens == 0:
        return _zero_grid_tokens(full=full, num_tokens=0, points_per_token=points_per_token)

    safe_indices = flat_indices.clamp_min(0)

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        return _gather_strided_candidates(
            tensor,
            batch_idx=batch_idx,
            flat_indices=safe_indices,
            stride=stride,
            downsampled_height=downsampled_height,
            downsampled_width=downsampled_width,
        )

    xyz = _zero_where_invalid(gather(full.xyz), valid)
    prev_flow = _zero_where_invalid(gather(full.prev_flow), valid)
    next_flow = _zero_where_invalid(gather(full.next_flow), valid)
    ts_src = _zero_where_invalid(gather(full.ts_src).squeeze(-1), valid)
    ts_prev = _zero_where_invalid(gather(full.ts_prev).squeeze(-1), valid)
    ts_next = _zero_where_invalid(gather(full.ts_next).squeeze(-1), valid)
    footprints = gather(full.footprints)
    distance = gather(full.distance).detach()
    pixel_scale = _zero_where_invalid(footprints * distance * stride, valid)
    opacity = _zero_where_invalid(gather(full.opacity_mask), valid)
    ctx_rgb = _zero_where_invalid(gather(full.ctx_rgb), valid)
    sem_class = _zero_where_invalid(gather(full.sem_class), valid)
    world_normal = _zero_where_invalid(gather(full.world_normal), valid)
    ray_origins = _zero_where_invalid(gather(full.ray_origins), valid)
    ray_directions = _zero_where_invalid(gather(full.ray_directions), valid)
    dynamic_mask = valid if dynamic else torch.zeros_like(valid)

    return GridTokens(
        query_xyz=xyz.unsqueeze(0),
        gs_xyz=xyz.reshape(1, num_tokens * points_per_token, 3),
        pixel_scale=pixel_scale.reshape(1, num_tokens * points_per_token, 1),
        point_valid_mask=opacity.reshape(1, num_tokens * points_per_token, 1),
        gs_ctx_rgb=ctx_rgb.reshape(1, num_tokens * points_per_token, ctx_rgb.shape[-1]),
        gs_sem_class=sem_class.reshape(1, num_tokens * points_per_token),
        gs_prev_flow=prev_flow.reshape(1, num_tokens * points_per_token, prev_flow.shape[-1]),
        gs_next_flow=next_flow.reshape(1, num_tokens * points_per_token, next_flow.shape[-1]),
        gs_dyn_mask=dynamic_mask.reshape(1, num_tokens * points_per_token),
        gs_ts_src=ts_src.reshape(1, num_tokens * points_per_token),
        gs_ts_prev=ts_prev.reshape(1, num_tokens * points_per_token),
        gs_ts_next=ts_next.reshape(1, num_tokens * points_per_token),
        gs_normals=world_normal.reshape(1, num_tokens * points_per_token, world_normal.shape[-1]),
        gs_ray_origins=ray_origins.reshape(1, num_tokens * points_per_token, 3),
        gs_ray_directions=ray_directions.reshape(1, num_tokens * points_per_token, 3),
    )


def _select_static_voxel_representatives(
    *,
    detached_xyz: torch.Tensor,
    flat_source_indices: torch.Tensor,
    voxel_size_m: float,
) -> tuple[torch.Tensor, int]:
    """Select the real point nearest each metric voxel center and Morton-sort it.

    Exact squared-distance ties are resolved by the smallest original flat
    source index. All inputs used by this discrete decision are detached by the
    caller; returned indices gather the original differentiable fields later.
    """
    if flat_source_indices.numel() == 0:
        return flat_source_indices, 0

    voxel_coords = torch.round(detached_xyz / voxel_size_m).to(torch.int64)
    unique_voxels, inverse = torch.unique(voxel_coords, dim=0, return_inverse=True)
    voxel_centers = unique_voxels[inverse].to(detached_xyz.dtype) * voxel_size_m
    center_distance_sq = torch.sum((detached_xyz - voxel_centers) ** 2, dim=-1)

    num_voxels = unique_voxels.shape[0]
    min_distance_sq = torch.full(
        (num_voxels,),
        float("inf"),
        dtype=center_distance_sq.dtype,
        device=center_distance_sq.device,
    )
    min_distance_sq.scatter_reduce_(0, inverse, center_distance_sq, reduce="amin", include_self=True)
    is_min_distance = center_distance_sq == min_distance_sq[inverse]

    sentinel = flat_source_indices.new_full((num_voxels,), torch.iinfo(flat_source_indices.dtype).max)
    tied_source_indices = torch.where(
        is_min_distance,
        flat_source_indices,
        torch.iinfo(flat_source_indices.dtype).max,
    )
    sentinel.scatter_reduce_(0, inverse, tied_source_indices, reduce="amin", include_self=True)
    representative_indices = sentinel

    shifted_voxels = unique_voxels - unique_voxels.amin(dim=0, keepdim=True)
    if int(shifted_voxels.amax().item()) >= _MORTON_AXIS_LIMIT:
        raise ValueError(
            "Voxel-coordinate span exceeds the shared 21-bit Morton encoder after per-sample minimum offset; "
            f"maximum shifted coordinate is {int(shifted_voxels.amax().item())}."
        )
    morton = _encode_morton_3d(shifted_voxels[:, 0], shifted_voxels[:, 1], shifted_voxels[:, 2])
    morton_order = morton.argsort(stable=True)
    return representative_indices[morton_order], num_voxels


def _dynamic_grid_indices(
    *,
    valid_dynamic: torch.Tensor,
    grid_center_stride: int,
    downsampled_height: int,
    downsampled_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group every dynamic point, padding partial boundary groups with invalid entries."""
    padded_height = ((downsampled_height + grid_center_stride - 1) // grid_center_stride) * grid_center_stride
    padded_width = ((downsampled_width + grid_center_stride - 1) // grid_center_stride) * grid_center_stride
    points_per_token = grid_center_stride**2
    if padded_height == 0 or padded_width == 0:
        empty = torch.empty((0, points_per_token), dtype=torch.int64, device=valid_dynamic.device)
        return empty, torch.empty_like(empty, dtype=torch.bool)

    if padded_height != downsampled_height or padded_width != downsampled_width:
        valid_dynamic = torch.nn.functional.pad(
            valid_dynamic,
            (0, padded_width - downsampled_width, 0, padded_height - downsampled_height),
            value=False,
        )
    grouped_valid = rearrange(
        valid_dynamic,
        "V (h cs1) (w cs2) -> (V h w) (cs1 cs2)",
        cs1=grid_center_stride,
        cs2=grid_center_stride,
    )
    kept_token_indices = grouped_valid.any(dim=-1).nonzero(as_tuple=False).flatten()
    if kept_token_indices.numel() == 0:
        return kept_token_indices.reshape(0, points_per_token), grouped_valid[:0]

    groups_y = padded_height // grid_center_stride
    groups_x = padded_width // grid_center_stride
    groups_per_view = groups_y * groups_x
    view_idx = kept_token_indices // groups_per_view
    local_token_idx = kept_token_indices % groups_per_view
    group_y = local_token_idx // groups_x
    group_x = local_token_idx % groups_x

    subpoint_idx = torch.arange(points_per_token, device=valid_dynamic.device)
    sub_y = subpoint_idx // grid_center_stride
    sub_x = subpoint_idx % grid_center_stride
    y_idx = group_y[:, None] * grid_center_stride + sub_y[None]
    x_idx = group_x[:, None] * grid_center_stride + sub_x[None]
    flat_indices = view_idx[:, None] * (downsampled_height * downsampled_width) + y_idx * downsampled_width + x_idx
    kept_valid = grouped_valid[kept_token_indices]
    # Padding must not index beyond the source image or into the next view.
    # Indexed token gathering clamps this sentinel before zeroing invalid fields.
    return torch.where(kept_valid, flat_indices, -1), kept_valid


def build_voxel_dedup_tokens(
    *,
    stride: int,
    grid_center_stride: int,
    voxel_size_m: float,
    max_total_tokens: int,
    full: FullResInputs,
) -> tuple[GridTokens, list[VoxelDedupDiagnostics]]:
    """Build bounded-budget dynamic + geometry-only deduplicated static tokens.

    Static candidates are deduplicated across every source view by detached
    metric xyz. One actual point nearest each rounded voxel center is retained;
    an exact distance tie selects the smallest camera-major flat source index.
    Representatives are Morton-ordered and grouped into ``K`` points per query.

    Dynamic points are never deduplicated across cameras or time. Their original
    per-view 2D groups are compacted when at least one valid dynamic point is
    present, and all non-dynamic/invalid subpoints in those groups are zeroed.
    Dynamic tokens consume the budget first and are never truncated. If dynamic
    coverage alone exceeds ``max_total_tokens``, all dynamic tokens are retained
    and no static tokens are emitted for that sample. Static overflow is spatially
    stratified by evenly sampling complete tokens over Morton order. Samples are
    padded only to the largest retained token count in the local batch, never to
    ``max_total_tokens`` itself; padding is finite zero with validity and dynamic
    masks both false. This keeps the configured budget from turning deduplicated-
    away points back into expensive empty cross-attention queries while making a
    dynamic-only overrun explicit in diagnostics.

    Depth confidence is deliberately absent from ``FullResInputs`` and from this
    selection path.
    """
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}.")
    if grid_center_stride < 1:
        raise ValueError(f"grid_center_stride must be positive, got {grid_center_stride}.")
    if voxel_size_m <= 0.0:
        raise ValueError(f"voxel_size_m must be positive, got {voxel_size_m}.")
    if max_total_tokens <= 0:
        raise ValueError(f"max_total_tokens must be positive, got {max_total_tokens}.")

    batch_size, num_views, _, _, _ = full.xyz.shape
    xyz_downsampled = full.xyz[:, :, ::stride, ::stride, :]
    downsampled_height, downsampled_width = xyz_downsampled.shape[2:4]
    points_per_token = grid_center_stride**2
    raw_candidates = num_views * downsampled_height * downsampled_width

    sample_tokens: list[GridTokens] = []
    diagnostics: list[VoxelDedupDiagnostics] = []
    for batch_idx in range(batch_size):
        detached_distance = full.distance[batch_idx, :, ::stride, ::stride, 0].detach()
        detached_opacity = full.opacity_mask[batch_idx, :, ::stride, ::stride, 0].detach()
        detached_xyz = xyz_downsampled[batch_idx].detach()
        dynamic_mask = full.dyn_mask[batch_idx, :, ::stride, ::stride].bool()
        valid = (
            torch.isfinite(detached_distance)
            & (detached_distance > 0.0)
            & torch.isfinite(detached_xyz).all(dim=-1)
            & torch.isfinite(detached_opacity)
            & (detached_opacity > 0.0)
        )
        valid_dynamic = valid & dynamic_mask
        valid_static = valid & ~dynamic_mask

        dynamic_indices, dynamic_valid = _dynamic_grid_indices(
            valid_dynamic=valid_dynamic,
            grid_center_stride=grid_center_stride,
            downsampled_height=downsampled_height,
            downsampled_width=downsampled_width,
        )
        num_dynamic_tokens = dynamic_indices.shape[0]
        dynamic_overflow_tokens = max(0, num_dynamic_tokens - max_total_tokens)

        dynamic_tokens = _build_indexed_grid_tokens(
            full=full,
            batch_idx=batch_idx,
            flat_indices=dynamic_indices,
            valid=dynamic_valid,
            stride=stride,
            downsampled_height=downsampled_height,
            downsampled_width=downsampled_width,
            dynamic=True,
        )

        static_source_indices = valid_static.reshape(-1).nonzero(as_tuple=False).flatten()
        static_xyz = _gather_strided_candidates(
            full.xyz.detach(),
            batch_idx=batch_idx,
            flat_indices=static_source_indices,
            stride=stride,
            downsampled_height=downsampled_height,
            downsampled_width=downsampled_width,
        )
        representative_indices, num_occupied_voxels = _select_static_voxel_representatives(
            detached_xyz=static_xyz,
            flat_source_indices=static_source_indices,
            voxel_size_m=voxel_size_m,
        )

        num_static_tokens_before_cap = (representative_indices.numel() + points_per_token - 1) // points_per_token
        static_padding_points = num_static_tokens_before_cap * points_per_token - representative_indices.numel()
        if static_padding_points:
            representative_indices = torch.cat(
                [representative_indices, representative_indices.new_full((static_padding_points,), -1)]
            )
        static_indices = representative_indices.reshape(num_static_tokens_before_cap, points_per_token)
        static_valid = static_indices >= 0

        remaining_static_tokens = max(0, max_total_tokens - num_dynamic_tokens)
        static_overflow_tokens = max(0, num_static_tokens_before_cap - remaining_static_tokens)
        if static_overflow_tokens:
            if remaining_static_tokens == 0:
                selected_static_tokens = static_indices[:0]
                selected_static_valid = static_valid[:0]
            else:
                selected_token_indices = (
                    torch.linspace(
                        0,
                        num_static_tokens_before_cap - 1,
                        steps=remaining_static_tokens,
                        device=static_indices.device,
                    )
                    .round()
                    .to(torch.int64)
                )
                selected_static_tokens = static_indices[selected_token_indices]
                selected_static_valid = static_valid[selected_token_indices]
        else:
            selected_static_tokens = static_indices
            selected_static_valid = static_valid

        static_tokens = _build_indexed_grid_tokens(
            full=full,
            batch_idx=batch_idx,
            flat_indices=selected_static_tokens,
            valid=selected_static_valid,
            stride=stride,
            downsampled_height=downsampled_height,
            downsampled_width=downsampled_width,
            dynamic=False,
        )

        num_kept_static_tokens = selected_static_tokens.shape[0]
        combined = concat_grid_tokens([dynamic_tokens, static_tokens])
        assert combined.query_xyz.shape[1] <= max(max_total_tokens, num_dynamic_tokens)
        sample_tokens.append(combined)

        kept_static_indices = selected_static_tokens[selected_static_valid]
        kept_source_views = kept_static_indices // (downsampled_height * downsampled_width)
        kept_source_counts = torch.bincount(kept_source_views, minlength=num_views)
        diagnostics.append(
            VoxelDedupDiagnostics(
                raw_candidates=raw_candidates,
                valid_static_candidates=int(valid_static.sum().item()),
                valid_dynamic_candidates=int(valid_dynamic.sum().item()),
                occupied_static_voxels=num_occupied_voxels,
                static_tokens_before_cap=num_static_tokens_before_cap,
                dynamic_tokens=num_dynamic_tokens,
                dynamic_overflow_tokens=dynamic_overflow_tokens,
                kept_static_tokens=num_kept_static_tokens,
                kept_static_points=int(selected_static_valid.sum().item()),
                # Filled after every sample's retained count is known. Padding
                # aligns only the local batch, not the configured safety ceiling.
                padded_tokens=0,
                static_overflow_tokens=static_overflow_tokens,
                kept_static_source_counts=tuple(int(count) for count in kept_source_counts.tolist()),
            )
        )

    batch_token_count = max(1, max(tokens.query_xyz.shape[1] for tokens in sample_tokens))
    padded_sample_tokens: list[GridTokens] = []
    for sample_idx, tokens in enumerate(sample_tokens):
        num_padding_tokens = batch_token_count - tokens.query_xyz.shape[1]
        if num_padding_tokens:
            tokens = concat_grid_tokens(
                [
                    tokens,
                    _zero_grid_tokens(
                        full=full,
                        num_tokens=num_padding_tokens,
                        points_per_token=points_per_token,
                    ),
                ]
            )
        assert tokens.query_xyz.shape[1] == batch_token_count
        padded_sample_tokens.append(tokens)
        diagnostics[sample_idx] = replace(diagnostics[sample_idx], padded_tokens=num_padding_tokens)

    return GridTokens(
        **{
            field.name: torch.cat([getattr(tokens, field.name) for tokens in padded_sample_tokens], dim=0)
            for field in fields(GridTokens)
        }
    ), diagnostics


def _make_pipeline(
    *,
    stride: int,
    cs: int,
    H_t: int,
    W_t: int,
    sel_idx: torch.Tensor | None,
):
    """Build the slice -> crop -> ``cs x cs`` group -> (optional) gather -> V/M-flatten pipeline.

    Routing by ``x.dim()``: 5 means a trailing channel dim is present and is preserved;
    4 means no channel dim. ``squeeze_c1=True`` is used for tensors that start as
    ``(B, V, H, W, 1)`` but should collapse to ``(B, V, M, K)`` (no trailing 1) -- this
    matches the historical handling of ``ts_*``.
    """

    def _gather_token(x: torch.Tensor) -> torch.Tensor:
        assert sel_idx is not None
        tail_dims = x.shape[3:]
        idx = sel_idx
        for _ in tail_dims:
            idx = idx.unsqueeze(-1)
        idx = idx.expand(*sel_idx.shape, *tail_dims)
        return torch.gather(x, dim=2, index=idx)

    def run(x: torch.Tensor, *, squeeze_c1: bool = False) -> torch.Tensor:
        # 1) stride-slice
        if x.dim() == 5:
            x = x[:, :, ::stride, ::stride, :]
        else:
            x = x[:, :, ::stride, ::stride]
        # 2) crop to cs-multiple
        if x.dim() == 5:
            x = x[:, :, :H_t, :W_t, :]
        else:
            x = x[:, :, :H_t, :W_t]
        # 3) cs x cs group
        if x.dim() == 5:
            x = rearrange(x, "B V (h cs1) (w cs2) C -> B V (h w) (cs1 cs2) C", cs1=cs, cs2=cs)
            if squeeze_c1:
                x = x.squeeze(-1)
        else:
            x = rearrange(x, "B V (h cs1) (w cs2) -> B V (h w) (cs1 cs2)", cs1=cs, cs2=cs)
        # 4) optional gather
        if sel_idx is not None:
            x = _gather_token(x)
        # 5) V/M-flatten
        if x.dim() == 5:
            x = rearrange(x, "B V M K C -> B (V M) K C")
        else:
            x = rearrange(x, "B V M K -> B (V M) K")
        return x

    return run


def build_grid_tokens(
    *,
    stride: int,
    cs: int,
    full: FullResInputs,
    keep_pixel_mask: torch.Tensor | None = None,
    keep_token_threshold: float = 0.0,
    max_tokens_per_view: int | None = None,
) -> GridTokens:
    """Build point-query tokens by 2D-strided downsampling and ``cs x cs`` grouping.

    All ``full.*`` tensors are at full resolution ``(B, V, H, W[, C])``.
    ``stride`` is an integer >= 1.
    When ``keep_pixel_mask`` is provided, only tokens whose ``cs*cs`` sub-cells have a
    fraction ``>= keep_token_threshold`` of true entries are kept; per ``(B, V)`` slot the
    kept tokens are then randomly subsampled to ``max_tokens_per_view`` (slots with fewer
    candidates are padded with zero-mask tokens so M_road stays fixed across the batch).
    """
    # Build pixel_scale at full-res first; pipeline will then slice/crop/group it like
    # any other 5D per-pixel tensor. distance is detached to match the original code.
    full_pscale = full.footprints * full.distance.detach() * stride

    # Compute H_t / W_t (cs-multiple crop sizes) from the strided xyz.
    _, _, Hds, Wds, _ = full.xyz[:, :, ::stride, ::stride, :].shape
    H_t = (Hds // cs) * cs
    W_t = (Wds // cs) * cs
    n_per_view = (H_t // cs) * (W_t // cs)

    # --- Optional gather setup (selects k_cap tokens per (B,V) slot) ---
    sel_idx: torch.Tensor | None = None
    token_valid: torch.Tensor | None = None
    if keep_pixel_mask is not None:
        assert max_tokens_per_view is not None, "max_tokens_per_view must be set when keep_pixel_mask is provided"
        keep_pixel_ds = keep_pixel_mask[:, :, ::stride, ::stride][:, :, :H_t, :W_t]
        keep_pixel_grp = rearrange(
            keep_pixel_ds, "B V (h cs1) (w cs2) -> B V (h w) (cs1 cs2)", cs1=cs, cs2=cs
        ).float()  # (B, V, n_per_view, K)
        keep_ratio = keep_pixel_grp.mean(dim=-1)  # (B, V, n_per_view)
        keep_token = keep_ratio >= keep_token_threshold

        k_cap = min(max_tokens_per_view, n_per_view)
        rand_scores = torch.rand_like(keep_ratio)
        rand_scores = rand_scores.masked_fill(~keep_token, float("-inf"))
        sel_scores, sel_idx = torch.topk(rand_scores, k=k_cap, dim=-1)  # (B, V, k_cap)
        token_valid = torch.isfinite(sel_scores)  # (B, V, k_cap) bool

    run = _make_pipeline(stride=stride, cs=cs, H_t=H_t, W_t=W_t, sel_idx=sel_idx)

    xyz_g = run(full.xyz)
    prev_flow_g = run(full.prev_flow)
    next_flow_g = run(full.next_flow)
    dyn_mask_g = run(full.dyn_mask)
    ts_src_g = run(full.ts_src, squeeze_c1=True)
    ts_prev_g = run(full.ts_prev, squeeze_c1=True)
    ts_next_g = run(full.ts_next, squeeze_c1=True)
    pscale_g = run(full_pscale)
    opacity_g = run(full.opacity_mask)
    ctx_rgb_g = run(full.ctx_rgb)
    sem_class_g = run(full.sem_class)
    world_normal_g = run(full.world_normal)
    ray_origins_g = run(full.ray_origins)
    ray_directions_g = run(full.ray_directions)

    # token_valid was computed pre-gather in (B, V, k_cap) space; the pipeline has now
    # flattened V and M, so we collapse the same way before multiplying into opacity.
    if token_valid is not None:
        token_valid_flat = rearrange(token_valid, "B V M -> B (V M)")
        opacity_g = opacity_g * token_valid_flat[..., None, None].to(opacity_g.dtype)

    return GridTokens(
        query_xyz=xyz_g,
        gs_xyz=_flatten_mk(xyz_g),
        pixel_scale=_flatten_mk(pscale_g),
        point_valid_mask=_flatten_mk(opacity_g),
        gs_ctx_rgb=_flatten_mk(ctx_rgb_g),
        gs_sem_class=_flatten_mk(sem_class_g),
        gs_prev_flow=_flatten_mk(prev_flow_g),
        gs_next_flow=_flatten_mk(next_flow_g),
        gs_dyn_mask=_flatten_mk(dyn_mask_g),
        gs_ts_src=_flatten_mk(ts_src_g),
        gs_ts_prev=_flatten_mk(ts_prev_g),
        gs_ts_next=_flatten_mk(ts_next_g),
        gs_normals=_flatten_mk(world_normal_g),
        gs_ray_origins=_flatten_mk(ray_origins_g),
        gs_ray_directions=_flatten_mk(ray_directions_g),
    )
