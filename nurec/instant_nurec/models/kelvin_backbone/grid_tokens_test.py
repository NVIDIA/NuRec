# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for geometry-only voxel dedup in the Kelvin point-query tokenizer."""

import pytest
import torch

from nurec.instant_nurec.models.kelvin_backbone.grid_tokens import (
    FullResInputs,
    _select_static_voxel_representatives,
    build_voxel_dedup_tokens,
)


def _full_inputs(
    xyz: torch.Tensor,
    *,
    distance: torch.Tensor | None = None,
    opacity: torch.Tensor | None = None,
    dynamic: torch.Tensor | None = None,
) -> FullResInputs:
    """Construct source fields with recognizable values around a supplied xyz grid."""
    batch_size, num_views, height, width, _ = xyz.shape
    scalar_shape = (batch_size, num_views, height, width, 1)
    pixel_shape = (batch_size, num_views, height, width)
    if distance is None:
        distance = torch.ones(scalar_shape, dtype=xyz.dtype)
    if opacity is None:
        opacity = torch.ones(scalar_shape, dtype=xyz.dtype)
    if dynamic is None:
        dynamic = torch.zeros(pixel_shape, dtype=torch.bool)

    flat_value = (
        torch.arange(num_views * height * width, dtype=xyz.dtype)
        .reshape(1, num_views, height, width, 1)
        .expand(batch_size, -1, -1, -1, -1)
        .clone()
    )
    return FullResInputs(
        xyz=xyz,
        prev_flow=torch.zeros_like(xyz),
        next_flow=torch.zeros_like(xyz),
        dyn_mask=dynamic,
        ts_src=flat_value.to(torch.int64),
        ts_prev=(flat_value - 1).to(torch.int64),
        ts_next=(flat_value + 1).to(torch.int64),
        footprints=torch.ones(scalar_shape, dtype=xyz.dtype),
        distance=distance,
        opacity_mask=opacity,
        ctx_rgb=flat_value.expand(-1, -1, -1, -1, 3).clone(),
        sem_class=flat_value.squeeze(-1).to(torch.int64),
        world_normal=torch.zeros_like(xyz),
        ray_origins=torch.zeros_like(xyz),
        ray_directions=torch.ones_like(xyz),
    )


def _valid_xyz(tokens) -> torch.Tensor:
    valid = tokens.point_valid_mask[0, :, 0] > 0
    return tokens.gs_xyz[0, valid]


def test_voxel_representative_is_nearest_center_with_source_index_tie_break() -> None:
    xyz = torch.tensor(
        [
            [0.40, 0.0, 0.0],
            [0.10, 0.0, 0.0],  # nearest voxel-0 center
            [1.25, 0.0, 0.0],  # voxel 1, exact-distance tie, smaller source index
            [0.75, 0.0, 0.0],
            [-1.20, 0.0, 0.0],  # negative voxel is retained and Morton-offset safely
        ]
    )
    source_indices = torch.arange(len(xyz), dtype=torch.int64)

    representatives, occupied = _select_static_voxel_representatives(
        detached_xyz=xyz,
        flat_source_indices=source_indices,
        voxel_size_m=1.0,
    )

    assert occupied == 3
    assert set(representatives.tolist()) == {1, 2, 4}


def test_voxel_quantization_locks_torch_round_half_to_even_for_signed_boundaries() -> None:
    # The shared merge convention is torch.round: +/-0.5 map to voxel 0, while
    # +/-1.5 map to the even signed voxels +/-2. Lock both sides of zero so a
    # later floor-based hash cannot silently change boundary ownership.
    xyz = torch.tensor(
        [
            [-1.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ]
    )
    source_indices = torch.arange(len(xyz), dtype=torch.int64)

    representatives, occupied = _select_static_voxel_representatives(
        detached_xyz=xyz,
        flat_source_indices=source_indices,
        voxel_size_m=1.0,
    )

    assert occupied == 3
    # -0.5 and +0.5 tie inside voxel 0; the smaller flat source index wins.
    assert set(representatives.tolist()) == {0, 1, 3}


def test_voxel_dedup_gathers_only_representatives_and_preserves_selected_gradients() -> None:
    xyz = torch.tensor(
        [[[[[0.40, 0.0, 0.0], [0.10, 0.0, 0.0], [1.10, 0.0, 0.0], [1.40, 0.0, 0.0]]]]],
        requires_grad=True,
    )
    full = _full_inputs(xyz)

    tokens, diagnostics = build_voxel_dedup_tokens(
        stride=1,
        grid_center_stride=8,
        voxel_size_m=1.0,
        max_total_tokens=2,
        full=full,
    )

    torch.testing.assert_close(_valid_xyz(tokens)[:, 0].sort().values, torch.tensor([0.10, 1.10]))
    assert diagnostics[0].valid_static_candidates == 4
    assert diagnostics[0].occupied_static_voxels == 2
    assert diagnostics[0].kept_static_points == 2
    assert diagnostics[0].padded_tokens == 0
    assert tokens.query_xyz.shape == (1, 1, 64, 3)

    _valid_xyz(tokens).sum().backward()
    assert xyz.grad is not None
    torch.testing.assert_close(xyz.grad[0, 0, 0, :, 0], torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_voxel_dedup_removes_invalid_depth_xyz_and_zero_opacity() -> None:
    xyz = torch.tensor([[[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [3.0, 0.0, 0.0]]]]])
    distance = torch.tensor([[[[[1.0], [-1.0], [1.0], [1.0]]]]])
    opacity = torch.tensor([[[[[1.0], [1.0], [1.0], [0.0]]]]])

    tokens, diagnostics = build_voxel_dedup_tokens(
        stride=1,
        grid_center_stride=8,
        voxel_size_m=0.1,
        max_total_tokens=1,
        full=_full_inputs(xyz, distance=distance, opacity=opacity),
    )

    torch.testing.assert_close(_valid_xyz(tokens), torch.tensor([[0.0, 0.0, 0.0]]))
    assert diagnostics[0].valid_static_candidates == 1
    assert torch.isfinite(tokens.query_xyz).all()


def test_dynamic_points_are_not_deduplicated_and_invalid_subpoints_stay_non_dynamic() -> None:
    xyz = torch.zeros((1, 2, 8, 8, 3))
    xyz[0, 0, 0, 0] = torch.tensor([5.0, 0.0, 0.0])
    xyz[0, 1, 0, 0] = torch.tensor([5.0, 0.0, 0.0])  # same xyz in another source view
    xyz[0, :, 1:, :, 0] = 20.0  # duplicate static pool
    dynamic = torch.zeros((1, 2, 8, 8), dtype=torch.bool)
    dynamic[0, :, 0, 0] = True

    tokens, diagnostics = build_voxel_dedup_tokens(
        stride=1,
        grid_center_stride=8,
        voxel_size_m=1.0,
        max_total_tokens=4,
        full=_full_inputs(xyz, dynamic=dynamic),
    )

    diag = diagnostics[0]
    assert diag.dynamic_tokens == 2
    assert diag.valid_dynamic_candidates == 2
    assert diag.occupied_static_voxels == 2
    assert diag.kept_static_tokens == 1
    assert diag.padded_tokens == 0
    assert tokens.query_xyz.shape == (1, 3, 64, 3)
    assert tokens.gs_dyn_mask.sum().item() == 2
    torch.testing.assert_close(tokens.gs_xyz[0, tokens.gs_dyn_mask[0]], torch.tensor([[5.0, 0.0, 0.0]] * 2))
    assert (tokens.point_valid_mask[0, tokens.gs_dyn_mask[0], 0] > 0.0).all()
    assert torch.isfinite(tokens.query_xyz).all()


@pytest.mark.parametrize("height,width,stride", [(9, 8, 1), (8, 9, 1), (7, 8, 1), (8, 7, 1), (1, 1, 1), (17, 15, 2)])
def test_dynamic_boundary_groups_preserve_every_candidate_and_gradient(height: int, width: int, stride: int) -> None:
    xyz = torch.arange(2 * height * width * 3, dtype=torch.float32).reshape(1, 2, height, width, 3).requires_grad_()
    dynamic = torch.ones((1, 2, height, width), dtype=torch.bool)

    tokens, diagnostics = build_voxel_dedup_tokens(
        stride=stride,
        grid_center_stride=8,
        voxel_size_m=1.0,
        max_total_tokens=8,
        full=_full_inputs(xyz, dynamic=dynamic),
    )

    expected = xyz[:, :, ::stride, ::stride].reshape(-1, 3)
    actual = tokens.gs_xyz[0, tokens.gs_dyn_mask[0]]
    torch.testing.assert_close(actual[actual[:, 0].argsort()], expected[expected[:, 0].argsort()])
    assert diagnostics[0].valid_dynamic_candidates == expected.shape[0]
    assert tokens.gs_dyn_mask.sum().item() == expected.shape[0]
    assert diagnostics[0].kept_static_tokens == 0
    assert torch.isfinite(tokens.query_xyz).all()
    invalid = ~tokens.gs_dyn_mask[0]
    assert not tokens.point_valid_mask[0, invalid].any()
    assert not tokens.gs_xyz[0, invalid].any()
    assert not tokens.gs_ts_src[0, invalid].any()
    assert not tokens.gs_ctx_rgb[0, invalid].any()

    actual.sum().backward()
    expected_grad = torch.zeros_like(xyz)
    expected_grad[:, :, ::stride, ::stride] = 1.0
    torch.testing.assert_close(xyz.grad, expected_grad)


def test_static_overflow_evenly_samples_morton_tokens_instead_of_prefix() -> None:
    # Four source views x 64 unique x-axis voxels -> four Morton-ordered static tokens.
    xyz = torch.zeros((1, 4, 8, 8, 3))
    xyz[..., 0] = torch.arange(256, dtype=torch.float32).reshape(1, 4, 8, 8)

    tokens, diagnostics = build_voxel_dedup_tokens(
        stride=1,
        grid_center_stride=8,
        voxel_size_m=1.0,
        max_total_tokens=2,
        full=_full_inputs(xyz),
    )

    selected_x = _valid_xyz(tokens)[:, 0]
    assert selected_x.min().item() == 0.0
    assert selected_x.max().item() == 255.0  # proves the last spatial stratum survived
    assert not ((selected_x >= 64.0) & (selected_x < 192.0)).any()
    assert diagnostics[0].static_tokens_before_cap == 4
    assert diagnostics[0].static_overflow_tokens == 2


def test_samples_pad_only_to_local_batch_maximum_not_configured_ceiling() -> None:
    xyz = torch.zeros((2, 1, 1, 128, 3))
    xyz[..., 0] = torch.arange(128, dtype=torch.float32)
    opacity = torch.ones((2, 1, 1, 128, 1))
    opacity[0, :, :, 1:] = 0.0  # one voxel/token in sample 0; two tokens in sample 1

    tokens, diagnostics = build_voxel_dedup_tokens(
        stride=1,
        grid_center_stride=8,
        voxel_size_m=1.0,
        max_total_tokens=4,
        full=_full_inputs(xyz, opacity=opacity),
    )

    assert tokens.query_xyz.shape == (2, 2, 64, 3)
    assert diagnostics[0].kept_static_tokens == 1
    assert diagnostics[0].padded_tokens == 1
    assert diagnostics[1].kept_static_tokens == 2
    assert diagnostics[1].padded_tokens == 0
    assert not tokens.point_valid_mask[0, 64:].any()
    assert torch.isfinite(tokens.query_xyz).all()


def test_dynamic_token_overflow_preserves_all_dynamic_and_drops_static() -> None:
    xyz = torch.zeros((1, 3, 8, 8, 3))
    dynamic = torch.zeros((1, 3, 8, 8), dtype=torch.bool)
    dynamic[0, :, 0, 0] = True  # one retained dynamic grid token per view

    tokens, diagnostics = build_voxel_dedup_tokens(
        stride=1,
        grid_center_stride=8,
        voxel_size_m=1.0,
        max_total_tokens=2,
        full=_full_inputs(xyz, dynamic=dynamic),
    )

    diag = diagnostics[0]
    assert diag.dynamic_tokens == 3
    assert diag.dynamic_overflow_tokens == 1
    assert diag.static_tokens_before_cap == 1
    assert diag.kept_static_tokens == 0
    assert diag.static_overflow_tokens == 1
    assert tokens.query_xyz.shape == (1, 3, 64, 3)
    assert tokens.gs_dyn_mask.sum().item() == 3
    assert tokens.point_valid_mask.sum().item() == 3
