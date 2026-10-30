# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nurec.utils.morton import VOXEL_SIZE, morton_sort_order


def test_order_is_a_permutation():
    rng = torch.Generator().manual_seed(0)
    positions = torch.rand(1024, 3, generator=rng) * 100.0

    order = morton_sort_order(positions).order

    assert torch.equal(order.sort().values, torch.arange(1024))


def test_codes_are_nondecreasing_under_the_order():
    rng = torch.Generator().manual_seed(1)
    positions = torch.rand(512, 3, generator=rng) * 100.0

    result = morton_sort_order(positions)

    assert bool(torch.all(torch.diff(result.codes_low[result.order]) >= 0))


def test_sorting_brings_neighbouring_rows_closer_together():
    rng = torch.Generator().manual_seed(2)
    positions = torch.rand(4096, 3, generator=rng) * 100.0

    sorted_positions = positions[morton_sort_order(positions).order]

    original_step = (positions[1:] - positions[:-1]).norm(dim=1).mean()
    sorted_step = (sorted_positions[1:] - sorted_positions[:-1]).norm(dim=1).mean()
    assert sorted_step < original_step / 2.0


def test_rows_in_one_voxel_keep_their_relative_order():
    # All four fall in the same voxel, so the stable sort must not move them.
    positions = torch.full((4, 3), VOXEL_SIZE / 4.0)

    assert morton_sort_order(positions).order.tolist() == [0, 1, 2, 3]


def test_empty_input_returns_empty_order():
    result = morton_sort_order(torch.zeros(0, 3))

    assert result.order.shape == (0,)
    assert result.codes_high is None


def test_second_code_segment_kicks_in_for_wide_scenes():
    extent = float((1 << 21) * VOXEL_SIZE)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [extent, 0.0, 0.0],
            [VOXEL_SIZE, 0.0, 0.0],
            [extent + VOXEL_SIZE, 0.0, 0.0],
        ]
    )

    result = morton_sort_order(positions)

    assert result.codes_high is not None
    assert result.order.tolist() == [0, 2, 1, 3]


def test_narrow_scenes_use_a_single_code_segment():
    rng = torch.Generator().manual_seed(4)
    positions = torch.rand(64, 3, generator=rng) * 100.0

    assert morton_sort_order(positions).codes_high is None


def test_rejects_nonpositive_voxel_size():
    with pytest.raises(ValueError, match="voxel_size must be positive"):
        morton_sort_order(torch.zeros(1, 3), voxel_size=0.0)


def test_rejects_wrongly_shaped_positions():
    with pytest.raises(ValueError, match=r"must be \(N, 3\)"):
        morton_sort_order(torch.zeros(4, 2))


def test_rejects_nonfinite_positions():
    with pytest.raises(ValueError, match="non-finite"):
        morton_sort_order(torch.tensor([[0.0, 0.0, float("nan")]]))


def test_rejects_coordinates_beyond_the_two_segment_range():
    positions = torch.tensor([[0.0, 0.0, 0.0], [float((1 << 42) * VOXEL_SIZE), 0.0, 0.0]])

    with pytest.raises(ValueError, match="exceeds 42-bit Morton limit"):
        morton_sort_order(positions)


def test_error_messages_use_the_supplied_label():
    with pytest.raises(ValueError, match=r"background\.positions"):
        morton_sort_order(torch.zeros(4, 2), label="background.positions")


def test_negative_coordinates_are_shifted_to_the_origin_before_encoding():
    """The encoder masks to 21 unsigned bits, so a negative voxel coord would wrap without the shift.

    Two points either side of the origin must order by position, not by wrapped bit pattern.
    """
    positions = torch.tensor([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result = morton_sort_order(positions, voxel_size=2.0)

    ordered_x = positions[result.order][:, 0].tolist()
    assert ordered_x == sorted(ordered_x), ordered_x
    assert result.codes_low.min() >= 0


def test_equal_codes_keep_their_original_relative_order():
    """The sort is stable, which the track-partitioned rigid order depends on to keep Morton inside a track."""
    # All rows land in one voxel at this size, so every code ties and only stability decides the result.
    # 64 rows because torch falls back to an insertion sort below about that, which is stable by accident.
    n = 64
    positions = torch.tensor([[0.01 * i, 0.0, 0.0] for i in range(n)])
    result = morton_sort_order(positions, voxel_size=100.0)

    assert result.codes_low.unique().numel() == 1
    assert result.order.tolist() == list(range(n))
