# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from nurec.lidar.firing_patterns.schema import grid_from_offsets


def _grid_offsets(n_columns: int, n_rows: int, step_us: int = 49) -> np.ndarray:
    """Times for a unit that fires a whole column at once, one column every ``step_us``."""
    return np.repeat(np.arange(n_columns, dtype=np.int64) * step_us, n_rows)


class TestGridFromOffsets:
    """The grid is read off the firing times, so it is only claimed when they show one."""

    def test_recovers_the_grid_from_column_wise_firing(self):
        assert grid_from_offsets(_grid_offsets(1214, 360)) == (1214, 360)

    def test_recovers_a_small_grid(self):
        assert grid_from_offsets(_grid_offsets(4, 3)) == (4, 3)

    def test_no_grid_when_every_ray_shares_one_instant(self):
        # Nothing distinguishes columns, so there is no addressing to offer.
        assert grid_from_offsets(np.zeros(24, dtype=np.int64)) is None

    def test_no_grid_when_the_run_length_does_not_divide_the_rays(self):
        offsets = np.concatenate([_grid_offsets(3, 5), np.array([999], dtype=np.int64)])
        assert grid_from_offsets(offsets) is None

    def test_no_grid_when_a_column_spans_several_instants(self):
        # A ragged column means index // n_rows is not the column, so refuse rather than
        # hand back an addressing that silently mis-maps the rest of the sweep.
        offsets = _grid_offsets(4, 3)
        offsets[7] += 1
        assert grid_from_offsets(offsets) is None

    def test_grid_survives_columns_that_arrive_out_of_time_order(self):
        # Source order is authoritative over timestamps, so a reordered frame still addresses
        # correctly and keeps its grid.
        offsets = _grid_offsets(4, 3)
        offsets[6:9], offsets[9:12] = offsets[9:12].copy(), offsets[6:9].copy()
        assert grid_from_offsets(offsets) == (4, 3)

    def test_grid_survives_two_columns_sharing_an_instant(self):
        offsets = _grid_offsets(4, 3)
        offsets[6:9] = offsets[3:6]  # third column fires with the second
        assert grid_from_offsets(offsets) == (4, 3)

    def test_no_grid_when_every_ray_fires_at_its_own_instant(self):
        # column_id would be the index and row_id always zero: no addressing to offer.
        assert grid_from_offsets(np.arange(24, dtype=np.int64)) is None

    def test_no_grid_for_a_pattern_whose_order_was_destroyed(self):
        # What a clustering merge produces: the right times, in label order rather than
        # sweep order. It must not be mistaken for a grid.
        rng = np.random.default_rng(0)
        assert grid_from_offsets(rng.permutation(_grid_offsets(8, 6))) is None
