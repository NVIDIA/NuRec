# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

import nurec.models.gaussians  # noqa: F401  registers opt-in Gaussian variants
import nurec.models.gaussians.collect as collect

from nurec.datasets.tracks import CuboidTracks
from nurec.models.gaussians.gaussians_model import BaseGaussianModel, DeformableGaussianModel, RigidGaussianModel
from nurec.models.gaussians.interpolated_table_gaussians import (
    InterpolatedDeformationTable,
    InterpolatedTableDeformableGaussianModel,
)
from nurec.models.gaussians.temporal_deformation_storage import (
    FrameMajorRaggedStorage,
)
from nurec.utils.batch import RenderingData


def _make_table() -> InterpolatedDeformationTable:
    return InterpolatedDeformationTable(
        reference_positions=torch.zeros(4, 3),
        gaussian_cuboid_ids=torch.tensor([0, 0, 1, 1]),
        track_timestamp_ranges_us=torch.tensor([[0, 200_000], [100_000, 300_000]]),
        start_timestamp_us=0,
        end_timestamp_us=300_000,
        frame_period_us=100_000,
        interpolation="linear_slerp",
        storage_backend="frame_major",
    )


def _logical_values(column_index: int, stable_rows: torch.Tensor) -> torch.Tensor:
    values = torch.zeros((stable_rows.numel(), 7))
    values[:, :3] = stable_rows.float().unsqueeze(1) + torch.tensor([0.0, 1.0, 2.0]) + column_index
    values[:, 5] = 0.05 * (column_index + 1)
    return values


def _fill_table(table: InterpolatedDeformationTable) -> None:
    with torch.no_grad():
        for column_index in range(table.num_frames):
            column = table.storage.gather_compressed_column(column_index)
            column.values.copy_(_logical_values(column_index, column.row_indices))


def test_existing_and_table_models_are_separate_config_selected_variants() -> None:
    assert BaseGaussianModel.GAUSSIANS_VARIANTS["deformable-gaussians"] is DeformableGaussianModel
    assert (
        BaseGaussianModel.GAUSSIANS_VARIANTS["interpolated-table-deformable-gaussians"]
        is InterpolatedTableDeformableGaussianModel
    )


def test_empty_tracks_create_an_empty_deformation_table() -> None:
    tracks = CuboidTracks.Factory.empty(device=torch.device("cpu"))
    ranges = InterpolatedTableDeformableGaussianModel._track_timestamp_ranges_us(tracks)
    assert ranges.shape == (0, 2)
    assert ranges.dtype == torch.int64
    assert ranges.device == tracks.device

    model = object.__new__(InterpolatedTableDeformableGaussianModel)
    object.__setattr__(model, "trainer_config", SimpleNamespace(world_size=1))
    object.__setattr__(model, "cuboid_tracks", tracks)
    object.__setattr__(model, "start_end_timestamp_us", None)
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(
            table=SimpleNamespace(
                frame_period_us=100_000,
                interpolation="linear_slerp",
                storage="frame_major",
            )
        ),
    )

    table = model._make_deformation_table(
        reference_positions=torch.empty((0, 3)),
        gaussian_cuboid_ids=torch.empty((0,), dtype=torch.int32),
        allocate_values=True,
    )

    assert table.num_gaussians == 0
    assert table.num_frames == 1
    assert table.num_active_slots == 0
    assert table.start_timestamp_us == table.end_timestamp_us == 0


def test_scalar_and_per_row_nearest_knot_round_half_up_consistently() -> None:
    table = _make_table()

    assert table.frame_index(50_000) == 1
    torch.testing.assert_close(table.row_frame_indices(torch.tensor([50_000])), torch.tensor([1]))


def test_table_model_uses_each_gaussians_resolved_track_timestamp() -> None:
    gaussian_track_ids = torch.tensor([1, 0, 1], dtype=torch.int32)
    base_layer_data = collect.LayerDataRigid(
        positions=torch.zeros(3, 3),
        rotations=torch.zeros(3, 4),
        scales=torch.zeros(3, 3),
        densities=torch.zeros(3, 1),
        extra_signal=torch.zeros(3, 0),
        camera_extra_signal=torch.zeros(3, 0),
        lidar_extra_signal=torch.zeros(3, 0),
        features_albedo=torch.zeros(3, 3),
        features_specular=torch.zeros(3, 0),
        embed_data=None,
        poses=torch.zeros(2, 7),
        keep_mask=torch.ones(2, dtype=torch.bool),
        tracks_ids=gaussian_track_ids,
    )
    context = BaseGaussianModel.CollectionContext(
        # This code path consumes only additional_data; keep the fixture small
        # while making that deliberate test double visible to the type checker.
        rendering_data=cast(RenderingData, SimpleNamespace()),
        is_training_batch=False,
        tracks_edit=None,
    )
    resolved_track_timestamps = torch.tensor([1120, 1480], dtype=torch.int64)
    context.additional_data["interpolated_track_timestamps"] = (resolved_track_timestamps, False)
    captured_timestamps: list[torch.Tensor] = []

    class _Table:
        num_gaussians = 3

        def __call__(self, timestamp_us, *args, **kwargs):
            captured_timestamps.append(timestamp_us)
            return {
                "positions": torch.zeros(3, 3),
                "rotations": torch.zeros(3, 4),
            }

    model = object.__new__(InterpolatedTableDeformableGaussianModel)
    object.__setattr__(model, "use_deformation", True)
    object.__setattr__(model, "deformation_table", _Table())
    object.__setattr__(model, "config", SimpleNamespace(table=SimpleNamespace(position_smoothness=True)))

    with patch.object(RigidGaussianModel, "get_layer_data", return_value=base_layer_data):
        layer_data = InterpolatedTableDeformableGaussianModel.get_layer_data(
            model,
            context,
            {"deformation_table_global_row_ids": torch.arange(3)},
        )

    assert len(captured_timestamps) == 1
    torch.testing.assert_close(captured_timestamps[0], torch.tensor([1480, 1120, 1480], dtype=torch.int64))
    assert layer_data.deform_positions is not None
    assert layer_data.deform_rotations is not None


def test_frame_major_gather_returns_the_existing_column_parameter() -> None:
    table = _make_table()
    assert isinstance(table.storage, FrameMajorRaggedStorage)
    column = table.storage.gather_compressed_column(1)
    assert column.values is next(table.storage.columns["1"].parameters())
    torch.testing.assert_close(column.row_indices, torch.arange(4))


def test_frozen_rows_clear_adam_momentum_before_each_step() -> None:
    """A frozen actor stays fixed even when restored Adam moments are nonzero."""

    table = _make_table()
    assert isinstance(table.storage, FrameMajorRaggedStorage)
    values = next(table.storage.columns["1"].parameters())
    optimizer = torch.optim.Adam([values], lr=0.1)

    # Populate Adam state for every row, then freeze actor 0 (rows 0 and 1).
    values.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    table.storage.register_frozen_groups([0], [cast(Any, {"optimizer": optimizer})])
    frozen_values = values[:2].detach().clone()

    # Simulate optimizer-state restoration after the freeze was registered.
    state = optimizer.state[values]
    state["exp_avg"][:2].fill_(1.0)
    state["exp_avg_sq"][:2].fill_(1.0)
    trainable_values = values[2:].detach().clone()

    values.sum().backward()
    optimizer.step()

    torch.testing.assert_close(values[:2], frozen_values)
    assert not torch.equal(values[2:], trainable_values)
    torch.testing.assert_close(state["exp_avg"][:2], torch.zeros_like(state["exp_avg"][:2]))
    torch.testing.assert_close(state["exp_avg_sq"][:2], torch.zeros_like(state["exp_avg_sq"][:2]))


def test_table_track_ply_override_ignores_tracks_owned_by_other_layers() -> None:
    """An unrelated PLY replacement must not be rejected by the table layer."""

    model = object.__new__(InterpolatedTableDeformableGaussianModel)
    torch.nn.Module.__init__(model)
    model.cuboid_tracks = CuboidTracks.Factory.empty(device=torch.device("cpu"))

    assert not model.track_ply_override("vehicle", cast(Any, object()), torch.eye(4), torch.zeros(3))

    model.cuboid_tracks = cast(Any, SimpleNamespace(tracks_id=["vehicle"]))
    with pytest.raises(NotImplementedError, match="fixed-size deformation tables"):
        model.track_ply_override("vehicle", cast(Any, object()), torch.eye(4), torch.zeros(3))


def test_resolved_per_actor_timestamps_select_different_control_intervals() -> None:
    table = _make_table()
    _fill_table(table)
    reference = torch.zeros(4, 3)
    stable_rows = torch.arange(4)
    timestamps = torch.tensor([50_000, 50_000, 250_000, 250_000], dtype=torch.int64)

    resolved = table(timestamps, stable_rows, reference)
    first_actor = table(50_000, stable_rows, reference)
    second_actor = table(250_000, stable_rows, reference)

    torch.testing.assert_close(resolved["positions"][:2], first_actor["positions"][:2])
    torch.testing.assert_close(resolved["rotations"][:2], first_actor["rotations"][:2])
    torch.testing.assert_close(resolved["positions"][2:], second_actor["positions"][2:])
    torch.testing.assert_close(resolved["rotations"][2:], second_actor["rotations"][2:])


def test_per_actor_timestamp_curvature_uses_each_rows_nearest_knot() -> None:
    table = _make_table()
    _fill_table(table)
    reference = torch.zeros(4, 3)
    stable_rows = torch.arange(4)

    per_actor_curvature, per_actor_valid = table.position_curvature(
        torch.tensor([100_000, 100_000, 200_000, 200_000], dtype=torch.int64),
        stable_rows,
        reference,
    )
    first_curvature, first_valid = table.position_curvature(100_000, stable_rows, reference)
    second_curvature, second_valid = table.position_curvature(200_000, stable_rows, reference)

    torch.testing.assert_close(per_actor_curvature[:2], first_curvature[:2])
    torch.testing.assert_close(per_actor_valid[:2], first_valid[:2])
    torch.testing.assert_close(per_actor_curvature[2:], second_curvature[2:])
    torch.testing.assert_close(per_actor_valid[2:], second_valid[2:])


def test_filtered_rows_preserve_stable_gaussian_identity() -> None:
    table = _make_table()
    _fill_table(table)
    result = table(
        150_000,
        torch.tensor([0, 3]),
        torch.zeros(2, 3),
        rows_were_filtered=True,
    )
    expected = torch.stack(
        (
            _logical_values(1, torch.tensor([0]))[0, :3] + 0.5,
            _logical_values(1, torch.tensor([3]))[0, :3] + 0.5,
        )
    )
    torch.testing.assert_close(result["positions"], expected)


def test_position_curvature_is_zero_for_constant_velocity() -> None:
    table = InterpolatedDeformationTable(
        reference_positions=torch.zeros(1, 3),
        gaussian_cuboid_ids=torch.tensor([0]),
        track_timestamp_ranges_us=torch.tensor([[0, 200_000]]),
        start_timestamp_us=0,
        end_timestamp_us=200_000,
        frame_period_us=100_000,
        interpolation="linear_slerp",
        storage_backend="frame_major",
    )
    _fill_table(table)

    curvature, valid = table.position_curvature(100_000, torch.tensor([0]), torch.zeros(1, 3))
    torch.testing.assert_close(curvature, torch.zeros_like(curvature))
    torch.testing.assert_close(valid, torch.ones_like(valid))


def test_mcmc_copies_full_same_actor_trajectory_into_recycled_slot() -> None:
    table = _make_table()
    _fill_table(table)
    # Old rows 1 and 3 are recycled, receiving trajectories from rows 0 and 2.
    table.apply_mcmc_relocation(
        source_row_map=torch.tensor([0, 2, 0, 2]),
        retained_count=2,
        recycled_row_indices=torch.tensor([1, 3]),
        optimizers=[],
    )
    torch.testing.assert_close(table.global_row_ids, torch.tensor([0, 2, 1, 3]))

    result = table(100_000, table.global_row_ids, torch.zeros(4, 3))
    torch.testing.assert_close(result["positions"][2], result["positions"][0])
    torch.testing.assert_close(result["positions"][3], result["positions"][1])
    torch.testing.assert_close(result["rotations"][2], result["rotations"][0])
    torch.testing.assert_close(result["rotations"][3], result["rotations"][1])


def test_morton_permutation_only_changes_current_to_stable_row_mapping() -> None:
    table = _make_table()
    _fill_table(table)
    original_column = table.storage.gather_compressed_column(1).values.detach().clone()

    table.apply_row_permutation(torch.tensor([2, 3, 0, 1]))

    torch.testing.assert_close(table.global_row_ids, torch.tensor([2, 3, 0, 1]))
    torch.testing.assert_close(table.storage.gather_compressed_column(1).values, original_column)
    result = table(100_000, table.global_row_ids, torch.zeros(4, 3))
    torch.testing.assert_close(result["positions"][0], original_column[2, :3])
    torch.testing.assert_close(result["positions"][2], original_column[0, :3])


def test_state_dict_restores_relocated_stable_row_mapping() -> None:
    trained = _make_table()
    _fill_table(trained)
    trained.apply_mcmc_relocation(
        source_row_map=torch.tensor([0, 2, 0, 2]),
        retained_count=2,
        recycled_row_indices=torch.tensor([1, 3]),
        optimizers=[],
    )

    loaded = InterpolatedDeformationTable(
        reference_positions=torch.zeros(4, 3),
        gaussian_cuboid_ids=torch.tensor([0, 0, 1, 1]),
        track_timestamp_ranges_us=torch.tensor([[0, 200_000], [100_000, 300_000]]),
        start_timestamp_us=0,
        end_timestamp_us=300_000,
        frame_period_us=100_000,
        interpolation="linear_slerp",
        storage_backend="frame_major",
        allocate_values=False,
    )
    loaded.load_state_dict(trained.state_dict(), assign=True)

    torch.testing.assert_close(loaded.global_row_ids, trained.global_row_ids)
    trained_result = trained(150_000, trained.global_row_ids, torch.zeros(4, 3))
    loaded_result = loaded(150_000, loaded.global_row_ids, torch.zeros(4, 3))
    torch.testing.assert_close(loaded_result["positions"], trained_result["positions"])
    torch.testing.assert_close(loaded_result["rotations"], trained_result["rotations"])
