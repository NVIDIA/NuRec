# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import patch
from weakref import ReferenceType, ref

import pytest
import safetensors.torch
import torch

from apex.optimizers import FusedAdam

import nurec.models.gaussians  # noqa: F401  registers opt-in Gaussian variants
import nurec.models.gaussians.collect as collect

from nurec.compression import gaussian_codec
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


@pytest.mark.parametrize("kept", [[0, 1, 2, 3], [0, 1], [1, 3], []], ids=["all", "partial", "one_actor", "empty"])
@pytest.mark.parametrize("level", [0, 9])
def test_compact_table_preserves_surviving_motion(tmp_path, kept, level) -> None:
    table = _make_table()
    _fill_table(table)
    table.apply_row_permutation(torch.tensor([2, 0, 3, 1]))
    prefix = "model.gaussians_nodes.dynamic_deformables."
    positions = torch.tensor([[3.0, 0, 0], [2.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0]])
    densities = torch.full((4, 1), -10.0)
    densities[kept] = 10.0
    state = {
        prefix + "positions": positions,
        prefix + "densities": densities,
        prefix + "gaussian_cuboid_ids": torch.tensor([1, 0, 1, 0]),
        prefix + "camera_extra_signal": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        prefix + "aabb_extent": torch.ones(2, 3),
        prefix + "_extra_state": {"use_deformation": True},
        **{prefix + "deformation_table." + key: value for key, value in table.state_dict().items()},
    }
    original = deepcopy(state)
    tensors, metadata = gaussian_codec.pack(
        state, compression_level=level, preserve_extra_signal_tensors=("camera_extra_signal",) if level == 0 else ()
    )
    path = tmp_path / "compact.safetensors"
    safetensors.torch.save_file(tensors, path, metadata=metadata)
    with safetensors.safe_open(path, framework="pt") as payload:
        restored = gaussian_codec.unpack(safetensors.torch.load_file(path), payload.metadata())
    for key, expected in original.items():
        if isinstance(expected, torch.Tensor):
            assert torch.equal(state[key], expected)
        else:
            assert state[key] == expected
    assert restored[prefix + "_extra_state"] == {"use_deformation": True}
    assert torch.equal(restored[prefix + "aabb_extent"], state[prefix + "aabb_extent"])
    # Integral, separated positions identify original rows even at level 9.
    source_rows = (3 - restored[prefix + "positions"][:, 0]).round().long()
    assert sorted(source_rows.tolist()) == kept
    signal = restored[prefix + "camera_extra_signal"]
    expected_signal = state[prefix + "camera_extra_signal"][source_rows] if level == 0 else torch.zeros_like(signal)
    assert torch.equal(signal, expected_signal)
    restored_table_state = {
        key.removeprefix(prefix + "deformation_table."): value
        for key, value in restored.items()
        if key.startswith(prefix + "deformation_table.")
    }
    stable_keep = torch.zeros(4, dtype=torch.bool)
    stable_keep[table.global_row_ids[kept]] = True
    for column in range(table.num_frames):
        original_column = table.storage.gather_compressed_column(column)
        expected = original_column.values[stable_keep[original_column.row_indices]]
        key = f"storage.columns.{column}.values"
        if len(expected):
            assert torch.equal(restored_table_state[key], expected)
        else:
            assert key not in restored_table_state
    for lazy in (False, True):
        loaded = InterpolatedDeformationTable(
            reference_positions=restored[prefix + "positions"],
            gaussian_cuboid_ids=restored[prefix + "gaussian_cuboid_ids"],
            track_timestamp_ranges_us=torch.tensor([[0, 200_000], [100_000, 300_000]]),
            start_timestamp_us=0,
            end_timestamp_us=300_000,
            frame_period_us=100_000,
            interpolation="linear_slerp",
            storage_backend="frame_major",
            allocate_values=not lazy,
        )
        loaded.load_state_dict(restored_table_state, strict=True, assign=lazy)
        for timestamp in (-1, 0, 50_000, 100_000, 150_000, 200_000, 300_000, 300_001):
            expected = table(timestamp, table.global_row_ids, positions)
            actual = loaded(timestamp, loaded.global_row_ids, positions[source_rows])
            for name in ("positions", "rotations"):
                torch.testing.assert_close(actual[name], expected[name][source_rows], rtol=0, atol=0)


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


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("position_smoothness", [False, True])
@pytest.mark.parametrize("kept_rows", [None, [2, 0]], ids=["full", "filtered"])
def test_table_model_passes_resolved_timestamps_to_track_evaluation(
    training: bool, position_smoothness: bool, kept_rows: list[int] | None
) -> None:
    selected_rows = torch.arange(3) if kept_rows is None else torch.tensor(kept_rows)
    gaussian_track_ids = torch.tensor([1, 0, 1], dtype=torch.int32)[selected_rows]
    gathered_row_ids = torch.tensor([2, 0, 1])[selected_rows]
    num_rows = selected_rows.numel()
    compute_curvature = training and position_smoothness
    lifetime_mask = torch.tensor([[1.0], [1.0], [0.0]])[selected_rows]
    base_layer_data = collect.LayerDataRigid(
        positions=torch.zeros(num_rows, 3),
        rotations=torch.zeros(num_rows, 4),
        scales=torch.zeros(num_rows, 3),
        densities=torch.zeros(num_rows, 1),
        extra_signal=torch.zeros(num_rows, 0),
        camera_extra_signal=torch.zeros(num_rows, 0),
        lidar_extra_signal=torch.zeros(num_rows, 0),
        features_albedo=torch.zeros(num_rows, 3),
        features_specular=torch.zeros(num_rows, 0),
        embed_data=None,
        poses=torch.zeros(2, 7),
        keep_mask=torch.ones(2, dtype=torch.bool),
        tracks_ids=gaussian_track_ids,
    )
    context = BaseGaussianModel.CollectionContext(
        # This code path consumes only additional_data; keep the fixture small
        # while making that deliberate test double visible to the type checker.
        rendering_data=cast(RenderingData, SimpleNamespace()),
        is_training_batch=training,
        tracks_edit=None,
        kept_rows=None if kept_rows is None else selected_rows,
    )
    resolved_track_timestamps = torch.tensor([1120, 1480], dtype=torch.int64)
    context.additional_data["interpolated_track_timestamps"] = (resolved_track_timestamps, False)
    context.additional_data["interpolated_mask"] = (torch.tensor([False, True]), False)
    captured_timestamps: list[torch.Tensor] = []

    class _Table:
        num_gaussians = 3

        def evaluate_tracks(self, timestamp_us, *args, **kwargs):
            captured_timestamps.append(timestamp_us)
            assert kwargs["compute_curvature"] is compute_curvature
            assert kwargs["rows_were_filtered"] is (kept_rows is not None)
            torch.testing.assert_close(args[0], gathered_row_ids)
            assert args[1] is base_layer_data.positions
            return {
                "positions": torch.zeros(num_rows, 3),
                "rotations": torch.zeros(num_rows, 4),
            }, (torch.ones(num_rows, 3), lifetime_mask) if compute_curvature else None

    model = object.__new__(InterpolatedTableDeformableGaussianModel)
    object.__setattr__(model, "use_deformation", True)
    object.__setattr__(model, "deformation_table", _Table())
    object.__setattr__(model, "config", SimpleNamespace(table=SimpleNamespace(position_smoothness=position_smoothness)))

    with patch.object(RigidGaussianModel, "get_layer_data", return_value=base_layer_data):
        layer_data = InterpolatedTableDeformableGaussianModel.get_layer_data(
            model,
            context,
            {"deformation_table_global_row_ids": gathered_row_ids},
        )

    assert len(captured_timestamps) == 1
    torch.testing.assert_close(captured_timestamps[0], resolved_track_timestamps)
    assert layer_data.deform_positions is not None
    assert layer_data.deform_rotations is not None
    if compute_curvature:
        torch.testing.assert_close(context.additional_data["deform_smoothness"][0], torch.ones(num_rows, 3))
        torch.testing.assert_close(
            context.additional_data["deform_smoothness_mask"][0], torch.tensor([[1.0], [0.0], [0.0]])[selected_rows]
        )
    else:
        assert "deform_smoothness" not in context.additional_data
        assert "deform_smoothness_mask" not in context.additional_data


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


@pytest.mark.parametrize("interpolation", ["nearest", "linear_slerp"])
def test_resolved_per_actor_timestamps_select_different_control_intervals(
    interpolation: Literal["nearest", "linear_slerp"],
) -> None:
    table = _make_table()
    table.interpolation = interpolation
    _fill_table(table)
    reference = torch.zeros(4, 3)
    stable_rows = torch.arange(4)
    timestamps = torch.tensor([25_000, 25_000, 275_000, 275_000], dtype=torch.int64)

    resolved = table(timestamps, stable_rows, reference)
    first_actor = table(25_000, stable_rows, reference)
    second_actor = table(275_000, stable_rows, reference)

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


@pytest.mark.parametrize("checkpoint_dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize(
    ("allocate_values", "assign"),
    [(True, False), (True, True), (False, True)],
    ids=["copy", "assign", "lazy_assign"],
)
def test_historical_frame_major_checkpoint_loads_without_migration(
    checkpoint_dtype: torch.dtype, allocate_values: bool, assign: bool
) -> None:
    """Load the original frame-major schema, not a current-writer round trip."""
    # Frozen state and output captured from c910a0902f2fd88fc8578fddd13a1ffe3ae0a95f.
    # Execute that revision's InterpolatedDeformationTable, storage and quat_slerp
    # to regenerate; no Git or old source is needed when this test runs in CI.
    # Writer groups [0, 2, 0], ranges below, permutation [1, 2, 0]. For stable row r
    # and column c: xyz = [(r+1)*(c+1)**2/8, -(r+1)*(c+1)/4, (r+1)/2],
    # rotation residual = [0, 0, (c+1)/8, 0]. All stored values are exact in fp16.
    # Column 3 is empty; group 1 has no rows. Never derive this fixture from the
    # current table's state_dict(), which could hide a backwards-incompatible key change.
    checkpoint = {
        "global_row_ids": torch.tensor([1, 2, 0], dtype=torch.int64),
        "storage.row_start_columns": torch.tensor([0, 4, 0], dtype=torch.int64),
        "storage.row_end_columns": torch.tensor([2, 6, 2], dtype=torch.int64),
        "storage.row_group_ids": torch.tensor([0, 2, 0], dtype=torch.int64),
        "storage.columns.0.values": torch.tensor(
            [[0.125, -0.25, 0.5, 0, 0, 0.125, 0], [0.375, -0.75, 1.5, 0, 0, 0.125, 0]], dtype=checkpoint_dtype
        ),
        "storage.columns.1.values": torch.tensor(
            [[0.5, -0.5, 0.5, 0, 0, 0.25, 0], [1.5, -1.5, 1.5, 0, 0, 0.25, 0]], dtype=checkpoint_dtype
        ),
        "storage.columns.2.values": torch.tensor(
            [[1.125, -0.75, 0.5, 0, 0, 0.375, 0], [3.375, -2.25, 1.5, 0, 0, 0.375, 0]], dtype=checkpoint_dtype
        ),
        "storage.columns.4.values": torch.tensor([[6.25, -2.5, 1, 0, 0, 0.625, 0]], dtype=checkpoint_dtype),
        "storage.columns.5.values": torch.tensor([[9, -3, 1, 0, 0, 0.75, 0]], dtype=checkpoint_dtype),
        "storage.columns.6.values": torch.tensor([[12.25, -3.5, 1, 0, 0, 0.875, 0]], dtype=checkpoint_dtype),
    }
    positions = torch.zeros(3, 3)
    current_cuboid_ids = torch.tensor([2, 0, 0])
    loaded = InterpolatedDeformationTable(
        reference_positions=positions,
        gaussian_cuboid_ids=current_cuboid_ids,
        track_timestamp_ranges_us=torch.tensor([[0, 200_000], [0, 0], [400_000, 600_000]]),
        start_timestamp_us=0,
        end_timestamp_us=600_000,
        frame_period_us=100_000,
        interpolation="linear_slerp",
        storage_backend="frame_major",
        allocate_values=allocate_values,
    )
    # Exercise recursive strict loading, including assignment into lazy columns.
    root = torch.nn.ModuleDict({"deformation_table": loaded})
    with BytesIO() as stream:
        torch.save({f"deformation_table.{key}": value for key, value in checkpoint.items()}, stream)
        stream.seek(0)
        root.load_state_dict(torch.load(stream, weights_only=True), strict=True, assign=assign)

    restored = loaded.state_dict()
    assert restored.keys() == checkpoint.keys()  # Derived mappings must remain nonpersistent.
    for key, expected in checkpoint.items():
        expected_dtype = torch.float32 if expected.is_floating_point() and not assign else expected.dtype
        assert restored[key].dtype == expected_dtype
        torch.testing.assert_close(restored[key], expected.to(expected_dtype), rtol=0, atol=0)
    torch.testing.assert_close(loaded.current_row_for_global_id, torch.tensor([2, 0, 1]))
    torch.testing.assert_close(loaded.represented_track_ids, torch.tensor([0, 2]))
    torch.testing.assert_close(loaded.track_index_for_global_row, torch.tensor([0, 1, 0]))
    torch.testing.assert_close(loaded.storage.row_group_ids[loaded.global_row_ids], current_cuboid_ids)
    for column_index, rows in enumerate(([0, 2], [0, 2], [0, 2], [], [1], [1], [1])):
        column = loaded.storage.gather_compressed_column(column_index)
        torch.testing.assert_close(column.row_indices, torch.tensor(rows, dtype=torch.int64))
        if rows:
            torch.testing.assert_close(
                column.values,
                checkpoint[f"storage.columns.{column_index}.values"].to(column.values.dtype),
                rtol=0,
                atol=0,
            )
        else:
            assert column.values.shape == (0, 7)

    # Training resume upcasts fp16 values. Evaluate both load modes in fp32
    # against the old writer's output, including the new production track path.
    loaded.float()
    track_times = torch.tensor([125_000, 0, 525_000])
    expected_positions = torch.tensor([[9.8125, -3.125, 1], [1.96875, -1.6875, 1.5], [0.65625, -0.5625, 0.5]])
    expected_rotations = torch.tensor(
        [
            [0, 0, 0.6149585247039795, -0.21144044399261475],
            [0, 0, 0.2700323760509491, -0.03714877367019653],
            [0, 0, 0.2700323760509491, -0.03714877367019653],
        ]
    )
    expected_curvature = torch.tensor([[0.25, 0, 0], [0.375, 0, 0], [0.125, 0, 0]])
    direct_output = loaded(track_times[current_cuboid_ids], loaded.global_row_ids, positions)
    output, smoothness = loaded.evaluate_tracks(track_times, loaded.global_row_ids, positions, compute_curvature=True)
    for result in (direct_output, output):
        torch.testing.assert_close(result["positions"], expected_positions, rtol=0, atol=0)
        torch.testing.assert_close(result["rotations"], expected_rotations, rtol=1e-6, atol=1e-7)
    assert smoothness is not None
    curvature, mask = smoothness
    torch.testing.assert_close(curvature, expected_curvature, rtol=0, atol=0)
    torch.testing.assert_close(mask, torch.ones(3, 1), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("interpolation", ["nearest", "linear_slerp"])
def test_batched_columns_preserve_values_gradients_and_adam_state(device, filtered, interpolation) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    table = _make_table()
    assert isinstance(table.storage, FrameMajorRaggedStorage)
    _fill_table(table)
    with torch.no_grad():
        next(table.storage.columns["2"].parameters())[:, 3:] = torch.tensor([0.2, -0.3, 0.4, -2.0])
    table = table.to(device)
    table.interpolation = interpolation
    table.apply_row_permutation(torch.tensor([2, 0, 3, 1], device=device))
    reference = _make_table().to(device)
    reference.interpolation = interpolation
    reference.load_state_dict(table.state_dict())

    # Keep the pre-MR (cecd61049e) serial algorithm as an independent oracle.
    def serial_columns(
        column_indices: tuple[torch.Tensor, ...],
        gathered_global_row_ids: torch.Tensor,
        reference_positions: torch.Tensor,
        *,
        rows_were_filtered: bool,
    ) -> tuple[torch.Tensor, ...]:
        results = []
        for indices in column_indices:
            result = reference_positions.new_zeros((reference_positions.shape[0], 7))
            for column_index in torch.unique(indices.detach()).cpu().tolist():
                values = reference._dense_residuals_at_frame(
                    column_index,
                    gathered_global_row_ids,
                    reference_positions,
                    rows_were_filtered=rows_were_filtered,
                )
                result = torch.where((indices == column_index).unsqueeze(1), values, result)
            results.append(result)
        return tuple(results)

    optimizers = [torch.optim.Adam(model.parameters(), lr=0.01) for model in (table, reference)]
    # Successive calls cover exact knots, mixed actor timestamps, clamping,
    # and reuse after optimizer updates without retaining an old autograd graph.
    timestamps = ([100_000] * 4, [150_000, 0, 250_000, 200_000], [-1, 400_000, 299_999, 1])
    with patch.object(reference, "_dense_residuals_at_row_columns", side_effect=serial_columns) as serial:
        for step, times in enumerate(timestamps):
            row_ids = table.global_row_ids[[2, 1]] if filtered else table.global_row_ids
            times_tensor = torch.tensor(times, dtype=torch.int64, device=device)
            if filtered:
                times_tensor = times_tensor[[2, 1]]
            positions = torch.zeros((row_ids.numel(), 3), device=device)
            outputs = []
            for model, optimizer in zip((table, reference), optimizers, strict=True):
                optimizer.zero_grad(set_to_none=True)
                output = model(times_tensor, row_ids, positions, rows_were_filtered=filtered)
                loss = output["positions"].square().sum() + output["rotations"].square().sum()
                if step:
                    curvature, mask = model.position_curvature(
                        times_tensor, row_ids, positions, rows_were_filtered=filtered
                    )
                    output.update(curvature=curvature, mask=mask)
                    loss = loss + (curvature * mask).square().sum()
                loss.backward()
                outputs.append(output)
            for key in outputs[0]:
                torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
            for candidate, baseline in zip(table.parameters(), reference.parameters(), strict=True):
                assert (candidate.grad is None) == (baseline.grad is None)
                if candidate.grad is not None:
                    torch.testing.assert_close(candidate.grad, baseline.grad)
            for optimizer in optimizers:
                optimizer.step()
            for candidate, baseline in zip(table.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(candidate, baseline)
                candidate_state, baseline_state = optimizers[0].state.get(candidate), optimizers[1].state.get(baseline)
                assert (candidate_state is None) == (baseline_state is None)
                if candidate_state is not None:
                    assert baseline_state is not None
                    for key in candidate_state:
                        torch.testing.assert_close(candidate_state[key], baseline_state[key])
            if step == 0:
                assert isinstance(table.storage, FrameMajorRaggedStorage)
                unused = next(table.storage.columns["2"].parameters())
                assert unused.grad is None
                assert unused not in optimizers[0].state
        assert serial.call_count == 5  # Step 0: forward; steps 1-2: forward + curvature.


def test_per_row_forward_and_curvature_group_column_requests() -> None:
    table = _make_table()
    timestamps = torch.tensor([25_000, 125_000, 175_000, 275_000], dtype=torch.int64)
    positions = torch.zeros(4, 3)
    with patch.object(table, "_dense_residuals_at_row_columns", wraps=table._dense_residuals_at_row_columns) as gather:
        table(timestamps, table.global_row_ids, positions)
        gather.assert_called_once()
        assert gather.call_args is not None
        assert isinstance(gather.call_args.args[0], tuple)
        assert len(gather.call_args.args[0]) == 2

        gather.reset_mock()
        table.position_curvature(timestamps, table.global_row_ids, positions)
        gather.assert_called_once()
        assert gather.call_args is not None
        assert isinstance(gather.call_args.args[0], tuple)
        assert len(gather.call_args.args[0]) == 3


def test_batched_columns_materialize_each_column_once_and_keep_separate_dependencies() -> None:
    table = _make_table()
    assert isinstance(table.storage, FrameMajorRaggedStorage)
    _fill_table(table)
    with patch.object(table, "_dense_residuals_at_frame", wraps=table._dense_residuals_at_frame) as gather:
        first, _ = table._dense_residuals_at_row_columns(
            (torch.tensor([1, 1, 1, 1]), torch.tensor([1, 2, 1, 2])),
            table.global_row_ids,
            torch.zeros(4, 3),
            rows_were_filtered=False,
        )
        assert gather.call_count == 2
    first.sum().backward()
    assert next(table.storage.columns["1"].parameters()).grad is not None
    assert next(table.storage.columns["2"].parameters()).grad is None


def test_batched_columns_release_dense_columns_after_their_last_request() -> None:
    table = _make_table()
    _fill_table(table)
    dense_columns: list[ReferenceType[torch.Tensor]] = []
    live_before_gather: list[int] = []
    gather_frame = table._dense_residuals_at_frame

    def track_dense_column(
        frame_index: int,
        gathered_global_row_ids: torch.Tensor,
        reference_positions: torch.Tensor,
        *,
        rows_were_filtered: bool,
    ) -> torch.Tensor:
        live_before_gather.append(sum(column() is not None for column in dense_columns))
        values = gather_frame(
            frame_index,
            gathered_global_row_ids,
            reference_positions,
            rows_were_filtered=rows_were_filtered,
        )
        dense_columns.append(ref(values))
        return values

    with patch.object(table, "_dense_residuals_at_frame", side_effect=track_dense_column) as gather:
        results = table._dense_residuals_at_row_columns(
            # Every request spans multiple columns, so no result needs to
            # retain a dense column as a singleton-request fast path.
            (torch.tensor([0, 1, 2, 3]), torch.tensor([3, 2, 1, 0]), torch.tensor([1, 2, 3, 0])),
            table.global_row_ids,
            torch.zeros(4, 3),
            rows_were_filtered=False,
        )
        assert gather.call_count == table.num_frames

    # The loop may retain the previous column while building the next one,
    # but must not retain every column until all requests are complete.
    assert max(live_before_gather) <= 1
    assert all(column() is None for column in dense_columns)
    torch.stack([result.sum() for result in results]).sum().backward()


@pytest.mark.parametrize("interpolation", ["nearest", "linear_slerp"])
@pytest.mark.parametrize("empty_table", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_per_row_empty_collection(interpolation, empty_table, device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    table = (
        InterpolatedDeformationTable(
            reference_positions=torch.empty(0, 3),
            gaussian_cuboid_ids=torch.empty(0, dtype=torch.int64),
            track_timestamp_ranges_us=torch.empty(0, 2, dtype=torch.int64),
            start_timestamp_us=0,
            end_timestamp_us=0,
            frame_period_us=100_000,
            interpolation=interpolation,
            storage_backend="frame_major",
        )
        if empty_table
        else _make_table()
    )
    table = table.to(device)
    table.interpolation = interpolation
    timestamps = torch.empty(0, dtype=torch.int64, device=device)
    rows = torch.empty(0, dtype=torch.int64, device=device)
    positions = torch.empty(0, 3, device=device)
    output = table(timestamps, rows, positions, rows_were_filtered=True)
    curvature, valid = table.position_curvature(timestamps, rows, positions, rows_were_filtered=True)
    assert output["positions"].shape == (0, 3)
    assert output["rotations"].shape == (0, 4)
    assert curvature.shape == (0, 3)
    assert valid.shape == (0, 1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("interpolation", ["nearest", "linear_slerp"])
def test_single_frame_table_with_rows_preserves_values_and_gradients(
    device: str, interpolation: Literal["nearest", "linear_slerp"]
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    table = InterpolatedDeformationTable(
        reference_positions=torch.zeros(2, 3),
        gaussian_cuboid_ids=torch.tensor([0, 0]),
        track_timestamp_ranges_us=torch.tensor([[0, 0]]),
        start_timestamp_us=0,
        end_timestamp_us=0,
        frame_period_us=100_000,
        interpolation=interpolation,
        storage_backend="frame_major",
    )
    _fill_table(table)
    table = table.to(device)
    table.apply_row_permutation(torch.tensor([1, 0], device=device))
    timestamps = torch.tensor([-1, 1], dtype=torch.int64, device=device)
    positions = torch.zeros(2, 3, device=device)
    output = table(timestamps, table.global_row_ids, positions)
    curvature, valid = table.position_curvature(timestamps, table.global_row_ids, positions)

    expected = _logical_values(0, torch.tensor([1, 0])).to(device)
    expected_rotations = expected[:, 3:].clone()
    if interpolation == "linear_slerp":
        expected_rotations[:, 3] += 1.0
        expected_rotations = torch.nn.functional.normalize(expected_rotations, dim=-1)
        expected_rotations[:, 3] -= 1.0
    torch.testing.assert_close(output["positions"], expected[:, :3])
    torch.testing.assert_close(output["rotations"], expected_rotations)
    torch.testing.assert_close(curvature, torch.zeros_like(curvature))
    torch.testing.assert_close(valid, torch.zeros_like(valid))

    (output["positions"].sum() + output["rotations"].square().sum() + curvature.square().sum()).backward()
    values = table.storage.gather_compressed_column(0).values
    assert values.grad is not None
    torch.testing.assert_close(values.grad[:, :3], torch.ones_like(values[:, :3]))


def test_exact_knot_preserves_existing_unused_adam_moments() -> None:
    table = _make_table()
    assert isinstance(table.storage, FrameMajorRaggedStorage)
    _fill_table(table)
    optimizer = torch.optim.Adam(table.parameters(), lr=0.1)
    unused = next(table.storage.columns["2"].parameters())
    unused.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    previous_value = unused.detach().clone()
    previous_state = {key: value.clone() for key, value in optimizer.state[unused].items()}
    output = table(torch.full((4,), 100_000, dtype=torch.int64), table.global_row_ids, torch.zeros(4, 3))
    output["positions"].sum().backward()
    assert unused.grad is None
    optimizer.step()
    torch.testing.assert_close(unused, previous_value, rtol=0, atol=0)
    for key, value in previous_state.items():
        torch.testing.assert_close(optimizer.state[unused][key], value, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("filtered", [False, True])
def test_duplicate_column_requests_support_deterministic_algorithms(device: str, filtered: bool) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    table = _make_table()
    _fill_table(table)
    with torch.no_grad():
        table.storage.gather_compressed_column(2).values[:, :3].square_()
    table = table.to(device)
    table.apply_row_permutation(torch.tensor([2, 0, 3, 1], device=device))
    row_ids = table.global_row_ids[[2, 1]] if filtered else table.global_row_ids
    timestamps = torch.tensor([175_000, 125_000, 225_000, 25_000], dtype=torch.int64, device=device)
    if filtered:
        timestamps = timestamps[[2, 1]]
    positions = torch.zeros(row_ids.numel(), 3, device=device)
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    outputs, gradients = [], []
    try:
        for deterministic in (False, True):
            torch.use_deterministic_algorithms(deterministic)
            table.zero_grad(set_to_none=True)
            output = table(timestamps, row_ids, positions, rows_were_filtered=filtered)
            curvature, mask = table.position_curvature(timestamps, row_ids, positions, rows_were_filtered=filtered)
            output.update(curvature=curvature, mask=mask)
            loss = (
                output["positions"].square().sum()
                + output["rotations"].square().sum()
                + (curvature * mask).square().sum()
            )
            loss.backward()
            outputs.append(output)
            gradients.append([None if p.grad is None else p.grad.clone() for p in table.parameters()])
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
    for normal, deterministic in zip(*gradients, strict=True):
        assert (normal is None) == (deterministic is None)
        if normal is not None:
            torch.testing.assert_close(normal, deterministic, rtol=0, atol=0)


@pytest.mark.parametrize("device,optimizer_kind", [("cpu", "adam"), ("cuda", "adam"), ("cuda", "fused_adam")])
@pytest.mark.parametrize("kept_rows", [None, [5, 1, 0], [5, 1, 3]], ids=["full", "all_tracks", "missing_track"])
@pytest.mark.parametrize("interpolation", ["nearest", "linear_slerp"])
@pytest.mark.parametrize(
    "compute_curvature,backward_output", [(False, "forward"), (True, "forward"), (True, "curvature"), (True, "both")]
)
def test_track_evaluation_matches_independent_values_gradients_and_adam(
    device: str,
    kept_rows: list[int] | None,
    interpolation: Literal["nearest", "linear_slerp"],
    compute_curvature: bool,
    backward_output: str,
    optimizer_kind: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    filtered = kept_rows is not None
    period = 100_001  # Preserve integer rounding for odd control periods.
    groups = torch.tensor([4, 1, 4, 1, 7, 4, 7, 1])
    ranges = torch.tensor([[0, 4 * period]] * 8)
    ranges[1] = torch.tensor([0, 2 * period])
    ranges[4] = torch.tensor([period, 3 * period])
    ranges[7] = torch.tensor([2 * period, 4 * period])

    def make_table() -> InterpolatedDeformationTable:
        return InterpolatedDeformationTable(
            reference_positions=torch.zeros(8, 3),
            gaussian_cuboid_ids=groups,
            track_timestamp_ranges_us=ranges,
            start_timestamp_us=0,
            end_timestamp_us=4 * period,
            frame_period_us=period,
            interpolation=interpolation,
            storage_backend="frame_major",
        ).to(device)

    table, reference = make_table(), make_table()
    with torch.no_grad():
        for index in range(table.num_frames):
            values = table.storage.gather_compressed_column(index).values
            values.copy_(torch.sin(torch.arange(values.numel(), device=device).reshape_as(values) + index) * 0.1)
        # Exercise shortest-arc interpolation and degenerate quaternion fallback.
        table.storage.gather_compressed_column(2).values[0, 3:] = torch.tensor([0, 0, 0, -1], device=device)
        table.storage.gather_compressed_column(3).values[0, 3:] = torch.tensor([0.2, -0.3, 0.4, -2], device=device)
    permutation = torch.tensor([6, 0, 3, 5, 1, 7, 2, 4], device=device)
    table.apply_row_permutation(permutation)
    current_cuboid_ids = groups.to(device)[permutation]
    torch.testing.assert_close(table.storage.row_group_ids[table.global_row_ids], current_cuboid_ids)
    reference.load_state_dict(table.state_dict())
    torch.testing.assert_close(reference.storage.row_group_ids[reference.global_row_ids], current_cuboid_ids)
    optimizer_cls = FusedAdam if optimizer_kind == "fused_adam" else torch.optim.Adam
    optimizers = [optimizer_cls(model.parameters(), lr=0.001) for model in (table, reference)]
    for model, optimizer in zip((table, reference), optimizers, strict=True):
        torch.stack([parameter.square().sum() for parameter in model.parameters()]).sum().backward()
        optimizer.step()  # Prepopulate moments, including subsequently unused columns.

    for times in (
        (period, period, period),
        (period // 2, 2 * period + 1, 3 * period),
        (-1, 5 * period, 0),
        (2 * period, period + period // 2 + 1, 3 * period - 1),
        (3 * period, 3 * period, period),
    ):
        # Tracks without rows intentionally request other columns; they must not
        # create Parameter dependencies. The final case has group 7 pre-start
        # at a stored column, which matters only when its rows are collected.
        track_times = torch.full((8,), 4 * period, dtype=torch.int64, device=device)
        track_times[torch.tensor([1, 4, 7], device=device)] = torch.tensor(times, device=device)
        rows = table.global_row_ids if kept_rows is None else table.global_row_ids[kept_rows]
        cuboid_ids = current_cuboid_ids if kept_rows is None else current_cuboid_ids[kept_rows]
        torch.testing.assert_close(table.storage.row_group_ids[rows], cuboid_ids)
        positions = torch.zeros(rows.numel(), 3, device=device)
        per_row_times = track_times[cuboid_ids]
        outputs = []
        for is_candidate, model, optimizer in zip((True, False), (table, reference), optimizers, strict=True):
            model.zero_grad(set_to_none=True)
            if is_candidate:
                output, smoothness = model.evaluate_tracks(
                    track_times, rows, positions, compute_curvature=compute_curvature, rows_were_filtered=filtered
                )
                assert (smoothness is not None) is compute_curvature
            else:
                output = model(per_row_times, rows, positions, rows_were_filtered=filtered)
                smoothness = (
                    model.position_curvature(per_row_times, rows, positions, rows_were_filtered=filtered)
                    if compute_curvature
                    else None
                )
            forward_loss = output["positions"].square().sum() + output["rotations"].square().sum()
            loss = forward_loss
            if smoothness is not None:
                curvature, mask = smoothness
                output.update(curvature=curvature, mask=mask)
                curvature_loss = (curvature * mask).square().sum()
                if backward_output == "curvature":
                    loss = curvature_loss
                elif backward_output == "both":
                    loss = forward_loss + curvature_loss
            loss.backward()
            outputs.append(output)
        for key in outputs[0]:
            torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
        for candidate, baseline in zip(table.parameters(), reference.parameters(), strict=True):
            assert (candidate.grad is None) == (baseline.grad is None)
            if candidate.grad is not None:
                torch.testing.assert_close(candidate.grad, baseline.grad)
        for optimizer in optimizers:
            optimizer.step()
        for candidate, baseline in zip(table.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(candidate, baseline)
            for key in optimizers[0].state[candidate]:
                torch.testing.assert_close(optimizers[0].state[candidate][key], optimizers[1].state[baseline][key])


@pytest.mark.parametrize("interpolation,num_requests", [("nearest", 3), ("linear_slerp", 5)])
def test_track_evaluation_discovers_once_and_materializes_shared_columns_once(
    interpolation: Literal["nearest", "linear_slerp"], num_requests: int
) -> None:
    table = _make_table()
    table.interpolation = interpolation
    with (
        patch.object(table, "_dense_residuals_at_row_columns", wraps=table._dense_residuals_at_row_columns) as discover,
        patch.object(table, "_dense_residuals_at_frame", wraps=table._dense_residuals_at_frame) as gather,
    ):
        table.evaluate_tracks(
            torch.tensor([125_000, 125_000]), table.global_row_ids, torch.zeros(4, 3), compute_curvature=True
        )
    assert discover.call_count == 1
    assert discover.call_args is not None
    requests = discover.call_args.args[0]
    assert len(requests) == num_requests
    assert all(request.shape == (2,) for request in requests)
    assert gather.call_count == 3


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("production_order", [False, True], ids=["renumbered_tracks", "current_row_order"])
def test_track_mapping_is_rebuilt_after_checkpoint_load(assign: bool, production_order: bool) -> None:
    table = _make_table()
    _fill_table(table)
    permutation = torch.tensor([2, 0, 3, 1])
    current_cuboid_ids = torch.tensor([0, 0, 1, 1])[permutation]
    table.apply_row_permutation(permutation)
    torch.testing.assert_close(table.storage.row_group_ids[table.global_row_ids], current_cuboid_ids)
    loaded = InterpolatedDeformationTable(
        reference_positions=torch.zeros(4, 3),
        gaussian_cuboid_ids=current_cuboid_ids if production_order else torch.tensor([2, 2, 3, 3]),
        track_timestamp_ranges_us=torch.tensor(
            [[0, 200_000], [100_000, 300_000]]
            if production_order
            else [[0, 0], [0, 0], [0, 200_000], [100_000, 300_000]]
        ),
        start_timestamp_us=0,
        end_timestamp_us=300_000,
        frame_period_us=100_000,
        interpolation="linear_slerp",
        storage_backend="frame_major",
        allocate_values=not assign,
    )
    loaded.load_state_dict(table.state_dict(), assign=assign)
    represented_tracks, track_indices = torch.unique(loaded.storage.row_group_ids, sorted=True, return_inverse=True)
    torch.testing.assert_close(loaded.represented_track_ids, represented_tracks)
    torch.testing.assert_close(loaded.track_index_for_global_row, track_indices)
    torch.testing.assert_close(loaded.storage.row_group_ids[loaded.global_row_ids], current_cuboid_ids)
    assert not any("track_index_for_global_row" in key or "represented_track_ids" in key for key in loaded.state_dict())
    times = torch.tensor([125_000, 225_000])
    expected = table(times[current_cuboid_ids], table.global_row_ids, torch.zeros(4, 3))
    actual, smoothness = loaded.evaluate_tracks(
        times, loaded.global_row_ids, torch.zeros(4, 3), compute_curvature=False
    )
    assert smoothness is None
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_checkpoint_load_rebuilds_mappings_on_clone_with_shallow_copied_hooks() -> None:
    source = _make_table()
    mapping_names = ("current_row_for_global_id", "represented_track_ids", "track_index_for_global_row")
    source_mappings = {name: getattr(source, name) for name in mapping_names}
    source_state = {name: value.clone() for name, value in source.state_dict().items()}
    cloned = deepcopy(source).to("meta")
    # Match context_composite._clone_module_tree: each shell owns a registry,
    # but the hook objects themselves are shallow-shared with the source.
    cloned._load_state_dict_post_hooks = source._load_state_dict_post_hooks.copy()
    cloned.storage._load_state_dict_post_hooks = source.storage._load_state_dict_post_hooks.copy()

    trained = _make_table()
    _fill_table(trained)
    trained.apply_row_permutation(torch.tensor([2, 0, 3, 1]))
    cloned.load_state_dict(trained.state_dict(), assign=True)

    for name in mapping_names:
        assert getattr(cloned, name).device.type == "cpu"
        torch.testing.assert_close(getattr(cloned, name), getattr(trained, name))
        assert getattr(source, name) is source_mappings[name]
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, source_state[name], rtol=0, atol=0)
    times = torch.tensor([225_000, 125_000, 225_000, 125_000])
    expected = trained(times, trained.global_row_ids, torch.zeros(4, 3))
    actual, _ = cloned.evaluate_tracks(
        torch.tensor([125_000, 225_000]), cloned.global_row_ids, torch.zeros(4, 3), compute_curvature=False
    )
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype,device", [(torch.int32, "cpu"), (torch.float32, "cpu"), (torch.int64, "meta")])
def test_track_evaluation_rejects_wrong_timestamp_dtype_or_device(dtype: torch.dtype, device: str) -> None:
    table = _make_table()
    timestamps = torch.zeros(2, dtype=dtype, device=device)
    with pytest.raises(ValueError, match="must be int64 on the Gaussian device"):
        table.evaluate_tracks(timestamps, table.global_row_ids, torch.zeros(4, 3), compute_curvature=False)


@pytest.mark.parametrize("shape", [(), (2, 1)])
def test_track_evaluation_rejects_non_vector_timestamps(shape: tuple[int, ...]) -> None:
    table = _make_table()
    timestamps = torch.zeros(shape, dtype=torch.int64)
    with pytest.raises(ValueError, match=r"must have shape \[num_tracks\]"):
        table.evaluate_tracks(timestamps, table.global_row_ids, torch.zeros(4, 3), compute_curvature=False)
