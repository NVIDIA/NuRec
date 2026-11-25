# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from nurec.models.gaussians.temporal_deformation_storage import FrameMajorRaggedStorage


_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
]


def _make_storage(
    device: str,
    *,
    allocate_values: bool = True,
    reorder_rows: bool = False,
) -> FrameMajorRaggedStorage:
    starts = torch.tensor([0, 4, 0, 1, 4], device=device)
    ends = torch.tensor([2, 6, 2, 1, 6], device=device)
    groups = torch.tensor([10, 20, 10, 30, 20], device=device)
    if reorder_rows:
        permutation = torch.tensor([4, 3, 2, 1, 0], device=device)
        starts, ends, groups = starts[permutation], ends[permutation], groups[permutation]
    return FrameMajorRaggedStorage(
        num_columns=7,
        row_start_columns=starts,
        row_end_columns=ends,
        row_group_ids=groups,
        value_dim=7,
        dtype=torch.float32,
        device=torch.device(device),
        allocate_values=allocate_values,
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_fixed_size_gather_preserves_row_order_parameter_identity_and_gradients(device: str) -> None:
    storage = _make_storage(device)
    for column_index in range(storage.num_columns):
        expected_rows = storage.active_rows(column_index)
        # The dynamic-output query remains available for infrequent mutation
        # paths, but must not be called by the hot column gather.
        with patch.object(storage, "active_rows", side_effect=AssertionError("dynamic active-row query")):
            column = storage.gather_compressed_column(column_index)
        torch.testing.assert_close(column.row_indices, expected_rows, rtol=0, atol=0)
        assert column.row_indices.dtype == torch.int64
        if expected_rows.numel() == 0:
            assert column.values.shape == (0, storage.value_dim)
            continue
        parameter = next(storage.columns[str(column_index)].parameters())
        assert column.values is parameter
        column.values.sum().backward()
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
        assert all(other.grad is None for other in storage.parameters() if other is not parameter)
        storage.zero_grad(set_to_none=True)
    with pytest.raises(IndexError, match="outside"):
        storage.gather_compressed_column(-1)
    with pytest.raises(IndexError, match="outside"):
        storage.gather_compressed_column(storage.num_columns)


@pytest.mark.parametrize("device", _DEVICES)
def test_empty_storage_has_only_empty_columns(device: str) -> None:
    empty = torch.empty(0, dtype=torch.int64, device=device)
    storage = FrameMajorRaggedStorage(
        num_columns=3,
        row_start_columns=empty,
        row_end_columns=empty,
        row_group_ids=empty,
        value_dim=7,
        dtype=torch.float32,
        device=torch.device(device),
    )
    storage.load_state_dict(storage.state_dict(), assign=True)
    for column_index in range(storage.num_columns):
        column = storage.gather_compressed_column(column_index)
        assert column.row_indices.shape == (0,)
        assert column.values.shape == (0, 7)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(("allocate_values", "assign"), [(True, False), (True, True), (False, True)])
def test_recursive_load_refreshes_membership_without_cached_indices(
    device: str, allocate_values: bool, assign: bool
) -> None:
    source = _make_storage(device, reorder_rows=True)
    with torch.no_grad():
        for column_index in range(source.num_columns):
            column = source.gather_compressed_column(column_index)
            column.values.copy_(column.row_indices.unsqueeze(1) + column_index)
    target = _make_storage(device, allocate_values=allocate_values)
    source_root = torch.nn.ModuleDict({"storage": source})
    target_root = torch.nn.ModuleDict({"storage": target})
    target_root.load_state_dict(source_root.state_dict(), assign=assign)
    for column_index in range(source.num_columns):
        expected = source.gather_compressed_column(column_index)
        actual = target.gather_compressed_column(column_index)
        torch.testing.assert_close(actual.row_indices, expected.row_indices, rtol=0, atol=0)
        torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
        if actual.values.numel():
            assert actual.values is next(target.columns[str(column_index)].parameters())


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize(
    ("buffer_name", "row", "value", "message"),
    [
        ("row_end_columns", 0, 1, "does not match"),  # Selection would pad.
        ("row_start_columns", 3, 0, "does not match"),  # Selection would truncate.
        ("row_end_columns", 0, 3, "missing active rows"),
        ("row_end_columns", 0, 7, "lifetimes are invalid"),
    ],
)
def test_load_rejects_lifetime_and_column_count_mismatches(
    device: str, assign: bool, buffer_name: str, row: int, value: int, message: str
) -> None:
    storage = _make_storage(device)
    checkpoint = {key: value.clone() for key, value in storage.state_dict().items()}
    checkpoint[buffer_name][row] = value
    with pytest.raises(RuntimeError, match=message):
        storage.load_state_dict(checkpoint, assign=assign)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("buffer_name", ["row_start_columns", "row_end_columns"])
def test_assign_load_rejects_non_int64_lifetimes(device: str, buffer_name: str) -> None:
    checkpoint = _make_storage(device).state_dict()
    checkpoint[buffer_name] = checkpoint[buffer_name].to(torch.int32)
    target = _make_storage(device, allocate_values=False)
    # assign=True preserves the checkpoint dtype instead of casting into the
    # existing int64 buffer, so post-load validation must reject it.
    with pytest.raises(RuntimeError, match="checkpoint deformation row lifetimes are invalid"):
        target.load_state_dict(checkpoint, assign=True)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("strict", [False, True])
def test_lazy_checkpoint_load_reports_missing_column(device: str, strict: bool) -> None:
    source_root = torch.nn.ModuleDict({"storage": _make_storage(device)})
    checkpoint = source_root.state_dict()
    del checkpoint["storage.columns.0.values"]
    target_root = torch.nn.ModuleDict({"storage": _make_storage(device, allocate_values=False)})
    with pytest.raises(
        RuntimeError,
        match=r"checkpoint deformation column 0 was not loaded \(missing columns\.0\.values\)",
    ):
        target_root.load_state_dict(checkpoint, strict=strict, assign=True)


@pytest.mark.parametrize("device", _DEVICES)
def test_shallow_copied_load_hook_validates_loaded_clone_not_uninitialized_original(device: str) -> None:
    original = _make_storage(device, allocate_values=False)
    clone = _make_storage(device, allocate_values=False)
    # Composite skeleton cloning shallow-copies hook registries. A bound hook
    # would still validate the original's uninitialized columns after this load.
    clone._load_state_dict_post_hooks = original._load_state_dict_post_hooks.copy()
    source = _make_storage(device, reorder_rows=True)
    clone.load_state_dict(source.state_dict(), assign=True)
    assert isinstance(next(original.columns["0"].parameters()), torch.nn.UninitializedParameter)
    for column_index in range(source.num_columns):
        expected = source.gather_compressed_column(column_index)
        actual = clone.gather_compressed_column(column_index)
        torch.testing.assert_close(actual.row_indices, expected.row_indices, rtol=0, atol=0)
        torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_shallow_copied_load_hook_rejects_invalid_clone_not_valid_original(device: str) -> None:
    original = _make_storage(device)
    clone = _make_storage(device, allocate_values=False)
    clone._load_state_dict_post_hooks = original._load_state_dict_post_hooks.copy()
    checkpoint = {key: value.clone() for key, value in original.state_dict().items()}
    checkpoint["row_end_columns"][0] = original.num_columns
    # The original remains valid; validating a bound method's owner would let
    # this invalid clone checkpoint load successfully.
    with pytest.raises(RuntimeError, match="checkpoint deformation row lifetimes are invalid"):
        clone.load_state_dict(checkpoint, assign=True)
    assert original.row_end_columns[0].item() == 2


@pytest.mark.parametrize("device", _DEVICES)
def test_lazy_checkpoint_load_rejects_wrong_column_shape(device: str) -> None:
    checkpoint = _make_storage(device).state_dict()
    checkpoint["columns.0.values"] = torch.zeros(3, 7, device=device)
    target = _make_storage(device, allocate_values=False)
    with pytest.raises(RuntimeError, match="column 0 does not match its 2 active rows"):
        target.load_state_dict(checkpoint, assign=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fixed_size_gather_is_cuda_graph_capturable() -> None:
    storage = _make_storage("cuda")
    expected = storage.active_rows(1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        column = storage.gather_compressed_column(1)
    graph.replay()
    torch.testing.assert_close(column.row_indices, expected, rtol=0, atol=0)
