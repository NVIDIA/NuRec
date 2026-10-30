# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nurec.libs.packed_ops.interface import packed_ops


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def _inputs(tracks: int = 3, offset: int = 0) -> list[torch.Tensor]:
    """Strided empty, singleton and duplicate timelines, including overlapping ranges."""
    timestamps = torch.tensor(
        [offset + value for value in [0, 0, 4, 4]] if tracks > 1 else [], dtype=torch.int64, device="cuda"
    )
    packinfo = torch.tensor(
        ([[0, 0], [2, 1], [0, 4]] * ((tracks + 2) // 3))[:tracks], dtype=torch.int64, device="cuda"
    ).reshape(tracks, 2)
    return [torch.stack((timestamps, timestamps), 1)[:, 0], torch.stack((packinfo, packinfo), 2)[:, :, 0]]


@pytest.mark.parametrize("tracks", [0, 1, packed_ops.TRACK_FRAME_SLOTS_THREADS_PER_BLOCK + 1])
@pytest.mark.parametrize("offset", [0, 2**53 + 1])
def test_frame_slots_match_reference(tracks: int, offset: int) -> None:
    inputs = _inputs(tracks, offset)
    storage = torch.full((tracks, 2), -1, dtype=torch.int64, device="cuda")
    actual = storage[:, 0]
    for delta, slot in [(-1, 0), (0, 0), (2, 1), (4, 2), (5, 2)]:
        storage.fill_(-1)  # Every query must overwrite all slots, but not padding.
        packed_ops.nearest_track_frame_slots(offset + delta, *inputs, actual)
        expected = torch.tensor(([0, 0, slot] * ((tracks + 2) // 3))[:tracks], dtype=torch.int64, device="cuda")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.all(storage[:, 1] == -1)


@pytest.mark.parametrize("argument", range(3))
@pytest.mark.parametrize("invalid", ["cpu", "dtype", "rank"])
def test_frame_slots_reject_invalid_tensor_metadata(argument: int, invalid: str) -> None:
    tensors = [*_inputs(), torch.empty(3, dtype=torch.int64, device="cuda")]
    value = tensors[argument]
    if invalid == "cpu":
        tensors[argument], message = value.cpu(), "must be a CUDA tensor"
    elif invalid == "dtype":
        tensors[argument], message = value.int(), "must have dtype int64"
    else:
        tensors[argument], message = value.unsqueeze(0), "must have rank"
    with pytest.raises(RuntimeError, match=message):
        packed_ops.nearest_track_frame_slots(2, *tensors)


@pytest.mark.parametrize("invalid", ["length", "columns", "overlap"])
def test_frame_slots_reject_invalid_layout(invalid: str) -> None:
    tensors = [*_inputs(), torch.empty(3, dtype=torch.int64, device="cuda")]
    if invalid == "length":
        tensors[2], message = tensors[2][:-1], "must have the same length"
    elif invalid == "columns":
        tensors[1], message = tensors[1][:, :1], "must have shape"
    else:
        tensors[2], message = tensors[2][:1].expand(3), "must not overlap internally"
    with pytest.raises(RuntimeError, match=message):
        packed_ops.nearest_track_frame_slots(2, *tensors)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_frame_slots_guard_device_and_reject_mixed_devices() -> None:
    inputs = [value.to("cuda:1") for value in _inputs()]
    output = torch.empty(3, dtype=torch.int64, device="cuda:1")
    with torch.cuda.device(0):
        packed_ops.nearest_track_frame_slots(2, *inputs, output)
        torch.testing.assert_close(output, torch.tensor([0, 0, 1], device="cuda:1"))
        with pytest.raises(RuntimeError, match="same CUDA device"):
            packed_ops.nearest_track_frame_slots(2, *inputs, output.to("cuda:0"))
