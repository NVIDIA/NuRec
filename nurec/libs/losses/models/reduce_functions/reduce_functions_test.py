# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nurec.libs.losses.models.reduce_functions.reduce_functions import MeanReduceFn
from nurec.libs.losses.orchestration.config import ReduceConfig


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_mean_of_single_value_matches_torch_mean(device, shape, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    value = torch.full(shape, 0.75, dtype=dtype, device=device, requires_grad=True)
    reference = value.detach().clone().requires_grad_()
    actual = MeanReduceFn(ReduceConfig(name="mean"))(value)
    expected = reference.mean()
    assert actual.shape == expected.shape == ()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)


def test_mean_of_single_value_is_a_view_without_kernel_work():
    value = torch.tensor([3.0])
    result = MeanReduceFn(ReduceConfig(name="mean"))(value)
    assert result.data_ptr() == value.data_ptr()


def test_mean_of_multiple_values_still_reduces():
    result = MeanReduceFn(ReduceConfig(name="mean"))(torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(result, torch.tensor(1.5))


def test_mean_of_single_value_keeps_reduce_mask():
    result = MeanReduceFn(ReduceConfig(name="mean"))(torch.tensor([3.0]), reduce_mask=torch.tensor([0.0]))
    torch.testing.assert_close(result, torch.tensor(0.0))


def test_mean_of_single_integer_value_still_rejected():
    with pytest.raises(RuntimeError):
        MeanReduceFn(ReduceConfig(name="mean"))(torch.tensor([3]))
