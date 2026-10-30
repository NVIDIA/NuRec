# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nurec.libs.losses.orchestration.config import LossReturn


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("weight", [0.0, 0.25, 1.0, 2.0, -1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_weighted_value_and_gradient(device, weight, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    value = torch.tensor([1.0, 3.0], device=device, dtype=dtype, requires_grad=True)
    reference = value.detach().clone().requires_grad_()
    mask = torch.tensor([1.0, 0.0], device=device, dtype=dtype)

    def reduce_fn(tensor, reduce_mask=None):
        return (tensor * reduce_mask).sum()

    loss = LossReturn(name="test", lambda_=weight, value=value, reduce_fn=reduce_fn, reduce_mask=mask)
    expected = weight * reduce_fn(reference, reduce_mask=mask)
    torch.testing.assert_close(loss.reduced_value, expected, rtol=0, atol=0)
    loss.reduced_value.backward()
    expected.backward()
    torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_unit_weight_skips_scaling(dtype):
    reduced = torch.tensor(3.0, dtype=dtype)
    loss = LossReturn(name="test", lambda_=1.0, value=torch.tensor([9.0]), reduce_fn=lambda x, **_: reduced)
    assert loss.reduced_value is reduced


def test_unit_weight_preserves_integer_promotion():
    value = torch.tensor([2, 3])
    loss = LossReturn(name="test", lambda_=1.0, value=value, reduce_fn=lambda x, **_: x.sum())
    torch.testing.assert_close(loss.reduced_value, 1.0 * value.sum())


def test_cached_reduction_is_not_recomputed():
    def unexpected_reduction(*args, **kwargs):
        raise AssertionError("cached reduction must be reused")

    reduced = torch.tensor(3.0)
    loss = LossReturn(
        name="test",
        lambda_=1.0,
        value=torch.tensor([9.0]),
        reduce_fn=unexpected_reduction,
        _reduced_value=reduced,
    )
    assert loss.reduced_value is reduced
