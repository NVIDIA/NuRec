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


@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize("weight", [0.0, 1.0, -2.0, 0.5])
def test_scalar_append_preserves_objective_gradients_and_logging(batch_size, weight):
    from nurec.libs.losses.orchestration.config import LossAggregatorBatchReturn, LossAggregatorReturn, LossResult

    samples = [torch.tensor(float(i + 1), requires_grad=True) for i in range(batch_size or 1)]
    items = [
        LossAggregatorReturn(
            {"sample": LossReturn(name="sample", lambda_=1.0, value=value, reduce_fn=lambda x, **_: x)}
        )
        for value in samples
    ]
    result: LossResult = items[0] if batch_size is None else LossAggregatorBatchReturn(items)
    scalar = torch.tensor(4.0, requires_grad=True)
    before = result.compute_total().detach().clone()
    result.append_loss(name="shared", value=scalar, weight=weight)
    torch.testing.assert_close(result.compute_total(), before + scalar * weight)
    assert result.names == ("sample", "shared")
    assert result.weighted_value_dtype("shared") == scalar.dtype
    roots = result.backward_roots()
    assert len(roots["shared"]) == 1
    logged = {}
    result.log(lambda name, value, **kwargs: logged.update({name: value}))
    torch.testing.assert_close(logged["loss/shared"], scalar * weight)
    assert logged["train_params/shared_lambda"] == weight
    result.compute_total().backward()
    torch.testing.assert_close(scalar.grad, scalar.new_tensor(weight))
    for sample in samples:
        torch.testing.assert_close(sample.grad, sample.new_tensor(1.0 / len(samples)))
    old_total = result.compute_total()
    with pytest.raises(ValueError, match="already contains"):
        result.append_loss(name="shared", value=scalar)
    with pytest.raises(ValueError, match="already contains"):
        result.append_loss(name="sample", value=scalar)
    with pytest.raises(ValueError, match="scalar"):
        result.append_loss(name="vector", value=torch.ones(2))
    assert result.compute_total() is old_total


def test_batch_named_dtype_matches_logged_promotion():
    from nurec.libs.losses.orchestration.config import LossAggregatorBatchReturn, LossAggregatorReturn, LossResult

    items = [
        LossAggregatorReturn(
            {
                "sample": LossReturn(
                    name="sample", value=torch.tensor(2.0, dtype=dtype), lambda_=1.0, reduce_fn=lambda x, **_: x
                )
            }
        )
        for dtype in (torch.float32, torch.float64)
    ]
    result: LossResult = LossAggregatorBatchReturn(items)
    assert result.names == ("sample",)
    logged = {}
    result.log(lambda name, value, **kwargs: logged.update({name: value}))
    assert result.weighted_value_dtype("sample") == logged["loss/sample"].dtype == torch.float64
    with pytest.raises(KeyError):
        result.weighted_value_dtype("missing")


@pytest.mark.parametrize("batch_size", [None, 3])
def test_protocol_rescale_preserves_sample_and_shared_gradients(batch_size):
    from nurec.libs.losses.orchestration.config import LossAggregatorBatchReturn, LossAggregatorReturn, LossResult

    samples = [torch.tensor(float(i + 2), requires_grad=True) for i in range(batch_size or 1)]
    items = [
        LossAggregatorReturn(
            {"sample": LossReturn(name="sample", value=value, lambda_=1.0, reduce_fn=lambda x, **_: x)}
        )
        for value in samples
    ]
    result: LossResult = items[0] if batch_size is None else LossAggregatorBatchReturn(items)
    scalar = torch.tensor(4.0, requires_grad=True)
    result.append_loss(name="shared", value=scalar, weight=-2.0)
    sample_scale = torch.tensor(0.25, requires_grad=True)
    shared_scale = torch.tensor(2.0, requires_grad=True)
    before = result.compute_total()
    result.rescale_reduced_values({})
    assert result.compute_total() is before
    if batch_size is not None:
        with pytest.raises(KeyError):
            result.rescale_reduced_values({"sample": sample_scale, "missing": shared_scale})
        assert result.compute_total() is before
    result.rescale_reduced_values({"sample": sample_scale, "shared": shared_scale})
    expected_sample_mean = sum(sample.detach() for sample in samples) / len(samples)
    expected = expected_sample_mean * sample_scale - 2.0 * scalar * shared_scale
    torch.testing.assert_close(result.compute_total(), expected)
    logged = {}
    result.log(lambda name, value, **kwargs: logged.update({name: value}))
    torch.testing.assert_close(logged["loss/sample"], expected_sample_mean * sample_scale)
    torch.testing.assert_close(logged["loss/shared"], scalar * -2.0 * shared_scale)
    result.compute_total().backward()
    for sample in samples:
        torch.testing.assert_close(sample.grad, sample.new_tensor(0.25 / len(samples)))
    torch.testing.assert_close(scalar.grad, scalar.new_tensor(-4.0))
    torch.testing.assert_close(sample_scale.grad, expected_sample_mean)
    torch.testing.assert_close(shared_scale.grad, scalar.detach() * -2.0)


@pytest.mark.parametrize("batch", [False, True])
def test_compute_total_does_not_repeat_legacy_reduction(batch):
    from unittest import mock

    from nurec.libs.losses.orchestration.config import LossAggregatorBatchReturn, LossAggregatorReturn, LossResult

    item = LossAggregatorReturn(
        {"sample": LossReturn(name="sample", value=torch.tensor(2.0), lambda_=1.0, reduce_fn=lambda x, **_: x)}
    )
    concrete = LossAggregatorBatchReturn([item]) if batch else item
    result: LossResult = concrete
    with mock.patch.object(type(concrete), "reduce", side_effect=AssertionError("unexpected repeat reduction")):
        assert result.compute_total() is concrete.total_value
        assert result.compute_total() is concrete.total_value
