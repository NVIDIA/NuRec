# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-CUDA contracts for the separate accumulator result API."""

from unittest import mock

import pytest
import torch

from nurec.libs.loss_aggregation_ops import finalize_accumulated_losses
from nurec.libs.losses.orchestration.config import AccumulatorLoss, LossAccumulatorReturn


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_weighted_prefix_raw_scalar_tail_and_diagnostic_gradients():
    vector = torch.tensor([2.0, -4.0, 0.0], device="cuda", requires_grad=True)
    ppisp = torch.tensor(3.0, device="cuda", requires_grad=True)
    ignored = torch.tensor(float("nan"), device="cuda", requires_grad=True)
    result = LossAccumulatorReturn(vector, {"rgb": 0, "ground": 1, "disabled": 2}, {"rgb": 7.0, "ground": -2.0})
    result.append_producer_loss(
        AccumulatorLoss(name="ppisp", lambda_=0.25, value=ppisp, reduce_fn=lambda value, **_: value.reshape(()))
    )
    result.append_producer_loss(
        AccumulatorLoss(name="zero", lambda_=0.0, value=ignored, reduce_fn=lambda value, **_: value.reshape(()))
    )
    assert result.names == ("rgb", "ground", "ppisp", "zero")
    torch.testing.assert_close(result.weighted_values, vector.new_tensor([2.0, -4.0, 0.0, 0.75, 0.0]))
    torch.testing.assert_close(result.compute_total(), vector.new_tensor(-1.25))
    # Logging metadata cannot multiply the already-weighted prefix again.
    assert result.weights_by_name["rgb"] == 7.0
    assert result.weighted_values is result.weighted_values
    with pytest.raises(KeyError):
        result.weighted_value("disabled")
    objective = 2.0 * result.compute_total() + 3.0 * result.weighted_value("rgb") + 4.0 * result.weighted_value("ppisp")
    objective.backward()
    torch.testing.assert_close(vector.grad, vector.new_tensor([5.0, 2.0, 2.0]))
    torch.testing.assert_close(ppisp.grad, ppisp.new_tensor(1.5))
    torch.testing.assert_close(ignored.grad, ignored.new_tensor(0.0))


@pytest.mark.parametrize("guard", [torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("lazy_map", [False, True])
def test_logging_under_disabled_grad_does_not_cache_a_detached_scalar_objective(guard, lazy_map):
    vector = torch.zeros(2, device="cuda")
    scalar = torch.tensor(3.0, device="cuda", requires_grad=True)
    result = LossAccumulatorReturn(vector, {"rgb": 0, "ground": 1}, {})
    reduce_fn = (lambda value, **_: value * 1.0) if lazy_map else (lambda value, **_: value.reshape(()))
    result.append_producer_loss(AccumulatorLoss(name="ppisp", lambda_=2.0, value=scalar, reduce_fn=reduce_fn))
    records = {}
    with guard():
        result.log(lambda name, value: records.__setitem__(name, value))
        assert not result.compute_total().requires_grad
    assert records["train_params/ppisp_lambda"] == 2.0
    assert result.compute_total().requires_grad
    result.compute_total().backward()
    torch.testing.assert_close(scalar.grad, scalar.new_tensor(2.0))


def test_normal_logging_keeps_the_cached_graph_for_backward():
    vector = torch.tensor([2.0, 4.0], device="cuda", requires_grad=True)
    result = LossAccumulatorReturn(vector, {"rgb": 0, "ground": 1}, {"rgb": 0.5, "ground": 4.0})
    result.log(lambda *_: None)
    result.compute_total().backward()
    torch.testing.assert_close(vector.grad, torch.ones_like(vector))


def test_component_rescale_then_calibration_append_preserves_all_gradients():
    vector = torch.tensor([2.0, 3.0], device="cuda", requires_grad=True)
    ppisp = torch.tensor(4.0, device="cuda", requires_grad=True)
    calibration = torch.tensor(5.0, device="cuda", requires_grad=True)
    scale = torch.tensor(0.25, device="cuda", requires_grad=True)
    result = LossAccumulatorReturn(vector, {"rgb": 0, "ground": 1}, {"rgb": 2.0, "ground": 7.0})
    result.append_producer_loss(
        AccumulatorLoss(name="ppisp", lambda_=0.5, value=ppisp, reduce_fn=lambda value, **_: value.reshape(()))
    )
    result.rescale_reduced_values({"rgb": scale})
    reducer = mock.Mock(side_effect=lambda value, **_: value.reshape(()))
    with mock.patch(
        "nurec.libs.losses.orchestration.config.finalize_accumulated_losses",
        wraps=finalize_accumulated_losses,
    ) as finalize:
        result.append_producer_loss(
            AccumulatorLoss(name="calibration", lambda_=-2.0, value=calibration, reduce_fn=reducer)
        )
        reducer.assert_not_called()
        assert result.weighted_value_dtype("rgb") == torch.float32
        finalize.assert_not_called()
        result.log(lambda *_: None)
        torch.testing.assert_close(result.weighted_values, vector.new_tensor([0.5, 3.0, 2.0, -10.0]))
        torch.testing.assert_close(result.compute_total(), vector.new_tensor(-4.5))
        result.compute_total().backward()
        reducer.assert_called_once()
        finalize.assert_called_once()
    torch.testing.assert_close(vector.grad, vector.new_tensor([0.25, 1.0]))
    torch.testing.assert_close(ppisp.grad, ppisp.new_tensor(0.5))
    torch.testing.assert_close(calibration.grad, calibration.new_tensor(-2.0))
    torch.testing.assert_close(scale.grad, scale.new_tensor(2.0))


def test_independent_map_reduction_stays_lazy_and_runs_once():
    vector = torch.zeros(2, device="cuda")
    raw = torch.tensor([2.0, 6.0], device="cuda", requires_grad=True)
    reducer = mock.Mock(side_effect=lambda value, **_: value.mean())
    loss = AccumulatorLoss(name="independent", lambda_=-3.0, value=raw, reduce_fn=reducer)
    result = LossAccumulatorReturn(vector, {}, {})
    result.append_producer_loss(loss)
    loss.lambda_ = 7.0  # Appending snapshots the producer's weight for this step.
    reducer.assert_not_called()
    result.rescale_reduced_values({})
    reducer.assert_not_called()
    torch.testing.assert_close(result.weighted_value(loss.name), raw.new_tensor(-12.0))
    result.compute_total().backward()
    reducer.assert_called_once()
    torch.testing.assert_close(raw.grad, raw.new_tensor([-1.5, -1.5]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("weight", [0.0, 1.0, -2.0, 0.5])
def test_protocol_scalar_append_is_lazy_and_invalidates_cache(weight):
    from nurec.libs.losses.orchestration.config import LossResult

    vector = torch.tensor([2.0], device="cuda", requires_grad=True)
    result = LossAccumulatorReturn(vector, {"sample": 0}, {"sample": 1.0})
    old_total = result.compute_total()
    scalar = torch.tensor(4.0, device="cuda", requires_grad=True)
    common: LossResult = result
    assert common.names == ("sample",)
    assert common.weighted_value_dtype("sample") == torch.float32
    common.append_loss(name="shared", value=scalar, weight=weight)
    assert common.names == ("sample", "shared")
    assert common.weighted_value_dtype("shared") == torch.float32
    assert result._finalized is None
    with torch.no_grad():
        torch.testing.assert_close(common.compute_total(), old_total.detach() + scalar * weight)
    assert result._finalized is None
    logged = {}
    common.log(lambda name, value: logged.update({name: value}))
    torch.testing.assert_close(logged["loss/shared"], scalar * weight)
    assert logged["train_params/shared_lambda"] == weight
    common.compute_total().backward()
    torch.testing.assert_close(vector.grad, torch.ones_like(vector))
    torch.testing.assert_close(scalar.grad, scalar.new_tensor(weight))
    cached = result._finalized
    with pytest.raises(ValueError, match="already contains"):
        common.append_loss(name="shared", value=scalar)
    with pytest.raises(ValueError, match="scalar"):
        common.append_loss(name="vector", value=torch.ones(2, device="cuda"))
    assert result._finalized is cached
