# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity and contracts for the CUDA accumulator finalizer."""

from unittest.mock import PropertyMock, patch

import pytest
import torch

from nurec.libs.loss_aggregation_ops import finalize_accumulated_losses
from nurec.libs.loss_aggregation_ops.finalize_weighted_losses import _load_native_ops


CUDA_AVAILABLE = torch.cuda.is_available() and _load_native_ops() is not None


def test_cuda_extension_loads_when_cuda_is_available() -> None:
    if torch.cuda.is_available():
        assert _load_native_ops() is not None


def test_cuda_input_requires_native_extension() -> None:
    vector = torch.tensor([1.0, 2.0])
    with (
        patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True),
        patch(f"{finalize_accumulated_losses.__module__}._load_native_ops", return_value=None),
        pytest.raises(RuntimeError, match="weighted-loss CUDA extension is unavailable"),
    ):
        finalize_accumulated_losses(vector, [], [])


def test_cpu_input_is_rejected_before_loading_extension() -> None:
    with (
        patch(f"{finalize_accumulated_losses.__module__}._load_native_ops") as load,
        pytest.raises(ValueError, match="requires a CUDA weighted vector"),
    ):
        finalize_accumulated_losses(torch.ones(2), [], [])
    load.assert_not_called()


def test_cuda_native_import_error_is_not_hidden() -> None:
    with (
        patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True),
        patch(f"{finalize_accumulated_losses.__module__}._load_native_ops", side_effect=ImportError("ABI mismatch")),
        pytest.raises(ImportError, match="ABI mismatch"),
    ):
        finalize_accumulated_losses(torch.tensor([1.0]), [], [])


def test_malformed_cuda_input_is_rejected_before_loading_extension() -> None:
    with (
        patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True),
        patch(f"{finalize_accumulated_losses.__module__}._load_native_ops") as load,
        pytest.raises(ValueError, match="1-D"),
    ):
        finalize_accumulated_losses(torch.zeros(2, 2), [], [])
    load.assert_not_called()


def _reference(
    weighted_vector: torch.Tensor,
    scalar_values: list[torch.Tensor],
    scalar_weights: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    weighted_scalars = [
        value.reshape(-1)[:0].sum(dtype=torch.float32) if weight == 0.0 else value.reshape(()).float() * weight
        for value, weight in zip(scalar_values, scalar_weights)
    ]
    weighted_values = torch.cat(
        (weighted_vector, torch.stack(weighted_scalars) if weighted_scalars else weighted_vector.new_empty((0,)))
    )
    return weighted_values.sum(), weighted_values


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension unavailable")
@pytest.mark.parametrize("scalar_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mixed_precision_scalars_accumulate_in_float32(scalar_dtype: torch.dtype) -> None:
    vector = torch.tensor([1.0, 2.0], device="cuda", requires_grad=True)
    scalar = torch.tensor(3.0, device="cuda", dtype=scalar_dtype, requires_grad=True)
    total, weighted_values = finalize_accumulated_losses(vector, [scalar], [0.25])
    assert total.dtype == weighted_values.dtype == torch.float32
    torch.testing.assert_close(total, torch.tensor(3.75, device="cuda"))
    torch.testing.assert_close(weighted_values, torch.tensor([1.0, 2.0, 0.75], device="cuda"))
    (2.0 * total + 3.0 * weighted_values.sum()).backward()
    torch.testing.assert_close(vector.grad, torch.full_like(vector, 5.0))
    assert scalar.grad is not None and scalar.grad.dtype == scalar_dtype
    torch.testing.assert_close(scalar.grad, torch.full_like(scalar, 1.25))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension unavailable")
def test_heterogeneous_scalar_dtypes_are_tagged_independently() -> None:
    vector = torch.tensor([1.0, -2.0], device="cuda", requires_grad=True)
    scalar_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    scalars = [
        torch.tensor(value, device="cuda", dtype=dtype, requires_grad=True)
        for value, dtype in zip((2.0, -3.0, 4.0), scalar_dtypes)
    ]
    weights = [0.5, -0.25, 2.0]
    total, weighted_values = finalize_accumulated_losses(vector, scalars, weights)
    torch.testing.assert_close(total, torch.tensor(8.75, device="cuda"))
    torch.testing.assert_close(weighted_values, torch.tensor([1.0, -2.0, 1.0, 0.75, 8.0], device="cuda"))
    side_upstream = torch.tensor([0.0, 0.0, 3.0, -1.0, 0.5], device="cuda")
    (2.0 * total + (side_upstream * weighted_values).sum()).backward()
    torch.testing.assert_close(vector.grad, torch.full_like(vector, 2.0))
    for scalar, dtype, expected in zip(scalars, scalar_dtypes, (2.5, -0.25, 5.0)):
        assert scalar.grad is not None and scalar.grad.dtype == dtype
        torch.testing.assert_close(scalar.grad, torch.full_like(scalar, expected))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension unavailable")
def test_cuda_runtime_sized_value_and_backward() -> None:
    vector = torch.randn(513, device="cuda", requires_grad=True)
    scalars = [torch.randn((), device="cuda", requires_grad=True) for _ in range(300)]
    weights = [(-0.01 if index % 2 else 0.02) * (index + 1) for index in range(300)]
    total, weighted_values = finalize_accumulated_losses(vector, scalars, weights)
    expected_total, expected_values = _reference(vector, scalars, weights)
    torch.testing.assert_close(total, expected_total, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(weighted_values, expected_values, rtol=0, atol=0)
    total_upstream = -0.75
    side_upstream = torch.randn_like(weighted_values)
    torch.autograd.backward((total, weighted_values), (total.new_tensor(total_upstream), side_upstream))
    torch.testing.assert_close(vector.grad, total_upstream + side_upstream[: vector.numel()], rtol=0, atol=0)
    for index, scalar in enumerate(scalars):
        expected = weights[index] * (total_upstream + side_upstream[vector.numel() + index])
        torch.testing.assert_close(scalar.grad, torch.ones_like(scalar) * expected, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension unavailable")
@pytest.mark.parametrize("n_descriptors", [1, 2, 3, 4, 5, 8, 9, 16, 17])
def test_cuda_descriptor_boundaries(n_descriptors: int) -> None:
    vector = torch.linspace(-1.0, 1.0, 33, device="cuda", requires_grad=True)
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    scalars = [
        torch.tensor(float(index % 7 - 3), device="cuda", dtype=dtypes[index % 3], requires_grad=True)
        for index in range(n_descriptors - 1)
    ]
    weights = [(0.0, 0.25, -0.5, 1.0)[index % 4] for index in range(len(scalars))]
    total, weighted_values = finalize_accumulated_losses(vector, scalars, weights)
    expected_total, expected_values = _reference(vector, scalars, weights)
    torch.testing.assert_close(total, expected_total, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(weighted_values, expected_values, rtol=0, atol=0)
    (total + weighted_values.sum()).backward()
    torch.testing.assert_close(vector.grad, torch.full_like(vector, 2.0), rtol=0, atol=0)
    for scalar, weight in zip(scalars, weights):
        torch.testing.assert_close(scalar.grad, torch.full_like(scalar, 2.0 * weight), rtol=0, atol=0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension unavailable")
@pytest.mark.parametrize("n_descriptors", [1, 2, 4, 8, 16])
def test_inline_descriptors_support_cuda_graph_forward_backward(n_descriptors: int) -> None:
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        warmup_vector = torch.arange(5, device="cuda", dtype=torch.float32, requires_grad=True)
        warmup_scalars = [
            torch.tensor(float(index), device="cuda", requires_grad=True) for index in range(n_descriptors - 1)
        ]
        warmup_total, warmup_values = finalize_accumulated_losses(
            warmup_vector, warmup_scalars, [0.25] * len(warmup_scalars)
        )
        (warmup_total + warmup_values.sum()).backward()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    vector = torch.arange(5, device="cuda", dtype=torch.float32, requires_grad=True)
    scalars = [torch.tensor(float(index), device="cuda", requires_grad=True) for index in range(n_descriptors - 1)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        total, weighted_values = finalize_accumulated_losses(vector, scalars, [0.25] * len(scalars))
        (total + 2.0 * weighted_values.sum()).backward()
    with torch.no_grad():
        vector.add_(1.0)
        for scalar in scalars:
            scalar.add_(2.0)
    graph.replay()
    torch.cuda.synchronize()
    expected_total, expected_values = _reference(vector, scalars, [0.25] * len(scalars))
    torch.testing.assert_close(weighted_values, expected_values, rtol=0, atol=0)
    torch.testing.assert_close(total, expected_total, rtol=0, atol=0)
    torch.testing.assert_close(vector.grad, torch.full_like(vector, 3.0), rtol=0, atol=0)
    for scalar in scalars:
        torch.testing.assert_close(scalar.grad, torch.full_like(scalar, 0.75), rtol=0, atol=0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension unavailable")
@pytest.mark.parametrize("scalar_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_zero_scalar_weight_is_exact_no_read(scalar_dtype: torch.dtype) -> None:
    vector = torch.tensor([2.0], device="cuda", requires_grad=True)
    scalar = torch.tensor(float("nan"), device="cuda", dtype=scalar_dtype, requires_grad=True)
    total, weighted_values = finalize_accumulated_losses(vector, [scalar], [0.0])
    assert total.item() == 2.0
    torch.testing.assert_close(weighted_values, torch.tensor([2.0, 0.0], device="cuda"), rtol=0, atol=0)
    torch.autograd.backward(
        (total, weighted_values),
        (torch.ones((), device="cuda"), torch.tensor([0.0, float("nan")], device="cuda")),
    )
    assert scalar.grad is not None and scalar.grad.item() == 0.0
    assert vector.grad is not None and vector.grad.item() == 1.0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension unavailable")
def test_nullable_upstreams_and_empty_scalar_list() -> None:
    vector = torch.tensor([1.0, 2.0], device="cuda", requires_grad=True)
    total, weighted_values = finalize_accumulated_losses(vector, [], [])
    torch.testing.assert_close(weighted_values, vector)
    total.backward()
    torch.testing.assert_close(vector.grad, torch.ones_like(vector), rtol=0, atol=0)
    vector = torch.empty(0, device="cuda", requires_grad=True)
    scalars = [
        torch.tensor(2.0, device="cuda", requires_grad=True),
        torch.tensor([-3.0], device="cuda", requires_grad=True),
    ]
    _, weighted_values = finalize_accumulated_losses(vector, scalars, [0.25, -2.0])
    (weighted_values * torch.tensor([3.0, -0.5], device="cuda")).sum().backward()
    torch.testing.assert_close(scalars[0].grad, torch.tensor(0.75, device="cuda"))
    torch.testing.assert_close(scalars[1].grad, torch.tensor([1.0], device="cuda"))
    torch.testing.assert_close(vector.grad, torch.empty_like(vector), rtol=0, atol=0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension unavailable")
def test_cuda_backward_is_value_independent_and_first_order_only() -> None:
    vector = torch.tensor([1.0, 2.0], device="cuda", requires_grad=True)
    scalar = torch.tensor(3.0, device="cuda", requires_grad=True)
    total, weighted_values = finalize_accumulated_losses(vector, [scalar], [0.5])
    with torch.no_grad():
        vector.add_(10.0)
        scalar.add_(10.0)
    scalar_grad = torch.autograd.grad(total + 2.0 * weighted_values.sum(), scalar, create_graph=True)[0]
    torch.testing.assert_close(scalar_grad, torch.tensor(1.5, device="cuda"))
    assert not scalar_grad.requires_grad


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension unavailable")
def test_native_entry_point_revalidates_outputs() -> None:
    native_ops = _load_native_ops()
    assert native_ops is not None
    vector = torch.ones(3, device="cuda")
    with pytest.raises(RuntimeError, match="weighted_values must have shape"):
        native_ops.finalize_accumulated_losses_fwd(
            vector,
            [torch.ones((), device="cuda")],
            [1.0],
            torch.empty((), device="cuda"),
            torch.empty(0, device="cuda"),
        )
    with pytest.raises(RuntimeError, match="v_weighted_values must have shape"):
        native_ops.finalize_accumulated_losses_bwd(
            None, torch.empty(0, device="cuda"), [], torch.empty_like(vector), []
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension unavailable")
@pytest.mark.parametrize("n_slots,n_scalars", [(0, 0), (0, 3), (20, 0), (20, 3), (513, 16)])
@pytest.mark.parametrize("objective", ["total", "slots", "both"])
def test_accumulator_finalizer_complete_vector_and_gradients(n_slots: int, n_scalars: int, objective: str) -> None:
    vector = torch.linspace(-1, 1, n_slots, device="cuda", requires_grad=True)
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    shapes = ((), (1,), (1, 1))
    weights = [(0.0, -0.25, 2.0)[index % 3] for index in range(n_scalars)]
    scalars = [
        torch.full(
            shapes[index % 3],
            float("nan") if weight == 0.0 else float(index + 1),
            device="cuda",
            dtype=dtypes[index % 3],
            requires_grad=True,
        )
        for index, weight in enumerate(weights)
    ]
    expected_scalars = torch.tensor(
        [0.0 if weight == 0.0 else float(index + 1) * weight for index, weight in enumerate(weights)], device="cuda"
    )
    expected_values = torch.cat((vector.detach(), expected_scalars))
    total, weighted_values = finalize_accumulated_losses(vector, scalars, weights)
    torch.testing.assert_close(weighted_values, expected_values, rtol=0, atol=0)
    torch.testing.assert_close(total, expected_values.sum(), rtol=1e-5, atol=1e-5)
    assert total.dtype == weighted_values.dtype == torch.float32

    # Strided upstream tests the bridge's contiguity handling. Unused outputs
    # reach backward as None, including an entirely empty output vector.
    upstream = (torch.arange(2 * weighted_values.numel(), device="cuda", dtype=torch.float32) / 8)[::2]
    total_scale = -0.5 if objective != "slots" else 0.0
    if objective == "total":
        torch.autograd.backward(total, total.new_tensor(total_scale))
    elif objective == "slots":
        torch.autograd.backward(weighted_values, upstream)
    else:
        torch.autograd.backward((total, weighted_values), (total.new_tensor(total_scale), upstream))
    expected_vector_grad = torch.full_like(vector, total_scale)
    if objective != "total":
        expected_vector_grad += upstream[:n_slots]
    torch.testing.assert_close(vector.grad, expected_vector_grad, rtol=0, atol=0)
    for index, (scalar, weight) in enumerate(zip(scalars, weights)):
        slot_upstream = upstream[n_slots + index] if objective != "total" else 0.0
        expected_grad = 0.0 if weight == 0.0 else weight * (total_scale + slot_upstream)
        torch.testing.assert_close(
            scalar.grad, (torch.ones_like(scalar) * expected_grad).to(scalar.dtype), rtol=0, atol=0
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension unavailable")
def test_accumulator_finalizer_zero_weight_masks_nonfinite_upstreams() -> None:
    vector = torch.ones(2, device="cuda", requires_grad=True)
    scalar = torch.tensor(float("inf"), device="cuda", requires_grad=True)
    total, weighted_values = finalize_accumulated_losses(vector, [scalar], [0.0])
    torch.testing.assert_close(weighted_values, torch.tensor([1.0, 1.0, 0.0], device="cuda"))
    torch.autograd.backward(
        (total, weighted_values), (total.new_tensor(float("nan")), torch.full_like(weighted_values, float("nan")))
    )
    torch.testing.assert_close(scalar.grad, torch.zeros_like(scalar), rtol=0, atol=0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension unavailable")
def test_accumulator_finalizer_cuda_graph_on_nondefault_stream() -> None:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    vector = torch.arange(20, device="cuda", dtype=torch.float32, requires_grad=True)
    scalar = torch.tensor(2.0, device="cuda", requires_grad=True)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            total, weighted_values = finalize_accumulated_losses(vector, [scalar], [-0.5])
            torch.autograd.grad(
                (total, weighted_values), (vector, scalar), (torch.ones_like(total), torch.ones_like(weighted_values))
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        total, weighted_values = finalize_accumulated_losses(vector, [scalar], [-0.5])
        vector_grad, scalar_grad = torch.autograd.grad(
            (total, weighted_values), (vector, scalar), (torch.ones_like(total), torch.full_like(weighted_values, 2.0))
        )
    with torch.no_grad():
        vector.add_(1)
        scalar.fill_(4)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(weighted_values, torch.cat((vector.detach(), scalar.detach().reshape(1) * -0.5)))
    torch.testing.assert_close(total, vector.sum() - 2)
    torch.testing.assert_close(vector_grad, torch.full_like(vector, 3))
    torch.testing.assert_close(scalar_grad, torch.full_like(scalar, -1.5))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension unavailable")
@pytest.mark.parametrize(
    "invalid",
    ["rank", "scalar_rank", "dtype", "stride", "device", "weights", "missing_weight", "scalar", "scalar_dtype"],
)
def test_accumulator_finalizer_rejects_invalid_inputs(invalid: str) -> None:
    vector = torch.ones(20, device="cuda")
    scalars = [torch.ones((), device="cuda")]
    weights = [1.0]
    match invalid:
        case "rank":
            vector = vector.reshape(4, 5)
        case "scalar_rank":
            vector = vector[0]
        case "dtype":
            vector = vector.half()
        case "stride":
            vector = vector[::2]
        case "device":
            scalars = [torch.ones(())]
        case "weights":
            weights = [float("nan")]
        case "missing_weight":
            weights = []
        case "scalar":
            scalars = [vector]
        case "scalar_dtype":
            scalars = [scalars[0].double()]
    with pytest.raises((ValueError, RuntimeError)):
        finalize_accumulated_losses(vector, scalars, weights)
