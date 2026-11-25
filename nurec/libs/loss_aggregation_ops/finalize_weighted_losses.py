# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autograd bridge for NuRec's CUDA accumulator finalizer."""

from functools import cache

import torch

from torch.autograd.function import once_differentiable

from nurec.libs.loss_aggregation_ops._types import LossAggregationOps
from nurec.utils.profiling import ScopedTimer


_EXTENSION_MODULE = "nurec.libs.loss_aggregation_ops.libloss_aggregation_ops_cc"


@cache
def _load_native_ops() -> LossAggregationOps | None:
    """Load the CUDA extension lazily."""
    try:
        from nurec.libs.loss_aggregation_ops.interface import loss_aggregation_ops
    except ModuleNotFoundError as error:
        if error.name == _EXTENSION_MODULE:
            return None
        raise
    return loss_aggregation_ops


def _require_native_ops() -> LossAggregationOps:
    native_ops = _load_native_ops()
    if native_ops is None:
        raise RuntimeError("weighted-loss CUDA extension is unavailable")
    return native_ops


class _FinalizeAccumulatedLosses(torch.autograd.Function):
    """First-order autograd bridge for the CUDA aggregation kernel."""

    @staticmethod
    def call(
        weighted_vector: torch.Tensor,
        scalar_values: list[torch.Tensor],
        scalar_weights: list[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pyrefly: ignore[bad-return]
        return _FinalizeAccumulatedLosses.apply(tuple(scalar_weights), weighted_vector, *scalar_values)

    @staticmethod
    def forward(
        ctx,
        scalar_weights: tuple[float, ...],
        weighted_vector: torch.Tensor,
        *scalar_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        native_ops = _require_native_ops()
        total = torch.empty((), dtype=torch.float32, device=weighted_vector.device)
        weighted_values = torch.empty(
            (weighted_vector.numel() + len(scalar_values),), dtype=torch.float32, device=weighted_vector.device
        )
        with ScopedTimer("finalize-accumulated-losses-fwd"):
            native_ops.finalize_accumulated_losses_fwd(
                weighted_vector, list(scalar_values), list(scalar_weights), total, weighted_values
            )

        # Backward is value-independent. Retain immutable allocation metadata,
        # not the input values or their version counters.
        ctx.input_specs = tuple((value.shape, value.dtype, value.device) for value in (weighted_vector, *scalar_values))
        ctx.scalar_weights = scalar_weights
        return total, weighted_values

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        v_total, v_weighted_values = grad_outputs
        input_grads = [torch.empty(shape, dtype=dtype, device=device) for shape, dtype, device in ctx.input_specs]
        native_ops = _require_native_ops()
        with ScopedTimer("finalize-accumulated-losses-bwd"):
            native_ops.finalize_accumulated_losses_bwd(
                v_total.contiguous() if v_total is not None else None,
                v_weighted_values.contiguous() if v_weighted_values is not None else None,
                list(ctx.scalar_weights),
                input_grads[0],
                input_grads[1:],
            )
        return (None, *(grad if ctx.needs_input_grad[index + 1] else None for index, grad in enumerate(input_grads)))


def finalize_accumulated_losses(
    weighted_vector: torch.Tensor,
    scalar_values: list[torch.Tensor],
    scalar_weights: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finalize CUDA accumulator slots followed by independently weighted scalars.

    The contiguous, 1-D float32 ``weighted_vector`` contains already weighted
    slots. Each contiguous, one-element scalar (float16, bfloat16, or float32)
    has one finite outer weight. All inputs must share a CUDA device. Outputs
    are a float32 total and ``weighted_values``, a float32 vector containing the
    input slots followed by the weighted scalar values. Both outputs support
    first-order gradients.

    Packing, scalar weighting and CUB summation share one forward kernel; one
    backward kernel combines gradients from both outputs. Exact zero scalar
    weights never read their inputs and produce exact zero gradients. Up to 16
    total descriptors (the vector plus scalars) are passed by value and support
    CUDA graphs; larger eager calls upload runtime-sized scalar descriptors.
    """
    if not weighted_vector.is_cuda:
        raise ValueError("finalize_accumulated_losses requires a CUDA weighted vector.")
    if weighted_vector.dim() != 1:
        raise ValueError("finalize_accumulated_losses weighted_vector must be 1-D.")
    return _FinalizeAccumulatedLosses.call(
        weighted_vector, list(scalar_values), [float(weight) for weight in scalar_weights]
    )


__all__ = ["finalize_accumulated_losses"]
