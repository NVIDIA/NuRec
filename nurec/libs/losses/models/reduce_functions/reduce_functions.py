# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from nurec.libs.losses.models.reduce_functions.registry import register as register_reduce_fn
from nurec.libs.losses.orchestration.config import RayReduceFn, ReduceConfig


@register_reduce_fn("mean")
class MeanReduceFn(RayReduceFn):
    """Standard mean reduction.

    Without a reduce mask, a single-value input is returned as a 0-d view that shares storage with it.
    """

    def __init__(self, config: ReduceConfig, **kwargs) -> None:
        pass

    def __call__(self, value: torch.Tensor, reduce_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if reduce_mask is not None:
            return (value * reduce_mask).sum() / reduce_mask.sum().clamp(min=1)
        if value.numel() == 1 and value.is_floating_point():
            # The mean of one value is the value itself; return a view instead of launching a kernel.
            return value.reshape(())
        return value.mean()


@register_reduce_fn("quantile")
class QuantileMeanReduceFn(RayReduceFn):
    """Filter out top quantile before applying mean"""

    def __init__(self, config: ReduceConfig, **kwargs) -> None:
        self.quantile = config.quantile
        if self.quantile is not None:
            assert self.quantile > 0 and self.quantile <= 1, "quantile needs to be in ]0,1]"

    def __call__(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        flattened_value = value.flatten()
        n_values = len(value.flatten())
        assert self.quantile is not None
        filtered_idxs = flattened_value.argsort()[: int(n_values * self.quantile)]
        if filtered_idxs.nelement() == 0:
            return torch.tensor(0.0, device=value.device)
        return value[filtered_idxs].mean()


@register_reduce_fn("sum")
class SumReduceFn(RayReduceFn):
    """Standard sum reduction"""

    def __init__(self, config: ReduceConfig, **kwargs) -> None:
        pass

    def __call__(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        return value.sum()
