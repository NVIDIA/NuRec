# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static type contract for the native loss-aggregation module."""

from typing import Protocol

import torch


class LossAggregationOps(Protocol):
    def finalize_accumulated_losses_fwd(
        self,
        weighted_vector: torch.Tensor,
        scalar_values: list[torch.Tensor],
        scalar_weights: list[float],
        total: torch.Tensor,
        weighted_values: torch.Tensor,
    ) -> None: ...

    def finalize_accumulated_losses_bwd(
        self,
        v_total: torch.Tensor | None,
        v_weighted_values: torch.Tensor | None,
        scalar_weights: list[float],
        v_weighted_vector: torch.Tensor,
        v_scalar_values: list[torch.Tensor],
    ) -> None: ...


__all__ = ["LossAggregationOps"]
