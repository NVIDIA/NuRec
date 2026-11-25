# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spatial partitioning helpers for Gaussian USD exports.

This module is intentionally self-contained so the workaround can be removed
when upstream USDZ/ZIP64 support makes it unnecessary.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch


# OVRTX indexes Gaussian rows with a 24-bit field. Keep one value available for
# sentinel/invalid-index handling rather than using the all-ones value.
OVRTX_MAX_GAUSSIANS_PER_LAYER = (1 << 24) - 1
# Conservative target for a single USDZ member. The ZIP format's nominal limit
# is larger, but older USDZ readers commonly fail well before it.
USDZ_SAFE_MEMBER_BYTES = 2 * 1024**3


def automatic_partition_cap(num_gaussians: int, estimated_bytes_per_gaussian: float) -> int | None:
    """Return a safe cap when either OVRTX or USDZ size limits require splitting."""
    cap = OVRTX_MAX_GAUSSIANS_PER_LAYER
    if estimated_bytes_per_gaussian > 0:
        cap = min(cap, int(USDZ_SAFE_MEMBER_BYTES * 0.8 / estimated_bytes_per_gaussian))
    cap = max(1, cap)
    return cap if num_gaussians > cap else None


def partition_gaussian_indices(positions: np.ndarray, max_per_partition: int) -> List[torch.Tensor]:
    """Split Gaussian rows by a balanced spatial KD-tree."""
    if max_per_partition < 1:
        raise ValueError("max_per_partition must be positive")
    if max_per_partition > OVRTX_MAX_GAUSSIANS_PER_LAYER:
        raise ValueError(
            f"max_per_partition={max_per_partition} exceeds the OVRTX limit "
            f"of {OVRTX_MAX_GAUSSIANS_PER_LAYER} Gaussians per layer"
        )
    n = len(positions)
    if n <= max_per_partition:
        return [torch.arange(n, dtype=torch.long)]

    pending: List[np.ndarray] = [np.arange(n, dtype=np.int64)]
    leaves: List[np.ndarray] = []
    while pending:
        indices = pending.pop()
        if len(indices) <= max_per_partition:
            leaves.append(indices)
            continue
        points = positions[indices]
        axis = int(np.argmax(np.ptp(points, axis=0)))
        order = np.argsort(points[:, axis], kind="stable")
        midpoint = len(indices) // 2
        pending.append(indices[order[:midpoint]])
        pending.append(indices[order[midpoint:]])
    return [torch.from_numpy(indices) for indices in leaves]
