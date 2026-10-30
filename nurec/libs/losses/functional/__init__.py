# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused-loss dispatch onto gsplat's fused loss modules."""

from nurec.libs.losses.functional.cuda_losses_function import CudaLossesFunction


__all__ = [
    "CudaLossesFunction",
]
