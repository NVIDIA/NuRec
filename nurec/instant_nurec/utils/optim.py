# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight parameter-marking utilities used by model architecture code (e.g.
`nurec.instant_nurec.models.blocks.mamba_scan`), independent of the optimizer/scheduler
construction logic that lives in a separate downstream training package.
"""

from __future__ import annotations

import torch


def mark_parameter_no_weight_decay(param: torch.nn.Parameter) -> None:
    """
    Mark a parameter as not requiring weight decay by setting an attribute within the parameter.
    """
    setattr(param, "_instant_nurec_no_weight_decay", True)


def check_parameter_no_weight_decay(param: torch.nn.Parameter) -> bool:
    """
    Check if a parameter is marked as not requiring weight decay.
    """
    flag = getattr(param, "_instant_nurec_no_weight_decay", False)
    assert isinstance(flag, bool), "Attribute '_instant_nurec_no_weight_decay' must be a boolean"
    return flag
