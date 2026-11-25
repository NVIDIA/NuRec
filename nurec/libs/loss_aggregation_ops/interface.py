# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed interface to the native weighted-loss aggregation extension."""

import importlib

from typing import cast

import torch  # noqa: F401

from nurec.libs.loss_aggregation_ops._types import LossAggregationOps


loss_aggregation_ops = cast(
    LossAggregationOps,
    importlib.import_module("nurec.libs.loss_aggregation_ops.libloss_aggregation_ops_cc"),
)


__all__ = ["LossAggregationOps", "loss_aggregation_ops"]
