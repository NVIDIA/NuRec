# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-sized aggregation for NuRec's weighted training objective."""

from nurec.libs.loss_aggregation_ops.finalize_weighted_losses import finalize_accumulated_losses


__all__ = ["finalize_accumulated_losses"]
