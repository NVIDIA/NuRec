# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch


@dataclass(kw_only=True)
class BaseInstantNuRecSupervisionPack:
    """
    Dataclass storing the supervision data required for instant_nurec training.
    """


@dataclass(kw_only=True)
class CelsiusInstantNuRecSupervisionPack(BaseInstantNuRecSupervisionPack):
    """
    Supervision pack for the Celsius model.

    The fields are:
    - context_distance: (B, H, W, 1)
    - context_velocity: (B, H, W, 3/6)
    - context_unscaled_falloff_sigma: (B, H, W, 1)
    """

    context_distance: torch.Tensor | None = None
    context_velocity: torch.Tensor | None = None
    context_unscaled_falloff_sigma: torch.Tensor | None = None


@dataclass(kw_only=True)
class KelvinMotionSupervision:
    """
    Motion supervision targets vs predicted flow.
    Fields:
    - source_timestamps_us: (..., 1)
    - target_timestamps_us: (..., 1)
    - context_flow: (..., 3)
    - reference_flow: (..., 3) | None
    """

    source_timestamps_us: torch.Tensor
    target_timestamps_us: torch.Tensor
    context_flow: torch.Tensor
    reference_flow: torch.Tensor | None = None


@dataclass(kw_only=True)
class KelvinInstantNuRecSupervisionPack(BaseInstantNuRecSupervisionPack):
    """
    Supervision pack for the Kelvin model.

    The fields are:
    - context_rgb: (B, H, W, 3) if pixel-aligned
    - context_depth: (B, H, W, 1) if pixel-aligned
    - context_depth_conf: (B, H, W, 1) if pixel-aligned
    - context_semantic_logits: (B, H, W, C) if pixel-aligned
    - context_xyz: (B, H, W, 3) if pixel-aligned
    - context_world_normal: (B, H, W, 3) if pixel-aligned (unit-norm, world space)
    - predicted_sky_cubemap: (6, S, S, 3)
    - reference_sky_cubemap: (6, S, S, 3)
    - reference_sky_cubemap_mask: (6, S, S, 1)
    - motion_supervisions: list[KelvinMotionSupervision] (... = B, H, W)
    """

    context_rgb: torch.Tensor | None = None
    context_depth: torch.Tensor | None = None
    context_depth_conf: torch.Tensor | None = None
    context_semantic_logits: torch.Tensor | None = None
    context_xyz: torch.Tensor | None = None
    context_world_normal: torch.Tensor | None = None
    predicted_sky_cubemap: torch.Tensor | None = None
    reference_sky_cubemap: torch.Tensor | None = None
    reference_sky_cubemap_mask: torch.Tensor | None = None
    motion_supervisions: list[KelvinMotionSupervision] = field(default_factory=list)
