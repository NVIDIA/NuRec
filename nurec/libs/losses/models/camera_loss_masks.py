# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for camera loss-validity masks."""

from __future__ import annotations

from enum import StrEnum, auto

import torch

from nurec.utils.types import RayFlags


class CameraLossKind(StrEnum):
    """Camera-pixel loss families with component-sensitive denominators."""

    RGB = auto()
    SSIM = auto()
    BACKGROUND = auto()
    SEMANTIC = auto()
    VELOCITY = auto()
    DISTANCE = auto()
    NORMAL = auto()


def _flags_have_all(flags: torch.Tensor, required: RayFlags) -> torch.Tensor:
    return torch.bitwise_and(flags, required.value).eq(required.value)


def _flags_have_none(flags: torch.Tensor, excluded: RayFlags) -> torch.Tensor:
    return torch.bitwise_and(flags, excluded.value).eq(0)


def camera_loss_valid_mask(
    kind: CameraLossKind,
    flags: torch.Tensor,
    *,
    metric_distance: torch.Tensor | None = None,
    distance_bounds: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Return the exact validity predicate used by one camera loss family."""
    valid = _flags_have_none(flags, RayFlags.INVALID)
    if kind == CameraLossKind.RGB:
        return valid & _flags_have_all(flags, RayFlags.RGB_LABEL)
    if kind in (CameraLossKind.SSIM, CameraLossKind.VELOCITY):
        return valid
    if kind == CameraLossKind.BACKGROUND:
        return valid & _flags_have_none(flags, RayFlags.HARMONIZED) & _flags_have_none(flags, RayFlags.SYNTHETIC)
    if kind == CameraLossKind.SEMANTIC:
        return (
            valid
            & _flags_have_all(flags, RayFlags.RGB_LABEL)
            & _flags_have_none(flags, RayFlags.SKY_SEMANTIC)
            & _flags_have_none(flags, RayFlags.HARMONIZED)
        )
    if kind == CameraLossKind.NORMAL:
        return (
            valid
            & _flags_have_none(flags, RayFlags.HARMONIZED)
            & _flags_have_none(flags, RayFlags.SKY_SEMANTIC)
            & _flags_have_all(flags, RayFlags.VALID_NORMAL)
        )

    assert kind == CameraLossKind.DISTANCE
    assert distance_bounds is not None, "Camera distance loss requires configured bounds"
    assert metric_distance is not None, "Camera distance loss requires metric-distance labels"
    min_distance, max_distance = distance_bounds
    return (
        valid
        & _flags_have_none(flags, RayFlags.HARMONIZED)
        & (metric_distance > min_distance)
        & (metric_distance < max_distance)
    )


__all__ = ["CameraLossKind", "camera_loss_valid_mask"]
