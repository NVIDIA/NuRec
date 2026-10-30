# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loss-orchestration support for component-only camera denominator selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from nurec.libs.losses.models.camera_loss_masks import (
    CameraLossKind,
)
from nurec.libs.losses.models.camera_loss_masks import (
    camera_loss_valid_mask as shared_camera_loss_valid_mask,
)
from nurec.libs.losses.orchestration.config import LossAggregatorReturn
from nurec.utils.batch import CameraFrameLabels


@dataclass(frozen=True, slots=True)
class ComponentOnlyLossNormalizationContext:
    """Camera validity captured before component masking mutates the labels."""

    original_flags: torch.Tensor

    @classmethod
    def capture(cls, flags: torch.Tensor) -> "ComponentOnlyLossNormalizationContext":
        return cls(original_flags=flags.clone())


def configured_camera_loss_kinds_by_name(configured_losses: Iterable[Any]) -> dict[str, CameraLossKind]:
    """Collect explicit camera-mean metadata declared by configured loss implementations."""
    return {
        loss.name: kind
        for loss in configured_losses
        if (kind := getattr(loss, "component_only_camera_loss_kind", None)) is not None
    }


def camera_loss_valid_mask(
    kind: CameraLossKind,
    labels: CameraFrameLabels,
    flags: torch.Tensor,
    *,
    distance_bounds: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Reconstruct one camera loss's validity predicate against arbitrary flags."""
    if kind == CameraLossKind.DISTANCE:
        assert distance_bounds is not None, "Missing configured bounds for camera distance loss"
        assert labels.metric_distance is not None, "Missing metric-distance labels for camera distance loss"
    return shared_camera_loss_valid_mask(
        kind,
        flags,
        metric_distance=labels.metric_distance,
        distance_bounds=distance_bounds,
    )


def configured_distance_bounds_by_name(configured_losses: Iterable[Any]) -> dict[str, tuple[float, float]]:
    """Collect the range predicate from configured camera distance losses."""
    return {
        loss.name: (float(loss.min_distance), float(loss.max_distance))
        for loss in configured_losses
        if getattr(loss, "component_only_camera_loss_kind", None) == CameraLossKind.DISTANCE
    }


def rescale_camera_mean_losses_to_original_valid_pixels(
    loss_return: LossAggregatorReturn,
    labels: CameraFrameLabels,
    context: ComponentOnlyLossNormalizationContext,
    *,
    camera_loss_kinds_by_name: Mapping[str, CameraLossKind],
    distance_bounds_by_name: Mapping[str, tuple[float, float]],
) -> None:
    """Convert masked-valid camera means to original-valid camera means in-place.

    Camera losses execute against component-masked flags first, retaining their
    normal safe empty-mask behavior. Multiplying the reduced result by
    ``masked_valid / original_valid`` then converts
    ``masked_numerator / masked_valid`` into
    ``masked_numerator / original_valid``. Fused means are represented as
    pre-scaled sums, so the same post-reduction conversion applies to both
    implementations.
    """
    assert labels.flags is not None, "all_valid_pixels normalization requires component-masked camera flags"
    current_flags = labels.flags
    assert current_flags.shape == context.original_flags.shape, (
        "Camera flag shape changed while applying the component-only mask"
    )

    scales_by_name: dict[str, torch.Tensor] = {}
    for loss in loss_return.loss_returns.values():
        kind = camera_loss_kinds_by_name.get(loss.name)
        if kind is None:
            continue
        distance_bounds = distance_bounds_by_name.get(loss.name) if kind == CameraLossKind.DISTANCE else None
        original_valid = camera_loss_valid_mask(
            kind,
            labels,
            context.original_flags,
            distance_bounds=distance_bounds,
        )
        masked_valid = camera_loss_valid_mask(
            kind,
            labels,
            current_flags,
            distance_bounds=distance_bounds,
        )
        original_count = original_valid.sum()
        masked_count = masked_valid.sum()
        scale = torch.where(
            original_count > 0,
            masked_count / original_count.clamp(min=1),
            original_count.to(torch.float32),
        ).to(loss.reduced_value.dtype)
        scales_by_name[loss.name] = scale

    loss_return.rescale_reduced_values(scales_by_name)


__all__ = [
    "ComponentOnlyLossNormalizationContext",
    "camera_loss_valid_mask",
    "configured_camera_loss_kinds_by_name",
    "configured_distance_bounds_by_name",
    "rescale_camera_mean_losses_to_original_valid_pixels",
]
