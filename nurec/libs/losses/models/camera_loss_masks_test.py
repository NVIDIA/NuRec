# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import pytest
import torch

from nurec.libs.losses.models.camera_loss_masks import CameraLossKind, camera_loss_valid_mask
from nurec.utils.types import RayFlags


def _flags_and_distance() -> tuple[torch.Tensor, torch.Tensor]:
    base = RayFlags.RGB_LABEL.value | RayFlags.VALID_NORMAL.value
    flags = torch.tensor(
        [
            base,
            base | RayFlags.INVALID.value,
            RayFlags.VALID_NORMAL.value,
            base | RayFlags.HARMONIZED.value,
            base | RayFlags.SYNTHETIC.value,
            base | RayFlags.SKY_SEMANTIC.value,
            RayFlags.RGB_LABEL.value,
            base,
        ],
        dtype=torch.int32,
    ).reshape(1, 1, 8, 1)
    distance = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 0.5, 10.0, 101.0]).reshape(1, 1, 8, 1)
    return flags, distance


def test_camera_loss_kind_uses_lowercase_generated_values() -> None:
    assert [kind.value for kind in CameraLossKind] == [
        "rgb",
        "ssim",
        "background",
        "semantic",
        "velocity",
        "distance",
        "normal",
    ]


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (CameraLossKind.RGB, [True, False, False, True, True, True, True, True]),
        (CameraLossKind.SSIM, [True, False, True, True, True, True, True, True]),
        (CameraLossKind.VELOCITY, [True, False, True, True, True, True, True, True]),
        (CameraLossKind.BACKGROUND, [True, False, True, False, False, True, True, True]),
        (CameraLossKind.SEMANTIC, [True, False, False, False, True, False, True, True]),
        (CameraLossKind.NORMAL, [True, False, True, False, True, False, False, True]),
        (CameraLossKind.DISTANCE, [True, False, True, False, True, False, True, False]),
    ],
)
def test_camera_loss_valid_mask_matches_loss_predicates(kind: CameraLossKind, expected: list[bool]) -> None:
    flags, distance = _flags_and_distance()
    actual = camera_loss_valid_mask(
        kind,
        flags,
        metric_distance=distance,
        distance_bounds=(1.0, 100.0),
    )
    assert actual.flatten().tolist() == expected


def test_distance_requires_bounds_and_metric_labels() -> None:
    flags, distance = _flags_and_distance()
    with pytest.raises(AssertionError, match="configured bounds"):
        camera_loss_valid_mask(CameraLossKind.DISTANCE, flags, metric_distance=distance)
    with pytest.raises(AssertionError, match="metric-distance labels"):
        camera_loss_valid_mask(CameraLossKind.DISTANCE, flags, distance_bounds=(1.0, 100.0))


def test_rejects_unknown_loss_kind() -> None:
    flags, _ = _flags_and_distance()
    with pytest.raises(AssertionError):
        camera_loss_valid_mask(cast(Any, "unknown"), flags)
