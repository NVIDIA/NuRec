# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from nurec.render.utils import transform_intrinsics_to_resolution


class _FakeCameraParameters:
    def __init__(self, resolution: tuple[int, int], *, return_wrong_resolution: bool = False) -> None:
        self.resolution = np.asarray(resolution, dtype=np.uint64)
        self.return_wrong_resolution = return_wrong_resolution
        self.transform_args: tuple[float | tuple[float, float], tuple[int, int]] | None = None

    def transform(
        self,
        *,
        image_domain_scale: float | tuple[float, float],
        new_resolution: tuple[int, int],
    ) -> _FakeCameraParameters:
        self.transform_args = (image_domain_scale, new_resolution)
        returned_resolution = (2, 2) if self.return_wrong_resolution else new_resolution
        return _FakeCameraParameters(returned_resolution)


@pytest.mark.parametrize(
    ("requested_resolution", "expected_resolution", "expected_scale"),
    [
        ((1920, 1), (1920, 1080), 0.5),
        ((1, 1080), (1920, 1080), 0.5),
        ((2000, 1000), (2000, 1125), 2000 / 3840),
        ((1000, 1200), (2133, 1200), 1200 / 2160),
    ],
)
def test_transform_intrinsics_to_resolution_preserves_aspect_ratio(
    requested_resolution: tuple[int, int],
    expected_resolution: tuple[int, int],
    expected_scale: float | tuple[float, float],
) -> None:
    parameters = _FakeCameraParameters((3840, 2160))

    transformed = transform_intrinsics_to_resolution(
        cast(Any, parameters),
        requested_resolution,
    )

    assert parameters.transform_args is not None
    actual_scale, actual_resolution = parameters.transform_args
    np.testing.assert_allclose(actual_scale, expected_scale)
    assert actual_resolution == expected_resolution
    np.testing.assert_array_equal(transformed.resolution, expected_resolution)


def test_transform_intrinsics_to_resolution_checks_transformed_resolution() -> None:
    parameters = _FakeCameraParameters((3840, 2160), return_wrong_resolution=True)

    with pytest.raises(AssertionError):
        transform_intrinsics_to_resolution(cast(Any, parameters), (1, 1080))
