# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from nurec.instant_nurec.utils.cubemap import cubemap_ray_directions, rotate_sky_cubemap
from nurec.models.background_sampling import sample_cubemap_seamless


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cubemap samplers require CUDA")


def _axis_rotation(axis: int, degrees: float, device: torch.device) -> torch.Tensor:
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotations = [
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
    ]
    return torch.tensor(rotations[axis], device=device)


def _compound_rotation(device: torch.device) -> torch.Tensor:
    return _axis_rotation(0, 11.0, device) @ _axis_rotation(1, -23.0, device) @ _axis_rotation(2, 17.0, device)


def _smooth_cubemap(size: int, n_channels: int, device: torch.device) -> torch.Tensor:
    directions = cubemap_ray_directions(size, device)
    features = [
        0.5 + 0.3 * directions[..., 0],
        0.5 + 0.3 * directions[..., 1],
        0.5 + 0.3 * directions[..., 2],
        torch.sigmoid(3.0 * directions[..., 2]),
        0.5 + 0.25 * directions[..., 0] * directions[..., 1],
    ]
    return torch.stack(features[:n_channels], dim=-1).contiguous()


def _edge_discontinuous_cubemap(size: int, n_channels: int, device: torch.device) -> torch.Tensor:
    coordinates = torch.arange(size, device=device)
    xx, yy = torch.meshgrid(coordinates, coordinates, indexing="xy")
    face = torch.arange(6, device=device)[:, None, None]
    if n_channels == 1:
        return (((3 * xx + 5 * yy + 7 * face) % 19) < 9).float()[..., None].contiguous()

    xx = xx / (size - 1)
    yy = yy / (size - 1)
    return torch.stack(
        [
            (face + 1) / 7 + 0.05 * xx,
            (6 - face) / 7 + 0.05 * yy,
            ((face % 2) * 0.8 + 0.1 * xx + 0.1 * yy),
        ],
        dim=-1,
    ).contiguous()


def _sample_rotated_cubemap(cubemap: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    cubemap_size = cubemap.shape[1]
    query_rays = (cubemap_ray_directions(cubemap_size, cubemap.device) @ rotation.float()).reshape(-1, 3)
    opengl_rays = torch.stack([query_rays[:, 0], -query_rays[:, 1], query_rays[:, 2]], dim=-1)
    return sample_cubemap_seamless(cubemap, opengl_rays).reshape_as(cubemap)


@pytest.mark.parametrize("n_channels", [1, 3, 5])
def test_rotate_sky_cubemap_identity_preserves_features(n_channels: int) -> None:
    cubemap = _smooth_cubemap(32, n_channels, torch.device("cuda"))

    rotated = rotate_sky_cubemap(cubemap, torch.eye(3, device="cuda"))

    assert rotated.shape == cubemap.shape
    assert rotated.dtype == cubemap.dtype
    torch.testing.assert_close(rotated, cubemap, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("n_channels", [1, 3])
def test_rotate_sky_cubemap_applies_requested_rotation(n_channels: int) -> None:
    cubemap = _edge_discontinuous_cubemap(256, n_channels, torch.device("cuda"))
    rotation = _compound_rotation(cubemap.device)

    actual = rotate_sky_cubemap(cubemap, rotation)
    expected = _sample_rotated_cubemap(cubemap, rotation)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("n_channels", [1, 3])
def test_rotate_sky_cubemap_preserves_bfloat16_dtype_and_values(n_channels: int) -> None:
    cubemap = _edge_discontinuous_cubemap(64, n_channels, torch.device("cuda")).bfloat16()
    rotation = _compound_rotation(cubemap.device)

    actual = rotate_sky_cubemap(cubemap, rotation)
    expected = _sample_rotated_cubemap(cubemap.float(), rotation)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected, atol=4e-3, rtol=4e-3)
