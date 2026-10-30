# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

from nurec.instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive
from nurec.utils.types import GaussiansRenderReturn


def test_postprocess_rendering_uses_shared_cubemap_sampler() -> None:
    primitive = KelvinInstantNuRecPrimitive.random(1, use_2dgs=True, device=torch.device("cpu"))
    primitive.affine_matrix = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
    ray_directions = torch.tensor([[0.2, 0.3, 0.9], [-0.4, 0.8, 0.1]])
    rendering_data = cast(
        Any,
        SimpleNamespace(
            rays=torch.cat([torch.zeros_like(ray_directions), ray_directions], dim=-1),
            _rays_is_sky=None,
        ),
    )
    output = GaussiansRenderReturn(
        rgb=torch.zeros((2, 3)),
        opacity=torch.zeros(2),
        distance=torch.zeros(2),
    )
    sampled_sky = torch.tensor([[0.1, 0.2, 0.3], [0.6, 0.5, 0.4]])

    with patch(
        "nurec.instant_nurec.primitives.kelvin_primitive.sample_cubemap_seamless", return_value=sampled_sky
    ) as sampler:
        result = primitive.postprocess_rendering(output, rendering_data, unique_sensor_idx=0)

    torch.testing.assert_close(result.rgb, sampled_sky)
    sampler.assert_called_once()
    torch.testing.assert_close(sampler.call_args.args[0], primitive.sky_cubemap)
    torch.testing.assert_close(
        sampler.call_args.args[1],
        torch.stack([ray_directions[:, 0], -ray_directions[:, 1], ray_directions[:, 2]], dim=-1),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cubemap sampling requires CUDA")
def test_postprocess_rendering_samples_constant_cubemap_on_cuda() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    primitive = KelvinInstantNuRecPrimitive.random(1, use_2dgs=True, device=device)
    sky_color = torch.tensor([0.125, 0.25, 0.5], device=device)
    sky_cubemap = sky_color.expand(6, 4, 4, 3).clone()
    primitive.sky_cubemap = sky_cubemap
    primitive.affine_matrix = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
        device=device,
    )
    ray_directions = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 1.0]],
        device=device,
    )
    rendering_data = cast(
        Any,
        SimpleNamespace(
            rays=torch.cat([torch.zeros_like(ray_directions), ray_directions], dim=-1),
            _rays_is_sky=None,
        ),
    )
    foreground_rgb = torch.tensor(
        [[0.0, 0.1, 0.2], [0.2, 0.0, 0.1], [0.1, 0.2, 0.0]],
        device=device,
    )
    opacity = torch.tensor([0.0, 0.25, 0.5], device=device)
    output = GaussiansRenderReturn(
        rgb=foreground_rgb.clone(),
        opacity=opacity,
        distance=torch.zeros(3, device=device),
    )

    result = primitive.postprocess_rendering(output, rendering_data, unique_sensor_idx=0)

    expected = foreground_rgb + (1.0 - opacity[:, None]) * sky_color
    result_rgb = result.rgb
    assert result_rgb is not None
    assert result_rgb.device == sky_cubemap.device == device
    torch.testing.assert_close(result_rgb, expected, atol=2e-6, rtol=0)
