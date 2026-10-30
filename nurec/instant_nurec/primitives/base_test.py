# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch

from nurec.instant_nurec.primitives.base import BaseGaussiansInstantNuRecPrimitive
from nurec.instant_nurec.primitives.celsius_primitive import CelsiusInstantNuRecPrimitive


@pytest.mark.parametrize("constructor", ["base", "celsius"])
@pytest.mark.parametrize("use_color_transfer", [None, False, True])
def test_camera_render_forwards_harmonizer_color_transfer(constructor: str, use_color_transfer: bool | None) -> None:
    """Forward explicit settings and retain legacy color transfer when the argument is omitted."""
    original_rgb = torch.zeros((6, 3))
    harmonized_rgb = torch.ones((6, 3))
    render_output = MagicMock(rgb=original_rgb)
    harmonizer_model = MagicMock(return_value=harmonized_rgb)
    renderer = MagicMock()
    renderer.render.return_value = render_output
    background_color = torch.tensor([0.1, 0.2, 0.3])
    options = {} if use_color_transfer is None else {"harmonizer_use_color_transfer": use_color_transfer}

    # Use the pre-existing positional order through solid_background_color.
    primitive = CelsiusInstantNuRecPrimitive(
        torch.zeros((1, 3)),  # positions
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),  # rotations
        torch.ones((1, 3)),  # scales
        torch.ones((1, 1)),  # densities
        torch.zeros((1, 3)),  # rgb
        torch.zeros((1, 1)),  # timestamps_us
        None,  # forward_speed_mps
        None,  # falloff_sigma
        None,  # dynamic_bbox_mask
        None,  # road_mask
        None,  # sky_mask
        None,  # sky_token
        None,  # sky_head
        None,  # sky_rotation
        None,  # affine_matrix
        None,  # affine_bias
        renderer,
        "none",
        harmonizer_model,
        True,  # sky_mask_enabled
        background_color,
        **options,
    )
    assert primitive.sky_mask_enabled is True
    assert primitive.solid_background_color is background_color

    if constructor == "base":
        # Exercise the base constructor's own default on the concrete primitive.
        BaseGaussiansInstantNuRecPrimitive.__init__(
            primitive, renderer, checkpointing="none", harmonizer_model=harmonizer_model, **options
        )

    rendering_data = MagicMock(b=1, h=2, w=3)
    frame_meta = MagicMock(unique_sensor_idx=0)
    with (
        patch.object(primitive, "get_gaussian_parameters", return_value={}),
        patch.object(primitive, "postprocess_rendering", return_value=render_output),
    ):
        result = primitive._render_single_camera_frame(rendering_data, frame_meta, {})

    expected = True if use_color_transfer is None else use_color_transfer
    assert result.rgb is harmonized_rgb
    harmonizer_model.assert_called_once_with(original_rgb, torch.Size([2, 3]), expected)
