# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict initialization contracts for the optional encoder-depth ablation."""

import pytest
import torch

from nurec.instant_nurec.config.models import (
    KelvinDAv3EncoderConfig,
    KelvinDPTDecoderConfig,
    KelvinModelConfig,
    KelvinPointQueryCADecoderConfig,
)
from nurec.instant_nurec.models.kelvin_backbone import decoders
from nurec.instant_nurec.models.kelvin_backbone.decoders import KelvinDPTDecoder, KelvinPointQueryCADecoder


@pytest.mark.parametrize("use_encoder_depth", [False, True])
def test_dav3_initialization_only_remaps_present_depth_head(use_encoder_depth: bool) -> None:
    config = KelvinDPTDecoderConfig(
        name="dpt-decoder",
        dpt_dim=16,
        dpt_reassemble_hidden_dims=[8, 16, 32, 64],
        motion_depth=0,
        use_encoder_depth=use_encoder_depth,
    )
    # Construct only the decoder; the encoder/renderer are not exercised by weight remapping.
    model_config = KelvinModelConfig.model_construct(
        encoder=KelvinDAv3EncoderConfig.model_construct(embed_dim=32, take_block_indices=[0, 1, 2, 3]),
    )
    decoder = KelvinDPTDecoder(config, model_config)
    dav3 = {"model.head.norm.weight": torch.ones(32)}
    if use_encoder_depth:
        decoder.initialize_weights({"dav3": dav3})
        assert not any(key.startswith("depth_head.") for key in decoder.state_dict())
    else:
        # A partial checkpoint must still fail strictly when the learned depth head is present.
        with pytest.raises(RuntimeError, match="Missing key"):
            decoder.initialize_weights({"dav3": dav3})


@pytest.mark.parametrize("point_query", [False, True])
def test_dav3_initialization_retains_gaussian_activation_buffers(point_query: bool, monkeypatch) -> None:
    model_config = KelvinModelConfig.model_construct(
        encoder=KelvinDAv3EncoderConfig.model_construct(embed_dim=32, n_heads=4, take_block_indices=[0, 1, 2, 3]),
    )
    decoder: KelvinDPTDecoder
    if point_query:
        ca_config = KelvinPointQueryCADecoderConfig(
            name="point-query-ca-decoder", dpt_dim=16, dpt_reassemble_hidden_dims=[8, 16, 32, 64], motion_depth=0
        )
        decoder = KelvinPointQueryCADecoder(ca_config, model_config)
    else:
        dpt_config = KelvinDPTDecoderConfig(
            name="dpt-decoder", dpt_dim=16, dpt_reassemble_hidden_dims=[8, 16, 32, 64], motion_depth=0
        )
        decoder = KelvinDPTDecoder(dpt_config, model_config)
    # Isolate strict state assembly after a complete DAV3 depth-head remap.
    depth_state = {f"depth_head.{key}": value for key, value in decoder.depth_head.state_dict().items()}
    monkeypatch.setattr(
        decoders, "convert_dav3_state_dict_to_instant_nurec", lambda *args, **kwargs: depth_state.copy()
    )
    activation_state = {key: value.clone() for key, value in decoder.gaussian_activations.state_dict().items()}
    assert activation_state
    decoder.initialize_weights({"dav3": {}})
    for key, value in activation_state.items():
        torch.testing.assert_close(decoder.gaussian_activations.state_dict()[key], value)
