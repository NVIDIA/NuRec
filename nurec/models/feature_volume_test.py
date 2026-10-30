# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from nurec.models.feature_volume import TcnnMultiLevelEncoding


def _stub_encoding(n_active_levels: int, *, encoding_has_level_input: bool = True) -> TcnnMultiLevelEncoding:
    """Construct an encoding without initializing TCNN or CUDA."""
    encoding = TcnnMultiLevelEncoding.__new__(TcnnMultiLevelEncoding)
    torch.nn.Module.__init__(encoding)
    encoding.n_input_dims = 3
    encoding.n_levels = 16
    encoding.n_active_levels = n_active_levels
    encoding.encoding_has_level_input = encoding_has_level_input
    encoding.encoding_config = {}
    return encoding


def test_tcnn_multi_level_encoding_restores_active_levels_from_extra_state() -> None:
    saved_state = _stub_encoding(n_active_levels=16).state_dict()
    restored_encoding = _stub_encoding(n_active_levels=10)

    restored_encoding.load_state_dict(saved_state)

    assert restored_encoding.n_active_levels == 16


def test_tcnn_multi_level_encoding_preserves_active_levels_without_level_input() -> None:
    initial_n_active_levels = 7
    saved_state = _stub_encoding(n_active_levels=16, encoding_has_level_input=False).state_dict()
    restored_encoding = _stub_encoding(
        n_active_levels=initial_n_active_levels,
        encoding_has_level_input=False,
    )

    restored_encoding.load_state_dict(saved_state)

    assert restored_encoding.n_active_levels == initial_n_active_levels
