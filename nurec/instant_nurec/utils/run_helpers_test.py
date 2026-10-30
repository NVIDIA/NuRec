# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-global backend settings must not leak between instant_nurec runs."""

import logging

from unittest.mock import Mock

import pytest
import torch

from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig
from nurec.instant_nurec.utils.run_helpers import _maybe_enable_cudnn_benchmark_for_static_shapes


@pytest.mark.parametrize("disabled_gate", ["config", "cuda", "cudnn", "deterministic", "dynamic"])
def test_cudnn_benchmark_is_cleared_after_previously_enabled_run(disabled_gate: str, monkeypatch) -> None:
    config = Mock(spec=InstantNuRecConfig)
    config.enable_cudnn_benchmark = True
    config.has_static_shapes.return_value = True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    logger = logging.getLogger(__name__)

    _maybe_enable_cudnn_benchmark_for_static_shapes(config, logger)
    assert torch.backends.cudnn.benchmark

    if disabled_gate == "config":
        config.enable_cudnn_benchmark = False
    elif disabled_gate == "cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    elif disabled_gate == "cudnn":
        monkeypatch.setattr(torch.backends.cudnn, "enabled", False)
    elif disabled_gate == "deterministic":
        monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    else:
        config.has_static_shapes.return_value = False
    _maybe_enable_cudnn_benchmark_for_static_shapes(config, logger)
    assert not torch.backends.cudnn.benchmark
