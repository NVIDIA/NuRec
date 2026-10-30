# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nurec.instant_nurec.systems.gaussians_instant_nurec import GaussiansInstantNuRecSystem


def _minimal_system() -> GaussiansInstantNuRecSystem:
    system = GaussiansInstantNuRecSystem.__new__(GaussiansInstantNuRecSystem)
    torch.nn.Module.__init__(system)
    # Only state-dict behavior is exercised; register a lightweight module stub.
    system.add_module("model", torch.nn.Linear(2, 1))
    return system


def test_inference_ignores_only_obsolete_validation_lpips_state() -> None:
    system = _minimal_system()
    checkpoint = system.state_dict()
    checkpoint["validation_lpips.net.legacy_weight"] = torch.ones(1)

    result = system.load_state_dict(checkpoint, strict=True, assign=True)

    assert result.missing_keys == result.unexpected_keys == []
    assert "validation_lpips.net.legacy_weight" in checkpoint
    assert not hasattr(system, "validation_lpips")


@pytest.mark.parametrize("invalid_key", ["model.unexpected", "loss.unexpected", "validation_lpips_extra.weight"])
def test_validation_compatibility_does_not_hide_unexpected_weights(invalid_key: str) -> None:
    system = _minimal_system()
    checkpoint = system.state_dict()
    checkpoint["validation_lpips.net.legacy_weight"] = torch.ones(1)
    checkpoint[invalid_key] = torch.ones(1)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        system.load_state_dict(checkpoint, strict=True)


def test_validation_compatibility_does_not_hide_missing_model_weights() -> None:
    system = _minimal_system()
    checkpoint = system.state_dict()
    del checkpoint["model.weight"]
    checkpoint["validation_lpips.net.legacy_weight"] = torch.ones(1)

    with pytest.raises(RuntimeError, match="Missing key"):
        system.load_state_dict(checkpoint, strict=True)


def test_validation_compatibility_does_not_hide_model_shape_mismatch() -> None:
    system = _minimal_system()
    checkpoint = system.state_dict()
    checkpoint["model.weight"] = torch.ones(3, 3)

    with pytest.raises(RuntimeError, match="size mismatch"):
        system.load_state_dict(checkpoint, strict=True)
