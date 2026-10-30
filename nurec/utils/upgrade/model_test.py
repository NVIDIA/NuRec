# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from omegaconf import DictConfig, OmegaConf

from nurec.config.version import Version
from nurec.utils.upgrade.model import upgrade_model


def _config(version_patch: int) -> DictConfig:
    """Minimal config carrying a version stamp and the layers referenced by the migration."""
    return OmegaConf.create(
        {
            "version": {"version_major": 26, "version_minor": 4, "version_patch": version_patch},
            "model": {"layers": {"background": {}, "road": {}}},
        }
    )


def _state_dict_with_n_active_features() -> dict:
    return {
        "gaussians_nodes.background.n_active_features": torch.tensor(3),
        "gaussians_nodes.road.n_active_features": torch.tensor(3),
        # An unrelated tensor that must survive untouched.
        "gaussians_nodes.background.positions": torch.zeros(4, 3),
    }


def test_n_active_features_migrates_for_late_2604_artifact():
    """A release/26.04 artifact must still get n_active_features migrated on a newer build.

    This is the regression guard for the forward-rendering failure: with the original
    to_version=(26, 4, 111) gate, a 26.4.146 artifact skipped the migration and failed to
    load on a 26.6 build. The corrected boundary (26.5.18 -> 26.5.19, the true landing
    version of NRE-3485 on main) migrates everything older than 26.5.19.
    """
    model = _state_dict_with_n_active_features()
    upgraded = upgrade_model(model, _config(version_patch=146), version_target=Version.from_components(26, 6, 6))

    # Old keys gone, new _extra_state keys present with the carried value.
    assert "gaussians_nodes.background.n_active_features" not in upgraded
    assert "gaussians_nodes.road.n_active_features" not in upgraded
    assert upgraded["gaussians_nodes.background._extra_state"]["n_active_features"] == 3
    assert upgraded["gaussians_nodes.road._extra_state"]["n_active_features"] == 3
    # Unrelated tensors are preserved.
    assert "gaussians_nodes.background.positions" in upgraded


def test_n_active_features_migrates_for_early_artifact():
    """Older artifacts (below the gate) were already migrated; keep them working."""
    model = _state_dict_with_n_active_features()
    upgraded = upgrade_model(model, _config(version_patch=50), version_target=Version.from_components(26, 6, 6))

    assert "gaussians_nodes.background._extra_state" in upgraded
    assert "gaussians_nodes.background.n_active_features" not in upgraded


def test_n_active_features_not_migrated_when_target_is_release_2604():
    """Loading an older artifact into a release/26.04 build (target 26.4.x) must NOT migrate,
    since that build still expects the nn.Buffer form."""
    model = _state_dict_with_n_active_features()
    upgraded = upgrade_model(model, _config(version_patch=50), version_target=Version.from_components(26, 4, 146))

    assert "gaussians_nodes.background.n_active_features" in upgraded
    assert "gaussians_nodes.background._extra_state" not in upgraded
