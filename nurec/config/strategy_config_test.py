# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pydantic import ValidationError

from nurec.config.model import (
    DensifyConfig,
    GsplatStrategyConfig,
    MCMCStrategyConfig,
    PruneConfig,
    ResetDensityConfig,
    TamingGSMCMCStrategyConfig,
)


REQUIRED_FIELDS = {"frequency": 100, "start_iteration": 500, "end_iteration": 15_000}


def test_densify_config_defaults() -> None:
    config = DensifyConfig(**REQUIRED_FIELDS)
    assert config.frequency == 100
    assert config.start_iteration == 500
    assert config.end_iteration == 15_000
    assert config.clone_grad_threshold == 0.0002
    assert config.split_grad_threshold == 0.0002
    assert config.relative_size_threshold == 0.01
    assert config.split_n_gaussians == 2


def test_prune_config_defaults() -> None:
    config = PruneConfig(**REQUIRED_FIELDS)
    assert config.density_threshold == 0.01


def test_reset_density_config_defaults() -> None:
    config = ResetDensityConfig(**REQUIRED_FIELDS)
    assert config.new_max_density == 0.01


@pytest.mark.parametrize("cls", [DensifyConfig, PruneConfig, ResetDensityConfig])
@pytest.mark.parametrize("missing", ["frequency", "start_iteration", "end_iteration"])
def test_required_fields_are_enforced(cls: type, missing: str) -> None:
    args = {k: v for k, v in REQUIRED_FIELDS.items() if k != missing}
    with pytest.raises(ValidationError):
        cls(**args)


@pytest.mark.parametrize("cls", [DensifyConfig, PruneConfig, ResetDensityConfig])
def test_frequency_must_be_positive(cls: type) -> None:
    args = {**REQUIRED_FIELDS, "frequency": 0}
    with pytest.raises(ValidationError):
        cls(**args)


@pytest.mark.parametrize("cls", [DensifyConfig, PruneConfig, ResetDensityConfig])
@pytest.mark.parametrize("field", ["start_iteration", "end_iteration"])
def test_iteration_fields_reject_negative(cls: type, field: str) -> None:
    args = {**REQUIRED_FIELDS, field: -1}
    with pytest.raises(ValidationError):
        cls(**args)


def test_prune_density_threshold_is_bounded() -> None:
    with pytest.raises(ValidationError):
        PruneConfig(**REQUIRED_FIELDS, density_threshold=-0.1)
    with pytest.raises(ValidationError):
        PruneConfig(**REQUIRED_FIELDS, density_threshold=1.5)


def test_reset_density_new_max_density_is_bounded() -> None:
    with pytest.raises(ValidationError):
        ResetDensityConfig(**REQUIRED_FIELDS, new_max_density=-0.1)
    with pytest.raises(ValidationError):
        ResetDensityConfig(**REQUIRED_FIELDS, new_max_density=1.5)


def test_gsplat_strategy_config_all_none_path() -> None:
    """Validates the all-None path that the rest of the codebase relies on."""
    config = GsplatStrategyConfig.model_validate({"name": "gsplat"})
    assert config.densify is None
    assert config.prune is None
    assert config.reset_density is None


def test_gsplat_strategy_config_with_sub_configs() -> None:
    config = GsplatStrategyConfig.model_validate(
        {
            "name": "gsplat",
            "densify": REQUIRED_FIELDS,
            "prune": REQUIRED_FIELDS,
            "reset_density": REQUIRED_FIELDS,
        }
    )
    assert config.densify is not None and config.densify.frequency == 100
    assert config.prune is not None and config.prune.frequency == 100
    assert config.reset_density is not None and config.reset_density.frequency == 100


def test_mcmc_same_track_layer_ids_reject_duplicates() -> None:
    with pytest.raises(ValidationError, match="same_track_layer_ids must not contain duplicates"):
        MCMCStrategyConfig.model_validate(
            {
                "name": "mcmc",
                "relocate": {
                    "same_track_layer_ids": ["vehicle_lights", "vehicle_lights"],
                },
            }
        )


def test_taminggs_mcmc_has_bounded_scoring_defaults() -> None:
    taminggs = TamingGSMCMCStrategyConfig.model_validate({"name": "taminggs_mcmc"}).add.taminggs
    assert taminggs.current_gradient_weight == 50.0
    assert taminggs.gradient_ema_weight == 10.0
    assert taminggs.opacity_weight == 100.0
    assert taminggs.scale_weight == 25.0
    assert taminggs.visibility_weight == 5.0
    assert taminggs.normalization_clip == 10.0
    assert taminggs.minimum_source_weight == 1.0e-6


def test_mcmc_taminggs_rejects_all_zero_or_invalid_score_controls() -> None:
    with pytest.raises(ValidationError, match="at least one positive score weight"):
        TamingGSMCMCStrategyConfig.model_validate(
            {
                "name": "taminggs_mcmc",
                "add": {
                    "taminggs": {
                        "current_gradient_weight": 0.0,
                        "gradient_ema_weight": 0.0,
                        "opacity_weight": 0.0,
                        "scale_weight": 0.0,
                        "visibility_weight": 0.0,
                    }
                },
            }
        )
    for field, value in (("current_gradient_weight", -1.0), ("normalization_clip", 0.0)):
        with pytest.raises(ValidationError):
            TamingGSMCMCStrategyConfig.model_validate({"name": "taminggs_mcmc", "add": {"taminggs": {field: value}}})


def test_mcmc_max_scale_guard_is_disabled_by_default() -> None:
    config = MCMCStrategyConfig.model_validate({"name": "mcmc"})
    assert not config.max_scale_guard.enabled
    assert config.max_scale_guard.frequency == 1
    assert config.max_scale_guard.relative_max_scale_to_scene_extent is None
    assert config.max_scale_guard.absolute_max_scale_m == 3.0
    assert config.max_scale_guard.relative_max_scale_to_rigid_cuboid == 1.0
    assert config.max_scale_guard.global_limit_layer_ids == ["background"]


@pytest.mark.parametrize(
    "max_scale_guard",
    [
        {"frequency": 0},
        {"relative_max_scale_to_scene_extent": -0.1},
        {"absolute_max_scale_m": -0.1},
        {"relative_max_scale_to_rigid_cuboid": -0.1},
        {"global_limit_layer_ids": []},
        {
            "enabled": True,
            "relative_max_scale_to_scene_extent": None,
            "absolute_max_scale_m": None,
            "relative_max_scale_to_rigid_cuboid": None,
            "global_limit_layer_ids": None,
        },
        {
            "relative_max_scale_to_scene_extent": None,
            "absolute_max_scale_m": None,
            "relative_max_scale_to_rigid_cuboid": None,
            "global_limit_layer_ids": ["background"],
        },
    ],
)
def test_mcmc_max_scale_guard_rejects_invalid_values(max_scale_guard: dict) -> None:
    with pytest.raises(ValidationError):
        MCMCStrategyConfig.model_validate({"name": "mcmc", "max_scale_guard": max_scale_guard})


@pytest.mark.parametrize("name,schema", [("mcmc", MCMCStrategyConfig), ("taminggs_mcmc", TamingGSMCMCStrategyConfig)])
def test_mcmc_strategy_names_select_exclusive_schemas(
    name: str, schema: type[MCMCStrategyConfig | TamingGSMCMCStrategyConfig]
) -> None:
    config = schema.model_validate({"name": name})
    assert config.opacity_threshold == 0.005
    assert ("taminggs" in config.add.model_dump()) == (name == "taminggs_mcmc")
    other_name = "mcmc" if name == "taminggs_mcmc" else "taminggs_mcmc"
    with pytest.raises(ValidationError):
        schema.model_validate({"name": other_name})


def test_ordinary_mcmc_rejects_taminggs_scoring() -> None:
    with pytest.raises(ValidationError):
        MCMCStrategyConfig.model_validate({"name": "mcmc", "add": {"taminggs": {}}})


def test_taminggs_scoring_rejects_removed_enable_flag() -> None:
    with pytest.raises(ValidationError):
        TamingGSMCMCStrategyConfig.model_validate({"name": "taminggs_mcmc", "add": {"taminggs": {"enabled": True}}})


@pytest.mark.parametrize("cap", [0.0, 1.0])
def test_gsplat_reset_cap_endpoints(cap: float) -> None:
    values = {
        "name": "gsplat",
        "reset_density": {**REQUIRED_FIELDS, "new_max_density": cap},
    }
    with pytest.raises(ValidationError, match="strictly between 0 and 1"):
        GsplatStrategyConfig.model_validate(values)
