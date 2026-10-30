# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for SelectiveAdam-related GaussiansComposite config helpers."""

from types import SimpleNamespace

import pytest

from omegaconf import OmegaConf

from nurec.config.model import GSplatRendererConfig
from nurec.models.gaussians.gaussians_composite import (
    assert_selective_adam_occlusion_supported,
    configure_selective_adam_renderer,
    ensure_selective_adam_renderer_outputs,
    uses_selective_adam,
)


def _optimizer_names(general: str, density: str, scale: str) -> SimpleNamespace:
    """Config stand-in exposing the three optimizer-name fields uses_selective_adam reads.

    Mirrors ModelConfig.gaussian_{,density_,scale_}optimizer_name; a namespace keeps the
    predicate tests CPU-only without constructing the full typed model config.
    """
    return SimpleNamespace(
        gaussian_optimizer_name=general,
        gaussian_density_optimizer_name=density,
        gaussian_scale_optimizer_name=scale,
    )


def test_selective_adam_renderer_outputs_handles_none_outputs():
    """Ensure missing renderer outputs are initialized with SelectiveAdam mask fields.

    Cached or minimal configs can have no output section at all; this verifies the helper
    creates the scene visibility settings needed before constructing the renderer.
    """
    renderer_cfg = GSplatRendererConfig(name="3dgut-gsplat")
    renderer_cfg.outputs = None  # type: ignore[assignment]

    ensure_selective_adam_renderer_outputs(renderer_cfg)

    assert renderer_cfg.outputs == {"scene": {"enable_cumulated_weights": True, "enable_visibility": True}}


def test_selective_adam_renderer_outputs_handles_dict_outputs():
    """Ensure dict-based renderer outputs are patched in place for SelectiveAdam.

    Some config paths still expose raw dictionaries, so this protects resume/override flows
    that would otherwise keep visibility disabled after parsing.
    """
    renderer_cfg = GSplatRendererConfig(name="3dgut-gsplat")
    renderer_cfg.outputs = {"scene": {"enable_visibility": False}}  # type: ignore[assignment]

    ensure_selective_adam_renderer_outputs(renderer_cfg)

    assert renderer_cfg.outputs["scene"]["enable_visibility"] is True
    assert renderer_cfg.outputs["scene"]["enable_cumulated_weights"] is True


def test_selective_adam_renderer_outputs_handles_typed_outputs():
    """Ensure typed renderer output configs are forced to emit visibility masks.

    This covers the normal schema-backed config path used by production app configs.
    """
    renderer_cfg = GSplatRendererConfig(name="3dgut-gsplat")
    renderer_cfg.outputs.scene.enable_visibility = False
    renderer_cfg.outputs.scene.enable_cumulated_weights = False

    ensure_selective_adam_renderer_outputs(renderer_cfg)

    assert renderer_cfg.outputs.scene.enable_visibility is True
    assert renderer_cfg.outputs.scene.enable_cumulated_weights is True


@pytest.mark.parametrize(
    "config_data",
    [
        {"outputs": {"scene": {}}},
        {"outputs": {}},
        {"outputs": None},
        {},
    ],
    ids=["empty-scene", "missing-scene", "none-outputs", "missing-outputs"],
)
def test_selective_adam_renderer_outputs_handles_locked_dictconfig_shapes(config_data):
    """Ensure locked DictConfigs get the complete output path and remain locked."""
    renderer_cfg = OmegaConf.create(config_data)
    OmegaConf.set_struct(renderer_cfg, True)
    OmegaConf.set_readonly(renderer_cfg, True)

    ensure_selective_adam_renderer_outputs(renderer_cfg)

    assert renderer_cfg.outputs.scene.enable_visibility is True
    assert renderer_cfg.outputs.scene.enable_cumulated_weights is True
    assert OmegaConf.is_struct(renderer_cfg) is True
    assert OmegaConf.is_readonly(renderer_cfg) is True


def test_selective_adam_renderer_outputs_handles_locked_nested_dictconfig():
    """Ensure a typed-like wrapper can expose locked DictConfig outputs."""
    outputs = OmegaConf.create({})
    OmegaConf.set_struct(outputs, True)
    OmegaConf.set_readonly(outputs, True)
    renderer_cfg = SimpleNamespace(outputs=outputs)

    ensure_selective_adam_renderer_outputs(renderer_cfg)

    assert outputs.scene.enable_visibility is True
    assert outputs.scene.enable_cumulated_weights is True
    assert OmegaConf.is_struct(outputs) is True
    assert OmegaConf.is_readonly(outputs) is True


class TestUsesSelectiveAdam:
    """uses_selective_adam gates SelectiveAdam-specific renderer setup and validation."""

    def test_all_fused_is_false(self):
        """Fused-only configs must not trigger SelectiveAdam renderer outputs or validation."""
        config = _optimizer_names("fused_adam", "fused_adam", "fused_adam")
        assert uses_selective_adam(config) is False

    def test_general_selective_is_true(self):
        """A selective general (positions/rotations/features) optimizer requires visibility outputs."""
        config = _optimizer_names("selective_adam", "fused_adam", "fused_adam")
        assert uses_selective_adam(config) is True

    def test_density_selective_is_true(self):
        """A selective density optimizer alone still requires visibility outputs."""
        config = _optimizer_names("fused_adam", "selective_adam", "fused_adam")
        assert uses_selective_adam(config) is True

    def test_scale_selective_is_true(self):
        """A selective scale optimizer alone still requires visibility outputs."""
        config = _optimizer_names("fused_adam", "fused_adam", "selective_adam")
        assert uses_selective_adam(config) is True

    def test_missing_fields_default_to_fused(self):
        """Configs lacking the optimizer-name fields are treated as fused-only (no SelectiveAdam)."""
        assert uses_selective_adam(SimpleNamespace()) is False


def test_fused_only_config_leaves_renderer_outputs_untouched():
    """The production renderer setup leaves fused-only outputs disabled."""
    config = _optimizer_names("fused_adam", "fused_adam", "fused_adam")
    renderer_cfg = GSplatRendererConfig(name="3dgut-gsplat")
    renderer_cfg.outputs.scene.enable_visibility = False
    renderer_cfg.outputs.scene.enable_cumulated_weights = False
    config.renderer = renderer_cfg
    config.selective_adam_occlusion_aware = True

    configure_selective_adam_renderer(config)

    assert renderer_cfg.outputs.scene.enable_visibility is False
    assert renderer_cfg.outputs.scene.enable_cumulated_weights is False


def test_selective_config_enables_renderer_outputs_through_gate():
    """The production renderer setup enables outputs for any selective optimizer."""
    config = _optimizer_names("fused_adam", "fused_adam", "selective_adam")
    renderer_cfg = GSplatRendererConfig(name="3dgut-gsplat")
    renderer_cfg.outputs.scene.enable_visibility = False
    renderer_cfg.outputs.scene.enable_cumulated_weights = False
    config.renderer = renderer_cfg
    config.selective_adam_occlusion_aware = False

    configure_selective_adam_renderer(config)

    assert renderer_cfg.outputs.scene.enable_visibility is True
    assert renderer_cfg.outputs.scene.enable_cumulated_weights is True


def test_fused_only_config_skips_gsplat_occlusion_validation():
    """The production renderer setup bypasses SelectiveAdam validation for fused-only configs."""
    config = _optimizer_names("fused_adam", "fused_adam", "fused_adam")
    renderer_cfg = GSplatRendererConfig(name="3dgut-gsplat")
    config.renderer = renderer_cfg
    config.selective_adam_occlusion_aware = True

    configure_selective_adam_renderer(config)


def test_selective_config_runs_gsplat_occlusion_validation():
    """The production renderer setup validates selective GSplat configs."""
    config = _optimizer_names("selective_adam", "fused_adam", "fused_adam")
    config.renderer = GSplatRendererConfig(name="3dgut-gsplat")
    config.selective_adam_occlusion_aware = True

    with pytest.raises(AssertionError, match="selective_adam_occlusion_aware=true.*GSplat"):
        configure_selective_adam_renderer(config)


def test_selective_adam_occlusion_aware_rejects_gsplat_renderer():
    """Ensure unsupported occlusion-aware masking fails before training starts.

    GSplat currently emits plain visibility but not cumulated_weights, so it cannot support
    model.selective_adam_occlusion_aware=true.
    """
    renderer_cfg = GSplatRendererConfig(name="3dgut-gsplat")

    with pytest.raises(AssertionError, match="selective_adam_occlusion_aware=true.*GSplat"):
        assert_selective_adam_occlusion_supported(renderer_cfg, selective_adam_occlusion_aware=True)


def test_selective_adam_non_occlusion_aware_allows_gsplat_renderer():
    """Ensure the default SelectiveAdam visibility mode remains valid for GSplat."""
    renderer_cfg = GSplatRendererConfig(name="3dgut-gsplat")

    assert_selective_adam_occlusion_supported(renderer_cfg, selective_adam_occlusion_aware=False)
