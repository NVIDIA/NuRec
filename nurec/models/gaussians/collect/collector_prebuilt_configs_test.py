# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The collector configurations of the shipped presets must be in the pre-built kernel set.

A configuration missing from ``nurec/libs/slang_gaussians/collector/collector_prebuilt_configs``
is compiled with ``slangc`` when the first artifact needing it is loaded, which costs tens of
seconds per process. The layer mixes below mirror the production presets; the configurations
are enumerated through the collector itself so the ordering rules stay the source of truth, and
the mirrors are checked against the composed presets so they cannot drift from the YAML.
"""

import json
import unittest

from typing import Any
from unittest import mock

import torch

from python.runfiles import runfiles

from nurec.config.model import ModelConfig
from nurec.config.parse import parse_typed_config
from nurec.libs.slang_gaussians.collector import CollectorConfiguration
from nurec.models.gaussians.collect import collector_slang
from nurec.models.gaussians.collect.collector import (
    DensityActivation,
    EmbeddingConfig,
    HolisticRemapTimeInputEmbeddingConfig,
    IndividualRemapTimeInputEmbeddingConfig,
    LayerConfigDeformable,
    LayerConfigRigid,
    LayerConfigSH,
    LayersConfig,
    RotationActivation,
    ScaleActivation,
)


def _layer(cls: type[LayerConfigSH], fourier_features_dim: int, embed_config: EmbeddingConfig | None) -> LayerConfigSH:
    return cls(
        rotation_activation=RotationActivation.NORMALIZE,
        scale_activation=ScaleActivation.EXP,
        density_activation=DensityActivation.SIGMOID,
        fourier_features_dim=fourier_features_dim,
        embed_config=embed_config,
    )


# apps/prod/Hyperion-8.1/car2sim (car2sim_6cam) layers in composition order: background (fourier 5,
# holistic time embedding), road, dynamic_rigids (fourier 20, per-track time embedding),
# dynamic_deformables, vehicle_lights (rigid, no temporal appearance) and cones (static).
_CAR2SIM_LAYERS: dict[str, LayerConfigSH] = {
    "background": _layer(
        LayerConfigSH,
        fourier_features_dim=5,
        embed_config=HolisticRemapTimeInputEmbeddingConfig(
            timestamps_us_min=0, timestamps_us_max=1_000_000, remap_min=0.0, remap_max=1.0
        ),
    ),
    "road": _layer(LayerConfigSH, fourier_features_dim=1, embed_config=None),
    "dynamic_rigids": _layer(
        LayerConfigRigid,
        fourier_features_dim=20,
        embed_config=IndividualRemapTimeInputEmbeddingConfig(
            timestamps_us_ranges=torch.tensor([[0, 1_000_000]], dtype=torch.int64),
            remap_min=0.0,
            remap_max=1.0,
        ),
    ),
    "dynamic_deformables": _layer(LayerConfigDeformable, fourier_features_dim=1, embed_config=None),
    "vehicle_lights": _layer(LayerConfigRigid, fourier_features_dim=1, embed_config=None),
    "cones": _layer(LayerConfigSH, fourier_features_dim=1, embed_config=None),
}


def _car2sim_layers_config(lidar_extra_signal_dim: int) -> LayersConfig:
    """semantic_logits (camera, 20) on every layer; fp16 SH output."""
    return LayersConfig(
        layers=list(_CAR2SIM_LAYERS.values()),
        extra_signal_dim=0,
        camera_extra_signal_dim=20,
        lidar_extra_signal_dim=lidar_extra_signal_dim,
        albedo_dim=3,
        specular_dim=45,
        sh_features_dtype="fp16",
    )


# Preset -> layers config. options/lidar_rendering: intensity_raydrop adds intensity (1) + raydrop (2).
# The sqa_* presets (Hyperion-8.1 and PAI) derive from car2sim and keep its layer mix: their
# dynamic_rigids temporal-appearance override keeps Fourier dim 20.
_PRESETS = {
    "apps/prod/Hyperion-8.1/car2sim": _car2sim_layers_config(lidar_extra_signal_dim=0),
    "apps/prod/Hyperion-8.1/car2sim_lidar": _car2sim_layers_config(lidar_extra_signal_dim=3),
    "apps/prod/Hyperion-8.1/sqa_default": _car2sim_layers_config(lidar_extra_signal_dim=0),
    "apps/prod/Hyperion-8.1/sqa_lidar_default": _car2sim_layers_config(lidar_extra_signal_dim=3),
    "apps/prod/Hyperion-8.1/sqa_lidar_1cam": _car2sim_layers_config(lidar_extra_signal_dim=3),
    "apps/prod/PAI/sqa_1cam": _car2sim_layers_config(lidar_extra_signal_dim=0),
    "apps/prod/PAI/sqa_lidar_1cam": _car2sim_layers_config(lidar_extra_signal_dim=3),
}

# Layer model name -> collector layer config built by the model's get_layer_config().
_LAYER_KINDS: dict[str, type[LayerConfigSH]] = {
    "sh-gaussians": LayerConfigSH,
    "rigid-gaussians": LayerConfigRigid,
    "vehicle-light-rigid-gaussians": LayerConfigRigid,
    "deformable-gaussians": LayerConfigDeformable,
}

_TIME_EMBEDDINGS: dict[str, type[EmbeddingConfig]] = {
    "holistic-remap-time-input-embedding": HolisticRemapTimeInputEmbeddingConfig,
    "individual-remap-time-input-embedding": IndividualRemapTimeInputEmbeddingConfig,
}


def _prebuilt_configurations() -> set[CollectorConfiguration]:
    path = runfiles.Create().Rlocation("nurec_repo/nurec/libs/slang_gaussians/collector-default-configs.json")
    if path is None:
        raise RuntimeError("Could not locate the pre-built collector configuration mapping")
    with open(path, "r") as f:
        return {CollectorConfiguration(parameters=tuple(entry["configuration"])) for entry in json.load(f)}


def _requested_configurations(layers_config: LayersConfig) -> list[CollectorConfiguration]:
    """Configurations the collector requests for ``layers_config``, without loading any kernel."""
    requested: list[CollectorConfiguration] = []

    def record(configurations: list[CollectorConfiguration]) -> list[Any]:
        requested.extend(configurations)
        return [None] * len(configurations)

    with mock.patch.object(collector_slang, "get_slang_kernels", side_effect=record):
        collector_slang.SlangGaussianParameterCollector(layers_config, run_validation=False)
    return requested


def _composed_model_config(preset: str) -> ModelConfig:
    """Compose a shipped preset the way ``nurec.run`` does; no dataset or GPU is touched."""
    return parse_typed_config(f"{preset}.yaml", ["dataset.path=/does/not/matter", "out_dir=/does/not/matter"]).model


class TestCollectorPrebuiltConfigs(unittest.TestCase):
    def test_shipped_presets_are_precompiled(self) -> None:
        prebuilt = _prebuilt_configurations()
        for preset, layers_config in _PRESETS.items():
            with self.subTest(preset=preset):
                requested = _requested_configurations(layers_config)
                # One configuration per layer; the deformable layer also requests its rigid fallback.
                self.assertEqual(len(requested), len(layers_config.layers) + 1)
                missing = [configuration for configuration in requested if configuration not in prebuilt]
                self.assertEqual(
                    missing,
                    [],
                    f"{preset} would compile {missing} at runtime; add them to "
                    "nurec/libs/slang_gaussians/collector/collector_prebuilt_configs/",
                )

    def test_mirrors_match_composed_presets(self) -> None:
        """Every fact the mirrors encode is read back from the composed preset."""
        for preset, layers_config in _PRESETS.items():
            with self.subTest(preset=preset):
                model = _composed_model_config(preset)
                self.assertEqual(list(model.layers), list(_CAR2SIM_LAYERS))
                for (name, mirror), composed in zip(_CAR2SIM_LAYERS.items(), model.layers.values()):
                    with self.subTest(layer=name):
                        self.assertIs(type(mirror), _LAYER_KINDS[composed.name])
                        self.assertEqual(mirror.rotation_activation.name.lower(), composed.rotation_activation)
                        self.assertEqual(mirror.scale_activation.name.lower(), composed.scale_activation)
                        self.assertEqual(mirror.density_activation.name.lower(), composed.density_activation)
                        fourier_features_dim = composed.fourier_features_dim
                        assert fourier_features_dim is not None
                        self.assertEqual(mirror.fourier_features_dim, fourier_features_dim)
                        # The collector consults the time embedding only when there is temporal appearance.
                        expected_embedding: type = type(None)
                        if fourier_features_dim > 1:
                            expected_embedding = _TIME_EMBEDDINGS[composed.time_embed["name"]]
                        self.assertIs(type(mirror.embed_config), expected_embedding)
                        particle = composed.particle
                        self.assertEqual(layers_config.extra_signal_dim, particle["extra_signal_dim"])
                        self.assertEqual(layers_config.camera_extra_signal_dim, particle["camera_extra_signal_dim"])
                        # The lidar extra signal is collected SH-expanded (GaussiansComposite._get_layers_config).
                        lidar_sph_dim = (particle["lidar_extra_signal_sph_degree"] + 1) ** 2
                        self.assertEqual(
                            layers_config.lidar_extra_signal_dim, particle["lidar_extra_signal_dim"] * lidar_sph_dim
                        )
                        # sh_degree_to_specular_dim: the SH coefficients above band 0, three channels each.
                        max_degree = composed.progressive_training["max_n_features"]
                        self.assertEqual(layers_config.specular_dim, 3 * ((max_degree + 1) ** 2 - 1))
                self.assertEqual(layers_config.sh_features_dtype, getattr(model.renderer, "sh_features_dtype", "fp32"))


if __name__ == "__main__":
    unittest.main()
