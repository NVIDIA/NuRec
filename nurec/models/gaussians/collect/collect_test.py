# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib
import os
import tempfile
import time
import unittest

from collections import namedtuple
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Literal, cast
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from omegaconf import OmegaConf
from python.runfiles import runfiles

from nurec.config.trainer import TrainerConfig
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.libs.slang_gaussians.collector import CollectorConfiguration, get_slang_kernels, load_prebuilt_configs
from nurec.libs.slang_gaussians.collector import collector as collector_module
from nurec.libs.slang_utils.utils import profile
from nurec.models.gaussians.collect import (
    CollectorResult,
    CreateGaussianParameterCollector,
    DensityActivation,
    DirectTracksCalibData,
    IndividualRemapTimeInputEmbeddingConfig,
    IndividualRemapTimeInputEmbeddingData,
    IndividualStepTimeInputEmbeddingConfig,
    IndividualStepTimeInputEmbeddingData,
    InputEmbeddingData,
    LayerConfigDeformable,
    LayerConfigRigid,
    LayerConfigSH,
    LayerDataDeformable,
    LayerDataRigid,
    LayerDataSH,
    LayersConfig,
    LayersData,
    RotationActivation,
    ScaleActivation,
    SceneContractorData,
    TracksInterpolationData,
    TracksTimestampsEstimationData,
    TracksTimestampsGlobalData,
    TracksTimestampsPerTrackData,
    WeightedInstanceInputEmbeddingData,
    swap_time_embedding_table,
    whole_layer_segments,
    windowed_layer_segments,
)
from nurec.models.tracks_calib import ElevationAwareUnicycleTracksCalib
from nurec.utils.types import AABB3D, SceneContractor


device = torch.device("cuda")
# Deterministic random for reproducibility
torch.manual_seed(123)

_generate_collector_code_for_jit = collector_module.generate_collector_code


class TestCollector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Register test-only AOT specializations after the normal production set.

        Detects unexpected JIT compilations, informing the kernel developer they should add it
        to the prebuilt configs to avoid slowing down the test suite.
        """
        get_slang_kernels([])

        test_slang_collector = importlib.import_module("nurec.models.gaussians.collect.libcollector_test_slang_cc")
        mapping_path = runfiles.Create().Rlocation(
            "nurec_repo/nurec/models/gaussians/collect/collector-test-configs.json"
        )
        if mapping_path is None:
            raise RuntimeError("Could not locate the test collector configuration mapping")
        # Test configs must remain disjoint from the production set loaded above;
        # load_prebuilt_configs rejects duplicate CollectorConfigurations.
        load_prebuilt_configs(test_slang_collector, mapping_path)

        def fail_on_unexpected_jit(configurations):
            raise RuntimeError(
                f"Collector kernel configuration(s) {configurations!r} were not precompiled and would slow "
                "collect_test down by triggering SlangTorch JIT compilation. Add the missing configuration to "
                "nurec/models/gaussians/collect/collector_test_prebuilt_configs.json so that "
                "//nurec/models/gaussians/collect:collector_test_slang_cc builds it AOT."
            )

        cls.jit_guard = mock.patch.object(
            collector_module,
            "generate_collector_code",
            side_effect=fail_on_unexpected_jit,
        )
        cls.jit_guard.start()

    @classmethod
    def tearDownClass(cls):
        cls.jit_guard.stop()

    def _apply_scene_contraction(
        self, xyzs: torch.Tensor, instance_idx: torch.Tensor, scene_contractor_config: SceneContractorData
    ) -> torch.Tensor:
        """Apply scene contraction using SceneContractor.to_contracted_space().

        We use the Python SceneContractor implementation here for ground truth comparison.
        """
        # Create SceneContractor from config
        aabb = AABB3D(scene_contractor_config.aabb_blb, scene_contractor_config.aabb_trf)
        scene_contractor = SceneContractor(
            degree=scene_contractor_config.degree,
            aabb=aabb,
            is_single=False,  # Multiple instances
            is_merf=scene_contractor_config.is_merf,
        )

        # Apply per-sample contraction by indexing the contractor per instance
        contracted_xyzs = scene_contractor[instance_idx].to_contracted_space(xyzs)

        # Clamp to [0, 1] as done in Slang kernel
        return torch.clamp(contracted_xyzs, 0.0, 1.0)

    def test_errors(self):
        layers_config = LayersConfig(
            layers=[
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
            ],
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
        )

        collector = CreateGaussianParameterCollector(layers_config)

        nb_gaussians = 1000
        layers_data = LayersData(
            layers=[
                LayerDataSH(
                    positions=torch.randn(nb_gaussians, 3, device=device),
                    rotations=torch.randn(nb_gaussians, 4, device=device),
                    scales=torch.randn(nb_gaussians, 3, device=device),
                    densities=torch.randn(nb_gaussians, 1, device=device),
                    extra_signal=torch.randn(nb_gaussians, 0, device=device),
                    camera_extra_signal=torch.randn(nb_gaussians, 20, device=device),
                    lidar_extra_signal=torch.randn(nb_gaussians, 0, device=device),
                    features_albedo=torch.randn(nb_gaussians, 3, device=device),
                    features_specular=torch.randn(nb_gaussians, 45, device=device),
                    embed_data=None,
                )
            ],
            frame_timestamp_us=None,
        )

        # Test successful collection.
        results = collector.collect(layers_data)
        self.assertEqual(results.positions.shape, layers_data.layers[0].positions.shape)
        self.assertEqual(results.rotations.shape, layers_data.layers[0].rotations.shape)
        self.assertEqual(results.scales.shape, layers_data.layers[0].scales.shape)
        self.assertEqual(results.densities.shape, layers_data.layers[0].densities.shape)
        self.assertEqual(results.extra_signal.shape, layers_data.layers[0].extra_signal.shape)
        self.assertEqual(results.camera_extra_signal.shape, layers_data.layers[0].camera_extra_signal.shape)
        self.assertEqual(results.lidar_extra_signal.shape, layers_data.layers[0].lidar_extra_signal.shape)
        self.assertEqual(
            results.features.shape,
            (
                nb_gaussians,
                layers_data.layers[0].features_albedo.shape[1] + layers_data.layers[0].features_specular.shape[1],
            ),
        )

        tensor_attributes = [
            field.name
            for field in fields(LayerDataSH)
            if isinstance(getattr(layers_data.layers[0], field.name), torch.Tensor)
        ]

        # For mismatch in number of gaussians.
        for name in tensor_attributes:
            layers_data_modified = copy.deepcopy(layers_data)
            modified_value = getattr(layers_data_modified.layers[0], name)
            modified_value = modified_value[:-1]
            setattr(layers_data_modified.layers[0], name, modified_value)
            with self.assertRaises(AssertionError):
                collector.collect(layers_data_modified)

        # For mismatch in component dimensions.
        for name in tensor_attributes:
            layers_data_modified = copy.deepcopy(layers_data)
            modified_value = getattr(layers_data_modified.layers[0], name)
            if modified_value.shape[1] == 0:
                continue
            modified_value = modified_value[:, :-1]
            setattr(layers_data_modified.layers[0], name, modified_value)
            with self.assertRaises(AssertionError):
                collector.collect(layers_data_modified)

        # For zero-sized layers:
        layer_data_zeroed = copy.deepcopy(layers_data)
        for name in tensor_attributes:
            modified_value = getattr(layer_data_zeroed.layers[0], name)
            modified_value = modified_value[0:0]
            setattr(layer_data_zeroed.layers[0], name, modified_value)
        results = collector.collect(layer_data_zeroed)
        for field in fields(results):
            result = getattr(results, field.name)
            self.assertEqual(result.shape[0], 0)

    def test_base_layer(self):
        # Forward/backward block boundaries, partial warps and nonzero layer offsets.
        layers_nb_gaussians = [1, 127, 128, 129, 255, 256, 257, 100000]
        layers_config = LayersConfig(
            layers=[
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
            ]
            * len(layers_nb_gaussians),
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
        )

        collector = CreateGaussianParameterCollector(layers_config)

        layers_data = LayersData(
            layers=[
                LayerDataSH(
                    positions=torch.randn(nb_gaussians, 6, device=device)[:, ::2],
                    rotations=torch.randn(nb_gaussians, 4, device=device),
                    scales=torch.randn(nb_gaussians, 3, device=device),
                    densities=torch.randn(nb_gaussians, 1, device=device),
                    extra_signal=torch.randn(nb_gaussians, 0, device=device),
                    camera_extra_signal=torch.randn(nb_gaussians, 20, device=device),
                    lidar_extra_signal=torch.randn(nb_gaussians, 0, device=device),
                    features_albedo=torch.randn(nb_gaussians, 3, device=device),
                    features_specular=torch.randn(nb_gaussians, 45, device=device),
                    embed_data=None,
                )
                for nb_gaussians in layers_nb_gaussians
            ],
            frame_timestamp_us=None,
        )

        # Test successful collection.
        results = collector.collect(layers_data)

        offset = 0
        for i in range(len(layers_nb_gaussians)):
            nb_gaussians = layers_nb_gaussians[i]

            positions = results.positions[offset : offset + nb_gaussians]
            original_positions = layers_data.layers[i].positions
            self.assertTrue(torch.equal(positions, original_positions))

            rotations = results.rotations[offset : offset + nb_gaussians]
            original_rotations = torch.nn.functional.normalize(layers_data.layers[i].rotations)
            self.assertTrue(torch.allclose(rotations, original_rotations))

            scales = results.scales[offset : offset + nb_gaussians]
            original_scales = torch.exp(layers_data.layers[i].scales)
            self.assertTrue(torch.allclose(scales, original_scales))

            densities = results.densities[offset : offset + nb_gaussians]
            original_densities = torch.sigmoid(layers_data.layers[i].densities)
            self.assertTrue(torch.allclose(densities, original_densities))

            extra_signal = results.extra_signal[offset : offset + nb_gaussians]
            original_extra_signal = layers_data.layers[i].extra_signal
            self.assertTrue(torch.equal(extra_signal, original_extra_signal))

            camera_extra_signal = results.camera_extra_signal[offset : offset + nb_gaussians]
            original_camera_extra_signal = layers_data.layers[i].camera_extra_signal
            self.assertTrue(torch.equal(camera_extra_signal, original_camera_extra_signal))

            lidar_extra_signal = results.lidar_extra_signal[offset : offset + nb_gaussians]
            original_lidar_extra_signal = layers_data.layers[i].lidar_extra_signal
            self.assertTrue(torch.equal(lidar_extra_signal, original_lidar_extra_signal))

            features = results.features[offset : offset + nb_gaussians]
            original_features = torch.cat(
                [layers_data.layers[i].features_albedo, layers_data.layers[i].features_specular], dim=1
            )
            self.assertTrue(torch.equal(features, original_features))

            offset += nb_gaussians

    def test_backward_pass(self):
        layers_nb_gaussians = [1, 127, 128, 129, 255, 256, 257, 100000]
        layers_config = LayersConfig(
            layers=[
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
            ]
            * len(layers_nb_gaussians),
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
        )

        collector = CreateGaussianParameterCollector(layers_config)

        layers_data = LayersData(
            layers=[
                LayerDataSH(
                    positions=torch.nn.Parameter(torch.randn(nb_gaussians, 3, device=device)),
                    rotations=torch.nn.Parameter(torch.randn(nb_gaussians, 4, device=device)),
                    scales=torch.nn.Parameter(torch.randn(nb_gaussians, 3, device=device)),
                    densities=torch.nn.Parameter(torch.randn(nb_gaussians, 1, device=device)),
                    extra_signal=torch.nn.Parameter(torch.randn(nb_gaussians, 0, device=device)),
                    camera_extra_signal=torch.nn.Parameter(torch.randn(nb_gaussians, 20, device=device)),
                    lidar_extra_signal=torch.nn.Parameter(torch.randn(nb_gaussians, 0, device=device)),
                    features_albedo=torch.nn.Parameter(torch.randn(nb_gaussians, 3, device=device)),
                    features_specular=torch.nn.Parameter(torch.randn(nb_gaussians, 45, device=device)),
                    embed_data=None,
                )
                for nb_gaussians in layers_nb_gaussians
            ],
            frame_timestamp_us=None,
        )

        results = collector.collect(layers_data)

        # Test that gradients which are just copies are correct.
        results_positions_grad = torch.randn_like(results.positions)
        results_extra_signal_grad = torch.randn_like(results.extra_signal)
        results_camera_extra_signal_grad = torch.randn_like(results.camera_extra_signal)
        results_lidar_extra_signal_grad = torch.randn_like(results.lidar_extra_signal)
        results_features_grad = torch.randn_like(results.features)

        torch.autograd.backward(
            [
                results.positions,
                results.extra_signal,
                results.camera_extra_signal,
                results.lidar_extra_signal,
                results.features,
            ],
            [
                results_positions_grad,
                results_extra_signal_grad,
                results_camera_extra_signal_grad,
                results_lidar_extra_signal_grad,
                results_features_grad,
            ],
        )

        offset = 0
        for i in range(len(layers_nb_gaussians)):
            nb_gaussians = layers_nb_gaussians[i]
            layer_data = layers_data.layers[i]

            results_positions = results_positions_grad[offset : offset + nb_gaussians]
            self.assertTrue(torch.equal(layer_data.positions.grad, results_positions))

            results_camera_extra_signal = results_camera_extra_signal_grad[offset : offset + nb_gaussians]
            self.assertTrue(torch.equal(layer_data.camera_extra_signal.grad, results_camera_extra_signal))

            results_features_albedo = results_features_grad[offset : offset + nb_gaussians, :3]
            self.assertTrue(torch.equal(layer_data.features_albedo.grad, results_features_albedo))
            results_features_specular = results_features_grad[offset : offset + nb_gaussians, 3:]
            self.assertTrue(torch.equal(layer_data.features_specular.grad, results_features_specular))

            offset += nb_gaussians

        # The fused distributed gather relies on CollectorFunction keeping
        # every selected layer active on every rank. Even when downstream
        # indexing touches rows from only one layer, the collector's single
        # concatenated output must return explicit zero gradients for the
        # other layers rather than disconnecting their autograd paths.
        for layer_data in layers_data.layers:
            for tensor in (
                layer_data.positions,
                layer_data.rotations,
                layer_data.scales,
                layer_data.densities,
                layer_data.extra_signal,
                layer_data.camera_extra_signal,
                layer_data.lidar_extra_signal,
                layer_data.features_albedo,
                layer_data.features_specular,
            ):
                tensor.grad = None

        partial_results = collector.collect(layers_data)
        partial_results.positions[: layers_nb_gaussians[0]].sum().backward()
        for layer_index, layer_data in enumerate(layers_data.layers):
            self.assertIsNotNone(layer_data.positions.grad)
            expected = (
                torch.ones_like(layer_data.positions) if layer_index == 0 else torch.zeros_like(layer_data.positions)
            )
            torch.testing.assert_close(layer_data.positions.grad, expected, rtol=0, atol=0)

    def test_backward_pass_sh_fp16(self):
        """Smoke test for the SH fp16 middle step.

        sh_features_dtype="fp16": albedo/specular Parameters stay fp32 (no Parameter dtype
        change), but the collector's flat output buffer is fp16. The kernel reads fp32
        inputs, computes in fp32, and writes fp16 via store_half.

        Verifies: the fp16 AOT specialization executes, the flat output is fp16, and
        Parameter gradients flow back as fp32 finite values close to the fp32 baseline
        within the precision budget of fp16 round-trip on the output gradient.
        """
        layers_nb_gaussians = [1, 127, 128, 129, 255, 256, 257, 100000]
        for sh_features_dtype, expected_out_dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
            layers_config = LayersConfig(
                layers=[
                    LayerConfigSH(
                        rotation_activation=RotationActivation.NORMALIZE,
                        scale_activation=ScaleActivation.EXP,
                        density_activation=DensityActivation.SIGMOID,
                        fourier_features_dim=1,
                        embed_config=None,
                    ),
                ]
                * len(layers_nb_gaussians),
                extra_signal_dim=0,
                camera_extra_signal_dim=0,
                lidar_extra_signal_dim=0,
                albedo_dim=3,
                specular_dim=45,
                sh_features_dtype=sh_features_dtype,
            )

            collector = CreateGaussianParameterCollector(layers_config)

            # Use fixed seed so fp32 and fp16 paths see the same inputs.
            torch.manual_seed(42)
            layers_data = LayersData(
                layers=[
                    LayerDataSH(
                        positions=torch.nn.Parameter(torch.randn(nb, 3, device=device)),
                        rotations=torch.nn.Parameter(torch.randn(nb, 4, device=device)),
                        scales=torch.nn.Parameter(torch.randn(nb, 3, device=device)),
                        densities=torch.nn.Parameter(torch.randn(nb, 1, device=device)),
                        extra_signal=torch.nn.Parameter(torch.randn(nb, 0, device=device)),
                        camera_extra_signal=torch.nn.Parameter(torch.randn(nb, 0, device=device)),
                        lidar_extra_signal=torch.nn.Parameter(torch.randn(nb, 0, device=device)),
                        features_albedo=torch.nn.Parameter(torch.randn(nb, 3, device=device)),
                        features_specular=torch.nn.Parameter(torch.randn(nb, 45, device=device)),
                        embed_data=None,
                    )
                    for nb in layers_nb_gaussians
                ],
                frame_timestamp_us=None,
            )

            results = collector.collect(layers_data)

            # Output dtype must match the configured sh_features_dtype.
            self.assertEqual(results.features.dtype, expected_out_dtype, msg=f"sh_features_dtype={sh_features_dtype}")

            # Backward with a fixed grad on the features so both paths see the same upstream signal.
            torch.manual_seed(99)
            features_grad = torch.randn_like(results.features)
            results.features.backward(features_grad)

            # Parameter grads exist, are fp32, and finite.
            offset = 0
            for layer_data in layers_data.layers:
                end = offset + layer_data.positions.shape[0]
                expected = torch.cat([layer_data.features_albedo, layer_data.features_specular], dim=1)
                torch.testing.assert_close(
                    results.features[offset:end], expected.to(expected_out_dtype), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    layer_data.features_albedo.grad, features_grad[offset:end, :3].float(), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    layer_data.features_specular.grad, features_grad[offset:end, 3:].float(), rtol=0, atol=0
                )
                offset = end
                self.assertIsNotNone(layer_data.features_albedo.grad)
                self.assertEqual(layer_data.features_albedo.grad.dtype, torch.float32)
                self.assertTrue(torch.isfinite(layer_data.features_albedo.grad).all())
                self.assertIsNotNone(layer_data.features_specular.grad)
                self.assertEqual(layer_data.features_specular.grad.dtype, torch.float32)
                self.assertTrue(torch.isfinite(layer_data.features_specular.grad).all())

    def test_fp16_jit_compilation_smoke(self):
        """A unique fp16 collector specialization compiled via JIT to make sure the JIT path works."""
        configuration = CollectorConfiguration(parameters=("SphericalFeaturesCollector_Copy_FP16<3,45>",))
        self.assertNotIn(
            configuration,
            collector_module._cached_kernels,
            f"{configuration.parameters[0]} is intentionally excluded from AOT compilation to smoke-test JIT "
            "code generation. Your changes appear to have added it to an AOT list; remove it from AOT "
            "compilation or choose a new configuration for this JIT smoke test.",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch.object(collector_module, "generate_collector_code", _generate_collector_code_for_jit),
                mock.patch.object(collector_module.tempfile, "gettempdir", return_value=temp_dir),
            ):
                collector_kernel = get_slang_kernels([configuration])[0]

            self.assertEqual(len(collector_kernel.kernels), 2)
            self.assertTrue(all(kernel is not None for kernel in collector_kernel.kernels))
            self.assertIsNotNone(collector_kernel.slang_module)

            compilation_source_dir = Path(temp_dir)
            if os.environ.get("CUDA_COVERAGE_MODE") == "1":
                # CUDA_COVERAGE_MODE overrides the compilation source dir
                if coverage_output_dir := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
                    compilation_source_dir = Path(coverage_output_dir)

            generated_sources = list(compilation_source_dir.glob("collector_*.slang"))
            self.assertEqual(len(generated_sources), 1)
            self.assertTrue(any(path.name == ".slangtorch_cache" for path in compilation_source_dir.rglob("*")))

    def test_backward_pass_sh_fp16_fourier(self):
        """fp16 Fourier grads match fp32 within budget, for IndividualRemap and IndividualStep.

        IndividualStep routes to the _FP16 IndividualStep kernel and has a trainable u, checked too.
        """
        for nb_gaussians in (257, 1024):
            with self.subTest(nb_gaussians=nb_gaussians):
                self._check_backward_pass_sh_fp16_fourier(nb_gaussians)

    def _check_backward_pass_sh_fp16_fourier(self, nb_gaussians):
        """Check Fourier output and gradient parity for one layer size."""
        fourier_dim = 20
        albedo_dim = 3
        specular_dim = 45
        n_steps = 15

        timestamps_us_ranges = torch.tensor([[0, 1000000]], dtype=torch.int64, device=device)
        remap_config = IndividualRemapTimeInputEmbeddingConfig(
            timestamps_us_ranges=timestamps_us_ranges,
            remap_min=0.0,
            remap_max=1.0,
        )
        step_config = IndividualStepTimeInputEmbeddingConfig(
            timestamps_us_ranges=timestamps_us_ranges,
            n_steps=n_steps,
            n_dims=1,
        )

        # Snapshot inputs once so fp32 and fp16 runs see identical values.
        torch.manual_seed(7)
        positions_init = torch.randn(nb_gaussians, 3, device=device)
        rotations_init = torch.randn(nb_gaussians, 4, device=device)
        scales_init = torch.randn(nb_gaussians, 3, device=device)
        densities_init = torch.randn(nb_gaussians, 1, device=device)
        features_albedo_init = torch.randn(nb_gaussians, fourier_dim, albedo_dim, device=device)
        features_specular_init = torch.randn(nb_gaussians, specular_dim, device=device)
        u_init = torch.randn(1, 1, n_steps, device=device)  # [num_instances, n_dims, n_steps]

        def make_embed_data(embed_config):
            instance_idx = torch.zeros(nb_gaussians, dtype=torch.int32, device=device)
            if isinstance(embed_config, IndividualStepTimeInputEmbeddingConfig):
                return IndividualStepTimeInputEmbeddingData(
                    instance_idx=instance_idx,
                    u=torch.nn.Parameter(u_init.clone()),
                    beta=torch.tensor(50000.0, device=device),
                )
            return IndividualRemapTimeInputEmbeddingData(instance_idx=instance_idx)

        def run(embed_config, sh_features_dtype: Literal["fp32", "fp16"]):
            layers_config = LayersConfig(
                layers=[
                    LayerConfigSH(
                        rotation_activation=RotationActivation.NORMALIZE,
                        scale_activation=ScaleActivation.EXP,
                        density_activation=DensityActivation.SIGMOID,
                        fourier_features_dim=fourier_dim,
                        embed_config=embed_config,
                    ),
                ],
                extra_signal_dim=0,
                camera_extra_signal_dim=0,
                lidar_extra_signal_dim=0,
                albedo_dim=albedo_dim,
                specular_dim=specular_dim,
                sh_features_dtype=sh_features_dtype,
            )
            collector = CreateGaussianParameterCollector(layers_config)

            embed_data = make_embed_data(embed_config)
            layers_data = LayersData(
                layers=[
                    LayerDataSH(
                        positions=torch.nn.Parameter(positions_init.clone()),
                        rotations=torch.nn.Parameter(rotations_init.clone()),
                        scales=torch.nn.Parameter(scales_init.clone()),
                        densities=torch.nn.Parameter(densities_init.clone()),
                        extra_signal=torch.nn.Parameter(torch.zeros(nb_gaussians, 0, device=device)),
                        camera_extra_signal=torch.nn.Parameter(torch.zeros(nb_gaussians, 0, device=device)),
                        lidar_extra_signal=torch.nn.Parameter(torch.zeros(nb_gaussians, 0, device=device)),
                        features_albedo=torch.nn.Parameter(features_albedo_init.clone()),
                        features_specular=torch.nn.Parameter(features_specular_init.clone()),
                        embed_data=embed_data,
                    )
                ],
                frame_timestamp_us=500000,
            )

            results = collector.collect(layers_data)
            expected_dtype = torch.float16 if sh_features_dtype == "fp16" else torch.float32
            self.assertEqual(results.features.dtype, expected_dtype)

            # Fixed upstream grad cast to the output dtype, so both runs see the same signal.
            torch.manual_seed(11)
            upstream_grad_fp32 = torch.randn(*results.features.shape, device=device)
            upstream_grad = upstream_grad_fp32.to(results.features.dtype)
            results.features.backward(upstream_grad)

            layer0 = cast(LayerDataSH, layers_data.layers[0])
            albedo_grad = layer0.features_albedo.grad
            specular_grad = layer0.features_specular.grad
            assert albedo_grad is not None and specular_grad is not None
            u_grad = embed_data.u.grad if isinstance(embed_data, IndividualStepTimeInputEmbeddingData) else None
            return (
                albedo_grad.detach().clone(),
                specular_grad.detach().clone(),
                None if u_grad is None else u_grad.detach().clone(),
            )

        # Only fp16 loss is the upstream-grad round-trip (~1e-3); Fourier math runs in fp64 in both.
        for embed_config in [remap_config, step_config]:
            with self.subTest(embedding=type(embed_config).__name__):
                albedo_grad_fp32, specular_grad_fp32, u_grad_fp32 = run(embed_config, "fp32")
                albedo_grad_fp16, specular_grad_fp16, u_grad_fp16 = run(embed_config, "fp16")

                checks = [
                    ("features_albedo", albedo_grad_fp32, albedo_grad_fp16),
                    ("features_specular", specular_grad_fp32, specular_grad_fp16),
                ]
                if isinstance(embed_config, IndividualStepTimeInputEmbeddingConfig):
                    assert u_grad_fp32 is not None and u_grad_fp16 is not None, "step embedding u should be trainable"
                    checks.append(("embed_data.u", u_grad_fp32, u_grad_fp16))

                for name, ref, got in checks:
                    self.assertEqual(got.dtype, torch.float32, msg=name)
                    self.assertTrue(torch.isfinite(got).all(), msg=name)
                    max_abs = (got - ref).abs().max().item()
                    ref_scale = ref.abs().max().item()
                    rel = max_abs / max(ref_scale, 1e-8)
                    self.assertLess(
                        rel,
                        2e-3,
                        msg=f"{name} [{type(embed_config).__name__}]: fp16 grad diverges from fp32 "
                        f"(max_abs={max_abs:.3e}, ref_scale={ref_scale:.3e}, rel={rel:.3e})",
                    )

    def test_backward_pass_sh_fp16_multilayer(self):
        """Multi-layer mix mirroring Hyperion-8.1 real training: Copy + Fourier together.

        background/road use Copy (fourier_features_dim=1), dynamic_rigids uses Fourier
        (fourier_features_dim=20). Asserts every layer's Parameter grads match fp32
        within fp16-quantization budget.
        """
        albedo_dim = 3
        specular_dim = 45
        fourier_dim = 20
        layers_nb_gaussians = [2048, 1024, 512]  # background-like (Copy), road-like (Copy), rigid-like (Fourier)
        fourier_dims = [1, 1, fourier_dim]

        time_embedding_config = IndividualRemapTimeInputEmbeddingConfig(
            timestamps_us_ranges=torch.tensor([[0, 1000000]], dtype=torch.int64, device=device),
            remap_min=0.0,
            remap_max=1.0,
        )

        torch.manual_seed(13)
        layer_inits = []
        for i, nb in enumerate(layers_nb_gaussians):
            albedo_shape = (nb, fourier_dims[i], albedo_dim) if fourier_dims[i] > 1 else (nb, albedo_dim)
            layer_inits.append(
                dict(
                    positions=torch.randn(nb, 3, device=device),
                    rotations=torch.randn(nb, 4, device=device),
                    scales=torch.randn(nb, 3, device=device),
                    densities=torch.randn(nb, 1, device=device),
                    features_albedo=torch.randn(*albedo_shape, device=device),
                    features_specular=torch.randn(nb, specular_dim, device=device),
                    instance_idx=torch.zeros(nb, dtype=torch.int32, device=device),
                )
            )

        def run(sh_features_dtype: Literal["fp32", "fp16"]):
            layers_config = LayersConfig(
                layers=[
                    LayerConfigSH(
                        rotation_activation=RotationActivation.NORMALIZE,
                        scale_activation=ScaleActivation.EXP,
                        density_activation=DensityActivation.SIGMOID,
                        fourier_features_dim=fourier_dims[i],
                        embed_config=time_embedding_config if fourier_dims[i] > 1 else None,
                    )
                    for i in range(len(layers_nb_gaussians))
                ],
                extra_signal_dim=0,
                camera_extra_signal_dim=0,
                lidar_extra_signal_dim=0,
                albedo_dim=albedo_dim,
                specular_dim=specular_dim,
                sh_features_dtype=sh_features_dtype,
            )
            collector = CreateGaussianParameterCollector(layers_config)

            layers_data = LayersData(
                layers=[
                    LayerDataSH(
                        positions=torch.nn.Parameter(layer_inits[i]["positions"].clone()),
                        rotations=torch.nn.Parameter(layer_inits[i]["rotations"].clone()),
                        scales=torch.nn.Parameter(layer_inits[i]["scales"].clone()),
                        densities=torch.nn.Parameter(layer_inits[i]["densities"].clone()),
                        extra_signal=torch.nn.Parameter(torch.zeros(layers_nb_gaussians[i], 0, device=device)),
                        camera_extra_signal=torch.nn.Parameter(torch.zeros(layers_nb_gaussians[i], 0, device=device)),
                        lidar_extra_signal=torch.nn.Parameter(torch.zeros(layers_nb_gaussians[i], 0, device=device)),
                        features_albedo=torch.nn.Parameter(layer_inits[i]["features_albedo"].clone()),
                        features_specular=torch.nn.Parameter(layer_inits[i]["features_specular"].clone()),
                        embed_data=(
                            IndividualRemapTimeInputEmbeddingData(instance_idx=layer_inits[i]["instance_idx"].clone())
                            if fourier_dims[i] > 1
                            else None
                        ),
                    )
                    for i in range(len(layers_nb_gaussians))
                ],
                frame_timestamp_us=500000,
            )

            results = collector.collect(layers_data)
            torch.manual_seed(17)
            upstream_grad_fp32 = torch.randn(*results.features.shape, device=device)
            upstream_grad = upstream_grad_fp32.to(results.features.dtype)
            results.features.backward(upstream_grad)

            grads = []
            for i in range(len(layers_nb_gaussians)):
                layer_i = cast(LayerDataSH, layers_data.layers[i])
                albedo_grad = layer_i.features_albedo.grad
                specular_grad = layer_i.features_specular.grad
                assert albedo_grad is not None and specular_grad is not None
                grads.append((albedo_grad.detach().clone(), specular_grad.detach().clone()))
            return grads

        grads_fp32 = run("fp32")
        grads_fp16 = run("fp16")

        for i in range(len(layers_nb_gaussians)):
            for name, ref, got in [
                (f"layer{i}_albedo", grads_fp32[i][0], grads_fp16[i][0]),
                (f"layer{i}_specular", grads_fp32[i][1], grads_fp16[i][1]),
            ]:
                self.assertEqual(got.dtype, torch.float32, msg=name)
                self.assertTrue(torch.isfinite(got).all(), msg=name)
                max_abs = (got - ref).abs().max().item()
                ref_scale = ref.abs().max().item()
                rel = max_abs / max(ref_scale, 1e-8)
                self.assertLess(
                    rel,
                    5e-3,
                    msg=f"{name}: fp16 grad diverges from fp32 (max_abs={max_abs:.3e}, ref_scale={ref_scale:.3e}, rel={rel:.3e})",
                )

    def test_static_cache_fourier_band0_frame_variance_split(self):
        """Pins the frame-variance split the static-layer-cache inference path relies on.

        the static cache holds the collected static layers once and re-mixes only SH band-0
        per frame (the holistic-Fourier fold). That is only correct if, holding the layer
        tensors fixed, collecting at two frame timestamps changes band-0
        (features[:, :albedo_dim]) while leaving geometry, density, and the specular SH
        bands (features[:, albedo_dim:]) bit-identical -- i.e. band-0 is the ONLY
        time-dependent output, so caching everything else across frames is exact.

        Uses the precompiled fp32 IndividualRemap config of test_backward_pass_sh_fp16_fourier
        so no kernel is JIT-compiled under this class's no-unexpected-JIT guard.
        """
        nb_gaussians = 1024
        fourier_dim = 20
        albedo_dim = 3
        specular_dim = 45

        timestamps_us_ranges = torch.tensor([[0, 1000000]], dtype=torch.int64, device=device)
        remap_config = IndividualRemapTimeInputEmbeddingConfig(
            timestamps_us_ranges=timestamps_us_ranges,
            remap_min=0.0,
            remap_max=1.0,
        )
        layers_config = LayersConfig(
            layers=[
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=fourier_dim,
                    embed_config=remap_config,
                ),
            ],
            extra_signal_dim=0,
            camera_extra_signal_dim=0,
            lidar_extra_signal_dim=0,
            albedo_dim=albedo_dim,
            specular_dim=specular_dim,
        )
        collector = CreateGaussianParameterCollector(layers_config)

        # Fixed inputs shared by both frames; only frame_timestamp_us differs.
        torch.manual_seed(0)
        layer = LayerDataSH(
            positions=torch.randn(nb_gaussians, 3, device=device),
            rotations=torch.randn(nb_gaussians, 4, device=device),
            scales=torch.randn(nb_gaussians, 3, device=device),
            densities=torch.randn(nb_gaussians, 1, device=device),
            extra_signal=torch.zeros(nb_gaussians, 0, device=device),
            camera_extra_signal=torch.zeros(nb_gaussians, 0, device=device),
            lidar_extra_signal=torch.zeros(nb_gaussians, 0, device=device),
            features_albedo=torch.randn(nb_gaussians, fourier_dim, albedo_dim, device=device),
            features_specular=torch.randn(nb_gaussians, specular_dim, device=device),
            embed_data=IndividualRemapTimeInputEmbeddingData(
                instance_idx=torch.zeros(nb_gaussians, dtype=torch.int32, device=device)
            ),
        )

        # Two in-range frame timestamps that remap to distinct time embeddings.
        with torch.no_grad():
            result_t0 = collector.collect(LayersData(layers=[layer], frame_timestamp_us=250000))
            band0_t0 = result_t0.features[:, :albedo_dim].clone()
            specular_t0 = result_t0.features[:, albedo_dim:].clone()
            positions_t0 = result_t0.positions.clone()
            rotations_t0 = result_t0.rotations.clone()
            scales_t0 = result_t0.scales.clone()
            densities_t0 = result_t0.densities.clone()

            result_t1 = collector.collect(LayersData(layers=[layer], frame_timestamp_us=750000))

        # Band-0 is time-dependent -> it must be re-mixed per frame (the fold),
        # never serve a statically cached band-0.
        self.assertFalse(
            torch.equal(band0_t0, result_t1.features[:, :albedo_dim]),
            "Fourier band-0 did not change across frame timestamps; the per-frame fold would be a no-op",
        )
        # Everything the cache reuses across frames must be frame-invariant.
        self.assertTrue(
            torch.equal(specular_t0, result_t1.features[:, albedo_dim:]), "specular must be frame-invariant"
        )
        self.assertTrue(torch.equal(positions_t0, result_t1.positions), "positions must be frame-invariant")
        self.assertTrue(torch.equal(rotations_t0, result_t1.rotations), "rotations must be frame-invariant")
        self.assertTrue(torch.equal(scales_t0, result_t1.scales), "scales must be frame-invariant")
        self.assertTrue(torch.equal(densities_t0, result_t1.densities), "densities must be frame-invariant")

    def _get_standalone_collector(self):
        layers_config = LayersConfig(
            layers=[
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
            ],
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
        )

        collector = CreateGaussianParameterCollector(layers_config)
        return collector

    @staticmethod
    def _make_unicycle_calib(
        target_device: torch.device,
        track_pose_counts: tuple[int, ...] = (4, 5),
    ) -> tuple[ElevationAwareUnicycleTracksCalib, CuboidTracks]:
        def track(count: int, offset: float) -> tuple[np.ndarray, np.ndarray]:
            poses = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
            poses[:, 0, 3] = offset + np.arange(count, dtype=np.float32)
            timestamps = np.cumsum(np.array([0, *([400_000, 700_000, 500_000, 900_000] * count)])[:count])
            return poses, timestamps.astype(np.int64)

        tracks_data = [track(count, 10.0 * track_idx) for track_idx, count in enumerate(track_pose_counts)]
        tracks = CuboidTracks.Factory.from_numpy(
            tracks_id=[f"track_{track_idx}" for track_idx in range(len(tracks_data))],
            tracks_poses=[poses for poses, _ in tracks_data],
            tracks_timestamps_us=[timestamps for _, timestamps in tracks_data],
            tracks_label_class=["unknown"] * len(tracks_data),
            tracks_flags=[TrackFlags.NONE] * len(tracks_data),
            cuboids_dims=[np.ones(3, dtype=np.float32)] * len(tracks_data),
            device=target_device,
        )
        trainer_config = TrainerConfig(
            max_epochs=1,
            check_val_every_n_epoch=1,
            precision="32",
            log_every_n_steps=1,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        calib = ElevationAwareUnicycleTracksCalib(
            config=OmegaConf.create(
                {
                    "visualize": False,
                    "start_global_step": 0,
                    "fix_first_pose": True,
                    "fix_last_pose": False,
                    "pitch_slope_clamp_rad": 1.2,
                    "cuboid_tracks": {"up_axis": [0.0, 0.0, 1.0], "forward_axis": [1.0, 0.0, 0.0]},
                }
            ),
            trainer_config=trainer_config,
            cuboid_tracks=tracks,
        )

        n = calib.xy.shape[0]
        pairs = calib.velocity.shape[0]
        with torch.no_grad():
            calib.xy.add_(
                torch.linspace(-0.5, 0.5, n, device=target_device).view(-1, 1)
                * torch.tensor([[1.0, -0.5]], device=target_device)
            )
            calib.height.add_(torch.linspace(0.1, 0.4, n, device=target_device))
            calib.theta.add_(torch.linspace(-0.3, 0.7, n, device=target_device))
            calib.pitch.add_(torch.linspace(-0.2, 0.2, n, device=target_device))
            calib.velocity.add_(torch.linspace(0.2, 0.8, pairs, device=target_device))
            calib.theta_velocity.add_(torch.linspace(-0.4, 0.4, pairs, device=target_device))
            calib.pitch_velocity.add_(torch.linspace(-0.1, 0.3, pairs, device=target_device))
            calib.ground_t.add_(torch.tensor([[0.5, -1.0, 2.0]], device=target_device))
            calib.ground_q.add_(torch.tensor([[0.1, -0.2, 0.15, 0.0]], device=target_device))
            calib.Tclc_tquat.add_(torch.tensor([[0.1, -0.2, 0.3, 0.05, -0.1, 0.15, 0.02]], device=target_device))
        return calib, tracks

    def _get_test_scene_contractor(self, n_instances=1):
        """Helper method to create a test SceneContractorData."""
        # Simple AABB centered at origin with range [-1, 1] for each instance
        aabb_blb = torch.tensor([[-1.0, -1.0, -1.0]] * n_instances, device=device, dtype=torch.float32)
        aabb_trf = torch.tensor([[1.0, 1.0, 1.0]] * n_instances, device=device, dtype=torch.float32)
        return SceneContractorData(
            aabb_blb=aabb_blb,
            aabb_trf=aabb_trf,
            degree=2.0,
            is_merf=False,
        )

    def _check_tracks_calib(self, left_multiply: bool) -> None:
        # Create the fake rotations.
        tracks_nb_poses = [123, 456, 789]
        total_count = sum(tracks_nb_poses)

        gradient_mask = torch.full((total_count,), True, dtype=torch.bool, device=device)
        offset = 0
        for count in tracks_nb_poses:
            gradient_mask[offset] = False
            gradient_mask[offset + count - 1] = False
            offset += count
        tracks_delta_q = torch.nn.functional.normalize(torch.randn(total_count, 4, device=device))
        tracks_delta_q.requires_grad = True
        tracks_delta_t = torch.randn(total_count, 3, device=device)
        tracks_delta_t.requires_grad = True
        tracks_poses = torch.zeros(total_count, 4, device=device)

        tracks_q = torch.nn.functional.normalize(torch.randn(total_count, 4, device=device))
        tracks_t = torch.randn(total_count, 3, device=device)
        tracks_poses = torch.cat([tracks_t, tracks_q], dim=1)

        # Compute the ground truth poses.
        import lietorch as lt

        current_tracks_delta_q = torch.where(gradient_mask.view(-1, 1), tracks_delta_q, tracks_delta_q.detach())
        current_tracks_delta_t = torch.where(gradient_mask.view(-1, 1), tracks_delta_t, tracks_delta_t.detach())

        current_tracks_delta_transform = lt.SE3.InitFromVec(
            torch.cat(
                [current_tracks_delta_t, current_tracks_delta_q / current_tracks_delta_q.norm(dim=1, keepdim=True)],
                dim=1,
            )
        )

        tracks_transform = lt.SE3.InitFromVec(tracks_poses)
        ground_truth_poses = (
            current_tracks_delta_transform * tracks_transform
            if left_multiply
            else tracks_transform * current_tracks_delta_transform
        )
        ground_truth_poses = ground_truth_poses.vec()

        # Calibrate the poses.
        collector = self._get_standalone_collector()

        tracks_calib_data = DirectTracksCalibData(
            gradient_mask=gradient_mask,
            tracks_delta_q=tracks_delta_q,
            tracks_delta_t=tracks_delta_t,
            left_multiply=left_multiply,
        )
        output_poses = collector.calibrate_tracks_poses(tracks_poses, tracks_calib_data)

        self.assertTrue(torch.allclose(output_poses, ground_truth_poses, atol=1e-5))

        # Backward pass.
        ground_truth_poses_grad = torch.randn_like(output_poses)
        output_poses.backward(ground_truth_poses_grad)
        input_grad_delta_t = tracks_delta_t.grad
        input_grad_delta_q = tracks_delta_q.grad

        tracks_delta_t.grad = None
        tracks_delta_q.grad = None

        ground_truth_poses.backward(ground_truth_poses_grad)

        ground_truth_input_grad_delta_t = tracks_delta_t.grad
        ground_truth_input_grad_delta_q = tracks_delta_q.grad

        assert input_grad_delta_t is not None
        assert input_grad_delta_q is not None
        assert ground_truth_input_grad_delta_t is not None
        assert ground_truth_input_grad_delta_q is not None

        # Since we don't implement tangent space propagation yet, to match lietorch's
        # results, we need to remove the parts of the gradient orthogonal to the tangent space.
        # This is done by projecting the gradient onto the tangent space.
        input_grad_delta_q = (
            input_grad_delta_q - (input_grad_delta_q * tracks_delta_q).sum(dim=1, keepdim=True) * tracks_delta_q
        )

        self.assertTrue(torch.allclose(input_grad_delta_t, ground_truth_input_grad_delta_t, atol=1e-5))
        self.assertTrue(torch.allclose(input_grad_delta_q, ground_truth_input_grad_delta_q, atol=1e-5))

    def test_tracks_calib(self) -> None:
        for left_multiply in (True, False):
            with self.subTest(left_multiply=left_multiply):
                self._check_tracks_calib(left_multiply)

    def test_unicycle_tracks_calib_matches_torch_reference(self) -> None:
        cpu_calib, cpu_tracks = self._make_unicycle_calib(torch.device("cpu"))
        gpu_calib = copy.deepcopy(cpu_calib).to(device)
        gpu_tracks = cpu_tracks.to_device(device)
        collector = self._get_standalone_collector()
        data = gpu_calib.collect_data()

        gpu_pose = gpu_calib._get_pose_from_collector(
            lambda: collector.calibrate_tracks_poses(gpu_tracks.tracks_poses.vec(), data)
        ).vec()
        gpu_residuals = collector.compute_tracks_calib_residuals(data)
        comparisons = (
            ("pose", cpu_calib.get_pose().vec(), gpu_pose, ("xy", "height", "theta", "pitch")),
            (
                "consistency",
                cpu_calib.get_consistency(),
                gpu_residuals.consistency,
                ("xy", "height", "theta", "pitch", "velocity", "theta_velocity", "pitch_velocity"),
            ),
            (
                "smoothness",
                cpu_calib.get_smoothness(),
                gpu_residuals.smoothness,
                ("height", "velocity", "theta_velocity", "pitch_velocity"),
            ),
            (
                "regularization",
                cpu_calib.get_pose_regularization(),
                gpu_residuals.pose_regularization,
                ("xy", "height"),
            ),
        )
        frozen_rows = tuple(int(start) for start in gpu_tracks.tracks_packinfo[:, 0].cpu())

        for name, reference, actual, channels in comparisons:
            self._assert_unicycle_output_vjp_matches(
                name,
                reference,
                actual,
                cpu_calib,
                gpu_calib,
                channels,
                frozen_rows,
            )

    def test_unicycle_residuals_cross_launch_blocks(self) -> None:
        # Use non-aligned counts that cross every residual kernel's launch boundary.
        cpu_calib, cpu_tracks = self._make_unicycle_calib(
            torch.device("cpu"),
            track_pose_counts=(131, 137),
        )
        gpu_calib = copy.deepcopy(cpu_calib).to(device)
        gpu_tracks = cpu_tracks.to_device(device)
        collector = self._get_standalone_collector()
        gpu_residuals = collector.compute_tracks_calib_residuals(gpu_calib.collect_data())
        comparisons = (
            (
                "consistency",
                cpu_calib.get_consistency(),
                gpu_residuals.consistency,
                ("xy", "height", "theta", "pitch", "velocity", "theta_velocity", "pitch_velocity"),
            ),
            (
                "smoothness",
                cpu_calib.get_smoothness(),
                gpu_residuals.smoothness,
                ("height", "velocity", "theta_velocity", "pitch_velocity"),
            ),
            (
                "regularization",
                cpu_calib.get_pose_regularization(),
                gpu_residuals.pose_regularization,
                ("xy", "height"),
            ),
        )
        frozen_rows = tuple(int(start) for start in gpu_tracks.tracks_packinfo[:, 0].cpu())

        for name, reference, actual, channels in comparisons:
            self._assert_unicycle_output_vjp_matches(
                name,
                reference,
                actual,
                cpu_calib,
                gpu_calib,
                channels,
                frozen_rows,
            )

    def test_unicycle_slang_returns_empty_smoothness_for_zero_triplets(self) -> None:
        calib, tracks = self._make_unicycle_calib(device, track_pose_counts=(2, 2))
        data = calib.collect_data()

        smoothness = self._get_standalone_collector().compute_tracks_calib_residuals(data).smoothness

        self.assertEqual(smoothness.shape, (0, 4))
        channels = (calib.height, calib.velocity, calib.theta_velocity, calib.pitch_velocity)
        grads = torch.autograd.grad(smoothness.sum(), channels)
        for grad in grads:
            torch.testing.assert_close(grad, torch.zeros_like(grad), rtol=0, atol=0)

    def _assert_unicycle_output_vjp_matches(
        self,
        name: str,
        reference: torch.Tensor,
        actual: torch.Tensor,
        cpu_calib: ElevationAwareUnicycleTracksCalib,
        gpu_calib: ElevationAwareUnicycleTracksCalib,
        channels: tuple[str, ...],
        frozen_rows: tuple[int, ...],
    ) -> None:
        self.assertEqual(reference.shape, actual.shape, name)
        self.assertLess((reference - actual.cpu()).abs().max().item(), 1e-4, name)

        cpu_cotangent = torch.linspace(-0.75, 1.25, reference.numel(), dtype=reference.dtype).reshape_as(reference)
        gpu_cotangent = cpu_cotangent.to(device)
        cpu_grads = torch.autograd.grad(
            reference,
            [getattr(cpu_calib, channel) for channel in channels],
            grad_outputs=cpu_cotangent,
            retain_graph=True,
        )
        gpu_grads = torch.autograd.grad(
            actual,
            [getattr(gpu_calib, channel) for channel in channels],
            grad_outputs=gpu_cotangent,
            retain_graph=True,
        )
        for channel, reference_grad, actual_grad in zip(channels, cpu_grads, gpu_grads, strict=True):
            self.assertLess(
                (reference_grad - actual_grad.cpu()).abs().max().item(),
                1e-3,
                f"{name}.{channel}",
            )
            if channel in {"xy", "height", "theta", "pitch"}:
                for frozen_row in frozen_rows:
                    self.assertEqual(actual_grad[frozen_row].abs().max().item(), 0.0, f"{name}.{channel}")

    def test_unicycle_regularization_preserves_shared_grad_output(self) -> None:
        left, _ = self._make_unicycle_calib(device)
        right, _ = self._make_unicycle_calib(device)
        with torch.no_grad():
            right.xy.mul_(2.0)
            right.height.mul_(2.0)

        collector = self._get_standalone_collector()
        left_regularization = collector.compute_tracks_calib_residuals(left.collect_data()).pose_regularization
        right_regularization = collector.compute_tracks_calib_residuals(right.collect_data()).pose_regularization
        shared_grad_output = torch.linspace(0.25, 0.75, left.xy.shape[0], device=device)
        expected_grad_output = shared_grad_output.clone()

        torch.autograd.backward(
            (left_regularization, right_regularization),
            (shared_grad_output, shared_grad_output),
        )

        for calib in (left, right):
            expected_xy = 2.0 * (calib.xy - calib.initial_pose_xy) * expected_grad_output[:, None]
            expected_height = 2.0 * (calib.height - calib.initial_pose_height) * expected_grad_output
            expected_xy = torch.where(calib.pose_gradient_mask[:, None], expected_xy, torch.zeros_like(expected_xy))
            expected_height = torch.where(calib.pose_gradient_mask, expected_height, torch.zeros_like(expected_height))
            torch.testing.assert_close(calib.xy.grad, expected_xy)
            torch.testing.assert_close(calib.height.grad, expected_height)
        torch.testing.assert_close(shared_grad_output, expected_grad_output)

    def test_unicycle_tracks_calib_rejects_invalid_tensor_contract(self) -> None:
        cases = (
            (
                "xy_shape",
                "xy",
                lambda data: data.xy[:, :1],
                None,
            ),
            (
                "tracks_idx_dtype",
                "tracks_idx",
                lambda data: data.tracks_idx.to(torch.int32),
                None,
            ),
            (
                "triplet_rank",
                "motion_triplet_start_indices",
                lambda data: data.motion_triplet_start_indices.view(1, -1),
                None,
            ),
            (
                "mixed_device",
                "time_delta_s",
                lambda data: data.time_delta_s.cpu(),
                "must share one CUDA device",
            ),
            (
                "track_index_range",
                "tracks_idx",
                lambda data: data.tracks_idx.clone().index_fill_(
                    0,
                    torch.tensor([0], device=data.tracks_idx.device),
                    data.ground_t.shape[0],
                ),
                "out-of-range track index",
            ),
            (
                "pair_topology",
                "pose_pair_start_indices",
                lambda data: data.pose_pair_start_indices.roll(1),
                "does not match the packed track topology",
            ),
            (
                "triplet_topology",
                "motion_triplet_start_indices",
                lambda data: data.motion_triplet_start_indices.flip(0),
                "does not match the packed pair topology",
            ),
        )

        collector = self._get_standalone_collector()

        calib, tracks = self._make_unicycle_calib(device)
        tracks_poses = tracks.tracks_poses.vec().detach().requires_grad_()
        with self.assertRaises(AssertionError):
            collector.calibrate_tracks_poses(tracks_poses, calib.collect_data())

        for case_name, field_name, make_bad_value, error_pattern in cases:
            with self.subTest(case=case_name):
                calib, tracks = self._make_unicycle_calib(device)
                data = calib.collect_data()
                data = replace(data, **{field_name: make_bad_value(data)})
                with self.assertRaisesRegex(AssertionError, error_pattern or ""):
                    collector.calibrate_tracks_poses(tracks.tracks_poses.vec(), data)

    def test_input_embeddings(self):
        """Test input embedding computation with IndividualRemapTimeInputEmbedding."""
        n_samples = 1000
        n_instances = 10
        instance_dim = 1  # Must be 1 for Slang [ForceUnroll] in differentiable code

        # Create input data
        xyzs = torch.randn(n_samples, 3, device=device, dtype=torch.float32)
        xyzs.requires_grad = True

        instance_idx = torch.randint(0, n_instances, (n_samples,), device=device, dtype=torch.int32)

        instance_embedding = nn.Embedding(n_instances, instance_dim, device=device, dtype=torch.float32)

        # Time embedding config
        timestamps_ranges = torch.tensor(
            [[1000 * i, 1000 * (i + 10)] for i in range(n_instances)], dtype=torch.int64, device=device
        )
        time_embedding_config = IndividualRemapTimeInputEmbeddingConfig(
            timestamps_us_ranges=timestamps_ranges,
            remap_min=0.0,
            remap_max=1.0,
        )

        # Use single timestamp for all samples
        timestamp_us = 5000

        # Get scene contractor for applying contraction
        scene_contractor_config = self._get_test_scene_contractor(n_instances=n_instances)

        # Compute ground truth in Python
        ground_truth_embeddings = []

        # xyz
        contracted_xyzs = self._apply_scene_contraction(xyzs, instance_idx, scene_contractor_config)
        ground_truth_embeddings.append(contracted_xyzs)

        # instance embedding
        instance_emb = instance_embedding(instance_idx)
        ground_truth_embeddings.append(instance_emb)

        # time embedding (single timestamp for all samples)
        ranges = timestamps_ranges[instance_idx]
        ratio = (float(timestamp_us) - ranges[:, 0].float()) / (ranges[:, 1].float() - ranges[:, 0].float())
        time_emb = torch.clamp(ratio, 0, 1).unsqueeze(-1)  # remap_min=0, remap_max=1
        ground_truth_embeddings.append(time_emb)

        ground_truth = torch.cat(ground_truth_embeddings, dim=-1)

        # Compute using Slang collector
        collector = self._get_standalone_collector()

        input_embedding_data = InputEmbeddingData(
            xyzs=xyzs,
            instance_idx=instance_idx,
            timestamps_us=timestamp_us,
            instance_embedding_weights=WeightedInstanceInputEmbeddingData(
                instance_embedding_weights=instance_embedding.weight
            ),
            time_embedding_config=time_embedding_config,
            scene_contractor=scene_contractor_config,
        )

        output = collector.prepare_input_embeddings(input_embedding_data)

        self.assertEqual(output.shape, (n_samples, 3 + instance_dim + 1))
        self.assertTrue(torch.allclose(output, ground_truth, atol=1e-5))

        # Test backward pass
        xyzs.grad = None
        instance_embedding.weight.grad = None

        ground_truth_grad = torch.randn_like(output)
        output.backward(ground_truth_grad)
        output_grad_xyz = xyzs.grad
        output_grad_emb = instance_embedding.weight.grad

        xyzs.grad = None
        instance_embedding.weight.grad = None

        ground_truth.backward(ground_truth_grad)
        ground_truth_grad_xyz = xyzs.grad
        ground_truth_grad_emb = instance_embedding.weight.grad

        self.assertTrue(torch.allclose(output_grad_xyz, ground_truth_grad_xyz, atol=1e-5))
        self.assertTrue(torch.allclose(output_grad_emb, ground_truth_grad_emb, atol=1e-5))

    def test_input_embeddings_with_timestamps_delta(self):
        """Test input embedding computation with IndividualRemapTimeInputEmbedding and timestamps_delta."""
        n_samples = 1000
        n_instances = 10
        instance_dim = 1  # Must be 1 for Slang [ForceUnroll] in differentiable code

        # Create input data
        xyzs = torch.randn(n_samples, 3, device=device, dtype=torch.float32)
        xyzs.requires_grad = True

        instance_idx = torch.randint(0, n_instances, (n_samples,), device=device, dtype=torch.int32)

        instance_embedding = nn.Embedding(n_instances, instance_dim, device=device, dtype=torch.float32)

        # Time embedding config
        timestamps_ranges = torch.tensor(
            [[1000 * i, 1000 * (i + 10)] for i in range(n_instances)], dtype=torch.int64, device=device
        )
        time_embedding_config = IndividualRemapTimeInputEmbeddingConfig(
            timestamps_us_ranges=timestamps_ranges,
            remap_min=0.0,
            remap_max=1.0,
        )

        # Use single timestamp for all samples
        timestamp_us = 5000

        # Per-sample deltas
        timestamps_delta = torch.randint(-500, 501, (n_samples,), device=device, dtype=torch.int64)

        # Get scene contractor for applying contraction
        scene_contractor_config = self._get_test_scene_contractor(n_instances=n_instances)

        # Compute ground truth in Python for all 3 variants: [t, t-delta, t+delta]
        ground_truth_variants = []

        time_offsets = [
            torch.zeros_like(timestamps_delta),
            -timestamps_delta,
            timestamps_delta,
        ]  # [t, t-delta, t+delta]

        for time_offset in time_offsets:
            contracted_xyzs = self._apply_scene_contraction(xyzs, instance_idx, scene_contractor_config)

            # instance embedding (same for all variants)
            instance_emb = instance_embedding(instance_idx)

            # time embedding with offset
            adjusted_timestamp = float(timestamp_us) + time_offset.float()
            ranges = timestamps_ranges[instance_idx]
            ratio = (adjusted_timestamp - ranges[:, 0].float()) / (ranges[:, 1].float() - ranges[:, 0].float())
            time_emb = torch.clamp(ratio, 0, 1).unsqueeze(-1)  # remap_min=0, remap_max=1

            # Concatenate this variant
            variant = torch.cat([contracted_xyzs, instance_emb, time_emb], dim=-1)
            ground_truth_variants.append(variant)

        # Stack all variants along batch dimension (dim=0) to get [3*N, base_dim]
        ground_truth = torch.cat(ground_truth_variants, dim=0)

        # Compute using Slang collector
        collector = self._get_standalone_collector()

        input_embedding_data = InputEmbeddingData(
            xyzs=xyzs,
            instance_idx=instance_idx,
            timestamps_us=timestamp_us,
            timestamps_delta=timestamps_delta,
            instance_embedding_weights=WeightedInstanceInputEmbeddingData(
                instance_embedding_weights=instance_embedding.weight
            ),
            time_embedding_config=time_embedding_config,
            scene_contractor=scene_contractor_config,
        )

        output = collector.prepare_input_embeddings(input_embedding_data)

        base_dim = 3 + instance_dim + 1
        self.assertEqual(output.shape, (n_samples * 3, base_dim))
        self.assertTrue(torch.allclose(output, ground_truth, atol=1e-5))

        # Test backward pass
        xyzs.grad = None
        instance_embedding.weight.grad = None

        ground_truth_grad = torch.randn_like(output)
        output.backward(ground_truth_grad)
        output_grad_xyz = xyzs.grad.clone() if xyzs.grad is not None else None
        output_grad_emb = instance_embedding.weight.grad.clone() if instance_embedding.weight.grad is not None else None

        xyzs.grad = None
        instance_embedding.weight.grad = None

        ground_truth.backward(ground_truth_grad)
        ground_truth_grad_xyz = xyzs.grad
        ground_truth_grad_emb = instance_embedding.weight.grad

        self.assertTrue(torch.allclose(output_grad_xyz, ground_truth_grad_xyz, atol=1e-5))
        self.assertTrue(torch.allclose(output_grad_emb, ground_truth_grad_emb, atol=1e-5))

    def test_input_embeddings_with_per_track_timestamps(self):
        """Compact track timestamps map through repeated, non-monotonic instance IDs."""
        n_samples = 8192
        n_instances = 10
        instance_dim = 1

        xyzs = torch.randn(n_samples, 3, device=device, dtype=torch.float32, requires_grad=True)
        instance_idx = ((torch.arange(n_samples, device=device) * 7 + 3) % n_instances).to(torch.int32)
        instance_embedding = nn.Embedding(n_instances, instance_dim, device=device, dtype=torch.float32)
        timestamps_ranges = torch.tensor(
            [[1000 * i, 1000 * (i + 10)] for i in range(n_instances)], dtype=torch.int64, device=device
        )
        # Every instance maps to a distinct, unclamped time embedding (.45 .. .54).
        timestamps_us = 4500 + 1100 * torch.arange(n_instances, device=device, dtype=torch.int64)
        scene_contractor_config = self._get_test_scene_contractor(n_instances=n_instances)

        contracted_xyzs = self._apply_scene_contraction(xyzs, instance_idx, scene_contractor_config)
        instance_emb = instance_embedding(instance_idx)
        ranges = timestamps_ranges[instance_idx]
        sample_timestamps_us = timestamps_us[instance_idx]
        ratio = (sample_timestamps_us.float() - ranges[:, 0].float()) / (ranges[:, 1].float() - ranges[:, 0].float())
        time_emb = torch.clamp(ratio, 0, 1).unsqueeze(-1)
        ground_truth = torch.cat([contracted_xyzs, instance_emb, time_emb], dim=-1)

        collector = self._get_standalone_collector()
        output = collector.prepare_input_embeddings(
            InputEmbeddingData(
                xyzs=xyzs,
                instance_idx=instance_idx,
                timestamps_us=timestamps_us,
                instance_embedding_weights=WeightedInstanceInputEmbeddingData(
                    instance_embedding_weights=instance_embedding.weight
                ),
                time_embedding_config=IndividualRemapTimeInputEmbeddingConfig(
                    timestamps_us_ranges=timestamps_ranges,
                    remap_min=0.0,
                    remap_max=1.0,
                ),
                scene_contractor=scene_contractor_config,
            )
        )

        self.assertEqual(output.shape, (n_samples, 3 + instance_dim + 1))
        self.assertTrue(torch.allclose(output, ground_truth, atol=1e-5))

        # Unit gradients make the shared-instance atomic accumulation exact:
        # any dropped or duplicated contribution changes the integer count.
        ground_truth_grad = torch.ones_like(output)
        output.backward(ground_truth_grad)
        output_grad_xyz = xyzs.grad.clone()
        output_grad_emb = instance_embedding.weight.grad.clone()
        xyzs.grad = None
        instance_embedding.weight.grad = None
        ground_truth.backward(ground_truth_grad)

        self.assertTrue(torch.allclose(output_grad_xyz, xyzs.grad, atol=1e-5))
        torch.testing.assert_close(output_grad_emb, instance_embedding.weight.grad, rtol=1e-5, atol=1e-5)

    def test_input_embeddings_with_per_track_timestamps_and_delta(self):
        """Smoothness samples stay centered on each Gaussian's owning track time."""
        n_samples = 8192
        n_instances = 10
        instance_dim = 1

        xyzs = torch.randn(n_samples, 3, device=device, dtype=torch.float32, requires_grad=True)
        instance_idx = ((torch.arange(n_samples, device=device) * 7 + 3) % n_instances).to(torch.int32)
        instance_embedding = nn.Embedding(n_instances, instance_dim, device=device, dtype=torch.float32)
        timestamps_ranges = torch.tensor(
            [[1000 * i, 1000 * (i + 10)] for i in range(n_instances)], dtype=torch.int64, device=device
        )
        # Every instance maps to a distinct, unclamped time embedding (.45 .. .54).
        timestamps_us = 4500 + 1100 * torch.arange(n_instances, device=device, dtype=torch.int64)
        timestamps_delta = torch.randint(1, 501, (n_samples,), device=device, dtype=torch.int64)
        scene_contractor_config = self._get_test_scene_contractor(n_instances=n_instances)

        contracted_xyzs = self._apply_scene_contraction(xyzs, instance_idx, scene_contractor_config)
        instance_emb = instance_embedding(instance_idx)
        ranges = timestamps_ranges[instance_idx]
        sample_timestamps_us = timestamps_us[instance_idx]
        ground_truth_variants = []
        for adjusted_timestamps_us in (
            sample_timestamps_us,
            sample_timestamps_us - timestamps_delta,
            sample_timestamps_us + timestamps_delta,
        ):
            ratio = (adjusted_timestamps_us.float() - ranges[:, 0].float()) / (
                ranges[:, 1].float() - ranges[:, 0].float()
            )
            time_emb = torch.clamp(ratio, 0, 1).unsqueeze(-1)
            ground_truth_variants.append(torch.cat([contracted_xyzs, instance_emb, time_emb], dim=-1))
        ground_truth = torch.cat(ground_truth_variants, dim=0)

        collector = self._get_standalone_collector()
        output = collector.prepare_input_embeddings(
            InputEmbeddingData(
                xyzs=xyzs,
                instance_idx=instance_idx,
                timestamps_us=timestamps_us,
                timestamps_delta=timestamps_delta,
                instance_embedding_weights=WeightedInstanceInputEmbeddingData(
                    instance_embedding_weights=instance_embedding.weight
                ),
                time_embedding_config=IndividualRemapTimeInputEmbeddingConfig(
                    timestamps_us_ranges=timestamps_ranges,
                    remap_min=0.0,
                    remap_max=1.0,
                ),
                scene_contractor=scene_contractor_config,
            )
        )

        self.assertEqual(output.shape, (n_samples * 3, 3 + instance_dim + 1))
        self.assertTrue(torch.allclose(output, ground_truth, atol=1e-5))

        # Unit gradients make the shared-instance atomic accumulation exact:
        # any dropped or duplicated contribution changes the integer count.
        ground_truth_grad = torch.ones_like(output)
        output.backward(ground_truth_grad)
        output_grad_xyz = xyzs.grad.clone()
        output_grad_emb = instance_embedding.weight.grad.clone()
        xyzs.grad = None
        instance_embedding.weight.grad = None
        ground_truth.backward(ground_truth_grad)

        self.assertTrue(torch.allclose(output_grad_xyz, xyzs.grad, atol=1e-5))
        torch.testing.assert_close(output_grad_emb, instance_embedding.weight.grad, rtol=1e-5, atol=1e-5)

    def test_tracks_interpolation(self):
        def get_translation(t):
            return torch.tensor([1.0, 2.0, 3.0], device=device) * t.unsqueeze(1)

        def get_rotation(t):
            half_angle = t.unsqueeze(1) * (10 * 0.5 * torch.pi / 180.0)
            c = torch.cos(half_angle)
            s = torch.sin(half_angle)
            return torch.cat([s, torch.zeros_like(s), torch.zeros_like(s), c], dim=1)

        def get_tracks_poses(factors):
            track0 = torch.cat([torch.zeros_like(get_translation(factors[0])), get_rotation(factors[0])], dim=1)
            track1 = torch.cat([get_translation(factors[1]), get_rotation(factors[1])], dim=1)
            tracks_poses = torch.cat([track0, track1], dim=0)
            return tracks_poses

        factors = torch.arange(4, device=device).expand(2, 4)
        tracks_poses = get_tracks_poses(factors)
        tracks_timestamps = torch.tensor(
            [1000, 2000, 3000, 4000, 3000, 4000, 5000, 6000], dtype=torch.int64, device=device
        )
        tracks_packinfo = torch.tensor([[0, 4], [4, 4]], dtype=torch.int32, device=device)

        InterpolationTest = namedtuple("InterpolationTest", ["timestamp", "expected_factors", "expected_inside"])
        tests = [
            InterpolationTest(timestamp=0, expected_factors=[0.0, 0.0], expected_inside=[False, False]),
            InterpolationTest(timestamp=1000, expected_factors=[0.0, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=1400, expected_factors=[0.4, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=1500, expected_factors=[0.5, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=1600, expected_factors=[0.6, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=2000, expected_factors=[1.0, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=2400, expected_factors=[1.4, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=2500, expected_factors=[1.5, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=2600, expected_factors=[1.6, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=3000, expected_factors=[2.0, 0.0], expected_inside=[True, True]),
            InterpolationTest(timestamp=3400, expected_factors=[2.4, 0.4], expected_inside=[True, True]),
            InterpolationTest(timestamp=3500, expected_factors=[2.5, 0.5], expected_inside=[True, True]),
            InterpolationTest(timestamp=3600, expected_factors=[2.6, 0.6], expected_inside=[True, True]),
            InterpolationTest(timestamp=4000, expected_factors=[3.0, 1.0], expected_inside=[True, True]),
            InterpolationTest(timestamp=4400, expected_factors=[3.0, 1.4], expected_inside=[False, True]),
            InterpolationTest(timestamp=4500, expected_factors=[3.0, 1.5], expected_inside=[False, True]),
            InterpolationTest(timestamp=4600, expected_factors=[3.0, 1.6], expected_inside=[False, True]),
            InterpolationTest(timestamp=5000, expected_factors=[3.0, 2.0], expected_inside=[False, True]),
            InterpolationTest(timestamp=5400, expected_factors=[3.0, 2.4], expected_inside=[False, True]),
            InterpolationTest(timestamp=5500, expected_factors=[3.0, 2.5], expected_inside=[False, True]),
            InterpolationTest(timestamp=5600, expected_factors=[3.0, 2.6], expected_inside=[False, True]),
            InterpolationTest(timestamp=6000, expected_factors=[3.0, 3.0], expected_inside=[False, True]),
            InterpolationTest(timestamp=6400, expected_factors=[3.0, 3.0], expected_inside=[False, False]),
        ]

        collector = self._get_standalone_collector()

        for test in tests:
            for nearest_neighbor in [False, True]:
                timestamps_data = TracksTimestampsGlobalData(timestamp=test.timestamp)
                tracks_interpolation_data = TracksInterpolationData(
                    tracks_poses=tracks_poses,
                    tracks_timestamps=tracks_timestamps,
                    tracks_packinfo=tracks_packinfo,
                    timestamps_data=timestamps_data,
                    nearest_neighbor=nearest_neighbor,
                    max_positive_timestamp_gap_us=torch.zeros(2, dtype=torch.int64, device=device),
                )
                results_interp, results_inside_interp, results_timestamps = collector.interpolate_tracks_poses(
                    tracks_interpolation_data
                )
                expected_timestamps = [test.timestamp, test.timestamp]
                for track_idx, expected_inside in enumerate(test.expected_inside):
                    track_start = int(tracks_packinfo[track_idx, 0])
                    track_length = int(tracks_packinfo[track_idx, 1])
                    first_track_timestamp = int(tracks_timestamps[track_start])
                    last_track_timestamp = int(tracks_timestamps[track_start + track_length - 1])
                    if expected_inside and test.timestamp < first_track_timestamp:
                        expected_timestamps[track_idx] = first_track_timestamp
                    elif expected_inside and test.timestamp > last_track_timestamp:
                        expected_timestamps[track_idx] = last_track_timestamp
                self.assertEqual(results_timestamps.tolist(), expected_timestamps)

                expected_inside = getattr(test, "expected_inside")
                self.assertEqual(expected_inside, results_inside_interp.tolist())

                expected_factors = test.expected_factors
                expected_factors = torch.tensor(expected_factors, device=device).unsqueeze(1)
                if nearest_neighbor:
                    # In case of a tie we round to the left / smallest value.
                    expected_factors = torch.ceil(expected_factors - 0.5)
                expected_poses = get_tracks_poses(expected_factors)
                self.assertTrue(torch.allclose(results_interp, expected_poses))

        # Render-time timestamp estimation can query slightly beyond the first/last
        # annotated pose because camera exposure spans an interval.  When the
        # tolerance is enabled, keep the endpoint pose instead of marking the
        # whole rigid track invisible.
        tolerance_tests = [
            InterpolationTest(timestamp=4400, expected_factors=[3.0, 1.4], expected_inside=[True, True]),
            InterpolationTest(timestamp=800, expected_factors=[0.0, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=6400, expected_factors=[3.0, 3.0], expected_inside=[False, True]),
        ]
        cadences_us = [[300, 0], [100, 100], [0, 300]]
        frame_exposure_us = 200
        expected_resolved_timestamps = ([4000, 4400], [1000, 800], [6400, 6000])
        for test, track_cadences_us, expected_timestamps in zip(
            tolerance_tests, cadences_us, expected_resolved_timestamps
        ):
            for nearest_neighbor in [False, True]:
                timestamps_data = TracksTimestampsGlobalData(timestamp=test.timestamp)
                tracks_interpolation_data = TracksInterpolationData(
                    tracks_poses=tracks_poses,
                    tracks_timestamps=tracks_timestamps,
                    tracks_packinfo=tracks_packinfo,
                    timestamps_data=timestamps_data,
                    nearest_neighbor=nearest_neighbor,
                    max_positive_timestamp_gap_us=torch.tensor(track_cadences_us, dtype=torch.int64, device=device),
                    frame_exposure_us=frame_exposure_us,
                )
                results_interp, results_inside_interp, results_timestamps = collector.interpolate_tracks_poses(
                    tracks_interpolation_data
                )
                self.assertEqual(results_timestamps.tolist(), expected_timestamps)
                self.assertEqual(test.expected_inside, results_inside_interp.tolist())

                expected_factors = torch.tensor(test.expected_factors, device=device).unsqueeze(1)
                if nearest_neighbor:
                    # In case of a tie we round to the left / smallest value.
                    expected_factors = torch.ceil(expected_factors - 0.5)
                expected_poses = get_tracks_poses(expected_factors)
                self.assertTrue(torch.allclose(results_interp, expected_poses))

        # Backward pass.
        # We are just going to test that gradients pass through in nearest neighbor mode.
        tracks_poses.requires_grad = True
        tracks_poses.grad = None
        # Rendering metadata can arrive on the CPU even though interpolation runs
        # on the GPU. The collector owns aligning it with the track tensors.
        timestamps = torch.tensor([2900, 4100], dtype=torch.int64)
        timestamps_data = TracksTimestampsPerTrackData(timestamps=timestamps)
        matching_indices = [2, 5]
        tracks_interpolation_data = TracksInterpolationData(
            tracks_poses=tracks_poses,
            tracks_timestamps=tracks_timestamps,
            tracks_packinfo=tracks_packinfo,
            timestamps_data=timestamps_data,
            nearest_neighbor=True,
            max_positive_timestamp_gap_us=torch.zeros(2, dtype=torch.int64, device=device),
        )
        results_interp, results_inside_interp, results_timestamps = collector.interpolate_tracks_poses(
            tracks_interpolation_data
        )
        self.assertTrue(torch.equal(results_timestamps, timestamps.to(device)))
        results_interp_grad = torch.randn_like(results_interp)
        results_interp.backward(results_interp_grad)
        tracks_poses_grad = tracks_poses.grad
        for i in range(len(matching_indices)):
            tracks_packinfo_start = tracks_packinfo[i, 0].item()
            tracks_packinfo_length = tracks_packinfo[i, 1].item()
            for j in range(tracks_packinfo_length):
                pose_index = tracks_packinfo_start + j
                ground_truth = results_interp_grad[i]
                if pose_index != matching_indices[i]:
                    ground_truth = torch.zeros_like(ground_truth)
                self.assertTrue(torch.equal(tracks_poses_grad[pose_index], ground_truth))

    def test_tracks_timestamp_estimation_dispatch(self):
        collector = self._get_standalone_collector()
        tracks_poses = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            requires_grad=True,
        )
        tracks_timestamps = torch.tensor([1000, 2000], dtype=torch.int64, device=device)
        tracks_packinfo = torch.tensor([[0, 2]], dtype=torch.int32, device=device)
        timestamps_data = TracksTimestampsEstimationData(
            rays=torch.tensor([[0.0, 0.0, -2.0, 0.0, 0.0, 1.0]], device=device),
            rays_timestamps=torch.tensor([1500], dtype=torch.int64, device=device),
            cuboids_dimensions=torch.tensor([[2.0, 2.0, 2.0]], device=device),
            # A miss would fall back to timestamp 2000 and return the second pose.
            # The nearest-neighbor and interpolation expectations below therefore
            # prove that the ray hit at 1500 and reached the corresponding kernel.
            default_timestamp=2000,
            frame_start_timestamp=1000,
            frame_end_timestamp=2000,
        )

        for nearest_neighbor in (False, True):
            with self.subTest(nearest_neighbor=nearest_neighbor):
                tracks_poses.grad = None
                interpolated, inside, resolved_timestamps = collector.interpolate_tracks_poses(
                    TracksInterpolationData(
                        tracks_poses=tracks_poses,
                        tracks_timestamps=tracks_timestamps,
                        tracks_packinfo=tracks_packinfo,
                        timestamps_data=timestamps_data,
                        nearest_neighbor=nearest_neighbor,
                        max_positive_timestamp_gap_us=torch.zeros(1, dtype=torch.int64, device=device),
                        frame_exposure_us=0,
                    )
                )

                expected_pose = tracks_poses[0].detach().clone()
                expected_grads = torch.zeros_like(tracks_poses)
                if nearest_neighbor:
                    expected_grads[0, 0] = 1.0
                else:
                    expected_pose[0] = 0.25
                    expected_grads[:, 0] = 0.5

                self.assertEqual(inside.tolist(), [True])
                self.assertEqual(resolved_timestamps.tolist(), [1500])
                torch.testing.assert_close(interpolated[0], expected_pose, rtol=0, atol=0)
                interpolated[0, 0].backward()
                torch.testing.assert_close(tracks_poses.grad, expected_grads, rtol=0, atol=0)

    def test_shutter_timestamp_cache_preserves_poses_and_gradients(self):
        """Exercise real cached dispatch, mixed track lengths, and refreshed scratch."""
        from nurec.models.gaussians.collect.collector_slang import TracksInterpolationFunction

        collector = self._get_standalone_collector()
        batch, height, width = 2, 3, 11
        poses = torch.zeros((9, 7), dtype=torch.float32, device=device)
        poses[:, 6] = 1
        poses[:, 0] = torch.linspace(0, 0.4, 9, device=device)
        poses.requires_grad_(True)
        rays = torch.zeros((batch, height, width, 6), device=device)
        rays[..., 0], rays[..., 3] = -3, 1
        pack = torch.tensor([[0, 3], [3, 2], [5, 4]], dtype=torch.int32, device=device)
        times = torch.tensor([1000, 2000, 3000, 1000, 3000, 1000, 1500, 2500, 3000], dtype=torch.int64, device=device)
        for case in ("rows", "rows-reverse", "columns", "columns-reverse", "global", "rows-mislabeled", None):
            layout = case.split("-")[0] if case is not None else None
            for nearest in (False, True):
                with self.subTest(layout=case, nearest=nearest):
                    stamps = torch.full((batch, height, width), 1600, dtype=torch.int64, device=device)
                    if case == "rows-mislabeled" or layout is None:
                        stamps += torch.arange(stamps.numel(), device=device).reshape(stamps.shape) * 7
                    elif layout == "rows":
                        stamps += torch.arange(height, device=device)[None, :, None] * 100
                    elif layout == "columns":
                        stamps += torch.arange(width, device=device)[None, None, :] * 20
                    if case == "rows-reverse":
                        stamps = stamps.flip(1)
                    elif case == "columns-reverse":
                        stamps = stamps.flip(2)
                    stamps[1] += 150
                    case_rays = rays
                    if case == "rows-mislabeled":
                        # Cross a moving cuboid boundary so an incorrectly reused pose changes visibility.
                        case_rays = torch.zeros_like(rays)
                        case_rays[..., 0] = poses[0, 0].detach() + 1.04
                        case_rays[..., 1], case_rays[..., 4] = -3, 1
                    estimation = TracksTimestampsEstimationData(
                        rays=case_rays.reshape(-1, 6),
                        rays_timestamps=stamps.reshape(-1),
                        cuboids_dimensions=torch.full((3, 3), 2.0, device=device),
                        default_timestamp=1500,
                        frame_start_timestamp=1000,
                        frame_end_timestamp=3000,
                        rays_timestamp_layout=layout,
                        rays_grid_shape=(batch, height, width),
                    )
                    self.assertEqual(
                        TracksInterpolationFunction._timestamp_cache_layout(estimation, 3, 9) is not None,
                        layout is not None,
                    )
                    data = TracksInterpolationData(
                        tracks_poses=poses,
                        tracks_timestamps=times,
                        tracks_packinfo=pack,
                        timestamps_data=estimation,
                        nearest_neighbor=nearest,
                        max_positive_timestamp_gap_us=torch.zeros(3, dtype=torch.int64, device=device),
                        frame_exposure_us=0,
                    )
                    for revision in range(2):
                        with torch.no_grad():
                            poses[:, 0].add_(0.01)
                        values, resolved_timestamp_values, gradients = [], [], []
                        for use_cache in (False, True):
                            poses.grad = None
                            variant = estimation if use_cache else replace(estimation, rays_timestamp_layout=None)
                            result, inside, resolved_timestamps = collector.interpolate_tracks_poses(
                                replace(data, timestamps_data=variant)
                            )
                            self.assertTrue(inside.all())
                            result[:, 0].sum().backward()
                            values.append(result.detach().clone())
                            resolved_timestamp_values.append(resolved_timestamps.clone())
                            gradients.append(poses.grad.clone())
                        torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
                        torch.testing.assert_close(
                            resolved_timestamp_values[0], resolved_timestamp_values[1], rtol=0, atol=0
                        )
                        torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
                        self.assertGreater(gradients[1].abs().sum().item(), 0)

    def test_timestamp_cache_declines_unknown_shape_and_large_scratch(self):
        """Untrusted metadata and expensive preparation retain the original path."""
        from nurec.models.gaussians.collect.collector_slang import TracksInterpolationFunction

        data = TracksTimestampsEstimationData(
            rays=torch.empty((66, 6)),
            rays_timestamps=torch.empty(66, dtype=torch.int64),
            cuboids_dimensions=torch.empty((3, 3)),
            default_timestamp=0,
            frame_start_timestamp=0,
            frame_end_timestamp=1,
            rays_timestamp_layout="rows",
            rays_grid_shape=(2, 3, 11),
        )
        choose = TracksInterpolationFunction._timestamp_cache_layout
        self.assertIsNotNone(choose(data, 3, 9))
        for variant in (
            replace(data, rays_timestamp_layout=None),
            replace(data, rays_grid_shape=None),
            replace(data, rays_grid_shape=(2, 3, 12)),
            replace(data, rays_grid_shape=(0, 3, 11)),
        ):
            self.assertIsNone(choose(variant, 3, 9))
        self.assertIsNone(choose(data, 3, 6))
        self.assertIsNone(choose(data, 1000000, 3000000))

    def test_tracks_interpolation_dispatch_matrix(self):
        """Cadence endpoint semantics match across representative timestamp and pose dispatches."""
        collector = self._get_standalone_collector()
        first_timestamp_us = 1000
        last_timestamp_us = 2000
        cadence_value_us = 1000
        tracks_timestamps = torch.tensor([first_timestamp_us, last_timestamp_us], dtype=torch.int64, device=device)
        tracks_packinfo = torch.tensor([[0, 2]], dtype=torch.int32, device=device)
        base_tracks_poses = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
        )
        cadence_us = torch.tensor([cadence_value_us], dtype=torch.int64, device=device)
        frame_exposure_us = 200
        endpoint_tolerance_us = cadence_value_us + frame_exposure_us

        cases = (
            ("midpoint", 1500),
            ("endpoint_tolerance_before", first_timestamp_us - endpoint_tolerance_us),
            ("endpoint_tolerance_before_minus_one", first_timestamp_us - endpoint_tolerance_us - 1),
            ("endpoint_tolerance_after", last_timestamp_us + endpoint_tolerance_us),
            ("endpoint_tolerance_after_plus_one", last_timestamp_us + endpoint_tolerance_us + 1),
        )

        def make_timestamps_data(mode: str, query_timestamp: int, case_name: str):
            if mode == "global":
                return TracksTimestampsGlobalData(timestamp=query_timestamp)
            if mode == "per_track":
                return TracksTimestampsPerTrackData(
                    timestamps=torch.tensor([query_timestamp], dtype=torch.int64, device=device)
                )
            if mode == "estimation":
                # The midpoint ray intersects the moving cuboid, exercising the estimated
                # timestamp. Endpoint cases deliberately miss and exercise the fetcher's
                # fallback timestamp before the same interpolation entrypoint runs.
                ray_x = 0.25 if case_name == "midpoint" else 100.0
                return TracksTimestampsEstimationData(
                    rays=torch.tensor([[ray_x, 0.0, -2.0, 0.0, 0.0, 1.0]], device=device),
                    rays_timestamps=torch.tensor([1500], dtype=torch.int64, device=device),
                    cuboids_dimensions=torch.tensor([[2.0, 2.0, 2.0]], device=device),
                    default_timestamp=query_timestamp,
                    frame_start_timestamp=1000,
                    frame_end_timestamp=2000,
                )
            raise AssertionError(f"Unsupported timestamp mode: {mode}")

        def reference_pose_inside_and_timestamp(query_timestamp: int, nearest_neighbor: bool):
            if query_timestamp <= first_timestamp_us:
                expected_pose = base_tracks_poses[0]
                endpoint_delta_us = first_timestamp_us - query_timestamp
                endpoint_timestamp_us = first_timestamp_us
            elif query_timestamp >= last_timestamp_us:
                expected_pose = base_tracks_poses[-1]
                endpoint_delta_us = query_timestamp - last_timestamp_us
                endpoint_timestamp_us = last_timestamp_us
            else:
                alpha = (query_timestamp - first_timestamp_us) / (last_timestamp_us - first_timestamp_us)
                if nearest_neighbor:
                    expected_pose = base_tracks_poses[0] if alpha <= 0.5 else base_tracks_poses[1]
                else:
                    expected_pose = torch.lerp(base_tracks_poses[0], base_tracks_poses[1], alpha)
                endpoint_delta_us = 0
                endpoint_timestamp_us = query_timestamp
            expected_inside = (
                first_timestamp_us <= query_timestamp <= last_timestamp_us or endpoint_delta_us <= endpoint_tolerance_us
            )
            expected_timestamp = (
                endpoint_timestamp_us
                if expected_inside and not first_timestamp_us <= query_timestamp <= last_timestamp_us
                else query_timestamp
            )
            return expected_pose, expected_inside, expected_timestamp

        dispatch_modes = (
            ("global", False),
            ("global", True),
            ("per_track", False),
            ("per_track", True),
            ("estimation", False),
            ("estimation", True),
        )
        for timestamp_mode, nearest_neighbor in dispatch_modes:
            for case_name, query_timestamp in cases:
                with self.subTest(
                    timestamp_mode=timestamp_mode,
                    nearest_neighbor=nearest_neighbor,
                    case=case_name,
                ):
                    interpolation_data = TracksInterpolationData(
                        tracks_poses=base_tracks_poses,
                        tracks_timestamps=tracks_timestamps,
                        tracks_packinfo=tracks_packinfo,
                        timestamps_data=make_timestamps_data(timestamp_mode, query_timestamp, case_name),
                        nearest_neighbor=nearest_neighbor,
                        max_positive_timestamp_gap_us=cadence_us,
                        frame_exposure_us=frame_exposure_us,
                    )
                    with torch.inference_mode():
                        interpolated, inside, resolved_timestamps = collector.interpolate_tracks_poses(
                            interpolation_data
                        )

                    expected_pose, expected_inside, expected_timestamp = reference_pose_inside_and_timestamp(
                        query_timestamp, nearest_neighbor
                    )
                    torch.testing.assert_close(interpolated[0], expected_pose, rtol=1e-6, atol=1e-6)
                    self.assertEqual(inside.tolist(), [expected_inside])
                    self.assertEqual(resolved_timestamps.tolist(), [expected_timestamp])

        # Exercise every generated backward wrapper with the same cadence ABI.
        for timestamp_mode, nearest_neighbor in dispatch_modes:
            with self.subTest(
                timestamp_mode=timestamp_mode,
                nearest_neighbor=nearest_neighbor,
                case="backward",
            ):
                differentiable_poses = base_tracks_poses.clone().requires_grad_(True)
                interpolated, _, _ = collector.interpolate_tracks_poses(
                    TracksInterpolationData(
                        tracks_poses=differentiable_poses,
                        tracks_timestamps=tracks_timestamps,
                        tracks_packinfo=tracks_packinfo,
                        timestamps_data=make_timestamps_data(timestamp_mode, 1500, "midpoint"),
                        nearest_neighbor=nearest_neighbor,
                        max_positive_timestamp_gap_us=cadence_us,
                        frame_exposure_us=frame_exposure_us,
                    )
                )
                output_gradient = torch.zeros_like(interpolated)
                output_gradient[0, 0] = 1.0
                interpolated.backward(output_gradient)

                reference_poses = base_tracks_poses.clone().requires_grad_(True)
                reference_pose = (
                    reference_poses[0] if nearest_neighbor else torch.lerp(reference_poses[0], reference_poses[1], 0.5)
                )
                reference_pose.unsqueeze(0).backward(output_gradient)
                self.assertIsNotNone(differentiable_poses.grad)
                self.assertIsNotNone(reference_poses.grad)
                assert differentiable_poses.grad is not None
                assert reference_poses.grad is not None
                torch.testing.assert_close(
                    differentiable_poses.grad,
                    reference_poses.grad,
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_tracks_timestamp_estimation_returns_absolute_average_and_fallback(self):
        """Resolved deformnet times retain the frame origin and default tracks with no hits."""
        default_timestamp = 10_100
        tracks_poses = torch.tensor(
            [
                [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0],
                [0.2, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0],
                [5.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0],
                [5.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=torch.float32,
        )
        tracks_timestamps = torch.tensor([9_000, 11_000, 9_000, 11_000], device=device, dtype=torch.int64)
        tracks_packinfo = torch.tensor([[0, 2], [2, 2]], device=device, dtype=torch.int32)
        cuboid_dimensions = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], device=device)
        rays = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.25, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=torch.float32,
        )
        rays_timestamps = torch.tensor([10_020, 10_060], device=device, dtype=torch.int64)

        interpolation_data = TracksInterpolationData(
            tracks_poses=tracks_poses,
            tracks_timestamps=tracks_timestamps,
            tracks_packinfo=tracks_packinfo,
            timestamps_data=TracksTimestampsEstimationData(
                rays=rays,
                rays_timestamps=rays_timestamps,
                cuboids_dimensions=cuboid_dimensions,
                default_timestamp=default_timestamp,
                frame_start_timestamp=10_000,
                frame_end_timestamp=10_200,
                frustum_cull=None,
            ),
            nearest_neighbor=False,
            max_positive_timestamp_gap_us=torch.tensor([2_000, 2_000], dtype=torch.int64, device=device),
            frame_exposure_us=200,
        )

        poses, inside, resolved_timestamps = self._get_standalone_collector().interpolate_tracks_poses(
            interpolation_data
        )

        self.assertEqual(inside.tolist(), [True, True])
        self.assertEqual(resolved_timestamps.tolist(), [10_040, default_timestamp])
        torch.testing.assert_close(poses[0, 0], torch.tensor(0.104, device=device), rtol=1e-6, atol=1e-6)

    def test_tracks_interpolation_returns_empty_timestamps_for_no_tracks(self):
        interpolation_data = TracksInterpolationData(
            tracks_poses=torch.empty((0, 7), device=device, dtype=torch.float32),
            tracks_timestamps=torch.empty((0,), device=device, dtype=torch.int64),
            tracks_packinfo=torch.empty((0, 2), device=device, dtype=torch.int32),
            timestamps_data=TracksTimestampsGlobalData(timestamp=10_100),
            nearest_neighbor=False,
            max_positive_timestamp_gap_us=torch.empty((0,), device=device, dtype=torch.int64),
            frame_exposure_us=200,
        )

        poses, inside, resolved_timestamps = self._get_standalone_collector().interpolate_tracks_poses(
            interpolation_data
        )

        self.assertEqual(poses.shape, (0, 7))
        self.assertEqual(inside.shape, (0,))
        self.assertEqual(resolved_timestamps.shape, (0,))
        self.assertEqual(resolved_timestamps.dtype, torch.int64)
        self.assertEqual(inside.device, interpolation_data.tracks_poses.device)
        self.assertEqual(resolved_timestamps.device, interpolation_data.tracks_poses.device)

    def test_zero_disabled_track_grads_matches_row_zero(self):
        from nurec.models.gaussians.collect.collector_slang import (
            launch_zero_gradients_for_disabled_tracks,
        )

        configuration = CollectorConfiguration(
            parameters=(
                "Collector_Copy<3>",
                "RotationsCollector_Normalize",
                "ScalesCollector_Exp",
                "DensitiesCollector_Sigmoid",
                "SphericalFeaturesCollector_Copy<3,45>",
            )
        )
        slang_module = get_slang_kernels([configuration])[0].slang_module
        torch.manual_seed(0)
        n = 97
        n_tracks = 7
        tracks_ids = torch.randint(0, n_tracks, (n,), device=device, dtype=torch.int32)
        keep_mask = torch.tensor([True, False, True, False, True, False, True], device=device)
        widths = (3, 4, 3, 1, 0, 20, 0, 3, 45)
        grads = []
        refs = []
        for width in widths:
            if width == 0:
                empty = torch.empty(0, device=device, dtype=torch.float32)
                grads.append(empty)
                refs.append(empty)
                continue
            tensor = torch.randn(n, width, device=device)
            grads.append(tensor)
            refs.append(tensor.clone())

        disabled = ~keep_mask[tracks_ids.long()]
        for ref, width in zip(refs, widths):
            if width == 0:
                continue
            ref[disabled] = 0

        launch_zero_gradients_for_disabled_tracks(slang_module, tracks_ids, keep_mask, tuple(grads))
        for got, expected, width in zip(grads, refs, widths):
            if width == 0:
                continue
            torch.testing.assert_close(got, expected, rtol=0, atol=0)

        with self.assertRaises(ValueError):
            launch_zero_gradients_for_disabled_tracks(slang_module, tracks_ids, keep_mask, tuple(grads[:-1]))

        empty_tracks_ids = tracks_ids[:0]
        empty_grads = tuple(grad[:0] for grad in grads)
        with self.assertRaises(ValueError):
            launch_zero_gradients_for_disabled_tracks(
                slang_module,
                empty_tracks_ids,
                keep_mask,
                empty_grads[:-1],
            )

        mismatched = list(grads)
        mismatched[0] = torch.randn(n + 1, widths[0], device=device)
        with self.assertRaises(ValueError):
            launch_zero_gradients_for_disabled_tracks(slang_module, tracks_ids, keep_mask, tuple(mismatched))

    def test_rigid_layer(self):
        nb_tracks = 2
        fourier_dim = 20
        timestamps_us_ranges = torch.tensor(
            [[0, 1_000_000]] * nb_tracks,
            dtype=torch.int64,
            device=device,
        )
        layers_config = LayersConfig(
            layers=[
                LayerConfigRigid(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=fourier_dim,
                    embed_config=IndividualRemapTimeInputEmbeddingConfig(
                        timestamps_us_ranges=timestamps_us_ranges,
                        remap_min=0.0,
                        remap_max=1.0,
                    ),
                ),
            ],
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
            sh_features_dtype="fp16",
        )

        collector = CreateGaussianParameterCollector(layers_config)

        tracks_poses = torch.cat(
            [
                torch.randn(nb_tracks, 3, device=device),
                torch.zeros(nb_tracks, 3, device=device),
                torch.ones(nb_tracks, 1, device=device),
            ],
            dim=1,
        )
        keep_mask = torch.tensor([True, False], device=device, dtype=torch.bool)
        tracks_ids = torch.tensor([0, 0, 1], device=device, dtype=torch.int32)

        nb_gaussians = 3
        positions = torch.randn(nb_gaussians, 3, device=device)
        rotations = torch.randn(nb_gaussians, 4, device=device)
        scales = torch.randn(nb_gaussians, 3, device=device)
        densities = torch.randn(nb_gaussians, 1, device=device)
        extra_signal = torch.randn(nb_gaussians, 0, device=device)
        camera_extra_signal = torch.randn(nb_gaussians, 20, device=device)
        lidar_extra_signal = torch.randn(nb_gaussians, 0, device=device)
        features_albedo = torch.randn(nb_gaussians, fourier_dim, 3, device=device)
        features_specular = torch.randn(nb_gaussians, 45, device=device)
        embed_data = IndividualRemapTimeInputEmbeddingData(instance_idx=tracks_ids)

        layers_data = LayersData(
            layers=[
                LayerDataRigid(
                    positions=positions,
                    rotations=rotations,
                    scales=scales,
                    densities=densities,
                    extra_signal=extra_signal,
                    camera_extra_signal=camera_extra_signal,
                    lidar_extra_signal=lidar_extra_signal,
                    features_albedo=features_albedo,
                    features_specular=features_specular,
                    embed_data=embed_data,
                    poses=tracks_poses,
                    keep_mask=keep_mask,
                    tracks_ids=tracks_ids,
                ),
            ],
            frame_timestamp_us=500_000,
        )

        for tensor in [
            positions,
            rotations,
            scales,
            densities,
            extra_signal,
            camera_extra_signal,
            lidar_extra_signal,
            features_albedo,
            features_specular,
            tracks_poses,
        ]:
            tensor.requires_grad = True
            tensor.grad = None

        # Forward pass.
        results = collector.collect(layers_data)
        ground_truth_positions = positions + tracks_poses[tracks_ids][:, :3]
        self.assertTrue(torch.allclose(results.positions, ground_truth_positions))
        enabled_densities = results.densities[keep_mask[tracks_ids]]
        disabled_densities = results.densities[~keep_mask[tracks_ids]]
        self.assertEqual(disabled_densities.shape, (1, 1))
        self.assertTrue(torch.all(disabled_densities == 0))
        self.assertTrue(torch.all(enabled_densities != 0))

        # Backward pass.
        buffers = [getattr(results, field.name) for field in fields(results)]
        gradients = [torch.randn_like(buffer) for buffer in buffers]
        torch.autograd.backward(buffers, gradients)

        for tensor in [extra_signal, lidar_extra_signal]:
            self.assertEqual(tensor.grad, None)

        for tensor in [
            positions,
            rotations,
            scales,
            densities,
            camera_extra_signal,
            features_albedo,
            features_specular,
        ]:
            self.assertTrue(torch.all(tensor.grad[keep_mask[tracks_ids]] != 0))
            self.assertTrue(torch.all(tensor.grad[~keep_mask[tracks_ids]] == 0))

        self.assertTrue(torch.all(tracks_poses.grad[keep_mask] != 0))
        self.assertTrue(torch.all(tracks_poses.grad[~keep_mask] == 0))

        self.assertTrue(torch.all(positions.grad[keep_mask[tracks_ids]] == gradients[0][keep_mask[tracks_ids]]))

    def _rigid_windows_fixture(self):
        """A single rigid layer whose 8 rows are grouped into 3 tracks: [0,3), [3,5), [5,8)."""
        nb_tracks = 3
        fourier_dim = 20
        timestamps_us_ranges = torch.tensor([[0, 1_000_000]] * nb_tracks, dtype=torch.int64, device=device)
        layers_config = LayersConfig(
            layers=[
                LayerConfigRigid(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=fourier_dim,
                    embed_config=IndividualRemapTimeInputEmbeddingConfig(
                        timestamps_us_ranges=timestamps_us_ranges,
                        remap_min=0.0,
                        remap_max=1.0,
                    ),
                ),
            ],
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
            sh_features_dtype="fp16",
        )
        collector = CreateGaussianParameterCollector(layers_config)

        tracks_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], device=device, dtype=torch.int32)
        nb_gaussians = tracks_ids.shape[0]
        tracks_poses = torch.cat(
            [
                torch.randn(nb_tracks, 3, device=device),
                torch.zeros(nb_tracks, 3, device=device),
                torch.ones(nb_tracks, 1, device=device),
            ],
            dim=1,
        )
        layers_data = LayersData(
            layers=[
                LayerDataRigid(
                    positions=torch.randn(nb_gaussians, 3, device=device),
                    rotations=torch.randn(nb_gaussians, 4, device=device),
                    scales=torch.randn(nb_gaussians, 3, device=device),
                    densities=torch.randn(nb_gaussians, 1, device=device),
                    extra_signal=torch.randn(nb_gaussians, 0, device=device),
                    camera_extra_signal=torch.randn(nb_gaussians, 20, device=device),
                    lidar_extra_signal=torch.randn(nb_gaussians, 0, device=device),
                    features_albedo=torch.randn(nb_gaussians, fourier_dim, 3, device=device),
                    features_specular=torch.randn(nb_gaussians, 45, device=device),
                    embed_data=IndividualRemapTimeInputEmbeddingData(instance_idx=tracks_ids),
                    poses=tracks_poses,
                    keep_mask=torch.tensor([True, True, False], device=device),
                    tracks_ids=tracks_ids,
                ),
            ],
            frame_timestamp_us=500_000,
        )
        return collector, layers_data

    def test_segmented_collect_writes_only_the_selected_windows(self):
        """A segmented collect matches the dense one inside the windows and leaves other rows alone."""
        collector, layers_data = self._rigid_windows_fixture()

        with torch.no_grad():
            dense = collector.collect(layers_data)

            # Collect tracks 0 and 2 only, into a destination pre-filled with NaN.
            windows = windowed_layer_segments(
                starts=torch.tensor([0, 5], dtype=torch.int32, device=device),
                window_size_prefix=torch.tensor([0, 3, 6], dtype=torch.int32, device=device),
                active_count=6,
            )
            destination = CollectorResult(
                *[torch.full_like(getattr(dense, field.name), float("nan")) for field in fields(dense)]
            )
            sparse = collector.collect(layers_data, out=destination, segments=[windows])
            self.assertIs(sparse.positions, destination.positions, "the destination buffers must be written in place")

            selected = torch.tensor([True] * 3 + [False] * 2 + [True] * 3, device=device)
            for field in fields(dense):
                dense_rows = getattr(dense, field.name)
                sparse_rows = getattr(sparse, field.name)
                if dense_rows.shape[1] == 0:
                    continue
                self.assertTrue(
                    torch.equal(dense_rows[selected], sparse_rows[selected]),
                    f"{field.name}: selected rows differ from the dense collect",
                )
                self.assertTrue(
                    torch.isnan(sparse_rows[~selected]).all(),
                    f"{field.name}: rows outside the windows were written",
                )

            # An empty window list touches nothing.
            untouched = CollectorResult(
                *[torch.full_like(getattr(dense, field.name), float("nan")) for field in fields(dense)]
            )
            empty = windowed_layer_segments(
                starts=torch.empty(0, dtype=torch.int32, device=device),
                window_size_prefix=torch.zeros(1, dtype=torch.int32, device=device),
                active_count=0,
            )
            collector.collect(layers_data, out=untouched, segments=[empty])
            self.assertTrue(torch.isnan(untouched.positions).all())

    def test_windowed_and_caller_owned_collects_require_no_grad(self):
        """Both preconditions are read in `collect`, since a Function's forward never sees grad mode."""
        collector, layers_data = self._rigid_windows_fixture()

        with torch.no_grad():
            dense = collector.collect(layers_data)
        windows = windowed_layer_segments(
            starts=torch.tensor([0, 5], dtype=torch.int32, device=device),
            window_size_prefix=torch.tensor([0, 3, 6], dtype=torch.int32, device=device),
            active_count=6,
        )
        destination = CollectorResult(*[torch.empty_like(getattr(dense, f.name)) for f in fields(dense)])

        with torch.enable_grad():
            with self.assertRaisesRegex(ValueError, "windowed collect requires no_grad"):
                collector.collect(layers_data, out=None, segments=[windows])
            with self.assertRaisesRegex(ValueError, "caller-owned collect destination requires no_grad"):
                collector.collect(layers_data, out=destination)
            # Whole-layer segments are the dense collect, so they stay allowed under grad.
            whole = whole_layer_segments(int(layers_data.layers[0].positions.shape[0]))
            collector.collect(layers_data, segments=[whole])

    def test_deformable_layer(self):
        layers_config = LayersConfig(
            layers=[
                LayerConfigDeformable(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
            ],
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
        )

        collector = CreateGaussianParameterCollector(layers_config)

        nb_tracks = 2
        tracks_poses = torch.cat(
            [
                torch.randn(nb_tracks, 3, device=device),
                torch.zeros(nb_tracks, 3, device=device),
                torch.ones(nb_tracks, 1, device=device),
            ],
            dim=1,
        )
        keep_mask = torch.tensor([True, False], device=device, dtype=torch.bool)
        tracks_ids = torch.tensor([0, 0, 1], device=device, dtype=torch.int32)

        nb_gaussians = 3
        positions = torch.randn(nb_gaussians, 3, device=device)
        rotations = torch.randn(nb_gaussians, 4, device=device)
        scales = torch.randn(nb_gaussians, 3, device=device)
        densities = torch.randn(nb_gaussians, 1, device=device)
        extra_signal = torch.randn(nb_gaussians, 0, device=device)
        camera_extra_signal = torch.randn(nb_gaussians, 20, device=device)
        lidar_extra_signal = torch.randn(nb_gaussians, 0, device=device)
        features_albedo = torch.randn(nb_gaussians, 3, device=device)
        features_specular = torch.randn(nb_gaussians, 45, device=device)
        embed_data = None
        deform_positions = torch.randn(nb_gaussians, 3, device=device)
        deform_rotations = torch.randn(nb_gaussians, 4, device=device)

        layers_data = LayersData(
            layers=[
                LayerDataDeformable(
                    positions=positions,
                    rotations=rotations,
                    scales=scales,
                    densities=densities,
                    extra_signal=extra_signal,
                    camera_extra_signal=camera_extra_signal,
                    lidar_extra_signal=lidar_extra_signal,
                    features_albedo=features_albedo,
                    features_specular=features_specular,
                    embed_data=embed_data,
                    poses=tracks_poses,
                    keep_mask=keep_mask,
                    tracks_ids=tracks_ids,
                    deform_positions=deform_positions,
                    deform_rotations=deform_rotations,
                ),
            ],
            frame_timestamp_us=None,
        )

        for tensor in [
            positions,
            rotations,
            scales,
            densities,
            extra_signal,
            camera_extra_signal,
            lidar_extra_signal,
            features_albedo,
            features_specular,
            tracks_poses,
            deform_positions,
            deform_rotations,
        ]:
            tensor.requires_grad = True
            tensor.grad = None

        # Forward pass.
        results = collector.collect(layers_data)
        ground_truth_positions = positions + deform_positions + tracks_poses[tracks_ids][:, :3]
        self.assertTrue(torch.allclose(results.positions, ground_truth_positions))
        enabled_densities = results.densities[keep_mask[tracks_ids]]
        disabled_densities = results.densities[~keep_mask[tracks_ids]]
        self.assertEqual(disabled_densities.shape, (1, 1))
        self.assertTrue(torch.all(disabled_densities == 0))
        self.assertTrue(torch.all(enabled_densities != 0))

        # Backward pass.
        buffers = [getattr(results, field.name) for field in fields(results)]
        gradients = [torch.randn_like(buffer) for buffer in buffers]
        torch.autograd.backward(buffers, gradients)

        for tensor in [extra_signal, lidar_extra_signal]:
            self.assertEqual(tensor.grad, None)

        for tensor in [
            positions,
            rotations,
            scales,
            densities,
            camera_extra_signal,
            features_albedo,
            features_specular,
        ]:
            self.assertTrue(torch.all(tensor.grad[keep_mask[tracks_ids]] != 0))
            self.assertTrue(torch.all(tensor.grad[~keep_mask[tracks_ids]] == 0))

        self.assertTrue(torch.all(tracks_poses.grad[keep_mask] != 0))
        self.assertTrue(torch.all(tracks_poses.grad[~keep_mask] == 0))

        self.assertTrue(torch.all(positions.grad[keep_mask[tracks_ids]] == gradients[0][keep_mask[tracks_ids]]))
        self.assertTrue(torch.all(deform_positions.grad[keep_mask[tracks_ids]] == gradients[0][keep_mask[tracks_ids]]))

    def test_performance(self):
        layers_nb_gaussians = [303609, 57965, 102079]
        layers_config = LayersConfig(
            layers=[
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=20,
                    embed_config=IndividualRemapTimeInputEmbeddingConfig(
                        timestamps_us_ranges=torch.tensor([[0, 1000000]], dtype=torch.int64, device=device),
                        remap_min=0.0,
                        remap_max=1.0,
                    ),
                ),
            ],
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
        )

        layers_data = LayersData(
            layers=[
                LayerDataSH(
                    positions=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 3, device=device)),
                    rotations=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 4, device=device)),
                    scales=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 3, device=device)),
                    densities=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 1, device=device)),
                    extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 0, device=device)),
                    camera_extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 20, device=device)),
                    lidar_extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 0, device=device)),
                    features_albedo=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 3, device=device)),
                    features_specular=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 45, device=device)),
                    embed_data=None,
                ),
                LayerDataSH(
                    positions=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 3, device=device)),
                    rotations=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 4, device=device)),
                    scales=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 3, device=device)),
                    densities=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 1, device=device)),
                    extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 0, device=device)),
                    camera_extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 20, device=device)),
                    lidar_extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 0, device=device)),
                    features_albedo=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 3, device=device)),
                    features_specular=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 45, device=device)),
                    embed_data=None,
                ),
                LayerDataSH(
                    positions=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 3, device=device)),
                    rotations=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 4, device=device)),
                    scales=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 3, device=device)),
                    densities=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 1, device=device)),
                    extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 0, device=device)),
                    camera_extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 20, device=device)),
                    lidar_extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 0, device=device)),
                    features_albedo=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 20, 3, device=device)),
                    features_specular=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 45, device=device)),
                    embed_data=IndividualRemapTimeInputEmbeddingData(
                        instance_idx=torch.zeros(layers_nb_gaussians[2], dtype=torch.int32, device=device),
                    ),
                ),
            ],
            frame_timestamp_us=500000,
        )

        def zero_grad(obj):
            if isinstance(obj, torch.Tensor):
                obj.grad = None
            elif is_dataclass(obj):
                for field in fields(obj):
                    zero_grad(getattr(obj, field.name))
            elif isinstance(obj, list):
                for item in obj:
                    zero_grad(item)

        with profile("Creation"):
            collector = CreateGaussianParameterCollector(layers_config)

        NB_WARMUP = 10
        NB_MEASURE = 100

        with profile("Warmup"):
            for _ in range(NB_WARMUP):
                results = collector.collect(layers_data)
                buffers = [getattr(results, field.name) for field in fields(results)]
                gradients = [torch.randn_like(buffer) for buffer in buffers]
                torch.autograd.backward(buffers, gradients)

        with profile("Benchmark"):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(NB_MEASURE):
                with profile("Zero grad"):
                    zero_grad(layers_data)
                with profile("Forward"):
                    results = collector.collect(layers_data)
                with profile("Backward"):
                    buffers = [getattr(results, field.name) for field in fields(results)]
                    gradients = [torch.randn_like(buffer) for buffer in buffers]
                    torch.autograd.backward(buffers, gradients)
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            print(f"Collect took {(end_time - start_time) / NB_MEASURE * 1e3:.6f} ms")

        with profile("Benchmark Sync"):
            for _ in range(NB_MEASURE):
                with profile("Zero grad"):
                    zero_grad(layers_data)
                    torch.cuda.synchronize()
                with profile("Forward"):
                    results = collector.collect(layers_data)
                    torch.cuda.synchronize()
                with profile("Backward"):
                    buffers = [getattr(results, field.name) for field in fields(results)]
                    gradients = [torch.randn_like(buffer) for buffer in buffers]
                    torch.autograd.backward(buffers, gradients)
                    torch.cuda.synchronize()


class TestSwapTimeEmbeddingTable(unittest.TestCase):
    """The validating table swap is the ONLY safety net between a bad table and an unbounded kernel gather
    (handler build-time asserts never re-run; the Slang kernel has no bounds check)."""

    @staticmethod
    def _config(table: torch.Tensor) -> IndividualRemapTimeInputEmbeddingConfig:
        return IndividualRemapTimeInputEmbeddingConfig(timestamps_us_ranges=table, remap_min=0.0, remap_max=1.0)

    def test_swap_installs_valid_table(self):
        config = self._config(torch.tensor([[0, 10]], dtype=torch.int64))
        new = torch.tensor([[0, 10], [5, 20]], dtype=torch.int64)
        installed = swap_time_embedding_table(config, new, min_rows=2)
        self.assertIs(config.timestamps_us_ranges, installed)
        self.assertIs(installed, new)  # no widening needed -> installed as-is

    def test_degenerate_row_widened_on_a_copy(self):
        config = self._config(torch.tensor([[0, 10]], dtype=torch.int64))
        new = torch.tensor([[0, 10], [7, 7]], dtype=torch.int64)  # one-frame track from the min/max oracle
        installed = swap_time_embedding_table(config, new, min_rows=2)
        self.assertIsNot(installed, new)  # widened copy; the caller's tensor is untouched
        self.assertEqual(new[1, 1].item(), 7)
        self.assertEqual(installed[1, 1].item(), 8)  # +1 us satisfies the strict col0 < col1 invariant
        self.assertTrue(bool((installed[:, 0] < installed[:, 1]).all()))

    def test_swap_rejects_invalid_tables(self):
        config = self._config(torch.tensor([[0, 10]], dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "int64"):
            swap_time_embedding_table(config, torch.tensor([[0, 10]], dtype=torch.int32), min_rows=1)
        with self.assertRaisesRegex(ValueError, r"\[n, 2\]"):
            swap_time_embedding_table(config, torch.zeros(2, 3, dtype=torch.int64), min_rows=1)
        # (requires_grad=True is unconstructible for int64 tensors, so the helper's no-grad check is
        # transitively guaranteed by the dtype check; it stays as belt-and-suspenders.)
        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            swap_time_embedding_table(config, torch.tensor([[0, 10]], dtype=torch.int64), min_rows=2)
        with self.assertRaisesRegex(ValueError, "col0 < col1"):
            swap_time_embedding_table(config, torch.tensor([[10, 0]], dtype=torch.int64), min_rows=1)
        # A failed swap must leave the installed table untouched.
        self.assertEqual(config.timestamps_us_ranges.shape, (1, 2))

    def test_swap_rejects_config_without_table(self):
        with self.assertRaisesRegex(TypeError, "no per-instance table"):
            swap_time_embedding_table(
                cast(IndividualRemapTimeInputEmbeddingConfig, object()),
                torch.tensor([[0, 10]], dtype=torch.int64),
                min_rows=1,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "device-change guard needs a real CUDA device")
    def test_swap_rejects_device_change(self):
        # A CPU table swapped into a CUDA-built collector passes every value/shape check and then faults
        # at the kernel gather. The device guard must reject it up front (before dtype/shape/row checks).
        cuda_config = self._config(torch.tensor([[0, 10]], dtype=torch.int64, device="cuda"))
        cpu_table = torch.tensor([[0, 10], [5, 20]], dtype=torch.int64)  # valid but on the wrong device
        with self.assertRaisesRegex(ValueError, "device change not allowed"):
            swap_time_embedding_table(cuda_config, cpu_table, min_rows=2)
        # A rejected swap leaves the installed (CUDA) table untouched.
        self.assertEqual(cuda_config.timestamps_us_ranges.device.type, "cuda")

    def test_swapped_table_takes_effect_on_next_collect(self):
        """A post-construction swap must be indistinguishable from constructing with the new table.

        This is the mechanism the in-place insert fix relies on:
        the handler re-reads `embed_config.timestamps_us_ranges` every collect, so the appended row is
        live for the next render with no rebuild (and table length is not a kernel-spec key)."""
        nb_gaussians = 256
        fourier_dim = 4
        torch.manual_seed(3)
        tensors = dict(
            positions=torch.randn(nb_gaussians, 3, device=device),
            rotations=torch.randn(nb_gaussians, 4, device=device),
            scales=torch.randn(nb_gaussians, 3, device=device),
            densities=torch.randn(nb_gaussians, 1, device=device),
            extra_signal=torch.zeros(nb_gaussians, 0, device=device),
            camera_extra_signal=torch.zeros(nb_gaussians, 0, device=device),
            lidar_extra_signal=torch.zeros(nb_gaussians, 0, device=device),
            features_albedo=torch.randn(nb_gaussians, fourier_dim, 3, device=device),
            features_specular=torch.randn(nb_gaussians, 45, device=device),
        )
        old_table = torch.tensor([[0, 1_000_000]], dtype=torch.int64, device=device)
        new_table = torch.tensor([[0, 1_000_000], [400_000, 2_000_000]], dtype=torch.int64, device=device)

        def collect_features(config: IndividualRemapTimeInputEmbeddingConfig, instance: int) -> torch.Tensor:
            layers_config = LayersConfig(
                layers=[
                    LayerConfigSH(
                        rotation_activation=RotationActivation.NORMALIZE,
                        scale_activation=ScaleActivation.EXP,
                        density_activation=DensityActivation.SIGMOID,
                        fourier_features_dim=fourier_dim,
                        embed_config=config,
                    )
                ],
                extra_signal_dim=0,
                camera_extra_signal_dim=0,
                lidar_extra_signal_dim=0,
                albedo_dim=3,
                specular_dim=45,
                sh_features_dtype="fp32",
            )
            collector = CreateGaussianParameterCollector(layers_config)
            layers_data = LayersData(
                layers=[
                    LayerDataSH(
                        **{name: tensor.clone() for name, tensor in tensors.items()},
                        embed_data=IndividualRemapTimeInputEmbeddingData(
                            instance_idx=torch.full((nb_gaussians,), instance, dtype=torch.int32, device=device)
                        ),
                    )
                ],
                frame_timestamp_us=500_000,
            )
            return collector.collect(layers_data).features.detach().clone()

        # Swap on a live config, then read the NEW row (index 1) -- vs a fresh collector built with the
        # new table directly: bit-equal. And row 1's range differs from row 0's -> features differ.
        swapped_config = self._config(old_table)
        baseline_row0 = collect_features(swapped_config, instance=0)
        swap_time_embedding_table(swapped_config, new_table, min_rows=2)
        after_swap_row1 = collect_features(swapped_config, instance=1)
        fresh_row1 = collect_features(self._config(new_table), instance=1)
        self.assertTrue(torch.equal(after_swap_row1, fresh_row1))
        self.assertFalse(torch.equal(after_swap_row1, baseline_row0))


if __name__ == "__main__":
    unittest.main()
