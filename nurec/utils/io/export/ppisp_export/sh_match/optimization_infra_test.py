# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ``nurec.utils.io.export.ppisp_export.sh_match.optimization``.

Hermetic: a tiny ``_MockPPISPModule`` stands in for the trained PPISP,
and ``mock.patch.object`` redirects the warm-start function so the
adapter test never needs the real ``ppisp.ppisp_apply``.

Coverage:
* ``SHMatchingOptimizationConfig`` defaults, every validator branch,
  resolved specular LR semantics.
* ``FixedCameraFramePPISP`` deep-copy independence, vignetting zeroing,
  controller forced off, forward routing to the fixed indices.
* ``PPISPSHOptimizationAdapter`` construction validation, warmstart
  delegation, identity fit-transform clamp, optimizer setup
  delegation.
* ``_setup_optimization_parameters`` parameter freezing, group LRs,
  attribute / type guards.
"""

from __future__ import annotations

import copy
import math
import unittest

from typing import Any, Dict, Iterable, List, Tuple
from unittest import mock

import torch
import torch.nn as nn

from nurec.utils.io.export.ppisp_export.sh_match import optimization as optimization_module
from nurec.utils.io.export.ppisp_export.sh_match.optimization import (
    DEFAULT_LR_ALBEDO,
    DEFAULT_LR_DENSITY,
    DEFAULT_NUM_ITERATIONS,
    SPECULAR_LR_RATIO,
    FixedCameraFramePPISP,
    PPISPSHOptimizationAdapter,
    SHMatchingOptimizationConfig,
    _setup_optimization_parameters,
    _validate_camera_frame_ids,
)


# =============================================================================
# Test doubles
# =============================================================================


class _MockPPISPConfig:
    """Minimal stand-in for ``PPISP.config``."""

    def __init__(self, use_controller: bool = True) -> None:
        self.use_controller = use_controller


class _MockPPISPModule(nn.Module):
    """Trained-PPISP-shaped ``nn.Module`` with deep-copyable parameters.

    The forward signature mirrors the real PPISP's ``__call__`` so
    :class:`FixedCameraFramePPISP` can route to it transparently.
    """

    def __init__(
        self,
        num_cameras: int = 2,
        num_frames: int = 3,
        seed: int = 0,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.num_cameras = num_cameras
        self.config = _MockPPISPConfig(use_controller=True)

        self.exposure_params = nn.Parameter(torch.randn(num_frames, generator=generator), requires_grad=False)
        self.color_params = nn.Parameter(torch.randn(num_frames, 8, generator=generator), requires_grad=False)
        self.vignetting_params = nn.Parameter(torch.randn(num_cameras, 3, 5, generator=generator), requires_grad=False)
        self.crf_params = nn.Parameter(torch.randn(num_cameras, 3, 4, generator=generator), requires_grad=False)

        self.calls: List[Dict[str, Any]] = []

    def forward(
        self,
        rgb: torch.Tensor,
        pixel_coords: torch.Tensor,
        resolution: Tuple[int, int],
        camera_idx: int,
        frame_idx: int,
        exposure_prior: object = None,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "rgb_id": id(rgb),
                "pixel_coords_id": id(pixel_coords),
                "resolution": resolution,
                "camera_idx": camera_idx,
                "frame_idx": frame_idx,
                "exposure_prior": exposure_prior,
            }
        )
        return rgb


class _MockGaussianModel(nn.Module):
    """Minimal SH-coefficient bearer that satisfies ``nn.Module.parameters()``."""

    def __init__(self, num_gaussians: int = 4, num_specular_dims: int = 9, seed: int = 1) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.features_albedo = nn.Parameter(torch.randn(num_gaussians, 3, generator=generator))
        self.features_specular = nn.Parameter(torch.randn(num_gaussians, num_specular_dims, generator=generator))
        self.densities = nn.Parameter(torch.randn(num_gaussians, 1, generator=generator))
        self.unrelated_buffer = nn.Parameter(torch.zeros(num_gaussians, 4))


class _BareSHBearer:
    """Non-``nn.Module`` SH bearer used to test the ``isinstance(param, nn.Parameter)`` guard.

    ``nn.Module.__setattr__`` rejects non-``nn.Parameter`` assignment to
    a name already registered as a parameter, which makes it impossible
    to construct a "wrong-type albedo" via the production model. This
    minimal bearer skips ``nn.Module``'s strict bookkeeping so the
    guard branch is reachable from the test.
    """

    def __init__(self) -> None:
        self.features_albedo: object = torch.zeros(4, 3)
        self.features_specular = nn.Parameter(torch.zeros(4, 9))
        self.densities = nn.Parameter(torch.zeros(4, 1))

    def parameters(self) -> Iterable[nn.Parameter]:
        return iter([self.features_specular, self.densities])


def _make_cpu_config(**overrides: Any) -> SHMatchingOptimizationConfig:
    """Build a config pinned to CPU so tests run without CUDA."""
    base: Dict[str, Any] = {"device": "cpu"}
    base.update(overrides)
    return SHMatchingOptimizationConfig(**base)


# =============================================================================
# SHMatchingOptimizationConfig
# =============================================================================


class SHMatchingOptimizationConfigTest(unittest.TestCase):
    def test_defaults_match_documented_values(self) -> None:
        config = _make_cpu_config()
        self.assertEqual(config.num_iterations, DEFAULT_NUM_ITERATIONS)
        self.assertAlmostEqual(config.learning_rate_albedo, DEFAULT_LR_ALBEDO)
        self.assertIsNone(config.learning_rate_specular)
        self.assertAlmostEqual(config.learning_rate_density, DEFAULT_LR_DENSITY)
        self.assertEqual(config.device, "cpu")

    def test_specular_lr_resolves_to_albedo_ratio_when_none(self) -> None:
        config = _make_cpu_config(learning_rate_albedo=2.0e-3, learning_rate_specular=None)
        self.assertAlmostEqual(
            config.resolved_learning_rate_specular,
            2.0e-3 * SPECULAR_LR_RATIO,
        )

    def test_specular_lr_uses_explicit_value_when_set(self) -> None:
        config = _make_cpu_config(learning_rate_specular=7.5e-4)
        self.assertAlmostEqual(config.resolved_learning_rate_specular, 7.5e-4)

    def test_zero_num_iterations_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_iterations"):
            _make_cpu_config(num_iterations=0)

    def test_negative_num_iterations_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_iterations"):
            _make_cpu_config(num_iterations=-1)

    def test_non_int_num_iterations_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "num_iterations"):
            _make_cpu_config(num_iterations=1.5)  # type: ignore[arg-type]

    def test_bool_num_iterations_raises(self) -> None:
        # ``bool`` subclasses ``int``; explicit guard rejects True/False.
        with self.assertRaisesRegex(TypeError, "num_iterations"):
            _make_cpu_config(num_iterations=True)  # type: ignore[arg-type]

    def test_non_positive_albedo_lr_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "learning_rate_albedo"):
            _make_cpu_config(learning_rate_albedo=0.0)
        with self.assertRaisesRegex(ValueError, "learning_rate_albedo"):
            _make_cpu_config(learning_rate_albedo=-1.0)

    def test_non_finite_albedo_lr_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "learning_rate_albedo"):
            _make_cpu_config(learning_rate_albedo=math.nan)
        with self.assertRaisesRegex(ValueError, "learning_rate_albedo"):
            _make_cpu_config(learning_rate_albedo=math.inf)

    def test_non_numeric_albedo_lr_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "learning_rate_albedo"):
            _make_cpu_config(learning_rate_albedo="1e-3")  # type: ignore[arg-type]

    def test_non_positive_density_lr_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "learning_rate_density"):
            _make_cpu_config(learning_rate_density=0.0)

    def test_non_positive_specular_lr_raises_when_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "learning_rate_specular"):
            _make_cpu_config(learning_rate_specular=0.0)
        with self.assertRaisesRegex(ValueError, "learning_rate_specular"):
            _make_cpu_config(learning_rate_specular=-1.0e-4)

    def test_empty_device_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "device"):
            _make_cpu_config(device="")

    def test_non_string_device_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "device"):
            _make_cpu_config(device=None)  # type: ignore[arg-type]

    def test_invalid_device_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "device"):
            _make_cpu_config(device="not-a-device")

    def test_config_is_frozen(self) -> None:
        config = _make_cpu_config()
        with self.assertRaises(Exception):  # noqa: B017 -- dataclasses.FrozenInstanceError
            config.num_iterations = 999  # type: ignore[misc]


# =============================================================================
# FixedCameraFramePPISP
# =============================================================================


class FixedCameraFramePPISPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ppisp = _MockPPISPModule(num_cameras=2, num_frames=3)
        self.original_vignetting = self.ppisp.vignetting_params.detach().clone()
        self.fixed = FixedCameraFramePPISP(
            ppisp_module=self.ppisp,
            camera_id=1,
            frame_id=2,
            device=torch.device("cpu"),
        )

    def test_records_camera_and_frame_ids(self) -> None:
        self.assertEqual(self.fixed.camera_id, 1)
        self.assertEqual(self.fixed.frame_id, 2)

    def test_zeros_vignetting_in_copy(self) -> None:
        self.assertTrue(
            torch.equal(self.fixed.ppisp.vignetting_params, torch.zeros_like(self.fixed.ppisp.vignetting_params))
        )

    def test_does_not_mutate_source_module(self) -> None:
        # Source vignetting must be untouched after the deep copy.
        torch.testing.assert_close(
            self.ppisp.vignetting_params.detach(),
            self.original_vignetting,
            rtol=0.0,
            atol=0.0,
        )

    def test_disables_controller(self) -> None:
        self.assertFalse(self.fixed.ppisp.config.use_controller)
        # Source's controller flag must be untouched.
        self.assertTrue(self.ppisp.config.use_controller)

    def test_forward_routes_to_fixed_indices(self) -> None:
        rgb = torch.rand(4, 3)
        pixel_coords = torch.rand(4, 2)
        out = self.fixed(rgb, pixel_coords, resolution=(800, 600))
        torch.testing.assert_close(out, rgb)
        self.assertEqual(len(self.fixed.ppisp.calls), 1)
        call = self.fixed.ppisp.calls[0]
        self.assertEqual(call["camera_idx"], 1)
        self.assertEqual(call["frame_idx"], 2)
        self.assertEqual(call["resolution"], (800, 600))
        self.assertIsNone(call["exposure_prior"])

    def test_forward_ignores_caller_camera_and_frame_kwargs(self) -> None:
        rgb = torch.rand(4, 3)
        pixel_coords = torch.rand(4, 2)
        self.fixed(
            rgb,
            pixel_coords,
            resolution=(800, 600),
            camera_idx=999,
            frame_idx=999,
            exposure_prior="should-be-ignored",
        )
        call = self.fixed.ppisp.calls[0]
        self.assertEqual(call["camera_idx"], 1)
        self.assertEqual(call["frame_idx"], 2)
        self.assertIsNone(call["exposure_prior"])

    def test_eval_mode(self) -> None:
        self.assertFalse(self.fixed.ppisp.training)


# =============================================================================
# _validate_camera_frame_ids
# =============================================================================


class ValidateCameraFrameIdsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ppisp = _MockPPISPModule(num_cameras=2, num_frames=3)

    def test_accepts_valid_indices(self) -> None:
        _validate_camera_frame_ids(self.ppisp, 0, 0)
        _validate_camera_frame_ids(self.ppisp, 1, 2)

    def test_rejects_negative_camera_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "camera_id"):
            _validate_camera_frame_ids(self.ppisp, -1, 0)

    def test_rejects_out_of_bounds_camera_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "camera_id"):
            _validate_camera_frame_ids(self.ppisp, 2, 0)

    def test_rejects_negative_frame_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame_id"):
            _validate_camera_frame_ids(self.ppisp, 0, -1)

    def test_rejects_out_of_bounds_frame_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame_id"):
            _validate_camera_frame_ids(self.ppisp, 0, 3)

    def test_rejects_non_int_camera_id(self) -> None:
        with self.assertRaisesRegex(TypeError, "camera_id"):
            _validate_camera_frame_ids(self.ppisp, 0.5, 0)  # type: ignore[arg-type]

    def test_rejects_bool_frame_id(self) -> None:
        with self.assertRaisesRegex(TypeError, "frame_id"):
            _validate_camera_frame_ids(self.ppisp, 0, True)  # type: ignore[arg-type]

    def test_missing_ppisp_attribute_raises(self) -> None:
        broken = _MockPPISPModule()
        del broken.exposure_params
        with self.assertRaisesRegex(AttributeError, "exposure_params"):
            _validate_camera_frame_ids(broken, 0, 0)


# =============================================================================
# PPISPSHOptimizationAdapter
# =============================================================================


class PPISPSHOptimizationAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _make_cpu_config()
        self.ppisp = _MockPPISPModule(num_cameras=2, num_frames=3)
        self.adapter = PPISPSHOptimizationAdapter(
            config=self.config,
            ppisp_module=self.ppisp,
            camera_id=1,
            frame_id=2,
        )

    def test_properties_record_construction_args(self) -> None:
        self.assertIs(self.adapter.config, self.config)
        self.assertEqual(self.adapter.camera_id, 1)
        self.assertEqual(self.adapter.frame_id, 2)
        self.assertEqual(self.adapter.device, torch.device("cpu"))
        self.assertIsInstance(self.adapter.fixed_ppisp, FixedCameraFramePPISP)

    def test_rejects_non_config(self) -> None:
        with self.assertRaisesRegex(TypeError, "config"):
            PPISPSHOptimizationAdapter(
                config="not-a-config",  # type: ignore[arg-type]
                ppisp_module=self.ppisp,
                camera_id=0,
                frame_id=0,
            )

    def test_rejects_bad_camera_id_at_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "camera_id"):
            PPISPSHOptimizationAdapter(
                config=self.config,
                ppisp_module=self.ppisp,
                camera_id=99,
                frame_id=0,
            )

    def test_rejects_bad_frame_id_at_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame_id"):
            PPISPSHOptimizationAdapter(
                config=self.config,
                ppisp_module=self.ppisp,
                camera_id=0,
                frame_id=99,
            )

    def test_apply_fit_transform_clamps_to_unit_range(self) -> None:
        rgb = torch.tensor([-0.5, 0.0, 0.25, 0.75, 1.0, 1.5])
        clamped = self.adapter.apply_fit_transform(rgb)
        torch.testing.assert_close(clamped, torch.tensor([0.0, 0.0, 0.25, 0.75, 1.0, 1.0]))

    def test_apply_fit_transform_does_not_mutate_input(self) -> None:
        rgb = torch.tensor([-0.5, 1.5])
        original = rgb.clone()
        _ = self.adapter.apply_fit_transform(rgb)
        torch.testing.assert_close(rgb, original)

    def test_warmstart_delegates_with_correct_kwargs(self) -> None:
        model = _MockGaussianModel()
        with mock.patch.object(optimization_module, "warmstart_sh_dc_from_ppisp") as warm_mock:
            warm_mock.return_value = (0.42, torch.zeros(8))
            exposure, color = self.adapter.warmstart(model)
        warm_mock.assert_called_once_with(
            model=model,
            ppisp_module=self.ppisp,
            camera_id=1,
            frame_id=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(exposure, 0.42)
        self.assertTrue(torch.equal(color, torch.zeros(8)))

    def test_setup_optimizer_returns_three_group_adam(self) -> None:
        model = _MockGaussianModel()
        optimizer = self.adapter.setup_optimizer(model)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(len(optimizer.param_groups), 3)


# =============================================================================
# _setup_optimization_parameters
# =============================================================================


class SetupOptimizationParametersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _make_cpu_config()
        self.model = _MockGaussianModel()

    def test_freezes_all_other_parameters(self) -> None:
        _setup_optimization_parameters(self.model, self.config)
        self.assertFalse(self.model.unrelated_buffer.requires_grad)

    def test_unfreezes_fit_parameters(self) -> None:
        _setup_optimization_parameters(self.model, self.config)
        self.assertTrue(self.model.features_albedo.requires_grad)
        self.assertTrue(self.model.features_specular.requires_grad)
        self.assertTrue(self.model.densities.requires_grad)

    def test_returns_three_groups_with_correct_lrs(self) -> None:
        config = _make_cpu_config(
            learning_rate_albedo=4.0e-3,
            learning_rate_specular=1.0e-4,
            learning_rate_density=8.0e-2,
        )
        optimizer = _setup_optimization_parameters(self.model, config)
        self.assertEqual(len(optimizer.param_groups), 3)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 4.0e-3)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1.0e-4)
        self.assertAlmostEqual(optimizer.param_groups[2]["lr"], 8.0e-2)

    def test_specular_lr_uses_resolved_default(self) -> None:
        config = _make_cpu_config(learning_rate_albedo=2.0e-3, learning_rate_specular=None)
        optimizer = _setup_optimization_parameters(self.model, config)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 2.0e-3 * SPECULAR_LR_RATIO)

    def test_param_groups_target_correct_tensors(self) -> None:
        optimizer = _setup_optimization_parameters(self.model, self.config)
        self.assertIs(optimizer.param_groups[0]["params"][0], self.model.features_albedo)
        self.assertIs(optimizer.param_groups[1]["params"][0], self.model.features_specular)
        self.assertIs(optimizer.param_groups[2]["params"][0], self.model.densities)

    def test_missing_features_albedo_raises(self) -> None:
        broken = _MockGaussianModel()
        del broken.features_albedo
        with self.assertRaisesRegex(AttributeError, "features_albedo"):
            _setup_optimization_parameters(broken, self.config)

    def test_missing_densities_raises(self) -> None:
        broken = _MockGaussianModel()
        del broken.densities
        with self.assertRaisesRegex(AttributeError, "densities"):
            _setup_optimization_parameters(broken, self.config)

    def test_features_albedo_not_parameter_raises(self) -> None:
        # Production ``nn.Module`` cannot hold a non-Parameter under a name
        # already registered as a parameter, so the "wrong-type albedo"
        # branch is exercised through the bare bearer fixture.
        broken = _BareSHBearer()
        with self.assertRaisesRegex(TypeError, "features_albedo"):
            _setup_optimization_parameters(broken, self.config)  # type: ignore[arg-type]

    def test_model_without_parameters_method_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "parameters"):
            _setup_optimization_parameters(object(), self.config)  # type: ignore[arg-type]


# =============================================================================
# Module exports / contracts
# =============================================================================


class ModuleConstantsTest(unittest.TestCase):
    def test_specular_lr_ratio_is_twentieth(self) -> None:
        self.assertAlmostEqual(SPECULAR_LR_RATIO, 1.0 / 20.0)

    def test_default_num_iterations(self) -> None:
        self.assertEqual(DEFAULT_NUM_ITERATIONS, 3000)

    def test_default_lrs(self) -> None:
        self.assertAlmostEqual(DEFAULT_LR_ALBEDO, 2.5e-3)
        self.assertAlmostEqual(DEFAULT_LR_DENSITY, 5.0e-2)


# =============================================================================
# Sanity: deep-copy isolates fixed PPISP from original
# =============================================================================


class DeepCopyIsolationTest(unittest.TestCase):
    def test_mutating_fixed_ppisp_does_not_affect_source(self) -> None:
        ppisp = _MockPPISPModule(num_cameras=1, num_frames=1)
        snapshot = copy.deepcopy(ppisp.vignetting_params.detach())
        fixed = FixedCameraFramePPISP(ppisp, camera_id=0, frame_id=0, device=torch.device("cpu"))
        with torch.no_grad():
            fixed.ppisp.vignetting_params.add_(torch.full_like(fixed.ppisp.vignetting_params, 100.0))
        torch.testing.assert_close(ppisp.vignetting_params.detach(), snapshot, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
