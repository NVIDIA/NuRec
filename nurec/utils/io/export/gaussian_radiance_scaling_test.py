# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ``nurec.utils.io.export.gaussian_radiance_scaling``.

Covers:
* Public ``apply_radiance_scale_to_sh`` behavior across success and error paths
* Internal ``_compute_dc_offset_compensation`` numeric correctness
* Internal ``_validate_radiance_scaling_inputs`` fail-loud branches
* Round-trip property: forward DC RGB after scaling equals ``s`` * original
"""

from __future__ import annotations

import math
import unittest

from typing import Any, cast

import torch

from nurec.models.gaussians.utils import C0
from nurec.utils.io.export.gaussian_radiance_scaling import (
    _compute_dc_offset_compensation,
    _validate_radiance_scaling_inputs,
    apply_radiance_scale_to_sh,
    apply_radiance_scale_to_sh_nodes,
)


# =============================================================================
# Test fixtures
# =============================================================================


class _MockSHModel:
    """Minimal model exposing the two SH parameter tensors.

    Mirrors the surface ``apply_radiance_scale_to_sh`` actually touches, so
    tests can run without instantiating a full ``SHGaussianModel`` (which
    requires a configured datasource and trainer context).
    """

    def __init__(self, num_gaussians: int = 4, num_specular_dims: int = 9, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.features_albedo = torch.randn(num_gaussians, 3, generator=generator, dtype=torch.float32)
        self.features_specular = torch.randn(num_gaussians, num_specular_dims, generator=generator, dtype=torch.float32)


def _evaluate_dc_rgb(features_albedo: torch.Tensor) -> torch.Tensor:
    """Evaluate the SH DC term: ``rgb = features_albedo * C0 + 0.5``."""
    return features_albedo * C0 + 0.5


# =============================================================================
# Public API: apply_radiance_scale_to_sh
# =============================================================================


class ApplyRadianceScaleToShTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _MockSHModel(num_gaussians=8, num_specular_dims=9, seed=42)
        self.original_albedo = self.model.features_albedo.clone()
        self.original_specular = self.model.features_specular.clone()

    def test_brightening_scale_doubles_dc_rgb(self) -> None:
        scale = 2.0
        original_dc_rgb = _evaluate_dc_rgb(self.original_albedo)
        apply_radiance_scale_to_sh(self.model, scale)
        new_dc_rgb = _evaluate_dc_rgb(self.model.features_albedo)
        torch.testing.assert_close(new_dc_rgb, scale * original_dc_rgb, rtol=1e-6, atol=1e-6)

    def test_dimming_scale_halves_dc_rgb(self) -> None:
        scale = 0.5
        original_dc_rgb = _evaluate_dc_rgb(self.original_albedo)
        apply_radiance_scale_to_sh(self.model, scale)
        new_dc_rgb = _evaluate_dc_rgb(self.model.features_albedo)
        torch.testing.assert_close(new_dc_rgb, scale * original_dc_rgb, rtol=1e-6, atol=1e-6)

    def test_unity_scale_is_bitwise_noop(self) -> None:
        apply_radiance_scale_to_sh(self.model, 1.0)
        self.assertTrue(torch.equal(self.model.features_albedo, self.original_albedo))
        self.assertTrue(torch.equal(self.model.features_specular, self.original_specular))

    def test_specular_bands_scaled_linearly(self) -> None:
        scale = 1.7
        apply_radiance_scale_to_sh(self.model, scale)
        torch.testing.assert_close(
            self.model.features_specular,
            scale * self.original_specular,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_dc_offset_compensation_applied_to_albedo(self) -> None:
        scale = 0.3
        apply_radiance_scale_to_sh(self.model, scale)
        expected_albedo = scale * self.original_albedo + (scale - 1.0) * 0.5 / C0
        torch.testing.assert_close(self.model.features_albedo, expected_albedo, rtol=1e-6, atol=1e-6)

    def test_zero_scale_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            apply_radiance_scale_to_sh(self.model, 0.0)

    def test_negative_scale_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            apply_radiance_scale_to_sh(self.model, -0.5)

    def test_nan_scale_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            apply_radiance_scale_to_sh(self.model, float("nan"))

    def test_inf_scale_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            apply_radiance_scale_to_sh(self.model, float("inf"))

    def test_string_scale_raises_type_error(self) -> None:
        with self.assertRaisesRegex(TypeError, "real number"):
            apply_radiance_scale_to_sh(self.model, "1.0")  # type: ignore[arg-type]

    def test_bool_scale_raises_type_error(self) -> None:
        # ``bool`` is a subclass of ``int`` in Python; this guard makes the
        # contract explicit and rejects accidental True/False arguments.
        with self.assertRaisesRegex(TypeError, "real number"):
            apply_radiance_scale_to_sh(self.model, True)  # type: ignore[arg-type]

    def test_missing_features_albedo_raises_attribute_error(self) -> None:
        broken_model = _MockSHModel()
        del broken_model.features_albedo
        with self.assertRaisesRegex(AttributeError, "features_albedo"):
            apply_radiance_scale_to_sh(broken_model, 1.5)

    def test_missing_features_specular_raises_attribute_error(self) -> None:
        broken_model = _MockSHModel()
        del broken_model.features_specular
        with self.assertRaisesRegex(AttributeError, "features_specular"):
            apply_radiance_scale_to_sh(broken_model, 1.5)

    def test_features_albedo_wrong_type_raises_type_error(self) -> None:
        broken_model = _MockSHModel()
        broken_model.features_albedo = [[0.0, 0.0, 0.0]]  # type: ignore[assignment]
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            apply_radiance_scale_to_sh(broken_model, 1.5)

    def test_validation_runs_before_mutation(self) -> None:
        original_albedo = self.model.features_albedo.clone()
        original_specular = self.model.features_specular.clone()
        with self.assertRaises(ValueError):
            apply_radiance_scale_to_sh(self.model, -1.0)
        self.assertTrue(torch.equal(self.model.features_albedo, original_albedo))
        self.assertTrue(torch.equal(self.model.features_specular, original_specular))


class ApplyRadianceScaleToShNodesTest(unittest.TestCase):
    def test_scales_each_node(self) -> None:
        node_a = _MockSHModel(num_gaussians=4, seed=0)
        node_b = _MockSHModel(num_gaussians=5, seed=1)
        original_a = node_a.features_albedo.clone()
        original_b = node_b.features_albedo.clone()

        apply_radiance_scale_to_sh_nodes([("static", node_a), ("dynamic", node_b)], 2.0)

        torch.testing.assert_close(_evaluate_dc_rgb(node_a.features_albedo), 2.0 * _evaluate_dc_rgb(original_a))
        torch.testing.assert_close(_evaluate_dc_rgb(node_b.features_albedo), 2.0 * _evaluate_dc_rgb(original_b))

    def test_validates_all_nodes_before_mutating_any_node(self) -> None:
        good_node = _MockSHModel(seed=0)
        original_albedo = good_node.features_albedo.clone()
        original_specular = good_node.features_specular.clone()

        with self.assertRaisesRegex(AttributeError, "gaussian node 'broken'.*features_albedo"):
            apply_radiance_scale_to_sh_nodes([("good", good_node), ("broken", cast(Any, object()))], 2.0)

        self.assertTrue(torch.equal(good_node.features_albedo, original_albedo))
        self.assertTrue(torch.equal(good_node.features_specular, original_specular))

    def test_empty_node_list_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one Gaussian node"):
            apply_radiance_scale_to_sh_nodes([], 2.0)


# =============================================================================
# Private helper: _compute_dc_offset_compensation
# =============================================================================


class ComputeDcOffsetCompensationTest(unittest.TestCase):
    def test_unity_scale_yields_zero_offset(self) -> None:
        self.assertAlmostEqual(_compute_dc_offset_compensation(1.0), 0.0, places=12)

    def test_doubling_scale_matches_closed_form(self) -> None:
        self.assertAlmostEqual(_compute_dc_offset_compensation(2.0), 0.5 / C0, places=12)

    def test_halving_scale_matches_closed_form(self) -> None:
        self.assertAlmostEqual(_compute_dc_offset_compensation(0.5), -0.25 / C0, places=12)

    def test_zero_scale_matches_closed_form(self) -> None:
        # Independent of validation; this helper has no guard, by design.
        self.assertAlmostEqual(_compute_dc_offset_compensation(0.0), -0.5 / C0, places=12)


# =============================================================================
# Private helper: _validate_radiance_scaling_inputs
# =============================================================================


class ValidateRadianceScalingInputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _MockSHModel()

    def test_accepts_valid_inputs(self) -> None:
        _validate_radiance_scaling_inputs(self.model, 1.0)
        _validate_radiance_scaling_inputs(self.model, 0.001)
        _validate_radiance_scaling_inputs(self.model, 1000.0)

    def test_accepts_int_scale(self) -> None:
        _validate_radiance_scaling_inputs(self.model, 2)

    def test_rejects_non_finite_scale(self) -> None:
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(scale=bad):
                with self.assertRaises(ValueError):
                    _validate_radiance_scaling_inputs(self.model, bad)

    def test_rejects_non_positive_scale(self) -> None:
        for bad in (0.0, -1e-9, -1.0):
            with self.subTest(scale=bad):
                with self.assertRaises(ValueError):
                    _validate_radiance_scaling_inputs(self.model, bad)

    def test_validates_scale_before_model_attributes(self) -> None:
        # An invalid scale on a model missing attributes must surface the
        # scalar error first; this keeps error messages stable and
        # predictable for callers.
        empty_model = object()
        with self.assertRaises(ValueError):
            _validate_radiance_scaling_inputs(empty_model, -1.0)


if __name__ == "__main__":
    unittest.main()
