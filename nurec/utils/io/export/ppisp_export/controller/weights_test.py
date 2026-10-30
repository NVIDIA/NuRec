# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`nurec.utils.io.export.ppisp_export.controller.weights`.

Tests are hermetic and CPU-only:

* :class:`_FakeController` mirrors ``ppisp._PPISPController``'s
  public surface (``cnn_encoder``, ``mlp_trunk``, ``exposure_head``,
  ``color_head``) on CPU. Each architectural axis can be twisted
  independently so the validator's failure branches are reachable
  without monkey-patching.
* :class:`_FakePPISPModule` exposes only ``controllers`` so we can
  exercise :func:`select_camera_controller` without any real PPISP /
  GPU dependency.
"""

from __future__ import annotations

import unittest

from typing import Callable, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn

from nurec.utils.io.export.ppisp_export.controller.weights import (
    EXPECTED_CONTROLLER_ARCHITECTURE,
    EXPECTED_CONTROLLER_WEIGHTS_LEN,
    ControllerArchitectureSpec,
    flatten_controller_weights,
    select_camera_controller,
    validate_controller_architecture,
    validate_controller_weights_finite,
)


# =============================================================================
# Test fixtures
# =============================================================================


class _FakeController(nn.Module):
    """CPU mirror of ``ppisp._PPISPController`` for hermetic tests.

    Each architectural axis is overridable via ``__init__`` kwargs
    so any single axis can be perturbed to drive the corresponding
    validator branch.
    """

    def __init__(
        self,
        input_downsampling: int = 3,
        cnn_in_channels: int = 3,
        cnn_layer_1_channels: int = 16,
        cnn_layer_2_channels: int = 32,
        cnn_feature_dim: int = 64,
        pool_grid_size: Tuple[int, int] = (5, 5),
        mlp_hidden_dim: int = 128,
        num_mlp_trunk_layers: int = 3,
        color_params_per_frame: int = 8,
        conv1_kernel_size: int = 1,
        omit_avgpool: bool = False,
    ) -> None:
        super().__init__()

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(cnn_in_channels, cnn_layer_1_channels, kernel_size=conv1_kernel_size),
            nn.MaxPool2d(kernel_size=input_downsampling, stride=input_downsampling),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_layer_1_channels, cnn_layer_2_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_layer_2_channels, cnn_feature_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(pool_grid_size) if not omit_avgpool else nn.Identity(),
            nn.Flatten(),
        )

        pool_h, pool_w = pool_grid_size
        trunk_input_dim = cnn_feature_dim * pool_h * pool_w + 1

        trunk_layers: List[nn.Module] = [
            nn.Linear(trunk_input_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_mlp_trunk_layers - 1):
            trunk_layers.extend([nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.ReLU(inplace=True)])
        self.mlp_trunk = nn.Sequential(*trunk_layers)

        self.exposure_head = nn.Linear(mlp_hidden_dim, 1)
        self.color_head = nn.Linear(mlp_hidden_dim, color_params_per_frame)


class _FakePPISPModule:
    """Bare-bones PPISP mock exposing only ``controllers``."""

    def __init__(self, controllers: Optional[List[nn.Module]]) -> None:
        if controllers is None:
            self.controllers = nn.ModuleList()
        else:
            self.controllers = nn.ModuleList(controllers)


def _cnn_param(controller: _FakeController, index: int, name: str) -> torch.Tensor:
    """Fetch a cnn_encoder parameter. Attribute access on nn.Module is typed as a union, so the
    concrete Sequential entry and its tensor have to be named for the type checker."""
    layer = cast(nn.Sequential, controller.cnn_encoder)[index]
    return cast(torch.Tensor, getattr(layer, name))


def _make_default_fake_controller() -> _FakeController:
    """Return a controller that satisfies the default architecture spec."""
    return _FakeController()


def _set_layer_constants(controller: _FakeController, constants: List[float]) -> None:
    """Fill controller layers with distinguishable per-layer constants.

    Layer order (matches the documented flatten layout):

    0: conv1.weight, 1: conv1.bias,
    2: conv2.weight, 3: conv2.bias,
    4: conv3.weight, 5: conv3.bias,
    6: trunk[0].weight, 7: trunk[0].bias,
    8: trunk[1].weight, 9: trunk[1].bias,
    10: trunk[2].weight, 11: trunk[2].bias,
    12: exposure_head.weight, 13: exposure_head.bias,
    14: color_head.weight, 15: color_head.bias.
    """
    if len(constants) != 16:
        raise ValueError(f"expected 16 layer constants, got {len(constants)}")

    mlp_trunk = controller.mlp_trunk
    trunk_linear = [m for m in mlp_trunk.modules() if isinstance(m, nn.Linear)]

    pairs: List[Tuple[torch.Tensor, float]] = [
        (_cnn_param(controller, 0, "weight"), constants[0]),
        (_cnn_param(controller, 0, "bias"), constants[1]),
        (_cnn_param(controller, 3, "weight"), constants[2]),
        (_cnn_param(controller, 3, "bias"), constants[3]),
        (_cnn_param(controller, 5, "weight"), constants[4]),
        (_cnn_param(controller, 5, "bias"), constants[5]),
        (trunk_linear[0].weight, constants[6]),
        (trunk_linear[0].bias, constants[7]),
        (trunk_linear[1].weight, constants[8]),
        (trunk_linear[1].bias, constants[9]),
        (trunk_linear[2].weight, constants[10]),
        (trunk_linear[2].bias, constants[11]),
        (controller.exposure_head.weight, constants[12]),
        (controller.exposure_head.bias, constants[13]),
        (controller.color_head.weight, constants[14]),
        (controller.color_head.bias, constants[15]),
    ]
    with torch.no_grad():
        for tensor, value in pairs:
            tensor.fill_(value)


# =============================================================================
# Architecture spec
# =============================================================================


class ControllerArchitectureSpecTest(unittest.TestCase):
    def test_expected_weights_len_matches_lavoisier_constant(self) -> None:
        # 241,961 is the locked SPG-shader contract value; mirrors
        # the lavoisier reference implementation. Any change to the
        # default architecture must update this constant in lockstep
        # with the SPG shader's TOTAL_WEIGHTS.
        self.assertEqual(EXPECTED_CONTROLLER_ARCHITECTURE.expected_weights_len, 241961)
        self.assertEqual(EXPECTED_CONTROLLER_WEIGHTS_LEN, 241961)

    def test_trunk_input_dim_includes_prior_exposure(self) -> None:
        spec = EXPECTED_CONTROLLER_ARCHITECTURE
        self.assertEqual(spec.trunk_input_dim, 64 * 5 * 5 + 1)

    def test_custom_spec_recomputes_expected_length(self) -> None:
        spec = ControllerArchitectureSpec(cnn_feature_dim=32)
        # 32 * 5 * 5 + 1 = 801; total length should drop accordingly.
        self.assertEqual(spec.trunk_input_dim, 801)
        self.assertNotEqual(spec.expected_weights_len, 241961)


# =============================================================================
# validate_controller_architecture
# =============================================================================


class ValidateControllerArchitectureTest(unittest.TestCase):
    def test_default_controller_validates(self) -> None:
        validate_controller_architecture(_make_default_fake_controller())

    def test_missing_cnn_encoder_raises_type_error(self) -> None:
        controller = _make_default_fake_controller()
        del controller.cnn_encoder
        with self.assertRaisesRegex(TypeError, "cnn_encoder"):
            validate_controller_architecture(controller)

    def test_cnn_encoder_wrong_type_raises_type_error(self) -> None:
        controller = _make_default_fake_controller()
        controller.cnn_encoder = nn.Linear(8, 8)  # type: ignore[assignment]
        with self.assertRaisesRegex(TypeError, "must be a Sequential|cnn_encoder"):
            validate_controller_architecture(controller)

    def test_missing_mlp_trunk_raises_type_error(self) -> None:
        controller = _make_default_fake_controller()
        del controller.mlp_trunk
        with self.assertRaisesRegex(TypeError, "mlp_trunk"):
            validate_controller_architecture(controller)

    def test_missing_exposure_head_raises_type_error(self) -> None:
        controller = _make_default_fake_controller()
        del controller.exposure_head
        with self.assertRaisesRegex(TypeError, "exposure_head"):
            validate_controller_architecture(controller)

    def test_missing_color_head_raises_type_error(self) -> None:
        controller = _make_default_fake_controller()
        del controller.color_head
        with self.assertRaisesRegex(TypeError, "color_head"):
            validate_controller_architecture(controller)

    def test_architecture_drift_cases_raise(self) -> None:
        cases: List[Tuple[str, Callable[[], _FakeController], str]] = [
            ("conv1 wrong input channels", lambda: _FakeController(cnn_in_channels=4), r"cnn_encoder\[0\]"),
            ("conv1 wrong output channels", lambda: _FakeController(cnn_layer_1_channels=8), r"cnn_encoder\[0\]"),
            ("conv1 non-unit kernel", lambda: _FakeController(conv1_kernel_size=3), "kernel_size must be"),
            ("conv2 wrong channels", lambda: _FakeController(cnn_layer_2_channels=64), r"cnn_encoder\[3\]"),
            ("conv3 wrong channels", lambda: _FakeController(cnn_feature_dim=128), r"cnn_encoder\[5\]"),
            ("maxpool wrong kernel", lambda: _FakeController(input_downsampling=2), "MaxPool2d"),
            ("avgpool wrong grid", lambda: _FakeController(pool_grid_size=(4, 4)), "AdaptiveAvgPool2d|in_features"),
            ("avgpool wrong kind", lambda: _FakeController(omit_avgpool=True), "AdaptiveAvgPool2d"),
            (
                "too few trunk layers",
                lambda: _FakeController(num_mlp_trunk_layers=2),
                "mlp_trunk must contain exactly",
            ),
            (
                "too many trunk layers",
                lambda: _FakeController(num_mlp_trunk_layers=4),
                "mlp_trunk must contain exactly",
            ),
            (
                "wrong trunk hidden dim",
                lambda: _FakeController(mlp_hidden_dim=64),
                "mlp_trunk|in_features|out_features",
            ),
            (
                "wrong color head width",
                lambda: _FakeController(color_params_per_frame=4),
                "color_head.out_features",
            ),
        ]
        for name, make_controller, expected_error in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, expected_error):
                validate_controller_architecture(make_controller())

    def test_custom_spec_validates_matching_controller(self) -> None:
        spec = ControllerArchitectureSpec(cnn_feature_dim=32)
        controller = _FakeController(cnn_feature_dim=32)
        validate_controller_architecture(controller, spec)


# =============================================================================
# validate_controller_weights_finite
# =============================================================================


class ValidateControllerWeightsFiniteTest(unittest.TestCase):
    def test_default_controller_passes(self) -> None:
        controller = _make_default_fake_controller()
        validate_controller_weights_finite(controller)

    def test_nan_in_conv_weight_raises(self) -> None:
        controller = _make_default_fake_controller()
        with torch.no_grad():
            _cnn_param(controller, 0, "weight")[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            validate_controller_weights_finite(controller)

    def test_inf_in_trunk_bias_raises(self) -> None:
        controller = _make_default_fake_controller()
        trunk_linear = [m for m in controller.mlp_trunk.modules() if isinstance(m, nn.Linear)]
        with torch.no_grad():
            trunk_linear[1].bias[0] = float("inf")
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            validate_controller_weights_finite(controller)

    def test_negative_inf_in_color_head_raises(self) -> None:
        controller = _make_default_fake_controller()
        with torch.no_grad():
            controller.color_head.weight[0, 0] = float("-inf")
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            validate_controller_weights_finite(controller)


# =============================================================================
# flatten_controller_weights
# =============================================================================


class FlattenControllerWeightsTest(unittest.TestCase):
    def test_default_controller_returns_expected_length_float32(self) -> None:
        controller = _make_default_fake_controller()
        flat = flatten_controller_weights(controller)
        self.assertEqual(flat.dtype, np.float32)
        self.assertEqual(flat.ndim, 1)
        self.assertEqual(flat.size, EXPECTED_CONTROLLER_WEIGHTS_LEN)

    def test_layout_matches_documented_order(self) -> None:
        # Fill each layer with a distinguishable constant; assert
        # that the flat buffer's slices read back those constants in
        # the documented order. This locks the byte layout against
        # accidental reorderings.
        controller = _make_default_fake_controller()
        constants = [float(i + 1) for i in range(16)]
        _set_layer_constants(controller, constants)
        flat = flatten_controller_weights(controller)

        spec = EXPECTED_CONTROLLER_ARCHITECTURE
        sizes = [
            spec.cnn_layer_1_channels * spec.cnn_in_channels,  # conv1.weight
            spec.cnn_layer_1_channels,  # conv1.bias
            spec.cnn_layer_2_channels * spec.cnn_layer_1_channels,
            spec.cnn_layer_2_channels,
            spec.cnn_feature_dim * spec.cnn_layer_2_channels,
            spec.cnn_feature_dim,
            spec.mlp_hidden_dim * spec.trunk_input_dim,
            spec.mlp_hidden_dim,
            spec.mlp_hidden_dim * spec.mlp_hidden_dim,
            spec.mlp_hidden_dim,
            spec.mlp_hidden_dim * spec.mlp_hidden_dim,
            spec.mlp_hidden_dim,
            1 * spec.mlp_hidden_dim,
            1,
            spec.color_params_per_frame * spec.mlp_hidden_dim,
            spec.color_params_per_frame,
        ]
        self.assertEqual(sum(sizes), EXPECTED_CONTROLLER_WEIGHTS_LEN)

        offset = 0
        for layer_index, (size, value) in enumerate(zip(sizes, constants)):
            slice_ = flat[offset : offset + size]
            self.assertTrue(
                np.allclose(slice_, np.full(size, value, dtype=np.float32)),
                msg=(
                    f"layer {layer_index} (size {size}) at offset {offset} "
                    f"should be filled with {value}; got "
                    f"min={slice_.min()} max={slice_.max()}"
                ),
            )
            offset += size
        self.assertEqual(offset, flat.size)

    def test_conv_weight_reshape_is_out_then_in(self) -> None:
        # Conv weights are emitted as [out_channels, in_channels]
        # row-major. We verify by setting weight[0, :, 0, 0] = 7 and
        # weight[1, :, 0, 0] = 9 on conv1, then asserting that the
        # first 3 floats are 7s (out=0 row) and the next 3 are 9s
        # (out=1 row).
        controller = _make_default_fake_controller()
        with torch.no_grad():
            _cnn_param(controller, 0, "weight").zero_()
            _cnn_param(controller, 0, "weight")[0, :, 0, 0] = 7.0
            _cnn_param(controller, 0, "weight")[1, :, 0, 0] = 9.0
            _cnn_param(controller, 0, "bias").zero_()
        flat = flatten_controller_weights(controller)
        np.testing.assert_array_equal(flat[0:3], np.array([7.0, 7.0, 7.0], dtype=np.float32))
        np.testing.assert_array_equal(flat[3:6], np.array([9.0, 9.0, 9.0], dtype=np.float32))

    def test_invalid_architecture_propagates(self) -> None:
        controller = _FakeController(cnn_feature_dim=128)
        with self.assertRaisesRegex(ValueError, r"cnn_encoder\[5\]"):
            flatten_controller_weights(controller)

    def test_non_finite_propagates(self) -> None:
        controller = _make_default_fake_controller()
        with torch.no_grad():
            _cnn_param(controller, 3, "bias")[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            flatten_controller_weights(controller)

    def test_idempotent_repeated_calls(self) -> None:
        controller = _make_default_fake_controller()
        flat_a = flatten_controller_weights(controller)
        flat_b = flatten_controller_weights(controller)
        np.testing.assert_array_equal(flat_a, flat_b)

    def test_does_not_mutate_controller_parameters(self) -> None:
        controller = _make_default_fake_controller()
        before = {name: p.detach().clone() for name, p in controller.named_parameters()}
        flatten_controller_weights(controller)
        for name, p in controller.named_parameters():
            self.assertTrue(
                torch.equal(before[name], p),
                msg=f"parameter {name!r} was mutated by flatten_controller_weights",
            )


# =============================================================================
# select_camera_controller
# =============================================================================


class SelectCameraControllerTest(unittest.TestCase):
    def test_returns_correct_controller(self) -> None:
        c0 = _make_default_fake_controller()
        c1 = _make_default_fake_controller()
        ppisp = _FakePPISPModule(controllers=[c0, c1])
        self.assertIs(select_camera_controller(ppisp, 0), c0)
        self.assertIs(select_camera_controller(ppisp, 1), c1)

    def test_camera_id_must_be_int(self) -> None:
        ppisp = _FakePPISPModule(controllers=[_make_default_fake_controller()])
        with self.assertRaisesRegex(TypeError, "camera_id must be int"):
            select_camera_controller(ppisp, 0.0)  # type: ignore[arg-type]

    def test_camera_id_bool_rejected(self) -> None:
        ppisp = _FakePPISPModule(controllers=[_make_default_fake_controller()])
        with self.assertRaisesRegex(TypeError, "camera_id must be int"):
            select_camera_controller(ppisp, True)  # type: ignore[arg-type]

    def test_missing_controllers_attribute_raises(self) -> None:
        class _NoControllers:
            pass

        with self.assertRaisesRegex(TypeError, "controllers"):
            select_camera_controller(_NoControllers(), 0)  # type: ignore[arg-type]

    def test_empty_controllers_raises(self) -> None:
        ppisp = _FakePPISPModule(controllers=None)
        with self.assertRaisesRegex(ValueError, "use_controller=False"):
            select_camera_controller(ppisp, 0)

    def test_negative_camera_id_raises(self) -> None:
        ppisp = _FakePPISPModule(controllers=[_make_default_fake_controller()])
        with self.assertRaisesRegex(ValueError, r"camera_id must be in"):
            select_camera_controller(ppisp, -1)

    def test_out_of_range_camera_id_raises(self) -> None:
        ppisp = _FakePPISPModule(controllers=[_make_default_fake_controller(), _make_default_fake_controller()])
        with self.assertRaisesRegex(ValueError, r"camera_id must be in \[0, 1\]"):
            select_camera_controller(ppisp, 2)


if __name__ == "__main__":
    unittest.main()
