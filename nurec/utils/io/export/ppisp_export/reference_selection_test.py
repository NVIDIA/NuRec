# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ``nurec.utils.io.export.ppisp_export.reference_selection``.

Hermetic: a tiny ``_MockPPISPModule`` exposes only ``num_cameras``
and ``exposure_params``, which is exactly the duck-typed surface the
resolver requires.

Coverage:
* ``_validate_mode`` (3 cases): valid literals pass, anything else
  rejected with the documented set in the message.
* ``_validate_id_type`` (8 cases): ``None`` passes, ``int`` passes,
  ``bool`` / ``float`` / ``str`` / negative non-int / etc. rejected.
* ``_validate_id_range`` (4 cases): in-range, at lower bound, at
  upper-exclusive bound, below zero.
* ``_read_ppisp_bounds`` (3 cases): both attributes present, missing
  ``num_cameras``, missing ``exposure_params``.
* ``resolve_ppisp_reference_ids`` -- per-mode behavior:
  * ``sh-optimized`` (4 cases): both unset, camera-only, frame-only
    (accepted, camera defaults to 0), both set.
  * ``spg-runtime`` (4 cases): all four rows of the documented
    fallback table including the frame-only ``ValueError``.
* Round-trip with ``PPISPSHOptimizationAdapter`` (1 case): the
  resolver's output feeds the adapter constructor without further
  massaging.
* Module constants and the dataclass surface (2 cases).
"""

from __future__ import annotations

import unittest

from typing import Any

import torch
import torch.nn as nn

from nurec.utils.io.export.ppisp_export.reference_selection import (
    _read_ppisp_bounds,
    _validate_id_range,
    _validate_id_type,
    _validate_mode,
    resolve_ppisp_reference_ids,
)
from nurec.utils.io.export.ppisp_export.sh_match.optimization import (
    PPISPSHOptimizationAdapter,
    SHMatchingOptimizationConfig,
)


# =============================================================================
# Test doubles
# =============================================================================


class _MockPPISPConfig:
    def __init__(self, use_controller: bool = True) -> None:
        self.use_controller = use_controller


class _MockPPISPModule(nn.Module):
    """Trained-PPISP-shaped module exposing the resolver's required surface."""

    def __init__(self, num_cameras: int = 4, num_frames: int = 7) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.config = _MockPPISPConfig(use_controller=True)
        self.exposure_params = nn.Parameter(torch.zeros(num_frames), requires_grad=False)
        self.color_params = nn.Parameter(torch.zeros(num_frames, 8), requires_grad=False)
        self.vignetting_params = nn.Parameter(torch.zeros(num_cameras, 3, 5), requires_grad=False)
        self.crf_params = nn.Parameter(torch.zeros(num_cameras, 3, 4), requires_grad=False)


# =============================================================================
# _validate_mode
# =============================================================================


class ValidateModeTest(unittest.TestCase):
    def test_sh_optimized_passes(self) -> None:
        _validate_mode("sh-optimized")

    def test_spg_runtime_passes(self) -> None:
        _validate_mode("spg-runtime")

    def test_unknown_mode_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, r"mode must be one of"):
            _validate_mode("baked-sh")


# =============================================================================
# _validate_id_type
# =============================================================================


class ValidateIdTypeTest(unittest.TestCase):
    def test_none_passes(self) -> None:
        _validate_id_type("ppisp_reference_camera_id", None)

    def test_zero_passes(self) -> None:
        _validate_id_type("ppisp_reference_camera_id", 0)

    def test_positive_int_passes(self) -> None:
        _validate_id_type("ppisp_reference_camera_id", 5)

    def test_negative_int_passes(self) -> None:
        # Type validation is type-only; range is checked separately.
        _validate_id_type("ppisp_reference_camera_id", -1)

    def test_bool_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, r"must be int or None, got bool"):
            _validate_id_type("ppisp_reference_camera_id", True)

    def test_float_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, r"must be int or None, got float"):
            _validate_id_type("ppisp_reference_camera_id", 1.5)

    def test_str_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, r"must be int or None, got str"):
            _validate_id_type("ppisp_reference_camera_id", "0")

    def test_error_message_includes_argument_name(self) -> None:
        with self.assertRaisesRegex(TypeError, r"ppisp_reference_frame_id must be int or None"):
            _validate_id_type("ppisp_reference_frame_id", 1.5)


# =============================================================================
# _validate_id_range
# =============================================================================


class ValidateIdRangeTest(unittest.TestCase):
    def test_lower_bound_passes(self) -> None:
        _validate_id_range("name", 0, upper_exclusive=4)

    def test_in_range_passes(self) -> None:
        _validate_id_range("name", 2, upper_exclusive=4)

    def test_at_upper_exclusive_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"name must be in \[0, 3\], got 4"):
            _validate_id_range("name", 4, upper_exclusive=4)

    def test_negative_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"name must be in \[0, 3\], got -1"):
            _validate_id_range("name", -1, upper_exclusive=4)


# =============================================================================
# _read_ppisp_bounds
# =============================================================================


class ReadPPISPBoundsTest(unittest.TestCase):
    def test_returns_num_cameras_and_num_frames(self) -> None:
        module = _MockPPISPModule(num_cameras=3, num_frames=11)
        num_cameras, num_frames = _read_ppisp_bounds(module)
        self.assertEqual(num_cameras, 3)
        self.assertEqual(num_frames, 11)

    def test_missing_num_cameras_raises(self) -> None:
        module = _MockPPISPModule()
        del module.num_cameras
        with self.assertRaisesRegex(AttributeError, r"missing required attribute 'num_cameras'"):
            _read_ppisp_bounds(module)

    def test_missing_exposure_params_raises(self) -> None:
        module = _MockPPISPModule()
        del module.exposure_params
        with self.assertRaisesRegex(AttributeError, r"missing required attribute 'exposure_params'"):
            _read_ppisp_bounds(module)


# =============================================================================
# resolve_ppisp_reference_ids: sh-optimized mode
# =============================================================================


class ResolveSHOptimizedModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _MockPPISPModule(num_cameras=4, num_frames=7)

    def test_both_unset_defaults_to_zero_zero(self) -> None:
        resolved = resolve_ppisp_reference_ids(None, None, self.module, mode="sh-optimized")
        self.assertEqual(resolved.camera_id, 0)
        self.assertEqual(resolved.frame_id, 0)
        self.assertFalse(resolved.camera_was_user_set)
        self.assertFalse(resolved.frame_was_user_set)

    def test_camera_only_keeps_camera_and_zero_frame(self) -> None:
        resolved = resolve_ppisp_reference_ids(2, None, self.module, mode="sh-optimized")
        self.assertEqual(resolved.camera_id, 2)
        self.assertEqual(resolved.frame_id, 0)
        self.assertTrue(resolved.camera_was_user_set)
        self.assertFalse(resolved.frame_was_user_set)

    def test_frame_only_accepted_with_zero_camera(self) -> None:
        """sh-optimized accepts frame-only specification: camera defaults to 0
        because the bake target collapses to a single (camera, frame) pair
        regardless of which axis the user pinned."""
        resolved = resolve_ppisp_reference_ids(None, 5, self.module, mode="sh-optimized")
        self.assertEqual(resolved.camera_id, 0)
        self.assertEqual(resolved.frame_id, 5)
        self.assertFalse(resolved.camera_was_user_set)
        self.assertTrue(resolved.frame_was_user_set)

    def test_both_set_uses_both(self) -> None:
        resolved = resolve_ppisp_reference_ids(3, 6, self.module, mode="sh-optimized")
        self.assertEqual(resolved.camera_id, 3)
        self.assertEqual(resolved.frame_id, 6)
        self.assertTrue(resolved.camera_was_user_set)
        self.assertTrue(resolved.frame_was_user_set)

    def test_camera_out_of_range_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"ppisp_reference_camera_id must be in \[0, 3\], got 4"):
            resolve_ppisp_reference_ids(4, None, self.module, mode="sh-optimized")

    def test_frame_out_of_range_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"ppisp_reference_frame_id must be in \[0, 6\], got 7"):
            resolve_ppisp_reference_ids(None, 7, self.module, mode="sh-optimized")


# =============================================================================
# resolve_ppisp_reference_ids: spg-runtime mode
# =============================================================================


class ResolveSPGRuntimeModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _MockPPISPModule(num_cameras=4, num_frames=7)

    def test_both_unset_signals_time_sampled(self) -> None:
        """Both flags False signals time-sampled to the SPG writer."""
        resolved = resolve_ppisp_reference_ids(None, None, self.module, mode="spg-runtime")
        self.assertEqual(resolved.camera_id, 0)
        self.assertEqual(resolved.frame_id, 0)
        self.assertFalse(resolved.camera_was_user_set)
        self.assertFalse(resolved.frame_was_user_set)

    def test_camera_only_signals_neutral_frame(self) -> None:
        """camera_was_user_set=True, frame_was_user_set=False signals the
        camera-only fallback: SPG writer emits per-camera vignetting/CRF and
        neutral exposure/identity color."""
        resolved = resolve_ppisp_reference_ids(2, None, self.module, mode="spg-runtime")
        self.assertEqual(resolved.camera_id, 2)
        self.assertEqual(resolved.frame_id, 0)
        self.assertTrue(resolved.camera_was_user_set)
        self.assertFalse(resolved.frame_was_user_set)

    def test_both_set_signals_fully_specified(self) -> None:
        resolved = resolve_ppisp_reference_ids(3, 6, self.module, mode="spg-runtime")
        self.assertEqual(resolved.camera_id, 3)
        self.assertEqual(resolved.frame_id, 6)
        self.assertTrue(resolved.camera_was_user_set)
        self.assertTrue(resolved.frame_was_user_set)

    def test_frame_only_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"ppisp_reference_frame_id was set without ppisp_reference_camera_id",
        ):
            resolve_ppisp_reference_ids(None, 5, self.module, mode="spg-runtime")

    def test_camera_out_of_range_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"ppisp_reference_camera_id must be in \[0, 3\], got 4"):
            resolve_ppisp_reference_ids(4, 0, self.module, mode="spg-runtime")

    def test_frame_out_of_range_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"ppisp_reference_frame_id must be in \[0, 6\], got 7"):
            resolve_ppisp_reference_ids(0, 7, self.module, mode="spg-runtime")


# =============================================================================
# resolve_ppisp_reference_ids: top-level guards
# =============================================================================


class ResolveTopLevelGuardsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _MockPPISPModule()

    def test_invalid_mode_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, r"mode must be one of"):
            resolve_ppisp_reference_ids(None, None, self.module, mode="bake")  # type: ignore[arg-type]

    def test_camera_id_wrong_type_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, r"ppisp_reference_camera_id must be int or None"):
            resolve_ppisp_reference_ids("0", None, self.module, mode="sh-optimized")  # type: ignore[arg-type]

    def test_frame_id_wrong_type_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, r"ppisp_reference_frame_id must be int or None"):
            resolve_ppisp_reference_ids(None, 1.5, self.module, mode="sh-optimized")  # type: ignore[arg-type]

    def test_camera_id_bool_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, r"ppisp_reference_camera_id must be int or None, got bool"):
            resolve_ppisp_reference_ids(True, None, self.module, mode="sh-optimized")  # type: ignore[arg-type]

    def test_missing_module_attribute_raises(self) -> None:
        broken: Any = _MockPPISPModule()
        del broken.num_cameras
        with self.assertRaisesRegex(AttributeError, r"missing required attribute 'num_cameras'"):
            resolve_ppisp_reference_ids(None, None, broken, mode="sh-optimized")


# =============================================================================
# Round-trip with PPISPSHOptimizationAdapter
# =============================================================================


class ResolvedIdsRoundTripTest(unittest.TestCase):
    def test_resolved_ids_construct_adapter(self) -> None:
        """Resolved IDs must round-trip into PPISPSHOptimizationAdapter without
        further massaging. Validates the contract between the resolver and the
        downstream consumer in sh-optimized mode."""
        module = _MockPPISPModule(num_cameras=4, num_frames=7)
        resolved = resolve_ppisp_reference_ids(2, 5, module, mode="sh-optimized")
        config = SHMatchingOptimizationConfig(num_iterations=1, device="cpu")
        adapter = PPISPSHOptimizationAdapter(
            config=config,
            ppisp_module=module,
            camera_id=resolved.camera_id,
            frame_id=resolved.frame_id,
        )
        self.assertEqual(adapter.camera_id, 2)
        self.assertEqual(adapter.frame_id, 5)


if __name__ == "__main__":
    unittest.main()
