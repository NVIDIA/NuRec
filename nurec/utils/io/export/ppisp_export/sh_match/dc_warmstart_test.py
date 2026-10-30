# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ``nurec.utils.io.export.ppisp_export.sh_match.dc_warmstart``.

The tests are hermetic: they substitute a recorder for the
``ppisp.ppisp_apply`` symbol so behavior depends only on this module's
wiring (vignetting zeroed, single-frame indexing, identity-CRF mapping,
device/dtype handling), not on the ppisp library's pipeline.

Coverage:
* ``warmstart_sh_dc_from_ppisp`` end-to-end: DC band updated, specular
  preserved, returned ``(exposure, color)`` correct, identity PPISP is
  a no-op modulo float round-trip.
* ``_validate_warmstart_inputs`` rejects every documented bad input.
* ``_get_fixed_frame_params`` extracts the right slice and detaches.
* ``_apply_ppisp_to_dc_colors`` calls ``ppisp_apply`` with vignetting
  zeroed, single-frame parameter shapes, and pixel coords pinned to
  the optical center.
"""

from __future__ import annotations

import unittest

from typing import Any, Dict, List
from unittest import mock

import torch

from nurec.models.gaussians.utils import RGB2SH, SH2RGB
from nurec.utils.io.export.ppisp_export.sh_match import dc_warmstart as warmstart_module
from nurec.utils.io.export.ppisp_export.sh_match.dc_warmstart import (
    _apply_ppisp_to_dc_colors,
    _get_fixed_frame_params,
    _validate_warmstart_inputs,
    warmstart_sh_dc_from_ppisp,
)


# =============================================================================
# Test doubles
# =============================================================================


class _MockPPISPModule:
    """Trained-PPISP-shaped object exposing only the attributes warmstart reads."""

    def __init__(
        self,
        num_cameras: int = 2,
        num_frames: int = 3,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.num_cameras = num_cameras
        self.exposure_params = torch.randn(num_frames, generator=generator, dtype=dtype, device=device)
        self.color_params = torch.randn(num_frames, 8, generator=generator, dtype=dtype, device=device)
        self.vignetting_params = torch.randn(num_cameras, 3, 5, generator=generator, dtype=dtype, device=device)
        self.crf_params = torch.randn(num_cameras, 3, 4, generator=generator, dtype=dtype, device=device)


class _MockSHModel:
    """SH-coefficient bearer with arbitrary albedo/specular tensors."""

    def __init__(
        self,
        num_gaussians: int = 8,
        num_specular_dims: int = 9,
        seed: int = 1,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.features_albedo = torch.randn(num_gaussians, 3, generator=generator, dtype=torch.float32)
        self.features_specular = torch.randn(num_gaussians, num_specular_dims, generator=generator, dtype=torch.float32)


class _PpispApplyRecorder:
    """Replacement for ``ppisp.ppisp_apply`` that records its kwargs.

    The default ``transform`` returns its ``rgb_in`` unchanged so the
    warm-start round-trip is the identity in SH space, which makes
    DC-update tests trivially predictable. Tests that need to exercise
    a non-identity bake pass a custom ``transform``.
    """

    def __init__(
        self,
        transform: Any = None,
    ) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._transform = transform if transform is not None else (lambda rgb_in, **_: rgb_in)

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        # Snapshot tensor inputs so later in-place mutation of the source
        # tensors cannot retroactively change recorded values.
        snapshot = {
            key: (value.detach().clone() if isinstance(value, torch.Tensor) else value) for key, value in kwargs.items()
        }
        self.calls.append(snapshot)
        return self._transform(**kwargs)


# =============================================================================
# warmstart_sh_dc_from_ppisp end-to-end
# =============================================================================


class WarmstartShDcFromPpispTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _MockSHModel(num_gaussians=8)
        self.ppisp_module = _MockPPISPModule(num_cameras=2, num_frames=3)
        self.original_albedo = self.model.features_albedo.clone()
        self.original_specular = self.model.features_specular.clone()

    def test_specular_preserved(self) -> None:
        recorder = _PpispApplyRecorder()
        with mock.patch.object(warmstart_module, "ppisp_apply", recorder):
            warmstart_sh_dc_from_ppisp(self.model, self.ppisp_module, camera_id=0, frame_id=0)
        self.assertTrue(torch.equal(self.model.features_specular, self.original_specular))

    def test_identity_ppisp_round_trips_albedo(self) -> None:
        # ppisp_apply returns rgb_in unchanged -> SH2RGB then RGB2SH is the
        # identity up to floating-point rounding.
        recorder = _PpispApplyRecorder()
        with mock.patch.object(warmstart_module, "ppisp_apply", recorder):
            warmstart_sh_dc_from_ppisp(self.model, self.ppisp_module, camera_id=0, frame_id=0)
        torch.testing.assert_close(self.model.features_albedo, self.original_albedo, rtol=1e-6, atol=1e-6)

    def test_albedo_changes_when_ppisp_is_not_identity(self) -> None:
        recorder = _PpispApplyRecorder(transform=lambda rgb_in, **_: rgb_in * 2.0 + 0.1)
        with mock.patch.object(warmstart_module, "ppisp_apply", recorder):
            warmstart_sh_dc_from_ppisp(self.model, self.ppisp_module, camera_id=0, frame_id=0)
        self.assertFalse(torch.equal(self.model.features_albedo, self.original_albedo))

    def test_returned_exposure_and_color_match_module(self) -> None:
        recorder = _PpispApplyRecorder()
        with mock.patch.object(warmstart_module, "ppisp_apply", recorder):
            exposure, color = warmstart_sh_dc_from_ppisp(self.model, self.ppisp_module, camera_id=1, frame_id=2)
        self.assertAlmostEqual(exposure, float(self.ppisp_module.exposure_params[2].item()), places=6)
        torch.testing.assert_close(color, self.ppisp_module.color_params[2].detach(), rtol=0.0, atol=0.0)

    def test_double_warmstart_with_identity_is_idempotent(self) -> None:
        recorder = _PpispApplyRecorder()
        with mock.patch.object(warmstart_module, "ppisp_apply", recorder):
            warmstart_sh_dc_from_ppisp(self.model, self.ppisp_module, camera_id=0, frame_id=0)
            mid_albedo = self.model.features_albedo.clone()
            warmstart_sh_dc_from_ppisp(self.model, self.ppisp_module, camera_id=0, frame_id=0)
        torch.testing.assert_close(self.model.features_albedo, mid_albedo, rtol=1e-6, atol=1e-6)

    def test_invocation_passes_zero_vignetting_and_unit_resolution(self) -> None:
        recorder = _PpispApplyRecorder()
        with mock.patch.object(warmstart_module, "ppisp_apply", recorder):
            warmstart_sh_dc_from_ppisp(self.model, self.ppisp_module, camera_id=1, frame_id=0)
        self.assertEqual(len(recorder.calls), 1)
        call = recorder.calls[0]
        self.assertTrue(torch.equal(call["vignetting_params"], torch.zeros_like(call["vignetting_params"])))
        self.assertEqual(call["resolution_w"], 1)
        self.assertEqual(call["resolution_h"], 1)
        self.assertEqual(call["camera_idx"], 1)
        self.assertEqual(call["frame_idx"], 0)
        self.assertEqual(call["exposure_params"].shape, (1,))
        self.assertEqual(call["color_params"].shape, (1, 8))
        self.assertTrue(torch.equal(call["pixel_coords"], torch.zeros_like(call["pixel_coords"])))


# =============================================================================
# _validate_warmstart_inputs
# =============================================================================


class ValidateWarmstartInputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _MockSHModel()
        self.ppisp_module = _MockPPISPModule(num_cameras=2, num_frames=3)

    def test_accepts_valid_inputs(self) -> None:
        _validate_warmstart_inputs(self.model, self.ppisp_module, 0, 0)
        _validate_warmstart_inputs(self.model, self.ppisp_module, 1, 2)

    def test_camera_id_negative_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "camera_id"):
            _validate_warmstart_inputs(self.model, self.ppisp_module, -1, 0)

    def test_camera_id_out_of_bounds_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "camera_id"):
            _validate_warmstart_inputs(self.model, self.ppisp_module, 2, 0)

    def test_frame_id_negative_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame_id"):
            _validate_warmstart_inputs(self.model, self.ppisp_module, 0, -1)

    def test_frame_id_out_of_bounds_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame_id"):
            _validate_warmstart_inputs(self.model, self.ppisp_module, 0, 3)

    def test_camera_id_non_int_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "camera_id must be int"):
            _validate_warmstart_inputs(self.model, self.ppisp_module, 0.5, 0)  # type: ignore[arg-type]

    def test_camera_id_bool_raises(self) -> None:
        # ``bool`` subclasses ``int`` in Python; the explicit guard keeps
        # callers from accidentally indexing with ``True`` / ``False``.
        with self.assertRaisesRegex(TypeError, "camera_id must be int"):
            _validate_warmstart_inputs(self.model, self.ppisp_module, True, 0)  # type: ignore[arg-type]

    def test_frame_id_non_int_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "frame_id must be int"):
            _validate_warmstart_inputs(self.model, self.ppisp_module, 0, 1.5)  # type: ignore[arg-type]

    def test_missing_features_albedo_raises(self) -> None:
        broken = _MockSHModel()
        del broken.features_albedo
        with self.assertRaisesRegex(AttributeError, "features_albedo"):
            _validate_warmstart_inputs(broken, self.ppisp_module, 0, 0)

    def test_missing_features_specular_raises(self) -> None:
        broken = _MockSHModel()
        del broken.features_specular
        with self.assertRaisesRegex(AttributeError, "features_specular"):
            _validate_warmstart_inputs(broken, self.ppisp_module, 0, 0)

    def test_features_albedo_wrong_dtype_raises(self) -> None:
        broken = _MockSHModel()
        broken.features_albedo = [[0.0, 0.0, 0.0]]  # type: ignore[assignment]
        with self.assertRaisesRegex(TypeError, "features_albedo"):
            _validate_warmstart_inputs(broken, self.ppisp_module, 0, 0)

    def test_features_albedo_wrong_shape_raises(self) -> None:
        broken = _MockSHModel()
        broken.features_albedo = torch.zeros(8, 4)
        with self.assertRaisesRegex(ValueError, r"\[N, 3\]"):
            _validate_warmstart_inputs(broken, self.ppisp_module, 0, 0)

    def test_missing_ppisp_attribute_raises(self) -> None:
        broken = _MockPPISPModule()
        del broken.exposure_params
        with self.assertRaisesRegex(AttributeError, "exposure_params"):
            _validate_warmstart_inputs(self.model, broken, 0, 0)


# =============================================================================
# _get_fixed_frame_params
# =============================================================================


class GetFixedFrameParamsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ppisp_module = _MockPPISPModule(num_cameras=1, num_frames=4, seed=42)

    def test_returns_exposure_as_python_float(self) -> None:
        exposure, _ = _get_fixed_frame_params(self.ppisp_module, frame_id=2)
        self.assertIsInstance(exposure, float)
        self.assertAlmostEqual(exposure, float(self.ppisp_module.exposure_params[2].item()), places=6)

    def test_returns_color_with_correct_shape_and_values(self) -> None:
        _, color = _get_fixed_frame_params(self.ppisp_module, frame_id=3)
        self.assertEqual(color.shape, (8,))
        torch.testing.assert_close(color, self.ppisp_module.color_params[3].detach(), rtol=0.0, atol=0.0)

    def test_returned_color_is_detached(self) -> None:
        ppisp_module = _MockPPISPModule(num_cameras=1, num_frames=2)
        ppisp_module.color_params = ppisp_module.color_params.requires_grad_(True)
        _, color = _get_fixed_frame_params(ppisp_module, frame_id=0)
        self.assertFalse(color.requires_grad)


# =============================================================================
# _apply_ppisp_to_dc_colors
# =============================================================================


class ApplyPpispToDcColorsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ppisp_module = _MockPPISPModule(num_cameras=2, num_frames=3, seed=7)
        self.dc_rgb_linear = torch.rand(5, 3, dtype=torch.float32)

    def test_zeros_vignetting_independent_of_module(self) -> None:
        # Module's vignetting is non-zero by construction; the call must
        # still pass an all-zero tensor of the right shape.
        recorder = _PpispApplyRecorder()
        with mock.patch.object(warmstart_module, "ppisp_apply", recorder):
            _apply_ppisp_to_dc_colors(
                dc_rgb_linear=self.dc_rgb_linear,
                ppisp_module=self.ppisp_module,
                camera_id=1,
                exposure=0.25,
                color=self.ppisp_module.color_params[0].detach(),
                device=torch.device("cpu"),
            )
        call = recorder.calls[0]
        self.assertEqual(call["vignetting_params"].shape, self.ppisp_module.vignetting_params.shape)
        self.assertTrue(torch.equal(call["vignetting_params"], torch.zeros_like(call["vignetting_params"])))

    def test_pixel_coords_at_optical_center(self) -> None:
        recorder = _PpispApplyRecorder()
        with mock.patch.object(warmstart_module, "ppisp_apply", recorder):
            _apply_ppisp_to_dc_colors(
                dc_rgb_linear=self.dc_rgb_linear,
                ppisp_module=self.ppisp_module,
                camera_id=0,
                exposure=0.0,
                color=torch.zeros(8),
                device=torch.device("cpu"),
            )
        call = recorder.calls[0]
        self.assertEqual(call["pixel_coords"].shape, (5, 2))
        self.assertTrue(torch.equal(call["pixel_coords"], torch.zeros_like(call["pixel_coords"])))

    def test_single_frame_param_shapes(self) -> None:
        recorder = _PpispApplyRecorder()
        with mock.patch.object(warmstart_module, "ppisp_apply", recorder):
            _apply_ppisp_to_dc_colors(
                dc_rgb_linear=self.dc_rgb_linear,
                ppisp_module=self.ppisp_module,
                camera_id=0,
                exposure=1.5,
                color=torch.arange(8, dtype=torch.float32),
                device=torch.device("cpu"),
            )
        call = recorder.calls[0]
        self.assertEqual(call["exposure_params"].shape, (1,))
        self.assertAlmostEqual(float(call["exposure_params"].item()), 1.5, places=6)
        self.assertEqual(call["color_params"].shape, (1, 8))
        torch.testing.assert_close(call["color_params"][0], torch.arange(8, dtype=torch.float32))
        self.assertEqual(call["frame_idx"], 0)

    def test_returns_ppisp_apply_output(self) -> None:
        out = torch.full_like(self.dc_rgb_linear, 0.42)
        recorder = _PpispApplyRecorder(transform=lambda rgb_in, **_: out)
        with mock.patch.object(warmstart_module, "ppisp_apply", recorder):
            result = _apply_ppisp_to_dc_colors(
                dc_rgb_linear=self.dc_rgb_linear,
                ppisp_module=self.ppisp_module,
                camera_id=0,
                exposure=0.0,
                color=torch.zeros(8),
                device=torch.device("cpu"),
            )
        torch.testing.assert_close(result, out)


# =============================================================================
# Round-trip / cross-check
# =============================================================================


class WarmstartCrossCheckTest(unittest.TestCase):
    def test_dc_rgb_after_warmstart_matches_ppisp_output(self) -> None:
        # Predict what features_albedo should become: SH2RGB -> ppisp_apply ->
        # RGB2SH. With ``transform = rgb*2 + 0.1``, the closed-form result
        # is RGB2SH(2 * SH2RGB(albedo) + 0.1).
        model = _MockSHModel(num_gaussians=4)
        original_albedo = model.features_albedo.clone()
        ppisp_module = _MockPPISPModule(num_cameras=1, num_frames=1)

        recorder = _PpispApplyRecorder(transform=lambda rgb_in, **_: rgb_in * 2.0 + 0.1)
        with mock.patch.object(warmstart_module, "ppisp_apply", recorder):
            warmstart_sh_dc_from_ppisp(model, ppisp_module, camera_id=0, frame_id=0)

        expected_albedo = RGB2SH(2.0 * SH2RGB(original_albedo) + 0.1)
        torch.testing.assert_close(model.features_albedo, expected_albedo, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
