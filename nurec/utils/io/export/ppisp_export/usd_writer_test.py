# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ``nurec.utils.io.export.ppisp_export.usd_writer``.

Scope:

* PPISP responsivity authoring (validation + propagation).
* Static single-frame authoring branches (camera-only and
  camera+frame fixing) introduced by the export-mode integration.
* Batch entrypoint dispatch between the time-sampled, static, and
  automatic-parameter (controller) authoring paths.
"""

from __future__ import annotations

import math
import unittest

from typing import List, Optional

import torch
import torch.nn as nn

from pxr import Sdf, Usd, UsdGeom, UsdShade

from nurec.utils.io.export.ppisp_export.reference_selection import ResolvedPPISPReferenceIds
from nurec.utils.io.export.ppisp_export.usd_writer import (
    DEFAULT_PPISP_RESPONSIVITY,
    PPISP_ATTR_NAMESPACE,
    PPISP_CAMERA_API_SCHEMAS,
    PPISP_CAMERA_AUTO_EXPOSURE_ENABLED,
    PPISP_CAMERA_EXPOSURE,
    PPISP_CAMERA_EXPOSURE_FSTOP,
    PPISP_CAMERA_EXPOSURE_ISO,
    PPISP_CAMERA_EXPOSURE_RESPONSIVITY,
    PPISP_CAMERA_EXPOSURE_TIME,
    PPISP_CAMERA_SUFFIX,
    PPISP_RESPONSIVITY_INPUT_NAME,
    _author_ppisp_camera_attributes,
    _create_ppisp_camera,
    _set_responsivity_params,
    _validate_ppisp_responsivity,
    add_ppisp_shader_to_render_product,
    add_ppisp_to_all_render_products,
)


# =============================================================================
# Minimal PPISP stand-in
# =============================================================================


class _MockPPISP:
    """Smallest PPISP-shaped object the writer touches.

    Exposes the public attributes consumed by ``add_ppisp_shader_to_render_product``:
    ``num_cameras``, ``vignetting_params``, ``crf_params``, ``exposure_params``,
    ``color_params``, and (optionally for the controller path) ``controllers``.
    Concrete values do not matter for responsivity tests; they only need to be
    shape-correct so the existing parameter setters run. The ``frame_value_seed``
    knob lets static-frame tests verify that authored attributes truly come from
    the resolved frame index by setting ``exposure_params[i] = 10*i + seed``.
    """

    def __init__(self, num_cameras: int = 1, num_frames: int = 1, frame_value_seed: float = 0.0) -> None:
        self.num_cameras = num_cameras
        self.vignetting_params = torch.zeros(num_cameras, 3, 5, dtype=torch.float32)
        self.crf_params = torch.zeros(num_cameras, 3, 4, dtype=torch.float32)
        self.exposure_params = torch.tensor(
            [10.0 * float(i) + frame_value_seed for i in range(num_frames)],
            dtype=torch.float32,
        )
        self.color_params = torch.zeros(num_frames, 8, dtype=torch.float32)
        for i in range(num_frames):
            for j in range(8):
                self.color_params[i, j] = 0.01 * float(i) + 0.001 * float(j) + 0.1 * frame_value_seed


# =============================================================================
# Stage fixture builder
# =============================================================================


def _build_minimal_render_stage(camera_names: List[str]) -> Usd.Stage:
    """Build an in-memory stage with the minimum prims the writer requires.

    Creates ``/Render`` scope, one ``RenderProduct`` per ``camera_name`` with
    a ``camera`` relationship targeting ``/World/Cameras/<name>``, and an
    ``HdrColor`` ``RenderVar`` child. References from the writer to the SPG
    ``.usda`` file may not resolve in this stage, but locally-authored
    ``inputs:`` attributes are stored on the session layer and read back
    deterministically.
    """
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Render", "Scope")
    stage.DefinePrim("/World", "Scope")
    cameras_scope = stage.DefinePrim("/World/Cameras", "Scope")
    del cameras_scope

    for name in camera_names:
        camera_path = f"/World/Cameras/{name}"
        stage.DefinePrim(camera_path, "Camera")

        render_product_path = f"/Render/{name}"
        render_product = stage.DefinePrim(render_product_path, "RenderProduct")
        camera_rel = render_product.CreateRelationship("camera")
        camera_rel.SetTargets([Sdf.Path(camera_path)])
        render_product.CreateRelationship("orderedVars")

        hdr_color_path = f"{render_product_path}/HdrColor"
        stage.DefinePrim(hdr_color_path, "RenderVar")

    return stage


def _camera_path_for_render_product(stage: Usd.Stage, render_product_path: str) -> str:
    """Return the camera prim path bound to a RenderProduct (the cam_ppisp shim)."""
    render_product = stage.GetPrimAtPath(render_product_path)
    assert render_product.IsValid(), f"RenderProduct missing at {render_product_path}"
    targets = list(render_product.GetRelationship("camera").GetTargets())
    assert targets, f"no camera target on {render_product_path}"
    return str(targets[0])


def _read_camera_ppisp_float(stage: Usd.Stage, render_product_path: str, name: str) -> float:
    """Read a scalar ``ppisp:<name>`` attribute off the RenderProduct's bound camera."""
    camera_path = _camera_path_for_render_product(stage, render_product_path)
    prim = stage.GetPrimAtPath(camera_path)
    assert prim.IsValid(), f"camera prim missing at {camera_path}"
    attr = prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{name}")
    assert attr.IsValid(), f"{PPISP_ATTR_NAMESPACE}{name} missing on {camera_path}"
    return float(attr.Get())


def _read_camera_ppisp_color_latent(stage: Usd.Stage, render_product_path: str, name: str) -> tuple:
    """Read a ``ppisp:<name>`` float2 attribute off the RenderProduct's bound camera."""
    camera_path = _camera_path_for_render_product(stage, render_product_path)
    prim = stage.GetPrimAtPath(camera_path)
    assert prim.IsValid(), f"camera prim missing at {camera_path}"
    attr = prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{name}")
    assert attr.IsValid(), f"{PPISP_ATTR_NAMESPACE}{name} missing on {camera_path}"
    value = attr.Get()
    return (float(value[0]), float(value[1]))


def _camera_ppisp_attr_has_time_samples(stage: Usd.Stage, render_product_path: str, name: str) -> bool:
    """Return True iff the camera's ``ppisp:<name>`` attribute has time samples."""
    camera_path = _camera_path_for_render_product(stage, render_product_path)
    prim = stage.GetPrimAtPath(camera_path)
    assert prim.IsValid(), f"camera prim missing at {camera_path}"
    attr = prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{name}")
    assert attr.IsValid(), f"{PPISP_ATTR_NAMESPACE}{name} missing on {camera_path}"
    return attr.GetNumTimeSamples() > 0


def _shader_input_connection(stage: Usd.Stage, render_product_path: str, name: str) -> str:
    """Return the single connection target of a PPISP shader ``inputs:<name>``."""
    shader_path = f"{render_product_path}/PPISP"
    prim = stage.GetPrimAtPath(shader_path)
    assert prim.IsValid(), f"PPISP shader prim missing at {shader_path}"
    attr = prim.GetAttribute(f"inputs:{name}")
    assert attr.IsValid(), f"inputs:{name} missing on {shader_path}"
    connections = list(attr.GetConnections())
    assert len(connections) == 1, f"expected exactly one connection on inputs:{name}, got {connections}"
    return str(connections[0])


def _read_responsivity_input(stage: Usd.Stage, render_product_path: str) -> float:
    """Read the camera-sourced ``ppisp:responsivity`` for a RenderProduct.

    The batch entrypoint now authors responsivity onto the ``<cam>_ppisp``
    camera (the source of truth) and connects the shader input to it, so the
    value is read off the camera rather than the shader literal.
    """
    return _read_camera_ppisp_float(stage, render_product_path, PPISP_RESPONSIVITY_INPUT_NAME)


# =============================================================================
# _validate_ppisp_responsivity
# =============================================================================


class ValidatePpispResponsivityTest(unittest.TestCase):
    def test_accepts_unity(self) -> None:
        _validate_ppisp_responsivity(1.0)

    def test_accepts_int(self) -> None:
        _validate_ppisp_responsivity(2)

    def test_accepts_small_and_large_positives(self) -> None:
        _validate_ppisp_responsivity(1e-6)
        _validate_ppisp_responsivity(1e6)

    def test_rejects_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            _validate_ppisp_responsivity(0.0)

    def test_rejects_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            _validate_ppisp_responsivity(-0.5)

    def test_rejects_nan(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            _validate_ppisp_responsivity(math.nan)

    def test_rejects_positive_inf(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            _validate_ppisp_responsivity(math.inf)

    def test_rejects_negative_inf(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            _validate_ppisp_responsivity(-math.inf)

    def test_rejects_string(self) -> None:
        with self.assertRaisesRegex(TypeError, "real number"):
            _validate_ppisp_responsivity("1.0")

    def test_rejects_bool(self) -> None:
        # ``bool`` subclasses ``int`` in Python; explicit rejection avoids
        # silent ``True``/``False`` being treated as 1.0/0.0.
        with self.assertRaisesRegex(TypeError, "real number"):
            _validate_ppisp_responsivity(True)
        with self.assertRaisesRegex(TypeError, "real number"):
            _validate_ppisp_responsivity(False)

    def test_rejects_none(self) -> None:
        with self.assertRaisesRegex(TypeError, "real number"):
            _validate_ppisp_responsivity(None)


# =============================================================================
# _set_responsivity_params
# =============================================================================


class SetResponsivityParamsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = Usd.Stage.CreateInMemory()
        shader = UsdShade.Shader.Define(self.stage, "/PPISP")
        self.shader = shader

    def test_default_value_is_one(self) -> None:
        _set_responsivity_params(self.shader, DEFAULT_PPISP_RESPONSIVITY)
        attr = self.shader.GetPrim().GetAttribute(f"inputs:{PPISP_RESPONSIVITY_INPUT_NAME}")
        self.assertTrue(attr.IsValid())
        self.assertAlmostEqual(float(attr.Get()), 1.0, places=6)

    def test_custom_value_round_trips(self) -> None:
        _set_responsivity_params(self.shader, 2.5)
        attr = self.shader.GetPrim().GetAttribute(f"inputs:{PPISP_RESPONSIVITY_INPUT_NAME}")
        self.assertAlmostEqual(float(attr.Get()), 2.5, places=6)

    def test_attribute_is_float_typed(self) -> None:
        _set_responsivity_params(self.shader, 1.0)
        attr = self.shader.GetPrim().GetAttribute(f"inputs:{PPISP_RESPONSIVITY_INPUT_NAME}")
        self.assertEqual(attr.GetTypeName(), Sdf.ValueTypeNames.Float)

    def test_overwrites_previous_value(self) -> None:
        _set_responsivity_params(self.shader, 0.25)
        _set_responsivity_params(self.shader, 4.0)
        attr = self.shader.GetPrim().GetAttribute(f"inputs:{PPISP_RESPONSIVITY_INPUT_NAME}")
        self.assertAlmostEqual(float(attr.Get()), 4.0, places=6)


# =============================================================================
# add_ppisp_to_all_render_products: responsivity propagation
# =============================================================================


class AddPpispToAllRenderProductsResponsivityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_names = ["camera_front", "camera_rear"]
        self.stage = _build_minimal_render_stage(self.camera_names)
        self.ppisp = _MockPPISP(num_cameras=len(self.camera_names), num_frames=1)
        self.camera_name_to_index = {name: i for i, name in enumerate(self.camera_names)}
        self.camera_frame_mappings = {name: [0] for name in self.camera_names}

    def _run(self, **overrides: object) -> None:
        kwargs: dict[str, object] = dict(
            stage=self.stage,
            ppisp=self.ppisp,
            camera_name_to_index=self.camera_name_to_index,
            camera_frame_mappings=self.camera_frame_mappings,
        )
        kwargs.update(overrides)
        add_ppisp_to_all_render_products(**kwargs)  # type: ignore[arg-type]

    def test_default_responsivity_is_unity(self) -> None:
        self._run()
        for name in self.camera_names:
            value = _read_responsivity_input(self.stage, f"/Render/{name}")
            self.assertAlmostEqual(value, 1.0, places=6)

    def test_custom_responsivity_propagates_to_every_shader(self) -> None:
        self._run(responsivity=0.75)
        for name in self.camera_names:
            value = _read_responsivity_input(self.stage, f"/Render/{name}")
            self.assertAlmostEqual(value, 0.75, places=6)

    def test_zero_responsivity_raises_before_mutation(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            self._run(responsivity=0.0)
        for name in self.camera_names:
            shader_prim = self.stage.GetPrimAtPath(f"/Render/{name}/PPISP")
            self.assertFalse(shader_prim.IsValid(), f"shader created despite invalid responsivity for {name}")

    def test_negative_responsivity_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            self._run(responsivity=-1.0)

    def test_nan_responsivity_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            self._run(responsivity=math.nan)

    def test_inf_responsivity_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            self._run(responsivity=math.inf)

    def test_string_responsivity_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "real number"):
            self._run(responsivity="1.0")

    def test_bool_responsivity_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "real number"):
            self._run(responsivity=True)

    def test_does_not_clobber_other_inputs(self) -> None:
        self._run(responsivity=2.0)
        # Vignetting and CRF are also authored on the PPISP shader; verify
        # that responsivity authoring did not displace any of them.
        for name in self.camera_names:
            shader_prim = self.stage.GetPrimAtPath(f"/Render/{name}/PPISP")
            self.assertTrue(shader_prim.GetAttribute("inputs:exposureOffset").IsValid())
            self.assertTrue(shader_prim.GetAttribute("inputs:vignettingCenterR").IsValid())
            self.assertTrue(shader_prim.GetAttribute("inputs:crfToeR").IsValid())
            self.assertTrue(shader_prim.GetAttribute("inputs:colorLatentBlue").IsValid())

    def test_int_responsivity_accepted(self) -> None:
        self._run(responsivity=2)
        value = _read_responsivity_input(self.stage, f"/Render/{self.camera_names[0]}")
        self.assertAlmostEqual(value, 2.0, places=6)


# =============================================================================
# Module export contracts (catch accidental rename in future commits)
# =============================================================================


class ModuleConstantsTest(unittest.TestCase):
    def test_input_name_is_responsivity(self) -> None:
        self.assertEqual(PPISP_RESPONSIVITY_INPUT_NAME, "responsivity")


# =============================================================================
# Helpers shared by the new reference-id / controller-dispatch tests
# =============================================================================


def _read_float_input(stage: Usd.Stage, shader_path: str, name: str) -> float:
    prim = stage.GetPrimAtPath(shader_path)
    assert prim.IsValid(), f"shader prim missing at {shader_path}"
    attr = prim.GetAttribute(f"inputs:{name}")
    assert attr.IsValid(), f"inputs:{name} missing on {shader_path}"
    return float(attr.Get())


def _read_color_latent(stage: Usd.Stage, shader_path: str, name: str) -> tuple:
    prim = stage.GetPrimAtPath(shader_path)
    assert prim.IsValid(), f"shader prim missing at {shader_path}"
    attr = prim.GetAttribute(f"inputs:{name}")
    assert attr.IsValid(), f"inputs:{name} missing on {shader_path}"
    value = attr.Get()
    return (float(value[0]), float(value[1]))


def _attribute_has_time_samples(stage: Usd.Stage, shader_path: str, name: str) -> bool:
    """Return True iff the attribute has any USD time samples authored.

    Used to distinguish the static branch (no time samples, just a default
    value) from the time-sampled branch (one sample per frame plus a default).
    """
    prim = stage.GetPrimAtPath(shader_path)
    assert prim.IsValid(), f"shader prim missing at {shader_path}"
    attr = prim.GetAttribute(f"inputs:{name}")
    assert attr.IsValid(), f"inputs:{name} missing on {shader_path}"
    return attr.GetNumTimeSamples() > 0


# =============================================================================
# add_ppisp_shader_to_render_product: resolved_reference_ids dispatch
# =============================================================================


class AddPpispShaderResolvedReferenceIdsTest(unittest.TestCase):
    """Per-RenderProduct branches keyed by ``resolved_reference_ids``.

    Backward-compat is enforced by the responsivity tests above; this
    suite focuses exclusively on the static single-frame and
    camera-only branches introduced by the export-mode integration.
    """

    def setUp(self) -> None:
        self.camera_names = ["camera_front"]
        self.stage = _build_minimal_render_stage(self.camera_names)
        self.ppisp = _MockPPISP(num_cameras=1, num_frames=4, frame_value_seed=2.0)

    def _author(self, ids: Optional[ResolvedPPISPReferenceIds]) -> str:
        path = "/Render/camera_front"
        add_ppisp_shader_to_render_product(
            stage=self.stage,
            render_product_path=path,
            camera_index=0,
            ppisp=self.ppisp,
            frame_indices=[0, 1, 2, 3],
            resolved_reference_ids=ids,
        )
        return f"{path}/PPISP"

    def test_none_preserves_time_sampled_authoring(self) -> None:
        shader_path = self._author(ids=None)
        self.assertTrue(_attribute_has_time_samples(self.stage, shader_path, "exposureOffset"))
        self.assertTrue(_attribute_has_time_samples(self.stage, shader_path, "colorLatentBlue"))

    def test_camera_only_writes_neutral_exposure(self) -> None:
        ids = ResolvedPPISPReferenceIds(camera_id=0, frame_id=0, camera_was_user_set=True, frame_was_user_set=False)
        shader_path = self._author(ids=ids)
        self.assertFalse(_attribute_has_time_samples(self.stage, shader_path, "exposureOffset"))
        self.assertAlmostEqual(_read_float_input(self.stage, shader_path, "exposureOffset"), 0.0, places=6)

    def test_camera_only_writes_identity_color_latents(self) -> None:
        ids = ResolvedPPISPReferenceIds(camera_id=0, frame_id=0, camera_was_user_set=True, frame_was_user_set=False)
        shader_path = self._author(ids=ids)
        for name in ("colorLatentBlue", "colorLatentRed", "colorLatentGreen", "colorLatentNeutral"):
            self.assertEqual(_read_color_latent(self.stage, shader_path, name), (0.0, 0.0))
            self.assertFalse(_attribute_has_time_samples(self.stage, shader_path, name))

    def test_camera_and_frame_writes_static_exposure_from_resolved_frame(self) -> None:
        # frame_id=2 maps to exposure_params[2] = 10*2 + 2 = 22.0.
        ids = ResolvedPPISPReferenceIds(camera_id=0, frame_id=2, camera_was_user_set=True, frame_was_user_set=True)
        shader_path = self._author(ids=ids)
        self.assertFalse(_attribute_has_time_samples(self.stage, shader_path, "exposureOffset"))
        self.assertAlmostEqual(_read_float_input(self.stage, shader_path, "exposureOffset"), 22.0, places=4)

    def test_camera_and_frame_writes_static_color_from_resolved_frame(self) -> None:
        # color_params[2,j] = 0.01*2 + 0.001*j + 0.1*2 = 0.22 + 0.001*j.
        ids = ResolvedPPISPReferenceIds(camera_id=0, frame_id=2, camera_was_user_set=True, frame_was_user_set=True)
        shader_path = self._author(ids=ids)
        # colorLatentBlue == (color[2,0], color[2,1]) == (0.22, 0.221).
        b = _read_color_latent(self.stage, shader_path, "colorLatentBlue")
        self.assertAlmostEqual(b[0], 0.22, places=4)
        self.assertAlmostEqual(b[1], 0.221, places=4)
        self.assertFalse(_attribute_has_time_samples(self.stage, shader_path, "colorLatentBlue"))


# =============================================================================
# add_ppisp_to_all_render_products: reference-id propagation
# =============================================================================


class AddPpispToAllRenderProductsReferenceIdsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_names = ["camera_front", "camera_rear"]
        self.stage = _build_minimal_render_stage(self.camera_names)
        self.ppisp = _MockPPISP(num_cameras=len(self.camera_names), num_frames=4, frame_value_seed=2.0)
        self.camera_name_to_index = {name: i for i, name in enumerate(self.camera_names)}
        self.camera_frame_mappings = {name: [0, 1, 2, 3] for name in self.camera_names}

    def _run(self, **overrides: object) -> None:
        kwargs: dict[str, object] = dict(
            stage=self.stage,
            ppisp=self.ppisp,
            camera_name_to_index=self.camera_name_to_index,
            camera_frame_mappings=self.camera_frame_mappings,
        )
        kwargs.update(overrides)
        add_ppisp_to_all_render_products(**kwargs)  # type: ignore[arg-type]

    def test_default_unchanged_uses_time_sampled_path(self) -> None:
        # Backward compat: existing call sites that pass none of the
        # new flags must continue to author time-sampled exposure.
        self._run()
        for name in self.camera_names:
            self.assertTrue(_camera_ppisp_attr_has_time_samples(self.stage, f"/Render/{name}", "exposureOffset"))

    def test_camera_only_neutralizes_every_render_product(self) -> None:
        # Setting only camera_id forces the camera-only branch on every
        # RenderProduct, regardless of which physical camera the
        # iteration is on. The resolver pins the camera index globally.
        self._run(ppisp_reference_camera_id=1)
        for name in self.camera_names:
            rp_path = f"/Render/{name}"
            self.assertAlmostEqual(_read_camera_ppisp_float(self.stage, rp_path, "exposureOffset"), 0.0, places=6)
            self.assertFalse(_camera_ppisp_attr_has_time_samples(self.stage, rp_path, "exposureOffset"))

    def test_camera_and_frame_pins_static_attributes(self) -> None:
        # frame_id=3 -> exposure 32.0, color[3,0]=0.23.
        self._run(ppisp_reference_camera_id=0, ppisp_reference_frame_id=3)
        for name in self.camera_names:
            rp_path = f"/Render/{name}"
            self.assertAlmostEqual(_read_camera_ppisp_float(self.stage, rp_path, "exposureOffset"), 32.0, places=4)
            b = _read_camera_ppisp_color_latent(self.stage, rp_path, "colorLatentBlue")
            self.assertAlmostEqual(b[0], 0.23, places=4)

    def test_frame_only_rejected_with_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "ppisp_reference_frame_id was set"):
            self._run(ppisp_reference_frame_id=2)

    def test_invalid_camera_id_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ppisp_reference_camera_id"):
            self._run(ppisp_reference_camera_id=99)

    def test_invalid_frame_id_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ppisp_reference_frame_id"):
            self._run(ppisp_reference_camera_id=0, ppisp_reference_frame_id=99)

    def test_resolution_runs_before_any_authoring(self) -> None:
        # Frame-only is rejected; no PPISP shader prim should have been
        # authored on either RenderProduct.
        with self.assertRaises(ValueError):
            self._run(ppisp_reference_frame_id=1)
        for name in self.camera_names:
            self.assertFalse(self.stage.GetPrimAtPath(f"/Render/{name}/PPISP").IsValid())


# =============================================================================
# add_ppisp_to_all_render_products: ppisp_disable_controller dispatch
# =============================================================================


class _FakePPISPController(nn.Module):
    """Minimal CPU controller matching the architecture spec.

    Mirrors :class:`_FakeController` from the controller-writer tests
    so we can exercise the automatic-parameter branch
    without depending on the real ``ppisp`` package.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Flatten(),
        )
        trunk_input_dim = 64 * 5 * 5 + 1
        self.mlp_trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.exposure_head = nn.Linear(128, 1)
        self.color_head = nn.Linear(128, 8)


class _MockPPISPWithControllers(_MockPPISP):
    """``_MockPPISP`` extension that also exposes per-camera controllers."""

    def __init__(self, num_cameras: int = 2, num_frames: int = 1) -> None:
        super().__init__(num_cameras=num_cameras, num_frames=num_frames)
        self.controllers = nn.ModuleList([_FakePPISPController() for _ in range(num_cameras)])


class AddPpispToAllRenderProductsControllerDispatchTest(unittest.TestCase):
    """Dispatch between the static and automatic paths."""

    def setUp(self) -> None:
        self.camera_names = ["camera_front", "camera_rear"]
        self.stage = _build_minimal_render_stage(self.camera_names)
        self.ppisp = _MockPPISPWithControllers(num_cameras=len(self.camera_names))
        self.camera_name_to_index = {name: i for i, name in enumerate(self.camera_names)}
        self.camera_frame_mappings = {name: [0] for name in self.camera_names}

    def _run_with_controller_disabled(self, disabled: bool) -> None:
        add_ppisp_to_all_render_products(
            stage=self.stage,
            ppisp=self.ppisp,
            camera_name_to_index=self.camera_name_to_index,
            camera_frame_mappings=self.camera_frame_mappings,
            ppisp_disable_controller=disabled,
        )

    def test_default_dispatches_to_static_path(self) -> None:
        # Backward-compat: PPISP shader prim authored on every RP, no
        # PPISPAuto / PPISPController_* prims.
        self._run_with_controller_disabled(disabled=True)
        for name in self.camera_names:
            self.assertTrue(self.stage.GetPrimAtPath(f"/Render/{name}/PPISP").IsValid())
            self.assertFalse(self.stage.GetPrimAtPath(f"/Render/{name}/PPISPAuto").IsValid())

    def test_disabled_false_dispatches_to_auto_path(self) -> None:
        self._run_with_controller_disabled(disabled=False)
        for cam_idx, name in enumerate(self.camera_names):
            self.assertFalse(self.stage.GetPrimAtPath(f"/Render/{name}/PPISP").IsValid())
            self.assertTrue(self.stage.GetPrimAtPath(f"/Render/{name}/PPISPAuto").IsValid())
            self.assertTrue(self.stage.GetPrimAtPath(f"/Render/{name}/PPISPController_{cam_idx}").IsValid())

    def test_auto_path_uses_embedded_controller_sources(self) -> None:
        self._run_with_controller_disabled(disabled=False)
        for cam_idx, name in enumerate(self.camera_names):
            controller_pool_prim = self.stage.GetPrimAtPath(f"/Render/{name}/PPISPControllerPool_{cam_idx}")
            self.assertFalse(controller_pool_prim.GetAttribute("inputs:weights").IsValid())
            pool_source = controller_pool_prim.GetAttribute("info:spg:sourceAsset").Get()
            self.assertIsInstance(pool_source, Sdf.AssetPath)
            self.assertEqual(pool_source.path, f"ppisp_controller_{cam_idx}.cu")

            controller_prim = self.stage.GetPrimAtPath(f"/Render/{name}/PPISPController_{cam_idx}")
            self.assertFalse(controller_prim.GetAttribute("inputs:weights").IsValid())
            source = controller_prim.GetAttribute("info:spg:sourceAsset").Get()
            self.assertIsInstance(source, Sdf.AssetPath)
            self.assertEqual(source.path, f"ppisp_controller_{cam_idx}.cu")


# =============================================================================
# RenderVar `omni:rtx:aov` schema-attribute contract
# =============================================================================


class RenderVarAovIsSchemaAttributeTest(unittest.TestCase):
    """Pin that the ``omni:rtx:aov`` attribute on HdrColor/LdrColor RenderVars
    is authored as a non-custom (schema) attribute.

    Kit's SPG reflection layer looks up the AOV binding through the
    schema-defined ``omni:rtx:aov``. Authoring it as a custom attribute
    (the default for ``Prim.CreateAttribute``) silently breaks the
    lookup at runtime with:

        Failed to find binding index for input 'inputs:HdrColor'
        Failed to find binding index for output 'outputs:PPISPColor'

    and (because Kit short-circuits parameter binding when the texture
    I/O bindings fail) cascades into a flood of
    ``Failed to find parameter 'params:<X>' in shader reflection``
    warnings for every PPISP shader uniform.

    This test exercises ``add_ppisp_shader_to_render_product`` against
    an in-memory stage and reads back the ``omni:rtx:aov`` attribute
    specs from the layer to verify the ``custom`` flag is ``False`` on
    both RenderVars.
    """

    def setUp(self) -> None:
        self.camera_name = "camera1"
        self.stage = _build_minimal_render_stage([self.camera_name])
        self.ppisp = _MockPPISP(num_cameras=1, num_frames=1)
        self.render_product_path = f"/Render/{self.camera_name}"
        add_ppisp_shader_to_render_product(
            stage=self.stage,
            render_product_path=self.render_product_path,
            camera_index=0,
            ppisp=self.ppisp,
            frame_indices=[0],
        )

    def _attr_spec_is_custom(self, prim_path: str, attr_name: str) -> bool:
        prim = self.stage.GetPrimAtPath(prim_path)
        self.assertTrue(prim.IsValid(), f"prim missing at {prim_path}")
        attr = prim.GetAttribute(attr_name)
        self.assertTrue(attr.IsValid(), f"attribute {attr_name} missing on {prim_path}")
        layer = self.stage.GetRootLayer()
        attr_spec = layer.GetAttributeAtPath(attr.GetPath())
        self.assertIsNotNone(attr_spec, f"attribute spec not found for {attr.GetPath()}")
        return bool(attr_spec.custom)

    def test_hdr_color_aov_is_not_custom(self) -> None:
        self.assertFalse(
            self._attr_spec_is_custom(f"{self.render_product_path}/HdrColor", "omni:rtx:aov"),
            "HdrColor.omni:rtx:aov must be a schema (non-custom) attribute for Kit SPG reflection",
        )

    def test_ldr_color_aov_is_not_custom(self) -> None:
        self.assertFalse(
            self._attr_spec_is_custom(f"{self.render_product_path}/LdrColor", "omni:rtx:aov"),
            "LdrColor.omni:rtx:aov must be a schema (non-custom) attribute for Kit SPG reflection",
        )

    def test_ldr_color_aov_connects_to_ppisp_output(self) -> None:
        ldr_aov = self.stage.GetPrimAtPath(f"{self.render_product_path}/LdrColor").GetAttribute("omni:rtx:aov")
        targets = list(ldr_aov.GetConnections())
        self.assertEqual(
            targets,
            [Sdf.Path(f"{self.render_product_path}/PPISP.outputs:PPISPColor")],
            "LdrColor.omni:rtx:aov must connect to the PPISP shader output",
        )


# =============================================================================
# _create_ppisp_camera: per-RenderProduct neutral-exposure shim
# =============================================================================


class CreatePpispCameraTest(unittest.TestCase):
    """Regression coverage for the lavoisier-parity camera ISP-disable shim.

    Two prior regressions, both pinned here:

    1.  Resolvability gating (``IsValid()`` check) -- the production
        call site (``compose_default_usd_stage``) brings the rig-camera
        payload in via a reference moments before invoking the shim, so
        the rig camera is *not* yet resolvable on the authoring stage.
        Gating shim creation on ``GetPrimAtPath(source).IsValid()``
        silently skipped authoring in production, leaving the
        RenderProduct bound to the rig camera with Kit's exposure
        stage active. The shim must author purely from the camera
        path.

    2.  Placement -- USD's ``inherits`` arc is a specialization arc
        that brings only the target prim's own opinions into the
        inheritor, NOT ancestor transforms. NCORE rigs put the time-
        sampled pose on the rig-parent Xform (``sensor_rig_0``) and
        leave the camera prim with intrinsics only, so a shim placed
        under ``/Render/<rp>/<leaf>`` was rig-detached and rendered
        from an identity-transform pose. The shim must be authored as
        a sibling of the source camera so it inherits the rig pose
        via standard parent-transform composition.
    """

    def setUp(self) -> None:
        self.stage = Usd.Stage.CreateInMemory()
        self.stage.DefinePrim("/Render", "Scope")
        self.render_product_path = "/Render/camera1"
        self.source_camera_path = "/World/rig_trajectories/sensor_rig_0/camera1"
        self.shim_path = "/World/rig_trajectories/sensor_rig_0/camera1_ppisp"

        self.render_product = self.stage.DefinePrim(self.render_product_path, "RenderProduct")
        self.camera_rel = self.render_product.CreateRelationship("camera")
        self.camera_rel.SetTargets([Sdf.Path(self.source_camera_path)])

    def _shim_prim(self) -> Usd.Prim:
        prim = self.stage.GetPrimAtPath(self.shim_path)
        self.assertTrue(prim.IsValid(), f"shim missing at {self.shim_path}")
        return prim

    def test_authors_shim_when_source_camera_prim_absent_from_stage(self) -> None:
        # Source camera prim is intentionally NOT defined on this stage.
        # This mirrors the cross-layer composition timing in production.
        self.assertFalse(self.stage.GetPrimAtPath(self.source_camera_path).IsValid())

        _create_ppisp_camera(self.stage, self.render_product)

        prim = self._shim_prim()
        self.assertEqual(prim.GetTypeName(), "Camera")

    def test_shim_is_hidden_and_invisible(self) -> None:
        _create_ppisp_camera(self.stage, self.render_product)
        prim = self._shim_prim()
        self.assertTrue(prim.IsHidden(), "shim must be hidden from UI listings")
        visibility = UsdGeom.Imageable(prim).GetVisibilityAttr().Get()
        self.assertEqual(visibility, "invisible")

    def test_shim_inherits_from_source_camera_path(self) -> None:
        _create_ppisp_camera(self.stage, self.render_product)
        prim = self._shim_prim()
        # ``GetAllDirectInherits`` reports paths authored by ``AddInherit``
        # without requiring the target prim to be composed.
        inherit_paths = list(prim.GetInherits().GetAllDirectInherits())
        self.assertIn(Sdf.Path(self.source_camera_path), inherit_paths)

    def test_shim_authors_neutral_exposure_attributes(self) -> None:
        _create_ppisp_camera(self.stage, self.render_product)
        prim = self._shim_prim()
        expected = [
            ("exposure", PPISP_CAMERA_EXPOSURE),
            ("exposure:fStop", PPISP_CAMERA_EXPOSURE_FSTOP),
            ("exposure:iso", PPISP_CAMERA_EXPOSURE_ISO),
            ("exposure:responsivity", PPISP_CAMERA_EXPOSURE_RESPONSIVITY),
            ("exposure:time", PPISP_CAMERA_EXPOSURE_TIME),
        ]
        for attr_name, expected_value in expected:
            attr = prim.GetAttribute(attr_name)
            self.assertTrue(attr.IsValid(), f"missing {attr_name} on shim")
            self.assertEqual(attr.GetTypeName(), Sdf.ValueTypeNames.Float)
            self.assertAlmostEqual(float(attr.Get()), float(expected_value), places=6)

    def test_shim_applies_rtx_camera_api_schemas(self) -> None:
        _create_ppisp_camera(self.stage, self.render_product)
        prim = self._shim_prim()
        api_schemas = prim.GetMetadata("apiSchemas")
        self.assertIsInstance(api_schemas, Sdf.TokenListOp)
        assert isinstance(api_schemas, Sdf.TokenListOp)
        self.assertEqual(list(api_schemas.explicitItems), PPISP_CAMERA_API_SCHEMAS)

    def test_shim_authors_rtx_auto_exposure_attribute(self) -> None:
        _create_ppisp_camera(self.stage, self.render_product)
        prim = self._shim_prim()

        auto_exposure = prim.GetAttribute("omni:rtx:autoExposure:enabled")
        self.assertTrue(auto_exposure.IsValid())
        self.assertEqual(auto_exposure.GetTypeName(), Sdf.ValueTypeNames.Bool)
        self.assertEqual(auto_exposure.Get(), PPISP_CAMERA_AUTO_EXPOSURE_ENABLED)

    def test_render_product_camera_rel_repointed_to_shim(self) -> None:
        _create_ppisp_camera(self.stage, self.render_product)
        targets = list(self.camera_rel.GetTargets())
        self.assertEqual(targets, [Sdf.Path(self.shim_path)])

    def test_empty_camera_targets_is_no_op(self) -> None:
        empty_render_product = self.stage.DefinePrim("/Render/empty_rp", "RenderProduct")
        empty_render_product.CreateRelationship("camera")
        camera_prims_before = {p.GetPath() for p in self.stage.Traverse() if p.GetTypeName() == "Camera"}

        _create_ppisp_camera(self.stage, empty_render_product)

        camera_prims_after = {p.GetPath() for p in self.stage.Traverse() if p.GetTypeName() == "Camera"}
        self.assertEqual(
            camera_prims_after,
            camera_prims_before,
            "no Camera prim must be authored when the source-camera rel is empty",
        )
        self.assertEqual(list(empty_render_product.GetRelationship("camera").GetTargets()), [])

    def test_shim_authored_as_sibling_of_source_camera(self) -> None:
        # The shim must live as a sibling of the source camera so it
        # inherits the rig-parent (``sensor_rig_0``) transform. Authoring
        # under the RenderProduct (``/Render/camera1/...``) detaches the
        # shim from the rig pose, breaking lavoisier parity for rigs that
        # keep intrinsics on the camera and the pose on the parent.
        _create_ppisp_camera(self.stage, self.render_product)

        prim = self._shim_prim()
        self.assertEqual(
            prim.GetPath().GetParentPath(),
            Sdf.Path(self.source_camera_path).GetParentPath(),
            "shim must be a sibling of the source camera",
        )
        self.assertFalse(
            self.stage.GetPrimAtPath(f"{self.render_product_path}/camera1_ppisp").IsValid(),
            "shim must NOT be authored under the RenderProduct (regression: rig-detached pose)",
        )

    def test_shim_inherits_parent_transform_via_composition(self) -> None:
        # Build a rig where the parent Xform (``sensor_rig_0``) carries a
        # non-identity time-sampled transform and the camera carries only
        # intrinsics. The shim, as a sibling of the source camera, must
        # compose the same world transform as the source camera at every
        # sample. The OLD placement (shim under RenderProduct) would not
        # compose this transform, yielding an identity world matrix.
        from pxr import Gf
        from pxr import UsdGeom as _UsdGeom

        rig_parent = self.stage.OverridePrim("/World/rig_trajectories/sensor_rig_0")
        rig_parent.SetTypeName("Xform")
        xform_op = _UsdGeom.Xformable(rig_parent).AddTransformOp()
        sample_time = 7.0
        sample_matrix = Gf.Matrix4d(1.0).SetTranslateOnly(Gf.Vec3d(11.0, 22.0, 33.0))
        xform_op.Set(sample_matrix, time=sample_time)

        source_cam = self.stage.DefinePrim(self.source_camera_path, "Camera")
        self.assertTrue(source_cam.IsValid())

        _create_ppisp_camera(self.stage, self.render_product)

        xform_cache = _UsdGeom.XformCache(sample_time)
        source_world = xform_cache.GetLocalToWorldTransform(source_cam)
        shim_world = xform_cache.GetLocalToWorldTransform(self._shim_prim())

        for row in range(4):
            for col in range(4):
                self.assertAlmostEqual(
                    shim_world[row][col],
                    source_world[row][col],
                    places=6,
                    msg=f"shim world xform diverges from source at [{row}][{col}]",
                )

    def test_shim_creation_is_idempotent_when_called_twice(self) -> None:
        # Re-running export onto an existing layer (or invoking
        # ``add_ppisp_to_all_render_products`` twice in a single session)
        # must not double-author the shim or duplicate inherit arcs.
        _create_ppisp_camera(self.stage, self.render_product)
        _create_ppisp_camera(self.stage, self.render_product)

        camera_prims = [
            p for p in self.stage.Traverse() if p.GetTypeName() == "Camera" and p.GetName().endswith("_ppisp")
        ]
        self.assertEqual(len(camera_prims), 1, "shim must be authored exactly once")

        inherit_paths = list(camera_prims[0].GetInherits().GetAllDirectInherits())
        self.assertEqual(
            inherit_paths,
            [Sdf.Path(self.source_camera_path)],
            "inherit arc must not be duplicated on repeated invocations",
        )

        self.assertEqual(
            list(self.camera_rel.GetTargets()),
            [Sdf.Path(self.shim_path)],
            "RenderProduct camera rel must remain bound to the shim",
        )


# =============================================================================
# camera_ppisp source-of-truth attributes + shader connections
# =============================================================================


class PpispCameraSourceOfTruthTest(unittest.TestCase):
    """Task 1: the ``<cam>_ppisp`` camera carries the PPISP parameters."""

    def setUp(self) -> None:
        self.camera_names = ["camera_front"]
        self.stage = _build_minimal_render_stage(self.camera_names)
        self.ppisp = _MockPPISP(num_cameras=1, num_frames=4, frame_value_seed=2.0)
        self.render_product = self.stage.GetPrimAtPath("/Render/camera_front")

    def test_ppisp_camera_named_ppisp_and_neutralizes_exposure(self) -> None:
        cam_path = _create_ppisp_camera(self.stage, self.render_product)
        assert cam_path is not None
        self.assertTrue(str(cam_path).endswith(PPISP_CAMERA_SUFFIX))
        prim = self.stage.GetPrimAtPath(cam_path)
        self.assertEqual(prim.GetTypeName(), "Camera")
        self.assertAlmostEqual(float(prim.GetAttribute("exposure").Get()), float(PPISP_CAMERA_EXPOSURE), places=6)

    def test_ppisp_camera_authors_namespaced_attrs(self) -> None:
        cam_path = _create_ppisp_camera(self.stage, self.render_product)
        assert cam_path is not None
        _author_ppisp_camera_attributes(
            self.stage,
            cam_path,
            self.ppisp,
            0,
            responsivity=0.5,
            resolved_reference_ids=None,
            frame_indices=[0, 1, 2, 3],
        )
        prim = self.stage.GetPrimAtPath(cam_path)
        for name in (
            "responsivity",
            "vignettingCenterR",
            "vignettingAlpha1R",
            "crfToeR",
            "exposureOffset",
            "colorLatentBlue",
            "colorLatentNeutral",
        ):
            self.assertTrue(
                prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{name}").IsValid(),
                msg=f"missing {PPISP_ATTR_NAMESPACE}{name} on cam_ppisp",
            )
        self.assertAlmostEqual(float(prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}responsivity").Get()), 0.5, places=6)

    def test_ppisp_camera_exposure_time_samples(self) -> None:
        cam_path = _create_ppisp_camera(self.stage, self.render_product)
        assert cam_path is not None
        _author_ppisp_camera_attributes(
            self.stage,
            cam_path,
            self.ppisp,
            0,
            responsivity=1.0,
            resolved_reference_ids=None,
            frame_indices=[0, 1, 2, 3],
        )
        prim = self.stage.GetPrimAtPath(cam_path)
        self.assertGreater(prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}exposureOffset").GetNumTimeSamples(), 0)

    def test_ppisp_camera_omits_tile_counts(self) -> None:
        # Tiling is a RenderProduct property: it lives as a constant shader
        # input, never as a ppisp:* camera attribute.
        cam_path = _create_ppisp_camera(self.stage, self.render_product)
        assert cam_path is not None
        _author_ppisp_camera_attributes(
            self.stage,
            cam_path,
            self.ppisp,
            0,
            responsivity=1.0,
            resolved_reference_ids=None,
            frame_indices=[0, 1, 2, 3],
        )
        prim = self.stage.GetPrimAtPath(cam_path)
        for name in ("tileCountX", "tileCountY"):
            attr = prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{name}")
            self.assertFalse(attr.IsValid(), msg=f"unexpected {PPISP_ATTR_NAMESPACE}{name} on camera")


class PpispShaderConnectionTest(unittest.TestCase):
    """SPG shader inputs are hardcoded literals; camera ``ppisp:*`` attrs are
    still authored as the source of truth (connections disabled, see
    ``TODO(kit)`` in ``add_ppisp_shader_to_render_product``)."""

    def setUp(self) -> None:
        self.camera_names = ["camera_front", "camera_rear"]
        self.stage = _build_minimal_render_stage(self.camera_names)
        self.ppisp = _MockPPISP(num_cameras=len(self.camera_names), num_frames=1)
        self.camera_name_to_index = {name: i for i, name in enumerate(self.camera_names)}
        self.camera_frame_mappings = {name: [0] for name in self.camera_names}

    def test_shader_inputs_hardcoded_and_camera_attrs_present(self) -> None:
        add_ppisp_to_all_render_products(
            stage=self.stage,
            ppisp=self.ppisp,
            camera_name_to_index=self.camera_name_to_index,
            camera_frame_mappings=self.camera_frame_mappings,
        )
        for name in self.camera_names:
            rp_path = f"/Render/{name}"
            cam_path = _camera_path_for_render_product(self.stage, rp_path)
            cam_prim = self.stage.GetPrimAtPath(cam_path)
            shader = self.stage.GetPrimAtPath(f"{rp_path}/PPISP")
            for inp in (
                "responsivity",
                "exposureOffset",
                "vignettingCenterR",
                "crfToeR",
                "colorLatentBlue",
            ):
                shader_attr = shader.GetAttribute(f"inputs:{inp}")
                self.assertTrue(shader_attr.IsValid(), msg=f"missing inputs:{inp} on shader")
                # Hardcoded literal: not connected to the camera.
                self.assertEqual(list(shader_attr.GetConnections()), [], msg=f"inputs:{inp} must not be connected")
                # Camera source-of-truth attr still authored for other workflows.
                self.assertTrue(
                    cam_prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{inp}").IsValid(),
                    msg=f"missing {PPISP_ATTR_NAMESPACE}{inp} on camera",
                )

    def test_shader_authors_literal_tile_counts(self) -> None:
        # Tiling is a RenderProduct property authored as a constant shader
        # input (untiled default 1x1), not connected to the camera.
        add_ppisp_to_all_render_products(
            stage=self.stage,
            ppisp=self.ppisp,
            camera_name_to_index=self.camera_name_to_index,
            camera_frame_mappings=self.camera_frame_mappings,
        )
        for name in self.camera_names:
            shader_prim = self.stage.GetPrimAtPath(f"/Render/{name}/PPISP")
            for inp in ("tileCountX", "tileCountY"):
                attr = shader_prim.GetAttribute(f"inputs:{inp}")
                self.assertTrue(attr.IsValid(), msg=f"missing inputs:{inp}")
                self.assertEqual(attr.GetTypeName(), Sdf.ValueTypeNames.Int)
                self.assertEqual(int(attr.Get()), 1)
                self.assertEqual(list(attr.GetConnections()), [])

    def test_ppisp_camera_path_none_authors_literal(self) -> None:
        # Direct caller without a camera path keeps the legacy literal
        # authoring (no connection, value lives on the shader input).
        add_ppisp_shader_to_render_product(
            stage=self.stage,
            render_product_path="/Render/camera_front",
            camera_index=0,
            ppisp=self.ppisp,
            frame_indices=[0],
        )
        shader = self.stage.GetPrimAtPath("/Render/camera_front/PPISP")
        attr = shader.GetAttribute("inputs:responsivity")
        self.assertTrue(attr.IsValid())
        self.assertEqual(list(attr.GetConnections()), [])
        self.assertAlmostEqual(float(attr.Get()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
