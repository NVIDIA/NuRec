# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`nurec.utils.io.export.ppisp_export.controller.writer`.

Tests are hermetic and CPU-only. The PPISP module surface is mocked
with the smallest possible stand-ins:

* :class:`_FakeController` mirrors ``ppisp._PPISPController`` on CPU
  and matches the architecture spec locked in
  :mod:`nurec.utils.io.export.ppisp_export.controller.weights`.
* :class:`_FakePPISPModule` exposes only the attributes consumed by
  the writer (``num_cameras``, ``vignetting_params``, ``crf_params``,
  ``controllers``).

The smoke tests for the static SPG ``.usda`` sidecars open them as
USD layers in-memory and assert their default-prim names and the
schema (input / output names) the writer relies on. Generated
controller sidecars are checked as serialized CUDA/Lua text because
the controller no longer ships a shared ``.usda`` schema file.
"""

from __future__ import annotations

import math
import unittest

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from pxr import Sdf, Usd

from nurec.utils.io.export.ppisp_export.controller.weights import EXPECTED_CONTROLLER_WEIGHTS_LEN
from nurec.utils.io.export.ppisp_export.controller.writer import (
    CONTROLLER_CU_FILE,
    CONTROLLER_EMBEDDED_CU_FILE_TEMPLATE,
    CONTROLLER_EMBEDDED_LUA_FILE_TEMPLATE,
    CONTROLLER_FEATURES_RENDER_VAR,
    CONTROLLER_LUA_FILE,
    CONTROLLER_PARAMS_RENDER_VAR,
    CONTROLLER_POOL_PRIM_NAME_TEMPLATE,
    CONTROLLER_PRIM_NAME_TEMPLATE,
    CONTROLLER_SUB_IDENTIFIER,
    CONTROLLER_WEIGHTS_CAMERA_ATTR,
    DEFAULT_PRIOR_EXPOSURE,
    DEFAULT_RESPONSIVITY,
    HDR_COLOR_RENDER_VAR,
    LDR_COLOR_RENDER_VAR,
    PPISP_ATTR_NAMESPACE,
    PPISP_AUTO_CU_FILE,
    PPISP_AUTO_LUA_FILE,
    PPISP_AUTO_OUTPUT_RENDER_VAR,
    PPISP_AUTO_PRIM_NAME,
    PPISP_AUTO_SUB_IDENTIFIER,
    PPISP_AUTO_USDA_FILE,
    PPISP_SPG_DIR,
    add_ppisp_auto_shader_to_render_product,
    get_ppisp_auto_spg_files,
    get_ppisp_embedded_controller_spg_files,
    render_embedded_controller_cuda_source,
    render_embedded_controller_lua_source,
)


# =============================================================================
# Test fixtures
# =============================================================================


class _FakeController(nn.Module):
    """CPU mirror of ``ppisp._PPISPController`` (default architecture)."""

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


def _make_distinguishable_controller(seed: int) -> _FakeController:
    """Return a controller whose weights depend deterministically on ``seed``."""
    torch.manual_seed(seed)
    controller = _FakeController()
    with torch.no_grad():
        for param in controller.parameters():
            param.copy_(torch.randn_like(param))
    return controller


class _FakePPISPModule:
    """PPISP stand-in exposing only the attributes the writer touches."""

    def __init__(self, num_cameras: int = 2) -> None:
        self.num_cameras = num_cameras
        # Distinguishable per-camera vignetting / CRF so propagation
        # tests can read back specific values.
        vig = torch.zeros(num_cameras, 3, 5, dtype=torch.float32)
        crf = torch.zeros(num_cameras, 3, 4, dtype=torch.float32)
        for cam in range(num_cameras):
            for ch in range(3):
                vig[cam, ch, 0] = 0.01 * cam + 0.001 * ch  # cx
                vig[cam, ch, 1] = 0.02 * cam + 0.002 * ch  # cy
                vig[cam, ch, 2] = 0.03 * cam + 0.003 * ch  # alpha1
                vig[cam, ch, 3] = 0.04 * cam + 0.004 * ch  # alpha2
                vig[cam, ch, 4] = 0.05 * cam + 0.005 * ch  # alpha3
                crf[cam, ch, 0] = 0.10 * cam + 0.010 * ch  # toe
                crf[cam, ch, 1] = 0.20 * cam + 0.020 * ch  # shoulder
                crf[cam, ch, 2] = 0.30 * cam + 0.030 * ch  # gamma
                crf[cam, ch, 3] = 0.40 * cam + 0.040 * ch  # center
        self.vignetting_params = vig
        self.crf_params = crf
        self.exposure_params = torch.zeros(1, dtype=torch.float32)
        self.color_params = torch.zeros(1, 8, dtype=torch.float32)
        self.controllers = nn.ModuleList([_make_distinguishable_controller(seed=cam + 1) for cam in range(num_cameras)])


def _build_minimal_render_stage(camera_names: List[str]) -> Usd.Stage:
    """Build an in-memory stage with the minimum prims the writer requires.

    Mirrors :file:`ppisp_export/usd_writer_test.py` so the two writers'
    test scaffolds stay aligned. References to the SPG
    ``.usda`` files may not resolve in-memory, but locally
    authored ``inputs:`` attributes and AOV connections round-trip
    deterministically.
    """
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Render", "Scope")
    stage.DefinePrim("/World", "Scope")
    stage.DefinePrim("/World/Cameras", "Scope")
    for name in camera_names:
        camera_path = f"/World/Cameras/{name}"
        stage.DefinePrim(camera_path, "Camera")

        render_product_path = f"/Render/{name}"
        render_product = stage.DefinePrim(render_product_path, "RenderProduct")
        camera_rel = render_product.CreateRelationship("camera")
        camera_rel.SetTargets([Sdf.Path(camera_path)])
        render_product.CreateRelationship("orderedVars")

        hdr_color_path = f"{render_product_path}/{HDR_COLOR_RENDER_VAR}"
        stage.DefinePrim(hdr_color_path, "RenderVar")
    return stage


def _author_default_graph(
    stage: Usd.Stage,
    render_product_path: str,
    camera_index: int = 0,
    *,
    responsivity: float = DEFAULT_RESPONSIVITY,
    prior_exposure: float = DEFAULT_PRIOR_EXPOSURE,
) -> Tuple[Usd.Prim, _FakePPISPModule]:
    """Helper that authors a default graph and returns (auto_prim, ppisp)."""
    ppisp = _FakePPISPModule()
    auto_prim = add_ppisp_auto_shader_to_render_product(
        stage=stage,
        render_product_path=render_product_path,
        camera_index=camera_index,
        ppisp=ppisp,
        controller=ppisp.controllers[camera_index],
        responsivity=responsivity,
        prior_exposure=prior_exposure,
    )
    return auto_prim, ppisp


def _read_float_input(stage: Usd.Stage, shader_path: str, input_name: str) -> float:
    prim = stage.GetPrimAtPath(shader_path)
    assert prim.IsValid(), f"shader prim missing at {shader_path}"
    attr = prim.GetAttribute(f"inputs:{input_name}")
    assert attr.IsValid(), f"inputs:{input_name} missing on {shader_path}"
    return float(attr.Get())


def _read_aov_connections(stage: Usd.Stage, render_var_path: str) -> List[str]:
    prim = stage.GetPrimAtPath(render_var_path)
    assert prim.IsValid(), f"RenderVar missing at {render_var_path}"
    attr = prim.GetAttribute("omni:rtx:aov")
    assert attr.IsValid(), f"omni:rtx:aov missing on {render_var_path}"
    return [str(p) for p in attr.GetConnections()]


def _decode_serialized_text(payload: str | bytes) -> str:
    if isinstance(payload, str):
        return payload
    return payload.decode("utf-8")


# =============================================================================
# add_ppisp_auto_shader_to_render_product: success path
# =============================================================================


class AddPpispAutoShaderToRenderProductTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = _build_minimal_render_stage(["cam0"])
        self.render_product_path = "/Render/cam0"

    def test_returns_auto_ppisp_prim(self) -> None:
        auto_prim, _ = _author_default_graph(self.stage, self.render_product_path)
        self.assertEqual(
            str(auto_prim.GetPath()),
            f"{self.render_product_path}/{PPISP_AUTO_PRIM_NAME}",
        )
        self.assertTrue(auto_prim.IsValid())

    def test_authors_controller_shader_prim(self) -> None:
        _author_default_graph(self.stage, self.render_product_path, camera_index=0)
        controller_pool_path = f"{self.render_product_path}/{CONTROLLER_POOL_PRIM_NAME_TEMPLATE.format(camera_index=0)}"
        controller_pool_prim = self.stage.GetPrimAtPath(controller_pool_path)
        self.assertTrue(controller_pool_prim.IsValid())
        self.assertEqual(controller_pool_prim.GetTypeName(), "Shader")

        controller_path = f"{self.render_product_path}/{CONTROLLER_PRIM_NAME_TEMPLATE.format(camera_index=0)}"
        controller_prim = self.stage.GetPrimAtPath(controller_path)
        self.assertTrue(controller_prim.IsValid())
        self.assertEqual(controller_prim.GetTypeName(), "Shader")

    def test_authors_auto_ppisp_shader_prim(self) -> None:
        _author_default_graph(self.stage, self.render_product_path)
        auto_path = f"{self.render_product_path}/{PPISP_AUTO_PRIM_NAME}"
        auto_prim = self.stage.GetPrimAtPath(auto_path)
        self.assertTrue(auto_prim.IsValid())
        self.assertEqual(auto_prim.GetTypeName(), "Shader")

    def test_controller_source_asset_metadata_authored_locally(self) -> None:
        # Locally duplicating sourceAsset metadata defends against the
        # USDZ-packaging case where Kit may not resolve referenced
        # Shader metadata. Mirrors the static PPISP writer's policy.
        _author_default_graph(self.stage, self.render_product_path)
        controller_path = f"{self.render_product_path}/{CONTROLLER_PRIM_NAME_TEMPLATE.format(camera_index=0)}"
        prim = self.stage.GetPrimAtPath(controller_path)
        impl_attr = prim.GetAttribute("info:implementationSource")
        asset_attr = prim.GetAttribute("info:spg:sourceAsset")
        sub_attr = prim.GetAttribute("info:spg:sourceAsset:subIdentifier")
        self.assertEqual(str(impl_attr.Get()), "sourceAsset")
        self.assertEqual(
            str(asset_attr.Get().path),
            CONTROLLER_EMBEDDED_CU_FILE_TEMPLATE.format(camera_index=0),
        )
        self.assertEqual(str(sub_attr.Get()), CONTROLLER_SUB_IDENTIFIER)

    def test_auto_ppisp_source_asset_metadata_authored_locally(self) -> None:
        _author_default_graph(self.stage, self.render_product_path)
        auto_path = f"{self.render_product_path}/{PPISP_AUTO_PRIM_NAME}"
        prim = self.stage.GetPrimAtPath(auto_path)
        self.assertEqual(str(prim.GetAttribute("info:spg:sourceAsset").Get().path), PPISP_AUTO_CU_FILE)
        self.assertEqual(
            str(prim.GetAttribute("info:spg:sourceAsset:subIdentifier").Get()),
            PPISP_AUTO_SUB_IDENTIFIER,
        )

    def test_default_controller_mode_does_not_author_weight_asset(self) -> None:
        _author_default_graph(self.stage, self.render_product_path, camera_index=0)
        controller_pool_path = f"{self.render_product_path}/{CONTROLLER_POOL_PRIM_NAME_TEMPLATE.format(camera_index=0)}"
        controller_pool_prim = self.stage.GetPrimAtPath(controller_pool_path)
        self.assertFalse(controller_pool_prim.GetAttribute("inputs:weights").IsValid())
        self.assertEqual(
            controller_pool_prim.GetAttribute("info:spg:sourceAsset").Get().path,
            CONTROLLER_EMBEDDED_CU_FILE_TEMPLATE.format(camera_index=0),
        )

        controller_path = f"{self.render_product_path}/{CONTROLLER_PRIM_NAME_TEMPLATE.format(camera_index=0)}"
        prim = self.stage.GetPrimAtPath(controller_path)
        self.assertFalse(prim.GetAttribute("inputs:weights").IsValid())
        self.assertEqual(
            prim.GetAttribute("info:spg:sourceAsset").Get().path,
            CONTROLLER_EMBEDDED_CU_FILE_TEMPLATE.format(camera_index=0),
        )

    def test_controller_launch_dimensions_authored_locally(self) -> None:
        _author_default_graph(self.stage, self.render_product_path, camera_index=0)
        controller_pool_path = f"{self.render_product_path}/{CONTROLLER_POOL_PRIM_NAME_TEMPLATE.format(camera_index=0)}"
        controller_pool_prim = self.stage.GetPrimAtPath(controller_pool_path)
        self.assertEqual(controller_pool_prim.GetAttribute("inputs:blockDimX").Get(), 256)
        self.assertEqual(controller_pool_prim.GetAttribute("inputs:gridDimX").Get(), 25)

        controller_path = f"{self.render_product_path}/{CONTROLLER_PRIM_NAME_TEMPLATE.format(camera_index=0)}"
        controller_prim = self.stage.GetPrimAtPath(controller_path)
        self.assertEqual(controller_prim.GetAttribute("inputs:blockDimX").Get(), 128)
        self.assertEqual(controller_prim.GetAttribute("inputs:gridDimX").Get(), 1)

    def test_embedded_controller_sidecars_differ_per_camera(self) -> None:
        ppisp = _FakePPISPModule(num_cameras=2)
        sidecars = get_ppisp_embedded_controller_spg_files(ppisp, [0, 1])
        self.assertEqual(
            [s.filename for s in sidecars],
            [
                CONTROLLER_EMBEDDED_CU_FILE_TEMPLATE.format(camera_index=0),
                CONTROLLER_EMBEDDED_LUA_FILE_TEMPLATE.format(camera_index=0),
                CONTROLLER_EMBEDDED_CU_FILE_TEMPLATE.format(camera_index=1),
                CONTROLLER_EMBEDDED_LUA_FILE_TEMPLATE.format(camera_index=1),
            ],
        )
        cuda0 = _decode_serialized_text(sidecars[0].serialized)
        lua0 = _decode_serialized_text(sidecars[1].serialized)
        cuda1 = _decode_serialized_text(sidecars[2].serialized)
        lua1 = _decode_serialized_text(sidecars[3].serialized)
        self.assertIn("kControllerWeights", cuda0)
        self.assertIn("controllerPoolProcess", cuda0)
        self.assertNotEqual(cuda0, cuda1)
        self.assertEqual(lua0, lua1)
        self.assertNotIn('inputs["weights"]', lua0)

    def test_embedded_controller_source_uses_no_weight_kernel_argument(self) -> None:
        ppisp = _FakePPISPModule()
        source = render_embedded_controller_cuda_source(ppisp.controllers[0])
        self.assertIn("static __device__ const float kControllerWeights[TOTAL_WEIGHTS]", source)
        self.assertIn("const float* __restrict__ weights = kControllerWeights;", source)
        self.assertNotIn(
            'extern "C" __global__ void controllerPoolProcess(\n'
            "    int inW,\n"
            "    int inH,\n"
            "    cudaTextureObject_t inHdrColor,\n"
            "    const float* __restrict__ weights",
            source,
        )
        self.assertNotIn(
            'extern "C" __global__ void controllerProcess(\n'
            "    const float* __restrict__ controllerFeatures,\n"
            "    const float* __restrict__ weights",
            source,
        )

    def test_embedded_controller_lua_source_uses_no_weight_input(self) -> None:
        source = render_embedded_controller_lua_source()
        self.assertIn("controllerPoolProcess", source)
        self.assertIn("controllerProcess", source)
        self.assertNotIn('inputs["weights"]', source)
        self.assertNotIn("cuda.array(weights", source)

    def test_controller_prior_exposure_authored(self) -> None:
        _author_default_graph(self.stage, self.render_product_path, prior_exposure=0.7)
        controller_path = f"{self.render_product_path}/{CONTROLLER_PRIM_NAME_TEMPLATE.format(camera_index=0)}"
        self.assertAlmostEqual(
            _read_float_input(self.stage, controller_path, "priorExposure"),
            0.7,
            places=6,
        )

    def test_controller_prior_exposure_default_zero(self) -> None:
        _author_default_graph(self.stage, self.render_product_path)
        controller_path = f"{self.render_product_path}/{CONTROLLER_PRIM_NAME_TEMPLATE.format(camera_index=0)}"
        self.assertAlmostEqual(
            _read_float_input(self.stage, controller_path, "priorExposure"),
            0.0,
            places=6,
        )

    def test_auto_ppisp_responsivity_authored(self) -> None:
        _author_default_graph(self.stage, self.render_product_path, responsivity=2.5)
        auto_path = f"{self.render_product_path}/{PPISP_AUTO_PRIM_NAME}"
        self.assertAlmostEqual(_read_float_input(self.stage, auto_path, "responsivity"), 2.5, places=6)

    def test_auto_ppisp_responsivity_default(self) -> None:
        _author_default_graph(self.stage, self.render_product_path)
        auto_path = f"{self.render_product_path}/{PPISP_AUTO_PRIM_NAME}"
        self.assertAlmostEqual(_read_float_input(self.stage, auto_path, "responsivity"), 1.0, places=6)

    def test_controller_pool_responsivity_authored(self) -> None:
        _author_default_graph(self.stage, self.render_product_path, responsivity=2.5)
        pool_path = f"{self.render_product_path}/{CONTROLLER_POOL_PRIM_NAME_TEMPLATE.format(camera_index=0)}"
        self.assertAlmostEqual(_read_float_input(self.stage, pool_path, "responsivity"), 2.5, places=6)

    def test_controller_pool_responsivity_default(self) -> None:
        _author_default_graph(self.stage, self.render_product_path)
        pool_path = f"{self.render_product_path}/{CONTROLLER_POOL_PRIM_NAME_TEMPLATE.format(camera_index=0)}"
        self.assertAlmostEqual(_read_float_input(self.stage, pool_path, "responsivity"), 1.0, places=6)

    def test_auto_ppisp_vignetting_propagated_per_camera(self) -> None:
        # Camera 1 has different vignetting from camera 0 (see
        # _FakePPISPModule); write camera_index=1 and verify the
        # alpha1R/alpha1G/alpha1B values come from row 1.
        stage = _build_minimal_render_stage(["cam1"])
        ppisp = _FakePPISPModule(num_cameras=2)
        add_ppisp_auto_shader_to_render_product(
            stage=stage,
            render_product_path="/Render/cam1",
            camera_index=1,
            ppisp=ppisp,
            controller=ppisp.controllers[1],
        )
        auto_path = f"/Render/cam1/{PPISP_AUTO_PRIM_NAME}"
        self.assertAlmostEqual(
            _read_float_input(stage, auto_path, "vignettingAlpha1R"),
            float(ppisp.vignetting_params[1, 0, 2]),
            places=6,
        )
        self.assertAlmostEqual(
            _read_float_input(stage, auto_path, "vignettingAlpha1G"),
            float(ppisp.vignetting_params[1, 1, 2]),
            places=6,
        )
        self.assertAlmostEqual(
            _read_float_input(stage, auto_path, "vignettingAlpha1B"),
            float(ppisp.vignetting_params[1, 2, 2]),
            places=6,
        )

    def test_auto_ppisp_crf_propagated_per_camera(self) -> None:
        stage = _build_minimal_render_stage(["cam1"])
        ppisp = _FakePPISPModule(num_cameras=2)
        add_ppisp_auto_shader_to_render_product(
            stage=stage,
            render_product_path="/Render/cam1",
            camera_index=1,
            ppisp=ppisp,
            controller=ppisp.controllers[1],
        )
        auto_path = f"/Render/cam1/{PPISP_AUTO_PRIM_NAME}"
        self.assertAlmostEqual(
            _read_float_input(stage, auto_path, "crfToeR"),
            float(ppisp.crf_params[1, 0, 0]),
            places=6,
        )
        self.assertAlmostEqual(
            _read_float_input(stage, auto_path, "crfShoulderG"),
            float(ppisp.crf_params[1, 1, 1]),
            places=6,
        )
        self.assertAlmostEqual(
            _read_float_input(stage, auto_path, "crfCenterB"),
            float(ppisp.crf_params[1, 2, 3]),
            places=6,
        )

    def test_controller_params_render_var_wired_to_controller_output(self) -> None:
        _author_default_graph(self.stage, self.render_product_path, camera_index=0)
        var_path = f"{self.render_product_path}/{CONTROLLER_PARAMS_RENDER_VAR}"
        connections = _read_aov_connections(self.stage, var_path)
        controller_output = (
            f"{self.render_product_path}/"
            f"{CONTROLLER_PRIM_NAME_TEMPLATE.format(camera_index=0)}.outputs:{CONTROLLER_PARAMS_RENDER_VAR}"
        )
        self.assertEqual(connections, [controller_output])

    def test_auto_ppisp_controller_input_wired_to_render_var(self) -> None:
        # USD resolves the relative ``../ControllerParams.omni:rtx:aov``
        # connection authored by the writer to its absolute equivalent
        # when read back, so we assert against the absolute path.
        _author_default_graph(self.stage, self.render_product_path)
        auto_path = f"{self.render_product_path}/{PPISP_AUTO_PRIM_NAME}"
        prim = self.stage.GetPrimAtPath(auto_path)
        attr = prim.GetAttribute(f"inputs:{CONTROLLER_PARAMS_RENDER_VAR}")
        self.assertTrue(attr.IsValid())
        connections = [str(p) for p in attr.GetConnections()]
        self.assertEqual(
            connections,
            [f"{self.render_product_path}/{CONTROLLER_PARAMS_RENDER_VAR}.omni:rtx:aov"],
        )

    def test_ldr_color_render_var_wired_to_auto_ppisp_output(self) -> None:
        _author_default_graph(self.stage, self.render_product_path)
        var_path = f"{self.render_product_path}/{LDR_COLOR_RENDER_VAR}"
        connections = _read_aov_connections(self.stage, var_path)
        auto_output = f"{self.render_product_path}/{PPISP_AUTO_PRIM_NAME}.outputs:{PPISP_AUTO_OUTPUT_RENDER_VAR}"
        self.assertEqual(connections, [auto_output])

    def test_ordered_vars_extended_with_intermediates(self) -> None:
        _author_default_graph(self.stage, self.render_product_path)
        render_product = self.stage.GetPrimAtPath(self.render_product_path)
        targets = [str(p) for p in render_product.GetRelationship("orderedVars").GetTargets()]
        self.assertIn(f"{self.render_product_path}/{CONTROLLER_FEATURES_RENDER_VAR}", targets)
        self.assertIn(f"{self.render_product_path}/{CONTROLLER_PARAMS_RENDER_VAR}", targets)
        self.assertIn(f"{self.render_product_path}/{LDR_COLOR_RENDER_VAR}", targets)


# =============================================================================
# add_ppisp_auto_shader_to_render_product: validation + error paths
# =============================================================================


class AddPpispAutoShaderValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = _build_minimal_render_stage(["cam0"])

    def test_invalid_render_product_path_raises(self) -> None:
        ppisp = _FakePPISPModule()
        with self.assertRaisesRegex(ValueError, "RenderProduct not found"):
            add_ppisp_auto_shader_to_render_product(
                stage=self.stage,
                render_product_path="/Render/missing",
                camera_index=0,
                ppisp=ppisp,
                controller=ppisp.controllers[0],
            )

    def test_camera_index_out_of_range_raises(self) -> None:
        ppisp = _FakePPISPModule(num_cameras=1)
        with self.assertRaisesRegex(ValueError, "out of range"):
            add_ppisp_auto_shader_to_render_product(
                stage=self.stage,
                render_product_path="/Render/cam0",
                camera_index=5,
                ppisp=ppisp,
                controller=ppisp.controllers[0],
            )

    def test_negative_camera_index_raises(self) -> None:
        ppisp = _FakePPISPModule()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            add_ppisp_auto_shader_to_render_product(
                stage=self.stage,
                render_product_path="/Render/cam0",
                camera_index=-1,
                ppisp=ppisp,
                controller=ppisp.controllers[0],
            )

    def test_camera_index_must_be_int(self) -> None:
        ppisp = _FakePPISPModule()
        with self.assertRaisesRegex(TypeError, "camera_index must be int"):
            add_ppisp_auto_shader_to_render_product(
                stage=self.stage,
                render_product_path="/Render/cam0",
                camera_index=0.5,  # type: ignore[arg-type]
                ppisp=ppisp,
                controller=ppisp.controllers[0],
            )

    def test_camera_index_bool_rejected(self) -> None:
        ppisp = _FakePPISPModule()
        with self.assertRaisesRegex(TypeError, "camera_index must be int"):
            add_ppisp_auto_shader_to_render_product(
                stage=self.stage,
                render_product_path="/Render/cam0",
                camera_index=True,  # type: ignore[arg-type]
                ppisp=ppisp,
                controller=ppisp.controllers[0],
            )

    def test_responsivity_zero_rejected(self) -> None:
        ppisp = _FakePPISPModule()
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            add_ppisp_auto_shader_to_render_product(
                stage=self.stage,
                render_product_path="/Render/cam0",
                camera_index=0,
                ppisp=ppisp,
                controller=ppisp.controllers[0],
                responsivity=0.0,
            )

    def test_responsivity_nan_rejected(self) -> None:
        ppisp = _FakePPISPModule()
        with self.assertRaisesRegex(ValueError, "finite"):
            add_ppisp_auto_shader_to_render_product(
                stage=self.stage,
                render_product_path="/Render/cam0",
                camera_index=0,
                ppisp=ppisp,
                controller=ppisp.controllers[0],
                responsivity=math.nan,
            )

    def test_prior_exposure_inf_rejected(self) -> None:
        ppisp = _FakePPISPModule()
        with self.assertRaisesRegex(ValueError, "finite"):
            add_ppisp_auto_shader_to_render_product(
                stage=self.stage,
                render_product_path="/Render/cam0",
                camera_index=0,
                ppisp=ppisp,
                controller=ppisp.controllers[0],
                prior_exposure=math.inf,
            )

    def test_prior_exposure_negative_accepted(self) -> None:
        # Prior exposure can legitimately be negative (under-exposed
        # frames); the validator only enforces finiteness, not sign.
        ppisp = _FakePPISPModule()
        add_ppisp_auto_shader_to_render_product(
            stage=self.stage,
            render_product_path="/Render/cam0",
            camera_index=0,
            ppisp=ppisp,
            controller=ppisp.controllers[0],
            prior_exposure=-1.5,
        )
        controller_path = f"/Render/cam0/{CONTROLLER_PRIM_NAME_TEMPLATE.format(camera_index=0)}"
        self.assertAlmostEqual(
            _read_float_input(self.stage, controller_path, "priorExposure"),
            -1.5,
            places=6,
        )

    def test_non_finite_controller_weights_rejected(self) -> None:
        ppisp = _FakePPISPModule()
        controller = ppisp.controllers[0]
        assert isinstance(controller, _FakeController)
        conv1 = controller.cnn_encoder[0]
        assert isinstance(conv1, nn.Conv2d)
        with torch.no_grad():
            conv1.weight[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            add_ppisp_auto_shader_to_render_product(
                stage=self.stage,
                render_product_path="/Render/cam0",
                camera_index=0,
                ppisp=ppisp,
                controller=controller,
            )


# =============================================================================
# Sidecar packaging
# =============================================================================


class GetSpgFilesTest(unittest.TestCase):
    def _filenames(self, files: List) -> List[str]:
        return [f.filename for f in files]

    def test_auto_files_returned(self) -> None:
        files = get_ppisp_auto_spg_files()
        self.assertEqual(
            sorted(self._filenames(files)),
            sorted(
                [
                    "ppisp_usd_spg_auto.cu",
                    "ppisp_usd_spg_auto.cu.lua",
                    "ppisp_usd_spg_auto.usda",
                ]
            ),
        )

    def test_auto_files_have_payloads(self) -> None:
        for entry in get_ppisp_auto_spg_files():
            self.assertGreater(len(entry.serialized), 0, msg=f"{entry.filename} has empty payload")

    def test_embedded_controller_and_auto_filesets_disjoint(self) -> None:
        ppisp = _FakePPISPModule(num_cameras=2)
        controller = set(self._filenames(get_ppisp_embedded_controller_spg_files(ppisp, [0, 1])))
        auto = set(self._filenames(get_ppisp_auto_spg_files()))
        self.assertEqual(controller & auto, set())


# =============================================================================
# SPG asset .usda smoke tests
# =============================================================================


class SpgAssetSmokeTest(unittest.TestCase):
    """Open the .usda sidecars in-memory and assert the schema.

    The runtime SPG path is exercised by Kit; these tests catch
    schema drift (renamed inputs/outputs, defaultPrim rename, sub
    identifier rename) without requiring Kit at test time. The
    ``.cu`` and ``.cu.lua`` files are not parsed -- they are
    treated as opaque text whose contents are guarded by the byte
    payload checks in ``GetSpgFilesTest``.
    """

    def _open_layer(self, filename: str) -> Optional[Usd.Stage]:
        path = PPISP_SPG_DIR / filename
        if not path.exists():
            return None
        return Usd.Stage.Open(str(path))

    def test_auto_ppisp_usda_schema(self) -> None:
        stage = self._open_layer(PPISP_AUTO_USDA_FILE)
        self.assertIsNotNone(stage)
        assert stage is not None
        prim = stage.GetDefaultPrim()
        self.assertTrue(prim.IsValid())
        self.assertEqual(prim.GetName(), "PPISPAuto")
        for input_name in (
            "responsivity",
            "vignettingCenterR",
            "vignettingAlpha1R",
            "crfToeR",
            "HdrColor",
            "tileCountX",
            "tileCountY",
            CONTROLLER_PARAMS_RENDER_VAR,
        ):
            self.assertTrue(
                prim.GetAttribute(f"inputs:{input_name}").IsValid(),
                msg=f"auto-PPISP usda missing inputs:{input_name}",
            )
        for tile_input in ("tileCountX", "tileCountY"):
            self.assertEqual(int(prim.GetAttribute(f"inputs:{tile_input}").Get()), 1)
        self.assertTrue(prim.GetAttribute(f"outputs:{PPISP_AUTO_OUTPUT_RENDER_VAR}").IsValid())

    def test_auto_usda_subidentifier_matches_writer_constant(self) -> None:
        stage = self._open_layer(PPISP_AUTO_USDA_FILE)
        assert stage is not None
        prim = stage.GetDefaultPrim()
        attr = prim.GetAttribute("info:spg:sourceAsset:subIdentifier")
        self.assertEqual(str(attr.Get()), PPISP_AUTO_SUB_IDENTIFIER)

    def test_template_and_auto_sidecar_files_present(self) -> None:
        # Catches accidental BUILD.bazel data list omission as well as
        # filename drift in the sidecar set.
        expected = [
            CONTROLLER_CU_FILE,
            CONTROLLER_LUA_FILE,
            PPISP_AUTO_CU_FILE,
            PPISP_AUTO_LUA_FILE,
            PPISP_AUTO_USDA_FILE,
        ]
        for filename in expected:
            path: Path = PPISP_SPG_DIR / filename
            self.assertTrue(path.exists(), msg=f"missing SPG sidecar: {path}")
            self.assertGreater(path.stat().st_size, 0, msg=f"empty SPG sidecar: {path}")


# =============================================================================
# add_ppisp_auto_shader_to_render_product: camera-connection mode (Task 3)
# =============================================================================


class AddPpispAutoShaderCameraConnectionTest(unittest.TestCase):
    """When a ``ppisp_camera_path`` is given, responsivity / vignetting / CRF
    are hardcoded as literal ``inputs:*`` on the shader (connections disabled,
    see ``TODO(kit)``), while the values + controller weights are still
    authored on the camera ``ppisp:*`` attributes as the source of truth."""

    def setUp(self) -> None:
        self.stage = _build_minimal_render_stage(["cam0"])
        self.render_product_path = "/Render/cam0"
        self.ppisp = _FakePPISPModule(num_cameras=2)
        self.camera_path = Sdf.Path("/World/Cameras/cam0_ppisp")
        self.stage.DefinePrim(self.camera_path, "Camera")

    def _author(self, camera_index: int = 1) -> Usd.Prim:
        return add_ppisp_auto_shader_to_render_product(
            stage=self.stage,
            render_product_path=self.render_product_path,
            camera_index=camera_index,
            ppisp=self.ppisp,
            controller=self.ppisp.controllers[camera_index],
            ppisp_camera_path=self.camera_path,
        )

    def test_auto_shader_inputs_hardcoded_and_camera_attrs_present(self) -> None:
        self._author(camera_index=1)
        auto = self.stage.GetPrimAtPath(f"{self.render_product_path}/{PPISP_AUTO_PRIM_NAME}")
        cam = self.stage.GetPrimAtPath(self.camera_path)
        for inp in ("responsivity", "vignettingCenterR", "vignettingAlpha1R", "crfToeR", "crfCenterB"):
            attr = auto.GetAttribute(f"inputs:{inp}")
            self.assertTrue(attr.IsValid(), msg=f"inputs:{inp} missing on auto shader")
            # Hardcoded literal: connections disabled for now.
            self.assertEqual(list(attr.GetConnections()), [], msg=f"inputs:{inp} must not be connected")
            # Camera source-of-truth attr still authored for other workflows.
            self.assertTrue(
                cam.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{inp}").IsValid(),
                msg=f"missing ppisp:{inp} on camera",
            )

    def test_auto_shader_vignetting_value_lives_on_camera(self) -> None:
        self._author(camera_index=1)
        cam = self.stage.GetPrimAtPath(self.camera_path)
        attr = cam.GetAttribute(f"{PPISP_ATTR_NAMESPACE}vignettingAlpha1R")
        self.assertTrue(attr.IsValid())
        self.assertAlmostEqual(float(attr.Get()), float(self.ppisp.vignetting_params[1, 0, 2]), places=6)

    def test_ppisp_camera_authors_controller_weights_float_array(self) -> None:
        self._author(camera_index=1)
        cam = self.stage.GetPrimAtPath(self.camera_path)
        attr = cam.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{CONTROLLER_WEIGHTS_CAMERA_ATTR}")
        self.assertTrue(attr.IsValid(), msg="ppisp:controllerWeights missing on camera")
        self.assertEqual(attr.GetTypeName(), Sdf.ValueTypeNames.FloatArray)
        self.assertEqual(len(attr.Get()), EXPECTED_CONTROLLER_WEIGHTS_LEN)

    def test_ppisp_camera_omits_tile_counts(self) -> None:
        # Tiling is a RenderProduct property: it never lives as a ppisp:*
        # camera attribute (it is a constant shader input instead).
        self._author(camera_index=1)
        cam = self.stage.GetPrimAtPath(self.camera_path)
        for name in ("tileCountX", "tileCountY"):
            attr = cam.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{name}")
            self.assertFalse(attr.IsValid(), msg=f"unexpected ppisp:{name} on camera")

    def test_auto_graph_shaders_author_literal_tile_counts_with_camera(self) -> None:
        self._author(camera_index=1)
        # Auto-PPISP shader, controller pool and controller MLP all consume
        # the tile grid as a constant shader input (untiled default 1x1),
        # not as a camera connection -- tiling is a RenderProduct property.
        shader_names = (
            PPISP_AUTO_PRIM_NAME,
            CONTROLLER_POOL_PRIM_NAME_TEMPLATE.format(camera_index=1),
            CONTROLLER_PRIM_NAME_TEMPLATE.format(camera_index=1),
        )
        for shader_name in shader_names:
            shader = self.stage.GetPrimAtPath(f"{self.render_product_path}/{shader_name}")
            self.assertTrue(shader.IsValid(), msg=f"{shader_name} missing")
            for name in ("tileCountX", "tileCountY"):
                attr = shader.GetAttribute(f"inputs:{name}")
                self.assertTrue(attr.IsValid(), msg=f"inputs:{name} missing on {shader_name}")
                self.assertEqual(list(attr.GetConnections()), [])
                self.assertEqual(int(attr.Get()), 1)

    def test_auto_graph_shaders_author_literal_tile_counts_without_camera(self) -> None:
        # Without a camera path the tile counts are still authored as literal 1x1.
        add_ppisp_auto_shader_to_render_product(
            stage=self.stage,
            render_product_path=self.render_product_path,
            camera_index=1,
            ppisp=self.ppisp,
            controller=self.ppisp.controllers[1],
        )
        shader_names = (
            PPISP_AUTO_PRIM_NAME,
            CONTROLLER_POOL_PRIM_NAME_TEMPLATE.format(camera_index=1),
            CONTROLLER_PRIM_NAME_TEMPLATE.format(camera_index=1),
        )
        for shader_name in shader_names:
            shader = self.stage.GetPrimAtPath(f"{self.render_product_path}/{shader_name}")
            for name in ("tileCountX", "tileCountY"):
                attr = shader.GetAttribute(f"inputs:{name}")
                self.assertTrue(attr.IsValid(), msg=f"inputs:{name} missing on {shader_name}")
                self.assertEqual(list(attr.GetConnections()), [])
                self.assertEqual(int(attr.Get()), 1)


class DuplicatedConstantsInSyncTest(unittest.TestCase):
    """Guard the constants intentionally duplicated to avoid a circular import.

    ``usd_writer`` imports ``controller.writer``, so the latter cannot import
    the former. The constants below are re-declared as local literals in both
    modules; this test fails if they ever drift apart.
    """

    def test_constants_match_usd_writer(self) -> None:
        from nurec.utils.io.export.ppisp_export import usd_writer
        from nurec.utils.io.export.ppisp_export.controller import writer

        self.assertEqual(writer.PPISP_ATTR_NAMESPACE, usd_writer.PPISP_ATTR_NAMESPACE)
        self.assertEqual(writer.PPISP_TILE_COUNT_INPUT_NAMES, usd_writer.PPISP_TILE_COUNT_INPUT_NAMES)
        self.assertEqual(writer.DEFAULT_PPISP_TILE_COUNT, usd_writer.DEFAULT_PPISP_TILE_COUNT)


if __name__ == "__main__":
    unittest.main()
