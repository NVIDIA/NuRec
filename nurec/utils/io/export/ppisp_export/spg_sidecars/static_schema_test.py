# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static schema smoke tests for the runtime PPISP SPG sidecars."""

from __future__ import annotations

import unittest

from pathlib import Path

from pxr import Sdf, Usd


_SPG_DIR = Path(__file__).parent
_CU_FILE = "ppisp_usd_spg.cu"
_LUA_FILE = "ppisp_usd_spg.cu.lua"
_USDA_FILE = "ppisp_usd_spg.usda"
_DEFAULT_PRIM = "PPISP"
_ENTRY_POINT = "ppispProcess"
_INPUT_TEXTURE = "HdrColor"
_OUTPUT_TEXTURE = "PPISPColor"

_FLOAT_INPUTS = [
    "responsivity",
    "exposureOffset",
    "vignettingAlpha1R",
    "vignettingAlpha2R",
    "vignettingAlpha3R",
    "vignettingAlpha1G",
    "vignettingAlpha2G",
    "vignettingAlpha3G",
    "vignettingAlpha1B",
    "vignettingAlpha2B",
    "vignettingAlpha3B",
    "crfToeR",
    "crfShoulderR",
    "crfGammaR",
    "crfCenterR",
    "crfToeG",
    "crfShoulderG",
    "crfGammaG",
    "crfCenterG",
    "crfToeB",
    "crfShoulderB",
    "crfGammaB",
    "crfCenterB",
]

_FLOAT2_INPUTS = [
    "vignettingCenterR",
    "vignettingCenterG",
    "vignettingCenterB",
    "colorLatentBlue",
    "colorLatentRed",
    "colorLatentGreen",
    "colorLatentNeutral",
]


class PPISPStaticSPGSidecarSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        stage = Usd.Stage.Open(str(_SPG_DIR / _USDA_FILE))
        self.assertIsNotNone(stage)
        assert stage is not None
        self.stage = stage
        self.prim = stage.GetDefaultPrim()
        self.lua_source = (_SPG_DIR / _LUA_FILE).read_text(encoding="utf-8")

    def test_usda_shader_metadata_points_to_static_spg_entrypoint(self) -> None:
        self.assertTrue(self.prim.IsValid())
        self.assertEqual(self.prim.GetName(), _DEFAULT_PRIM)
        self.assertEqual(self.prim.GetTypeName(), "Shader")
        self.assertEqual(self.prim.GetAttribute("info:implementationSource").Get(), "sourceAsset")
        source_asset = self.prim.GetAttribute("info:spg:sourceAsset").Get()
        self.assertIsInstance(source_asset, Sdf.AssetPath)
        assert isinstance(source_asset, Sdf.AssetPath)
        self.assertEqual(source_asset.path, f"./{_CU_FILE}")
        self.assertEqual(self.prim.GetAttribute("info:spg:sourceAsset:subIdentifier").Get(), _ENTRY_POINT)

    def test_usda_declares_writer_authored_inputs_and_output(self) -> None:
        for name in _FLOAT_INPUTS:
            attr = self.prim.GetAttribute(f"inputs:{name}")
            self.assertTrue(attr.IsValid(), msg=f"missing inputs:{name}")
            self.assertEqual(attr.GetTypeName(), Sdf.ValueTypeNames.Float, msg=f"wrong type for inputs:{name}")
        for name in _FLOAT2_INPUTS:
            attr = self.prim.GetAttribute(f"inputs:{name}")
            self.assertTrue(attr.IsValid(), msg=f"missing inputs:{name}")
            self.assertEqual(attr.GetTypeName(), Sdf.ValueTypeNames.Float2, msg=f"wrong type for inputs:{name}")
        self.assertEqual(
            self.prim.GetAttribute(f"inputs:{_INPUT_TEXTURE}").GetTypeName(),
            Sdf.ValueTypeNames.Opaque,
        )
        self.assertEqual(
            self.prim.GetAttribute(f"outputs:{_OUTPUT_TEXTURE}").GetTypeName(),
            Sdf.ValueTypeNames.Opaque,
        )

    def test_lua_launcher_uses_usda_entrypoint_and_parameter_names(self) -> None:
        self.assertIn(f"function {_ENTRY_POINT}(", self.lua_source)
        self.assertIn(f'inputs["{_INPUT_TEXTURE}"]', self.lua_source)
        self.assertIn(f'outputs["{_OUTPUT_TEXTURE}"]', self.lua_source)
        for name in _FLOAT_INPUTS:
            self.assertIn(f'getFloat("{name}"', self.lua_source, msg=f"Lua launcher does not read {name}")
        for name in _FLOAT2_INPUTS:
            self.assertIn(f'getFloat2("{name}")', self.lua_source, msg=f"Lua launcher does not read {name}")

    def test_cuda_sidecar_does_not_require_cuda_runtime_headers(self) -> None:
        # Kit's CUDA SPG backend compiles sidecars with NVRTC and does
        # not expose the normal CUDA toolkit include search path.
        source = (_SPG_DIR / _CU_FILE).read_text(encoding="utf-8")
        self.assertNotIn("cuda_runtime.h", source)


if __name__ == "__main__":
    unittest.main()
