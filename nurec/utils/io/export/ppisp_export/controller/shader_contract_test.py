# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests tying the Python controller layout to the SPG shaders.

Two complementary scopes:

* :class:`PPISPControllerShaderContractTest` parses the CUDA controller
  source and pins the architecture / weight-offset constants
  against ``ControllerArchitectureSpec``.
* :class:`PPISPControllerCudaTemplateContractTest` parses the CUDA and
  Lua controller templates and pins the embedded-weight SPG contract:
  kernels do not take a weight pointer, the source contains the
  replacement marker used for per-camera generated sidecars, and the
  Lua launcher does not bind a ``weights`` input.

The CUDA offset constants stay expressed as ``static const int OFF_*``
declarations and are parsed via :func:`_read_cu_constants`.
"""

from __future__ import annotations

import ast
import re
import unittest

from pathlib import Path
from typing import Dict, Tuple

from nurec.utils.io.export.ppisp_export.controller.weights import (
    EXPECTED_CONTROLLER_ARCHITECTURE,
    EXPECTED_CONTROLLER_WEIGHTS_LEN,
    ControllerArchitectureSpec,
)
from nurec.utils.io.export.ppisp_export.controller.writer import (
    CONTROLLER_CU_FILE,
    CONTROLLER_LUA_FILE,
    PPISP_AUTO_CU_FILE,
    PPISP_SPG_DIR,
)


_EMBEDDED_WEIGHTS_MARKER = "// __PPISP_CONTROLLER_EMBEDDED_WEIGHTS__"
_STATIC_PPISP_SPG_CU_FILE = "ppisp_usd_spg.cu"
_STATIC_PPISP_SPG_DIR = PPISP_SPG_DIR.parent / "spg_sidecars"
_CRF_CLAMP_SNIPPETS = (
    "const float eps      = 1.0e-6f;",
    "const float center   = fminf(fmaxf(sigmoid(centerRaw), eps), 1.0f - eps);",
    "const float lerpVal = fmaxf((shoulder - toe) * center + toe, eps);",
)
_LUA_SANDBOX_FORBIDDEN_PATTERNS = (
    "require",
    "dofile",
    "loadfile",
    "load(",
    "loadstring",
    "os.",
    "io.",
    "debug.",
    "package.",
    "getfenv",
    "setfenv",
    "rawget",
    "rawset",
    "getmetatable",
    "setmetatable",
    "collectgarbage",
    "pcall",
    "xpcall",
    "coroutine",
    "module",
)


_CU_CONST_RE = re.compile(r"^\s*static\s+const\s+int\s+([A-Z0-9_]+)\s*=\s*(.+?);\s*(?://.*)?$")


def _eval_cu_int_expr(expr: str, constants: Dict[str, int]) -> int:
    node = ast.parse(expr, mode="eval")
    return _eval_ast_int(node.body, constants)


def _eval_ast_int(node: ast.AST, constants: Dict[str, int]) -> int:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    if isinstance(node, ast.Name):
        return constants[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_int(node.operand, constants)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
    if isinstance(node, ast.BinOp):
        left = _eval_ast_int(node.left, constants)
        right = _eval_ast_int(node.right, constants)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    raise ValueError(f"unsupported CUDA integer expression: {ast.dump(node)}")


def _read_cu_constants(path: Path) -> Dict[str, int]:
    constants: Dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _CU_CONST_RE.match(line)
        if match is not None:
            name, expr = match.groups()
            constants[name] = _eval_cu_int_expr(expr, constants)
    return constants


def _expected_layout(spec: ControllerArchitectureSpec) -> Tuple[Dict[str, int], int]:
    offset = 0
    offsets: Dict[str, int] = {}
    blocks = [
        ("OFF_CONV1_W", spec.cnn_layer_1_channels * spec.cnn_in_channels),
        ("OFF_CONV1_B", spec.cnn_layer_1_channels),
        ("OFF_CONV2_W", spec.cnn_layer_2_channels * spec.cnn_layer_1_channels),
        ("OFF_CONV2_B", spec.cnn_layer_2_channels),
        ("OFF_CONV3_W", spec.cnn_feature_dim * spec.cnn_layer_2_channels),
        ("OFF_CONV3_B", spec.cnn_feature_dim),
        ("OFF_TRUNK0_W", spec.mlp_hidden_dim * spec.trunk_input_dim),
        ("OFF_TRUNK0_B", spec.mlp_hidden_dim),
        ("OFF_TRUNK1_W", spec.mlp_hidden_dim * spec.mlp_hidden_dim),
        ("OFF_TRUNK1_B", spec.mlp_hidden_dim),
        ("OFF_TRUNK2_W", spec.mlp_hidden_dim * spec.mlp_hidden_dim),
        ("OFF_TRUNK2_B", spec.mlp_hidden_dim),
        ("OFF_EXP_W", spec.mlp_hidden_dim),
        ("OFF_EXP_B", 1),
        ("OFF_COL_W", spec.color_params_per_frame * spec.mlp_hidden_dim),
        ("OFF_COL_B", spec.color_params_per_frame),
    ]
    for name, size in blocks:
        offsets[name] = offset
        offset += size
    return offsets, offset


class PPISPControllerShaderContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.constants = _read_cu_constants(PPISP_SPG_DIR / CONTROLLER_CU_FILE)
        self.spec = EXPECTED_CONTROLLER_ARCHITECTURE

    def test_architecture_constants_match_python_spec(self) -> None:
        pool_h, pool_w = self.spec.pool_grid_size
        expected = {
            "CNN_FEATURE_DIM": self.spec.cnn_feature_dim,
            "POOL_GRID_H": pool_h,
            "POOL_GRID_W": pool_w,
            "POOL_CELL_COUNT": pool_h * pool_w,
            "POOL_FEATURE_LEN": self.spec.cnn_feature_dim * pool_h * pool_w,
            "MLP_INPUT_DIM": self.spec.trunk_input_dim,
            "MLP_HIDDEN_DIM": self.spec.mlp_hidden_dim,
            "COLOR_PARAMS_PER_FRAME": self.spec.color_params_per_frame,
            "INPUT_DOWNSAMPLING": self.spec.input_downsampling,
        }
        for name, value in expected.items():
            self.assertEqual(self.constants[name], value, msg=f"{name} drifted")

    def test_weight_offsets_match_python_flatten_order(self) -> None:
        expected_offsets, expected_total = _expected_layout(self.spec)
        for name, value in expected_offsets.items():
            self.assertEqual(self.constants[name], value, msg=f"{name} drifted")
        self.assertEqual(expected_total, EXPECTED_CONTROLLER_WEIGHTS_LEN)
        self.assertEqual(self.constants["TOTAL_WEIGHTS"], EXPECTED_CONTROLLER_WEIGHTS_LEN)


# =============================================================================
# CUDA controller template contract
# =============================================================================


class PPISPControllerCudaTemplateContractTest(unittest.TestCase):
    """Pin the embedded-weight CUDA controller template contract."""

    def setUp(self) -> None:
        self.cuda_source = (PPISP_SPG_DIR / CONTROLLER_CU_FILE).read_text(encoding="utf-8")
        self.lua_source = (PPISP_SPG_DIR / CONTROLLER_LUA_FILE).read_text(encoding="utf-8")

    def test_cuda_template_contains_embedded_weight_marker(self) -> None:
        self.assertIn(_EMBEDDED_WEIGHTS_MARKER, self.cuda_source)
        self.assertIn("kControllerWeights", self.cuda_source)

    def test_kernel_entry_points_have_no_weight_argument(self) -> None:
        self.assertIn("controllerPoolProcess", self.cuda_source)
        self.assertIn("controllerProcess", self.cuda_source)
        self.assertNotIn(
            'extern "C" __global__ void controllerPoolProcess(\n'
            "    int inW,\n"
            "    int inH,\n"
            "    cudaTextureObject_t inHdrColor,\n"
            "    const float* __restrict__ weights",
            self.cuda_source,
        )
        self.assertNotIn(
            'extern "C" __global__ void controllerProcess(\n'
            "    const float* __restrict__ controllerFeatures,\n"
            "    const float* __restrict__ weights",
            self.cuda_source,
        )

    def test_lua_launcher_has_no_weight_input(self) -> None:
        self.assertIn("controllerPoolProcess", self.lua_source)
        self.assertIn("controllerProcess", self.lua_source)
        self.assertNotIn('inputs["weights"]', self.lua_source)
        self.assertNotIn("cuda.array(weights", self.lua_source)

    def test_controller_template_files_exist(self) -> None:
        for filename in (CONTROLLER_CU_FILE, CONTROLLER_LUA_FILE):
            self.assertTrue(
                (PPISP_SPG_DIR / filename).exists(),
                msg=f"{filename} missing from controller package",
            )

    def test_cuda_sidecars_do_not_require_cuda_runtime_headers(self) -> None:
        # Kit's CUDA SPG backend compiles sidecars with NVRTC and does
        # not expose the normal CUDA toolkit include search path.
        for path in PPISP_SPG_DIR.glob("*.cu"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("cuda_runtime.h", source, msg=f"{path.name} must be NVRTC self-contained")

    def test_auto_ppisp_crf_clamps_match_static_sidecar(self) -> None:
        static_source = (_STATIC_PPISP_SPG_DIR / _STATIC_PPISP_SPG_CU_FILE).read_text(encoding="utf-8")
        auto_source = (PPISP_SPG_DIR / PPISP_AUTO_CU_FILE).read_text(encoding="utf-8")
        for snippet in _CRF_CLAMP_SNIPPETS:
            self.assertIn(snippet, static_source)
            self.assertIn(snippet, auto_source)
        self.assertNotIn("const float center   = sigmoid(centerRaw);", auto_source)
        self.assertNotIn("const float lerpVal = (shoulder - toe) * center + toe;", auto_source)

    def test_lua_sidecars_pass_kit_sandbox_source_scan(self) -> None:
        # Kit's SPG Lua sandbox scans the raw source text, including
        # comments and assertion strings, for these substrings.
        for path in PPISP_SPG_DIR.glob("*.lua"):
            source = path.read_text(encoding="utf-8").lower()
            for pattern in _LUA_SANDBOX_FORBIDDEN_PATTERNS:
                self.assertNotIn(pattern, source, msg=f"{pattern!r} is forbidden in {path.name}")


if __name__ == "__main__":
    unittest.main()
