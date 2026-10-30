# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast

from pathlib import Path

from python.runfiles import runfiles


_SOURCE_PATHS = (
    "nurec/instant_nurec/models/blocks/dpt.py",
    "nurec/instant_nurec/models/blocks/embeds.py",
    "nurec/instant_nurec/models/blocks/layers.py",
    "nurec/instant_nurec/models/kelvin_model.py",
    "nurec/instant_nurec/primitives/kelvin_primitive.py",
    "nurec/instant_nurec/systems/base.py",
)


def test_runtime_assignments_are_not_inside_assertions() -> None:
    """Assignments needed at runtime must survive ``python -O``."""

    runfiles_dir = runfiles.Create()
    offending_assertions: dict[str, list[str]] = {}
    for source_path in _SOURCE_PATHS:
        resolved_path = runfiles_dir.Rlocation(f"_main/{source_path}")
        assert resolved_path is not None
        tree = ast.parse(Path(resolved_path).read_text())
        assertions = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assert) and any(isinstance(child, ast.NamedExpr) for child in ast.walk(node.test))
        ]
        if assertions:
            offending_assertions[source_path] = assertions

    assert offending_assertions == {}
