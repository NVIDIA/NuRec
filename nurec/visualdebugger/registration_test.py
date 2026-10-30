# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from nurec.visualdebugger import VisualDebugger, get_visualdebugger


def test_visualdebugger_missing_backend_import_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "nurec.visualdebugger.polyscope", None)  # type: ignore[arg-type]

    with pytest.raises(ModuleNotFoundError, match="nurec.visualdebugger.polyscope"):
        get_visualdebugger()


def test_visualdebugger_loads_public_backend_on_first_use() -> None:
    assert "nurec.visualdebugger.polyscope" not in sys.modules

    visualdebugger = get_visualdebugger()

    assert "nurec.visualdebugger.polyscope" in sys.modules
    assert isinstance(visualdebugger, VisualDebugger)
    assert get_visualdebugger() is visualdebugger

    from nurec.visualdebugger.polyscope import PolyscopeVisualDebugger

    assert type(visualdebugger) is PolyscopeVisualDebugger
