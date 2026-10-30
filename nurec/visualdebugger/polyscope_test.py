# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, call

import numpy as np
import pytest

import nurec.visualdebugger.polyscope as polyscope_module

from nurec.visualdebugger.interface import VisualDebugger
from nurec.visualdebugger.polyscope import PolyscopeVisualDebugger


def test_polyscope_backend_implements_visualdebugger() -> None:
    assert issubclass(PolyscopeVisualDebugger, VisualDebugger)
    assert not PolyscopeVisualDebugger.__abstractmethods__
    assert PolyscopeVisualDebugger.__module__ == "nurec.visualdebugger.polyscope"


def test_polyscope_initialization_is_lazy_and_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_polyscope = MagicMock()
    monkeypatch.setattr(polyscope_module, "ps", mock_polyscope)

    visualdebugger = PolyscopeVisualDebugger()

    assert visualdebugger._initialized is False
    mock_polyscope.assert_not_called()

    visualdebugger.show()

    assert mock_polyscope.mock_calls[:4] == [
        call.set_verbosity(0),
        call.set_allow_headless_backends(True),
        call.init(),
        call.show(),
    ]
    assert visualdebugger._initialized is True

    visualdebugger.clear()

    mock_polyscope.init.assert_called_once_with()
    mock_polyscope.remove_all_structures.assert_called_once_with()


def test_polyscope_initialization_failure_is_deferred_until_use(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_polyscope = MagicMock()
    mock_polyscope.init.side_effect = RuntimeError("no Polyscope backend is available")
    monkeypatch.setattr(polyscope_module, "ps", mock_polyscope)

    visualdebugger = PolyscopeVisualDebugger()

    mock_polyscope.init.assert_not_called()
    with pytest.raises(RuntimeError, match="no Polyscope backend is available"):
        visualdebugger.clear()
    assert visualdebugger._initialized is False


def test_polyscope_screenshot_api(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use upstream's deterministic backend; display/EGL availability is a host concern.
    init = polyscope_module.ps.init
    monkeypatch.setattr(polyscope_module.ps, "init", lambda: init("openGL_mock"))

    visualdebugger = PolyscopeVisualDebugger()
    visualdebugger.set_properties(window_size=(64, 48))
    visualdebugger.add_point_cloud("headless-test", np.array([[0.0, 0.0, 0.0]], dtype=np.float32))

    screenshot = visualdebugger.screenshot_to_buffer()

    assert screenshot.shape == (48, 64, 4)
    visualdebugger.clear()
