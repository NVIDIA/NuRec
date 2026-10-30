# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight PPISP SPG sidecar selection helpers."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from nurec.utils.io.export.ppisp_export.controller.writer import (
    get_ppisp_auto_spg_files,
    get_ppisp_embedded_controller_spg_files,
)
from nurec.utils.io.export.ppisp_export.usd_writer import get_ppisp_spg_files
from nurec.utils.types import NamedSerialized


PPISP_INTEGRATION_MODE_SPG_RUNTIME = "spg-runtime"


def ppisp_has_controller(ppisp_module: Any | None) -> bool:
    """Return whether the loaded PPISP module exposes trained controllers."""
    if ppisp_module is None:
        return False

    use_controller = getattr(getattr(ppisp_module, "config", None), "use_controller", None)
    if use_controller is not None and not bool(use_controller):
        return False

    controllers = getattr(ppisp_module, "controllers", None)
    if controllers is None:
        return False

    try:
        return len(controllers) > 0
    except TypeError:
        return False


def resolve_ppisp_controller_export_enabled(
    *,
    requested: Optional[bool],
    ppisp_module: Any | None,
    ppisp_integration_mode: str,
) -> bool:
    """Resolve the tri-state controller export setting."""
    if ppisp_integration_mode != PPISP_INTEGRATION_MODE_SPG_RUNTIME:
        return False
    if requested is not None:
        return requested
    return ppisp_has_controller(ppisp_module)


def select_spg_files_for_export(
    *,
    enable_ppisp_controller_export: bool,
    ppisp_module: Any | None = None,
    camera_indices: Sequence[int] | None = None,
) -> List[NamedSerialized]:
    """Choose which SPG sidecars to ship alongside the USD asset."""
    if not enable_ppisp_controller_export:
        if ppisp_module is not None or camera_indices is not None:
            raise ValueError(
                "ppisp_module and camera_indices must be omitted when enable_ppisp_controller_export=False"
            )
        return list(get_ppisp_spg_files())
    if ppisp_module is None or camera_indices is None:
        raise ValueError("ppisp_module and camera_indices are required when enable_ppisp_controller_export=True")
    files: List[NamedSerialized] = []
    deduped_camera_indices = list(dict.fromkeys(camera_indices))
    files.extend(get_ppisp_embedded_controller_spg_files(ppisp_module, deduped_camera_indices))
    files.extend(get_ppisp_auto_spg_files())
    return files
