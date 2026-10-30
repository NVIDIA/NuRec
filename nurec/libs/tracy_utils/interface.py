# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracy profiler Python integration for NuRec - interface module for compiled extensions."""

import importlib

from typing import Any


# Import Tracy utils - the correct variant is provided by the Bazel target
# - pylib → tracy_utils_py.so (default CPU-only Tracy)
# - pylib_gpu → tracy_utils_gpu_py.so (full GPU Tracy)
# - pylib_disabled → tracy_utils_disabled_py.so (no Tracy symbols)
try:
    tracy_utils_py: Any = importlib.import_module("nurec.libs.tracy_utils.tracy_utils_py")

    TRACY_AVAILABLE = tracy_utils_py.is_available()
    PlotType = tracy_utils_py.PlotType if TRACY_AVAILABLE else None
except ImportError:
    try:
        tracy_utils_py: Any = importlib.import_module("nurec.libs.tracy_utils.tracy_utils_gpu_py")

        TRACY_AVAILABLE = tracy_utils_py.is_available()
        PlotType = tracy_utils_py.PlotType if TRACY_AVAILABLE else None
    except ImportError:
        try:
            tracy_utils_py: Any = importlib.import_module("nurec.libs.tracy_utils.tracy_utils_disabled_py")

            TRACY_AVAILABLE = tracy_utils_py.is_available()
            PlotType = tracy_utils_py.PlotType if TRACY_AVAILABLE else None
        except ImportError:
            # No Tracy variant available
            tracy_utils_py = None
            TRACY_AVAILABLE = False
            PlotType = None
