# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

from typing import Any

# Pre-load dynamic torch dependencies, otherwise runtime-lookup will fail for torch-specific .so's
import torch  # noqa: F401


# Import the C++ compiled modules
slang_collector: Any = importlib.import_module("nurec.libs.slang_gaussians.libcollector_slang_cc")
gsplat_strategy_cuda: Any = importlib.import_module("nurec.libs.slang_gaussians.libgsplat_strategy_cuda_cc")

# Lazy import for Slang GSplat strategy (test-only; avoids SlangTorch build dep at runtime)
_gsplat_strategy_slang: Any = None

# Import the Python wrapper module that uses the C++ kernels
gsplat_strategy: Any = importlib.import_module("nurec.libs.slang_gaussians.strategy.gsplat")


def __getattr__(name: str) -> Any:
    global _gsplat_strategy_slang
    if name == "gsplat_strategy_slang":
        if _gsplat_strategy_slang is None:
            _gsplat_strategy_slang = importlib.import_module("nurec.libs.slang_gaussians.libgsplat_strategy_slang_cc")
        return _gsplat_strategy_slang
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
