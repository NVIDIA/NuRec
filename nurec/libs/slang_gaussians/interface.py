# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

from typing import Any

# Pre-load dynamic torch dependencies, otherwise runtime-lookup will fail for torch-specific .so's
import torch  # noqa: F401


# Import the C++ compiled modules
slang_collector: Any = importlib.import_module("nurec.libs.slang_gaussians.libcollector_slang_cc")
