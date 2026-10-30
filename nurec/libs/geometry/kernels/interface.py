# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

from typing import Any

# Pre-load dynamic torch dependencies, otherwise runtime-lookup will fail for torch-specific .so's
import torch  # noqa: F401


libquaternion_slang_cc: Any = importlib.import_module("nurec.libs.geometry.kernels.libquaternion_slang_cc")
libpose_kernels_slang_cc: Any = importlib.import_module("nurec.libs.geometry.kernels.libpose_kernels_slang_cc")
