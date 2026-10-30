# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

from typing import Any

# Pre-load dynamic torch dependencies before importing the extension.
import torch  # noqa: F401


gaussian_collective_payload_ops: Any = importlib.import_module(
    "nurec.libs.gaussian_collective_payload_ops.libgaussian_collective_payload_ops_cc"
)
