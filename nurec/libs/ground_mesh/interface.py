# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python entry point for the nurec/libs/ground_mesh CUDA extension."""

import importlib

from typing import Any

# Pre-load dynamic torch dependencies before importing the .so, otherwise
# torch's runtime symbol lookup can fail when the extension is imported first.
import torch  # noqa: F401  (kept for the side-effect of loading libtorch)


ground_mesh: Any = importlib.import_module("nurec.libs.ground_mesh.libground_mesh_cc")
