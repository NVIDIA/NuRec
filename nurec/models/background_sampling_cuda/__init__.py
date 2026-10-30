# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

from typing import Any

import torch  # noqa: F401  # Pre-load dynamic torch dependencies for the extension.


background_sampling_cuda: Any = importlib.import_module(
    "nurec.models.background_sampling_cuda.libbackground_sampling_cuda_cc"
)
