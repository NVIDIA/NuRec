# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

from typing import Any


ray_utils: Any = importlib.import_module("nurec.libs.ray_utils.libray_utils_cc")
