# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from pathlib import Path


__reporoot__ = (
    Path(__file__).parent.parent.resolve()
    if "BUILD_WORKING_DIRECTORY" not in os.environ
    else Path(os.environ["BUILD_WORKING_DIRECTORY"])
)
