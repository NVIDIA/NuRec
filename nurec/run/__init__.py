# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys


# HACK: Remove project root from sys.path if it exists
# torchrun adds the project root to the python path, which
# breaks importing our cuda python modules.
# The importing of cuda modules should be made more robust.
# Note: this assumes that the project root is two dir levels above.
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
while project_root in sys.path:
    sys.path.remove(project_root)

from nurec.run.main import main  # noqa: E402
from nurec.run.profile_dataloader import profile_dataloader  # noqa: E402
from nurec.run.run_script import run_script  # noqa: E402


__all__ = [
    "main",
    "profile_dataloader",
    "run_script",
]
