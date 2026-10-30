# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nurec.datamodules.base as base

from nurec.datamodules.base import BaseDataModule, SODataModule
from nurec.datamodules.registry import make


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "base",
    "make",
    "BaseDataModule",
    "SODataModule",
]
