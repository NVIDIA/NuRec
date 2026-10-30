# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nurec.instant_nurec.config.dataset import InstantNuRecEpochSplitConfig, InstantNuRecSplitsConfig
from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig, InstantNuRecSystemConfig
from nurec.instant_nurec.config.models import CelsiusModelConfig


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "InstantNuRecSystemConfig",
    "InstantNuRecConfig",
    "InstantNuRecSplitsConfig",
    "InstantNuRecEpochSplitConfig",
    "CelsiusModelConfig",
]
