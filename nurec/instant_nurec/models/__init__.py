# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nurec.instant_nurec.config.models import CelsiusModelConfig, KelvinModelConfig
from nurec.instant_nurec.models.base import BaseInstantNuRec
from nurec.instant_nurec.models.celsius_model import CelsiusInstantNuRec
from nurec.instant_nurec.models.checkpoint import load_checkpoint_state_dict
from nurec.instant_nurec.models.kelvin_model import KelvinInstantNuRec


def make_instant_nurec_model(config: CelsiusModelConfig | KelvinModelConfig) -> BaseInstantNuRec:
    """Build the instant_nurec model selected by its typed configuration."""

    if isinstance(config, CelsiusModelConfig):
        return CelsiusInstantNuRec(config)
    if isinstance(config, KelvinModelConfig):
        return KelvinInstantNuRec(config)
    raise TypeError(f"Unsupported instant_nurec model config {type(config).__name__}.")


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "BaseInstantNuRec",
    "CelsiusInstantNuRec",
    "KelvinInstantNuRec",
    "load_checkpoint_state_dict",
    "make_instant_nurec_model",
]
