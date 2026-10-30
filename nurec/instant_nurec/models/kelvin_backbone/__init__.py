# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nurec.instant_nurec.models.kelvin_backbone.base import KelvinInstantNuRecSupervisionPack
from nurec.instant_nurec.models.kelvin_backbone.decoders import make_decoder
from nurec.instant_nurec.models.kelvin_backbone.encoders import make_encoder
from nurec.instant_nurec.models.kelvin_backbone.sky import make_sky


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "KelvinInstantNuRecSupervisionPack",
    "make_decoder",
    "make_encoder",
    "make_sky",
]
