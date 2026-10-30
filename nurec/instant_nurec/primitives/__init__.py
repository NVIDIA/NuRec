# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nurec.instant_nurec.primitives.base import BaseGaussiansInstantNuRecPrimitive
from nurec.instant_nurec.primitives.celsius_primitive import CelsiusInstantNuRecPrimitive
from nurec.instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "BaseGaussiansInstantNuRecPrimitive",
    "CelsiusInstantNuRecPrimitive",
    "KelvinInstantNuRecPrimitive",
]
