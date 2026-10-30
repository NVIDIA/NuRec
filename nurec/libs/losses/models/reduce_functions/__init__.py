# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nurec.libs.losses.models.reduce_functions.reduce_functions as reduce_functions

from nurec.libs.losses.models.reduce_functions.reduce_functions import MeanReduceFn, QuantileMeanReduceFn, SumReduceFn
from nurec.libs.losses.models.reduce_functions.registry import make as make_reduce_fn


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "reduce_functions",
    "make_reduce_fn",
    "MeanReduceFn",
    "QuantileMeanReduceFn",
    "SumReduceFn",
]
