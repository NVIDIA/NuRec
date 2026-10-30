# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nurec.libs.losses.models.lambda_schedulers.lambda_schedulers as lambda_schedulers

from nurec.libs.losses.models.lambda_schedulers.lambda_schedulers import BaseLambdaScheduler, LinearLambdaScheduler
from nurec.libs.losses.models.lambda_schedulers.registry import make


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "lambda_schedulers",
    "make",
    "BaseLambdaScheduler",
    "LinearLambdaScheduler",
]
