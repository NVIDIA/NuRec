# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:  # Needed to avoid a circular dependency due to type annotations
    from nurec.config.trainer import TrainerConfig
    from nurec.libs.losses.models.lambda_schedulers.lambda_schedulers import BaseLambdaScheduler

lambda_schedulers: dict[str, Callable[..., "BaseLambdaScheduler"]] = {}


def register(name: str):
    def decorator(cls):
        lambda_schedulers[name] = cls
        return cls

    return decorator


def make(name, config, trainer_config: "TrainerConfig", lambda_init, **kwargs) -> "BaseLambdaScheduler":
    lambda_scheduler_class = lambda_schedulers[name]
    return lambda_scheduler_class(config, trainer_config, lambda_init, **kwargs)
