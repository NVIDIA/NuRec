# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:  # Needed to avoid a circular dependency due to type annotations
    from nurec.libs.losses.orchestration.config import RayReduceFn

reduce_fns: dict[str, Callable[..., "RayReduceFn"]] = {}


def make(name: str, config):
    return reduce_fns[name](config)


def register(name: str):
    def decorator(fn: Callable[..., "RayReduceFn"]):
        reduce_fns[name] = fn
        return fn

    return decorator
