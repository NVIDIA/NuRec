# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:  # Needed to avoid a circular dependency due to type annotations
    from nurec.datamodules.base import BaseDataModule

datamodules: dict[str, Callable[..., "BaseDataModule"]] = {}


def register(name):
    def decorator(cls):
        datamodules[name] = cls
        return cls

    return decorator


def make(name, config, **kwargs) -> "BaseDataModule":
    datamodule = datamodules[name](config, **kwargs)
    return datamodule
