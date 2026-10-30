# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:  # Needed to avoid a circular dependency due to type annotations
    from nurec.datasets.base import BaseDataset

datasets: dict[str, Callable[..., "BaseDataset"]] = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls

    return decorator


def make(name, config, split) -> "BaseDataset":
    # ncore dataset need some values from other parts of the config.
    # In the long run we should move towards this not being the case.
    if name == "ncore":
        dataset = datasets[name](config, split)
    else:
        dataset = datasets[name](config.dataset, split)

    return dataset
