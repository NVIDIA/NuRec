# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:  # Needed to avoid a circular dependency due to type annotations
    from nurec.instant_nurec.datasets.instant_nurec_base import BaseInstantNuRecDataset

datasets: dict[str, Callable[..., "BaseInstantNuRecDataset"]] = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls

    return decorator


def make(name, config, split, *, seed: int | None = None) -> "BaseInstantNuRecDataset":
    """Construct a registered dataset while preserving omitted-seed behavior.

    When ``seed`` is ``None``, the registered callable is invoked as
    ``callable(config, split)`` for compatibility with legacy constructors.
    Constructors used with an explicit seed must accept the keyword-only
    ``seed`` argument and forward it to ``BaseInstantNuRecDataset.__init__``.
    """

    if seed is None:
        return datasets[name](config, split)
    dataset = datasets[name](config, split, seed=seed)
    if dataset._rng_seed != seed:
        raise RuntimeError(
            f"Dataset {name!r} accepted seed={seed} but did not forward it to BaseInstantNuRecDataset.__init__"
        )
    return dataset
