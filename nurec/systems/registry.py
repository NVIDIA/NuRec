# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch


# Safeguard to avoid a circular dependency due to type annotations
# in case a system inside .base is registered by using the decorator @register.
if TYPE_CHECKING:
    from nurec.config.nurec import NuRecConfig
    from nurec.systems.base import BaseSystem
    from nurec.systems.gaussians import InProcessInstantNuRecInitializer


systems: dict[str, BaseSystem] = {}


def make(
    name: str,
    config: NuRecConfig,
    load_from_checkpoint: Optional[str] = None,
    in_process_instant_nurec_initializer: InProcessInstantNuRecInitializer | None = None,
) -> BaseSystem:
    """Construct or restore a system with its optional setup-time instant_nurec producer.

    The factory only forwards ``in_process_instant_nurec_initializer``. The constructed
    system stores it per instance and invokes it later from ``setup()``.
    """

    if load_from_checkpoint is not None:
        # Load the model and force it to be on the GPU.
        # Usually checkpoints are saved with the model on the CPU,
        # so we need to move it to the GPU, as that's what callers expect.
        # Without it, we might run into issues where tensors in different devices are
        # used in the same torch operation.
        return systems[name].load_from_checkpoint(
            load_from_checkpoint,
            strict=False,
            config=config,
            map_location=torch.device("cuda"),
            in_process_instant_nurec_initializer=in_process_instant_nurec_initializer,
        )

    return systems[name](config, in_process_instant_nurec_initializer=in_process_instant_nurec_initializer)


def register(name: str):
    def decorator(cls):
        systems[name] = cls
        return cls

    return decorator
