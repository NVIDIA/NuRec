# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from typing import TYPE_CHECKING, Optional, TypeVar

from nurec.instant_nurec.models import load_checkpoint_state_dict
from nurec.instant_nurec.systems.base import BaseInstantNuRecSystem
from nurec.instant_nurec.systems.gaussians_instant_nurec import GaussiansInstantNuRecSystem


if TYPE_CHECKING:
    from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig

logger = logging.getLogger(__name__)

SystemT = TypeVar("SystemT", bound=BaseInstantNuRecSystem)


def load_or_construct_system(
    system_cls: type[SystemT],
    config: "InstantNuRecConfig",
    load_from_checkpoint: Optional[str] = None,
) -> SystemT:
    """Construct a system, optionally loading its weights from a checkpoint.

    Shared by nurec.instant_nurec.systems.make and the equivalent `systems.make` in a separate
    downstream training package, since the checkpoint-loading logic is identical
    across the inference-only and training packages; only the name->system_cls
    dispatch differs between them.
    """
    if load_from_checkpoint is not None and config.resume_weights_only:
        system = system_cls(config)
        # NuRec Lightning checkpoints can contain trusted OmegaConf/internal
        # metadata that PyTorch's restricted weights-only unpickler rejects.
        load_checkpoint_state_dict(system, load_from_checkpoint, torch_load_weights_only=False)
        return system
    if load_from_checkpoint is not None:
        return system_cls.load_from_checkpoint(load_from_checkpoint, strict=True, config=config)

    return system_cls(config)


def make(
    name: str, config: "InstantNuRecConfig", load_from_checkpoint: Optional[str] = None
) -> "BaseInstantNuRecSystem":
    system_cls: type[BaseInstantNuRecSystem]
    if name == "base-instant_nurec-system":
        system_cls = GaussiansInstantNuRecSystem
    else:
        raise NotADirectoryError(f"Unknown instant_nurec system name: {name}")

    return load_or_construct_system(system_cls, config, load_from_checkpoint)


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "BaseInstantNuRecSystem",
    "GaussiansInstantNuRecSystem",
]
