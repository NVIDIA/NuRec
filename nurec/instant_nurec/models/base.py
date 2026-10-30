# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, Iterator, TypeVar

import torch
import torch.nn as nn

from nurec.datasets.tracks import CuboidTracks
from nurec.instant_nurec.config.models import BaseModelConfig
from nurec.instant_nurec.primitives.base import InstantNuRecPrimitiveType
from nurec.instant_nurec.supervision_pack import BaseInstantNuRecSupervisionPack
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.log import BatchMediaLogger


# Export BaseInstantNuRecSupervisionPack & InstantNuRecSupervisionPackType for compatibility.
# New imports should use nurec.instant_nurec.supervision_pack instead.
__all__ = ["BaseInstantNuRec", "BaseInstantNuRecSupervisionPack", "InstantNuRecSupervisionPackType"]


InstantNuRecSupervisionPackType = TypeVar("InstantNuRecSupervisionPackType", bound=BaseInstantNuRecSupervisionPack)


class BaseInstantNuRec(nn.Module, Generic[InstantNuRecPrimitiveType, InstantNuRecSupervisionPackType]):
    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__()
        self.config = config

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        """
        Hook function invoked at system.on_train_batch_start(), i.e. the very start of each train step.
        """
        return {}

    def on_train_from_scratch_start(self, system, **kwargs) -> None:
        """
        Hook function for system, invoked at system.on_train_start() only when system.resume is False.
        Useful for e.g. setting scalers, shape initialization process for sdf models, etc.
        """
        pass

    @torch.no_grad()
    def serialize_to_json_dict(self, with_state_dict: bool = True) -> dict[str, Any]:
        """
        Returns a model-config JSON dict consumed by GSplatRenderer.__init__.
        The renderer requires a renderable primitive at construction time, but the primitive
        is not available until the first feed-forward pass.  Subclasses override this to
        provide a placeholder primitive for renderer initialization.
        TODO [JH]: Use lazy initialization for BaseGaussianRenderer.factory
        """
        return {}

    @abstractmethod
    def reconstruct(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        media_logger: BatchMediaLogger | None,
        compute_supervision_pack: bool = False,
    ) -> tuple[list[InstantNuRecPrimitiveType], list[InstantNuRecSupervisionPackType] | None]:
        """
        Perform a "reconstruction" step on the provided images.
        Note that even if compute_supervision_pack is False, the supervision packs can still be returned if they're cheap (e.g. for Kelvin models)
        """
        pass

    @abstractmethod
    def prepare_supervision(
        self,
        context: list[DataAndRenderingBatch],
        supervision: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        supervision_packs: list[InstantNuRecSupervisionPackType],
        media_logger: BatchMediaLogger | None,
    ) -> tuple[list[DataAndRenderingBatch], list[InstantNuRecSupervisionPackType]]:
        """
        Hook function to prepare supervision data for the model.
        e.g. for the static celsius model we have to use cuboid tracks to generate the supervision valid mask.
        """
        pass

    @abstractmethod
    def prepare_context(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
    ) -> list[DataAndRenderingBatch]:
        """
        Hook function to prepare context data for the model.
        e.g. for the dynamic celsius model we have to compute the velocity for the context.
        """
        pass

    def get_potential_unused_parameters(self) -> Iterator[nn.Parameter]:
        """
        Return an iterator of potential unused parameters that can be used to sink the parameters during training.
        This is useful for avoiding unused parameters in DDP setting.
        """
        return iter([])
