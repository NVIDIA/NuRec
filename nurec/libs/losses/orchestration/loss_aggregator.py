# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer 3: Orchestration - Centralized aggregator for all loss functions."""

import logging

import torch

from nurec.datasets.base import BaseDataset
from nurec.instant_nurec.models.base import BaseInstantNuRecSupervisionPack
from nurec.instant_nurec.primitives.base import BaseInstantNuRecPrimitive
from nurec.libs.losses.models.base_losses import BaseLoss
from nurec.libs.losses.orchestration.config import LossAggregatorReturn, LossConfig, LossReturn
from nurec.libs.losses.orchestration.loss_producers import (
    call_independent_loss,
    configure_losses,
    initialize_losses,
    update_losses,
)
from nurec.models.base import BaseModel
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.profiling import ScopedTimer
from nurec.utils.trainer import TrainerConfig
from nurec.utils.types import GaussiansCompositeReturn


log = logging.getLogger(__name__)


class LossAggregator:
    """Centralized aggregator for all the loss functions."""

    losses: list[BaseLoss]

    def __init__(self, config: LossConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        self.losses, _ = configure_losses(config, trainer_config, lambda *_: False, True)

    @property
    def configured_losses(self) -> list[BaseLoss]:
        return self.losses

    def __call__(
        self,
        *,  # Enforce keyword arguments
        step: int,
        model: BaseModel,
        results: GaussiansCompositeReturn | None = None,
        target: DataAndRenderingBatch | None = None,
        primitive: BaseInstantNuRecPrimitive | None = None,
        supervision_pack: BaseInstantNuRecSupervisionPack | None = None,
        context: DataAndRenderingBatch | None = None,
    ) -> LossAggregatorReturn:
        ret: dict[str, LossReturn] = {}

        for loss in self.losses:
            with ScopedTimer(f"LossAggregator/{loss.name}"):
                if loss.should_run_fn(step):
                    loss_ret = call_independent_loss(
                        loss,
                        model=model,
                        results=results,
                        target=target,
                        primitive=primitive,
                        supervision_pack=supervision_pack,
                        context=context,
                    )

                    if loss_ret is not None:
                        ret[loss.name] = loss_ret

        with ScopedTimer("LossAggregator/reduce"):
            return LossAggregatorReturn(ret)

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        return update_losses(self.losses, epoch, global_step, system, **kwargs)

    def initialize(self, train_dataset: BaseDataset) -> None:
        initialize_losses(self.losses, train_dataset)


__all__ = ["LossAggregator"]
