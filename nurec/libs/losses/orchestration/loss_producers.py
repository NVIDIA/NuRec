# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared construction, lifecycle and invocation for the two loss paths."""

from collections.abc import Callable, Sequence
from typing import Any

import torch

import nurec.libs.losses.models.loss_fns  # noqa: F401
import nurec.libs.losses.models.primitive_losses  # noqa: F401
import nurec.libs.losses.models.render_losses  # noqa: F401

from nurec.datasets.base import BaseDataset
from nurec.instant_nurec.models.base import BaseInstantNuRecSupervisionPack
from nurec.instant_nurec.primitives.base import BaseInstantNuRecPrimitive
from nurec.libs.losses.models.base_losses import BaseLoss, BasePrimitiveLoss, BaseRenderLoss
from nurec.libs.losses.models.registry import make_loss
from nurec.libs.losses.orchestration.config import AccumulatorLoss, LossConfig, LossItemConfig, LossReturn
from nurec.models.base import BaseModel
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.trainer import TrainerConfig
from nurec.utils.types import GaussiansCompositeReturn


def configure_losses(
    config: LossConfig,
    trainer_config: TrainerConfig,
    use_fused: Callable[[str, LossItemConfig], bool],
    force_disable_cuda: bool,
) -> tuple[list[BaseLoss], list[BaseLoss]]:
    independent: list[BaseLoss] = []
    fused: list[BaseLoss] = []
    for loss_type, loss_config in config.items():
        loss_name = str(loss_type)
        full_name = f"{loss_name}_{loss_config.fn}_{loss_config.reduce.name}"
        loss = make_loss(loss_name, loss_config, trainer_config)
        destination = fused if not force_disable_cuda and use_fused(full_name, loss_config) else independent
        destination.append(loss)
    return independent, fused


def initialize_losses(losses: Sequence[BaseLoss], train_dataset: BaseDataset) -> None:
    for loss in losses:
        loss.initialize(train_dataset)


def update_losses(
    losses: Sequence[BaseLoss], epoch: int, global_step: int, system, **kwargs
) -> dict[str, torch.Tensor]:
    additional_params: dict[str, torch.Tensor] = {}
    for loss in losses:
        additional_params |= loss.update_step_train_batch_start(epoch, global_step, system, **kwargs)
    return additional_params


def independent_loss_kwargs(
    loss: BaseLoss,
    *,
    model: BaseModel,
    results: GaussiansCompositeReturn | None = None,
    target: DataAndRenderingBatch | None = None,
    primitive: BaseInstantNuRecPrimitive | None = None,
    supervision_pack: BaseInstantNuRecSupervisionPack | None = None,
    context: DataAndRenderingBatch | None = None,
) -> dict[str, Any]:
    """Prepare one producer's arguments using its existing input contract."""
    if isinstance(loss, BaseRenderLoss):
        assert results is not None and target is not None, f"Loss {loss.name} requires results and target!"
        return dict(results=results, target=target, model=model)
    if isinstance(loss, BasePrimitiveLoss):
        assert primitive is not None and context is not None and supervision_pack is not None, (
            f"Loss {loss.name} is a BasePrimitiveLoss and requires primitive, context, and supervision_pack!"
        )
        primitive_loss_kwargs: dict[str, Any] = dict(
            primitive=primitive, supervision_pack=supervision_pack, context=context, model=model
        )
        if loss.needs_target:
            primitive_loss_kwargs["target"] = target
        return primitive_loss_kwargs
    raise ValueError(f"Unknown loss type {type(loss)}")


def call_independent_loss(loss: BaseLoss, **kwargs: Any) -> LossReturn | None:
    return loss(**independent_loss_kwargs(loss, **kwargs))


def accumulate_independent_loss(loss: BaseLoss, **kwargs: Any) -> AccumulatorLoss | None:
    return loss.accumulate(**independent_loss_kwargs(loss, **kwargs))
