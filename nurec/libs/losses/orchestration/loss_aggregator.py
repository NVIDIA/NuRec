# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer 3: Orchestration - Centralized aggregator for all loss functions."""

import logging

from typing import Any

import torch

# Import loss modules to trigger their registration decorators.
# This ensures all losses are available when LossAggregator is used.
import nurec.libs.losses.models.loss_fns  # noqa: F401
import nurec.libs.losses.models.primitive_losses  # noqa: F401
import nurec.libs.losses.models.render_losses  # noqa: F401

from nurec.datasets.base import BaseDataset
from nurec.instant_nurec.models.base import BaseInstantNuRecSupervisionPack
from nurec.instant_nurec.primitives.base import BaseInstantNuRecPrimitive
from nurec.libs.losses.models.base_losses import BaseLoss, BasePrimitiveLoss, BaseRenderLoss
from nurec.libs.losses.models.losses_module import ModuleLosses
from nurec.libs.losses.models.registry import make_loss
from nurec.libs.losses.orchestration.config import LossAggregatorReturn, LossConfig, LossReturn
from nurec.models.base import BaseModel
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.profiling import ScopedTimer
from nurec.utils.trainer import TrainerConfig
from nurec.utils.types import GaussiansCompositeReturn


log = logging.getLogger(__name__)


class LossAggregator:
    """Centralized aggregator for all the loss functions."""

    losses: list[BaseLoss]
    in_cuda: ModuleLosses  # Fused loss integration; see nurec/libs/losses/README.md.

    def __init__(self, config: LossConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        # Optional override to disable all fused CUDA losses used for testing
        force_disable_cuda: bool = kwargs.pop("force_disable_cuda", False)
        self.losses = []
        self.in_cuda = ModuleLosses()
        for loss_type, loss_config in config.items():
            loss_name = str(loss_type)
            full_name = f"{loss_name}_{loss_config.fn}_{loss_config.reduce.name}"
            has_semantic_weights = bool(loss_config.semantic_lambdas)
            use_module_losses = full_name in self.in_cuda.available and not (
                full_name == "ssim_ssim_mean" and has_semantic_weights
            )
            if not force_disable_cuda and use_module_losses:
                self.in_cuda.losses.append(make_loss(loss_name, loss_config, trainer_config))
            else:
                self.losses.append(make_loss(loss_name, loss_config, trainer_config))

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

        if self.in_cuda.losses:
            with ScopedTimer("LossAggregator/ModuleLosses"):
                ret = self.in_cuda(step, model, results, target)

        for loss in self.losses:
            with ScopedTimer(f"LossAggregator/{loss.name}"):
                if loss.should_run_fn(step):
                    if isinstance(loss, BaseRenderLoss):
                        assert results is not None and target is not None, (
                            f"Loss {loss.name} requires results and target!"
                        )
                        loss_ret = loss(results=results, target=target, model=model)

                    elif isinstance(loss, BasePrimitiveLoss):
                        assert primitive is not None and context is not None and supervision_pack is not None, (
                            f"Loss {loss.name} is a BasePrimitiveLoss and requires primitive, context, and supervision_pack!"
                        )
                        primitive_loss_kwargs: dict[str, Any] = dict(
                            primitive=primitive,
                            supervision_pack=supervision_pack,
                            context=context,
                            model=model,
                        )
                        if loss.needs_target:
                            primitive_loss_kwargs["target"] = target
                        loss_ret = loss(**primitive_loss_kwargs)

                    else:
                        raise ValueError(f"Unknown loss type {type(loss)}")

                    if loss_ret is not None:
                        ret[loss.name] = loss_ret

        with ScopedTimer("LossAggregator/reduce"):
            return LossAggregatorReturn(ret)

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        additional_params: dict[str, torch.Tensor] = {}
        for loss in self.losses + self.in_cuda.losses:
            additional_params |= loss.update_step_train_batch_start(epoch, global_step, system, **kwargs)
        return additional_params

    def initialize(self, train_dataset: BaseDataset) -> None:
        for loss in self.losses + self.in_cuda.losses:
            loss.initialize(train_dataset)


__all__ = ["LossAggregator"]
