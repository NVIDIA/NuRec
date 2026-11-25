# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure accumulator producers and return a lazily finalized loss vector."""

import torch

from nurec.datasets.base import BaseDataset
from nurec.instant_nurec.models.base import BaseInstantNuRecSupervisionPack
from nurec.instant_nurec.primitives.base import BaseInstantNuRecPrimitive
from nurec.libs.losses.functional.cuda_losses_accumulator import LOSS_SLOT_BY_NAME, N_LOSS_SLOTS
from nurec.libs.losses.models.accumulator_module import AccumulatorModuleLosses
from nurec.libs.losses.models.base_losses import BaseLoss
from nurec.libs.losses.models.render_losses import ground_rotation_weight_is_safe
from nurec.libs.losses.orchestration.config import LossAccumulatorReturn, LossConfig, LossItemConfig
from nurec.libs.losses.orchestration.loss_producers import (
    accumulate_independent_loss,
    configure_losses,
    initialize_losses,
    update_losses,
)
from nurec.models.base import BaseModel
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.profiling import ScopedTimer
from nurec.utils.trainer import TrainerConfig
from nurec.utils.types import GaussiansCompositeReturn


class LossAccumulator:
    """Own only the producers and modules used by the accumulator path."""

    def __init__(self, config: LossConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        force_disable_cuda: bool = kwargs.pop("force_disable_cuda", False)
        self.in_accumulator = AccumulatorModuleLosses()
        self.accumulator_losses, self.in_accumulator.losses = configure_losses(
            config, trainer_config, self._uses_fused, force_disable_cuda
        )

    def _uses_fused(self, full_name: str, config: LossItemConfig) -> bool:
        return full_name in self.in_accumulator.available and not (
            (full_name == "ssim_ssim_mean" and (bool(config.semantic_lambdas) or config.window_size != 11))
            or (full_name == "road_gaussians_abs_mean" and not ground_rotation_weight_is_safe(config.rotation_lambda))
        )

    @property
    def configured_losses(self) -> list[BaseLoss]:
        return self.accumulator_losses + self.in_accumulator.losses

    def initialize(self, train_dataset: BaseDataset) -> None:
        initialize_losses(self.configured_losses, train_dataset)

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        return update_losses(self.configured_losses, epoch, global_step, system, **kwargs)

    def __call__(
        self,
        *,
        step: int,
        model: BaseModel,
        results: GaussiansCompositeReturn | None = None,
        target: DataAndRenderingBatch | None = None,
        primitive: BaseInstantNuRecPrimitive | None = None,
        supervision_pack: BaseInstantNuRecSupervisionPack | None = None,
        context: DataAndRenderingBatch | None = None,
    ) -> LossAccumulatorReturn:
        """Return weighted CUDA slots and independent losses for lazy finalization."""
        if self.in_accumulator.losses:
            if results is None or target is None:
                raise ValueError("Fused accumulator losses require rendered results and target data.")
            with ScopedTimer("LossAccumulator/ModuleLosses"):
                vector, weights = self.in_accumulator.accumulate(step, model, results, target)
        else:
            vector = torch.zeros((N_LOSS_SLOTS,), dtype=torch.float32, device=model.device)
            weights = {}
        result = LossAccumulatorReturn(vector, LOSS_SLOT_BY_NAME, weights)
        for producer in self.accumulator_losses:
            if not producer.should_run_fn(step):
                continue
            with ScopedTimer(f"LossAccumulator/{producer.name}"):
                loss = accumulate_independent_loss(
                    producer,
                    model=model,
                    results=results,
                    target=target,
                    primitive=primitive,
                    supervision_pack=supervision_pack,
                    context=context,
                )
                if loss is not None:
                    result.append_producer_loss(loss)

        return result


__all__ = ["LossAccumulator"]
