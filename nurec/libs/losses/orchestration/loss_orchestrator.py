# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    import torch

    from nurec.datasets.base import BaseDataset
    from nurec.instant_nurec.models.base import BaseInstantNuRecSupervisionPack
    from nurec.instant_nurec.primitives.base import BaseInstantNuRecPrimitive
    from nurec.libs.losses.models.base_losses import BaseLoss
    from nurec.libs.losses.orchestration.config import LossResult
    from nurec.models.base import BaseModel
    from nurec.utils.batch import DataAndRenderingBatch
    from nurec.utils.types import GaussiansCompositeReturn


class LossOrchestrator(Protocol):
    """Producer lifecycle and invocation shared by both loss computation paths."""

    @property
    def configured_losses(self) -> list[BaseLoss]: ...

    def initialize(self, train_dataset: BaseDataset) -> None: ...

    def update_step_train_batch_start(
        self, epoch: int, global_step: int, system: Any, **kwargs: Any
    ) -> dict[str, torch.Tensor]: ...

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
    ) -> LossResult: ...
