# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from torch import nn
from tqdm import tqdm

from nurec.datasets.tracks import CuboidTracks
from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig
from nurec.instant_nurec.config.models import PrimitiveExportPreprocessConfig
from nurec.instant_nurec.config.predict import PredictConfig
from nurec.instant_nurec.models import load_checkpoint_state_dict, make_instant_nurec_model
from nurec.instant_nurec.models.base import BaseInstantNuRec, BaseInstantNuRecSupervisionPack
from nurec.instant_nurec.predict.primitive_merge import make as make_primitive_merge
from nurec.instant_nurec.primitives.base import BaseInstantNuRecPrimitive
from nurec.utils.batch import InstantNuRecDataBatch
from nurec.utils.log import BatchMediaLogger
from nurec.utils.types import GaussiansCompositeReturn


@dataclass(slots=True)
class InstantNuRecPredictionOutput:
    """Framework-neutral result of one instant_nurec prediction batch."""

    primitives: list[BaseInstantNuRecPrimitive]
    batch: InstantNuRecDataBatch


InstantNuRecForwardChunk = Callable[
    [InstantNuRecDataBatch],
    tuple[list[BaseInstantNuRecPrimitive], list[BaseInstantNuRecSupervisionPack], list[GaussiansCompositeReturn]],
]


def reconstruct_instant_nurec_context(
    *,
    model: BaseInstantNuRec,
    batch: InstantNuRecDataBatch,
    media_logger: BatchMediaLogger | None,
    compute_supervision_pack: bool,
) -> tuple[list[BaseInstantNuRecPrimitive], list[BaseInstantNuRecSupervisionPack] | None, list[CuboidTracks] | None]:
    """Prepare context and reconstruct primitives through the shared instant_nurec model API."""

    cuboid_tracks = None
    if batch.cuboid_tracks is not None:
        cuboid_tracks = [CuboidTracks.Factory.from_pack(pack) for pack in batch.cuboid_tracks]
    batch.context = model.prepare_context(batch.context, cuboid_tracks)
    primitives, supervision_packs = model.reconstruct(
        batch.context,
        cuboid_tracks,
        media_logger=media_logger,
        compute_supervision_pack=compute_supervision_pack,
    )
    return primitives, supervision_packs, cuboid_tracks


def run_instant_nurec_prediction(
    *,
    batch: InstantNuRecDataBatch,
    device: torch.device,
    export_preprocess_config: PrimitiveExportPreprocessConfig,
    predict_config: PredictConfig,
    forward_chunk: InstantNuRecForwardChunk,
) -> InstantNuRecPredictionOutput:
    """Run the model, export preprocessing, and primitive merge without a training framework."""

    if predict_config.chunk_size <= 0:
        raise ValueError(f"predict.chunk_size must be positive, got {predict_config.chunk_size}")

    # In the future maybe rendering data is not required any more for model forwarding.
    batch.maybe_compute_rendering_data(device=device)

    # Note that we don't guarantee that batch.supervision is not None here.

    # For large batch sizes, process in chunks.
    primitives_list: list[BaseInstantNuRecPrimitive] = []

    inner_batch_idx = 0
    progress_bar = tqdm(total=len(batch), desc="Predicting in chunks")
    while inner_batch_idx < len(batch):
        batch_chunk = batch[inner_batch_idx : inner_batch_idx + predict_config.chunk_size]
        primitives_chunk_list, supervision_packs_chunk, _ = forward_chunk(batch_chunk)

        if export_preprocess_config.enabled:
            context_rig_list = batch_chunk.context_rig
            for i in range(len(primitives_chunk_list)):
                context_rig_i = context_rig_list[i] if context_rig_list is not None else None
                supervision_pack_i = supervision_packs_chunk[i] if i < len(supervision_packs_chunk) else None
                primitives_chunk_list[i] = primitives_chunk_list[i].preprocess_for_export(
                    batch_chunk.context[i],
                    export_preprocess_config,
                    context_rig=context_rig_i,
                    supervision_pack=supervision_pack_i,
                )

        primitives_list.extend(primitives_chunk_list)
        inner_batch_idx += predict_config.chunk_size
        progress_bar.update(predict_config.chunk_size)
    progress_bar.close()

    # Merge the primitives if enabled.
    if predict_config.primitive_merge.enabled:
        primitive_merge = make_primitive_merge(
            type(primitives_list[0]),
            predict_config.primitive_merge,
            export_preprocess_config,
        )
        merged_primitive, batch = primitive_merge.merge_primitives_and_batch(primitives_list, batch)
        primitives_list = [merged_primitive]

    # Release memory if possible.
    gc.collect()
    torch.cuda.empty_cache()

    return InstantNuRecPredictionOutput(primitives=primitives_list, batch=batch)


class InstantNuRecPredictionRunner(nn.Module):
    """Inference-only instant_nurec module with no Lightning trainer lifecycle.

    Callers own device transfer and inference/autocast contexts. In particular,
    ``config.system.precision`` is not applied automatically; callers must enter
    the corresponding ``torch.autocast`` context before invoking ``forward``.
    """

    model: BaseInstantNuRec

    def __init__(self, config: InstantNuRecConfig, *, model: BaseInstantNuRec | None = None) -> None:
        super().__init__()
        self.model = make_instant_nurec_model(config.model) if model is None else model
        self.export_preprocess_config = config.model.export_preprocess
        self.predict_config = config.predict

    @property
    def device(self) -> torch.device:
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.model.buffers(), None)
        if buffer is not None:
            return buffer.device
        raise RuntimeError("instant_nurec prediction model has no parameters or buffers from which to infer its device")

    def load_weights_only_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Strictly load a raw or Lightning-wrapped weights-only ``model.*`` state dict."""

        # The runner's sole child is named ``model``, matching the checkpoint
        # namespace written by BaseInstantNuRecSystem without key rewriting.
        load_checkpoint_state_dict(self, checkpoint_path, torch_load_weights_only=True)

    def forward(self, batch: InstantNuRecDataBatch) -> InstantNuRecPredictionOutput:
        if batch.supervision is not None:
            raise ValueError("Standalone instant_nurec prediction supports context-only batches without supervision")
        return run_instant_nurec_prediction(
            batch=batch,
            device=self.device,
            export_preprocess_config=self.export_preprocess_config,
            predict_config=self.predict_config,
            forward_chunk=self._forward_chunk,
        )

    def prepare_prediction(self) -> None:
        """Enter evaluation mode and apply the step-zero hook before inference/autocast."""

        self.eval()
        self.model.update_step_train_batch_start(epoch=0, global_step=0, system=None)

    def _forward_chunk(
        self, batch: InstantNuRecDataBatch
    ) -> tuple[list[BaseInstantNuRecPrimitive], list[BaseInstantNuRecSupervisionPack], list[GaussiansCompositeReturn]]:
        primitives, supervision_packs, _ = reconstruct_instant_nurec_context(
            model=self.model,
            batch=batch,
            media_logger=None,
            compute_supervision_pack=False,
        )
        return primitives, [] if supervision_packs is None else supervision_packs, []
