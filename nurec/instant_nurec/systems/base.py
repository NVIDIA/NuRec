# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from abc import ABC

import torch

from pytorch_lightning import LightningModule

from nurec.config.base_schema import config_to_primitive
from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig, InstantNuRecSystemConfig
from nurec.instant_nurec.datasets.datamodule import InstantNuRecPredictDataModule
from nurec.instant_nurec.models.base import BaseInstantNuRec
from nurec.utils.batch import InstantNuRecDataBatch
from nurec.utils.log import BatchMediaLogger
from nurec.utils.types import Checkpoint


logger = logging.getLogger(__name__)


class BaseInstantNuRecSystem(LightningModule, ABC):
    """
    Inference-only instant_nurec system: builds the model/datamodule and supports Lightning's
    `predict` loop. Training-loop methods (`training_step`, `configure_optimizers`,
    validation/test loops, loss computation) live on `TrainBaseInstantNuRecSystem` in a
    separate downstream training package, which subclasses this.
    """

    config: InstantNuRecSystemConfig
    model: BaseInstantNuRec
    datamodule: InstantNuRecPredictDataModule

    device: torch.device  # mypy type fix for obtaininig system's device

    def __init__(self, config: InstantNuRecConfig) -> None:
        super().__init__()

        self.save_hyperparameters(config_to_primitive(config.to_dictconfig()))

        # save what we need from the config
        self.cached_config = config
        self.mode = config.mode
        self.out_dir = config.out_dir
        self.run_id = config.run_id
        self.resume = config.resume
        self.config = config.system
        self.predict_config = config.predict

        # `_build_datamodule` runs before this subclass's own `__init__` body (the code
        # after its `super().__init__()` call), see the hook's docstring for details.
        self.datamodule = self._build_datamodule(config)

        self.media_logger = BatchMediaLogger(self, self.config)

    def _build_datamodule(self, config: InstantNuRecConfig) -> InstantNuRecPredictDataModule:
        """
        Hook for subclasses to construct their datamodule. Overridden by `TrainBaseInstantNuRecSystem`
        to build a training-capable `InstantNuRecDataModule` instead, so only one datamodule is ever
        constructed (rather than building the predict-only one here and immediately replacing it).

        Warning: this is called from within `BaseInstantNuRecSystem.__init__`, before the subclass's
        own `__init__` body (the code after its `super().__init__()` call) has executed.
        Overrides must not read `self` attributes that the subclass's own `__init__` sets
        (e.g. anything derived from `self.model`, which `GaussiansInstantNuRecSystem.__init__` only
        assigns after its `super().__init__()` call returns) — they will not exist yet.
        """
        return InstantNuRecPredictDataModule(config)

    # ---- Predict loop methods ----

    def on_predict_batch_start(
        self, batch: InstantNuRecDataBatch, batch_local_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.model.update_step_train_batch_start(self.current_epoch, self.global_step, self)

    # ---- Checkpoint-related methods ----

    def on_load_checkpoint(self, checkpoint: Checkpoint) -> None:
        """
        The issue here is that the buffers can change in size so we cannot initialize them in advance and load them directly from the checkpoint
        """
        if self._trainer is not None:
            # Manually load all the loops as otherwise global_step and current epoch are not set correctly in the validation mode
            # see: https://github.com/Lightning-AI/lightning/issues/17127
            self.trainer.fit_loop.load_state_dict(checkpoint["loops"]["fit_loop"])
            self.trainer.validate_loop.load_state_dict(checkpoint["loops"]["validate_loop"])
            self.trainer.test_loop.load_state_dict(checkpoint["loops"]["test_loop"])
            self.trainer.predict_loop.load_state_dict(checkpoint["loops"]["predict_loop"])

        # Load the remainder of the state_dict
        self.load_state_dict(checkpoint["state_dict"], assign=True)
