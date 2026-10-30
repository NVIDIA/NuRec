# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from typing import cast

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from nurec.instant_nurec.config.dataset import InstantNuRecSplitsConfig
from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig
from nurec.instant_nurec.datasets.instant_nurec_base import BaseInstantNuRecDataset
from nurec.instant_nurec.datasets.registry import make as make_dataset
from nurec.utils.batch import InstantNuRecDataBatch
from nurec.utils.misc import unpack_optional


log = logging.getLogger(__name__)


class InstantNuRecPredictDataModule(LightningDataModule):
    """
    Inference-only datamodule: builds just the predict dataloader. Training/validation/test
    dataloaders, mid-epoch resume, and skip-batch sampling live on `InstantNuRecDataModule` in a
    separate downstream training package, which subclasses this.
    """

    instant_nurec_config: InstantNuRecConfig
    dataset_config: InstantNuRecSplitsConfig

    predict_dataset: BaseInstantNuRecDataset | None = None

    def __init__(self, instant_nurec_config: InstantNuRecConfig, *, seed: int | None = None) -> None:
        """Construct a datamodule with an optional explicit dataset seed.

        Each dataloader forwards ``seed`` to its dataset. ``None`` preserves
        the existing lazy ``PL_GLOBAL_SEED`` behavior.
        """
        super().__init__()
        self.instant_nurec_config = instant_nurec_config
        self._dataset_seed = seed
        assert isinstance(instant_nurec_config.dataset, InstantNuRecSplitsConfig)
        self.dataset_config = instant_nurec_config.dataset

    def predict_dataloader(self) -> DataLoader:
        """
        Returns a predict dataloader
        """
        dataset_config = self.dataset_config.predict
        assert dataset_config is not None, "dataset.predict has to be specified in the config to use the predict mode"

        self.predict_dataset = cast(
            BaseInstantNuRecDataset,
            make_dataset(dataset_config.name, dataset_config, "predict", seed=self._dataset_seed),
        )

        if self.trainer is not None and self.predict_dataset is not None:
            # Not calling set_rng_epoch here, as predict RNGs should not depend on epoch.
            self.predict_dataset.set_epoch(self.trainer.current_epoch)

        # Dataset is guaranteed to be loaded in the main process sequentially, typically with batch size 1.
        return DataLoader(
            unpack_optional(self.predict_dataset),
            num_workers=self.instant_nurec_config.system.predict_num_workers,
            persistent_workers=False,
            batch_size=self.instant_nurec_config.system.predict_batch_size,
            pin_memory=True,
            collate_fn=InstantNuRecDataBatch.collate_fn,
        )
