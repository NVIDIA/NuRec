# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest

import nurec.instant_nurec.datasets.datamodule as datamodule_module

from nurec.instant_nurec.config.dataset import DummyInstantNuRecDatasetConfig, InstantNuRecSplitsConfig
from nurec.instant_nurec.config.dataset import TestIndexInstantNuRecDatasetConfig as IndexDatasetConfig
from nurec.instant_nurec.datasets.datamodule import InstantNuRecPredictDataModule
from nurec.instant_nurec.datasets.instant_nurec_base import TestIndexInstantNuRecDataset as IndexDataset


@pytest.mark.parametrize("seed", [None, 29])
def test_predict_datamodule_forwards_optional_constructor_seed(
    monkeypatch: pytest.MonkeyPatch,
    seed: int | None,
) -> None:
    predict_config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)
    dummy_config = DummyInstantNuRecDatasetConfig(name="instant_nurec-dummy")
    splits = InstantNuRecSplitsConfig(
        name="instant_nurec",
        train=dummy_config,
        val=dummy_config,
        predict=predict_config,
    )
    config = SimpleNamespace(
        dataset=splits,
        system=SimpleNamespace(predict_num_workers=0, predict_batch_size=1),
    )
    calls: list[tuple[str, str, int | None]] = []

    def make_dataset(
        name: str,
        dataset_config: Any,
        split: str,
        *,
        seed: int | None = None,
    ) -> IndexDataset:
        calls.append((name, split, seed))
        return IndexDataset(dataset_config, split, seed=seed)

    monkeypatch.setattr(datamodule_module, "make_dataset", make_dataset)
    datamodule = InstantNuRecPredictDataModule(config, seed=seed)  # type: ignore[arg-type]

    datamodule.predict_dataloader()

    assert calls == [("instant_nurec-test-index", "predict", seed)]
