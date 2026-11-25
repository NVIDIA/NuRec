# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent ownership and shared producer lifecycle contracts."""

from types import SimpleNamespace
from unittest import mock

import pytest

from nurec.libs.losses.orchestration import loss_producers
from nurec.libs.losses.orchestration.config import LossConfig
from nurec.libs.losses.orchestration.loss_accumulator import LossAccumulator
from nurec.libs.losses.orchestration.loss_aggregator import LossAggregator


@pytest.mark.parametrize("owner", [LossAggregator, LossAccumulator])
def test_paths_construct_only_their_own_modules_and_run_lifecycle_once(owner):
    config = LossConfig.model_validate(
        {
            "rgb": {"lambda_": 1.0, "fn": "l1", "reduce": {"name": "mean"}},
            "psnr": {"lambda_": 0.0, "fn": "psnr", "reduce": {"name": "mean"}},
        }
    )
    producers = [mock.Mock(name=name) for name in ("rgb", "psnr")]
    for producer in producers:
        producer.update_step_train_batch_start.return_value = {}
    modules = SimpleNamespace(available=["rgb_l1_mean"], losses=[])
    with (
        mock.patch(
            "nurec.libs.losses.orchestration.loss_accumulator.AccumulatorModuleLosses",
            return_value=modules,
        ) as constructed,
        mock.patch.object(loss_producers, "make_loss", side_effect=producers) as made,
    ):
        instance = owner(config, SimpleNamespace())
    if owner is LossAccumulator:
        constructed.assert_called_once_with()
        assert modules.losses == [producers[0]]
    else:
        constructed.assert_not_called()
        assert instance.losses == producers
    assert made.call_count == 2
    assert hasattr(instance, "in_accumulator") == (owner is LossAccumulator)
    dataset, system = object(), object()
    instance.initialize(dataset)
    assert instance.update_step_train_batch_start(3, 17, system, extra=True) == {}
    for producer in producers:
        producer.initialize.assert_called_once_with(dataset)
        producer.update_step_train_batch_start.assert_called_once_with(3, 17, system, extra=True)
