# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pytorch_lightning.strategies.strategy as strategy_module
import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.plugins import TorchCheckpointIO
from pytorch_lightning.plugins.precision import Precision
from pytorch_lightning.trainer.states import TrainerFn
from torch.nn.parallel import DistributedDataParallel

from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.run.main import _DDPDevicePreservingStrategySO, _SingleDeviceStrategySO


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("stage", [TrainerFn.FITTING, TrainerFn.VALIDATING])
def test_teardown_preserves_devices_and_releases_resources(monkeypatch, distributed, stage):
    """Skip model/optimizer offload while retaining DDP and plugin cleanup."""
    events = []
    accelerator, precision, checkpoint_io = CUDAAccelerator(), Precision(), TorchCheckpointIO()
    for name, plugin in [("precision", precision), ("accelerator", accelerator), ("checkpoint", checkpoint_io)]:
        monkeypatch.setattr(plugin, "teardown", lambda name=name: events.append(name))
    kwargs = dict(accelerator=accelerator, precision_plugin=precision, checkpoint_io=checkpoint_io)
    strategy = _DDPDevicePreservingStrategySO(**kwargs) if distributed else _SingleDeviceStrategySO(**kwargs)
    module = LightningModule()
    module._trainer = SimpleNamespace(state=SimpleNamespace(fn=stage))
    module.cpu = Mock(side_effect=AssertionError("Model CPU offload must be skipped"))
    offload = Mock(side_effect=AssertionError("Optimizer CPU offload must be skipped"))
    monkeypatch.setattr(strategy_module, "_optimizers_to_device", offload)
    strategy.connect(module)
    if distributed:
        group = object()
        module.model = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(module.model)
        module.model.gaussian_process_group = group
        strategy.gaussian_process_group = group
        strategy.cluster_environment = Mock()
        strategy.cluster_environment.teardown.side_effect = lambda: events.append("cluster")
        strategy._layer_sync = Mock()
        strategy._layer_sync.revert.return_value = module
        wrapper = DistributedDataParallel.__new__(DistributedDataParallel)
        torch.nn.Module.__init__(wrapper)
        wrapper.module, wrapper.static_graph = module, True
        strategy.model = wrapper
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        destroyed = Mock(side_effect=lambda group: events.append("group"))
        monkeypatch.setattr(torch.distributed, "destroy_process_group", destroyed)

    strategy.teardown()

    assert events == (["group", "cluster"] if distributed else []) + ["precision", "accelerator", "checkpoint"]
    assert strategy.model is module
    module.cpu.assert_not_called()
    offload.assert_not_called()
    if distributed:
        destroyed.assert_called_once_with(group)
        assert strategy.gaussian_process_group is module.model.gaussian_process_group is None
        assert strategy._layer_sync.revert.call_count == int(stage == TrainerFn.FITTING)
