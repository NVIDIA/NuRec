# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint reproducibility for random streams used outside the custom CUDA generator."""

import io
import random

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pytorch_lightning import LightningModule

from nurec.systems.base import BaseSystem, BaseSystemSO
from nurec.utils.cuda_rng import get_cuda_generator
from nurec.utils.training_rng import capture_global_rng_state, preserve_training_rng_state, restore_global_rng_state


def _system(device=torch.device("cpu")):
    return SimpleNamespace(
        training_step_outputs={},
        trainer_config=SimpleNamespace(world_size=1),
        trainer=SimpleNamespace(global_step=7),
        device=device,
        global_rank=0,
        is_trainer_attached=False,
        resume_weights_only=False,
        load_state_dict=lambda *args, **kwargs: None,
    )


def _seed(value, device):
    random.seed(value)
    np.random.seed(value)
    torch.random.default_generator.manual_seed(value)
    if device.type == "cuda":
        torch.cuda.manual_seed(value)
        get_cuda_generator(device).manual_seed(value)


def _draw(device):
    values = [random.random(), np.random.random(4), torch.rand(4)]
    if device.type == "cuda":
        values += [torch.rand(4, device=device), torch.rand(4, device=device, generator=get_cuda_generator(device))]
    return values


def _assert_same(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    for actual_tensor, expected_tensor in zip(actual[2:], expected[2:], strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    device = torch.device(request.param)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        torch.empty(0, device=device)
    with preserve_training_rng_state(device):
        yield device


def test_full_checkpoint_restores_python_numpy_and_torch_streams(device):
    system = _system(device)
    _seed(42, device)
    _draw(device)
    checkpoint = {"state_dict": {}}
    BaseSystem.on_save_checkpoint(system, checkpoint)
    # The next uninterrupted values are the oracle for the resumed streams.
    expected = _draw(device)

    # Exercise the persisted representation, including device remapping of RNG tensors.
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=False)

    # Move every stream away from the checkpoint before restoring it.
    _seed(777, device)
    _draw(device)
    BaseSystem.on_load_checkpoint(system, checkpoint)
    _assert_same(_draw(device), expected)


@pytest.mark.parametrize("raises", [False, True])
def test_checkpoint_serialization_preserves_training_randomness(device, raises):
    class ArtifactSystem(BaseSystemSO):
        def __init__(self):
            LightningModule.__init__(self)
            self._device = device
            self.trainer = SimpleNamespace(global_step=7, global_rank=0)
            self.training_step_outputs = {}
            self.trainer_config = SimpleNamespace(world_size=1)
            self.artifact_cache = SimpleNamespace(artifact_config=SimpleNamespace(enabled=True))
            self.cached_config = SimpleNamespace(checkpoint=SimpleNamespace(strip_optimizer=False, fp16=False))

        def forward(self, batch):
            raise NotImplementedError

        def training_losses(self, batch, batch_local_idx):
            raise NotImplementedError

        def _populate_and_save_artifacts_rank_zero(self, checkpoint):
            # Artifact generation may consume any of the training RNG streams.
            _draw(device)
            if raises:
                raise RuntimeError("synthetic serialization failure")

    system = ArtifactSystem()
    _seed(42, device)
    expected = _draw(device)

    # Rewind to the same point so checkpoint serialization must leave the oracle unchanged.
    _seed(42, device)
    checkpoint = {"state_dict": {}}
    if raises:
        with pytest.raises(RuntimeError, match="synthetic serialization failure"):
            system.on_save_checkpoint(checkpoint)
    else:
        system.on_save_checkpoint(checkpoint)
    _assert_same(_draw(device), expected)

    # The saved state must reproduce the same next values when loaded later.
    BaseSystem.on_load_checkpoint(_system(device), checkpoint)
    _assert_same(_draw(device), expected)


def test_rank_selects_its_own_global_stream():
    device = torch.device("cpu")
    with preserve_training_rng_state(device):
        system = _system(device)
        checkpoint = {"state_dict": {}}
        states = []
        expected = []
        # Build one saved state and next-draw oracle for each simulated rank.
        for seed in (42, 43):
            _seed(seed, device)
            states.append(capture_global_rng_state(device))
            expected.append(_draw(device))
        system.trainer_config.world_size = 2
        BaseSystem.on_save_checkpoint(system, checkpoint)
        # Distributed gathering stores RNG records in global-rank order.
        checkpoint["nre.rng.global_states"] = states
        system.trainer_config.world_size = 3
        for rank in (0, 1):
            system.global_rank = rank
            BaseSystem.on_load_checkpoint(system, checkpoint)
            _assert_same(_draw(device), expected[rank])
        system.global_rank = 2
        _seed(777, device)
        expected_new_rank = _draw(device)
        _seed(777, device)
        BaseSystem.on_load_checkpoint(system, checkpoint)
        _assert_same(_draw(device), expected_new_rank)

        checkpoint["nre.rng.global_states"] = states[:1]
        system.global_rank = 1
        with pytest.raises(ValueError, match="no global RNG state for rank 1"):
            BaseSystem.on_load_checkpoint(system, checkpoint)


@pytest.mark.parametrize("weights_only", [False, True])
def test_missing_or_weights_only_global_state_leaves_new_streams_unchanged(weights_only):
    device = torch.device("cpu")
    with preserve_training_rng_state(device):
        system = _system(device)
        system.resume_weights_only = weights_only
        checkpoint = {"state_dict": {}}
        BaseSystem.on_save_checkpoint(system, checkpoint)
        if not weights_only:
            del checkpoint["nre.rng.global_states"]

        # A weights-only load or absent RNG record retains streams initialized for the current run.
        _seed(777, device)
        expected = _draw(device)
        _seed(777, device)
        BaseSystem.on_load_checkpoint(system, checkpoint)
        _assert_same(_draw(device), expected)


def test_cpu_capture_does_not_initialize_cuda(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("CPU RNG capture touched CUDA")

    monkeypatch.setattr(torch.cuda, "get_rng_state", unexpected)
    monkeypatch.setattr(torch.cuda, "_lazy_init", unexpected)
    state = capture_global_rng_state(torch.device("cpu"))
    assert state["torch_cuda"] is None
    with pytest.raises(ValueError, match="Unsupported global RNG checkpoint schema"):
        restore_global_rng_state({**state, "schema": -1}, torch.device("cpu"))
