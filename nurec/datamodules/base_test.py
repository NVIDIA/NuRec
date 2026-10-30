# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from nurec.datamodules.base import SODataModule


def _datamodule() -> SODataModule:
    datamodule = object.__new__(SODataModule)
    object.__setattr__(datamodule, "trainer", SimpleNamespace(global_rank=0))
    object.__setattr__(
        datamodule,
        "config",
        SimpleNamespace(
            prefetch=SimpleNamespace(enabled=False),
            mock_dataloader=False,
            train_num_workers=0,
            train_batch_size=1,
        ),
    )
    object.__setattr__(datamodule, "train_dataset", [object()])
    datamodule._train_loader_generator = None
    return datamodule


def test_training_iterator_does_not_advance_default_cpu_rng():
    datamodule = _datamodule()

    initial_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(42)
        expected = torch.rand(4)

        # Rewind the global stream so any DataLoader consumption changes the comparison below.
        torch.manual_seed(42)
        # Each iterator consumes a worker seed from the DataLoader's private generator.
        iter(datamodule.train_dataloader())
        assert datamodule._train_loader_generator is not None
        generator_state_after_first_iterator = datamodule._train_loader_generator.get_state()
        iter(datamodule.train_dataloader())
        generator_state_after_second_iterator = datamodule._train_loader_generator.get_state()
        actual = torch.rand(4)
    finally:
        torch.random.set_rng_state(initial_state)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.equal(generator_state_after_first_iterator, generator_state_after_second_iterator)


def test_training_loader_generator_state_continues_after_checkpoint():
    source = _datamodule()
    iter(source.train_dataloader())
    state_dict = source.state_dict()
    assert state_dict["train_loader_generator_rank_count"] == len(state_dict["train_loader_generator_states"])

    resumed = _datamodule()
    resumed.load_state_dict(state_dict)
    assert source._train_loader_generator is not None
    assert resumed._train_loader_generator is not None
    torch.testing.assert_close(
        resumed._train_loader_generator.get_state(),
        source._train_loader_generator.get_state(),
        atol=0,
        rtol=0,
    )

    # Matching iterator creation must advance both restored streams identically.
    iter(source.train_dataloader())
    iter(resumed.train_dataloader())
    torch.testing.assert_close(
        resumed._train_loader_generator.get_state(),
        source._train_loader_generator.get_state(),
        atol=0,
        rtol=0,
    )


def test_training_loader_generator_state_selects_global_rank():
    states = [torch.Generator(device="cpu").manual_seed(seed).get_state() for seed in (42, 43)]
    datamodule = _datamodule()
    datamodule.trainer.global_rank = 1

    datamodule.load_state_dict({"train_loader_generator_states": states})

    assert datamodule._train_loader_generator is not None
    torch.testing.assert_close(datamodule._train_loader_generator.get_state(), states[1], atol=0, rtol=0)


def test_missing_training_loader_generator_state_keeps_lazy_initialization():
    datamodule = _datamodule()

    datamodule.load_state_dict({})

    assert datamodule._train_loader_generator is None


def test_new_rank_keeps_lazy_training_loader_generator_initialization(monkeypatch):
    saved_state = torch.Generator(device="cpu").manual_seed(42).get_state()
    initialized_state = torch.Generator(device="cpu").manual_seed(777).get_state()
    monkeypatch.setattr("nurec.datamodules.base.compute_process_local_rng_seed", lambda: 777)
    datamodule = _datamodule()
    datamodule.trainer.global_rank = 1

    datamodule.load_state_dict({"train_loader_generator_states": [saved_state], "train_loader_generator_rank_count": 1})

    assert datamodule._train_loader_generator is None
    datamodule.train_dataloader()
    assert datamodule._train_loader_generator is not None
    torch.testing.assert_close(datamodule._train_loader_generator.get_state(), initialized_state, atol=0, rtol=0)


def test_missing_saved_rank_training_loader_generator_state_raises():
    saved_state = torch.Generator(device="cpu").manual_seed(42).get_state()
    datamodule = _datamodule()
    datamodule.trainer.global_rank = 1

    with pytest.raises(ValueError, match="no training-loader generator state for rank 1"):
        datamodule.load_state_dict(
            {"train_loader_generator_states": [saved_state], "train_loader_generator_rank_count": 2}
        )
