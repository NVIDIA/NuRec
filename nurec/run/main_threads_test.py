# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import os

import pytest
import torch


main_module = importlib.import_module("nurec.run.main")


@pytest.fixture
def cpus(monkeypatch):
    # setenv first so monkeypatch records the original value and restores it after the test
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("OMP_NUM_THREADS")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(64)))
    previous = torch.get_num_threads()
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("world_size, devices, threads", [(8, 8, 8), (16, 8, 8), (2, 2, 32), (4, 128, 1)])
def test_multi_rank_splits_cpus_per_rank(cpus, world_size, devices, threads):
    main_module.cap_cpu_threads_per_rank(world_size, devices)
    assert os.environ["OMP_NUM_THREADS"] == str(threads)
    assert torch.get_num_threads() == threads


def test_single_rank_unchanged(cpus):
    before = torch.get_num_threads()
    main_module.cap_cpu_threads_per_rank(1, 1)
    assert "OMP_NUM_THREADS" not in os.environ
    assert torch.get_num_threads() == before


def test_user_setting_kept(cpus, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    before = torch.get_num_threads()
    main_module.cap_cpu_threads_per_rank(8, 8)
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert torch.get_num_threads() == before
