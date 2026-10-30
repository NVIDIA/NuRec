# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint state for process-global training random streams."""

import random

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypedDict

import numpy as np
import torch

from nurec.utils.cuda_rng import get_cuda_generator


class GlobalRNGState(TypedDict):
    """Plain checkpoint data for one rank's default random generators.

    Private generators, including dataset sampling generators, have separate
    owners and must be checkpointed by those owners.
    """

    schema: int
    python: tuple
    numpy: tuple
    torch_cpu: torch.Tensor
    torch_cuda: torch.Tensor | None


def capture_global_rng_state(device: torch.device) -> GlobalRNGState:
    """Copy default generator states without drawing samples or initializing CUDA.

    CUDA training must already have initialized its rank-local device. Only that
    device is inspected, so capture does not allocate contexts on other GPUs.
    """
    if device.type == "cuda" and not torch.cuda.is_initialized():
        raise ValueError("CUDA RNG capture requires an initialized training device")
    numpy_state = np.random.get_state()
    assert isinstance(numpy_state, tuple)
    return {
        "schema": 1,
        "python": random.getstate(),
        "numpy": numpy_state,
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None,
    }


def restore_global_rng_state(state: GlobalRNGState, device: torch.device) -> None:
    """Restore default streams on this rank, accepting checkpoint tensors on any device."""
    if state["schema"] != 1:
        raise ValueError(f"Unsupported global RNG checkpoint schema: {state['schema']}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda" and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


@contextmanager
def preserve_training_rng_state(device: torch.device) -> Iterator[None]:
    """Isolate checkpoint serialization from training's default and custom streams.

    Call after training setup has initialized its device and custom CUDA
    generator. State is restored even if serialization raises an exception.
    """
    global_state = capture_global_rng_state(device)
    generator = get_cuda_generator(device) if device.type == "cuda" else None
    generator_state = generator.get_state() if generator is not None else None
    try:
        yield
    finally:
        restore_global_rng_state(global_state, device)
        if generator is not None:
            assert generator_state is not None
            generator.set_state(generator_state)
