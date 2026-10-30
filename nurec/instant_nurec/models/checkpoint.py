# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch

from torch import nn


def load_checkpoint_state_dict(
    module: nn.Module,
    checkpoint_path: str | Path,
    *,
    torch_load_weights_only: bool,
) -> None:
    """Strictly load module state from a raw or Lightning-wrapped checkpoint.

    ``torch_load_weights_only`` controls PyTorch's restricted unpickler, not
    whether non-state Lightning metadata is restored. This helper always loads
    only the checkpoint's state dict into ``module``.
    """

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=torch_load_weights_only,
        )
    except Exception as error:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {error}") from error

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Checkpoint state_dict must be a mapping, got {type(state_dict).__name__}")

    module.load_state_dict(cast(Mapping[str, torch.Tensor], state_dict), strict=True)
