# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nurec.utils.trainer import TrainerConfig


def make_trainer_cfg() -> TrainerConfig:
    return TrainerConfig(
        max_epochs=1,
        check_val_every_n_epoch=1,
        precision="32",
        log_every_n_steps=1,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
