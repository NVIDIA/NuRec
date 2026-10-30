# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared between the instant_nurec predict entrypoint (`nurec/instant_nurec/run.py`) and the instant_nurec
train/val/test/trainval/predict entrypoint in a separate downstream training package."""

from __future__ import annotations

import logging
import os

from typing import Any

import pytorch_lightning as pl
import torch

from pytorch_lightning.strategies import DDPStrategy, Strategy

from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig
from nurec.utils.misc import is_env_true


def _maybe_enable_cudnn_benchmark_for_static_shapes(config: InstantNuRecConfig, logger: logging.Logger) -> None:
    # This flag is process-global: every disabled gate must clear a previous run's setting.
    torch.backends.cudnn.benchmark = False
    if not config.enable_cudnn_benchmark:
        logger.info("Disabled cuDNN benchmark by instant_nurec configuration.")
        return

    if not torch.cuda.is_available() or not torch.backends.cudnn.enabled or torch.backends.cudnn.deterministic:
        return

    if not config.has_static_shapes():
        return

    torch.backends.cudnn.benchmark = True
    logger.info("Enabled cuDNN benchmark for static instant_nurec training shapes.")


def setup_environment_and_logger(
    config: InstantNuRecConfig,
    *,
    distributed_log_label: str,
    enable_cudnn_benchmark_gating: bool = False,
) -> logging.Logger:
    """Common environment/logger setup shared by the instant_nurec predict and train entrypoints.

    Args:
        config: The parsed top-level instant_nurec config (predict-only `InstantNuRecConfig` or its training
            superset `TrainInstantNuRecConfig`).
        distributed_log_label: Text used in the log message emitted when LOCAL_RANK-based device
            pinning kicks in, e.g. "Distributed inference detected" vs "Distributed training
            detected".
        enable_cudnn_benchmark_gating: Whether to enable cuDNN benchmark mode for datasets with
            static shapes. This only ever has an effect for training configs: predict-only
            `InstantNuRecConfig.mode` is fixed to `"predict"`, for which `has_static_shapes()` is always
            `False` (it resolves the *train* dataset config, which doesn't exist in predict mode).
            Predict-only callers should leave this disabled.
    """
    if (local_rank := os.environ.get("LOCAL_RANK")) is not None:
        torch.cuda.set_device(int(local_rank))
        logging.getLogger(__name__).info(
            f"[info] {distributed_log_label}. LOCAL_RANK = {local_rank}, device = {torch.cuda.current_device()}"
        )

    if is_env_true("CUDA_SYNC_DEBUG", False):
        torch.cuda.set_sync_debug_mode("warn")
        logging.getLogger(__name__).info("CUDA synchronization debug mode enabled (CUDA_SYNC_DEBUG=1)")

    logger = logging.getLogger(__name__)
    if config.verbose:
        logger.setLevel(logging.DEBUG)

    if enable_cudnn_benchmark_gating:
        _maybe_enable_cudnn_benchmark_for_static_shapes(config, logger)

    # Use the same seed for model initialization across all processes
    pl.seed_everything(config.seed, workers=True)

    return logger


def resolve_has_full_init(model_config: Any) -> bool:
    """Determine whether `model_config` (a `CelsiusModelConfig` or `KelvinModelConfig`) provides a
    full-model weight init, either via `init_weights_path` or a "full"/"tokengs" entry in
    `init_weights_paths`.

    Must stay in lockstep with KelvinModel.on_train_from_scratch_start's full-model check.
    """
    has_full_init = bool(getattr(model_config, "init_weights_path", None))
    init_weights_paths = getattr(model_config, "init_weights_paths", None)
    if not has_full_init and init_weights_paths is not None:
        if {"full", "tokengs"} & init_weights_paths.keys():
            has_full_init = True
    return has_full_init


def resolve_strategy(config: InstantNuRecConfig) -> "Strategy | str":
    """Resolve the pytorch-lightning DDP strategy to use for `config` (a predict-only `InstantNuRecConfig`
    or its training superset `TrainInstantNuRecConfig`).

    Maps the `"ddp_find_unused_parameters"` shorthand to a `DDPStrategy(find_unused_parameters=True)`
    instance; any other value in `config.system.strategy` is passed through unchanged.
    """
    strategy: Strategy | str = config.system.strategy
    if strategy == "ddp_find_unused_parameters":
        strategy = DDPStrategy(find_unused_parameters=True)
    return strategy
