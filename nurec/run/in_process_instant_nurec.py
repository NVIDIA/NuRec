# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process instant_nurec-to-NuRec Gaussian initialization.

This module intentionally lives at the run/orchestration layer. instant_nurec already
depends on core NuRec models, so importing it from those layers would create a
Bazel dependency cycle.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import random
import time

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import torch

from lightning_fabric import Fabric
from lightning_fabric.plugins import MixedPrecision, Precision
from torch.utils.data import DataLoader

from nurec.datasets.ncore import NCOREDataSource
from nurec.datasets.summary import DataSourceSummary
from nurec.instant_nurec.config.dataset import NCoreInstantNuRecDatasetConfig
from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig
from nurec.instant_nurec.datasets.datamodule import InstantNuRecPredictDataModule
from nurec.instant_nurec.predict.export_ply import build_ply_gaussian_loader
from nurec.instant_nurec.predict.prediction_runner import InstantNuRecPredictionRunner
from nurec.models.gaussians.utils import PLYGaussianLoader
from nurec.utils.batch import InstantNuRecDataBatch


log = logging.getLogger(__name__)


def _convert_input_like_standalone_instant_nurec(
    batch: InstantNuRecDataBatch, precision: Precision
) -> InstantNuRecDataBatch:
    """Apply the input conversion used by the standalone instant_nurec Lightning loop."""

    # PyTorch Lightning's MixedPrecision plugin leaves inputs unchanged and
    # applies the requested precision only through its forward autocast context.
    # Fabric's corresponding plugin eagerly casts every floating input instead.
    # All other built-in precision plugins share Lightning's input-conversion
    # behavior, so delegate to the resolved plugin without a local dtype table.
    if isinstance(precision, MixedPrecision):
        return batch
    return cast(InstantNuRecDataBatch, precision.convert_input(batch))


@contextlib.contextmanager
def _isolated_instant_nurec_rng(seed: int, device: torch.device) -> Iterator[None]:
    """Run instant_nurec with isolated RNG state, then restore NuRec's state."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device.index]):
            random.seed(seed)
            np.random.seed(seed)
            torch.default_generator.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _validate_instant_nurec_prediction_config(config: InstantNuRecConfig) -> None:
    """Reject configurations that cannot satisfy the in-process runner contract."""

    if not config.resume:
        raise ValueError("In-process instant_nurec requires a non-empty pretrained checkpoint in config.resume")
    if not config.resume_weights_only:
        raise ValueError("In-process instant_nurec requires config.resume_weights_only=true")

    predict_dataset = config.dataset.predict
    if not isinstance(predict_dataset, NCoreInstantNuRecDatasetConfig):
        actual_name = None if predict_dataset is None else predict_dataset.name
        raise ValueError(
            f"In-process instant_nurec requires the typed prediction split to be NCore (name='instant_nurec-ncore'); got {actual_name!r}"
        )

    ignored_outputs = [
        name
        for name, enabled in (
            ("predict.artifact", config.predict.artifact.enabled),
            ("predict.export_ply disk write", config.predict.export_ply.enabled),
            ("predict.render_video", config.predict.render_video.enabled),
        )
        if enabled
    ]
    if ignored_outputs:
        log.warning(
            "In-process instant_nurec returns an in-memory payload and skips enabled standalone output hooks: %s. "
            "Primitive-merge and export_ply codec/filter settings still apply.",
            ", ".join(ignored_outputs),
        )


def _cleanup_after_instant_nurec_generation() -> None:
    """Best-effort release without replacing a prediction failure."""

    for description, cleanup in (("Python objects", gc.collect), ("the CUDA allocator cache", torch.cuda.empty_cache)):
        try:
            cleanup()
        except Exception:
            log.exception("Failed to release %s after in-memory instant_nurec generation", description)


def _generate(config: InstantNuRecConfig, datasource: NCOREDataSource) -> PLYGaussianLoader:
    """Predict one sequence and release the instant_nurec producer before returning its payload."""

    start_time = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("In-process instant_nurec requires CUDA")

    device = torch.device("cuda", torch.cuda.current_device())
    _validate_instant_nurec_prediction_config(config)
    log.info("Generating in-process instant_nurec payload for sequence %s", datasource.sequence_id)
    # Keep explicit references so ``finally`` can drop the heavyweight owners
    # before allocator cleanup: Fabric/model objects own model CUDA storage;
    # data objects own workers and prefetched batches; outputs own inference
    # tensors. The original runner serves preparation and forward without
    # Fabric's eager input conversion. Only a completed ``loader`` survives the
    # cleanup.
    fabric: Fabric | None = None
    datamodule: InstantNuRecPredictDataModule | None = None
    runner: InstantNuRecPredictionRunner | None = None
    fabric_runner: Any = None
    dataloader: DataLoader | None = None
    iterator: Iterator[InstantNuRecDataBatch] | None = None
    batch: InstantNuRecDataBatch | None = None
    output: Any = None
    loader: PLYGaussianLoader | None = None
    try:
        assert config.resume is not None
        with _isolated_instant_nurec_rng(config.seed, device):
            # Fabric owns precision parsing, matching standalone instant_nurec; pass the
            # typed instant_nurec setting through unchanged instead of defining local policy.
            precision: Any = config.system.precision
            fabric = Fabric(
                accelerator="cuda",
                devices=[device.index],
                strategy="auto",
                precision=precision,
            )
            datamodule = InstantNuRecPredictDataModule(config, seed=config.seed)
            runner = InstantNuRecPredictionRunner(config)
            runner.load_weights_only_checkpoint(config.resume)

            dataloader = datamodule.predict_dataloader()
            if datamodule.predict_dataset is not None:
                datamodule.predict_dataset.set_epoch(0)
            batch_count = len(dataloader)
            if batch_count != 1:
                raise ValueError(
                    f"In-process instant_nurec expects exactly one prediction batch for one sequence; got {batch_count} batches"
                )
            # Creating the iterator starts DataLoader workers and queues their
            # prefetch before the blocking model transfer on the main process.
            iterator = iter(dataloader)
            fabric_runner = fabric.setup_module(runner)
            batch = cast(InstantNuRecDataBatch, next(iterator))
            # Match standalone instant_nurec's prediction-loop ordering: apply the
            # resolved precision plugin's input policy before device transfer.
            batch = _convert_input_like_standalone_instant_nurec(batch, fabric.strategy.precision)
            batch = cast(InstantNuRecDataBatch, fabric.to_device(batch))

            # This hook may update ordinary module state, so keep it outside both
            # inference mode and autocast. Only the actual forward is inference-only.
            runner.prepare_prediction()
            with torch.inference_mode(), fabric.autocast():
                # Call the prepared module without Fabric's wrapper because the
                # input policy was applied above and standalone Lightning does
                # not apply Fabric's output conversion. In-process instant_nurec is currently
                # single-device, so there is no distributed strategy wrapper to
                # bypass here. Fabric still owns model conversion and the forward
                # precision context, matching the current instant_nurec CLI.
                output = fabric_runner.module(batch)

            # The prediction output contains inference tensors, but the payload
            # becomes training initialization state. Build and copy it only after
            # leaving inference mode and autocast so the generic tensor loader
            # does not need to override its caller's execution mode.
            if len(output.primitives) != 1 or len(output.batch) != 1 or output.batch.context_rig is None:
                raise ValueError(
                    "instant_nurec primitive merging must produce one primitive and one rig trajectory for in-memory initialization"
                )
            loader = build_ply_gaussian_loader(
                config=config.predict.export_ply,
                primitives=output.primitives[0],
                rig_trajectories=output.batch.context_rig[0],
                device=device,
            )

    except BaseException:
        loader = None
        raise
    finally:
        # Drop every heavyweight producer reference before returning to NuRec. On
        # failure, also release a partially constructed payload.
        fabric_runner = None
        datamodule = None
        runner = None
        fabric = None
        iterator = None
        dataloader = None
        batch = None
        output = None
        _cleanup_after_instant_nurec_generation()

    assert loader is not None
    elapsed_seconds = time.perf_counter() - start_time
    log.info(
        "Generated in-process instant_nurec payload for sequence %s in %.3f seconds",
        datasource.sequence_id,
        elapsed_seconds,
    )
    return loader


def _generate_prepared(config: InstantNuRecConfig, datasource: NCOREDataSource) -> PLYGaussianLoader:
    """Generate a payload, then publish it device-ready in NuRec coordinates."""

    loader = _generate(config, datasource)
    device = loader.positions.device
    loader.transform(torch.FloatTensor(datasource.world_to_nurec.matrix).to(device))
    return loader


def initialize_from_instant_nurec_in_memory(
    config: InstantNuRecConfig, summary: DataSourceSummary
) -> PLYGaussianLoader:
    """Generate and prepare an in-process instant_nurec payload for NuRec world size one.

    Args:
        config: Fully materialized nested in-process instant_nurec configuration.
        summary: Outer NuRec datasource summary supplying the active sequence and
            world-to-NuRec transform.

    Returns:
        A device-local, NuRec-coordinate loader whose ordinary tensors are safe
        for Gaussian layer initializers to borrow read-only.
    """

    datasource = summary.datasource
    if not isinstance(datasource, NCOREDataSource):
        raise TypeError("In-process instant_nurec supports only NCOREDataSource")

    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if world_size != 1:
            raise RuntimeError(f"in_process_instant_nurec requires distributed world size 1; got {world_size}.")

    return _generate_prepared(config, datasource)
