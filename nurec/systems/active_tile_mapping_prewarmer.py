# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot active-tile mapping prewarm lifecycle."""

from __future__ import annotations

import logging

from contextlib import contextmanager
from typing import Iterator

import torch

from pytorch_lightning import Trainer

from nurec.datasets.base import BaseDataset
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.utils.profiling import ScopedTimer
from nurec.utils.trainer import distributed_context


log = logging.getLogger(__name__)


class ActiveTileMappingPrewarmer:
    """Populate fixed camera-to-LiDAR mappings before the first training forward."""

    # All ranks use the same seed so synthetic calibration builds the same mapping.
    # _isolated_rng restores both default RNG streams before real training resumes.
    _SEED = 0

    def __init__(self) -> None:
        self._finished = False

    def reset(self) -> None:
        """Re-arm the one-shot prewarm for a new fit or restored model state."""
        self._finished = False

    def run_once(
        self,
        *,
        model: GaussiansComposite,
        train_dataset: BaseDataset,
        device: torch.device,
        trainer: Trainer,
    ) -> None:
        """Run synchronously after DDP pre-forward setup and before the first render."""
        if self._finished:
            return
        try:
            self._run(
                model=model,
                train_dataset=train_dataset,
                device=device,
                trainer=trainer,
            )
        finally:
            self._finished = True

    @contextmanager
    def _isolated_rng(self, device: torch.device) -> Iterator[None]:
        """Install the rank-invariant mapping seed without advancing training RNG streams.

        Every rank must synthesize the same calibration mapping. The surrounding
        ``fork_rng`` scope restores the caller's CPU and CUDA RNG states afterward.
        """
        device_idx = device.index if device.index is not None else torch.cuda.current_device()
        resolved_device = torch.device("cuda", device_idx)

        # Mapping construction randomly subsamples bounded pose perturbations.
        # Seed both RNG streams for a deterministic mapping, then restore them
        # so the synthetic prewarm cannot advance the real training sequence.
        with torch.random.fork_rng(devices=[device_idx]):
            cpu_generator = torch.Generator(device="cpu")
            cpu_generator.manual_seed(self._SEED)
            torch.set_rng_state(cpu_generator.get_state())
            cuda_generator = torch.Generator(device=resolved_device)
            cuda_generator.manual_seed(self._SEED)
            torch.cuda.set_rng_state(cuda_generator.get_state(), device=resolved_device)
            yield

    def _run(
        self,
        *,
        model: GaussiansComposite,
        train_dataset: BaseDataset,
        device: torch.device,
        trainer: Trainer,
    ) -> None:
        """Populate every supported fixed mapping under coordinated error handling.

        This runs from the first underlying ``training_step``, after DDP's
        pre-forward buffer broadcast and DataLoader worker creation. It does not
        create a background thread or private CUDA stream.
        """
        if not model.active_tile_mapping_prewarm_enabled():
            return

        # Keep rank-local failures from leaving peers to enter the next training
        # collective alone. This is the only cross-rank synchronization added by
        # the prewarm.
        with distributed_context(trainer):
            with ScopedTimer("ActiveTileMappingPrewarmer/run"):
                plan = train_dataset.get_active_tile_mapping_prewarm_plan()
                if plan.skip_reason is not None:
                    log.info("Skipping active-tile mapping prewarm: %s", plan.skip_reason)
                    return

                try:
                    with self._isolated_rng(device):
                        for entry in plan.entries:
                            data_batch = entry.data_batch.to(device)
                            with torch.no_grad():
                                rendering_batch = model.calib(
                                    data_batch,
                                    skip_calib=False,
                                    # This synthetic metadata-only batch must not claim
                                    # TensorProber output for the real training batch.
                                    global_step_for_prober=None,
                                )
                                if rendering_batch.camera is None or rendering_batch.lidar is None:
                                    raise RuntimeError(
                                        "Active-tile mapping prewarm calibration did not produce camera and LiDAR data"
                                    )
                                model.prewarm_camera_to_lidar_tile_mapping(
                                    entry.camera_sensor_idx,
                                    rendering_batch.camera,
                                    rendering_batch.lidar,
                                )
                finally:
                    # The next render is stream-ordered already. This wait exists only
                    # to surface deferred CUDA errors inside BroadcastExceptions.
                    torch.cuda.current_stream(device=device).synchronize()
