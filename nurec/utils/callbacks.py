# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import glob
import logging
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

from types import FrameType
from typing import Any, Dict, Optional

import imageio.v2 as iio
import numpy as np
import torch

from einops import rearrange
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.loggers.logger import DummyExperiment, DummyLogger, Logger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer

from nurec.config.base_schema import config_to_primitive
from nurec.config.fixed_camera_render import FixedCameraRenderConfig
from nurec.config.logger import DummyLoggerConfig, LoggerConfigType, TensorboardLoggerConfig, WandbLoggerConfig
from nurec.config.nurec import NuRecConfig
from nurec.datasets.ncore import NCORESequentialDataset
from nurec.systems.base import BaseSystem, BaseSystemSO
from nurec.utils.media_log import media_backend
from nurec.utils.misc import rank_zero_only
from nurec.utils.types import RayFlags
from nurec.utils.visualize import scalar2img


logger = logging.getLogger(__name__)


def get_latest_version(folder):
    versions = [int(pathlib.PurePath(path).name.split("_")[-1]) for path in glob.glob(f"{folder}/version_*/")]

    if len(versions) == 0:
        return -1

    versions.sort()
    return versions[-1]


def get_latest_checkpoint(out_dir, version):
    chkpts = []
    while version > -1:
        folder = os.path.join(out_dir, f"version_{version}", "checkpoints")

        latest = f"{folder}/last.ckpt"
        if os.path.exists(latest):
            return latest, version

        chkpts = glob.glob(f"{folder}/epoch=*.ckpt")

        if len(chkpts) > 0:
            break

        version -= 1

    if len(chkpts) == 0:
        return None, None

    latest = max(chkpts, key=os.path.getctime)

    return latest, version


def _compute_target_linear_idx(
    camera_ids: list[str],
    camera_frames: Dict[str, list[int]],
    target_camera_id: str,
    target_frame_idx: int,
) -> Optional[int]:
    """Linear `__getitem__` index for (target_camera_id, target_frame_idx).

    Walks cameras in `camera_ids` order, summing prior cameras' frame counts.
    Returns None if the camera or frame is absent. Tolerant of inconsistency
    between `camera_ids` and `camera_frames` (treats missing keys as empty).
    """
    if target_camera_id not in camera_frames:
        return None
    linear_idx = 0
    for cam_id in camera_ids:
        frames = camera_frames.get(cam_id, [])
        if cam_id == target_camera_id:
            if target_frame_idx not in frames:
                return None
            return linear_idx + frames.index(target_frame_idx)
        linear_idx += len(frames)
    return None


def _should_render(global_step: int, start_step: int, every_n_steps: int) -> bool:
    """Schedule predicate: render iff step >= start and (step - start) % every == 0."""
    if global_step < start_step:
        return False
    return (global_step - start_step) % every_n_steps == 0


class TimingLogger(Callback):
    """
    Simple callback to log the elapsed time for training and validation.
    Specifically, we measure the total time spent in `train_epoch`s and the total time spent in `validation_epoch`s.

    Note: we can't `self.log()` inside `on_train/validation_end` so we use `self.logger.experiment.log()`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.train_elapsed_time_s: float = 0.0
        self.validation_elapsed_time_s: float = 0.0

        self.train_epoch_start_end_timer_s: Optional[float] = None
        self.validation_epoch_start_end_timer_s: Optional[float] = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_start(trainer, pl_module)
        self.train_epoch_start_end_timer_s = time.time()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        now = time.time()
        if self.train_epoch_start_end_timer_s is None:
            # on_train_epoch_start is not invoked for an epoch that is resumed mid-way from a
            # checkpoint, so there is no start time to measure against. Skip timing this partial
            # epoch instead of asserting and taking the whole (multi-node) job down.
            logger.warning(
                f"{self.__class__.__name__}: train epoch-start timer is None "
                "(expected on a mid-epoch resume); skipping elapsed-time accounting for this epoch."
            )
            return
        assert now > self.train_epoch_start_end_timer_s, (
            f"{self.__class__.__name__}: contract error, time went backwards"
        )
        self.train_elapsed_time_s += now - self.train_epoch_start_end_timer_s

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.validation_epoch_start_end_timer_s = time.time()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        now = time.time()
        if self.validation_epoch_start_end_timer_s is None:
            # See on_train_epoch_end: a mid-epoch resume can skip on_validation_epoch_start,
            # leaving no start time. Skip timing rather than crashing the job.
            logger.warning(
                f"{self.__class__.__name__}: validation epoch-start timer is None "
                "(expected on a mid-epoch resume); skipping elapsed-time accounting for this epoch."
            )
            return
        assert now > self.validation_epoch_start_end_timer_s, (
            f"{self.__class__.__name__}: contract error, time went backwards"
        )
        self.validation_elapsed_time_s += now - self.validation_epoch_start_end_timer_s

    @staticmethod
    def log_scalar(name: str, value: float | int, pl_module: LightningModule, step: Optional[int] = None) -> None:
        logger = pl_module.logger
        match logger:
            case WandbLogger():
                logger.experiment.log({name: value}, step=step)
            case TensorBoardLogger():
                logger.experiment.add_scalar(name, value, global_step=step)
            case DummyLogger():
                pass
            case _:
                raise ValueError(f"Unsupported logger type {type(logger)=}")

        if isinstance(pl_module, BaseSystemSO):
            pl_module.collect_metric(name, value, timing_method="wall_clock")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_end(trainer, pl_module)
        TimingLogger.log_scalar("train/elapsed_time_s", self.train_elapsed_time_s, pl_module)
        logger.info(f"Train elapsed time: {self.train_elapsed_time_s:.2f} seconds")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_end(trainer, pl_module)
        TimingLogger.log_scalar("test/elapsed_time_s", self.validation_elapsed_time_s, pl_module)
        logger.info(f"Validation elapsed time: {self.validation_elapsed_time_s:.2f} seconds")


class SmoothedProgressBar(TQDMProgressBar):
    """TQDMProgressBar that uses tqdm's EMA smoothing for it/s display.

    This will more accurately reflect the actual training rate, since that changes
    as the model converges and strategy changes.

    PyTorch Lightning's default TQDMProgressBar bypasses tqdm.update() by
    directly setting bar.n and calling bar.refresh(). This means tqdm's EMA
    (controlled by the smoothing parameter) is never fed, and the displayed
    rate always falls back to the global average (n / elapsed).

    This subclass overrides on_train_batch_end to call bar.update() instead,
    which properly feeds the EMA and gives a responsive it/s reading.
    """

    SMOOTHING_WINDOW_99P = 20  # ~N iterations contribute 99% of the it/s

    def init_train_tqdm(self) -> Tqdm:
        smoothing_alpha = 1 - 0.01 ** (1 / self.SMOOTHING_WINDOW_99P)

        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=smoothing_alpha,
            bar_format=self.BAR_FORMAT,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            # Use update() instead of directly setting .n so tqdm's EMA is fed
            delta = n - self.train_progress_bar.n
            if delta > 0:
                self.train_progress_bar.update(delta)
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.val_progress_bar.total):
            # Use update() instead of directly setting .n so tqdm's EMA is fed
            delta = n - self.val_progress_bar.n
            if delta > 0:
                self.val_progress_bar.update(delta)
            self.val_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))


def notify_before_checkpoint_capture(trainer: Trainer) -> None:
    """Tell the system a checkpoint is about to be captured.

    trainer.save_checkpoint reads the state dict before any on_save_checkpoint hook runs. The call sites below are
    the last point a model change can still reach the file. Every checkpoint writer in this module has to route
    through here. Runs on every rank, as distributed checkpointing requires.
    """
    if isinstance(trainer.lightning_module, BaseSystem):
        trainer.lightning_module.on_before_checkpoint_capture()


class _EpochEndModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that ensures last.ckpt captures post-batch-end model state.

    PL calls callback on_train_batch_end BEFORE LightningModule on_train_batch_end.
    When every_n_train_steps coincides with the last training step, the checkpoint
    saved during the callback misses model updates (densification, SH degree
    increment) that run in the LightningModule hook.  The epoch-end save
    (save_on_train_epoch_end=True) should fix this, but PL suppresses it via
    _last_global_step_saved deduplication.  We clear that flag and explicitly
    refresh `last` when PL's step-based path does not, so the final checkpoint
    contains the fully-updated model state.
    """

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        had_saved_step = hasattr(self, "_last_global_step_saved")
        previous_saved_step = getattr(self, "_last_global_step_saved", -1)
        epoch_end_completed = False
        explicit_last_saved = False
        try:
            if had_saved_step:
                self._last_global_step_saved = -1

            super().on_train_epoch_end(trainer, pl_module)

            # With step-based checkpointing, PL 2.6 does not run the epoch-based
            # top-k save and therefore leaves _last_global_step_saved at -1.
            # Explicitly refresh `last` now, after LightningModule's batch-end
            # hook, so it and the artifact cache include post-batch mutations.
            if (
                self.save_last
                and getattr(self, "_last_global_step_saved", None) != trainer.global_step
                and not self._should_skip_saving_checkpoint(trainer)
                and self._should_save_on_train_epoch_end(trainer)
            ):
                self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))
                explicit_last_saved = True
            epoch_end_completed = True
        finally:
            if had_saved_step:
                if explicit_last_saved:
                    # PL currently records this in _save_checkpoint, but doing it
                    # here keeps the state deterministic if that internal changes.
                    self._last_global_step_saved = trainer.global_step
                elif not epoch_end_completed or getattr(self, "_last_global_step_saved", None) == -1:
                    # Roll back our sentinel when saving was skipped or failed.
                    self._last_global_step_saved = previous_saved_step

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        # No public PL hook fires between the model being final and the state dict being read, hence this override.
        notify_before_checkpoint_capture(trainer)
        super()._save_checkpoint(trainer, filepath)

    @staticmethod
    def _validation_has_run(trainer: Trainer) -> bool:
        """Best-effort PL validation-loop state check.

        PyTorch Lightning does not expose a public flag for this; review this
        helper on PL upgrades.
        """
        try:
            val_loop = trainer.fit_loop.epoch_loop.val_loop
        except AttributeError:
            return False
        return bool(getattr(val_loop, "_has_run", False))

    def _save_topk_checkpoint(self, trainer: Trainer, monitor_candidates: Dict[str, Any]) -> None:
        validation_has_run = self._validation_has_run(trainer)
        if self.monitor is not None and self.monitor not in monitor_candidates and validation_has_run:
            logger.warning(
                "Skipping top-k checkpoint update at step %s because monitor %r is absent from logged metrics. "
                "The last checkpoint will still be saved.",
                trainer.global_step,
                self.monitor,
            )
            return
        super()._save_topk_checkpoint(trainer, monitor_candidates)


def make_callbacks(config: NuRecConfig) -> list[Callback]:
    callbacks = [
        _EpochEndModelCheckpoint(
            dirpath=config.ckpt_dir,
            filename="{step:06d}",
            save_on_train_epoch_end=config.checkpoint.save_on_train_epoch_end,
            every_n_train_steps=config.checkpoint.every_n_train_steps,
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_top_k=config.checkpoint.save_top_k,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if config.trainer.enable_progress_bar:
        callbacks.append(SmoothedProgressBar(refresh_rate=1))

    callbacks.append(TimingLogger())

    if os.environ.get("PID_PATH", ""):
        callbacks.append(
            CheckpointAndExitOnSlurmTimeoutCallback(
                config.ckpt_dir if config.checkpoint.save_on_preemption else None, os.environ["PID_PATH"]
            )
        )

    if config.mode == "trainval" and config.force_validate:
        callbacks.append(ForceValidateCallback())

    if config.fixed_camera_render.enabled:
        callbacks.append(
            FixedCameraRenderCallback(
                cfg=config.fixed_camera_render,
                out_dir=config.out_dir,
                run_id=config.run_id,
            )
        )

    return callbacks


class ForceValidateCallback(Callback):
    """
    Forces validation to happen at least once at the end of training if the mode is "trainval". This is useful for resuming
    preemptable runs since otherwise resuming a checkpoint interrupted after training but before validation finished
    will not re-run validation.
    """

    def __init__(self):
        self.has_validated = False

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.has_validated = True

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.getLogger(__name__).info("Validation has not yet run, forcing validation")

        if not self.has_validated:
            assert isinstance(trainer.lightning_module, BaseSystem)
            trainer.validate(trainer.lightning_module, datamodule=trainer.lightning_module.datamodule)


class PreemptionInterruptException(Exception):
    """
    Exception to be raised when the training process is preempted (e.g. by a signal).
    This will be raised from the CheckpointAndExitOnSignalCallback and its subclasses.
    """

    def __init__(self, message: str = "Preempted"):
        super().__init__(message)
        self.message = message


class CheckpointAndExitOnSignalCallback(Callback):
    """
    A callback to automatically save a checkpoint and exit the training process when a signal is received.
    """

    def __init__(
        self,
        ckpt_dir_path: str | None,
        signal_type: int,
        check_signal_every_n_batches: int = 5,
    ):
        self.preempting = False
        self.ckpt_dir_path = ckpt_dir_path
        self.check_signal_every_n_batches = check_signal_every_n_batches
        self.signal_type = signal_type
        signal.signal(self.signal_type, self._mark_preempting)

    def _mark_preempting(self, sig: int, frame: Optional[FrameType]) -> None:
        logger.info(f"Received signal {sig}, marking run as preempting")
        self.preempting = True

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if batch_idx % self.check_signal_every_n_batches == 0:
            self._check_preempting(trainer)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx % self.check_signal_every_n_batches == 0:
            self._check_preempting(trainer)

    def _check_preempting(self, trainer: Trainer) -> None:
        # Broadcast the preempting state:
        self.preempting = trainer.strategy.broadcast(self.preempting, src=0)

        if self.preempting:
            logger.info("Saving checkpoint and exiting in response to the preempting signal")

            # Set both the train and val loop to a clean state:
            # - For the train loop, mock completion of the batch by advancing "completed" to "processed".
            #   This is mainly to fix a bug in PL2.4 where the current batch might be processed again.
            #   Fixed in PL2.6 via https://github.com/Lightning-AI/pytorch-lightning/pull/20379
            # - For the val loop, simply reset so it could be re-run.
            # Here we're free to alter PL's internal state since we're exiting anyway.
            train_progress = trainer.fit_loop.epoch_loop.batch_progress
            train_progress.total.completed = train_progress.total.processed  # type: ignore[attr-defined]
            train_progress.current.completed = train_progress.current.processed  # type: ignore[attr-defined]
            val_progress = trainer.fit_loop.epoch_loop.val_loop.batch_progress
            val_progress.reset()

            if self.ckpt_dir_path is not None:
                # First write to a temporary checkpoint, then rename it to the final checkpoint.
                # This is to ensure that the checkpoint is not corrupted in case of an unexpected exit
                # (e.g. job gets killed while saving -- note this is less likely to happen since we have 300s but just in case).
                notify_before_checkpoint_capture(trainer)
                trainer.save_checkpoint(f"{self.ckpt_dir_path}/last.ckpt.tmp")
                if trainer.is_global_zero:
                    os.replace(f"{self.ckpt_dir_path}/last.ckpt.tmp", f"{self.ckpt_dir_path}/last.ckpt")
                logger.info(f"Checkpoint saved to {self.ckpt_dir_path}/last.ckpt")

            # Close the tracker run **after** the checkpoint is saved: it can hang while
            # uploading files, which would time the job out.
            backend = media_backend()
            if trainer.is_global_zero and backend is not None:
                backend.finalize_preempted_run()

            # We cannot use trainer.should_stop = True because it will trigger additional validation runs
            raise PreemptionInterruptException()

        # Due to unknown reason, when using the mamba backbone, the signal cannot be detected properly.
        # This might be due to mamba JIT's complicated implementation destroys the signal handler.
        # We find that setting this up again regularly can fix the issue.
        signal.signal(self.signal_type, self._mark_preempting)


class TimeBasedCheckpointCallback(Callback):
    """
    Save a rolling checkpoint every N minutes of wall-clock training time.
    """

    def __init__(
        self,
        ckpt_dir_path: str,
        every_n_minutes: int,
        filename: str = "last.ckpt",
    ) -> None:
        if every_n_minutes <= 0:
            raise ValueError(f"every_n_minutes must be > 0, got {every_n_minutes}")
        self.ckpt_dir_path = ckpt_dir_path
        self.every_n_seconds = every_n_minutes * 60
        self.filename = filename
        self._last_save_time: Optional[float] = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Anchor the timer at fit start so the first save fires N minutes in.
        self._last_save_time = time.monotonic()

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if self._last_save_time is None:
            return

        # Decide on rank 0, broadcast so save_checkpoint is collective in DDP.
        should_save = (time.monotonic() - self._last_save_time) >= self.every_n_seconds
        should_save = trainer.strategy.broadcast(should_save, src=0)
        if not should_save:
            return

        # At on_train_batch_end, processed > completed (PL increments completed AFTER this hook).
        # Saving in this state would make the resumed run re-execute the just-finished batch, so we
        # advance completed to match processed for the duration of save_checkpoint and restore the
        # real values right after — training continues from the unmodified counters in this run.
        train_progress = trainer.fit_loop.epoch_loop.batch_progress
        old_total_completed = train_progress.total.completed
        old_current_completed = train_progress.current.completed
        train_progress.total.completed = train_progress.total.processed  # type: ignore[attr-defined]
        train_progress.current.completed = train_progress.current.processed  # type: ignore[attr-defined]

        path = os.path.join(self.ckpt_dir_path, self.filename)
        tmp_path = f"{path}.tmp"
        try:
            notify_before_checkpoint_capture(trainer)
            trainer.save_checkpoint(tmp_path)
            if trainer.is_global_zero:
                os.replace(tmp_path, path)
                logger.info(f"Time-based checkpoint saved to {path} (every {self.every_n_seconds // 60} min)")
        finally:
            train_progress.total.completed = old_total_completed  # type: ignore[attr-defined]
            train_progress.current.completed = old_current_completed  # type: ignore[attr-defined]

        # Reset timer relative to the completion of this save so latency from
        # large-checkpoint writes doesn't shorten the next interval.
        self._last_save_time = time.monotonic()


class CheckpointAndExitOnSlurmTimeoutCallback(CheckpointAndExitOnSignalCallback):
    """
    A callback to automatically save a checkpoint and exit the training process on Slurm cluster.
    This is supposed to be used in junction with the cluster toolbox, so that the job could be properly requeued.
    The process is done by writing the pid of the training process to the specified path, which is used by the slurm script
    to send a SIGUSR1 signal to the training process before the job times out. The training process then marks the
    wandb run as preempting, which will cause the wandb agent to requeue the run, and terminate with a non-zero exit code
    (which is needed for wandb to requeue the run).

    The entire process follows the following execution flow:

    --------------> X                      X
    (TRAINING)      (300s before 4h limit) (SIGUSR1 is passed through bash to training processes)

    ----------------------------> X
    (After current batch ends)    (This callback marks preempting=True, saves the checkpoint, and marks trainer to stop)

    ------------------------------------> X ---------------(around 10s poll)------------> X
    (Trainer finishes cleaning up)        (Process does not exist anymore)                (Call scontrol requeue)

    X
    (requeue happens and current slurm job is terminated)

    This has been tested to work with multi-node training.

    Known Limitations:
    - Even when we set check_signal_every_n_batches=1, in cases where the dataloader hangs (e.g. due to slow S3 downloading) and
    we cannot get to on_**_batch_end step, the checkpoints will still not be saved. Fortunately such hanging usually happens just
    due to latency at the beginning of each epoch, so no work will be lost.
    """

    def __init__(
        self,
        ckpt_dir_path: str | None,
        pid_path: str,
        check_signal_every_n_batches: int = 5,
    ):
        # Note that you might see the following in the logs:
        #   SLURM auto-requeueing enabled. Setting signal handlers.
        # This message originates from PL's SlurmClusterEnvironment plugin, which gets automatically set-up.
        # That plugin is however not use-able since 'scontrol requeue' is not available in the docker container that we run in.
        # Here we make it disfunctional by completely **rewriting** the signal handler for SIGUSR1.
        logger.info(
            f"Registering signal handler for SIGUSR1, replacing existing handler {signal.getsignal(signal.SIGUSR1)}"
        )

        super().__init__(ckpt_dir_path, signal.SIGUSR1, check_signal_every_n_batches)

        # Write the pid of the training process so the external driving script can send SIGUSR1 to it.
        # Signal is only sent to the process with global rank 0 (hence the PID writing); state is broadcast by the DDP strategy.
        if int(os.environ["SLURM_PROCID"]) == 0:
            pid = str(os.getpid())
            logger.info(f"[Main Rank] Writing pid {pid} to {pid_path}")
            with open(pid_path, "w") as f:
                f.write(pid)


class VersionedCallback(Callback):
    def __init__(self, save_root, version=None, use_version=True):
        self.save_root = save_root
        self._version = version
        self.use_version = use_version

    @property
    def version(self) -> int:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        existing_versions = []
        if os.path.isdir(self.save_root):
            for f in os.listdir(self.save_root):
                bn = os.path.basename(f)
                if bn.startswith("version_"):
                    dir_ver = os.path.splitext(bn)[0].split("_")[1].replace("/", "")
                    existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0
        return max(existing_versions) + 1

    @property
    def savedir(self):
        if not self.use_version:
            return self.save_root
        return os.path.join(
            self.save_root, self.version if isinstance(self.version, str) else f"version_{self.version}"
        )


class CodeSnapshotCallback(VersionedCallback):
    def __init__(self, save_root: str, version=None, use_version=True):
        super().__init__(save_root, version, use_version)

    def get_file_list(self):
        return [
            b.decode()
            for b in set(subprocess.check_output("git ls-files", shell=True).splitlines())
            | set(subprocess.check_output("git ls-files --others --exclude-standard", shell=True).splitlines())
        ]

    def save_code_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        for f in self.get_file_list():
            if not os.path.exists(f) or os.path.isdir(f):
                continue
            os.makedirs(os.path.join(self.savedir, os.path.dirname(f)), exist_ok=True)
            shutil.copyfile(f, os.path.join(self.savedir, f))

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        try:
            self.save_code_snapshot()
        except Exception:
            logger.warning(
                "Code snapshot is not saved. Please make sure you have git installed and are in a git repository."
            )


def _make_wandb_logger(config: WandbLoggerConfig) -> WandbLogger:
    logger = WandbLogger(
        name=config.run_name,
        save_dir=config.save_dir or ".",
        offline=config.offline,
        id=config.run_id,
        project=config.project,
        anonymous=config.anonymous,
        log_model=config.log_model,
        group=config.group,
        tags=config.tags,
        job_type=config.job_type,
        entity=config.entity,
    )

    if not isinstance(logger.experiment, DummyExperiment):
        config.run_name = logger.experiment.name

        wandb_config: Dict = config_to_primitive(config.to_dictconfig())

        # Allow user to quickly associate and find the original maglev workflows that created this run.
        # MagLev environment variables are documented here:
        # https://maglev.nvda.ai/docs/components/workflows/defining-workflows/setting-up-environment-variables.md
        if os.environ.get("WORKFLOW_NAME", ""):
            base_url = "https://maglev.nvda.ai/ide/workflows"
            wf = os.environ.get("WORKFLOW_NAME", "")
            run = os.environ.get("WORKFLOW_RUN_ID", "")
            task = os.environ.get("WORKFLOW_TASK_NAME", "")
            job = os.environ.get("WORKFLOW_JOB_ID", "")
            wandb_config["maglev_url"] = f"{base_url}/workflow/{wf}/run/{run}/task/{task}/job/{job}"

        logger.experiment.config.update(wandb_config, allow_val_change=True)

    return logger


def _make_tensorboard_logger(config: TensorboardLoggerConfig) -> TensorBoardLogger:
    return TensorBoardLogger(
        save_dir=config.save_dir or ".",
        name=config.run_name,
        version=config.run_id,
        log_graph=config.log_graph,
        default_hp_metric=config.default_hp_metric,
        prefix=config.prefix,
    )


def make_logger(config: LoggerConfigType) -> Logger:
    if isinstance(config, DummyLoggerConfig):
        return DummyLogger()

    # Wandb expects the folder to exist already
    # However, for non-rank-zero workers the logger is already replaced by DummyLogger so no need to create dir.
    rank_zero_only(os.makedirs)(config.save_dir or ".", exist_ok=True)

    if isinstance(config, WandbLoggerConfig):
        return _make_wandb_logger(config)
    elif isinstance(config, TensorboardLoggerConfig):
        return _make_tensorboard_logger(config)
    else:
        raise TypeError(f"Unknown config type {type(config)=}.")


class FixedCameraRenderCallback(Callback):
    """Periodically render a fixed (camera, frame) view during training.

    Saves one JPEG per scheduled fire and stitches an MP4 at on_train_end.
    Rank-0 only; non-zero ranks are silent. Failures during render are
    logged but never interrupt training.
    """

    def __init__(
        self,
        cfg: FixedCameraRenderConfig,
        out_dir: str,
        run_id: str,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._frames_dir = os.path.join(out_dir, run_id, "fixed_camera_render")
        # One cached batch per configured frame_idx. Sorted iteration is
        # enforced by always walking sorted(self._cfg.frame_idxs).
        self._cached_batches: Dict[int, Any] = {}
        self._disabled: bool = False
        # Per-frame counts; total saved frames = sum(self._frames_written.values()).
        self._frames_written: Dict[int, int] = {}
        self._initial_render_done: bool = False

    def _frame_subdir(self, frame_idx: int) -> str:
        """Per-camera-frame sub-dir holding all per-view step JPEGs."""
        return os.path.join(self._frames_dir, f"frame_{frame_idx:04d}")

    def _frame_view_video_path(self, frame_idx: int, view: str) -> str:
        return os.path.join(self._frames_dir, f"frame_{frame_idx:04d}_{view}.mp4")

    def _render_view(
        self,
        view: str,
        pred_hwc_u8: np.ndarray,
        batch_on_device: Any,
    ) -> Optional[np.ndarray]:
        """Compute the (H, W, 3) uint8 image for a named view, or return None to skip.

        Add a branch here to support new views (e.g. "semantic", "depth"). The
        forward pass result is reused across views via `pred_hwc_u8`; views that
        need GT or other batch-side data pull it from `batch_on_device`.
        """
        if view == "rgb":
            return pred_hwc_u8

        if view == "residual":
            labels = batch_on_device.data.camera.labels
            gt_tensor = getattr(labels, "rgb", None) if labels is not None else None
            if gt_tensor is None:
                return None  # GT absent — skip silently (e.g. inference-only batches)
            gt_hwc = gt_tensor[0].detach().cpu().numpy()  # (h, w, 3) in [0,1]
            pred_hwc_f32 = pred_hwc_u8.astype(np.float32) / 255.0
            residual = np.abs(pred_hwc_f32 - gt_hwc).mean(axis=-1)  # (h, w)
            residual_rgb = scalar2img(residual, vmin=0.0, vmax=0.3, cmap="turbo")
            # Black out pixels training ignores (RayFlags.INVALID — ego mask
            # plus lens edges). Matches nurec/libs/losses/models/losses_module.py:209.
            flags_tensor = getattr(labels, "flags", None) if labels is not None else None
            if flags_tensor is not None:
                invalid_mask = (flags_tensor[0].squeeze(-1) & int(RayFlags.INVALID.value)) != 0  # (h, w) bool tensor
                residual_rgb[invalid_mask.detach().cpu().numpy()] = 0
            return residual_rgb

        logger.warning("FixedCameraRenderCallback: unknown view %r; skipping", view)
        return None

    def _log_image_to_remote(
        self,
        trainer: Trainer,
        frame_idx: int,
        view: str,
        img: np.ndarray,
        step: int,
    ) -> None:
        """Log a rendered image to the remote experiment tracker, if attached.

        Routed off the configured logger (no extra config knob): W&B and
        TensorBoard get a streamed copy of every JPEG that lands on disk,
        keyed by (frame_idx, view) so the in-tracker timeline mirrors the
        local step_*_<view>.jpg sequence. The W&B route needs both an attached
        WandbLogger and a registered media backend; without the backend the
        exception is logged and the image is not sent.
        """
        match trainer.logger:
            case WandbLogger():
                try:
                    backend = media_backend()
                    if backend is None:
                        raise RuntimeError("no media backend is registered")
                    backend.log_image(trainer.logger, f"fixed_camera_render/frame_{frame_idx:04d}/{view}", img, step)
                except Exception:  # noqa: BLE001 - tracker failures must never interrupt training
                    logger.exception(
                        "FixedCameraRenderCallback: W&B image log failed for frame_idx=%d view=%s step=%d",
                        frame_idx,
                        view,
                        step,
                    )
            case TensorBoardLogger():
                try:
                    trainer.logger.experiment.add_image(
                        f"fixed_camera_render/frame_{frame_idx:04d}/{view}",
                        img,
                        global_step=step,
                        dataformats="HWC",
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "FixedCameraRenderCallback: TensorBoard image log failed for frame_idx=%d view=%s step=%d",
                        frame_idx,
                        view,
                        step,
                    )
            case _:
                pass

    def _log_video_to_remote(
        self,
        trainer: Trainer,
        frame_idx: int,
        view: str,
        path: str,
    ) -> None:
        """Log an assembled MP4 to the remote tracker, if a WandbLogger is attached.

        Needs both an attached WandbLogger and a registered media backend; without
        the backend the exception is logged and the video is not sent.

        TensorBoard's add_video expects an NCHW tensor — we'd have to re-decode
        the MP4 to a tensor to use it. Skip TB videos until there's a need.
        """
        match trainer.logger:
            case WandbLogger():
                try:
                    backend = media_backend()
                    if backend is None:
                        raise RuntimeError("no media backend is registered")
                    backend.log_video(
                        trainer.logger, f"fixed_camera_render/videos/frame_{frame_idx:04d}_{view}", path, fmt="mp4"
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "FixedCameraRenderCallback: W&B video log failed for frame_idx=%d view=%s",
                        frame_idx,
                        view,
                    )
            case _:
                pass

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Setup runs on EVERY rank — all ranks must cache batches so they can
        # participate in the collective forward pass during render (see
        # on_train_batch_end). Side effects (makedirs, info log) are rank-0 only.
        # The dataset is deterministic, so all ranks resolve to the same set of
        # targets and either all succeed or all fail.
        if trainer.is_global_zero:
            os.makedirs(self._frames_dir, exist_ok=True)

        try:
            # The train dataloader's dataset (NCORETrainDataset) samples random
            # rays and does not expose per-frame indexing. Use the train-sequential
            # split (NCORESequentialDataset), which carries `camera_frames`.
            # The asserts narrow pl_module / dataset so mypy can resolve the
            # NuRec-specific attributes (`.datamodule` to `SODataModule`,
            # `.dataset` to `NCORESequentialDataset` with `camera_frames` etc.).
            assert isinstance(pl_module, BaseSystemSO)
            dataset = pl_module.datamodule.train_dataloader_sequential().dataset
            assert isinstance(dataset, NCORESequentialDataset)

            sorted_frame_idxs = sorted(set(self._cfg.frame_idxs))
            orig_subsample = dataset.n_image_subsample
            for frame_idx in sorted_frame_idxs:
                linear_idx = _compute_target_linear_idx(
                    list(dataset.camera_ids),
                    dataset.camera_frames,
                    self._cfg.camera_id,
                    frame_idx,
                )
                if linear_idx is None:
                    if trainer.is_global_zero:
                        available = {
                            cid: ([*frames[:3], "..."] if len(frames) > 3 else list(frames))
                            for cid, frames in dataset.camera_frames.items()
                        }
                        logger.warning(
                            "FixedCameraRenderCallback: target (camera_id=%s, frame_idx=%d) not "
                            "found in train-sequential split. Disabling. Available (first 3 frames "
                            "per camera): %s",
                            self._cfg.camera_id,
                            frame_idx,
                            available,
                        )
                    # A single missing target disables the whole callback.
                    # Deterministic across ranks (same dataset everywhere).
                    self._disabled = True
                    self._cached_batches.clear()
                    return

                # One-shot fetch at our configured subsample. Mutates the local
                # dataset copy on the main process between training batches;
                # DataLoader workers hold forked copies and are unaffected.
                try:
                    dataset.n_image_subsample = self._cfg.subsample
                    self._cached_batches[frame_idx] = dataset[(linear_idx, None)]
                finally:
                    dataset.n_image_subsample = orig_subsample

                self._frames_written[frame_idx] = 0

                if trainer.is_global_zero:
                    os.makedirs(self._frame_subdir(frame_idx), exist_ok=True)
                    cam = self._cached_batches[frame_idx].data.camera
                    # `cam.h` and `cam.w` are wrapped Optionals in this codebase; unwrap defensively.
                    h_val = int(cam.h) if not hasattr(cam.h, "value") else int(cam.h.value)
                    w_val = int(cam.w) if not hasattr(cam.w, "value") else int(cam.w.value)
                    logger.info(
                        "FixedCameraRenderCallback: resolved (camera_id=%s, frame_idx=%d) at "
                        "linear_idx=%d, render resolution %dx%d",
                        self._cfg.camera_id,
                        frame_idx,
                        linear_idx,
                        w_val,
                        h_val,
                    )

            if trainer.is_global_zero:
                logger.info(
                    "FixedCameraRenderCallback: %d frame(s) cached; schedule every %d steps from start_step=%d",
                    len(self._cached_batches),
                    self._cfg.every_n_steps,
                    self._cfg.start_step,
                )
        except Exception:  # noqa: BLE001 - we deliberately swallow so callback setup never kills training
            if trainer.is_global_zero:
                logger.exception("FixedCameraRenderCallback: setup failed; disabling callback.")
            self._disabled = True
            self._cached_batches.clear()

    def _render_and_save(self, trainer: Trainer, pl_module: LightningModule, step: int) -> None:
        """Render every cached frame at this training step.

        All ranks participate in each forward (DDP collective ops). Only rank 0
        writes JPEGs. Sorted iteration order is enforced so every rank issues
        collectives in the same sequence. Exceptions per-frame are caught and
        logged so a render failure on one frame never blocks the others or
        interrupts training.
        """
        for frame_idx in sorted(self._cached_batches.keys()):
            batch = self._cached_batches[frame_idx]
            try:
                batch_on_device = batch.to(pl_module.device, non_blocking=True)
                was_training = pl_module.training
                pl_module.eval()
                try:
                    with torch.no_grad():
                        result = pl_module(batch_on_device)
                finally:
                    if was_training:
                        pl_module.train()

                if not trainer.is_global_zero:
                    continue  # forward done; only rank 0 writes the JPEGs

                rendered_cam = result.rendered_cam
                rgb = rendered_cam.rgb if rendered_cam is not None else None
                if rgb is None:
                    logger.warning(
                        "FixedCameraRenderCallback: no camera RGB in forward result "
                        "(lidar-only config?); skipping frame_idx=%d at step=%d",
                        frame_idx,
                        step,
                    )
                    continue
                cam = batch_on_device.data.camera
                h_val = int(cam.h) if not hasattr(cam.h, "value") else int(cam.h.value)

                pred_hwc_u8 = rearrange(rgb.detach().cpu().numpy(), "(h w) c -> h w c", h=h_val)
                pred_hwc_u8 = np.clip(pred_hwc_u8 * 255.0, 0, 255).astype(np.uint8)

                subdir = self._frame_subdir(frame_idx)
                fired = False
                for view in self._cfg.views:
                    # Each view has its own guard so one bad view doesn't drop
                    # the others (and doesn't fire the surrounding "render
                    # failed" log, which is reserved for the forward call).
                    try:
                        img = self._render_view(view, pred_hwc_u8, batch_on_device)
                        if img is None:
                            continue
                        path = os.path.join(subdir, f"step_{step:08d}_{view}.jpg")
                        iio.imsave(path, img, quality=self._cfg.jpeg_quality)
                        self._log_image_to_remote(trainer, frame_idx, view, img, step)
                        fired = True
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "FixedCameraRenderCallback: view=%s save failed for frame_idx=%d at step=%d",
                            view,
                            frame_idx,
                            step,
                        )
                if fired:
                    self._frames_written[frame_idx] = self._frames_written.get(frame_idx, 0) + 1
            except Exception:  # noqa: BLE001 - we deliberately swallow to avoid interrupting training
                logger.exception(
                    "FixedCameraRenderCallback: render failed for frame_idx=%d at step=%d",
                    frame_idx,
                    step,
                )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # All ranks must follow the same control flow up to and including the
        # forward call — DDP forward synchronizes buffers across ranks, so a
        # rank-0-only forward would deadlock the others on their next sync point.
        # Only the JPEG save is rank-0-only.
        if self._disabled or not self._cached_batches:
            return

        # Initial-state snapshot on the FIRST batch-end. Done here (not in
        # on_train_start) because the model's sensor_model buffers are only
        # broadcast to all DDP ranks after Lightning's first training_step;
        # rendering earlier triggers an `original_width != resolution[0]`
        # assertion on non-rank-0 ranks (observed live, job 27762297).
        # Filename uses step=0 so it sorts first in each frame's video.
        if not self._initial_render_done:
            self._initial_render_done = True
            self._render_and_save(trainer, pl_module, 0)
            # Skip the schedule branch on this same batch-end call so we don't
            # re-render step 0 if Lightning's global_step is still 0 here
            # (depends on optimization mode; observed Lightning would write
            # an identical JPEG over the initial one).
            return

        step = int(pl_module.global_step)
        # The initial-render branch above already owns step 0, but under gradient
        # accumulation or manual optimization Lightning can keep global_step at 0
        # past the first batch-end — re-running the schedule here would overwrite
        # the initial snapshot with the post-mutation state.
        if step == 0:
            return
        if not _should_render(step, self._cfg.start_step, self._cfg.every_n_steps):
            return

        self._render_and_save(trainer, pl_module, step)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._disabled or not trainer.is_global_zero:
            return

        total_frames = sum(self._frames_written.values())
        if total_frames == 0:
            logger.info("FixedCameraRenderCallback: no frames were written, skipping video assembly.")
            return

        for frame_idx in sorted(self._cached_batches.keys()):
            if self._frames_written.get(frame_idx, 0) == 0:
                continue
            subdir = self._frame_subdir(frame_idx)

            for view in self._cfg.views:
                try:
                    paths = sorted(glob.glob(os.path.join(subdir, f"step_*_{view}.jpg")))
                    if not paths:
                        continue
                    out_path = self._frame_view_video_path(frame_idx, view)
                    # Stream JPEGs into the encoder one at a time so memory use
                    # stays O(1) in the number of saved steps — buffering the
                    # full sequence can OOM rank 0 on long runs or high-res views.
                    with iio.get_writer(out_path, fps=self._cfg.video_fps, macro_block_size=1) as writer:
                        for p in paths:
                            img = np.asarray(iio.imread(p))
                            # ffmpeg requires even dimensions; chop the last row/col.
                            if img.shape[0] % 2 == 1:
                                img = img[:-1, :]
                            if img.shape[1] % 2 == 1:
                                img = img[:, :-1]
                            writer.append_data(img)  # type: ignore[attr-defined]
                    self._log_video_to_remote(trainer, frame_idx, view, out_path)
                    logger.info(
                        "FixedCameraRenderCallback: wrote %d-frame %s video to %s",
                        len(paths),
                        view,
                        out_path,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "FixedCameraRenderCallback: video assembly failed for frame_idx=%d view=%s",
                        frame_idx,
                        view,
                    )
