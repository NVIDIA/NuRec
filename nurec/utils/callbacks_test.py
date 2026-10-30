# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import signal

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.logger import DummyLogger

from nurec.systems.base import BaseSystem, BaseSystemSO
from nurec.utils.callbacks import (
    CheckpointAndExitOnSignalCallback,
    PreemptionInterruptException,
    SmoothedProgressBar,
    TimeBasedCheckpointCallback,
    TimingLogger,
    _compute_target_linear_idx,
    _EpochEndModelCheckpoint,
    _should_render,
    make_callbacks,
    notify_before_checkpoint_capture,
)


class TestTimingLogger:
    @pytest.mark.parametrize("phase", ["train", "validation"])
    def test_resumed_epoch_without_start_preserves_elapsed_time(self, phase: str, caplog) -> None:
        callback = TimingLogger()
        trainer: Any = SimpleNamespace()
        pl_module: Any = SimpleNamespace()
        elapsed_attr = f"{phase}_elapsed_time_s"
        timer_attr = f"{phase}_epoch_start_end_timer_s"
        setattr(callback, elapsed_attr, 7.0)

        with patch("nurec.utils.callbacks.time.time", return_value=100.0):
            getattr(callback, f"on_{phase}_epoch_end")(trainer, pl_module)

        assert getattr(callback, elapsed_attr) == 7.0
        assert getattr(callback, timer_attr) is None
        assert f"{phase} epoch-start timer is None" in caplog.text
        assert "skipping elapsed-time accounting" in caplog.text

    @pytest.mark.parametrize("phase", ["train", "validation"])
    def test_complete_epochs_accumulate_elapsed_time(self, phase: str, caplog) -> None:
        callback = TimingLogger()
        trainer: Any = SimpleNamespace()
        pl_module: Any = SimpleNamespace()
        start = getattr(callback, f"on_{phase}_epoch_start")
        end = getattr(callback, f"on_{phase}_epoch_end")

        with patch("nurec.utils.callbacks.time.time", side_effect=[100.0, 103.0, 200.0, 207.0]):
            start(trainer, pl_module)
            end(trainer, pl_module)
            start(trainer, pl_module)
            end(trainer, pl_module)

        assert getattr(callback, f"{phase}_elapsed_time_s") == 10.0
        assert "skipping elapsed-time accounting" not in caplog.text

    @pytest.mark.parametrize(
        ("hook", "attr", "metric"),
        [
            ("on_train_end", "train_elapsed_time_s", "train/elapsed_time_s"),
            ("on_validation_end", "validation_elapsed_time_s", "test/elapsed_time_s"),
        ],
    )
    def test_elapsed_time_reaches_the_collector_as_wall_clock(self, hook: str, attr: str, metric: str) -> None:
        # The elapsed times are perf-counter spans; the metrics file must say so, or a consumer
        # cannot tell them from CUDA-event timings.
        callback = TimingLogger()
        setattr(callback, attr, 12.5)
        pl_module: Any = Mock(spec=BaseSystemSO)
        pl_module.logger = DummyLogger()

        getattr(callback, hook)(SimpleNamespace(), pl_module)

        pl_module.collect_metric.assert_called_once_with(metric, 12.5, timing_method="wall_clock")


def test_checkpoint_skips_topk_when_monitor_disappears_after_validation(tmp_path, caplog) -> None:
    callback = _EpochEndModelCheckpoint(
        dirpath=str(tmp_path),
        filename="{step:06d}",
        monitor="train/psnr",
        mode="max",
        save_top_k=1,
    )
    trainer: Any = SimpleNamespace(
        global_step=31_000,
        fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(val_loop=SimpleNamespace(_has_run=True))),
    )

    with patch.object(_EpochEndModelCheckpoint.__mro__[1], "_save_topk_checkpoint", autospec=True) as save_topk:
        callback._save_topk_checkpoint(trainer, {"epoch": torch.tensor(1), "step": torch.tensor(31_000)})

    assert "Skipping top-k checkpoint update" in caplog.text
    save_topk.assert_not_called()


def test_validation_has_run_handles_missing_pl_internal() -> None:
    trainer: Any = SimpleNamespace()

    assert not _EpochEndModelCheckpoint._validation_has_run(trainer)


class TestNotifyBeforeCheckpointCapture:
    """Pins the pre-capture notification, the only point at which a model change still reaches the saved file.

    The feature is three call sites plus one isinstance dispatch, so each is asserted on its own.
    """

    @staticmethod
    def _recording_system(events: list[str]) -> Any:
        """A lightning_module that satisfies the isinstance dispatch and records when it is notified."""
        system = Mock(spec=BaseSystem)
        system.on_before_checkpoint_capture.side_effect = lambda: events.append("notify")
        return system

    @staticmethod
    def _trainer(lightning_module: Any, events: list[str]) -> Any:
        """The slice of Trainer the three checkpoint writers touch, recording every save."""
        batch_progress = SimpleNamespace(
            total=SimpleNamespace(completed=0, processed=1),
            current=SimpleNamespace(completed=0, processed=1),
        )
        return SimpleNamespace(
            lightning_module=lightning_module,
            # Rank zero renames the temp file, which does not exist here, so pose as a non-zero rank.
            is_global_zero=False,
            strategy=SimpleNamespace(broadcast=lambda value, src=0: value),
            save_checkpoint=lambda path: events.append("save"),
            fit_loop=SimpleNamespace(
                epoch_loop=SimpleNamespace(
                    batch_progress=batch_progress,
                    val_loop=SimpleNamespace(batch_progress=SimpleNamespace(reset=lambda: None)),
                )
            ),
        )

    def test_epoch_end_checkpoint_notifies_before_capture(self, tmp_path) -> None:
        events: list[str] = []
        callback = _EpochEndModelCheckpoint(dirpath=str(tmp_path))
        trainer = self._trainer(self._recording_system(events), events)

        with patch.object(_EpochEndModelCheckpoint.__mro__[1], "_save_checkpoint", autospec=True) as save_checkpoint:
            save_checkpoint.side_effect = lambda *args, **kwargs: events.append("save")
            callback._save_checkpoint(trainer, str(tmp_path / "last.ckpt"))

        # PL reads the state dict inside the super call, so notifying after it would land in memory only.
        assert events == ["notify", "save"]

    def test_preemption_checkpoint_notifies_before_capture(self, tmp_path) -> None:
        events: list[str] = []
        with patch.object(signal, "signal"):  # the constructor installs a real process-wide handler
            callback = CheckpointAndExitOnSignalCallback(ckpt_dir_path=str(tmp_path), signal_type=signal.SIGUSR1)
        callback.preempting = True
        trainer = self._trainer(self._recording_system(events), events)

        with pytest.raises(PreemptionInterruptException):
            callback._check_preempting(trainer)

        assert events == ["notify", "save"]

    def test_time_based_checkpoint_notifies_before_capture(self, tmp_path) -> None:
        events: list[str] = []
        callback = TimeBasedCheckpointCallback(ckpt_dir_path=str(tmp_path), every_n_minutes=1)
        # Back-date the timer past any interval so this batch is due for a save.
        callback._last_save_time = float("-inf")
        trainer = self._trainer(self._recording_system(events), events)

        callback.on_train_batch_end(trainer, trainer.lightning_module, None, None, 0)

        assert events == ["notify", "save"]

    def test_a_module_that_is_not_a_base_system_is_left_alone(self) -> None:
        events: list[str] = []

        class _DuckTypedModule(LightningModule):
            def on_before_checkpoint_capture(self) -> None:
                events.append("notify")

        notify_before_checkpoint_capture(self._trainer(_DuckTypedModule(), events))

        # The dispatch keys off BaseSystem, not off the method name, and instant_nurec runs a module that is not one.
        assert events == []


@pytest.mark.parametrize("enabled", [False, True])
def test_make_callbacks_respects_progress_bar_setting(tmp_path, monkeypatch, enabled) -> None:
    monkeypatch.delenv("PID_PATH", raising=False)
    config: Any = SimpleNamespace(
        ckpt_dir=str(tmp_path),
        checkpoint=SimpleNamespace(
            save_on_train_epoch_end=False,
            every_n_train_steps=100,
            monitor=None,
            mode="min",
            save_top_k=0,
            save_on_preemption=False,
        ),
        trainer=SimpleNamespace(enable_progress_bar=enabled),
        mode="train",
        fixed_camera_render=SimpleNamespace(enabled=False),
    )

    callbacks = make_callbacks(config)

    assert any(isinstance(callback, SmoothedProgressBar) for callback in callbacks) is enabled


def test_epoch_end_handles_missing_saved_step_internal(tmp_path) -> None:
    callback = _EpochEndModelCheckpoint(dirpath=str(tmp_path), save_top_k=0)
    delattr(callback, "_last_global_step_saved")
    trainer: Any = SimpleNamespace(global_step=1)
    pl_module: Any = SimpleNamespace()

    with (
        patch.object(_EpochEndModelCheckpoint.__mro__[1], "on_train_epoch_end", autospec=True),
        patch.object(callback, "_should_skip_saving_checkpoint", return_value=True),
    ):
        callback.on_train_epoch_end(trainer, pl_module)

    assert not hasattr(callback, "_last_global_step_saved")


def test_epoch_end_records_successful_explicit_last_save(tmp_path) -> None:
    callback = _EpochEndModelCheckpoint(dirpath=str(tmp_path), save_last=True, save_top_k=0)
    callback._last_global_step_saved = 3
    trainer: Any = SimpleNamespace(global_step=4)
    pl_module: Any = SimpleNamespace()

    with (
        patch.object(_EpochEndModelCheckpoint.__mro__[1], "on_train_epoch_end", autospec=True),
        patch.object(callback, "_should_skip_saving_checkpoint", return_value=False),
        patch.object(callback, "_should_save_on_train_epoch_end", return_value=True),
        patch.object(callback, "_monitor_candidates", return_value={}),
        patch.object(callback, "_save_last_checkpoint") as save_last,
    ):
        callback.on_train_epoch_end(trainer, pl_module)

    save_last.assert_called_once_with(trainer, {})
    assert callback._last_global_step_saved == trainer.global_step


def test_epoch_end_restores_saved_step_when_explicit_last_save_fails(tmp_path) -> None:
    callback = _EpochEndModelCheckpoint(dirpath=str(tmp_path), save_last=True, save_top_k=0)
    callback._last_global_step_saved = 3
    trainer: Any = SimpleNamespace(global_step=4)
    pl_module: Any = SimpleNamespace()

    with (
        patch.object(_EpochEndModelCheckpoint.__mro__[1], "on_train_epoch_end", autospec=True),
        patch.object(callback, "_should_skip_saving_checkpoint", return_value=False),
        patch.object(callback, "_should_save_on_train_epoch_end", return_value=True),
        patch.object(callback, "_monitor_candidates", return_value={}),
        patch.object(callback, "_save_last_checkpoint", side_effect=RuntimeError("checkpoint write failed")),
        pytest.raises(RuntimeError, match="checkpoint write failed"),
    ):
        callback.on_train_epoch_end(trainer, pl_module)

    assert callback._last_global_step_saved == 3


@pytest.mark.parametrize("checkpoint_interval", [1, 2])
def test_epoch_end_last_checkpoint_contains_post_batch_module_state(tmp_path, checkpoint_interval: int) -> None:
    import pytorch_lightning as pl

    from torch.utils.data import DataLoader, TensorDataset

    class _PostBatchMutationModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.register_buffer("post_batch_mutation", torch.zeros(()))

        def training_step(self, batch, batch_idx):
            del batch_idx
            return self.weight * batch[0].mean()

        def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
            del outputs, batch, batch_idx
            self.register_buffer("post_batch_mutation", torch.ones(()))

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.0)

    checkpoint = _EpochEndModelCheckpoint(
        dirpath=str(tmp_path),
        every_n_train_steps=checkpoint_interval,
        save_last=True,
        save_top_k=0,
        save_on_train_epoch_end=True,
    )
    trainer = pl.Trainer(
        max_steps=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=[checkpoint],
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        _PostBatchMutationModule(),
        DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1),
    )

    last_checkpoint = torch.load(tmp_path / "last.ckpt", weights_only=False, map_location="cpu")
    torch.testing.assert_close(last_checkpoint["state_dict"]["post_batch_mutation"], torch.tensor(1.0))


class TestComputeTargetLinearIdx:
    def test_target_camera_is_first(self):
        camera_ids = ["camera_front_wide_120fov", "camera_rear"]
        camera_frames = {
            "camera_front_wide_120fov": [100, 200, 300, 400],
            "camera_rear": [50, 150, 250],
        }
        idx = _compute_target_linear_idx(camera_ids, camera_frames, "camera_front_wide_120fov", 300)
        assert idx == 2  # third entry in first camera

    def test_target_camera_is_second(self):
        camera_ids = ["camera_front_wide_120fov", "camera_rear"]
        camera_frames = {
            "camera_front_wide_120fov": [100, 200, 300, 400],
            "camera_rear": [50, 150, 250],
        }
        idx = _compute_target_linear_idx(camera_ids, camera_frames, "camera_rear", 150)
        assert idx == 4 + 1  # 4 entries before + index 1 within second camera

    def test_target_camera_missing(self):
        camera_ids = ["camera_front_wide_120fov"]
        camera_frames = {"camera_front_wide_120fov": [100, 200]}
        idx = _compute_target_linear_idx(camera_ids, camera_frames, "camera_nonexistent", 100)
        assert idx is None

    def test_target_frame_missing(self):
        camera_ids = ["camera_front_wide_120fov"]
        camera_frames = {"camera_front_wide_120fov": [100, 200]}
        idx = _compute_target_linear_idx(camera_ids, camera_frames, "camera_front_wide_120fov", 999)
        assert idx is None


class TestShouldRender:
    @pytest.mark.parametrize(
        "step,start,every,expected",
        [
            (0, 0, 100, True),
            (99, 0, 100, False),
            (100, 0, 100, True),
            (199, 0, 100, False),
            (200, 0, 100, True),
            (49, 50, 100, False),  # before start_step
            (50, 50, 100, True),  # at start_step
            (149, 50, 100, False),
            (150, 50, 100, True),
            (0, 50, 100, False),
        ],
    )
    def test_schedule(self, step, start, every, expected):
        assert _should_render(step, start, every) is expected


# End-to-end exercise of the `logger=tensorboard` path via NuRec's make_logger entry point + a 2-step PL trainer.fit. Fails without tensorboardX (not a transitive PL dep).
def test_tensorboard_logger_in_thin_trainer_fit(tmp_path) -> None:
    import pytorch_lightning as pl

    from torch.utils.data import DataLoader, TensorDataset

    from nurec.config.logger import TensorboardLoggerConfig
    from nurec.utils.callbacks import make_logger

    class _M(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(1, 1)

        def training_step(self, batch, _):
            x, y = batch
            loss = ((self.lin(x) - y) ** 2).mean()
            self.log("train/loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    cfg = TensorboardLoggerConfig(
        name="tensorboard",
        save_dir=str(tmp_path),
        run_name="t",
        log_graph=False,
        default_hp_metric=False,
        prefix="",
    )
    logger = make_logger(cfg)
    trainer = pl.Trainer(
        max_steps=2,
        logger=logger,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(_M(), DataLoader(TensorDataset(torch.randn(8, 1), torch.randn(8, 1)), batch_size=2))
    events = list(tmp_path.rglob("events.out.tfevents.*"))
    assert events and events[0].stat().st_size > 0
