# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import logging
import os
import textwrap
import types

from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import nurec.systems.base as base_module

from apps.av.road_mesh.ground_mesh import GroundMeshReconstructionSettings
from nurec.libs.losses.orchestration.config import LossAggregatorBatchReturn, LossAggregatorReturn, LossReturn
from nurec.models.calib import BaseCalib
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.systems.base import BaseSystem, BaseSystemSO, DDPStrategySO, _loss_roots_by_name
from nurec.utils.batch import FrameMeta
from nurec.utils.io.artifact_cache import ArtifactCache, publish_existing_usdz
from nurec.utils.io.mesh import Mesh
from nurec.utils.metrics import MetricsCollector
from nurec.utils.types import NamedSerialized


class _TestSystem(BaseSystemSO):
    def forward(self, batch):
        raise NotImplementedError

    def training_losses(self, batch, batch_local_idx):
        raise NotImplementedError


class _FakeDataSource:
    def get_camera_sensor_ids(self, unique_sensors=True):
        return ["camera_front_wide_120fov"]

    def get_lidar_sensor_ids(self, unique_sensors=True):
        return ["lidar_gt_top_p128"]


def test_loss_roots_group_batch_elements_by_loss_name() -> None:
    """Group each batch element's reduced loss under its stable name."""
    first_rgb = torch.tensor(1.0, requires_grad=True)
    first_depth = torch.tensor(2.0, requires_grad=True)
    second_rgb = torch.tensor(3.0, requires_grad=True)
    first_rgb_return = LossReturn(
        name="rgb",
        lambda_=1.0,
        value=first_rgb,
        reduce_fn=lambda value, **_kwargs: value,
    )
    first_depth_return = LossReturn(
        name="depth",
        lambda_=1.0,
        value=first_depth,
        reduce_fn=lambda value, **_kwargs: value,
    )
    second_rgb_return = LossReturn(
        name="rgb",
        lambda_=1.0,
        value=second_rgb,
        reduce_fn=lambda value, **_kwargs: value,
    )
    batch_return = LossAggregatorBatchReturn(
        [
            LossAggregatorReturn(
                {
                    "rgb": first_rgb_return,
                    "depth": first_depth_return,
                }
            ),
            LossAggregatorReturn({"rgb": second_rgb_return}),
        ]
    )

    roots = _loss_roots_by_name(batch_return)

    assert set(roots) == {"rgb", "depth"}
    assert roots["rgb"][0] is first_rgb_return.reduced_value
    assert roots["rgb"][1] is second_rgb_return.reduced_value
    assert roots["depth"][0] is first_depth_return.reduced_value


def test_training_step_keeps_exactly_one_total_manual_backward_call() -> None:
    """training_step still calls manual_backward once on the summed total."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(BaseSystem.training_step)))
    manual_backward_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "manual_backward"
    ]

    assert len(manual_backward_calls) == 1
    argument = manual_backward_calls[0].args[0]
    assert isinstance(argument, ast.Attribute)
    assert argument.attr == "total_value"
    assert isinstance(argument.value, ast.Name)
    assert argument.value.id == "loss_return"


def test_setup_distributed_recreates_gaussian_group_after_world(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    gaussian_groups = iter((object(), object()))
    configured_timeout = object()

    def make_strategy() -> DDPStrategySO:
        # __init__ needs a live cluster environment; only _timeout is read below.
        strategy = object.__new__(DDPStrategySO)
        strategy._timeout = cast(Any, configured_timeout)
        return strategy

    # Assert the world init is delegated to pytorch-lightning, not how it works.
    monkeypatch.setattr(base_module.DDPStrategy, "setup_distributed", lambda _self: events.append("init_world"))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def new_group(*, backend: str, timeout: object) -> object:
        assert backend == "nccl"
        assert timeout is configured_timeout
        events.append("init_gaussian_group")
        return next(gaussian_groups)

    monkeypatch.setattr(torch.distributed, "new_group", new_group)

    first_strategy = make_strategy()
    second_strategy = make_strategy()
    first_strategy.setup_distributed()
    second_strategy.setup_distributed()

    # The Gaussian group is created after the world.
    assert events == ["init_world", "init_gaussian_group", "init_world", "init_gaussian_group"]
    assert second_strategy.gaussian_process_group is not first_strategy.gaussian_process_group


def test_teardown_releases_owned_gaussian_group(monkeypatch: pytest.MonkeyPatch) -> None:
    group = object()
    model = GaussiansComposite.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.gaussian_process_group = cast(Any, group)
    system = types.SimpleNamespace(model=model)
    strategy = object.__new__(DDPStrategySO)
    strategy._model = cast(Any, system)
    strategy.gaussian_process_group = cast(Any, group)
    destroyed: list[object] = []
    parent_teardowns: list[object] = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroyed.append)
    monkeypatch.setattr(base_module.DDPStrategy, "teardown", lambda self: parent_teardowns.append(self))

    strategy.teardown()

    assert destroyed == [group]
    assert model.gaussian_process_group is None
    assert strategy.gaussian_process_group is None
    assert parent_teardowns == [strategy]


def test_teardown_releases_group_after_partial_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    group = object()
    strategy = object.__new__(DDPStrategySO)
    strategy._model = None
    strategy.gaussian_process_group = cast(Any, group)
    destroyed: list[object] = []
    parent_teardowns: list[object] = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroyed.append)
    monkeypatch.setattr(base_module.DDPStrategy, "teardown", lambda self: parent_teardowns.append(self))

    strategy.teardown()

    assert destroyed == [group]
    assert strategy.gaussian_process_group is None
    assert parent_teardowns == [strategy]


class _SetupSystem(_TestSystem):
    @property
    def trainer(self) -> Any:
        return self.__dict__["_setup_trainer"]

    @trainer.setter
    def trainer(self, value: Any) -> None:
        self.__dict__["_setup_trainer"] = value


def _make_setup_system(group: Any) -> tuple[_SetupSystem, GaussiansComposite]:
    system = object.__new__(_SetupSystem)
    torch.nn.Module.__init__(system)
    model = GaussiansComposite.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.gaussian_process_group = None
    system.model = model
    strategy = object.__new__(DDPStrategySO)
    strategy.gaussian_process_group = group
    system.trainer = types.SimpleNamespace(strategy=strategy)
    return system, model


def test_setup_installs_strategy_owned_gaussian_group(monkeypatch: pytest.MonkeyPatch) -> None:
    group = cast(Any, object())
    system, model = _make_setup_system(group)
    parent_setups: list[tuple[str, Any]] = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        BaseSystem,
        "setup",
        lambda self, stage: parent_setups.append((stage, self.model.gaussian_process_group)),
    )

    BaseSystemSO.setup(system, "fit")

    assert model.gaussian_process_group is group
    assert parent_setups == [("fit", group)]


def test_setup_rejects_missing_gaussian_group(monkeypatch: pytest.MonkeyPatch) -> None:
    system, _model = _make_setup_system(None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    parent_setup = Mock()
    monkeypatch.setattr(BaseSystem, "setup", parent_setup)

    with pytest.raises(RuntimeError, match="requires the dedicated process group"):
        BaseSystemSO.setup(system, "fit")

    parent_setup.assert_not_called()


@pytest.mark.parametrize("materialize_grad", [True, False], ids=["with-grad", "without-grad"])
def test_training_step_launches_and_resolves_optimizer_preparation_around_backward(materialize_grad: bool) -> None:
    events: list[str] = []
    parameter = torch.nn.Parameter(torch.ones(()))

    class _Timer:
        def start(self) -> None:
            pass

        def stop(self, force: bool = False) -> None:
            del force

    class _Optimizer:
        param_groups = [{"params": [parameter]}]

        def zero_grad(self) -> None:
            parameter.grad = None

    optimizer = _Optimizer()

    class _Strategy:
        lr_scheduler_configs: list[object] = []

        def reduce_boolean_decision(self, decision: bool, *, all: bool) -> bool:
            del all
            return decision

        def optimizer_step(self, stepped_optimizer, closure) -> None:
            assert stepped_optimizer is optimizer
            del closure
            events.append("optimizer_step")

    strategy = _Strategy()
    progress = types.SimpleNamespace(
        increment_ready=lambda: None,
        increment_completed=lambda: None,
    )
    trainer = types.SimpleNamespace(
        strategy=strategy,
        should_stop=False,
        global_rank=0,
        profiler=types.SimpleNamespace(start=lambda _name: None, stop=lambda _name: None),
        fit_loop=types.SimpleNamespace(
            epoch_loop=types.SimpleNamespace(
                manual_optimization=types.SimpleNamespace(optim_step_progress=progress),
            )
        ),
    )
    pl_optimizer = types.SimpleNamespace(
        optimizer=optimizer,
        _optimizer=optimizer,
        _strategy=strategy,
    )

    def manual_backward(_loss: torch.Tensor) -> None:
        events.append("manual_backward")
        if materialize_grad:
            parameter.grad = torch.ones_like(parameter)

    harness = types.SimpleNamespace(
        outer_loop_timer=_Timer(),
        inner_loop_timer=_Timer(),
        global_step=0,
        trainer=trainer,
        optimizers=lambda **_kwargs: pl_optimizer,
        training_losses=lambda _batch, _batch_idx: types.SimpleNamespace(
            total_value=torch.ones((), requires_grad=True)
        ),
        on_training_losses_return=lambda _loss: None,
        manual_backward=manual_backward,
        _launch_optimizers_async_preparation=lambda: events.append("launch_optimizers"),
        _prepare_optimizers_after_backward=lambda: events.append("prepare_optimizers"),
    )

    BaseSystem.training_step(cast(Any, harness), batch=cast(Any, None), batch_local_idx=0)

    expected = ["launch_optimizers", "manual_backward", "prepare_optimizers"]
    if materialize_grad:
        expected.append("optimizer_step")
    assert events == expected


def _run_training_step_with_scheduler(
    *,
    scheduler: object,
    interval: str,
    frequency: int,
    batch_local_idx: int,
    epoch_steps: tuple[int, int] | None,
    is_last_batch: bool = False,
    current_epoch: int = 0,
) -> None:
    parameter = torch.nn.Parameter(torch.ones(()))

    class _Timer:
        def start(self) -> None:
            pass

        def stop(self, force: bool = False) -> None:
            del force

    optimizer = types.SimpleNamespace(
        param_groups=[{"params": [parameter]}],
        zero_grad=lambda: setattr(parameter, "grad", None),
    )
    scheduler_config = types.SimpleNamespace(
        scheduler=scheduler,
        interval=interval,
        frequency=frequency,
        monitor="metric",
        strict=True,
    )

    class _Strategy:
        lr_scheduler_configs = [scheduler_config]

        @staticmethod
        def reduce_boolean_decision(decision: bool, *, all: bool) -> bool:
            del all
            return decision

    progress = types.SimpleNamespace(increment_ready=lambda: None, increment_completed=lambda: None)
    trainer = types.SimpleNamespace(
        strategy=_Strategy(),
        should_stop=False,
        global_rank=0,
        is_last_batch=is_last_batch,
        current_epoch=current_epoch,
        callback_metrics={"metric": torch.tensor(1.0)},
        profiler=types.SimpleNamespace(start=lambda _name: None, stop=lambda _name: None),
        fit_loop=types.SimpleNamespace(
            epoch_loop=types.SimpleNamespace(
                manual_optimization=types.SimpleNamespace(optim_step_progress=progress),
            )
        ),
    )
    pl_optimizer = types.SimpleNamespace(optimizer=optimizer, _optimizer=optimizer, _strategy=None)
    harness = types.SimpleNamespace(
        outer_loop_timer=_Timer(),
        inner_loop_timer=_Timer(),
        global_step=0,
        trainer=trainer,
        optimizers=lambda **_kwargs: pl_optimizer,
        training_losses=lambda _batch, _batch_idx: types.SimpleNamespace(
            total_value=torch.ones((), requires_grad=True)
        ),
        on_training_losses_return=lambda _loss: None,
        manual_backward=lambda _loss: setattr(parameter, "grad", torch.ones_like(parameter)),
        _launch_optimizers_async_preparation=lambda: None,
        _prepare_optimizers_after_backward=lambda: None,
        _lr_scheduler_epoch_steps=epoch_steps,
    )

    BaseSystem.training_step(cast(Any, harness), batch=cast(Any, None), batch_local_idx=batch_local_idx)


@pytest.mark.parametrize(("batch_local_idx", "expected_calls"), [(0, 3), (1, 3), (2, 4)])
def test_training_step_accelerates_step_lr_scheduler(batch_local_idx: int, expected_calls: int) -> None:
    scheduler = Mock()
    _run_training_step_with_scheduler(
        scheduler=scheduler,
        interval="step",
        frequency=1,
        batch_local_idx=batch_local_idx,
        epoch_steps=(10, 3),
    )
    assert scheduler.step.call_count == expected_calls


def test_training_step_preserves_step_scheduler_frequency_without_update() -> None:
    scheduler = Mock()
    _run_training_step_with_scheduler(
        scheduler=scheduler,
        interval="step",
        frequency=2,
        batch_local_idx=0,
        epoch_steps=None,
    )
    scheduler.step.assert_not_called()


def test_training_step_keeps_epoch_scheduler_unscaled() -> None:
    scheduler = Mock()
    _run_training_step_with_scheduler(
        scheduler=scheduler,
        interval="epoch",
        frequency=1,
        batch_local_idx=0,
        epoch_steps=(10, 3),
        is_last_batch=True,
    )
    scheduler.step.assert_called_once_with()


def test_training_step_rejects_accelerated_reduce_on_plateau() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    with pytest.raises(RuntimeError, match="ReduceLROnPlateau"):
        _run_training_step_with_scheduler(
            scheduler=scheduler,
            interval="step",
            frequency=1,
            batch_local_idx=0,
            epoch_steps=(10, 3),
        )


def test_training_step_allows_single_reduce_on_plateau_update() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    _run_training_step_with_scheduler(
        scheduler=scheduler,
        interval="step",
        frequency=2,
        batch_local_idx=1,
        epoch_steps=(4, 3),
    )
    assert scheduler.best == pytest.approx(1.0)


def _make_final_epoch_harness(
    *,
    is_trainer_attached: bool = True,
    should_stop: bool = False,
    global_step: int = 0,
    max_steps: int = -1,
    current_epoch: int = 0,
    max_epochs: int | None = None,
) -> Any:
    trainer = types.SimpleNamespace(
        should_stop=should_stop,
        global_step=global_step,
        max_steps=max_steps,
        current_epoch=current_epoch,
        max_epochs=max_epochs,
    )
    return cast(Any, types.SimpleNamespace(is_trainer_attached=is_trainer_attached, trainer=trainer))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, False),
        ({"is_trainer_attached": False, "should_stop": True}, False),
        ({"should_stop": True}, True),
        ({"global_step": 30, "max_steps": 30}, True),
        ({"global_step": 29, "max_steps": 30}, False),
        ({"global_step": 30, "max_steps": -1}, False),
        ({"current_epoch": 4, "max_epochs": 5}, True),
        ({"current_epoch": 3, "max_epochs": 5}, False),
        ({"current_epoch": 99, "max_epochs": None}, False),
        ({"current_epoch": 99, "max_epochs": -1}, False),
    ],
    ids=[
        "no-limits",
        "trainer-detached",
        "should-stop",
        "max-steps-reached",
        "max-steps-pending",
        "max-steps-unset",
        "max-epochs-reached",
        "max-epochs-pending",
        "max-epochs-none",
        "max-epochs-unset",
    ],
)
def test_is_final_train_epoch(kwargs: dict[str, Any], expected: bool) -> None:
    harness = _make_final_epoch_harness(**kwargs)

    assert BaseSystem._is_final_train_epoch(harness) is expected


def test_validation_start_creates_metrics_only_output_directory(tmp_path: Path, monkeypatch):
    system = object.__new__(_TestSystem)
    monkeypatch.setattr(_TestSystem, "global_step", 30_000, raising=False)
    system.__dict__["out_dir"] = str(tmp_path)
    system.__dict__["run_id"] = "metrics-only"
    system.__dict__["config"] = types.SimpleNamespace(
        test=types.SimpleNamespace(separate_val_dir_per_step=True),
    )
    system.__dict__["harmonizer_config"] = types.SimpleNamespace(
        inference=types.SimpleNamespace(enabled=False),
    )
    BaseSystem.on_validation_start(system)

    assert Path(system.val_dir) == tmp_path / "metrics-only" / "val" / "030000"
    assert Path(system.val_dir).is_dir()


def test_publish_existing_usdz_preserves_bytes(tmp_path):
    source = tmp_path / "000001.usdz"
    destination = tmp_path / "last.usdz"
    source.write_bytes(b"artifact")

    publish_existing_usdz(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_ino == source.stat().st_ino


def test_publish_existing_usdz_isolated_from_numbered_artifact_rewrite(tmp_path):
    source = tmp_path / "000001.usdz"
    destination = tmp_path / "last.usdz"
    source.write_bytes(b"new artifact")
    destination.write_bytes(b"old artifact")

    publish_existing_usdz(source, destination)
    retry = tmp_path / ".000001.usdz.retry.tmp"
    retry.write_bytes(b"retried artifact")
    os.replace(retry, source)

    assert source.read_bytes() == b"retried artifact"
    assert destination.read_bytes() == b"new artifact"
    assert not list(tmp_path.glob(".last.usdz.*.tmp"))


def test_publish_existing_usdz_failure_preserves_previous_artifact(tmp_path, monkeypatch):
    source = tmp_path / "000001.usdz"
    destination = tmp_path / "last.usdz"
    source.write_bytes(b"new artifact")
    destination.write_bytes(b"old artifact")
    monkeypatch.setattr("nurec.utils.io.artifact_cache.os.link", Mock(side_effect=OSError("link failed")))

    with pytest.raises(OSError, match="link failed"):
        publish_existing_usdz(source, destination)

    assert destination.read_bytes() == b"old artifact"
    assert not list(tmp_path.glob(".last.usdz.*.tmp"))


def test_publish_existing_usdz_replace_failure_cleans_temporary_file(tmp_path, monkeypatch):
    source = tmp_path / "000001.usdz"
    destination = tmp_path / "last.usdz"
    source.write_bytes(b"new artifact")
    destination.write_bytes(b"old artifact")
    monkeypatch.setattr("nurec.utils.io.artifact_cache.os.replace", Mock(side_effect=OSError("replace failed")))

    with pytest.raises(OSError, match="replace failed"):
        publish_existing_usdz(source, destination)

    assert destination.read_bytes() == b"old artifact"
    assert not list(tmp_path.glob(".last.usdz.*.tmp"))


def test_start_ppisp_report_submit_failure_shuts_down_executor(monkeypatch):
    system = object.__new__(_TestSystem)
    system.__dict__["model"] = object()
    system.__dict__["_ppisp_report_future"] = None
    system.__dict__["_ppisp_report_executor"] = None
    executor = Mock()
    executor.submit.side_effect = RuntimeError("submit failed")
    executor.shutdown.side_effect = RuntimeError("shutdown failed")
    monkeypatch.setattr(base_module, "get_ppisp_from_model", Mock(return_value=object()))
    monkeypatch.setattr(base_module, "ThreadPoolExecutor", Mock(return_value=executor))

    with pytest.raises(RuntimeError, match="submit failed"):
        system._start_ppisp_report_export_rank_zero()

    executor.shutdown.assert_called_once_with(wait=True)
    assert system._ppisp_report_executor is None
    assert system._ppisp_report_future is None


def test_start_ppisp_report_executor_creation_failure_leaves_clean_state(monkeypatch):
    system = object.__new__(_TestSystem)
    system.__dict__["model"] = object()
    system.__dict__["_ppisp_report_future"] = None
    system.__dict__["_ppisp_report_executor"] = None
    monkeypatch.setattr(base_module, "get_ppisp_from_model", Mock(return_value=object()))
    monkeypatch.setattr(
        base_module,
        "ThreadPoolExecutor",
        Mock(side_effect=RuntimeError("executor creation failed")),
    )

    with pytest.raises(RuntimeError, match="executor creation failed"):
        system._start_ppisp_report_export_rank_zero()

    assert system._ppisp_report_executor is None
    assert system._ppisp_report_future is None


def test_start_ppisp_report_without_ppisp_skips_executor(monkeypatch):
    system = object.__new__(_TestSystem)
    system.__dict__["model"] = object()
    system.__dict__["_ppisp_report_future"] = None
    system.__dict__["_ppisp_report_executor"] = None
    executor_factory = Mock()
    monkeypatch.setattr(base_module, "get_ppisp_from_model", Mock(return_value=None))
    monkeypatch.setattr(base_module, "ThreadPoolExecutor", executor_factory)

    system._start_ppisp_report_export_rank_zero()

    executor_factory.assert_not_called()


def test_train_end_publishes_final_artifact_before_reporting_executor_failure(monkeypatch):
    system = object.__new__(_TestSystem)
    system.__dict__["model"] = object()
    system.__dict__["artifact_cache"] = types.SimpleNamespace(
        artifact_config=types.SimpleNamespace(enabled=True),
    )
    system.__dict__["_ppisp_report_future"] = None
    system.__dict__["_ppisp_report_executor"] = None
    write_last = Mock()
    monkeypatch.setattr(BaseSystem, "on_train_end", Mock())
    monkeypatch.setattr(base_module, "get_ppisp_from_model", Mock(return_value=object()))
    monkeypatch.setattr(
        base_module,
        "ThreadPoolExecutor",
        Mock(side_effect=RuntimeError("executor creation failed")),
    )
    monkeypatch.setattr(_TestSystem, "_write_last_artifact_rank_zero", write_last)

    with pytest.raises(RuntimeError, match="executor creation failed"):
        BaseSystemSO.on_train_end(system)

    write_last.assert_called_once_with()


def test_final_checkpoint_publication_does_not_start_ppisp_worker(tmp_path, monkeypatch):
    system = object.__new__(_TestSystem)
    system.__dict__["out_dir"] = str(tmp_path)
    system.__dict__["run_id"] = "run"
    system.__dict__["artifact_cache"] = object()
    system.__dict__["checkpoint_every_n_train_steps"] = 1
    system.__dict__["last_checkpoint_step"] = 7
    system.__dict__["_fabric"] = None
    system.__dict__["_jit_is_scripting"] = False
    system.__dict__["_trainer"] = types.SimpleNamespace(global_step=7, max_steps=7)
    populate = Mock()
    start_report = Mock(side_effect=AssertionError("worker started during checkpoint hook"))
    monkeypatch.setattr(_TestSystem, "populate_artifact_cache", populate)
    monkeypatch.setattr(_TestSystem, "_start_ppisp_report_export_rank_zero", start_report)
    monkeypatch.setattr(
        base_module,
        "write_artifact_cache",
        lambda _cache, path: path.write_bytes(b"artifact"),
    )

    system._populate_and_save_artifacts_rank_zero({})

    populate.assert_called_once_with({})
    start_report.assert_not_called()
    assert (tmp_path / "run" / "artifacts" / "000007.usdz").read_bytes() == b"artifact"


def test_failed_artifact_population_does_not_reuse_previous_numbered_artifact(tmp_path, monkeypatch):
    system = object.__new__(_TestSystem)
    system.__dict__["out_dir"] = str(tmp_path)
    system.__dict__["run_id"] = "run"
    system.__dict__["artifact_cache"] = object()
    previous_artifact = tmp_path / "run" / "artifacts" / "000006.usdz"
    previous_artifact.parent.mkdir(parents=True)
    previous_artifact.write_bytes(b"stale")
    system.__dict__["_last_step_artifact_path"] = previous_artifact
    monkeypatch.setattr(
        _TestSystem,
        "populate_artifact_cache",
        Mock(side_effect=OSError("population failed")),
    )

    with pytest.raises(OSError, match="population failed"):
        system._populate_and_save_artifacts_rank_zero({})

    monkeypatch.setattr(
        base_module,
        "write_artifact_cache",
        lambda _cache, path: path.write_bytes(b"fresh"),
    )
    system._write_last_artifact_rank_zero()

    assert system._last_step_artifact_path is None
    assert (previous_artifact.parent / "last.usdz").read_bytes() == b"fresh"


def test_failed_numbered_artifact_is_not_reused_as_last(tmp_path, monkeypatch):
    system = object.__new__(_TestSystem)
    system.__dict__["out_dir"] = str(tmp_path)
    system.__dict__["run_id"] = "run"
    system.__dict__["artifact_cache"] = object()
    system.__dict__["checkpoint_every_n_train_steps"] = 1
    system.__dict__["last_checkpoint_step"] = 7
    system.__dict__["_last_step_artifact_path"] = None
    system.__dict__["_ppisp_report_future"] = None
    system.__dict__["_ppisp_report_executor"] = None
    system.__dict__["_fabric"] = None
    system.__dict__["_jit_is_scripting"] = False
    system.__dict__["_trainer"] = types.SimpleNamespace(global_step=7, max_steps=7)
    monkeypatch.setattr(_TestSystem, "populate_artifact_cache", Mock())
    monkeypatch.setattr(
        base_module,
        "write_artifact_cache",
        Mock(side_effect=OSError("write failed")),
    )

    with pytest.raises(OSError, match="write failed"):
        system._populate_and_save_artifacts_rank_zero({})

    assert system._last_step_artifact_path is None


def test_train_end_joins_ppisp_worker_when_artifact_write_fails(monkeypatch):
    system = object.__new__(_TestSystem)
    system.__dict__["artifact_cache"] = types.SimpleNamespace(
        artifact_config=types.SimpleNamespace(enabled=True),
    )
    report_future = Mock()
    report_future.result.side_effect = BaseException("report failed")
    report_executor = Mock()
    report_executor.shutdown.side_effect = RuntimeError("shutdown failed")
    system.__dict__["_ppisp_report_future"] = report_future
    system.__dict__["_ppisp_report_executor"] = report_executor
    monkeypatch.setattr(BaseSystem, "on_train_end", Mock())
    monkeypatch.setattr(
        _TestSystem,
        "_write_last_artifact_rank_zero",
        Mock(side_effect=OSError("artifact failed")),
    )

    with pytest.raises(OSError, match="artifact failed"):
        BaseSystemSO.on_train_end(system)

    report_executor.shutdown.assert_called_once_with(wait=True)
    assert system._ppisp_report_executor is None
    assert system._ppisp_report_future is None


def _make_ground_mesh_config(**overrides):
    values = {
        "step_frame": 1,
        "lidar_ids": None,
        "camera_ids": None,
        "min_ray_length": 0.1,
        "ground_compat_max_distance": 0.5,
        "ground_compat_max_angle_deg": 60.0,
        "z_offset": 0.0,
        "num_plane_hypotheses": 100,
        "plane_max_distance": 0.3,
        "plane_max_angle_deg": 30,
        "plane_min_eval_points": 1000,
        "plane_max_eval_points": 10000,
        "plane_min_inliers": 1000,
        "enable_plane_extension": False,
        "voxel_size": 0.1,
        "min_points_per_voxel": 1,
        "smoothing_passes": 10,
        "colored": False,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _make_ground_mesh_settings(**overrides) -> GroundMeshReconstructionSettings:
    return GroundMeshReconstructionSettings.from_mapping(_make_ground_mesh_config(**overrides))


def test_collect_metric_with_lidar_metadata_uses_per_lidar_bucket():
    system = object.__new__(_TestSystem)
    system.__dict__["datamodule"] = types.SimpleNamespace(get_datasource=lambda: _FakeDataSource())
    system.__dict__["logger_enabled"] = False
    system.__dict__["_validation_metrics_collector"] = MetricsCollector(
        train_config_name="test_config",
        mode="val",
        run_id="test_run",
    )

    frame_meta = FrameMeta(unique_sensor_idx=0, unique_frame_idx=7, training_frame_idx=-1)

    _TestSystem.collect_metric(
        system,
        "test/raydrop_accuracy",
        0.25,
        is_lidar=True,
        frame_meta=frame_meta,
        sequence_id="test_sequence",
    )

    sequence_samples = system._validation_metrics_collector._samples.per_sequence["test_sequence"]
    assert sequence_samples.per_camera == {}
    assert set(sequence_samples.per_lidar) == {"lidar_gt_top_p128"}

    metric_samples = sequence_samples.per_lidar["lidar_gt_top_p128"]["test/raydrop_accuracy"]
    assert len(metric_samples) == 1
    assert metric_samples[0].value == 0.25
    assert metric_samples[0].unique_frame_idx == 7


def test_get_cached_ground_mesh_artifact_reuses_matching_mesh_without_colors():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

    model = object.__new__(GaussiansComposite)
    model.gaussians_nodes = {
        "road": types.SimpleNamespace(
            ground_mesh_artifact=Mesh(vertices=vertices, faces=faces, colors=colors),
            ground_mesh_artifact_settings=_make_ground_mesh_settings(),
        )
    }
    system = object.__new__(_TestSystem)
    system.__dict__["model"] = model

    mesh = _TestSystem._get_cached_ground_mesh_artifact(system, _make_ground_mesh_config(colored=False))

    assert mesh is not None
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)
    assert mesh.colors is None


def test_get_cached_ground_mesh_artifact_reuses_matching_mesh_with_colors():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

    model = object.__new__(GaussiansComposite)
    model.gaussians_nodes = {
        "road": types.SimpleNamespace(
            ground_mesh_artifact=Mesh(vertices=vertices, faces=faces, colors=colors),
            ground_mesh_artifact_settings=_make_ground_mesh_settings(),
        )
    }
    system = object.__new__(_TestSystem)
    system.__dict__["model"] = model

    mesh = _TestSystem._get_cached_ground_mesh_artifact(system, _make_ground_mesh_config(colored=True))

    assert mesh is not None
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)
    np.testing.assert_array_equal(mesh.colors, colors)


def test_get_cached_ground_mesh_artifact_rejects_missing_colors_when_required():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    model = object.__new__(GaussiansComposite)
    model.gaussians_nodes = {
        "road": types.SimpleNamespace(
            ground_mesh_artifact=Mesh(vertices=vertices, faces=faces, colors=None),
            ground_mesh_artifact_settings=_make_ground_mesh_settings(),
        )
    }
    system = object.__new__(_TestSystem)
    system.__dict__["model"] = model

    assert _TestSystem._get_cached_ground_mesh_artifact(system, _make_ground_mesh_config(colored=True)) is None


def test_get_cached_ground_mesh_artifact_rejects_mismatched_settings():
    model = object.__new__(GaussiansComposite)
    model.gaussians_nodes = {
        "road": types.SimpleNamespace(
            ground_mesh_artifact=Mesh(
                vertices=np.zeros((3, 3), dtype=np.float32), faces=np.zeros((1, 3), dtype=np.int32)
            ),
            ground_mesh_artifact_settings=_make_ground_mesh_settings(voxel_size=0.2),
        )
    }
    system = object.__new__(_TestSystem)
    system.__dict__["model"] = model

    assert _TestSystem._get_cached_ground_mesh_artifact(system, _make_ground_mesh_config(voxel_size=0.1)) is None


def test_get_cached_ground_mesh_artifact_rejects_mismatched_z_offset():
    model = object.__new__(GaussiansComposite)
    model.gaussians_nodes = {
        "road": types.SimpleNamespace(
            ground_mesh_artifact=Mesh(
                vertices=np.zeros((3, 3), dtype=np.float32), faces=np.zeros((1, 3), dtype=np.int32)
            ),
            ground_mesh_artifact_settings=_make_ground_mesh_settings(z_offset=-0.3),
        )
    }
    system = object.__new__(_TestSystem)
    system.__dict__["model"] = model

    assert _TestSystem._get_cached_ground_mesh_artifact(system, _make_ground_mesh_config(z_offset=0.0)) is None


def test_get_cached_ground_mesh_artifact_picks_largest_when_multiple_match():
    small_vertices = np.zeros((3, 3), dtype=np.float32)
    small_faces = np.zeros((1, 3), dtype=np.int32)
    large_vertices = np.zeros((10, 3), dtype=np.float32)
    large_faces = np.zeros((4, 3), dtype=np.int32)
    settings = _make_ground_mesh_settings()

    model = object.__new__(GaussiansComposite)
    model.gaussians_nodes = {
        "first": types.SimpleNamespace(
            ground_mesh_artifact=Mesh(vertices=small_vertices, faces=small_faces),
            ground_mesh_artifact_settings=settings,
        ),
        "second": types.SimpleNamespace(
            ground_mesh_artifact=Mesh(vertices=large_vertices, faces=large_faces),
            ground_mesh_artifact_settings=settings,
        ),
    }
    system = object.__new__(_TestSystem)
    system.__dict__["model"] = model

    mesh = _TestSystem._get_cached_ground_mesh_artifact(system, _make_ground_mesh_config(colored=False))

    assert mesh is not None
    assert len(mesh.vertices) == 10
    assert len(mesh.faces) == 4


def test_get_cached_ground_mesh_artifact_prefers_colored_when_shapes_tie():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    settings = _make_ground_mesh_settings()

    model = object.__new__(GaussiansComposite)
    model.gaussians_nodes = {
        "uncolored": types.SimpleNamespace(
            ground_mesh_artifact=Mesh(vertices=vertices, faces=faces, colors=None),
            ground_mesh_artifact_settings=settings,
        ),
        "colored": types.SimpleNamespace(
            ground_mesh_artifact=Mesh(vertices=vertices, faces=faces, colors=colors),
            ground_mesh_artifact_settings=settings,
        ),
    }
    system = object.__new__(_TestSystem)
    system.__dict__["model"] = model

    mesh = _TestSystem._get_cached_ground_mesh_artifact(system, _make_ground_mesh_config(colored=True))

    assert mesh is not None
    np.testing.assert_array_equal(mesh.colors, colors)


class _RecordingOptimizer:
    """Minimal optimizer stand-in recording load_state_dict calls.

    Only needs the surface DDPStrategySO.load_optimizer_state_dict touches: an empty ``state``
    (for the device-transfer no-op) and a recordable ``load_state_dict``.
    """

    def __init__(self):
        self.state: dict = {}
        self.loaded_state = None

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict


def test_load_optimizer_state_dict_mismatch_warns_once_and_skips(caplog, monkeypatch):
    """A checkpoint/model optimizer-count mismatch must warn once and restore no optimizer state.

    Guards against zip() silently truncating and positionally misaligning states after the
    Gaussian optimizer split (density/scale became separate optimizers).
    """
    strategy = object.__new__(DDPStrategySO)
    opt_a, opt_b = _RecordingOptimizer(), _RecordingOptimizer()
    strategy._optimizers = [opt_a, opt_b]  # 2 optimizers

    import lightning_fabric.utilities.optimizer as lf_opt

    moved: list = []
    monkeypatch.setattr(lf_opt, "_optimizer_to_device", lambda opt, device: moved.append(opt))

    checkpoint = {"optimizer_states": [{"step": 1}]}  # only 1 state -> mismatch

    with caplog.at_level(logging.WARNING, logger="nurec.systems.base"):
        DDPStrategySO.load_optimizer_state_dict(strategy, checkpoint)

    warnings = [r for r in caplog.records if "Optimizer count mismatch" in r.getMessage()]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "2 optimizers" in message
    assert "1 optimizer states" in message

    # No optimizer state restored (no positional partial load) and no device transfer performed.
    assert opt_a.loaded_state is None
    assert opt_b.loaded_state is None
    assert moved == []


def test_load_optimizer_state_dict_matching_loads_all_and_transfers(monkeypatch):
    """Matching optimizer counts restore every state and run the per-optimizer device transfer."""
    strategy = object.__new__(DDPStrategySO)
    opt_a, opt_b = _RecordingOptimizer(), _RecordingOptimizer()
    strategy._optimizers = [opt_a, opt_b]
    strategy._lightning_module = types.SimpleNamespace()
    # Passthrough split (instance attribute shadows the sharding method) so the state each
    # optimizer receives is exactly the checkpoint entry.
    strategy.split_optimizer_states = lambda opt, state: state

    monkeypatch.setattr(DDPStrategySO, "root_device", property(lambda self: torch.device("cpu")))

    import lightning_fabric.utilities.optimizer as lf_opt

    moved: list = []
    monkeypatch.setattr(lf_opt, "_optimizer_to_device", lambda opt, device: moved.append((opt, device)))

    states = [{"step": 1}, {"step": 2}]
    checkpoint = {"optimizer_states": states}

    DDPStrategySO.load_optimizer_state_dict(strategy, checkpoint)

    # Every state loaded into the matching optimizer.
    assert opt_a.loaded_state == states[0]
    assert opt_b.loaded_state == states[1]
    # Device transfer invoked once per optimizer, with the strategy's root device.
    assert [opt for opt, _ in moved] == [opt_a, opt_b]
    assert all(device == torch.device("cpu") for _, device in moved)


class _ArtifactCalib(BaseCalib):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.offset = 1.0

    @property
    def params(self):
        return []

    def forward(self, data_batch, skip_calib=False, global_step_for_prober=None):
        raise NotImplementedError

    def apply_to_rig_trajectories(self, rig_trajectories):
        copied = types.SimpleNamespace(
            rig_trajectories=[
                types.SimpleNamespace(
                    T_rig_world_timestamps_us=rig_trajectories.rig_trajectories[0]
                    .T_rig_world_timestamps_us.detach()
                    .clone(),
                )
            ],
            baked_offset=self.offset,
        )
        return copied


class _RigDataSource:
    def __init__(self) -> None:
        self.source_rig = types.SimpleNamespace(
            rig_trajectories=[types.SimpleNamespace(T_rig_world_timestamps_us=torch.tensor([1_000_000, 2_000_000]))],
            baked_offset=0.0,
        )

        self.lidar_sensors = {"lidar_fu": object()}

    def get_rig_trajectories(self):
        return self.source_rig


def test_populate_artifact_cache_bakes_and_refreshes_rig_json_and_usd(monkeypatch):
    datasource = _RigDataSource()
    calib = _ArtifactCalib()
    system = object.__new__(_TestSystem)
    system.__dict__["datamodule"] = types.SimpleNamespace(get_datasource=lambda: datasource)
    system.__dict__["model"] = types.SimpleNamespace(calib=calib)
    system.__dict__["dataset_path"] = None
    system.__dict__["cached_config"] = types.SimpleNamespace()

    artifact_config = types.SimpleNamespace(
        checkpoint=types.SimpleNamespace(enabled=False),
        parsed_config=types.SimpleNamespace(enabled=False),
        nrend=types.SimpleNamespace(enabled=False),
        rig_trajectories=types.SimpleNamespace(enabled=True, add_default_cameras=True),
        firing_patterns=types.SimpleNamespace(enabled=False),
    )
    artifact_cache = ArtifactCache(artifact_config=artifact_config)
    original_summary = NamedSerialized(filename="datasource_summary.json", serialized="original")
    artifact_cache.datasource_summary = original_summary
    system.__dict__["artifact_cache"] = artifact_cache

    serialized_inputs = []

    def fake_serialize(rig_trajectories, *, usd_timestamp_offset, add_default_cameras):
        serialized_inputs.append(rig_trajectories)
        assert usd_timestamp_offset == 1_000_000
        assert add_default_cameras is True
        value = str(rig_trajectories.baked_offset)
        return [
            NamedSerialized(filename="rig_trajectories.json", serialized=f"json:{value}"),
            NamedSerialized(filename="rig_trajectories.usda", serialized=f"usd:{value}"),
        ]

    monkeypatch.setattr("nurec.systems.base.serialize_rig_trajectories", fake_serialize)

    _TestSystem.populate_artifact_cache(system, {})
    assert [member.serialized for member in artifact_cache.rig_trajectories] == ["json:1.0", "usd:1.0"]
    assert serialized_inputs[-1] is not datasource.source_rig
    assert datasource.source_rig.baked_offset == 0.0
    assert artifact_cache.datasource_summary is original_summary

    calib.offset = 2.0
    _TestSystem.populate_artifact_cache(system, {})
    assert [member.serialized for member in artifact_cache.rig_trajectories] == ["json:2.0", "usd:2.0"]
    assert len(serialized_inputs) == 2
    assert datasource.source_rig.baked_offset == 0.0
    assert artifact_cache.datasource_summary is original_summary


def test_populate_artifact_cache_serializes_firing_patterns_independently(monkeypatch):
    """A model-free lidar pattern does not depend on rig-trajectory artifact export."""
    datasource = _RigDataSource()
    system = object.__new__(_TestSystem)
    system.__dict__["datamodule"] = types.SimpleNamespace(get_datasource=lambda: datasource)
    system.__dict__["model"] = types.SimpleNamespace(calib=_ArtifactCalib())
    system.__dict__["dataset_path"] = None
    system.__dict__["cached_config"] = types.SimpleNamespace()

    artifact_config = types.SimpleNamespace(
        checkpoint=types.SimpleNamespace(enabled=False),
        parsed_config=types.SimpleNamespace(enabled=False),
        nrend=types.SimpleNamespace(enabled=False),
        rig_trajectories=types.SimpleNamespace(enabled=False, add_default_cameras=True),
        firing_patterns=types.SimpleNamespace(enabled=True),
    )
    artifact_cache = ArtifactCache(artifact_config=artifact_config)
    # Pre-set so the summary block, which does not understand this stub, is skipped.
    artifact_cache.datasource_summary = NamedSerialized(filename="datasource_summary.json", serialized="{}")
    system.__dict__["artifact_cache"] = artifact_cache

    seen = []

    def fake_serialize_patterns(rig_trajectories, lidar_sensors):
        seen.append((rig_trajectories, lidar_sensors))
        return [NamedSerialized(filename="firing_patterns/lidar_fu.npz", serialized=b"npz")]

    monkeypatch.setattr("nurec.utils.io.firing_patterns.serialize_firing_patterns", fake_serialize_patterns)

    _TestSystem.populate_artifact_cache(system, {})

    assert [member.filename for member in artifact_cache.firing_patterns] == ["firing_patterns/lidar_fu.npz"]
    # Measured off the CALIBRATED trajectories, the same ones the artifact records.
    assert seen[-1][0] is not datasource.source_rig
    assert seen[-1][1] is datasource.lidar_sensors

    # Cached after the first export, so a second call does not re-measure.
    _TestSystem.populate_artifact_cache(system, {})
    assert len(seen) == 1


def test_populate_artifact_cache_skips_firing_patterns_when_disabled(monkeypatch):
    datasource = _RigDataSource()
    system = object.__new__(_TestSystem)
    system.__dict__["datamodule"] = types.SimpleNamespace(get_datasource=lambda: datasource)
    system.__dict__["model"] = types.SimpleNamespace(calib=_ArtifactCalib())
    system.__dict__["dataset_path"] = None
    system.__dict__["cached_config"] = types.SimpleNamespace()

    artifact_config = types.SimpleNamespace(
        checkpoint=types.SimpleNamespace(enabled=False),
        parsed_config=types.SimpleNamespace(enabled=False),
        nrend=types.SimpleNamespace(enabled=False),
        rig_trajectories=types.SimpleNamespace(enabled=True, add_default_cameras=True),
        firing_patterns=types.SimpleNamespace(enabled=False),
    )
    artifact_cache = ArtifactCache(artifact_config=artifact_config)
    artifact_cache.datasource_summary = NamedSerialized(filename="datasource_summary.json", serialized="{}")
    system.__dict__["artifact_cache"] = artifact_cache

    monkeypatch.setattr(
        "nurec.systems.base.serialize_rig_trajectories",
        lambda rig_trajectories, **_: [NamedSerialized(filename="rig_trajectories.json", serialized="{}")],
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("firing patterns must not be measured when the export is disabled")

    monkeypatch.setattr("nurec.utils.io.firing_patterns.serialize_firing_patterns", fail)

    _TestSystem.populate_artifact_cache(system, {})
    assert artifact_cache.firing_patterns is None
