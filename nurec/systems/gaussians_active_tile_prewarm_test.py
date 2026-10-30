# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused lifecycle tests for eager camera-to-LiDAR active-tile prewarm."""

import json
import logging
import subprocess
import sys
import time

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from pytorch_lightning import LightningModule

from nurec.datasets.ncore import NCORETrainDataset
from nurec.systems.active_tile_mapping_prewarmer import ActiveTileMappingPrewarmer
from nurec.systems.base import BaseSystemSO, DDPStrategySO
from nurec.systems.gaussians import GaussiansSystem
from nurec.utils.batch import CameraFrameLabels, DataBatch, FrameMeta, LidarFrameLabels, RectSubsampled


class _PrewarmModel(torch.nn.Module):
    def __init__(self, *, fail: bool = False, enabled: bool = True, supported: bool = True) -> None:
        super().__init__()
        self.fail = fail
        self.enabled = enabled
        self.supported = supported
        # DDP requires at least one trainable parameter. The focused buffer-sync
        # regression below does not optimize it.
        self.ddp_anchor = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("calibration_version", torch.tensor(0, dtype=torch.int64))
        self.cache: dict[int, torch.Tensor] = {}
        self.build_calls: list[int] = []
        self.calib_global_steps: list[int | None] = []
        self.failure_cuda_work: torch.Tensor | None = None
        self.on_train_from_scratch_start = MagicMock()

    def active_tile_mapping_prewarm_enabled(self) -> bool:
        return self.enabled

    def calib(
        self,
        data_batch: DataBatch,
        *,
        skip_calib: bool,
        global_step_for_prober: int | None = None,
    ):  # type: ignore[no-untyped-def]
        assert skip_calib is False
        self.calib_global_steps.append(global_step_for_prober)
        assert data_batch.camera is not None and data_batch.lidar is not None
        assert data_batch.camera.labels.rgb is None
        assert data_batch.lidar.labels.distance is None
        unique_sensor_frame_idx = data_batch.camera.meta[0].unique_sensor_frame_idx_tensor
        assert unique_sensor_frame_idx is not None
        device = unique_sensor_frame_idx.device
        if self.fail:
            assert device.type == "cuda"
            self.failure_cuda_work = torch.arange(4096, device=device, dtype=torch.float32).sin()
            raise ValueError("injected mapping failure after CUDA work")
        rays = torch.zeros((1, 2, 3, 6), device=device)
        return SimpleNamespace(camera=SimpleNamespace(rays=rays), lidar=SimpleNamespace(rays=rays.clone()))

    def prewarm_camera_to_lidar_tile_mapping(
        self,
        sensor_idx: int,
        rendering_data_cam,
        rendering_data_lidar,
    ) -> None:  # type: ignore[no-untyped-def]
        del rendering_data_lidar
        if not self.supported:
            return
        device = rendering_data_cam.rays.device
        assert device.type == "cuda"
        if sensor_idx not in self.cache:
            self.build_calls.append(sensor_idx)
            self.cache[sensor_idx] = torch.rand(8, device=device) > 0.5


@dataclass(frozen=True)
class _PrewarmEntry:
    camera_sensor_idx: int
    data_batch: DataBatch


@dataclass(frozen=True)
class _PrewarmPlan:
    entries: tuple[_PrewarmEntry, ...] = ()
    skip_reason: str | None = None


def _entry(sensor_idx: int) -> _PrewarmEntry:
    camera_subsample = RectSubsampled(
        i=0,
        j=0,
        height=3,
        width=4,
        subsample_factor=2.0,
        original_width=8,
        original_height=6,
    )
    return _PrewarmEntry(
        camera_sensor_idx=sensor_idx,
        data_batch=DataBatch(
            sequence_id=["sequence"],
            camera=DataBatch.Camera(
                meta=[
                    FrameMeta(
                        unique_sensor_idx=sensor_idx,
                        unique_frame_idx=2,
                        training_frame_idx=1,
                        subsample=camera_subsample,
                    )
                ],
                labels=CameraFrameLabels(),
            ),
            lidar=DataBatch.Lidar(
                meta=[FrameMeta(unique_sensor_idx=9, unique_frame_idx=4, training_frame_idx=3)],
                labels=LidarFrameLabels(),
            ),
        ),
    )


def _system(
    *,
    world_size: int,
    entries: tuple[_PrewarmEntry, ...] = (),
    fail: bool = False,
    enabled: bool = True,
    supported: bool = True,
    mock_distributed_state: bool = True,
) -> tuple[GaussiansSystem, _PrewarmModel]:
    system = GaussiansSystem.__new__(GaussiansSystem)
    LightningModule.__init__(system)
    system.resume_weights_only = False
    cast(Any, system)._trainer = SimpleNamespace(
        global_rank=0,
        global_step=0,
        should_stop=False,
        strategy=SimpleNamespace(reduce_boolean_decision=lambda decision, **_kwargs: decision),
    )
    model = _PrewarmModel(fail=fail, enabled=enabled, supported=supported)
    cast(Any, system).model = model
    prewarmer = ActiveTileMappingPrewarmer()
    system._active_tile_mapping_prewarmer = prewarmer
    if mock_distributed_state:
        run = prewarmer._run

        def run_with_distributed_state(**kwargs: Any) -> None:
            with (
                patch.object(torch.distributed, "is_initialized", return_value=True),
                patch.object(torch.distributed, "get_world_size", return_value=world_size),
            ):
                run(**kwargs)

        prewarmer._run = run_with_distributed_state  # type: ignore[method-assign]

    dataset = MagicMock(spec=NCORETrainDataset)
    dataset.get_active_tile_mapping_prewarm_plan.return_value = _PrewarmPlan(entries=entries)
    cast(Any, system).datamodule = SimpleNamespace(train_dataset=dataset)
    return system, model


def _run_prewarm(system: GaussiansSystem) -> None:
    system._active_tile_mapping_prewarmer.run_once(
        model=system.model,
        train_dataset=system.datamodule.train_dataset,
        device=system.device,
        trainer=system.trainer,
    )


def _move_to_current_cuda(system: GaussiansSystem) -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("Active-tile mapping prewarm requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    system.to(device)
    assert system.device == device
    return device


def _nccl_asymmetric_failure_worker(rank: int, init_file: str) -> dict[str, Any]:
    """Exercise production prewarm coordination on one fresh NCCL worker rank."""
    initialized = False
    reached_sentinel_collective = False
    trainer = None
    model = None
    backend = "uninitialized"
    try:
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=30),
        )
        initialized = True
        backend = str(torch.distributed.get_backend())
        assert backend == "nccl"

        system, model = _system(
            world_size=2,
            entries=(_entry(3),),
            fail=rank == 0,
            mock_distributed_state=False,
        )
        system.to(device)
        cluster_environment = cast(
            Any,
            SimpleNamespace(
                global_rank=lambda: rank,
                local_rank=lambda: rank,
                node_rank=lambda: 0,
                world_size=lambda: 2,
            ),
        )
        strategy = DDPStrategySO(
            parallel_devices=[torch.device("cuda", device_idx) for device_idx in range(2)],
            cluster_environment=cluster_environment,
        )
        assert strategy.root_device == device
        trainer = SimpleNamespace(strategy=strategy, global_rank=rank, global_step=17, should_stop=False)
        cast(Any, system)._trainer = trainer

        _run_prewarm(system)
        # This models the first ordinary training collective. A coordinated prewarm
        # failure must prevent either rank from even invoking it.
        reached_sentinel_collective = True
        torch.distributed.all_reduce(torch.ones(1, dtype=torch.int32, device=device))
        result = {
            "backend": backend,
            "calib_global_steps": model.calib_global_steps,
            "cuda_work_started": bool(model.failure_cuda_work is not None),
            "error_message": "",
            "error_type": "unexpected_success",
            "rank": rank,
            "reached_sentinel_collective": reached_sentinel_collective,
            "should_stop": trainer.should_stop,
            "strategy": type(strategy).__name__,
        }
    except Exception as error:
        result = {
            "backend": backend,
            "calib_global_steps": model.calib_global_steps if model is not None else [],
            "cuda_work_started": bool(model is not None and model.failure_cuda_work is not None),
            "error_message": str(error),
            "error_type": type(error).__name__,
            "rank": rank,
            "reached_sentinel_collective": reached_sentinel_collective,
            "should_stop": bool(trainer is not None and trainer.should_stop),
            "strategy": type(trainer.strategy).__name__ if trainer is not None else "uninitialized",
        }
    finally:
        if initialized:
            torch.distributed.destroy_process_group()
    return result


def _nccl_ddp_buffer_sync_worker(rank: int, init_file: str) -> dict[str, Any]:
    """Prove DDP's first pre-forward buffer sync precedes system prewarm."""
    initialized = False
    original_training_step = BaseSystemSO.training_step
    try:
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=30),
        )
        initialized = True

        system, model = _system(world_size=2)
        system.to(device)
        cluster_environment = cast(
            Any,
            SimpleNamespace(
                global_rank=lambda: rank,
                local_rank=lambda: rank,
                node_rank=lambda: 0,
                world_size=lambda: 2,
            ),
        )
        strategy = DDPStrategySO(
            parallel_devices=[torch.device("cuda", device_idx) for device_idx in range(2)],
            cluster_environment=cluster_environment,
            find_unused_parameters=True,
        )
        strategy.connect(system)
        strategy.configure_ddp()
        wrapped = isinstance(strategy.model, torch.nn.parallel.DistributedDataParallel)
        assert wrapped
        # Lightning constructs DDP on a side stream. Finish that work before
        # deliberately making the registered buffer disagree across ranks.
        torch.cuda.synchronize(device)
        trainer = SimpleNamespace(strategy=strategy, global_rank=rank, global_step=41, should_stop=False)
        cast(Any, system)._trainer = trainer

        # Mutate after DDP construction so constructor-time state agreement
        # cannot satisfy this regression. The first wrapper forward must copy
        # rank 0's value before GaussiansSystem.training_step runs prewarm.
        calibration_version = cast(torch.Tensor, model.calibration_version)
        calibration_version.fill_(73 if rank == 0 else -1)
        observed_prewarm_values: list[int] = []
        base_step_count = 0

        def observe_prewarm(**_kwargs: Any) -> None:
            observed_prewarm_values.append(int(calibration_version.item()))

        def stub_base_training_step(_system: BaseSystemSO, _batch: Any, _batch_idx: int) -> torch.Tensor:
            nonlocal base_step_count
            base_step_count += 1
            return torch.zeros((), device=device)

        system._active_tile_mapping_prewarmer._run = observe_prewarm  # type: ignore[method-assign]
        BaseSystemSO.training_step = stub_base_training_step  # type: ignore[assignment]
        strategy.training_step(None, 0)
        # A direct second call proves the one-shot guard independently of a
        # second DDP iteration/backward lifecycle.
        system.training_step(None, 1)  # type: ignore[arg-type]
        return {
            "base_step_count": base_step_count,
            "buffer_value": int(calibration_version.item()),
            "error_message": "",
            "error_type": "",
            "observed_prewarm_values": observed_prewarm_values,
            "rank": rank,
            "wrapped": wrapped,
        }
    except Exception as error:
        return {
            "error_message": str(error),
            "error_type": type(error).__name__,
            "rank": rank,
        }
    finally:
        BaseSystemSO.training_step = original_training_step  # type: ignore[method-assign]
        if initialized:
            torch.distributed.destroy_process_group()


def test_overlap_disabled_is_a_noop() -> None:
    system, model = _system(world_size=2, entries=(_entry(1),), enabled=False)

    with patch("nurec.systems.active_tile_mapping_prewarmer.distributed_context") as context:
        _run_prewarm(system)

    assert model.build_calls == []
    context.assert_not_called()


def test_plan_skip_is_nonfatal(caplog: pytest.LogCaptureFixture) -> None:
    system, model = _system(world_size=2, entries=(_entry(1),))
    _move_to_current_cuda(system)
    cast(Any, system.datamodule.train_dataset).get_active_tile_mapping_prewarm_plan = lambda: _PrewarmPlan(
        skip_reason="requires_modeled_lidar_rays"
    )

    with caplog.at_level(logging.INFO, logger="nurec.systems.active_tile_mapping_prewarmer"):
        _run_prewarm(system)

    assert model.build_calls == []
    assert caplog.messages == ["Skipping active-tile mapping prewarm: requires_modeled_lidar_rays"]


def test_unsupported_renderer_is_nonfatal_and_keeps_lazy_fallback() -> None:
    system, model = _system(world_size=2, entries=(_entry(1), _entry(4)), supported=False)
    _move_to_current_cuda(system)

    _run_prewarm(system)

    assert model.cache == {}


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="NCCL regression requires two CUDA devices")
def test_real_nccl_asymmetric_failure_aborts_both_ranks_before_training_collectives(tmp_path: Path) -> None:
    init_file = tmp_path / "nccl_store"
    # Fresh interpreters avoid inheriting the parent process's CUDA context.
    workers = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "nurec.systems.gaussians_active_tile_prewarm_test",
                "--nccl-worker",
                str(rank),
                str(init_file),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for rank in range(2)
    ]

    outputs: list[str] = []
    deadline = time.monotonic() + 90
    try:
        for worker in workers:
            output, _ = worker.communicate(timeout=max(0.0, deadline - time.monotonic()))
            outputs.append(output)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()

    results: dict[int, dict[str, Any]] = {}
    for rank, (worker, output) in enumerate(zip(workers, outputs, strict=True)):
        assert worker.returncode == 0, f"NCCL worker rank {rank} failed:\n{output}"
        decoded = json.loads(output.splitlines()[-1])
        assert isinstance(decoded, dict)
        results[int(decoded["rank"])] = decoded

    assert results[0]["backend"] == results[1]["backend"] == "nccl"
    assert results[0]["calib_global_steps"] == results[1]["calib_global_steps"] == [None]
    assert results[0]["strategy"] == results[1]["strategy"] == "DDPStrategySO"
    assert results[0]["cuda_work_started"] is True
    assert results[1]["cuda_work_started"] is False
    assert results[0]["error_type"] == "ValueError"
    assert "injected mapping failure after CUDA work" in results[0]["error_message"]
    assert results[1]["error_type"] == "RuntimeError"
    assert "Aborting training due to exception in another rank" in results[1]["error_message"]
    assert results[0]["should_stop"] is results[1]["should_stop"] is True
    assert results[0]["reached_sentinel_collective"] is results[1]["reached_sentinel_collective"] is False


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="NCCL regression requires two CUDA devices")
def test_real_ddp_first_forward_syncs_buffers_before_one_shot_prewarm(tmp_path: Path) -> None:
    init_file = tmp_path / "ddp_buffer_sync_store"
    workers = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "nurec.systems.gaussians_active_tile_prewarm_test",
                "--ddp-buffer-sync-worker",
                str(rank),
                str(init_file),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for rank in range(2)
    ]

    outputs: list[str] = []
    deadline = time.monotonic() + 90
    try:
        for worker in workers:
            output, _ = worker.communicate(timeout=max(0.0, deadline - time.monotonic()))
            outputs.append(output)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()

    for rank, (worker, output) in enumerate(zip(workers, outputs, strict=True)):
        assert worker.returncode == 0, f"DDP buffer worker rank {rank} failed:\n{output}"
        result = json.loads(output.splitlines()[-1])
        assert result["error_type"] == "", result["error_message"]
        assert result["rank"] == rank
        assert result["buffer_value"] == 73
        assert result["observed_prewarm_values"] == [73]
        assert result["base_step_count"] == 2
        assert result["wrapped"] is True


def test_fixed_rng_restores_cpu_and_current_cuda_default_generators() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    system, _ = _system(world_size=2)
    device = torch.device("cuda", torch.cuda.current_device())
    original_cpu_state = torch.random.get_rng_state().clone()
    original_cuda_state = torch.cuda.get_rng_state(device).clone()

    try:
        cpu_state = torch.Generator(device="cpu").manual_seed(1234).get_state()
        cuda_state = torch.Generator(device=device).manual_seed(5678).get_state()
        torch.random.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device=device)
        caller_cpu_state = torch.random.get_rng_state().clone()
        caller_cuda_state = torch.cuda.get_rng_state(device).clone()

        with system._active_tile_mapping_prewarmer._isolated_rng(device):
            first_cpu = torch.rand(16)
            first_cuda = torch.rand(16, device=device)
        assert torch.equal(torch.random.get_rng_state(), caller_cpu_state)
        assert torch.equal(torch.cuda.get_rng_state(device), caller_cuda_state)

        with system._active_tile_mapping_prewarmer._isolated_rng(device):
            second_cpu = torch.rand(16)
            second_cuda = torch.rand(16, device=device)
        assert torch.equal(first_cpu, second_cpu)
        assert torch.equal(first_cuda, second_cuda)
        assert torch.equal(torch.random.get_rng_state(), caller_cpu_state)
        assert torch.equal(torch.cuda.get_rng_state(device), caller_cuda_state)
    finally:
        torch.random.set_rng_state(original_cpu_state)
        torch.cuda.set_rng_state(original_cuda_state, device=device)


def test_peer_failure_propagates_after_local_success() -> None:
    system, model = _system(world_size=2, entries=(_entry(3),))
    _move_to_current_cuda(system)
    cast(Any, system)._trainer = SimpleNamespace(global_step=17)

    @contextmanager
    def remote_rank_fails_after_local_success():
        yield
        raise RuntimeError("injected remote-rank failure")

    with (
        patch(
            "nurec.systems.active_tile_mapping_prewarmer.distributed_context",
            side_effect=lambda _context: remote_rank_fails_after_local_success(),
        ),
        pytest.raises(RuntimeError, match="remote-rank failure"),
    ):
        _run_prewarm(system)

    assert model.build_calls == [3]


def test_setup_never_runs_training_prewarm() -> None:
    system, model = _system(world_size=2)
    cast(Any, system).cached_config = SimpleNamespace(in_process_instant_nurec=None)
    cast(Any, model).gaussians_strategy = SimpleNamespace(init_from_datasource=False)
    system._in_process_instant_nurec_initialized = False
    system._active_tile_mapping_prewarmer._finished = True
    initialize = MagicMock()
    cast(Any, model).initialize_strategy_and_maybe_gaussians = initialize
    cast(Any, system)._validate_training_harmonizer_component_only_compatibility = MagicMock()
    cast(Any, system)._validate_component_only_loss_normalization_compatibility = MagicMock()
    cast(Any, system)._register_serialization_state_dict_hook = MagicMock()
    cast(Any, system)._training_harmonizer = MagicMock()

    with (
        patch.object(BaseSystemSO, "setup") as base_setup,
        patch.object(ActiveTileMappingPrewarmer, "run_once") as prewarm,
        patch("nurec.systems.gaussians.PSNRMetric"),
        patch("nurec.systems.gaussians.MetricManager"),
        patch("nurec.systems.gaussians.MetricFactory"),
        patch("nurec.systems.gaussians.StructuralSimilarityIndexMeasure"),
        patch("nurec.systems.gaussians.LearnedPerceptualImagePatchSimilarity"),
    ):
        system.setup("fit")

    base_setup.assert_called_once_with("fit")
    initialize.assert_called_once_with()
    prewarm.assert_not_called()
    assert system._active_tile_mapping_prewarmer._finished is False


def test_on_train_start_does_not_prewarm_before_ddp_buffer_sync() -> None:
    system, model = _system(world_size=2)
    system.resume = None
    cast(Any, model)._log_component_inventory = MagicMock()

    with (
        patch.object(BaseSystemSO, "on_train_start"),
        patch.object(ActiveTileMappingPrewarmer, "run_once") as prewarm,
    ):
        system.on_train_start()

    prewarm.assert_not_called()
    model.on_train_from_scratch_start.assert_called_once_with(system)
    cast(Any, model)._log_component_inventory.assert_called_once_with(when="post-init")


def test_single_rank_training_step_does_not_invoke_prewarm() -> None:
    system, _ = _system(world_size=1)
    cast(Any, system)._trainer = None

    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=1),
        patch.object(ActiveTileMappingPrewarmer, "run_once") as prewarm,
        patch.object(BaseSystemSO, "training_step", return_value=torch.zeros(())),
    ):
        system.training_step(cast(Any, None), 0)

    prewarm.assert_not_called()


def test_first_training_step_uses_restored_state_once() -> None:
    system, model = _system(world_size=2)
    system.resume = "checkpoint.ckpt"
    fit_loop = MagicMock()
    trainer = SimpleNamespace(
        global_step=0,
        fit_loop=fit_loop,
        validate_loop=MagicMock(),
        test_loop=MagicMock(),
        predict_loop=MagicMock(),
        strategy=SimpleNamespace(),
    )
    fit_loop.load_state_dict.side_effect = lambda _state: setattr(trainer, "global_step", 41)
    cast(Any, system)._trainer = trainer
    cast(Any, system).trainer_config = SimpleNamespace(world_size=2)
    state_dict = system.state_dict()
    state_dict["model.calibration_version"] = torch.tensor(73, dtype=torch.int64)
    checkpoint = {
        "loops": {
            "fit_loop": {"restored_step": 41},
            "validate_loop": {},
            "test_loop": {},
            "predict_loop": {},
        },
        "state_dict": state_dict,
        "trainer.world_size": 2,
        "system.checkpoint.last_checkpoint_step": 40,
        "system.checkpoint.training_step_outputs": {},
    }
    # This focused model exposes ``calib`` as a method, whereas the production
    # compatibility shim expects a calibration module. Keep the actual
    # BaseSystem checkpoint-loop restore and use the standard Module loader for
    # this synthetic state dict.
    with patch.object(
        system,
        "load_state_dict",
        side_effect=lambda values, **kwargs: torch.nn.Module.load_state_dict(system, values, **kwargs),
    ):
        system.on_load_checkpoint(cast(Any, checkpoint))
    events: list[str] = []

    def prewarm(**_kwargs: Any) -> None:
        assert int(cast(torch.Tensor, model.calibration_version).item()) == 73
        assert system.global_step == 41
        events.append("prewarm")

    system._active_tile_mapping_prewarmer._run = prewarm  # type: ignore[method-assign]
    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(BaseSystemSO, "training_step", side_effect=lambda *_args: events.append("base")),
    ):
        system.training_step(cast(Any, None), 0)
        system.training_step(cast(Any, None), 1)

    fit_loop.load_state_dict.assert_called_once_with({"restored_step": 41})
    assert events == ["prewarm", "base", "base"]
    assert system._active_tile_mapping_prewarmer._finished is True


def test_in_process_state_restore_rearms_the_one_shot_prewarm() -> None:
    system, _ = _system(world_size=2)
    system._active_tile_mapping_prewarmer._finished = True
    cast(Any, system).model = torch.nn.Module()
    with patch.object(BaseSystemSO, "load_state_dict", return_value=None):
        system.load_state_dict({})
    events: list[str] = []
    system._active_tile_mapping_prewarmer._run = lambda **_kwargs: events.append(  # type: ignore[method-assign]
        "prewarm"
    )

    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(BaseSystemSO, "training_step", side_effect=lambda *_args: events.append("base")),
    ):
        system.training_step(cast(Any, None), 0)
        system.training_step(cast(Any, None), 1)

    assert events == ["prewarm", "base", "base"]
    assert system._active_tile_mapping_prewarmer._finished is True


def test_prewarm_populates_every_cache_entry_with_fixed_seed_equivalence() -> None:
    def run_once() -> tuple[dict[int, torch.Tensor], list[int], list[int | None]]:
        system, model = _system(world_size=2, entries=(_entry(1), _entry(4)))
        _move_to_current_cuda(system)
        caller_state = torch.random.get_rng_state().clone()
        _run_prewarm(system)
        assert torch.equal(torch.random.get_rng_state(), caller_state)
        return (
            {key: value.clone() for key, value in model.cache.items()},
            model.build_calls,
            model.calib_global_steps,
        )

    torch.manual_seed(999)
    first_cache, first_calls, first_calib_global_steps = run_once()
    torch.rand(7)
    second_cache, second_calls, second_calib_global_steps = run_once()

    assert first_calls == second_calls == [1, 4]
    assert first_calib_global_steps == second_calib_global_steps == [None, None]
    assert all(torch.equal(first_cache[key], second_cache[key]) for key in first_cache)


def test_failure_synchronizes_inside_coordinated_scope_without_a_barrier() -> None:
    system, _ = _system(world_size=2, entries=(_entry(3),), fail=True)
    device = _move_to_current_cuda(system)
    coordination = SimpleNamespace(entered=False, exited=False)
    current_stream = MagicMock()

    @contextmanager
    def coordinated_exceptions():
        coordination.entered = True
        try:
            yield
        finally:
            coordination.exited = True

    with (
        patch(
            "nurec.systems.active_tile_mapping_prewarmer.distributed_context",
            side_effect=lambda _context: coordinated_exceptions(),
        ),
        patch.object(torch.cuda, "current_stream", return_value=current_stream) as get_current_stream,
        patch.object(torch.distributed, "barrier") as barrier,
        pytest.raises(ValueError, match="injected"),
    ):
        _run_prewarm(system)

    assert coordination.entered and coordination.exited
    get_current_stream.assert_called_once_with(device=device)
    current_stream.synchronize.assert_called_once_with()
    barrier.assert_not_called()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--nccl-worker":
        result = _nccl_asymmetric_failure_worker(int(sys.argv[2]), sys.argv[3])
        print(json.dumps(result), flush=True)
    elif len(sys.argv) == 4 and sys.argv[1] == "--ddp-buffer-sync-worker":
        result = _nccl_ddp_buffer_sync_worker(int(sys.argv[2]), sys.argv[3])
        print(json.dumps(result), flush=True)
    else:
        raise SystemExit("Expected --nccl-worker or --ddp-buffer-sync-worker <rank> <init-file>")
