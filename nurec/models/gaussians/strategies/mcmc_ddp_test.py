# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
import time

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import gsplat
import pytest
import torch

from nurec.config.model import MCMCStrategyConfig, TamingGSMCMCStrategyConfig
from nurec.models.base import BaseModel
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import _NUREC_TO_MCMC_SPLAT, BaseGaussianModel
from nurec.models.gaussians.strategies.mcmc import MCMCStrategy, TamingGSMCMCStrategy, _finish_taminggs_addition_scores
from nurec.models.gaussians.strategies.test_utils import make_trainer_cfg
from nurec.models.nn_extensions import TypedModuleDict
from nurec.systems.base import DDPStrategySO


_WORLD_SIZE = 2
_LAYER_COUNTS = {"first": 5, "second": 4}
_CHECKPOINT_WORKER_TIMEOUT_SECONDS = 120
_CHECKPOINT_WORKER_CLEANUP_TIMEOUT_SECONDS = 10
_CHECKPOINT_WORKER_OUTPUT_MAX_BYTES = 1024 * 1024


class _RankLocalMCMCNode(BaseGaussianModel):
    config: Any
    row_ids: torch.Tensor

    def __init__(self, rank: int, count: int, device: torch.device) -> None:
        torch.nn.Module.__init__(self)
        self.register_buffer("_device_indicator", torch.tensor(0, device=device), persistent=False)
        self.config = cast(Any, SimpleNamespace(density_activation="sigmoid", scale_activation="exp"))
        self.density_activation = torch.nn.Sigmoid()
        self.scale_activation = cast(Any, torch.exp)
        rows = torch.arange(count, dtype=torch.float32, device=device) + rank * 100
        self.positions = torch.nn.Parameter(rows[:, None].expand(-1, 3).clone())
        self.rotations = torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(count, -1).clone()
        )
        self.scales = torch.nn.Parameter(torch.zeros(count, 3, device=device))
        self.densities = torch.nn.Parameter(torch.linspace(-0.5, 0.5, count, device=device)[:, None])
        self.extra_signal = torch.nn.Parameter(rows[:, None].expand(-1, 2).clone())
        self.camera_extra_signal = torch.nn.Parameter(rows[:, None].clone())
        self.lidar_extra_signal = torch.nn.Parameter((rows + 1000)[:, None].clone())
        self.register_buffer("row_ids", torch.arange(count, device=device, dtype=torch.int64) + rank * 100)
        parameters = list(self.get_parameters().values())
        optimizer = torch.optim.Adam(
            [
                {"params": [parameter], "name": f"model.row_{index}", "lr": 1.0e-3}
                for index, parameter in enumerate(parameters)
            ]
        )
        self.optimizers = [{"optimizer": optimizer}]
        self.refresh_mcmc_scene("background")

    def get_additional_buffers(self) -> dict[str, torch.Tensor]:
        return {"row_ids": self.row_ids}


def _make_concrete_mcmc_strategy(
    config: MCMCStrategyConfig | TamingGSMCMCStrategyConfig,
    trainer_config,
    init_from_datasource: bool,
    gaussians_nodes: TypedModuleDict,
) -> Any:
    if isinstance(config, TamingGSMCMCStrategyConfig):
        return TamingGSMCMCStrategy(config, trainer_config, init_from_datasource, gaussians_nodes)
    return MCMCStrategy(config, trainer_config, init_from_datasource, gaussians_nodes)


def _global_rows(count: int, width: int, base: float, device: torch.device) -> torch.Tensor:
    return torch.arange(count * width, dtype=torch.float32, device=device).reshape(count, width) + base


def _layer_base(layer_id: str) -> float:
    return 1000.0 if layer_id == "first" else 2000.0


class _CheckpointNode(torch.nn.Module):
    def __init__(
        self,
        layer_id: str,
        rank: int,
        device: torch.device,
        *,
        reset: bool = False,
        world_size: int = _WORLD_SIZE,
    ) -> None:
        super().__init__()
        rows = _global_rows(_LAYER_COUNTS[layer_id], 3, _layer_base(layer_id), device)[rank::world_size]
        self.positions = torch.nn.Parameter(torch.full_like(rows, -1) if reset else rows)

    def configure_sharded_params_and_buffers(self) -> list[str]:
        BaseModel.mark_as_sharded(self.positions)
        return ["positions"]


class _CheckpointStrategy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.invisible_steps = torch.nn.ParameterDict()

    def configure_sharded_params_and_buffers(self) -> list[str]:
        names: list[str] = []
        for layer_id, counter in self.invisible_steps.items():
            names.append(f"invisible_steps.{layer_id}")
            if not isinstance(counter, torch.nn.UninitializedParameter):
                BaseModel.mark_as_sharded(counter)
        return names


class _CheckpointComposite(torch.nn.Module):
    def __init__(
        self,
        rank: int,
        device: torch.device,
        *,
        reset: bool = False,
        initialize_visibility: bool = True,
        track_visibility: bool = True,
        world_size: int = _WORLD_SIZE,
    ) -> None:
        super().__init__()
        self.gaussian_process_group: torch.distributed.ProcessGroup | None = None
        self.register_load_state_dict_pre_hook(GaussiansComposite._reset_incompatible_mcmc_visibility_state_on_load)
        self.gaussians_nodes = TypedModuleDict(
            {
                layer_id: _CheckpointNode(
                    layer_id,
                    rank,
                    device,
                    reset=reset,
                    world_size=world_size,
                )
                for layer_id in _LAYER_COUNTS
            }
        )
        self.gaussians_strategy = _CheckpointStrategy()
        if not track_visibility:
            return
        for layer_index, (layer_id, count) in enumerate(_LAYER_COUNTS.items(), start=1):
            if not initialize_visibility:
                self.gaussians_strategy.invisible_steps[layer_id] = torch.nn.UninitializedParameter(requires_grad=False)
                continue
            values = torch.arange(count, dtype=torch.int32, device=device)[rank::world_size] + layer_index * 100
            if reset:
                values.fill_(-1)
            self.gaussians_strategy.invisible_steps[layer_id] = torch.nn.Parameter(values, requires_grad=False)

    def configure_sharded_params_and_buffers(self) -> list[str]:
        return GaussiansComposite.configure_sharded_params_and_buffers(self)  # type: ignore[arg-type]


class _CheckpointModule(torch.nn.Module):
    def __init__(
        self,
        rank: int,
        device: torch.device,
        *,
        reset: bool = False,
        initialize_visibility: bool = True,
        track_visibility: bool = True,
        world_size: int = _WORLD_SIZE,
    ) -> None:
        super().__init__()
        self.preflight_distributed_checkpoint_state_dict = MagicMock()
        self.model = _CheckpointComposite(
            rank,
            device,
            reset=reset,
            initialize_visibility=initialize_visibility,
            track_visibility=track_visibility,
            world_size=world_size,
        )

    def configure_sharded_params_and_buffers(self) -> list[str]:
        return [f"model.{name}" for name in self.model.configure_sharded_params_and_buffers()]


class _CheckpointDDPStrategy(DDPStrategySO):
    @property
    def base_system(self) -> Any:
        assert self.lightning_module is not None
        return self.lightning_module


def _make_ddp_strategy(
    module: _CheckpointModule,
    gaussian_process_group: Any | None = None,
) -> _CheckpointDDPStrategy:
    strategy = object.__new__(_CheckpointDDPStrategy)
    strategy._lightning_module = cast(Any, module)
    strategy._model = None
    strategy._sharded_params_and_buffers = module.configure_sharded_params_and_buffers()
    strategy._verbose = False
    strategy.gaussian_process_group = cast(
        Any,
        gaussian_process_group if gaussian_process_group is not None else object(),
    )
    module.model.gaussian_process_group = strategy.gaussian_process_group
    return strategy


def _full_state(device: torch.device, *, legacy_visibility: bool = False) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for layer_index, (layer_id, count) in enumerate(_LAYER_COUNTS.items(), start=1):
        state[f"model.gaussians_nodes.{layer_id}.positions"] = _global_rows(
            count,
            3,
            _layer_base(layer_id),
            device,
        )
        visibility = torch.arange(count, dtype=torch.int32, device=device) + layer_index * 100
        if legacy_visibility:
            visibility = visibility[0::_WORLD_SIZE]
        state[f"model.gaussians_strategy.invisible_steps.{layer_id}"] = visibility
    return state


def _assert_local_state(module: _CheckpointModule, rank: int, device: torch.device) -> None:
    for layer_index, (layer_id, count) in enumerate(_LAYER_COUNTS.items(), start=1):
        expected_positions = _global_rows(count, 3, _layer_base(layer_id), device)[rank::_WORLD_SIZE]
        expected_visibility = (torch.arange(count, dtype=torch.int32, device=device) + layer_index * 100)[
            rank::_WORLD_SIZE
        ]
        node = module.model.gaussians_nodes[layer_id]
        counter = module.model.gaussians_strategy.invisible_steps[layer_id]
        torch.testing.assert_close(node.positions, expected_positions)
        torch.testing.assert_close(counter, expected_visibility)
        assert BaseModel.is_sharded(node.positions)
        assert BaseModel.is_sharded(counter)


def _taminggs_coschedule_contract(rank: int, device: torch.device) -> None:
    config = TamingGSMCMCStrategyConfig.model_validate(
        {
            "name": "taminggs_mcmc",
            "relocate": {"start_iteration": 0, "end_iteration": 8, "frequency": 4},
            "add": {
                "start_iteration": 0,
                "end_iteration": 8,
                "frequency": 4,
                "max_n_gaussians": 44,
                "taminggs": {},
            },
            "perturb": {"start_iteration": 1, "end_iteration": 1, "frequency": 1, "noise_lr": {"default": 1.0}},
        }
    )
    trainer_config = make_trainer_cfg().model_copy(update={"world_size": _WORLD_SIZE})
    for layout in ((0, 40), (20, 20)):
        count = layout[rank]
        node = _RankLocalMCMCNode(rank, count, device)
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": node}))
        strategy = _make_concrete_mcmc_strategy(config, trainer_config, False, nodes).to(device)
        strategy.maybe_initialize_buffers(nodes)
        with torch.no_grad():
            if layout[0] == 0 and rank == 1:
                node.densities[:2] = -20
            elif layout[0] > 0 and rank == 1:
                node.densities.fill_(-20)
        node.positions.grad = (
            torch.arange(1, count + 1, device=device, dtype=torch.float32)[:, None].expand(-1, 3).clone()
        )
        initial_visibility = torch.arange(1, count + 1, device=device, dtype=torch.int32)
        strategy.invisible_steps["background"].copy_(initial_visibility)
        original_scores = strategy._taminggs_source_weights(strategy._addition_selected_nodes(nodes))
        with patch.object(strategy, "_add_to_pool", wraps=strategy._add_to_pool) as addition:
            strategy._update_step_train_batch_end_impl(0, 2, cast(Any, None), None, nodes)
        addition.assert_called_once()
        transported = addition.call_args.kwargs["source_weights_by_owner"]
        if count:
            torch.testing.assert_close(
                transported["background"], original_scores["background"][node.row_ids[:count] - rank * 100]
            )
            retained_count = count - 2 if layout[0] == 0 else count
            expected_visibility = torch.cat(
                [
                    initial_visibility[node.row_ids[:retained_count] - rank * 100],
                    initial_visibility.new_zeros(node.get_num_gaussians() - retained_count),
                ]
            )
            torch.testing.assert_close(strategy.invisible_steps["background"], expected_visibility)
        else:
            assert transported == {}
            assert node.get_num_gaussians() == 0
        assert int(strategy._taminggs_initial_count) == 40
        torch.testing.assert_close(strategy._taminggs_initial_counts_by_rank, torch.tensor(layout, device=device))
        total = torch.tensor(node.get_num_gaussians(), device=device)
        torch.distributed.all_reduce(total)
        assert int(total) == 44
        assert strategy.invisible_steps["background"].shape == (node.get_num_gaussians(),)
        scene = node.get_mcmc_scene()
        assert scene is not None
        for name, parameter in node.get_parameters().items():
            assert scene.splats[_NUREC_TO_MCMC_SPLAT.get(name, name)] is parameter


def _taminggs_numerics_contract(rank: int, device: torch.device) -> None:
    """Real NCCL partitions share global medians and match the eager single-device score."""
    config = TamingGSMCMCStrategyConfig.model_validate(
        {
            "name": "taminggs_mcmc",
            "opacity_threshold": 0.1,
            "add": {"start_iteration": 0, "end_iteration": 4, "frequency": 2, "max_n_gaussians": 16},
        }
    )
    trainer = make_trainer_cfg().model_copy(update={"world_size": _WORLD_SIZE})
    strategy = TamingGSMCMCStrategy(config, trainer, False, TypedModuleDict()).to(device)
    raw = torch.tensor([float("nan"), float("inf"), -float("inf"), -1.0, 0.0, 2.0, 4.0, 8.0], device=device)
    opacity = torch.tensor([0.3, 0.4, 0.0, 0.1, 0.5, 0.6, 0.0, 0.8], device=device)
    expected, expected_mass = _finish_taminggs_addition_scores(
        raw[:, None],
        torch.tensor([4.0], device=device),
        (1.0,),
        10.0,
        0.01,
        opacity,
        0.1,
        (8,),
        enable_torch_compile=False,
    )
    for layout in ((0, 8), (3, 5), (8, 0), (0, 0)):
        offset = 0 if rank == 0 else layout[0]
        count = layout[rank]
        local = raw[offset : offset + count] if sum(layout) else raw[:0]
        local_opacity = opacity[offset : offset + count] if sum(layout) else opacity[:0]
        matrix, medians = strategy._taminggs_component_medians({"gradient": local})
        torch.testing.assert_close(medians, torch.tensor([4.0 if sum(layout) else 1.0], device=device))
        score, masses = _finish_taminggs_addition_scores(
            matrix,
            medians,
            (1.0,),
            10.0,
            0.01,
            local_opacity,
            0.1,
            (count,),
            enable_torch_compile=True,
        )
        torch.testing.assert_close(score, expected[offset : offset + count] if sum(layout) else expected[:0])
        global_mass = masses[0].clone()
        torch.distributed.all_reduce(global_mass)
        torch.testing.assert_close(global_mass, expected_mass[0] if sum(layout) else expected_mass[0].new_zeros(()))
    strategy._taminggs_initial_count.fill_(8)
    assert (
        strategy._taminggs_local_growth_quota(
            global_step=1, local_count=4, local_score_statistics={"background": (0.0, rank + 1)}
        )
        == 0
    )


def _checkpoint_worker(rank: int, init_file: str) -> None:
    print(f"rank {rank}: starting", flush=True)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=45),
    )
    gaussian_process_group = torch.distributed.new_group(backend="nccl")
    print(f"rank {rank}: process group ready", flush=True)
    try:
        _taminggs_numerics_contract(rank, device)
        taminggs_config = TamingGSMCMCStrategyConfig.model_validate(
            {
                "name": "taminggs_mcmc",
                "relocate": {"start_iteration": 1, "end_iteration": 1, "frequency": 1},
                "add": {
                    "start_iteration": 0,
                    "end_iteration": 10,
                    "frequency": 2,
                    "max_n_gaussians": 10,
                    "taminggs": {},
                },
                "perturb": {
                    "start_iteration": 1,
                    "end_iteration": 1,
                    "frequency": 1,
                    "noise_lr": {"default": 1.0},
                },
                "binom_n_max": 51,
                "opacity_threshold": 0.005,
                "exclude_layer_ids": [],
            }
        )
        trainer_config = make_trainer_cfg().model_copy(update={"world_size": _WORLD_SIZE})

        for proxy_enabled in (False, True):
            cpu_config: MCMCStrategyConfig | TamingGSMCMCStrategyConfig = taminggs_config.model_copy(deep=True)
            if not proxy_enabled:
                ordinary_config = cpu_config.model_dump()
                ordinary_config["name"] = "mcmc"
                ordinary_config["add"].pop("taminggs")
                cpu_config = MCMCStrategyConfig.model_validate(ordinary_config)
            cpu_nodes: TypedModuleDict = TypedModuleDict(
                cast(Any, {"background": _RankLocalMCMCNode(rank, 3 - rank, torch.device("cpu"))})
            )
            cpu_strategy = _make_concrete_mcmc_strategy(cpu_config, trainer_config, False, cpu_nodes)
            cpu_strategy.maybe_initialize_buffers(cpu_nodes)
            assert cpu_strategy.binoms.device.type == "cpu"
            assert cpu_nodes["background"].positions.device.type == "cpu"
            if proxy_enabled:
                assert int(cpu_strategy._taminggs_initial_count.item()) == 5
                torch.testing.assert_close(cpu_strategy._taminggs_initial_counts_by_rank, torch.tensor([3, 2]))

        lazy_cpu_node = _RankLocalMCMCNode(rank, 2, torch.device("cpu"))
        if rank == 1:
            lazy_cpu_node.positions = torch.nn.UninitializedParameter()
        lazy_cpu_nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": lazy_cpu_node}))
        lazy_cpu_strategy = _make_concrete_mcmc_strategy(taminggs_config, trainer_config, False, lazy_cpu_nodes)
        lazy_cpu_strategy.maybe_initialize_buffers(lazy_cpu_nodes)
        assert int(lazy_cpu_strategy._taminggs_initial_count.item()) == -1
        torch.testing.assert_close(lazy_cpu_strategy._taminggs_initial_counts_by_rank, torch.tensor([-1, -1]))

        local_owner_count = 2 if rank == 0 else 1
        taminggs_nodes: TypedModuleDict = TypedModuleDict(
            cast(
                Any,
                {
                    f"owner_{owner_index}": _RankLocalMCMCNode(rank, local_owner_count, device)
                    for owner_index in range(3)
                },
            )
        )
        taminggs = _make_concrete_mcmc_strategy(taminggs_config, trainer_config, False, taminggs_nodes).to(device)
        taminggs.maybe_initialize_buffers(taminggs_nodes)
        torch.testing.assert_close(
            taminggs._taminggs_initial_counts_by_rank,
            torch.tensor([6, 3], dtype=torch.int64, device=device),
        )
        local_count = sum(node.get_num_gaussians() for node in taminggs_nodes.values())
        local_target = torch.tensor(
            local_count
            + taminggs._taminggs_local_growth_quota(
                global_step=4,
                local_count=local_count,
                local_score_statistics=taminggs._taminggs_local_score_statistics(
                    taminggs_nodes,
                    {
                        owner_id: torch.ones(node.get_num_gaussians(), device=device)
                        for owner_id, node in taminggs_nodes.items()
                    },
                ),
            ),
            dtype=torch.int64,
            device=device,
        )
        targets = [torch.zeros_like(local_target) for _ in range(_WORLD_SIZE)]
        torch.distributed.all_gather(targets, local_target)
        assert [int(target.item()) for target in targets] == [7, 3]

        # The shared budget is global: neither local shard exceeds it, but
        # their sum does. All ranks must preserve the same initial ceiling.
        overflow_config = taminggs_config.model_copy(deep=True)
        overflow_config.add.max_n_gaussians = 8
        overflow = _make_concrete_mcmc_strategy(overflow_config, trainer_config, False, taminggs_nodes).to(device)
        overflow.maybe_initialize_buffers(taminggs_nodes)
        assert int(overflow._taminggs_initial_count.item()) == 9
        torch.testing.assert_close(
            overflow._taminggs_initial_counts_by_rank, torch.tensor([6, 3], dtype=torch.int64, device=device)
        )
        for step in (2, 4):
            assert overflow._taminggs_global_target_count(step) == 9
            assert (
                overflow._taminggs_local_growth_quota(
                    global_step=step,
                    local_count=local_count,
                    local_score_statistics=overflow._taminggs_local_score_statistics(
                        taminggs_nodes,
                        {
                            owner_id: torch.ones(node.get_num_gaussians(), device=device)
                            for owner_id, node in taminggs_nodes.items()
                        },
                    ),
                )
                == 0
            )

        zero_donor_nodes: TypedModuleDict = TypedModuleDict(
            cast(Any, {"background": _RankLocalMCMCNode(rank, 0 if rank == 0 else 9, device)})
        )
        zero_donor_taminggs = _make_concrete_mcmc_strategy(taminggs_config, trainer_config, False, zero_donor_nodes).to(
            device
        )
        zero_donor_taminggs.maybe_initialize_buffers(zero_donor_nodes)
        local_target = torch.tensor(
            zero_donor_nodes["background"].get_num_gaussians()
            + zero_donor_taminggs._taminggs_local_growth_quota(
                global_step=4,
                local_count=zero_donor_nodes["background"].get_num_gaussians(),
                local_score_statistics=zero_donor_taminggs._taminggs_local_score_statistics(
                    zero_donor_nodes,
                    {"background": torch.ones(zero_donor_nodes["background"].get_num_gaussians(), device=device)},
                ),
            ),
            dtype=torch.int64,
            device=device,
        )
        targets = [torch.zeros_like(local_target) for _ in range(_WORLD_SIZE)]
        torch.distributed.all_gather(targets, local_target)
        assert [int(target.item()) for target in targets] == [0, 10]

        asymmetric_node = _RankLocalMCMCNode(rank, 2, device)
        if rank == 1:
            asymmetric_node.positions = torch.nn.UninitializedParameter()
        asymmetric_nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": asymmetric_node}))
        asymmetric_taminggs = _make_concrete_mcmc_strategy(taminggs_config, trainer_config, False, asymmetric_nodes).to(
            device
        )
        asymmetric_taminggs.maybe_initialize_buffers(asymmetric_nodes)
        assert int(asymmetric_taminggs._taminggs_initial_count.item()) == -1
        torch.testing.assert_close(
            asymmetric_taminggs._taminggs_initial_counts_by_rank,
            torch.tensor([-1, -1], dtype=torch.int64, device=device),
        )
        print(f"rank {rank}: TamingGS-inspired proxy allocation contract complete", flush=True)
        _taminggs_coschedule_contract(rank, device)
        print(f"rank {rank}: real TamingGS co-schedule contract complete", flush=True)

        module = _CheckpointModule(rank, device)
        strategy = _make_ddp_strategy(module, gaussian_process_group)
        print(f"rank {rank}: gathering checkpoint", flush=True)
        gathered = strategy.lightning_module_state_dict()
        expected = _full_state(device)
        assert gathered.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(gathered[name], expected[name])

        restored = _CheckpointModule(rank, device, reset=True, initialize_visibility=False)
        restore_strategy = _make_ddp_strategy(restored, gaussian_process_group)
        expected_visibility_names = {
            f"model.gaussians_strategy.invisible_steps.{layer_id}" for layer_id in _LAYER_COUNTS
        }
        assert expected_visibility_names <= set(restore_strategy.sharded_params_and_buffers)
        restore_strategy.load_model_state_dict({"state_dict": gathered})
        _assert_local_state(restored, rank, device)

        legacy = _CheckpointModule(rank, device, reset=True, initialize_visibility=False)
        legacy_strategy = _make_ddp_strategy(legacy, gaussian_process_group)
        legacy_strategy.load_model_state_dict({"state_dict": _full_state(device, legacy_visibility=True)})
        for layer_id, count in _LAYER_COUNTS.items():
            counter = legacy.model.gaussians_strategy.invisible_steps[layer_id]
            expected_count = len(range(rank, count, _WORLD_SIZE))
            torch.testing.assert_close(counter, torch.zeros(expected_count, dtype=torch.int32, device=device))
            assert BaseModel.is_sharded(counter)
        print(f"rank {rank}: checkpoint contract complete", flush=True)

    finally:
        torch.distributed.destroy_process_group()


def test_visibility_checkpoint_compatibility_on_single_rank_load(caplog: pytest.LogCaptureFixture) -> None:
    device = torch.device("cpu")
    restored = _CheckpointModule(
        rank=0,
        device=device,
        reset=True,
        initialize_visibility=False,
        world_size=1,
    )

    with caplog.at_level("WARNING", logger="nurec.models.gaussians.gaussians_composite"):
        restored.load_state_dict(
            _full_state(device, legacy_visibility=True),
            strict=True,
            assign=True,
        )

    for layer_id, count in _LAYER_COUNTS.items():
        torch.testing.assert_close(
            restored.model.gaussians_strategy.invisible_steps[layer_id],
            torch.zeros(count, dtype=torch.int32),
        )
    assert caplog.text.count("incompatible MCMC visibility counter") == len(_LAYER_COUNTS)

    disabled = _CheckpointModule(rank=0, device=device, track_visibility=False, world_size=1)
    disabled.load_state_dict(_full_state(device), strict=True, assign=True)
    assert len(disabled.model.gaussians_strategy.invisible_steps) == 0


def test_checkpoint_collective_groups_are_rank_invariant() -> None:
    def gather_groups(rank: int) -> list[tuple[str, ...]]:
        module = _CheckpointModule(rank, torch.device("cpu"))
        strategy = _make_ddp_strategy(module)
        groups: list[tuple[str, ...]] = []

        def gather(
            parameters: dict[str, torch.Tensor],
            *,
            stable: bool,
            group: Any,
        ) -> dict[str, torch.Tensor]:
            assert stable
            assert group is strategy.gaussian_process_group
            groups.append(tuple(parameters))
            return parameters

        with patch("nurec.systems.base.distributed_all_gather_gaussian_parameters", side_effect=gather):
            strategy.lightning_module_state_dict()
        return groups

    rank_zero_groups = gather_groups(0)
    assert gather_groups(1) == rank_zero_groups
    assert ("model.gaussians_strategy.invisible_steps.first",) in rank_zero_groups
    assert ("model.gaussians_strategy.invisible_steps.second",) in rank_zero_groups


def test_checkpoint_preflight_runs_before_state_dict() -> None:
    module = _CheckpointModule(rank=0, device=torch.device("cpu"), world_size=1)
    strategy = _make_ddp_strategy(module)
    module.preflight_distributed_checkpoint_state_dict.side_effect = RuntimeError("unsafe MCMC state")

    with (
        patch.object(module, "state_dict", wraps=module.state_dict) as state_dict,
        pytest.raises(RuntimeError, match="unsafe MCMC state"),
    ):
        strategy.lightning_module_state_dict()

    state_dict.assert_not_called()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not gsplat.has_reloc() or not gsplat.has_adam(),
    reason="MCMC strategy requires CUDA, RELOC, and ADAM",
)
def test_scheduled_topology_is_rank_local_aligned_and_collective_free() -> None:
    config = MCMCStrategyConfig.model_validate(
        {
            "name": "mcmc",
            "relocate": {
                "start_iteration": 4,
                "end_iteration": 4,
                "frequency": 1,
                "max_invisible_steps": 10,
            },
            "add": {"start_iteration": 0, "end_iteration": 4, "frequency": 1, "max_n_gaussians": 42},
            "perturb": {
                "start_iteration": 4,
                "end_iteration": 4,
                "frequency": 1,
                "noise_lr": {"default": 1.0},
            },
            "binom_n_max": 51,
            "opacity_threshold": 0.005,
            "exclude_layer_ids": [],
        }
    )
    device = torch.device("cuda")

    for rank in range(_WORLD_SIZE):
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _RankLocalMCMCNode(rank, 20, device)}))
        trainer_config = make_trainer_cfg()
        trainer_config.world_size = _WORLD_SIZE
        strategy = _make_concrete_mcmc_strategy(config, trainer_config, False, nodes).to(device)
        strategy.maybe_initialize_buffers(nodes)
        initial_visibility = torch.arange(20, dtype=torch.int32, device=device) + rank * 100 + 10
        strategy.invisible_steps["background"].copy_(initial_visibility)
        node = cast(_RankLocalMCMCNode, nodes["background"])
        initial_row_ids = node.row_ids.clone()

        with (
            patch.object(torch.distributed, "all_reduce") as all_reduce,
            patch.object(torch.distributed, "all_gather") as all_gather,
            patch.object(torch.distributed, "all_gather_into_tensor") as all_gather_into_tensor,
            patch.object(torch.distributed, "broadcast") as broadcast,
        ):
            strategy.update_step_train_batch_end(1, 1, cast(Any, None), None, nodes)

        for collective in (all_reduce, all_gather, all_gather_into_tensor, broadcast):
            collective.assert_not_called()

        scene = node.get_mcmc_scene()
        assert scene is not None
        assert scene.id == "background"
        assert scene.num_gaussians() == node.get_num_gaussians() == 21
        optimizer_parameters = {
            id(parameter) for group in node.optimizers[0]["optimizer"].param_groups for parameter in group["params"]
        }
        for parameter_name, parameter in node.get_parameters().items():
            splat_name = _NUREC_TO_MCMC_SPLAT.get(parameter_name, parameter_name)
            assert scene.splats[splat_name] is parameter
            assert id(parameter) in optimizer_parameters

        assert node.row_ids.shape == (21,)
        torch.testing.assert_close(node.row_ids[:20], initial_row_ids)
        assert rank * 100 <= int(node.row_ids[-1]) < rank * 100 + 20
        visibility = strategy.invisible_steps["background"]
        assert visibility.shape == (21,)
        torch.testing.assert_close(visibility[:20], initial_visibility)
        assert visibility[-1] == 0


def test_split_state_only_allows_missing_lazy_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _CheckpointModule(
        rank=0,
        device=torch.device("cpu"),
        initialize_visibility=False,
        world_size=1,
    )
    strategy = _make_ddp_strategy(module)
    state = {name: value for name, value in _full_state(torch.device("cpu")).items() if ".invisible_steps." not in name}
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

    strategy.split_params_and_buffers(state)

    module.model.gaussians_strategy.invisible_steps["first"] = torch.nn.Parameter(
        torch.zeros(_LAYER_COUNTS["first"], dtype=torch.int32),
        requires_grad=False,
    )
    with pytest.raises(KeyError, match="Missing materialized sharded model state"):
        strategy.split_params_and_buffers(state)


def _prepared_visibility_worker(rank: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=45),
    )
    try:

        class VisibilityNode(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.positions = torch.nn.Parameter(torch.zeros(2, 3))

            @property
            def device(self) -> torch.device:
                return self.positions.device

            def get_num_gaussians(self) -> int:
                return self.positions.shape[0]

        for taminggs in (False, True):
            config: MCMCStrategyConfig | TamingGSMCMCStrategyConfig
            config_values = {"relocate": {"max_invisible_steps": 10}, "add": {"max_n_gaussians": 4}}
            if taminggs:
                config = TamingGSMCMCStrategyConfig.model_validate({"name": "taminggs_mcmc", **config_values})
            else:
                config = MCMCStrategyConfig.model_validate({"name": "mcmc", **config_values})
            nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": VisibilityNode()}))
            trainer = make_trainer_cfg().model_copy(update={"world_size": _WORLD_SIZE})
            with patch("gsplat.has_reloc", return_value=True):
                strategy = _make_concrete_mcmc_strategy(config, trainer, False, nodes)
            strategy.maybe_initialize_buffers(nodes)
            strategy.invisible_steps["background"].fill_(1)
            strategy.frozen = True
            strategy.on_train_step_start()
            visibility = torch.zeros(4, dtype=torch.uint8)
            visibility[rank * 2] = 1
            # The production packet uses CUDA/NCCL. Exercise its resolved
            # global-union/owner-compaction contract on CPU/Gloo here.
            torch.distributed.all_reduce(visibility, op=torch.distributed.ReduceOp.MAX)
            local_visibility = visibility[rank * 2 : rank * 2 + 2]
            if taminggs:
                strategy.update_prepared_visibility(visibility=local_visibility, gaussians_nodes=nodes)
            else:
                results = object()
                strategy.update_visibility_counters(results=results, visibility=local_visibility, gaussians_nodes=nodes)
                # Ordinary MCMC deduplicates repeated regularizers on the same render.
                strategy.update_visibility_counters(results=results, visibility=torch.ones(2), gaussians_nodes=nodes)
            expected = torch.tensor([0, 2], dtype=torch.int32)
            torch.testing.assert_close(strategy.invisible_steps["background"], expected)
            strategy.update_step_train_batch_end(0, 0, cast(Any, None), None, nodes)
            torch.testing.assert_close(strategy.invisible_steps["background"], expected)
    finally:
        torch.distributed.destroy_process_group()


def _read_worker_log(path: Path, max_bytes: int = _CHECKPOINT_WORKER_OUTPUT_MAX_BYTES) -> str:
    with path.open("rb") as output_file:
        output_file.seek(0, os.SEEK_END)
        output_size = output_file.tell()
        output_file.seek(max(0, output_size - max_bytes))
        output = output_file.read().decode("utf-8", errors="replace")
    if output_size > max_bytes:
        return f"<output truncated to last {max_bytes} bytes>\n{output}"
    return output


def _run_checkpoint_workers(
    init_file: Path,
    *,
    timeout_seconds: float = _CHECKPOINT_WORKER_TIMEOUT_SECONDS,
    worker_flag: str = "--checkpoint-worker",
) -> None:
    workers: list[subprocess.Popen[bytes]] = []
    log_files: list[Any] = []
    log_paths: list[Path] = []
    timed_out = False
    cleanup_errors: list[str] = []
    worker_env = os.environ | {
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    try:
        for rank in range(_WORLD_SIZE):
            log_path = init_file.with_name(f"{init_file.name}-rank-{rank}.log")
            log_file = log_path.open("wb")
            log_paths.append(log_path)
            log_files.append(log_file)
            worker = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    worker_flag,
                    str(rank),
                    str(init_file),
                ],
                env=worker_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            workers.append(worker)

        deadline = time.monotonic() + timeout_seconds
        while any(worker.poll() is None for worker in workers) and time.monotonic() < deadline:
            time.sleep(0.1)
        timed_out = any(worker.poll() is None for worker in workers)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()

        cleanup_deadline = time.monotonic() + _CHECKPOINT_WORKER_CLEANUP_TIMEOUT_SECONDS
        for rank, worker in enumerate(workers):
            try:
                worker.wait(timeout=max(0.0, cleanup_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                cleanup_errors.append(f"rank {rank} did not exit after kill")

        for log_file in log_files:
            log_file.close()

    outputs = [_read_worker_log(log_path) for log_path in log_paths]
    diagnostics = "\n".join(f"rank {rank}:\n{output}" for rank, output in enumerate(outputs))
    failed_workers = [
        f"rank {rank} exited with code {worker.returncode}"
        for rank, worker in enumerate(workers)
        if worker.returncode != 0
    ]
    if timed_out or cleanup_errors or failed_workers:
        reasons = []
        if timed_out:
            reasons.append(f"did not finish within {timeout_seconds} seconds")
        reasons.extend(cleanup_errors)
        reasons.extend(failed_workers)
        pytest.fail(f"checkpoint workers {'; '.join(reasons)}:\n{diagnostics}")


def test_worker_log_is_bounded_and_decodes_invalid_bytes(tmp_path: Path) -> None:
    log_path = tmp_path / "worker.log"
    log_path.write_bytes(b"abcd\xff")

    assert _read_worker_log(log_path, max_bytes=4) == "<output truncated to last 4 bytes>\nbcd�"


def test_checkpoint_worker_partial_startup_is_reaped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_popen = subprocess.Popen
    started_workers: list[subprocess.Popen[bytes]] = []

    def fail_second_spawn(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        if started_workers:
            raise OSError("rank 1 spawn failed")
        worker = real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
        )
        started_workers.append(worker)
        return worker

    monkeypatch.setattr(subprocess, "Popen", fail_second_spawn)

    with pytest.raises(OSError, match="rank 1 spawn failed"):
        _run_checkpoint_workers(tmp_path / "partial-start-init")

    assert started_workers[0].poll() is not None


def test_checkpoint_worker_timeout_reaps_and_decodes_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_popen = subprocess.Popen
    started_workers: list[subprocess.Popen[bytes]] = []

    def spawn_stuck_worker(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        rank = len(started_workers)
        worker = real_popen(
            [
                sys.executable,
                "-c",
                f'import os, time; os.write(1, b"rank {rank}: \\xff\\n"); time.sleep(60)',
            ],
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
        )
        started_workers.append(worker)
        return worker

    monkeypatch.setattr(subprocess, "Popen", spawn_stuck_worker)

    with pytest.raises(pytest.fail.Exception) as error:
        _run_checkpoint_workers(tmp_path / "timeout-init", timeout_seconds=1)

    message = str(error.value)
    assert "did not finish within 1 seconds" in message
    assert "rank 0: �" in message
    assert "rank 1: �" in message
    assert all(worker.poll() is not None for worker in started_workers)


def test_checkpoint_worker_failure_reports_both_ranks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_popen = subprocess.Popen
    started_workers: list[subprocess.Popen[bytes]] = []

    def spawn_failed_worker(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        rank = len(started_workers)
        worker = real_popen(
            [sys.executable, "-c", f'print("rank {rank} root error"); raise SystemExit({rank + 1})'],
            env=kwargs["env"],
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
        )
        started_workers.append(worker)
        return worker

    monkeypatch.setattr(subprocess, "Popen", spawn_failed_worker)

    with pytest.raises(pytest.fail.Exception) as error:
        _run_checkpoint_workers(tmp_path / "failure-init", timeout_seconds=5)

    message = str(error.value)
    assert "rank 0 exited with code 1" in message
    assert "rank 1 exited with code 2" in message
    assert "rank 0 root error" in message
    assert "rank 1 root error" in message


@pytest.mark.skipif(
    torch.cuda.device_count() < _WORLD_SIZE or not gsplat.has_reloc(),
    reason="Requires two GPUs and a gsplat build with relocation support",
)
def test_two_rank_visibility_checkpoint_and_taminggs_allocation_contract() -> None:
    output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    _run_checkpoint_workers(output_dir / "mcmc-ddp-init")


def test_two_rank_prepared_visibility_tracks_optimizer_steps(tmp_path: Path) -> None:
    _run_checkpoint_workers(tmp_path / "prepared-visibility-init", worker_flag="--prepared-visibility-worker")


if __name__ == "__main__" and len(sys.argv) == 4:
    if sys.argv[1] == "--checkpoint-worker":
        _checkpoint_worker(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "--prepared-visibility-worker":
        _prepared_visibility_worker(int(sys.argv[2]), sys.argv[3])
