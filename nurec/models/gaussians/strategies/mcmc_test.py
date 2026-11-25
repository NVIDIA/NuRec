# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import MagicMock, patch

import apex
import gsplat
import pytest
import torch

from omegaconf import DictConfig
from pydantic import TypeAdapter

from nurec.config.model import MCMCStrategyConfig, TamingGSMCMCStrategyConfig
from nurec.models.gaussians.gaussians_model import BaseGaussianModel, RigidGaussianModel
from nurec.models.gaussians.strategies.base import BaseGaussianStrategy
from nurec.models.gaussians.strategies.gsplat_mcmc_adapter import add_gaussians as gsplat_add_gaussians
from nurec.models.gaussians.strategies.mcmc import (
    BaseMCMCStrategy,
    MCMCStrategy,
    TamingGSMCMCStrategy,
    _finish_taminggs_addition_scores,
)
from nurec.models.gaussians.strategies.selected_gaussian_nodes import SelectedGaussianNodes
from nurec.models.gaussians.strategies.test_utils import make_trainer_cfg
from nurec.models.nn_extensions import TypedModuleDict
from nurec.utils.optim import SelectiveAdam


def _make_concrete_mcmc_strategy(
    config: MCMCStrategyConfig | TamingGSMCMCStrategyConfig,
    trainer_config,
    init_from_datasource: bool,
    gaussians_nodes: TypedModuleDict,
) -> Any:
    if isinstance(config, TamingGSMCMCStrategyConfig):
        return TamingGSMCMCStrategy(config, trainer_config, init_from_datasource, gaussians_nodes)
    return MCMCStrategy(config, trainer_config, init_from_datasource, gaussians_nodes)


def make_gaussian_cfg(name: Literal["mcmc", "taminggs_mcmc"] = "mcmc") -> DictConfig:
    return DictConfig(
        {
            "name": name,
            "relocate": {"start_iteration": 1, "end_iteration": 1, "frequency": 1},
            "add": {"start_iteration": 1, "end_iteration": 1, "frequency": 1, "max_n_gaussians": 100},
            "perturb": {"start_iteration": 1, "end_iteration": 30_000, "frequency": 1, "noise_lr": {"default": 1.0}},
            "binom_n_max": 51,
            "opacity_threshold": 0.005,
            "exclude_layer_ids": [],
        }
    )


class _LazyNode(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.positions = torch.nn.UninitializedParameter()

    @property
    def device(self) -> torch.device:
        return self.positions.device

    def get_num_gaussians(self) -> int:
        return self.positions.shape[0]


class _LazyAnchoredNode(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.positions = torch.nn.UninitializedParameter()
        self.register_buffer("mcmc_position_anchor", torch.nn.UninitializedBuffer())


class _MaterializedNode(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.positions = torch.nn.Parameter(torch.zeros((1, 3)))


class _AnchoredNode(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.positions = torch.nn.Parameter(torch.tensor([[2.0, 0.0, 0.0]]))
        self.register_buffer("mcmc_position_anchor", torch.empty((0, 3)))

    def raise_if_publication_failed(self) -> None:
        pass


class _StrategyNode(torch.nn.Module):
    def __init__(self, activated_densities: torch.Tensor, *, publication_failed: bool = False) -> None:
        super().__init__()
        self.activated_densities = activated_densities
        self.positions = torch.nn.Parameter(
            torch.zeros((activated_densities.numel(), 3), device=activated_densities.device)
        )
        self.publication_failed = publication_failed
        self.density_queries = 0

    @property
    def device(self) -> torch.device:
        return self.activated_densities.device

    def get_num_gaussians(self) -> int:
        return self.activated_densities.numel()

    def get_densities(self, preactivation: bool = False) -> torch.Tensor:
        del preactivation
        self.density_queries += 1
        return self.activated_densities

    def raise_if_publication_failed(self) -> None:
        if self.publication_failed:
            raise RuntimeError("Gaussian publication previously failed")


class _TamingGSInspiredProxyScoreNode(torch.nn.Module):
    def __init__(
        self,
        position_gradients: torch.Tensor,
        gradient_second_moments: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
    ) -> None:
        super().__init__()
        self.positions = torch.nn.Parameter(torch.zeros(position_gradients.shape))
        self.positions.grad = position_gradients.clone()
        self._opacities = opacities
        self._scales = scales
        optimizer = torch.optim.Adam([{"params": [self.positions], "name": "model.positions"}])
        optimizer.state[self.positions]["exp_avg_sq"] = gradient_second_moments.clone()
        self.optimizers = [{"optimizer": optimizer}]

    @property
    def device(self) -> torch.device:
        return self.positions.device

    def get_num_gaussians(self) -> int:
        return self.positions.shape[0]

    def get_densities(self, preactivation: bool = False) -> torch.Tensor:
        del preactivation
        return self._opacities

    def get_scales(self, preactivation: bool = False) -> torch.Tensor:
        del preactivation
        return self._scales

    def raise_if_publication_failed(self) -> None:
        pass


class TestMCMCStrategy(unittest.TestCase):
    @staticmethod
    def _make_strategy(
        config: DictConfig | None = None,
        nodes: TypedModuleDict | None = None,
    ) -> Any:
        return _make_concrete_mcmc_strategy(
            config=TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(
                config or make_gaussian_cfg()
            ),
            trainer_config=make_trainer_cfg(),
            init_from_datasource=False,
            gaussians_nodes=nodes or TypedModuleDict(),
        )

    def test_concrete_strategy_registration_uses_sibling_classes(self) -> None:
        self.assertIs(BaseGaussianStrategy.GAUSSIAN_STRATEGY_VARIANTS["mcmc"], MCMCStrategy)
        self.assertIs(BaseGaussianStrategy.GAUSSIAN_STRATEGY_VARIANTS["taminggs_mcmc"], TamingGSMCMCStrategy)
        self.assertTrue(issubclass(MCMCStrategy, BaseMCMCStrategy))
        self.assertTrue(issubclass(TamingGSMCMCStrategy, BaseMCMCStrategy))
        self.assertFalse(issubclass(TamingGSMCMCStrategy, MCMCStrategy))
        with self.assertRaises(TypeError):
            cast(Any, BaseMCMCStrategy)(make_gaussian_cfg(), make_trainer_cfg(), False, TypedModuleDict())

    def test_direct_topology_entries_reject_failed_publication_before_noop_paths(self):
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy()
        strategy.config.add.max_n_gaussians = 4
        nodes = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(4), publication_failed=True)}))

        for operation in (strategy.relocate_gaussians, strategy.add_new_gaussians):
            with (
                self.subTest(operation=operation.__name__),
                patch("nurec.models.gaussians.strategies.mcmc.get_cuda_generator") as get_rng,
                patch("nurec.models.gaussians.strategies.mcmc.gsplat_relocate_gaussians") as relocate,
                patch("nurec.models.gaussians.strategies.mcmc.gsplat_add_gaussians") as add,
                self.assertRaisesRegex(RuntimeError, "publication previously failed"),
            ):
                operation(nodes)

            get_rng.assert_not_called()
            relocate.assert_not_called()
            add.assert_not_called()

    def test_track_quota_requires_same_track_layer(self) -> None:
        config = make_gaussian_cfg()
        config.relocate.min_gaussians_per_track = {"vehicle_lights": 16}

        with self.assertRaisesRegex(ValueError, "must also be listed in same_track_layer_ids"):
            self._make_strategy(config)

    def test_cuboid_grouping_requires_single_gpu_and_independent_cap(self) -> None:
        """Reject distributed grouped topology and incompatible or uncapped barrier policies."""
        config = make_gaussian_cfg()
        config.cuboid_grouped_layer_ids = ["barriers"]
        with self.assertRaisesRegex(ValueError, "independent growth caps"):
            self._make_strategy(config)
        config.independent_layer_max_n_gaussians = {"barriers": 256000}
        with self.assertRaisesRegex(ValueError, "requires one GPU"):
            _make_concrete_mcmc_strategy(
                TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(config),
                make_trainer_cfg().model_copy(update={"world_size": 2}),
                False,
                TypedModuleDict(),
            )
        config.add.fixed_count_layer_ids = ["barriers"]
        with self.assertRaisesRegex(ValueError, "cannot combine"):
            self._make_strategy(config)

    def test_taminggs_requires_track_local_layers_to_have_fixed_counts(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.relocate.same_track_layer_ids = ["vehicle_lights"]
        config.add.taminggs = {}

        with self.assertRaisesRegex(ValueError, "same-track relocation layers to have fixed counts"):
            self._make_strategy(config)

        config.add.fixed_count_layer_ids = ["vehicle_lights"]
        with patch("gsplat.has_reloc", return_value=True):
            self._make_strategy(config)

    def test_track_quota_is_adjusted_from_global_to_local_count(self) -> None:
        config = make_gaussian_cfg()
        config.relocate.same_track_layer_ids = ["vehicle_lights"]
        config.relocate.min_gaussians_per_track = {"vehicle_lights": 16}
        trainer_config = make_trainer_cfg().model_copy(update={"world_size": 2})
        local_quotas = []

        for rank in range(2):
            with (
                patch("nurec.utils.trainer.rank_zero_only.rank", rank),
                patch("gsplat.has_reloc", return_value=True),
            ):
                strategy = _make_concrete_mcmc_strategy(
                    TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(config),
                    trainer_config,
                    False,
                    TypedModuleDict(),
                )
            local_quotas.append(strategy.config.relocate.min_gaussians_per_track["vehicle_lights"])

        self.assertEqual(local_quotas, [8, 8])
        self.assertEqual(sum(local_quotas), 16)

    def test_taminggs_allows_native_growth_for_independent_same_track_layer(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.relocate.same_track_layer_ids = ["vehicle_lights"]
        config.independent_layer_max_n_gaussians = {"vehicle_lights": 256}
        config.add.taminggs = {}
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"vehicle_lights": _StrategyNode(torch.ones(20))}))
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)
        strategy.config.print_stats = False
        with (
            patch.object(strategy, "_relocate_same_track_layer", return_value=2) as relocate,
            patch.object(strategy, "_relocate_pool", return_value=0) as shared_relocate,
            patch.object(strategy, "_add_to_pool") as add,
        ):
            strategy.relocate_gaussians(nodes)
            strategy.add_new_gaussians(nodes, global_step=2)
        relocate.assert_called_once_with("vehicle_lights", nodes["vehicle_lights"])
        shared_relocate.assert_called_once()
        self.assertEqual(shared_relocate.call_args.args[0].layer_ids, [])
        self.assertEqual(add.call_count, 2)
        self.assertEqual(add.call_args_list[0].args[0].layer_ids, [])
        self.assertTrue(add.call_args_list[0].kwargs["use_taminggs"])
        self.assertEqual(add.call_args.args[0].layer_ids, ["vehicle_lights"])
        self.assertEqual(add.call_args.args[1], 256)
        self.assertEqual(add.call_args.kwargs, {})

    def test_track_quota_adjustment_does_not_mutate_input_config(self) -> None:
        config = make_gaussian_cfg()
        config.relocate.same_track_layer_ids = ["vehicle_lights"]
        config.relocate.min_gaussians_per_track = {"vehicle_lights": 16}
        source_config: MCMCStrategyConfig | TamingGSMCMCStrategyConfig = TypeAdapter(
            MCMCStrategyConfig | TamingGSMCMCStrategyConfig
        ).validate_python(config)
        trainer_config = make_trainer_cfg().model_copy(update={"world_size": 2})

        with (
            patch("nurec.utils.trainer.rank_zero_only.rank", 0),
            patch("gsplat.has_reloc", return_value=True),
        ):
            first = _make_concrete_mcmc_strategy(source_config, trainer_config, False, TypedModuleDict())
            second = _make_concrete_mcmc_strategy(source_config, trainer_config, False, TypedModuleDict())

        self.assertEqual(source_config.relocate.min_gaussians_per_track, {"vehicle_lights": 16})
        self.assertEqual(first.config.relocate.min_gaussians_per_track, {"vehicle_lights": 8})
        self.assertEqual(second.config.relocate.min_gaussians_per_track, {"vehicle_lights": 8})

    def test_track_quota_keeps_one_per_populated_rank_when_world_size_exceeds_target(self) -> None:
        config = make_gaussian_cfg()
        config.relocate.same_track_layer_ids = ["vehicle_lights"]
        config.relocate.min_gaussians_per_track = {"vehicle_lights": 16}
        world_size = 32
        trainer_config = make_trainer_cfg().model_copy(update={"world_size": world_size})
        local_quotas = []

        for rank in range(world_size):
            with (
                patch("nurec.utils.trainer.rank_zero_only.rank", rank),
                patch("gsplat.has_reloc", return_value=True),
            ):
                strategy = _make_concrete_mcmc_strategy(
                    TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(config),
                    trainer_config,
                    False,
                    TypedModuleDict(),
                )
            local_quota = strategy.config.relocate.min_gaussians_per_track["vehicle_lights"]
            local_quotas.append(local_quota)
            protected = strategy._protect_track_survivors(
                densities=torch.tensor([0.0]),
                track_mask=torch.tensor([True]),
                dead_mask=torch.tensor([True]),
                minimum_survivors=local_quota,
            )
            self.assertFalse(bool(protected.item()))

        self.assertEqual(local_quotas, [1] * world_size)

    def test_same_track_layer_must_be_rigid(self) -> None:
        config = make_gaussian_cfg()
        config.relocate.same_track_layer_ids = ["background"]
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config)

        with self.assertRaisesRegex(TypeError, "must be a RigidGaussianModel"):
            strategy._relocate_same_track_layer("background", cast(Any, _LazyNode()))

    @unittest.skipUnless(torch.cuda.is_available(), "same-track relocation requires CUDA")
    def test_same_track_relocation_batches_groups_into_one_owner_commit(self) -> None:
        device = torch.device("cuda")
        config = make_gaussian_cfg()
        config.relocate.same_track_layer_ids = ["vehicle_lights"]
        config.relocate.min_gaussians_per_track = {"vehicle_lights": 1}
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config)

        node: Any = RigidGaussianModel.__new__(RigidGaussianModel)
        torch.nn.Module.__init__(node)
        node.positions = torch.nn.Parameter(torch.zeros((4, 3), device=device))
        node.densities = torch.nn.Parameter(torch.tensor([[-10.0], [0.0], [-10.0], [0.0]], device=device))
        node.density_activation = torch.sigmoid
        node.register_buffer("gaussian_cuboid_ids", torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=device))

        with (
            patch("nurec.models.gaussians.strategies.mcmc.get_cuda_generator", return_value=MagicMock()),
            patch(
                "nurec.models.gaussians.strategies.mcmc.gsplat_relocate_gaussians_grouped",
                return_value=2,
            ) as relocate,
        ):
            relocated = strategy._relocate_same_track_layer("vehicle_lights", node)

        self.assertEqual(relocated, 2)
        relocate.assert_called_once()
        groups = relocate.call_args.kwargs["groups"]
        self.assertEqual(len(groups), 2)
        torch.testing.assert_close(groups[0][0], torch.tensor([True, False, False, False], device=device))
        torch.testing.assert_close(groups[0][1], torch.tensor([1], dtype=torch.int64, device=device))
        torch.testing.assert_close(groups[1][0], torch.tensor([False, False, True, False], device=device))
        torch.testing.assert_close(groups[1][1], torch.tensor([3], dtype=torch.int64, device=device))

    def test_track_local_only_relocation_honors_empty_cache_config(self) -> None:
        config = make_gaussian_cfg()
        config.relocate.same_track_layer_ids = ["vehicle_lights"]
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config)
        strategy.config.print_stats = False
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"vehicle_lights": _StrategyNode(torch.ones(1))}))
        selected = SimpleNamespace(layer_ids=[])

        for empty_cache_enabled in (False, True):
            with (
                patch.object(strategy, "_relocate_same_track_layer", return_value=2),
                patch("nurec.models.gaussians.strategies.mcmc.SelectedGaussianNodes", return_value=selected),
                patch.object(torch.cuda, "empty_cache") as empty_cache,
            ):
                strategy._empty_cache_after_parameter_update = empty_cache_enabled
                strategy.relocate_gaussians(nodes)

            self.assertEqual(empty_cache.call_count, int(empty_cache_enabled))

    def test_fixed_count_layers_are_excluded_from_addition(self) -> None:
        config = make_gaussian_cfg()
        config.add.fixed_count_layer_ids = ["vehicle_lights"]
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config)
        node = _StrategyNode(torch.ones(20))
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"vehicle_lights": node}))

        with patch(
            "nurec.models.gaussians.strategies.mcmc.SelectedGaussianNodes", wraps=SelectedGaussianNodes
        ) as factory:
            strategy.add_new_gaussians(nodes)

        factory.assert_called_once_with(nodes, ["vehicle_lights"])
        self.assertEqual(node.get_num_gaussians(), 20)
        self.assertEqual(node.density_queries, 0)

    def test_independent_pool_isolation_and_caps(self) -> None:
        """Isolated layers retain their own caps and respect exclusions and missing inputs."""
        config = make_gaussian_cfg()
        config.independent_layer_max_n_gaussians = {"barriers": 256}
        nodes: TypedModuleDict = TypedModuleDict(
            cast(Any, {name: _StrategyNode(torch.ones(20)) for name in ("background", "barriers", "road")})
        )
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)
        pools = list(strategy._population_pools(nodes, ["road"]))
        self.assertEqual([(pool.layer_ids, cap) for pool, cap in pools], [(["background"], 100), (["barriers"], 256)])
        strategy.config.exclude_layer_ids = ["road"]
        with patch.object(strategy, "_add_to_pool") as add:
            strategy.add_new_gaussians(nodes)
        self.assertEqual(
            [(args[0].layer_ids, args[1]) for args, _ in add.call_args_list],
            [(["background"], 100), (["barriers"], 256)],
        )
        with patch.object(strategy, "_relocate_pool", return_value=0) as relocate:
            strategy.relocate_gaussians(nodes)
        self.assertEqual([args[0].layer_ids for args, _ in relocate.call_args_list], [["background"], ["barriers"]])
        # Respect frozen/excluded independent layers and tolerate empty detection
        # inputs whose barrier node was skipped by native initialization.
        self.assertEqual(len(list(strategy._population_pools(nodes, ["barriers"]))), 1)
        del nodes["barriers"]
        self.assertEqual(len(list(strategy._population_pools(nodes, []))), 1)

    def test_taminggs_shared_anchor_and_scores_ignore_independent_populations(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update({"start_iteration": 0, "end_iteration": 10, "frequency": 2, "max_n_gaussians": 10})
        config.add.taminggs = {}
        config.add.fixed_count_layer_ids = ["lights"]
        config.exclude_layer_ids = ["road"]
        config.independent_layer_max_n_gaussians = {"barriers": 100}
        expected_weights = None
        for independent_count in (None, 1, 80, -1):
            shared = _TamingGSInspiredProxyScoreNode(
                torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                torch.ones(2, 3),
                torch.tensor([0.2, 0.4]),
                torch.ones(2, 3),
            )
            owners: dict[str, Any] = {
                "background": shared,
                "lights": _StrategyNode(torch.ones(3)),
                "road": _StrategyNode(torch.ones(7)),
            }
            if independent_count is not None:
                owners["barriers"] = (
                    _LazyNode() if independent_count < 0 else _StrategyNode(torch.ones(independent_count))
                )
            nodes: TypedModuleDict = TypedModuleDict(owners)
            with patch("gsplat.has_reloc", return_value=True):
                strategy = self._make_strategy(config, nodes)
            strategy.maybe_initialize_buffers(nodes)
            selected = strategy._addition_selected_nodes(nodes)
            self.assertEqual(selected.layer_ids, ["background"])
            self.assertEqual(int(strategy._taminggs_initial_count), 2)
            weights = strategy._taminggs_source_weights(selected)
            self.assertEqual(set(strategy._taminggs_local_score_statistics(nodes, weights)), {"background"})
            if expected_weights is None:
                expected_weights = weights["background"]
            torch.testing.assert_close(weights["background"], expected_weights)

    def test_taminggs_coschedule_keeps_independent_dispatch_and_carried_scores_separate(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update({"start_iteration": 0, "end_iteration": 4, "frequency": 2, "max_n_gaussians": 42})
        config.add.taminggs = {}
        config.relocate.update({"start_iteration": 0, "end_iteration": 4, "frequency": 2})
        config.independent_layer_max_n_gaussians = {"barriers": 41}
        nodes: TypedModuleDict = TypedModuleDict(
            cast(Any, {name: _StrategyNode(torch.ones(40)) for name in ("background", "barriers")})
        )
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)
        strategy.maybe_initialize_buffers(nodes)
        before = torch.arange(40, dtype=torch.float32)

        def relocate(selected: SelectedGaussianNodes, **kwargs: Any) -> int:
            if selected.layer_ids == ["background"]:
                self.assertEqual(set(kwargs["taminggs_source_weights"]), {"background"})
                kwargs["reindex_values_by_owner"]["background"] = before.flip(0)
            else:
                self.assertEqual(selected.layer_ids, ["barriers"])
                self.assertEqual(kwargs, {})
            return 0

        with (
            patch.object(strategy, "_taminggs_source_weights", return_value={"background": before}) as scores,
            patch.object(strategy, "_relocate_pool", side_effect=relocate) as relocation,
            patch.object(strategy, "_add_to_pool") as addition,
            patch.object(strategy, "perturb_gaussians"),
        ):
            strategy._update_step_train_batch_end_impl(0, 2, cast(Any, None), None, nodes)
        self.assertEqual(relocation.call_count, 2)
        self.assertEqual([call.args[0].layer_ids for call in scores.call_args_list], [["background"], ["background"]])
        shared_addition, independent_addition = addition.call_args_list
        self.assertEqual(shared_addition.args[1], 42)
        self.assertTrue(shared_addition.kwargs["use_taminggs"])
        torch.testing.assert_close(shared_addition.kwargs["source_weights_by_owner"]["background"], before.flip(0))
        self.assertEqual(independent_addition.args[0].layer_ids, ["barriers"])
        self.assertEqual(independent_addition.args[1], 41)
        self.assertEqual(independent_addition.kwargs, {})

    def test_taminggs_quadratic_schedule_reaches_exact_budget_at_training_end(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update(
            {
                "start_iteration": 2,
                "end_iteration": 100,
                "frequency": 2,
                "max_n_gaussians": 200,
                "taminggs": {},
            }
        )
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(100))}))
        trainer_config = make_trainer_cfg().model_copy(update={"max_steps": 10})
        with patch("gsplat.has_reloc", return_value=True):
            strategy = _make_concrete_mcmc_strategy(
                TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(config),
                trainer_config,
                False,
                nodes,
            )
        strategy.maybe_initialize_buffers(nodes)

        self.assertEqual(strategy._taminggs_schedule_events(), (4, 10, 4))
        self.assertEqual(
            [strategy._taminggs_global_target_count(step) for step in (4, 6, 8, 10)],
            [144, 175, 194, 200],
        )
        self.assertEqual(int(strategy._taminggs_initial_count.item()), 100)
        torch.testing.assert_close(strategy._taminggs_initial_counts_by_rank, torch.tensor([100]))
        for key in (
            "_taminggs_initial_count",
            "_taminggs_initial_counts_by_rank",
        ):
            self.assertIn(key, strategy.state_dict())

    def test_taminggs_initial_population_at_or_above_budget_preserves_rows(self) -> None:
        for initial_count in (20, 21):
            with self.subTest(initial_count=initial_count):
                config = make_gaussian_cfg("taminggs_mcmc")
                config.add.update(
                    {
                        "start_iteration": 0,
                        "end_iteration": 10,
                        "frequency": 2,
                        "max_n_gaussians": 20,
                        "taminggs": {"visibility_weight": 0.0},
                    }
                )
                nodes: TypedModuleDict = TypedModuleDict(
                    cast(Any, {"background": _StrategyNode(torch.ones(initial_count))})
                )
                positions = nodes["background"].positions
                strategy = self._make_strategy(config, nodes)
                with patch("nurec.models.gaussians.strategies.mcmc.log.warning") as warning:
                    strategy.maybe_initialize_buffers(nodes)
                    strategy.maybe_initialize_buffers(nodes)
                self.assertEqual(warning.call_count, int(initial_count > 20))
                self.assertIs(nodes["background"].positions, positions)
                self.assertEqual(nodes["background"].get_num_gaussians(), initial_count)
                self.assertEqual(strategy._max_n_gaussians, 20)
                for step in (2, 4, 6, 8):
                    self.assertEqual(strategy._taminggs_global_target_count(step), initial_count)
                    self.assertEqual(
                        strategy._taminggs_local_growth_quota(
                            global_step=step,
                            local_count=initial_count,
                            local_score_statistics={"background": (1.0, 1)},
                        ),
                        0,
                    )

                # Resume must retain the original ceiling, even when the
                # currently materialized population is smaller.
                resumed_nodes: TypedModuleDict = TypedModuleDict(
                    cast(Any, {"background": _StrategyNode(torch.ones(initial_count - 1))})
                )
                resumed = self._make_strategy(config, resumed_nodes)
                resumed.load_state_dict(strategy.state_dict(), strict=True)
                resumed.maybe_initialize_buffers(resumed_nodes)
                self.assertEqual(resumed._taminggs_global_target_count(8), initial_count)
                torch.testing.assert_close(resumed._taminggs_initial_counts_by_rank, torch.tensor([initial_count]))
                self.assertEqual(
                    resumed._taminggs_local_growth_quota(
                        global_step=8,
                        local_count=initial_count - 1,
                        local_score_statistics={"background": (1.0, 1)},
                    ),
                    1,
                )

    def test_taminggs_short_run_keeps_initial_population_and_warns_once(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update(
            {
                "start_iteration": 450,
                "end_iteration": 27000,
                "frequency": 450,
                "max_n_gaussians": 20,
                "taminggs": {"visibility_weight": 0.0},
            }
        )
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(10))}))
        strategy = self._make_strategy(config, nodes)
        strategy._training_end_step = 100
        with self.assertLogs("nurec.models.gaussians.strategies.mcmc", level="WARNING") as logs:
            strategy.maybe_initialize_buffers(nodes)
            self.assertEqual(strategy._taminggs_schedule_events(), (900, 0, 0))
            strategy.maybe_initialize_buffers(nodes)
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(int(strategy._taminggs_initial_count.item()), 10)
        self.assertTrue(all(not strategy._is_topology_mutation_due(step, strategy.config.add) for step in range(101)))
        self.assertEqual(nodes["background"].get_num_gaussians(), 10)

    def test_taminggs_checkpoint_restores_original_population_anchor(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update(
            {
                "start_iteration": 0,
                "end_iteration": 10,
                "frequency": 2,
                "max_n_gaussians": 20,
                "taminggs": {"visibility_weight": 0.0},
            }
        )
        original_nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(10))}))
        resumed_nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(14))}))
        with patch("gsplat.has_reloc", return_value=True):
            original = self._make_strategy(config, original_nodes)
            resumed = self._make_strategy(config, resumed_nodes)
        original.maybe_initialize_buffers(original_nodes)
        resumed.maybe_initialize_buffers(resumed_nodes)

        resumed.load_state_dict(original.state_dict(), strict=True)

        self.assertEqual(int(resumed._taminggs_initial_count.item()), 10)
        torch.testing.assert_close(resumed._taminggs_initial_counts_by_rank, torch.tensor([10]))
        self.assertEqual(resumed._taminggs_global_target_count(8), 20)

    def test_taminggs_resume_changed_schedule_waits_for_target_to_catch_up(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update(
            {
                "start_iteration": 0,
                "end_iteration": 20,
                "frequency": 2,
                "max_n_gaussians": 20,
                "taminggs": {"visibility_weight": 0.0},
            }
        )
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(10))}))
        original = self._make_strategy(config, nodes)
        original.maybe_initialize_buffers(nodes)
        for start, frequency, max_steps in ((0, 2, 18), (4, 3, 18)):
            with self.subTest(start=start, frequency=frequency, max_steps=max_steps):
                revised = config.copy()
                revised.add.start_iteration = start
                revised.add.frequency = frequency
                resumed = self._make_strategy(revised, nodes)
                resumed._training_end_step = max_steps
                resumed.load_state_dict(original.state_dict(), strict=True)
                first, last, _ = resumed._taminggs_schedule_events()
                with self.assertLogs("nurec.models.gaussians.strategies.mcmc", level="WARNING"):
                    self.assertEqual(
                        resumed._taminggs_local_growth_quota(
                            global_step=first, local_count=18, local_score_statistics={"background": (1.0, 1)}
                        ),
                        0,
                    )
                self.assertEqual(
                    resumed._taminggs_local_growth_quota(
                        global_step=last, local_count=18, local_score_statistics={"background": (1.0, 1)}
                    ),
                    2,
                )

    def test_taminggs_schedule_preserves_exclusive_boundaries_and_population_curve(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update(
            {
                "start_iteration": 450,
                "end_iteration": 27_000,
                "frequency": 450,
                "max_n_gaussians": 1_500_000,
                "taminggs": {},
            }
        )
        config.relocate.update({"start_iteration": 450, "end_iteration": 22_500, "frequency": 450})
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(100))}))
        trainer_config = make_trainer_cfg().model_copy(update={"max_steps": 27_000})
        with patch("gsplat.has_reloc", return_value=True):
            strategy = _make_concrete_mcmc_strategy(
                TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(config),
                trainer_config,
                False,
                nodes,
            )
        strategy.maybe_initialize_buffers(nodes)

        self.assertEqual(strategy._taminggs_schedule_events(), (900, 26_550, 58))
        addition_steps = [
            step for step in range(27_001) if strategy._is_topology_mutation_due(step, strategy.config.add)
        ]
        relocation_steps = [
            step for step in range(27_001) if strategy._is_topology_mutation_due(step, strategy.config.relocate)
        ]
        self.assertEqual(addition_steps, list(range(900, 27_000, 450)))
        self.assertEqual(relocation_steps, list(range(900, 22_500, 450)))
        self.assertEqual(
            [strategy._taminggs_global_target_count(step) for step in (900, 13_500, 26_550)],
            [51_375, 1_125_025, 1_500_000],
        )
        for step in (450, 899, 22_501, 27_000):
            with self.subTest(step=step), self.assertRaises(ValueError):
                strategy._taminggs_global_target_count(step)
        self.assertEqual([step for step in range(27_001) if strategy.may_mutate_topology_at(step)], addition_steps)

    def test_taminggs_score_combines_all_five_signals_without_fixed_or_excluded_rows(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update({"start_iteration": 0, "end_iteration": 10, "frequency": 2})
        config.exclude_layer_ids = ["road", "cones"]
        config.add.fixed_count_layer_ids = ["vehicle_lights"]
        config.add.taminggs = {
            "current_gradient_weight": 50.0,
            "gradient_ema_weight": 10.0,
            "opacity_weight": 100.0,
            "scale_weight": 25.0,
            "visibility_weight": 5.0,
            "normalization_clip": 10.0,
            "minimum_source_weight": 1.0e-6,
        }
        first = _TamingGSInspiredProxyScoreNode(
            torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 3.0, 0.0], [4.0] * 3]),
            torch.tensor([0.1, 0.2]),
            torch.tensor([[3.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        )
        second = _TamingGSInspiredProxyScoreNode(
            torch.tensor([[4.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
            torch.tensor([[9.0] * 3, [16.0] * 3]),
            torch.tensor([0.4, 0.8]),
            torch.tensor([[2.0, 2.0, 2.0], [2.0, 1.0, 1.0]]),
        )
        nodes: TypedModuleDict = TypedModuleDict(
            cast(
                Any,
                {
                    "background": first,
                    "dynamic": second,
                    "vehicle_lights": _StrategyNode(torch.ones(3)),
                    "road": _StrategyNode(torch.ones(7)),
                    "cones": _StrategyNode(torch.ones(5)),
                },
            )
        )
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)
        strategy.maybe_initialize_buffers(nodes)
        strategy.invisible_steps["background"].copy_(torch.tensor([0, 1]))
        strategy.invisible_steps["dynamic"].copy_(torch.tensor([3, 7]))

        weights = strategy._taminggs_source_weights(strategy._addition_selected_nodes(nodes))

        self.assertEqual(set(weights), {"background", "dynamic"})
        self.assertEqual(int(strategy._taminggs_initial_count), 4)
        torch.testing.assert_close(weights["background"], torch.tensor([137.5, 182.5]) + 1.0e-6)
        torch.testing.assert_close(weights["dynamic"], torch.tensor([420.0, 947.5]) + 1.0e-6)

    def test_taminggs_validation_setup_defers_lazy_population_until_checkpoint_load(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update(
            {
                "start_iteration": 0,
                "end_iteration": 10,
                "frequency": 2,
                "max_n_gaussians": 20,
                "taminggs": {},
            }
        )
        config.perturb.max_position_offset_m = {"background": 0.5}
        training_node = _StrategyNode(torch.ones(10))
        training_node.register_buffer("mcmc_position_anchor", torch.zeros(10, 3))
        training_nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": training_node}))
        validation_node = _LazyNode()
        validation_node.register_buffer("mcmc_position_anchor", torch.nn.UninitializedBuffer())
        validation_nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": validation_node}))
        with patch("gsplat.has_reloc", return_value=True):
            trained = self._make_strategy(config, training_nodes)
            validating = self._make_strategy(config, validation_nodes)
        trained.maybe_initialize_buffers(training_nodes)

        with patch("nurec.models.gaussians.strategies.mcmc.SelectedGaussianNodes") as selection:
            validating.maybe_initialize_buffers(validation_nodes)

        selection.assert_not_called()
        self.assertEqual(int(validating._taminggs_initial_count.item()), -1)
        torch.testing.assert_close(validating._taminggs_initial_counts_by_rank, torch.tensor([-1]))

        # Lightning materializes the Gaussian parameters and strategy buffers
        # together when it restores the validation checkpoint after setup.
        validation_node.load_state_dict(
            {"positions": torch.full((14, 3), 2.0), "mcmc_position_anchor": torch.zeros(14, 3)},
            strict=True,
            assign=True,
        )
        validating.load_state_dict(trained.state_dict(), strict=True)
        validating.maybe_initialize_buffers(validation_nodes)

        self.assertEqual(validation_node.get_num_gaussians(), 14)
        self.assertEqual(int(validating._taminggs_initial_count.item()), 10)
        torch.testing.assert_close(validating._taminggs_initial_counts_by_rank, torch.tensor([10]))
        validating._clamp_position_offset(cast(Any, validation_node), max_offset_m=0.5)
        torch.testing.assert_close(torch.linalg.vector_norm(validation_node.positions, dim=-1), torch.full((14,), 0.5))

    def test_ordinary_mcmc_keeps_lazy_setup_behavior(self) -> None:
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _LazyNode()}))
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(make_gaussian_cfg(), nodes)

        with patch("nurec.models.gaussians.strategies.mcmc.SelectedGaussianNodes") as selection:
            strategy.maybe_initialize_buffers(nodes)

        selection.assert_not_called()
        self.assertFalse(hasattr(strategy, "_taminggs_initial_count"))

    def test_initialization_collective_device_follows_backend_for_cpu_checkpoint_nodes(self) -> None:
        tensor_factory = torch.tensor
        for backend in (torch.distributed.Backend.GLOO, torch.distributed.Backend.NCCL):
            for node in (None, _LazyNode(), _StrategyNode(torch.ones(2))):
                with self.subTest(backend=backend, node=type(node).__name__):
                    nodes: TypedModuleDict = TypedModuleDict(cast(Any, {} if node is None else {"background": node}))
                    with patch("gsplat.has_reloc", return_value=True):
                        strategy = _make_concrete_mcmc_strategy(
                            TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(
                                make_gaussian_cfg()
                            ),
                            make_trainer_cfg().model_copy(update={"world_size": 2}),
                            False,
                            nodes,
                        )
                    requested_devices: list[torch.device] = []

                    def allocate_tensor(data: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
                        requested_devices.append(torch.device(kwargs["device"]))
                        return tensor_factory(data, *args, **{**kwargs, "device": "cpu"})

                    with (
                        patch("torch.distributed.is_initialized", return_value=True),
                        patch("torch.distributed.get_world_size", return_value=2),
                        patch("torch.distributed.get_backend", return_value=backend),
                        patch("torch.cuda.current_device", return_value=3) as current_device,
                        patch("torch.tensor", side_effect=allocate_tensor),
                        patch(
                            "torch.distributed.nn.functional.all_reduce", side_effect=lambda value, **kwargs: value * 2
                        ),
                    ):
                        strategy.maybe_initialize_buffers(nodes)

                    expected_device = (
                        torch.device("cuda", 3) if backend == torch.distributed.Backend.NCCL else torch.device("cpu")
                    )
                    self.assertEqual(requested_devices, [expected_device])
                    self.assertEqual(current_device.call_count, int(backend == torch.distributed.Backend.NCCL))
                    self.assertEqual(strategy.binoms.device.type, "cpu")

    def test_taminggs_rejects_checkpoint_without_allocation_state_even_when_non_strict(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.taminggs = {}
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config)

        with self.assertRaisesRegex(RuntimeError, "lacks its population-allocation state"):
            strategy.load_state_dict({}, strict=False)

    def test_taminggs_rejects_training_resume_with_different_world_size(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.taminggs = {}
        with patch("gsplat.has_reloc", return_value=True):
            single_rank = self._make_strategy(config)
            two_rank = _make_concrete_mcmc_strategy(
                TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(config),
                make_trainer_cfg().model_copy(update={"world_size": 2}),
                False,
                TypedModuleDict(),
            )

        two_rank.load_state_dict(single_rank.state_dict(), strict=True)
        with self.assertRaisesRegex(RuntimeError, "different world size"):
            two_rank.on_train_step_start()
        single_rank.on_train_step_start()

    def test_taminggs_multi_rank_state_loads_for_single_rank_validation_and_export(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.taminggs = {}
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(16))}))
        for world_size in (2, 8):
            for strict in (True, False):
                for assign in (True, False):
                    with self.subTest(world_size=world_size, strict=strict, assign=assign):
                        with patch("gsplat.has_reloc", return_value=True):
                            saved = _make_concrete_mcmc_strategy(
                                TypeAdapter(TamingGSMCMCStrategyConfig).validate_python(config),
                                make_trainer_cfg().model_copy(update={"world_size": world_size}),
                                False,
                                TypedModuleDict(),
                            )
                            restored = self._make_strategy(config, nodes)
                        saved._ensure_invisible_steps_buffer("background", 16, torch.device("cpu"))
                        saved._taminggs_initial_counts_by_rank.fill_(2)
                        saved._taminggs_initial_count.fill_(2 * world_size)
                        # Match validation's strict and export's non-strict loads;
                        # both production entry points use assign=True.
                        restored.load_state_dict(saved.state_dict(), strict=strict, assign=assign)
                        restored.eval()
                        restored.maybe_initialize_buffers(nodes)
                        torch.testing.assert_close(
                            restored._taminggs_initial_counts_by_rank, saved._taminggs_initial_counts_by_rank
                        )
                        torch.testing.assert_close(restored._taminggs_initial_count, saved._taminggs_initial_count)
                        with self.assertRaisesRegex(RuntimeError, "different world size"):
                            restored.train()
                            restored.on_train_step_start()

    def test_taminggs_rejects_malformed_rank_count_state(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.taminggs = {}
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config)
        for counts in (torch.tensor(1), torch.empty(0), torch.ones(1, 2)):
            with self.subTest(shape=counts.shape):
                state = strategy.state_dict()
                state["_taminggs_initial_counts_by_rank"] = counts
                with self.assertRaisesRegex(RuntimeError, "nonempty vector"):
                    strategy.load_state_dict(state)

    def test_taminggs_global_budget_is_partitioned_by_score_mass_across_ddp_ranks(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update(
            {
                "start_iteration": 0,
                "end_iteration": 10,
                "frequency": 2,
                "max_n_gaussians": 201,
                "taminggs": {},
            }
        )
        trainer_config = make_trainer_cfg().model_copy(update={"world_size": 2})
        local_targets = []

        for rank in range(2):
            with (
                patch("gsplat.has_reloc", return_value=True),
                patch("nurec.utils.trainer.rank_zero_only.rank", rank),
            ):
                strategy = _make_concrete_mcmc_strategy(
                    TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(config),
                    trainer_config,
                    False,
                    TypedModuleDict(),
                )
                strategy._taminggs_initial_count.fill_(101)
                strategy._taminggs_initial_counts_by_rank.copy_(torch.tensor([51, 50]))
                self.assertEqual(strategy.config.add.max_n_gaussians, 101 - rank)

            owner_key = strategy._taminggs_stable_key("background")

            def all_gather(outputs: list[torch.Tensor], _value: torch.Tensor) -> None:
                outputs[0].copy_(torch.tensor([51.0, owner_key, 3.0, 100.0], dtype=torch.float64))
                outputs[1].copy_(torch.tensor([50.0, owner_key, 1.0, 200.0], dtype=torch.float64))

            with (
                patch("torch.distributed.is_initialized", return_value=True),
                patch("torch.distributed.get_world_size", return_value=2),
                patch("torch.distributed.get_rank", return_value=rank),
                patch("torch.distributed.all_gather", side_effect=all_gather),
            ):
                quota = strategy._taminggs_local_growth_quota(
                    global_step=4,
                    local_count=(51, 50)[rank],
                    local_score_statistics={"background": ((3.0, 1.0)[rank], (100, 200)[rank])},
                )
            local_targets.append((51, 50)[rank] + quota)

        self.assertEqual(local_targets, [126, 75])
        self.assertEqual(sum(local_targets), 201)

    def test_taminggs_hamilton_ties_use_stable_owner_key(self) -> None:
        quotas = TamingGSMCMCStrategy._taminggs_hamilton_quotas(1, {"z-owner": 1.0, "a-owner": 1.0})
        self.assertEqual(quotas, {"z-owner": 0, "a-owner": 1})

    def test_taminggs_shard_tie_follows_content_key_across_rank_permutation(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update(
            {
                "start_iteration": 0,
                "end_iteration": 10,
                "frequency": 2,
                "max_n_gaussians": 3,
                "taminggs": {},
            }
        )
        trainer_config = make_trainer_cfg().model_copy(update={"world_size": 2})
        assigned_keys = []
        for keys_by_rank in ((100, 200), (200, 100)):
            quotas = []
            for rank in range(2):
                with patch("gsplat.has_reloc", return_value=True):
                    strategy = _make_concrete_mcmc_strategy(
                        TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(config),
                        trainer_config,
                        False,
                        TypedModuleDict(),
                    )
                strategy._taminggs_initial_count.fill_(2)
                owner_key = strategy._taminggs_stable_key("background")

                def all_gather(outputs: list[torch.Tensor], _value: torch.Tensor) -> None:
                    for output, shard_key in zip(outputs, keys_by_rank):
                        output.copy_(torch.tensor([1.0, owner_key, 1.0, shard_key], dtype=torch.float64))

                with (
                    patch("torch.distributed.is_initialized", return_value=True),
                    patch("torch.distributed.get_world_size", return_value=2),
                    patch("torch.distributed.get_rank", return_value=rank),
                    patch("torch.distributed.all_gather", side_effect=all_gather),
                ):
                    quotas.append(
                        strategy._taminggs_local_growth_quota(
                            global_step=4,
                            local_count=1,
                            local_score_statistics={"background": (1.0, keys_by_rank[rank])},
                        )
                    )
            self.assertEqual(sum(quotas), 1)
            assigned_keys.append(keys_by_rank[quotas.index(1)])

        self.assertEqual(assigned_keys, [100, 100])

    def test_taminggs_duplicate_positive_shard_key_fails_closed(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update(
            {
                "start_iteration": 0,
                "end_iteration": 10,
                "frequency": 2,
                "max_n_gaussians": 3,
                "taminggs": {},
            }
        )
        trainer_config = make_trainer_cfg().model_copy(update={"world_size": 2})
        with patch("gsplat.has_reloc", return_value=True):
            strategy = _make_concrete_mcmc_strategy(
                TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(config),
                trainer_config,
                False,
                TypedModuleDict(),
            )
        strategy._taminggs_initial_count.fill_(2)
        owner_key = strategy._taminggs_stable_key("background")

        def all_gather(outputs: list[torch.Tensor], _value: torch.Tensor) -> None:
            for output in outputs:
                output.copy_(torch.tensor([1.0, owner_key, 1.0, 100.0], dtype=torch.float64))

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.all_gather", side_effect=all_gather),
            self.assertRaisesRegex(RuntimeError, "duplicate positive-mass shard keys"),
        ):
            strategy._taminggs_local_growth_quota(
                global_step=4,
                local_count=1,
                local_score_statistics={"background": (1.0, 100)},
            )

    def test_taminggs_growth_delta_handles_two_rank_multi_owner_sharding(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update(
            {
                "start_iteration": 0,
                "end_iteration": 10,
                "frequency": 2,
                "max_n_gaussians": 10,
                "taminggs": {},
            }
        )
        trainer_config = make_trainer_cfg().model_copy(update={"world_size": 2})
        local_layouts = ((2, 2, 2), (1, 1, 1))
        final_targets: list[int] = []

        def all_reduce(value: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
            return value.new_tensor(10 if int(value.item()) == 5 else 9)

        gather_calls = 0

        def all_gather(outputs: list[torch.Tensor], _value: torch.Tensor) -> None:
            nonlocal gather_calls
            gather_calls += 1
            if outputs[0].numel() == 1:
                outputs[0].fill_(6)
                outputs[1].fill_(3)
            else:
                owner_keys = [strategy._taminggs_stable_key(f"owner_{index}") for index in range(3)]
                outputs[0].copy_(
                    torch.tensor(
                        [6.0, *sum(([key, 2.0, 1000 + 10 * index] for index, key in enumerate(owner_keys)), [])],
                        dtype=torch.float64,
                    )
                )
                outputs[1].copy_(
                    torch.tensor(
                        [3.0, *sum(([key, 1.0, 1001 + 10 * index] for index, key in enumerate(owner_keys)), [])],
                        dtype=torch.float64,
                    )
                )

        for rank, layout in enumerate(local_layouts):
            nodes: TypedModuleDict = TypedModuleDict(
                cast(
                    Any,
                    {
                        f"owner_{owner_index}": _StrategyNode(torch.ones(count))
                        for owner_index, count in enumerate(layout)
                    },
                )
            )
            with (
                patch("gsplat.has_reloc", return_value=True),
                patch("torch.distributed.is_initialized", return_value=True),
                patch("torch.distributed.get_world_size", return_value=2),
                patch("torch.distributed.get_rank", return_value=rank),
                patch("torch.distributed.all_reduce", return_value=None),
                patch("torch.distributed.get_backend", return_value=torch.distributed.Backend.GLOO),
                patch("torch.distributed.nn.functional.all_reduce", side_effect=all_reduce),
                patch("torch.distributed.all_gather", side_effect=all_gather),
            ):
                strategy = _make_concrete_mcmc_strategy(
                    TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(config),
                    trainer_config,
                    False,
                    nodes,
                )
                strategy.maybe_initialize_buffers(nodes)
                quota = strategy._taminggs_local_growth_quota(
                    global_step=4,
                    local_count=sum(layout),
                    local_score_statistics={
                        f"owner_{owner_index}": (float(count), 1000 + 10 * owner_index + rank)
                        for owner_index, count in enumerate(layout)
                    },
                )
                final_targets.append(sum(layout) + quota)

            self.assertEqual(int(strategy._taminggs_initial_count.item()), 9)
            torch.testing.assert_close(strategy._taminggs_initial_counts_by_rank, torch.tensor([6, 3]))

        self.assertEqual(final_targets, [7, 3])
        self.assertEqual(sum(final_targets), 10)
        self.assertEqual(gather_calls, 4)

    def test_taminggs_growth_skips_zero_donor_rank(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update(
            {
                "start_iteration": 0,
                "end_iteration": 10,
                "frequency": 2,
                "max_n_gaussians": 10,
                "taminggs": {},
            }
        )
        trainer_config = make_trainer_cfg().model_copy(update={"world_size": 2})
        final_targets: list[int] = []

        def all_reduce(value: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
            return value.new_tensor(10 if int(value.item()) == 5 else 9)

        def all_gather(outputs: list[torch.Tensor], _value: torch.Tensor) -> None:
            if outputs[0].numel() == 1:
                outputs[0].zero_()
                outputs[1].fill_(9)
            else:
                owner_key = strategy._taminggs_stable_key("background")
                outputs[0].copy_(torch.tensor([0.0, owner_key, 0.0, 100.0], dtype=torch.float64))
                outputs[1].copy_(torch.tensor([9.0, owner_key, 9.0, 200.0], dtype=torch.float64))

        for rank, count in enumerate((0, 9)):
            nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(count))}))
            with (
                patch("gsplat.has_reloc", return_value=True),
                patch("torch.distributed.is_initialized", return_value=True),
                patch("torch.distributed.get_world_size", return_value=2),
                patch("torch.distributed.get_rank", return_value=rank),
                patch("torch.distributed.all_reduce", return_value=None),
                patch("torch.distributed.get_backend", return_value=torch.distributed.Backend.GLOO),
                patch("torch.distributed.nn.functional.all_reduce", side_effect=all_reduce),
                patch("torch.distributed.all_gather", side_effect=all_gather),
            ):
                strategy = _make_concrete_mcmc_strategy(
                    TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(config),
                    trainer_config,
                    False,
                    nodes,
                )
                strategy.maybe_initialize_buffers(nodes)
                quota = strategy._taminggs_local_growth_quota(
                    global_step=4,
                    local_count=count,
                    local_score_statistics={"background": ((0.0, 9.0)[rank], (100, 200)[rank])},
                )
                final_targets.append(count + quota)

        self.assertEqual(final_targets, [0, 10])
        self.assertEqual(sum(final_targets), 10)

    def test_taminggs_initial_budget_excludes_fixed_and_strategy_excluded_layers(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.exclude_layer_ids = ["road"]
        config.add.update(
            {
                "start_iteration": 0,
                "end_iteration": 10,
                "frequency": 2,
                "max_n_gaussians": 20,
                "fixed_count_layer_ids": ["lights"],
                "taminggs": {},
            }
        )
        nodes: TypedModuleDict = TypedModuleDict(
            cast(
                Any,
                {
                    "background": _StrategyNode(torch.ones(10)),
                    "lights": _StrategyNode(torch.ones(3)),
                    "road": _StrategyNode(torch.ones(7)),
                },
            )
        )
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)

        strategy.maybe_initialize_buffers(nodes)

        self.assertEqual(int(strategy._taminggs_initial_count.item()), 10)

    def test_taminggs_rejects_growth_without_an_eligible_donor(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.independent_layer_max_n_gaussians = {"barriers": 100}
        config.add.update(
            {
                "start_iteration": 0,
                "end_iteration": 10,
                "frequency": 2,
                "max_n_gaussians": 20,
                "taminggs": {},
            }
        )
        nodes: TypedModuleDict = TypedModuleDict(
            cast(Any, {"background": _StrategyNode(torch.empty(0)), "barriers": _StrategyNode(torch.ones(40))})
        )
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)

        with self.assertRaisesRegex(ValueError, "at least one eligible Gaussian donor"):
            strategy.maybe_initialize_buffers(nodes)

    def test_taminggs_source_weights_are_pooled_normalized_and_owner_aligned(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.taminggs = {
            "current_gradient_weight": 1.0,
            "gradient_ema_weight": 0.0,
            "opacity_weight": 0.0,
            "scale_weight": 0.0,
            "visibility_weight": 0.0,
            "minimum_source_weight": 0.01,
        }
        first = _TamingGSInspiredProxyScoreNode(
            torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            torch.zeros(2, 3),
            torch.tensor([0.2, 0.3]),
            torch.ones(2, 3),
        )
        second = _TamingGSInspiredProxyScoreNode(
            torch.tensor([[3.0, 0.0, 0.0]]),
            torch.zeros(1, 3),
            torch.tensor([0.4]),
            torch.ones(1, 3),
        )
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"first": first, "second": second}))
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)
        selected = SelectedGaussianNodes(nodes, [])

        weights = strategy._taminggs_source_weights(selected)

        torch.testing.assert_close(weights["first"], torch.tensor([0.51, 1.01]))
        torch.testing.assert_close(weights["second"], torch.tensor([1.51]))

    def test_taminggs_addition_masks_dead_donors_after_minimum_and_rechecks_carried_opacity(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.opacity_threshold = 0.1
        config.add.taminggs = {
            "current_gradient_weight": 0.0,
            "gradient_ema_weight": 0.0,
            "opacity_weight": 1.0,
            "scale_weight": 0.0,
            "visibility_weight": 0.0,
            "minimum_source_weight": 2.0,
        }
        node = _StrategyNode(torch.tensor([0.0, 0.1, 0.2]))
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"owner": node}))
        strategy = self._make_strategy(config, nodes)
        selected = SelectedGaussianNodes(nodes, [])
        weights, masses = strategy._taminggs_addition_weights(selected, None)
        # Median includes the positive but ineligible threshold row; only final weights are masked.
        torch.testing.assert_close(weights["owner"], torch.tensor([0.0, 0.0, 4.0]))
        torch.testing.assert_close(masses["owner"], torch.tensor(4.0))
        node.activated_densities = torch.tensor([0.3, 0.0, 0.0])
        weights, masses = strategy._taminggs_addition_weights(selected, {"owner": torch.tensor([3.0, 5.0, 7.0])})
        torch.testing.assert_close(weights["owner"], torch.tensor([3.0, 0.0, 0.0]))
        torch.testing.assert_close(masses["owner"], torch.tensor(3.0))

    def test_taminggs_all_dead_addition_skips_without_sampling(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.update({"start_iteration": 0, "end_iteration": 4, "frequency": 2, "max_n_gaussians": 4})
        config.add.taminggs = {
            "current_gradient_weight": 0.0,
            "gradient_ema_weight": 0.0,
            "opacity_weight": 1.0,
            "scale_weight": 0.0,
            "visibility_weight": 0.0,
        }
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"owner": _StrategyNode(torch.zeros(2))}))
        strategy = self._make_strategy(config, nodes)
        strategy.maybe_initialize_buffers(nodes)
        with (
            self.assertLogs("nurec.models.gaussians.strategies.mcmc", level="WARNING"),
            patch("nurec.models.gaussians.strategies.mcmc.gsplat_add_gaussians") as add,
        ):
            strategy.add_new_gaussians(nodes, global_step=2)
        add.assert_not_called()
        self.assertEqual(nodes["owner"].get_num_gaussians(), 2)
        for mass in (-1.0, float("nan"), float("inf")):
            with self.subTest(mass=mass), self.assertRaisesRegex(ValueError, "score mass"):
                strategy._taminggs_local_growth_quota(
                    global_step=2, local_count=2, local_score_statistics={"owner": (mass, 1)}
                )

    def test_taminggs_gradient_state_requires_parameter_identity_and_full_shape(self) -> None:
        node = _TamingGSInspiredProxyScoreNode(
            torch.ones(2, 3), torch.full((2, 3), 4.0), torch.ones(2), torch.ones(2, 3)
        )
        strategy = self._make_strategy(make_gaussian_cfg("taminggs_mcmc"))
        decoy = torch.nn.Parameter(torch.zeros_like(node.positions))
        optimizer = torch.optim.Adam(
            [{"params": [decoy], "name": "positions"}, {"params": [node.positions], "name": "renamed"}]
        )
        optimizer.state[decoy]["exp_avg_sq"] = torch.full_like(decoy, 100.0)
        optimizer.state[node.positions]["exp_avg_sq"] = torch.full_like(node.positions, 4.0)
        node.optimizers = [{"optimizer": optimizer}]
        torch.testing.assert_close(strategy._position_gradient_ema(cast(Any, node), "owner"), torch.full((2,), 2.0))
        optimizer.state[node.positions]["exp_avg_sq"] = torch.ones(2, 1)
        with self.assertLogs("nurec.models.gaussians.strategies.mcmc", level="WARNING") as logs:
            for _ in range(2):
                torch.testing.assert_close(strategy._position_gradient_ema(cast(Any, node), "owner"), torch.zeros(2))
        self.assertEqual(len(logs.records), 1)
        optimizer.param_groups.pop()
        with self.assertLogs("nurec.models.gaussians.strategies.mcmc", level="WARNING"):
            torch.testing.assert_close(strategy._position_gradient_ema(cast(Any, node), "owner"), torch.zeros(2))

    def test_taminggs_disabled_gradient_components_are_not_read(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.taminggs = {
            "current_gradient_weight": 0.0,
            "gradient_ema_weight": 0.0,
            "opacity_weight": 1.0,
            "scale_weight": 0.0,
            "visibility_weight": 0.0,
        }
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"owner": _StrategyNode(torch.ones(2))}))
        strategy = self._make_strategy(config, nodes)
        with (
            patch.object(strategy, "_position_gradient", side_effect=AssertionError("disabled")),
            patch.object(strategy, "_position_gradient_ema", side_effect=AssertionError("disabled")),
        ):
            weights = strategy._taminggs_source_weights(SelectedGaussianNodes(nodes, []))
        self.assertTrue(torch.all(weights["owner"] > 0))

    def test_taminggs_normalization_uses_only_positive_rows_for_its_median(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.taminggs = {"normalization_clip": 10.0}
        strategy = self._make_strategy(config)

        normalized = strategy._normalized_taminggs_component(
            torch.tensor([0.0, 0.0, 2.0, 4.0, float("nan"), float("inf"), -1.0])
        )

        # torch.median([2, 4]) == 2, matching the proxy's intended
        # positive-value median.  The former all-row median was zero and fell
        # back to the mean, producing [0, 0, 3.5, 7, ...] instead.
        torch.testing.assert_close(normalized, torch.tensor([0.0, 0.0, 1.0, 2.0, 0.0, 10.0, 0.0]))

    def test_taminggs_normalization_keeps_empty_component_zero(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.taminggs = {}
        strategy = self._make_strategy(config)

        normalized = strategy._normalized_taminggs_component(torch.zeros(4))

        torch.testing.assert_close(normalized, torch.zeros(4))

    def test_taminggs_normalization_all_empty_ranks_use_identical_count_collective(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.taminggs = {}
        strategy = self._make_strategy(config)
        gather_calls = []

        def all_gather(outputs: list[torch.Tensor], value: torch.Tensor) -> None:
            gather_calls.append(value.shape)
            for output in outputs:
                output.zero_()

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_gather", side_effect=all_gather),
        ):
            normalized = strategy._normalized_taminggs_component(torch.empty(0))

        self.assertEqual(gather_calls, [torch.Size([])])
        torch.testing.assert_close(normalized, torch.empty(0))

    def test_taminggs_normalization_is_invariant_to_rank_partition(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.taminggs = {"normalization_clip": 10.0}
        strategy = self._make_strategy(config)

        def normalize_partition(partition: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            normalized_parts = []
            for local in partition:

                def all_gather(outputs: list[torch.Tensor], _value: torch.Tensor) -> None:
                    if outputs[0].ndim == 0:
                        outputs[0].fill_(partition[0].numel())
                        outputs[1].fill_(partition[1].numel())
                        return
                    outputs[0].zero_()
                    outputs[1].zero_()
                    outputs[0][: partition[0].numel(), 0].copy_(partition[0])
                    outputs[1][: partition[1].numel(), 0].copy_(partition[1])

                with (
                    patch("torch.distributed.is_initialized", return_value=True),
                    patch("torch.distributed.get_world_size", return_value=2),
                    patch("torch.distributed.all_gather", side_effect=all_gather),
                ):
                    normalized_parts.append(strategy._normalized_taminggs_component(local))
            return torch.cat(normalized_parts).sort().values

        raw = torch.tensor([float("nan"), float("inf"), -float("inf"), -1.0, 0.0, 2.0, 4.0])
        expected_nonfinite = strategy._normalized_taminggs_component(raw).sort().values
        for split in (0, 1, 5, 7):
            torch.testing.assert_close(normalize_partition((raw[:split], raw[split:])), expected_nonfinite)

        first = normalize_partition((torch.tensor([1.0, 9.0]), torch.tensor([3.0, 6.0])))
        second = normalize_partition((torch.tensor([1.0, 3.0, 6.0]), torch.tensor([9.0])))
        empty_shard = normalize_partition((torch.empty(0), torch.tensor([1.0, 3.0, 6.0, 9.0])))

        expected = torch.tensor([1.0 / 3.0, 1.0, 2.0, 3.0])
        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(second, expected)
        torch.testing.assert_close(empty_shard, expected)

    def test_taminggs_coschedule_reindexes_pre_relocation_gradient_scores(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.relocate.update({"start_iteration": 0, "end_iteration": 10, "frequency": 2})
        config.add.update(
            {
                "start_iteration": 0,
                "end_iteration": 10,
                "frequency": 2,
                "max_n_gaussians": 4,
                "taminggs": {
                    "current_gradient_weight": 1.0,
                    "gradient_ema_weight": 0.0,
                    "opacity_weight": 0.0,
                    "scale_weight": 0.0,
                    "visibility_weight": 0.0,
                    "minimum_source_weight": 0.01,
                },
            }
        )
        node = _TamingGSInspiredProxyScoreNode(
            torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            torch.zeros(2, 3),
            torch.tensor([0.2, 0.3]),
            torch.ones(2, 3),
        )
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": node}))
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)
        strategy.maybe_initialize_buffers(nodes)
        original_parameter = node.positions

        def replace_parameters(
            _nodes: TypedModuleDict,
            *,
            reindex_values_by_owner: Mapping[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            torch.testing.assert_close(reindex_values_by_owner["background"], torch.tensor([1.01, 3.01]))
            node.positions = torch.nn.Parameter(original_parameter.detach()[torch.tensor([1, 0])])
            self.assertIsNone(node.positions.grad)
            return {"background": reindex_values_by_owner["background"][torch.tensor([1, 0])]}

        def inspect_addition(
            _nodes: TypedModuleDict,
            *,
            global_step: int,
            source_weights_by_owner: Mapping[str, torch.Tensor],
        ) -> None:
            self.assertEqual(global_step, 2)
            self.assertIsNot(node.positions, original_parameter)
            self.assertIsNone(node.positions.grad)
            torch.testing.assert_close(source_weights_by_owner["background"], torch.tensor([3.01, 1.01]))

        with (
            patch.object(strategy, "relocate_gaussians", side_effect=replace_parameters) as relocate,
            patch.object(strategy, "add_new_gaussians", side_effect=inspect_addition) as add,
            patch.object(strategy, "perturb_gaussians"),
        ):
            strategy._update_step_train_batch_end_impl(0, 2, cast(Any, None), None, nodes)

        relocate.assert_called_once()
        add.assert_called_once()

    def test_survivor_quota_keeps_densest_rows_with_stable_ties(self) -> None:
        protected = MCMCStrategy._protect_track_survivors(
            densities=torch.tensor([0.1, 0.1, 0.9, 0.8]),
            track_mask=torch.tensor([True, True, True, False]),
            dead_mask=torch.ones(4, dtype=torch.bool),
            minimum_survivors=2,
        )

        torch.testing.assert_close(protected, torch.tensor([False, True, False, True]))

    def test_survivor_quota_preserves_existing_visible_donors(self) -> None:
        dead_mask = torch.tensor([False, True, False])
        protected = MCMCStrategy._protect_track_survivors(
            densities=torch.tensor([0.2, 0.9, 0.8]),
            track_mask=torch.ones(3, dtype=torch.bool),
            dead_mask=dead_mask,
            minimum_survivors=2,
        )

        torch.testing.assert_close(protected, dead_mask)

    def test_position_anchor_clamp_limits_absolute_displacement(self) -> None:
        node = SimpleNamespace(
            positions=torch.nn.Parameter(torch.tensor([[2.0, 0.0, 0.0], [0.1, 0.0, 0.0]])),
            mcmc_position_anchor=torch.zeros((2, 3)),
        )

        MCMCStrategy._clamp_position_offset(cast(Any, node), max_offset_m=0.5)

        torch.testing.assert_close(node.positions, torch.tensor([[0.5, 0.0, 0.0], [0.1, 0.0, 0.0]]))

    def test_lazy_position_anchor_survives_setup_and_materializes_from_checkpoint(self) -> None:
        config = make_gaussian_cfg()
        config.perturb.max_position_offset_m = {"vehicle_lights": 0.5}
        node = _LazyAnchoredNode()
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"vehicle_lights": node}))
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)

        strategy.maybe_initialize_buffers(nodes)
        self.assertIsInstance(node.positions, torch.nn.UninitializedParameter)
        self.assertIsInstance(node.mcmc_position_anchor, torch.nn.UninitializedBuffer)

        node.load_state_dict(
            {
                "positions": torch.tensor([[2.0, 0.0, 0.0]]),
                "mcmc_position_anchor": torch.zeros((1, 3)),
            },
            strict=True,
            assign=True,
        )
        MCMCStrategy._clamp_position_offset(cast(Any, node), max_offset_m=0.5)

        torch.testing.assert_close(node.positions, torch.tensor([[0.5, 0.0, 0.0]]))
        torch.testing.assert_close(node.mcmc_position_anchor, torch.zeros((1, 3)))

    def test_materialized_position_rejects_missing_or_uninitialized_anchor(self) -> None:
        config = make_gaussian_cfg()
        config.perturb.max_position_offset_m = {"vehicle_lights": 0.5}
        nodes_to_check = [_MaterializedNode(), _MaterializedNode()]
        nodes_to_check[1].register_buffer("mcmc_position_anchor", torch.nn.UninitializedBuffer())

        for node in nodes_to_check:
            with self.subTest(anchor=getattr(node, "mcmc_position_anchor", None)):
                nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"vehicle_lights": node}))
                with patch("gsplat.has_reloc", return_value=True):
                    strategy = self._make_strategy(config, nodes)

                with self.assertRaisesRegex(
                    ValueError,
                    "vehicle_lights must provide an initialized mcmc_position_anchor buffer",
                ):
                    strategy.maybe_initialize_buffers(nodes)

    def test_materialized_position_rejects_mismatched_anchor_shape(self) -> None:
        config = make_gaussian_cfg()
        config.perturb.max_position_offset_m = {"vehicle_lights": 0.5}
        node = _AnchoredNode()
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"vehicle_lights": node}))
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)

        with self.assertRaisesRegex(ValueError, "mcmc_position_anchor shape .* does not match positions"):
            strategy.maybe_initialize_buffers(nodes)

    def test_excluded_layer_is_not_anchor_validated_clamped_or_mutated(self) -> None:
        config = make_gaussian_cfg()
        config.exclude_layer_ids = ["vehicle_lights"]
        config.perturb.max_position_offset_m = {"vehicle_lights": 0.5}
        node = _AnchoredNode()
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"vehicle_lights": node}))
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)

        positions_before = node.positions.detach().clone()
        strategy.maybe_initialize_buffers(nodes)
        with patch.object(strategy, "_clamp_position_offset", wraps=strategy._clamp_position_offset) as clamp:
            strategy._update_step_train_batch_end_impl(0, 0, cast(Any, None), None, nodes)

        clamp.assert_not_called()
        torch.testing.assert_close(node.positions, positions_before)

    def test_topology_mutation_schedule_matches_batch_end_gates(self) -> None:
        config = make_gaussian_cfg()
        config.relocate.update({"start_iteration": 2, "end_iteration": 12, "frequency": 2})
        config.add.update({"start_iteration": 3, "end_iteration": 12, "frequency": 3})
        config.perturb.update({"start_iteration": 0, "end_iteration": 12, "frequency": 5})
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(1))}))
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)
        expected_operations = {
            1: (False, False, False),
            2: (False, False, False),
            3: (False, False, False),
            4: (True, False, False),
            5: (False, False, True),
            6: (True, True, False),
            9: (False, True, False),
            10: (True, False, True),
            12: (False, False, False),
        }
        operation_order: list[str] = []

        with (
            patch.object(
                strategy,
                "relocate_gaussians",
                side_effect=lambda *_args, **_kwargs: operation_order.append("relocate"),
            ) as relocate,
            patch.object(
                strategy,
                "add_new_gaussians",
                side_effect=lambda *_args, **_kwargs: operation_order.append("add"),
            ) as add,
            patch.object(strategy, "perturb_gaussians") as perturb,
        ):
            for step, expected in expected_operations.items():
                with self.subTest(step=step):
                    strategy.update_step_train_batch_end(0, step, cast(Any, None), None, nodes)

                    actual = (relocate.called, add.called, perturb.called)
                    self.assertEqual(actual, expected)
                    self.assertEqual(
                        operation_order,
                        [name for name, called in zip(("relocate", "add"), expected[:2], strict=True) if called],
                    )
                    self.assertEqual(strategy.may_mutate_topology_at(step), relocate.called or add.called)
                    operation_order.clear()
                    relocate.reset_mock()
                    add.reset_mock()
                    perturb.reset_mock()

            strategy.frozen = True
            strategy.update_step_train_batch_end(0, 6, cast(Any, None), None, nodes)
            self.assertFalse(strategy.may_mutate_topology_at(6))
            relocate.assert_not_called()
            add.assert_not_called()
            perturb.assert_not_called()

    def test_registered_callbacks_execute_with_and_without_coscheduled_scores(self) -> None:
        config = make_gaussian_cfg()
        config.relocate.update({"start_iteration": 2, "end_iteration": 12, "frequency": 2})
        config.add.update({"start_iteration": 3, "end_iteration": 12, "frequency": 3})
        nodes: TypedModuleDict = TypedModuleDict()
        before = {"background": torch.tensor([1.0, 2.0])}
        after = {"background": torch.tensor([2.0, 1.0])}
        for enabled in (False, True):
            config.name = "taminggs_mcmc" if enabled else "mcmc"
            with patch("gsplat.has_reloc", return_value=True):
                strategy = self._make_strategy(config, nodes)
            for step in (4, 6, 9, 12):
                with self.subTest(enabled=enabled, step=step):
                    registered_relocate = MagicMock(return_value=after)
                    registered_add = MagicMock()
                    with (
                        patch.object(
                            strategy,
                            "_topology_mutation_ops",
                            return_value=(
                                (strategy.config.relocate, registered_relocate),
                                (strategy.config.add, registered_add),
                            ),
                        ),
                        patch.object(TamingGSMCMCStrategy, "_taminggs_source_weights", return_value=before) as score,
                        patch.object(strategy, "relocate_gaussians") as unregistered_relocate,
                        patch.object(strategy, "add_new_gaussians") as unregistered_add,
                    ):
                        strategy._update_step_train_batch_end_impl(0, step, cast(Any, None), None, nodes)
                        self.assertEqual(strategy.may_mutate_topology_at(step), step != 12)
                    self.assertEqual(registered_relocate.called, step in (4, 6))
                    self.assertEqual(registered_add.called, step in (6, 9))
                    self.assertEqual(score.called, enabled and step == 6)
                    if registered_add.called:
                        self.assertIs(
                            registered_add.call_args.kwargs["source_weights_by_owner"],
                            after if enabled and step == 6 else None,
                        )
                    if enabled and step == 6:
                        self.assertIs(registered_relocate.call_args.kwargs["reindex_values_by_owner"], before)
                    unregistered_relocate.assert_not_called()
                    unregistered_add.assert_not_called()

    def test_topology_mutation_schedule_ignores_rank_local_layout(self) -> None:
        config = make_gaussian_cfg()
        config.relocate.update({"start_iteration": 2, "end_iteration": 12, "frequency": 2})
        config.add.update({"start_iteration": 3, "end_iteration": 12, "frequency": 3})
        layouts: tuple[TypedModuleDict[Any], TypedModuleDict[Any]] = (
            TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(1))})),
            TypedModuleDict(
                cast(
                    Any,
                    {
                        "road": _StrategyNode(torch.ones(7)),
                        "vehicle": _StrategyNode(torch.empty(0)),
                    },
                )
            ),
        )
        with patch("gsplat.has_reloc", return_value=True):
            strategies = tuple(self._make_strategy(config, nodes) for nodes in layouts)
        steps = (1, 2, 3, 4, 6, 9, 12)

        decisions = tuple(tuple(strategy.may_mutate_topology_at(step) for step in steps) for strategy in strategies)
        self.assertEqual(decisions[0], decisions[1])
        self.assertEqual(decisions[0], (False, False, False, True, True, True, False))

        for strategy in strategies:
            strategy.frozen = True
            self.assertFalse(any(strategy.may_mutate_topology_at(step) for step in steps))

    def test_runtime_scene_refresh_skips_non_gaussian_placeholders(self) -> None:
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy()
        gaussian = object.__new__(BaseGaussianModel)
        torch.nn.Module.__init__(gaussian)
        refresh_mcmc_scene = MagicMock()
        gaussian.refresh_mcmc_scene = cast(Any, refresh_mcmc_scene)
        nodes: TypedModuleDict[BaseGaussianModel] = TypedModuleDict(
            cast(Any, {"gaussian": gaussian, "placeholder": torch.nn.Identity()})
        )

        strategy.refresh_runtime_scenes(nodes)

        refresh_mcmc_scene.assert_called_once_with("gaussian")

    def test_addition_at_cap_is_a_noop_before_device_and_rng_checks(self):
        config = make_gaussian_cfg()
        config.add.max_n_gaussians = 20
        nodes = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(20))}))
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)

        with (
            patch("nurec.models.gaussians.strategies.mcmc.get_cuda_generator") as get_rng,
            patch("nurec.models.gaussians.strategies.mcmc.gsplat_add_gaussians") as operation,
            patch.object(torch.cuda, "empty_cache") as empty_cache,
        ):
            strategy.add_new_gaussians(nodes)

        get_rng.assert_not_called()
        operation.assert_not_called()
        empty_cache.assert_not_called()

    def test_relocation_with_no_selected_layers_is_a_noop(self):
        config = make_gaussian_cfg("taminggs_mcmc")
        config.exclude_layer_ids = ["background"]
        config.add.taminggs = {}
        nodes = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(4))}))
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)

        with (
            patch("nurec.models.gaussians.strategies.mcmc.get_cuda_generator") as get_rng,
            patch("nurec.models.gaussians.strategies.mcmc.gsplat_relocate_gaussians") as operation,
            patch.object(strategy, "_taminggs_source_weights", return_value={}) as source_weights,
        ):
            strategy.relocate_gaussians(nodes)

        source_weights.assert_called_once()
        get_rng.assert_not_called()
        operation.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "MCMC policy tests require CUDA")
    def test_relocation_global_noops_do_not_request_rng(self):
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.taminggs = {}
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config)

        for densities in (torch.full((3,), 0.5, device="cuda"), torch.zeros(3, device="cuda")):
            node = _StrategyNode(densities)
            nodes = TypedModuleDict(cast(Any, {"background": node}))
            with (
                patch("nurec.models.gaussians.strategies.mcmc.get_cuda_generator") as get_rng,
                patch("nurec.models.gaussians.strategies.mcmc.gsplat_relocate_gaussians") as operation,
                patch.object(
                    strategy,
                    "_taminggs_source_weights",
                    return_value={"background": torch.ones_like(densities)},
                ) as source_weights,
            ):
                strategy.relocate_gaussians(nodes)

            source_weights.assert_called_once()
            get_rng.assert_not_called()
            operation.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "MCMC policy tests require CUDA")
    def test_individually_all_dead_owner_keeps_deterministic_ineligible_survivor(self):
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy()
        all_dead_densities = torch.tensor([0.001, 0.003, 0.003], device="cuda")
        partly_live_densities = torch.tensor([0.0, 0.5], device="cuda")
        nodes = TypedModuleDict(
            cast(
                Any,
                {
                    "all_dead": _StrategyNode(all_dead_densities),
                    "partly_live": _StrategyNode(partly_live_densities),
                },
            )
        )

        with patch(
            "nurec.models.gaussians.strategies.mcmc.gsplat_relocate_gaussians",
            return_value=True,
        ) as operation:
            strategy.relocate_gaussians(nodes)

        call = operation.call_args.kwargs
        torch.testing.assert_close(
            call["dead_by_owner"]["all_dead"],
            torch.tensor([True, False, True], device="cuda"),
        )
        torch.testing.assert_close(
            call["source_eligible_by_owner"]["all_dead"],
            torch.tensor([False, False, False], device="cuda"),
        )
        torch.testing.assert_close(
            call["dead_by_owner"]["partly_live"],
            torch.tensor([True, False], device="cuda"),
        )
        torch.testing.assert_close(
            call["source_eligible_by_owner"]["partly_live"],
            torch.tensor([False, True], device="cuda"),
        )
        self.assertEqual(sum(mask.numel() for mask in call["dead_by_owner"].values()), 5)
        self.assertEqual([node.density_queries for node in nodes.values()], [1, 1])

    @unittest.skipUnless(torch.cuda.is_available(), "MCMC scheduling tests require CUDA")
    def test_same_step_addition_observes_post_relocation_topology(self):
        config = make_gaussian_cfg()
        config.relocate.start_iteration = 0
        config.relocate.end_iteration = 2
        config.add.start_iteration = 0
        config.add.end_iteration = 2
        config.perturb.start_iteration = 0
        config.perturb.end_iteration = 2
        config.add.max_n_gaussians = 21
        nodes = TypedModuleDict(
            cast(
                Any,
                {
                    "first": _StrategyNode(
                        torch.cat([torch.zeros(1, device="cuda"), torch.full((9,), 0.5, device="cuda")])
                    ),
                    "second": _StrategyNode(torch.full((10,), 0.5, device="cuda")),
                },
            )
        )
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)
        strategy.perturb_noise_lr = {"first": 1.0, "second": 1.0}
        selected_calls = []

        def relocate(selected, **_kwargs):
            selected_calls.append(selected)
            nodes["first"].activated_densities = torch.full((9,), 0.5, device="cuda")
            nodes["second"].activated_densities = torch.full((11,), 0.5, device="cuda")
            return True

        def add(selected, **_kwargs):
            selected_calls.append(selected)
            self.assertEqual(selected.layer_num_gaussians, {"first": 9, "second": 11})
            return True

        with (
            patch("nurec.models.gaussians.strategies.mcmc.gsplat_relocate_gaussians", side_effect=relocate),
            patch("nurec.models.gaussians.strategies.mcmc.gsplat_add_gaussians", side_effect=add),
            patch.object(strategy, "perturb_gaussians") as perturb,
        ):
            strategy.update_step_train_batch_end(1, 1, cast(Any, None), None, nodes)

        self.assertEqual(len(selected_calls), 2)
        self.assertIsNot(selected_calls[0], selected_calls[1])
        self.assertEqual([call.args[0] for call in perturb.call_args_list], ["first", "second"])

    def test_enabled_visibility_materializes_on_load(self):
        config = make_gaussian_cfg()
        config.relocate.max_invisible_steps = 10
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _LazyNode()}))
        strategy = self._make_strategy(config, nodes)
        strategy.maybe_initialize_buffers(nodes)
        strategy.invisible_steps.register_buffer("metadata", torch.tensor([7]), persistent=True)

        self.assertIsInstance(strategy.invisible_steps["background"], torch.nn.UninitializedParameter)
        state = strategy.state_dict()
        self.assertNotIn("invisible_steps.background", state)
        torch.testing.assert_close(state["invisible_steps.metadata"], torch.tensor([7]))
        expected = torch.arange(4, dtype=torch.int32)
        strategy.load_state_dict(
            {
                "invisible_steps.background": expected,
                "invisible_steps.metadata": torch.tensor([7]),
            },
            strict=True,
            assign=True,
        )
        torch.testing.assert_close(strategy.invisible_steps["background"], expected)

    def _make_visibility_strategy(self, counts, device):
        class _VisibilityNode(torch.nn.Module):
            def __init__(self, count):
                super().__init__()
                self.positions = torch.nn.Parameter(torch.zeros(count, 3, device=device))

            @property
            def device(self):
                return self.positions.device

            def get_num_gaussians(self):
                return self.positions.shape[0]

        config = make_gaussian_cfg()
        config.relocate.max_invisible_steps = 3
        nodes = TypedModuleDict(
            cast(
                Any, {name: _LazyNode() if count is None else _VisibilityNode(count) for name, count in counts.items()}
            )
        )
        strategy = _make_concrete_mcmc_strategy(
            config=TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(config),
            trainer_config=make_trainer_cfg(),
            init_from_datasource=False,
            gaussians_nodes=nodes,
        )
        return strategy, nodes

    def test_visibility_transition_preserves_nurec_layer_order_and_deduplication(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=device):
                strategy, nodes = self._make_visibility_strategy({"first": 2, "second": 1}, device)
                strategy.maybe_initialize_buffers(nodes)
                first_results = object()
                strategy.update_visibility_counters(
                    results=first_results,
                    visibility=torch.tensor([1.0, 0.0, 0.0], device=device),
                    gaussians_nodes=nodes,
                )
                strategy.update_visibility_counters(
                    results=first_results, visibility=torch.zeros(3, device=device), gaussians_nodes=nodes
                )
                strategy.update_visibility_counters(
                    results=object(), visibility=torch.tensor([0.0, 1.0, 1.0], device=device), gaussians_nodes=nodes
                )
                torch.testing.assert_close(
                    strategy.invisible_steps["first"], torch.tensor([1, 0], dtype=torch.int32, device=device)
                )
                torch.testing.assert_close(
                    strategy.invisible_steps["second"], torch.tensor([0], dtype=torch.int32, device=device)
                )

    def test_visibility_skips_empty_and_uninitialized_owners(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for skipped in (0, None):
                with self.subTest(device=device, skipped=skipped):
                    strategy, nodes = self._make_visibility_strategy(
                        {"first": 2, "skipped": skipped, "second": 1}, device
                    )
                    strategy.update_visibility_counters(
                        results=object(), visibility=torch.tensor([0.0, 1.0, 0.0], device=device), gaussians_nodes=nodes
                    )
                    torch.testing.assert_close(
                        strategy.invisible_steps["first"], torch.tensor([1, 0], dtype=torch.int32, device=device)
                    )
                    torch.testing.assert_close(
                        strategy.invisible_steps["second"], torch.tensor([1], dtype=torch.int32, device=device)
                    )
                    self.assertIsInstance(strategy.invisible_steps["skipped"], torch.nn.UninitializedParameter)

    def test_visibility_length_mismatch_updates_only_valid_prefix(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=device):
                strategy, nodes = self._make_visibility_strategy({"first": 2, "second": 1}, device)
                strategy.maybe_initialize_buffers(nodes)
                strategy.invisible_steps["second"].fill_(2)
                with self.assertLogs("nurec.models.gaussians.strategies.mcmc", level="WARNING"):
                    strategy.update_visibility_counters(
                        results=object(), visibility=torch.tensor([0.0, 1.0], device=device), gaussians_nodes=nodes
                    )
                torch.testing.assert_close(
                    strategy.invisible_steps["first"], torch.tensor([1, 0], dtype=torch.int32, device=device)
                )
                torch.testing.assert_close(
                    strategy.invisible_steps["second"], torch.tensor([2], dtype=torch.int32, device=device)
                )

    def test_taminggs_tracks_visibility_without_relocation_threshold(self):
        class _VisibilityNode(torch.nn.Module):
            def __init__(self, count: int):
                super().__init__()
                self.positions = torch.nn.Parameter(torch.zeros(count, 3))

            @property
            def device(self) -> torch.device:
                return self.positions.device

            def get_num_gaussians(self) -> int:
                return self.positions.shape[0]

        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.taminggs = {"visibility_weight": 5.0}
        config.add.max_n_gaussians = 3
        nodes = TypedModuleDict(cast(Any, {"background": _VisibilityNode(3)}))
        strategy = _make_concrete_mcmc_strategy(
            config=TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(config),
            trainer_config=make_trainer_cfg(),
            init_from_datasource=False,
            gaussians_nodes=nodes,
        )
        strategy.maybe_initialize_buffers(nodes)

        self.assertIsNone(strategy.config.relocate.max_invisible_steps)
        strategy.on_train_step_start()
        strategy.update_prepared_visibility(
            visibility=torch.tensor([1.0, 0.0, 0.0]),
            gaussians_nodes=nodes,
        )
        strategy.on_train_step_start()
        strategy.update_prepared_visibility(
            visibility=torch.tensor([0.0, 1.0, 0.0]),
            gaussians_nodes=nodes,
        )

        torch.testing.assert_close(
            strategy.invisible_steps["background"],
            torch.tensor([1, 0, 2], dtype=torch.int32),
        )
        self.assertEqual(strategy._visibility_counter_cap(), 2**30)

    def test_taminggs_visibility_consumes_pre_compaction_union_in_ddp(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.taminggs = {"visibility_weight": 5.0}
        config.add.max_n_gaussians = 3
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(3))}))
        strategy = self._make_strategy(config, nodes)
        strategy.maybe_initialize_buffers(nodes)

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
        ):
            strategy.update_prepared_visibility(
                visibility=torch.tensor([1.0, 0.0, 1.0]),
                gaussians_nodes=nodes,
            )

        torch.testing.assert_close(
            strategy.invisible_steps["background"],
            torch.tensor([0, 1, 0], dtype=torch.int32),
        )

    def test_taminggs_visibility_fails_before_partial_update_on_row_mismatch(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.taminggs = {"visibility_weight": 5.0}
        config.add.max_n_gaussians = 3
        nodes: TypedModuleDict = TypedModuleDict(
            cast(
                Any,
                {
                    "first": _StrategyNode(torch.ones(2)),
                    "second": _StrategyNode(torch.ones(1)),
                },
            )
        )
        strategy = self._make_strategy(config, nodes)
        strategy.maybe_initialize_buffers(nodes)
        strategy.on_train_step_start()
        for size in (0, 2, 4):
            with self.subTest(size=size), self.assertRaisesRegex(RuntimeError, f"{size} != 3 Gaussian rows"):
                strategy.update_prepared_visibility(visibility=torch.zeros(size), gaussians_nodes=nodes)
            torch.testing.assert_close(strategy.invisible_steps["first"], torch.zeros(2, dtype=torch.int32))
            torch.testing.assert_close(strategy.invisible_steps["second"], torch.zeros(1, dtype=torch.int32))

    def test_taminggs_prepared_visibility_survives_permutation_and_checkpoint_round_trip(self) -> None:
        config = make_gaussian_cfg("taminggs_mcmc")
        config.add.max_n_gaussians = 3
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(3))}))
        strategy = self._make_strategy(config, nodes)
        strategy.maybe_initialize_buffers(nodes)
        strategy.invisible_steps["background"].copy_(torch.tensor([4, 5, 6], dtype=torch.int32))
        strategy.frozen = True
        strategy.on_train_step_start()
        strategy.update_prepared_visibility(visibility=torch.tensor([1.0, 0.0, 0.0]), gaussians_nodes=nodes)
        order = torch.tensor([2, 0, 1])
        strategy.permute_gaussian_rows({"background": order})
        expected = torch.tensor([7, 0, 6], dtype=torch.int32)
        torch.testing.assert_close(strategy.invisible_steps["background"], expected)
        strategy.update_step_train_batch_end(0, 0, cast(Any, None), None, nodes)
        torch.testing.assert_close(strategy.invisible_steps["background"], expected)
        restored = self._make_strategy(config, nodes)
        restored.maybe_initialize_buffers(nodes)
        restored.load_state_dict(strategy.state_dict())
        torch.testing.assert_close(restored.invisible_steps["background"], expected)
        restored.update_prepared_visibility(visibility=torch.tensor([0.0, 1.0, 0.0]), gaussians_nodes=nodes)
        torch.testing.assert_close(restored.invisible_steps["background"], torch.tensor([8, 0, 7], dtype=torch.int32))

    def test_strict_checkpoint_requires_enabled_visibility_counters(self):
        for name in ("mcmc", "taminggs_mcmc"):
            with self.subTest(strategy=name):
                config = make_gaussian_cfg(name)
                config.relocate.max_invisible_steps = 10
                config.add.max_n_gaussians = 3
                nodes = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(3))}))
                strategy = self._make_strategy(config, nodes)
                strategy.maybe_initialize_buffers(nodes)
                incomplete = {
                    key: value.clone()
                    for key, value in strategy.state_dict().items()
                    if key != "invisible_steps.background"
                }
                strategy.invisible_steps["background"].fill_(9)
                with self.assertRaisesRegex(RuntimeError, "invisible_steps.background"):
                    strategy.load_state_dict(incomplete, strict=True)
                torch.testing.assert_close(
                    strategy.invisible_steps["background"], torch.full((3,), 9, dtype=torch.int32)
                )

    def test_zero_relocation_visibility_threshold_does_not_drive_relocation(self):
        config = make_gaussian_cfg("taminggs_mcmc")
        config.relocate.max_invisible_steps = 0
        config.add.taminggs = {"visibility_weight": 5.0}
        strategy = self._make_strategy(config)

        self.assertFalse(strategy._relocation_uses_visibility())
        self.assertTrue(strategy._should_track_visibility())
        self.assertEqual(strategy._visibility_counter_cap(), 2**30)

    def test_visibility_guard_advances_on_a_reused_id_in_the_next_step(self):
        """A render result whose Python id equals the previous step's (CPython reuses a freed object's address) is a
        new render once the next training step has started: the counters advance. This holds while the strategy is
        frozen too (controller distillation freezes it), whose batch end returns early. A repeated call within one step
        (the scale and the density loss both update the counters) stays deduplicated, and a new id within the step
        advances, as before."""
        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                strategy, nodes = self._make_visibility_strategy({"first": 2, "second": 1}, "cpu")
                strategy.maybe_initialize_buffers(nodes)
                strategy.frozen = frozen
                results = object()  # the same object stands in for a new result that got the freed object's address
                invisible = torch.zeros(3)

                def counters() -> list[int]:
                    return strategy.invisible_steps["first"].tolist() + strategy.invisible_steps["second"].tolist()

                def end_batch() -> None:  # step 0: nothing scheduled
                    strategy.update_step_train_batch_end(
                        epoch=0, global_step=0, batch=cast(Any, None), system=None, gaussians_nodes=TypedModuleDict()
                    )

                strategy.on_train_step_start()
                strategy.update_visibility_counters(results=results, visibility=invisible, gaussians_nodes=nodes)
                self.assertEqual(counters(), [1, 1, 1])
                strategy.update_visibility_counters(results=results, visibility=invisible, gaussians_nodes=nodes)
                self.assertEqual(counters(), [1, 1, 1])  # the same render within the step: deduplicated
                end_batch()
                strategy.on_train_step_start()
                strategy.update_visibility_counters(results=results, visibility=invisible, gaussians_nodes=nodes)
                self.assertEqual(counters(), [2, 2, 2])  # the reused id in the next step: a new render
                strategy.update_visibility_counters(results=results, visibility=invisible, gaussians_nodes=nodes)
                self.assertEqual(counters(), [2, 2, 2])  # its second call in the step: deduplicated
                strategy.update_visibility_counters(results=object(), visibility=invisible, gaussians_nodes=nodes)
                self.assertEqual(counters(), [3, 3, 3])  # another result within the step: advances


class TestMCMCPermuteGaussianRows(unittest.TestCase):
    def test_invisible_steps_follows_the_permutation(self):
        # invisible_steps is row-aligned but lives on the strategy, so a reorder inside a node cannot reach it.
        # Left behind, relocation recycles by another Gaussian's history.
        strategy = _make_concrete_mcmc_strategy(
            config=TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(make_gaussian_cfg()),
            trainer_config=make_trainer_cfg(),
            init_from_datasource=False,
            gaussians_nodes=TypedModuleDict(),
        )
        strategy.invisible_steps["background"] = torch.nn.Parameter(
            torch.tensor([10.0, 20.0, 30.0]), requires_grad=False
        )

        strategy.permute_gaussian_rows({"background": torch.tensor([2, 0, 1])})

        assert torch.equal(strategy.invisible_steps["background"], torch.tensor([30.0, 10.0, 20.0]))


OptimizerVariant = Literal["fused_adam", "selective_adam"]


class _TrackBufferNode(BaseGaussianModel):
    """Minimal owner with production Adam variants and a row sidecar."""

    row_ids: torch.Tensor

    def __init__(self, count: int, optimizer_variant: OptimizerVariant, scene_id: str = "dynamic") -> None:
        """Materialize a CUDA scene, sidecar row IDs and production Adam optimizer state."""
        torch.nn.Module.__init__(self)
        device = torch.device("cuda")
        self.register_buffer("_device_indicator", torch.tensor(0, device=device), persistent=False)
        self.config = cast(Any, SimpleNamespace(density_activation="sigmoid", scale_activation="exp"))
        self.density_activation = cast(Any, torch.sigmoid)
        self.scale_activation = cast(Any, torch.exp)
        self.positions = torch.nn.Parameter(torch.randn(count, 3, device=device))
        self.rotations = torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(count, -1).clone()
        )
        self.scales = torch.nn.Parameter(torch.full((count, 3), -2.0, device=device))
        self.densities = torch.nn.Parameter(torch.full((count, 1), 1.0, device=device))
        self.extra_signal = torch.nn.Parameter(torch.randn(count, 2, device=device))
        self.camera_extra_signal = torch.nn.Parameter(torch.randn(count, 1, device=device))
        self.lidar_extra_signal = torch.nn.Parameter(torch.randn(count, 1, device=device))
        self.register_buffer("row_ids", torch.arange(count, device=device))
        groups = [
            {"params": [self.positions], "name": "model.positions", "lr": 1.0e-3},
            {"params": [self.densities], "name": "model.densities", "lr": 1.0e-3},
            {"params": [self.scales], "name": "model.scales", "lr": 1.0e-3},
        ]
        if optimizer_variant == "fused_adam":
            optimizer = apex.optimizers.FusedAdam(groups)
        else:
            optimizer = SelectiveAdam(groups, selective_enabled=True)
        (self.positions.sum() + self.densities.sum() + self.scales.sum()).backward()
        if optimizer_variant == "selective_adam":
            optimizer.set_visibility(torch.ones(count, device=device, dtype=torch.bool))
        optimizer.step()
        optimizer.zero_grad()
        self.optimizers = cast(Any, [{"optimizer": optimizer}])
        self.refresh_mcmc_scene(scene_id)

    def get_additional_buffers(self) -> dict[str, torch.Tensor]:
        return {"row_ids": self.row_ids}


@pytest.mark.skipif(
    not torch.cuda.is_available() or not gsplat.has_reloc() or not gsplat.has_adam(),
    reason="Independent MCMC pools require CUDA, RELOC, and ADAM",
)
@pytest.mark.parametrize("taminggs_enabled", [False, True])
def test_independent_gpu_pool_respects_cap_and_never_borrows_background_donors(taminggs_enabled: bool) -> None:
    """Real GPU pools honor separate caps and cannot repopulate barriers from background."""
    nodes: TypedModuleDict[BaseGaussianModel] = TypedModuleDict(
        {name: _TrackBufferNode(40, "selective_adam", name) for name in ("background", "barriers")}
    )
    config = make_gaussian_cfg()
    config.add.max_n_gaussians = 42
    config.independent_layer_max_n_gaussians = {"barriers": 41}
    config.name = "taminggs_mcmc" if taminggs_enabled else "mcmc"
    config.add.update({"start_iteration": 0, "end_iteration": 4, "frequency": 2})
    strategy = TestMCMCStrategy._make_strategy(config, nodes).to("cuda")
    strategy.maybe_initialize_buffers(nodes)
    strategy.add_new_gaussians(nodes, global_step=2)
    assert nodes["background"].get_num_gaussians() == 42
    assert nodes["barriers"].get_num_gaussians() == 41
    with torch.no_grad():
        nodes["barriers"].densities.fill_(-20)
    positions = {name: node.positions.clone() for name, node in nodes.items()}
    strategy.relocate_gaussians(nodes)
    for name, node in nodes.items():
        torch.testing.assert_close(node.positions, positions[name])


@pytest.mark.skipif(
    not torch.cuda.is_available() or not gsplat.has_reloc() or not gsplat.has_adam(),
    reason="MCMC optimizer integration requires CUDA, RELOC, and ADAM",
)
@pytest.mark.parametrize("optimizer_variant", ["fused_adam", "selective_adam"])
@pytest.mark.parametrize("operation", ["addition", "relocation"])
@pytest.mark.parametrize("taminggs_enabled", [False, True])
def test_topology_keeps_production_adam_state_and_track_buffer_aligned(
    optimizer_variant: OptimizerVariant,
    operation: str,
    taminggs_enabled: bool,
) -> None:
    node = _TrackBufferNode(40, optimizer_variant)
    optimizer = node.optimizers[0]["optimizer"]
    old_moments: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for group in optimizer.param_groups:
            parameter = group["params"][0]
            parameter_name = group["name"].rsplit(".", 1)[-1]
            old_moments[parameter_name] = {}
            for state_name, offset in (("exp_avg", 0), ("exp_avg_sq", 100)):
                state = optimizer.state[parameter][state_name]
                row_values = torch.arange(1, 41, device=state.device, dtype=state.dtype).reshape(
                    40, *([1] * (state.ndim - 1))
                )
                state.copy_(row_values.expand_as(state) + offset)
                old_moments[parameter_name][state_name] = state.clone()
    nodes: TypedModuleDict[BaseGaussianModel] = TypedModuleDict(cast(Any, {"dynamic": node}))
    config = make_gaussian_cfg()
    config.add.max_n_gaussians = 42
    config.name = "taminggs_mcmc" if taminggs_enabled else "mcmc"
    if taminggs_enabled:
        config.add.update({"start_iteration": 0, "end_iteration": 4, "frequency": 2})
    strategy = TestMCMCStrategy._make_strategy(config, nodes).to("cuda")
    strategy.maybe_initialize_buffers(nodes)

    if operation == "addition":
        strategy.add_new_gaussians(nodes, global_step=2)
        assert node.get_num_gaussians() == 42
    else:
        with torch.no_grad():
            node.densities[0] = -10.0
        strategy.relocate_gaussians(nodes)
        assert node.get_num_gaussians() == 40

    assert node.row_ids.shape == (node.get_num_gaussians(),)
    assert set(node.row_ids.tolist()) <= set(range(40))
    retained_count = 40 if operation == "addition" else 39
    retained_rows = node.row_ids[:retained_count]
    donor_rows = node.row_ids[retained_count:]
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        parameter_name = group["name"].rsplit(".", 1)[-1]
        assert parameter is getattr(node, parameter_name)
        for state_name, state in optimizer.state[parameter].items():
            if isinstance(state, torch.Tensor) and state.ndim > 0:
                assert state.shape[0] == node.get_num_gaussians()
            if state_name not in old_moments[parameter_name]:
                continue
            old_state = old_moments[parameter_name][state_name]
            retained_state = old_state[retained_rows].clone()
            if operation == "relocation" and parameter_name in ("densities", "scales"):
                retained_state[torch.isin(retained_rows, donor_rows)] = 0
            expected = torch.cat([retained_state, torch.zeros_like(old_state[donor_rows])])
            torch.testing.assert_close(state, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not gsplat.has_reloc() or not gsplat.has_adam(),
    reason="Co-scheduled TamingGS integration requires CUDA, RELOC, and ADAM",
)
@pytest.mark.parametrize("owner_ids", [("dynamic",), ("background", "dynamic")])
def test_taminggs_coscheduled_cuda_mutations_preserve_scores_and_optimizer_progress(owner_ids: tuple[str, ...]) -> None:
    owners = {owner_id: _TrackBufferNode(40, "selective_adam", scene_id=owner_id) for owner_id in owner_ids}
    nodes: TypedModuleDict[BaseGaussianModel] = TypedModuleDict(cast(Any, owners))
    config = make_gaussian_cfg("taminggs_mcmc")
    config.add.update(
        {
            "start_iteration": 450,
            "end_iteration": 27_000,
            "frequency": 450,
            "max_n_gaussians": 40 * len(owners) + 2,
            "taminggs": {},
        }
    )
    config.relocate.update({"start_iteration": 450, "end_iteration": 22_500, "frequency": 450})
    strategy = TestMCMCStrategy._make_strategy(config, nodes).to("cuda")
    strategy.maybe_initialize_buffers(nodes)
    original_visibility = torch.arange(1, 41, device="cuda", dtype=torch.int32)
    for owner_index, (owner_id, node) in enumerate(owners.items()):
        with torch.no_grad():
            node.densities[0] = -10.0
        gradients = torch.arange(1, 41, device="cuda", dtype=torch.float32) * (owner_index + 1)
        node.positions.grad = gradients[:, None].expand(-1, 3).clone()
        strategy.invisible_steps[owner_id].copy_(original_visibility)
    original_scores = strategy._taminggs_source_weights(strategy._addition_selected_nodes(nodes))
    original_positions = {owner_id: node.positions for owner_id, node in owners.items()}

    with (
        patch.object(strategy, "add_new_gaussians", wraps=strategy.add_new_gaussians) as addition,
        patch(
            "nurec.models.gaussians.strategies.mcmc.gsplat_add_gaussians",
            wraps=gsplat_add_gaussians,
        ) as adapter_addition,
        patch.object(strategy, "perturb_gaussians"),
    ):
        strategy._update_step_train_batch_end_impl(0, 900, cast(Any, None), None, nodes)

    assert sum(node.get_num_gaussians() for node in owners.values()) == 40 * len(owners) + 1
    addition.assert_called_once()
    adapter_addition.assert_called_once()
    for owner_id, node in owners.items():
        assert node.positions is not original_positions[owner_id]
        assert node.positions.grad is None
        assert 0 not in node.row_ids.tolist()
        carried_scores = addition.call_args.kwargs["source_weights_by_owner"][owner_id]
        pre_addition_count = adapter_addition.call_args.args[0].layer_num_gaussians[owner_id]
        torch.testing.assert_close(carried_scores, original_scores[owner_id][node.row_ids[:pre_addition_count]])
        torch.testing.assert_close(
            adapter_addition.call_args.kwargs["source_weights_by_owner"][owner_id], carried_scores
        )
        expected_visibility = torch.cat(
            [original_visibility[node.row_ids[:39]], original_visibility.new_zeros(node.get_num_gaussians() - 39)]
        )
        torch.testing.assert_close(strategy.invisible_steps[owner_id], expected_visibility)
        optimizer = cast(SelectiveAdam, node.optimizers[0]["optimizer"])
        before_step = node.positions.detach().clone()
        (node.positions.sum() + node.densities.sum() + node.scales.sum()).backward()
        optimizer.set_visibility(torch.ones(node.get_num_gaussians(), device="cuda", dtype=torch.bool))
        optimizer.step()
        assert torch.isfinite(node.positions).all()
        assert torch.all(node.positions < before_step)
        for group in optimizer.param_groups:
            parameter = group["params"][0]
            assert parameter is getattr(node, group["name"].rsplit(".", 1)[-1])
            assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape
            assert optimizer.state[parameter]["exp_avg_sq"].shape == parameter.shape


class _ScaleOnlyNode(torch.nn.Module):
    def __init__(self, activated_scales: torch.Tensor):
        super().__init__()
        self.scales = torch.nn.Parameter(torch.log(activated_scales))
        self.scene_extent = 100.0
        self.scale_activation = cast(Any, torch.exp)
        self.scale_activation_inv = cast(Any, torch.log)

    def get_scales(self, preactivation: bool = False) -> torch.Tensor:
        return self.scales if preactivation else self.scale_activation(self.scales)


class _RigidScaleOnlyNode(RigidGaussianModel):
    def __init__(
        self,
        activated_scales: torch.Tensor,
        cuboid_dims: torch.Tensor,
        gaussian_cuboid_ids: torch.Tensor,
        activated_track_scales: torch.Tensor | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)  # intentionally skip RigidGaussianModel.__init__
        self.scales = torch.nn.Parameter(torch.log(activated_scales))
        self.scene_extent = 100.0
        self.scale_activation = cast(Any, torch.exp)
        self.scale_activation_inv = cast(Any, torch.log)
        self.gaussian_cuboid_ids = torch.nn.Buffer(gaussian_cuboid_ids)
        self.cuboid_tracks = cast(Any, SimpleNamespace(cuboids_dims=cuboid_dims))
        self.track_scales = (
            None if activated_track_scales is None else torch.nn.Parameter(torch.log(activated_track_scales))
        )

    def get_scales(self, preactivation: bool = False) -> torch.Tensor:
        return self.scales if preactivation else self.scale_activation(self.scales)


class TestMCMCMaxScaleGuard(unittest.TestCase):
    @staticmethod
    def _strategy(config: DictConfig, nodes: TypedModuleDict, *, max_steps: int = 30_000) -> Any:
        trainer_config = make_trainer_cfg().model_copy(update={"max_steps": max_steps})
        return _make_concrete_mcmc_strategy(
            config=TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(config),
            trainer_config=trainer_config,
            init_from_datasource=False,
            gaussians_nodes=nodes,
        )

    def test_disabled_guard_leaves_scales_unchanged(self) -> None:
        node = _ScaleOnlyNode(torch.tensor([[4.0, 2.0, 1.0]]))
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": node}))
        strategy = self._strategy(make_gaussian_cfg(), nodes)

        strategy.enforce_max_scale_guard(nodes)

        torch.testing.assert_close(node.get_scales(), torch.tensor([[4.0, 2.0, 1.0]]))

    def test_guard_maps_positive_infinity_to_upper_bound_and_other_invalid_values_to_minimum(self) -> None:
        node = _ScaleOnlyNode(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, float("inf"), float("nan")],
                    [0.0, float("-inf"), 2.0],
                ]
            )
        )
        valid_raw_scale = node.scales[0, 1].detach().clone()
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": node}))
        config = make_gaussian_cfg()
        config["max_scale_guard"] = {"enabled": True}
        strategy = self._strategy(config, nodes)

        strategy.enforce_max_scale_guard(nodes)

        guarded_scales = node.get_scales()
        self.assertEqual(guarded_scales.shape[0], 3)
        self.assertTrue(bool(torch.isfinite(guarded_scales).all()))
        self.assertTrue(bool((guarded_scales <= 3.0).all()))
        self.assertTrue(bool((guarded_scales > 0.0).all()))
        torch.testing.assert_close(guarded_scales[1, 1], torch.tensor(3.0))
        torch.testing.assert_close(guarded_scales[1, 2], torch.tensor(1.0e-8))
        torch.testing.assert_close(guarded_scales[2, 0], torch.tensor(1.0e-8))
        torch.testing.assert_close(guarded_scales[2, 1], torch.tensor(1.0e-8))
        self.assertTrue(torch.equal(node.scales[0, 1], valid_raw_scale))

    def test_guard_schedule_uses_exclusive_window_and_exact_training_end_flush(self) -> None:
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": _ScaleOnlyNode(torch.ones((1, 3)))}))
        config = make_gaussian_cfg()
        config["max_scale_guard"] = {
            "enabled": True,
            "start_iteration": 2,
            "end_iteration": 6,
            "frequency": 2,
        }
        strategy = self._strategy(config, nodes, max_steps=10)

        self.assertFalse(strategy._is_max_scale_guard_due(2))
        self.assertTrue(strategy._is_max_scale_guard_due(4))
        self.assertFalse(strategy._is_max_scale_guard_due(6))
        self.assertFalse(strategy._is_max_scale_guard_due(8))
        self.assertTrue(strategy._is_max_scale_guard_due(10))
        self.assertFalse(strategy._is_max_scale_guard_due(11))

    def test_serialization_override_is_scoped_idempotent_and_mutation_free(self) -> None:
        background = _ScaleOnlyNode(torch.tensor([[4.0, 2.0, 1.0]]))
        dynamic = _ScaleOnlyNode(torch.tensor([[9.0, 2.0, 1.0]]))
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": background, "dynamic": dynamic}))
        config = make_gaussian_cfg()
        config["max_scale_guard"] = {"enabled": True}
        strategy = self._strategy(config, nodes)

        first = strategy.serialization_state_dict_overrides(nodes)
        second = strategy.serialization_state_dict_overrides(nodes)

        torch.testing.assert_close(torch.exp(first["background.scales"]), torch.tensor([[3.0, 2.0, 1.0]]))
        torch.testing.assert_close(second["background.scales"], first["background.scales"])
        torch.testing.assert_close(background.get_scales(), torch.tensor([[4.0, 2.0, 1.0]]))
        torch.testing.assert_close(dynamic.get_scales(), torch.tensor([[9.0, 2.0, 1.0]]))
        self.assertNotIn("dynamic.scales", first)

    def test_rigid_guard_accounts_for_owning_track_scale(self) -> None:
        rigid = _RigidScaleOnlyNode(
            activated_scales=torch.tensor([[3.0, 1.0, 1.0], [3.0, 1.0, 1.0]]),
            cuboid_dims=torch.tensor([[4.0, 2.0, 1.0], [12.0, 2.0, 1.0]]),
            gaussian_cuboid_ids=torch.tensor([0, 1]),
            activated_track_scales=torch.tensor([[2.0, 1.0, 1.0], [2.0, 1.0, 1.0]]),
        )
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"dynamic_rigids": rigid}))
        config = make_gaussian_cfg()
        config["max_scale_guard"] = {
            "enabled": True,
            "global_limit_layer_ids": None,
            "absolute_max_scale_m": None,
        }
        strategy = self._strategy(config, nodes)
        strategy.maybe_initialize_buffers(nodes)

        strategy.enforce_max_scale_guard(nodes)

        base_scales = rigid.get_scales()
        assert rigid.track_scales is not None
        rendered_scales = base_scales * rigid.scale_activation(rigid.track_scales)[rigid.gaussian_cuboid_ids]
        torch.testing.assert_close(base_scales[:, 0], torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(rendered_scales[:, 0], torch.tensor([4.0, 6.0]))

    def test_rigid_metadata_is_cached_after_initialization(self) -> None:
        rigid = _RigidScaleOnlyNode(
            activated_scales=torch.ones((1, 3)),
            cuboid_dims=torch.tensor([[4.0, 2.0, 1.0]]),
            gaussian_cuboid_ids=torch.tensor([0]),
        )
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"dynamic_rigids": rigid}))
        config = make_gaussian_cfg()
        config["max_scale_guard"] = {"enabled": True, "global_limit_layer_ids": None}
        strategy = self._strategy(config, nodes)

        strategy.maybe_initialize_buffers(nodes)
        cached = strategy._rigid_scale_guard_cache["dynamic_rigids"].max_rendered_scale_m
        strategy.enforce_max_scale_guard(nodes)
        strategy.enforce_max_scale_guard(nodes)

        self.assertIs(strategy._rigid_scale_guard_cache["dynamic_rigids"].max_rendered_scale_m, cached)

    @unittest.skipUnless(torch.cuda.is_available(), "Rigid scale cache device transfer requires CUDA")
    def test_rigid_metadata_cache_follows_strategy_device_transfer(self) -> None:
        rigid = _RigidScaleOnlyNode(
            activated_scales=torch.tensor([[5.0, 1.0, 1.0]]),
            cuboid_dims=torch.tensor([[4.0, 2.0, 1.0]]),
            gaussian_cuboid_ids=torch.tensor([0]),
        )
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"dynamic_rigids": rigid}))
        config = make_gaussian_cfg()
        config["max_scale_guard"] = {
            "enabled": True,
            "global_limit_layer_ids": None,
            "absolute_max_scale_m": None,
        }
        strategy = self._strategy(config, nodes)
        strategy.maybe_initialize_buffers(nodes)
        cache = strategy._rigid_scale_guard_cache["dynamic_rigids"]
        self.assertEqual(cache.max_rendered_scale_m.device.type, "cpu")
        self.assertFalse(any(name.startswith("_rigid_scale_guard_cache.") for name in strategy.state_dict()))

        container = torch.nn.ModuleDict({"nodes": nodes, "strategy": strategy}).to("cuda")

        self.assertIs(container["strategy"], strategy)
        self.assertEqual(cache.max_rendered_scale_m.device, rigid.scales.device)
        strategy.enforce_max_scale_guard(nodes)
        torch.testing.assert_close(rigid.get_scales(), torch.tensor([[4.0, 1.0, 1.0]], device="cuda"))

    def test_rigid_metadata_rejects_invalid_inputs_during_initialization(self) -> None:
        cases = (
            (
                _RigidScaleOnlyNode(torch.ones((2, 3)), torch.ones((1, 3)), torch.tensor([0])),
                "does not match 2 Gaussian rows",
            ),
            (
                _RigidScaleOnlyNode(torch.ones((2, 3)), torch.ones((1, 3)), torch.tensor([0, 1])),
                "fall outside 1 tracks",
            ),
            (
                _RigidScaleOnlyNode(torch.ones((1, 3)), torch.tensor([[1.0, float("nan"), 1.0]]), torch.tensor([0])),
                "finite positive cuboid dimensions",
            ),
        )
        for rigid, message in cases:
            with self.subTest(message=message):
                nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"dynamic_rigids": rigid}))
                config = make_gaussian_cfg()
                config["max_scale_guard"] = {"enabled": True, "global_limit_layer_ids": None}
                strategy = self._strategy(config, nodes)
                with self.assertRaisesRegex(ValueError, message):
                    strategy.maybe_initialize_buffers(nodes)


class TestGsplatPerturbationDelegation(unittest.TestCase):
    def _make_strategy(self) -> Any:
        with (
            patch("gsplat.has_reloc", return_value=True),
            patch("gsplat.has_3dgs", return_value=True),
        ):
            return _make_concrete_mcmc_strategy(
                config=TypeAdapter(MCMCStrategyConfig | TamingGSMCMCStrategyConfig).validate_python(
                    make_gaussian_cfg()
                ),
                trainer_config=make_trainer_cfg(),
                init_from_datasource=False,
                gaussians_nodes=TypedModuleDict(),
            )

    def test_strategy_passes_nurec_policy_to_public_tensor_api(self):
        strategy = self._make_strategy()
        strategy.perturb_noise_lr = {"dynamic": 2.5}
        strategy.config.perturb.move_outside_of_cuboid = False
        node = object.__new__(RigidGaussianModel)
        torch.nn.Module.__init__(node)
        node.positions = torch.nn.Parameter(torch.zeros(3, 3))
        node.rotations = torch.nn.Parameter(torch.ones(3, 4))
        node.scales = torch.nn.Parameter(torch.zeros(3, 3))
        node.densities = torch.nn.Parameter(torch.zeros(3, 1))
        node.gaussian_cuboid_ids = torch.nn.Buffer(torch.tensor([1, 0, 1]))
        cuboid_dimensions = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        node.cuboid_tracks = cast(Any, SimpleNamespace(cuboids_dims=cuboid_dimensions))
        optimizer = torch.optim.SGD([{"params": [node.positions], "name": "model.positions", "lr": 0.2}])
        node.optimizers = [{"optimizer": optimizer}]
        generator = object()

        with (
            patch(
                "nurec.models.gaussians.strategies.mcmc.get_cuda_generator",
                return_value=generator,
            ),
            patch("nurec.models.gaussians.strategies.mcmc.perturb_positions") as operation,
        ):
            strategy.perturb_gaussians("dynamic", node)

        operation.assert_called_once()
        call = operation.call_args.kwargs
        self.assertIs(call["positions"], node.positions)
        self.assertIs(call["quats"], node.rotations)
        self.assertIs(call["scales"], node.scales)
        torch.testing.assert_close(call["opacities"], node.densities.squeeze(-1))
        self.assertEqual(call["noise_scale"], 0.5)
        self.assertIs(call["generator"], generator)
        torch.testing.assert_close(call["cuboid_dimensions"], cuboid_dimensions[node.gaussian_cuboid_ids])


@pytest.mark.parametrize("name", ["mcmc", "taminggs_mcmc"])
def test_mcmc_variants_use_their_concrete_runtime_factory(name: Literal["mcmc", "taminggs_mcmc"]) -> None:
    config: MCMCStrategyConfig | TamingGSMCMCStrategyConfig = TypeAdapter(
        MCMCStrategyConfig | TamingGSMCMCStrategyConfig
    ).validate_python(make_gaussian_cfg(name))
    with patch("gsplat.has_reloc", return_value=True):
        strategy = BaseGaussianStrategy.factory(name, config, make_trainer_cfg(), False, TypedModuleDict())
    assert type(strategy) is (TamingGSMCMCStrategy if name == "taminggs_mcmc" else MCMCStrategy)
    assert type(strategy.config) is type(config)
    assert hasattr(strategy, "_taminggs_initial_count") == (name == "taminggs_mcmc")


@pytest.mark.skipif(not torch.cuda.is_available() or not gsplat.has_adam(), reason="SelectiveAdam requires CUDA")
def test_taminggs_scores_use_actual_selective_adam_training_gradients() -> None:
    node = _TrackBufferNode(8, "selective_adam", "background")
    optimizer = cast(SelectiveAdam, node.optimizers[0]["optimizer"])
    optimizer.zero_grad()
    coefficients = torch.arange(1, 9, device="cuda", dtype=torch.float32)[:, None]
    (node.positions * coefficients).sum().backward()
    optimizer.set_visibility(torch.ones(8, device="cuda", dtype=torch.bool))
    optimizer.step()
    strategy = TestMCMCStrategy._make_strategy(make_gaussian_cfg("taminggs_mcmc"))
    assert node.positions.grad is not None
    torch.testing.assert_close(
        strategy._position_gradient("background", node), torch.linalg.vector_norm(node.positions.grad, dim=-1)
    )
    torch.testing.assert_close(
        strategy._position_gradient_ema(node, "background"),
        optimizer.state[node.positions]["exp_avg_sq"].mean(dim=-1).sqrt(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Compiled saliency requires CUDA")
@pytest.mark.parametrize("count", [0, 1, 17])
def test_compiled_taminggs_addition_matches_eager_with_nonfinite_components(count: int) -> None:
    matrix = torch.arange(count * 2, device="cuda", dtype=torch.float32).reshape(count, 2)
    if count:
        matrix[0, 0] = float("inf")
        matrix[-1, 1] = float("nan")
    medians = torch.tensor([2.0, 1.0], device="cuda")
    opacity = torch.arange(count, device="cuda", dtype=torch.float32) / max(count, 1)
    args = (matrix, medians, (2.0, 3.0), 10.0, 0.01, opacity, 0.1, (count,))
    eager = _finish_taminggs_addition_scores(*args, enable_torch_compile=False)
    compiled = _finish_taminggs_addition_scores(*args, enable_torch_compile=True)
    torch.testing.assert_close(compiled, eager)
