# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import MagicMock, patch

import apex
import gsplat
import pytest
import torch

from omegaconf import DictConfig

from nurec.config.model import MCMCStrategyConfig
from nurec.models.gaussians.gaussians_model import BaseGaussianModel, RigidGaussianModel
from nurec.models.gaussians.strategies.mcmc import MCMCStrategy
from nurec.models.gaussians.strategies.test_utils import make_trainer_cfg
from nurec.models.nn_extensions import TypedModuleDict
from nurec.utils.optim import SelectiveAdam


def make_gaussian_cfg() -> DictConfig:
    return DictConfig(
        {
            "name": "mcmc",
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

    def raise_if_mcmc_publication_failed(self) -> None:
        pass


class _StrategyNode(torch.nn.Module):
    def __init__(self, activated_densities: torch.Tensor, *, publication_failed: bool = False) -> None:
        super().__init__()
        self.activated_densities = activated_densities
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

    def raise_if_mcmc_publication_failed(self) -> None:
        if self.publication_failed:
            raise RuntimeError("MCMC publication previously failed")


class TestMCMCStrategy(unittest.TestCase):
    @staticmethod
    def _make_strategy(
        config: DictConfig | None = None,
        nodes: TypedModuleDict | None = None,
    ) -> MCMCStrategy:
        return MCMCStrategy(
            config=MCMCStrategyConfig.model_validate(config or make_gaussian_cfg()),
            trainer_config=make_trainer_cfg(),
            init_from_datasource=False,
            gaussians_nodes=nodes or TypedModuleDict(),
        )

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
            MCMCStrategy(
                MCMCStrategyConfig.model_validate(config),
                make_trainer_cfg().model_copy(update={"world_size": 2}),
                False,
                TypedModuleDict(),
            )
        config.add.fixed_count_layer_ids = ["barriers"]
        with self.assertRaisesRegex(ValueError, "cannot combine"):
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
                strategy = MCMCStrategy(
                    MCMCStrategyConfig.model_validate(config),
                    trainer_config,
                    False,
                    TypedModuleDict(),
                )
            local_quotas.append(strategy.config.relocate.min_gaussians_per_track["vehicle_lights"])

        self.assertEqual(local_quotas, [8, 8])
        self.assertEqual(sum(local_quotas), 16)

    def test_track_quota_adjustment_does_not_mutate_input_config(self) -> None:
        config = make_gaussian_cfg()
        config.relocate.same_track_layer_ids = ["vehicle_lights"]
        config.relocate.min_gaussians_per_track = {"vehicle_lights": 16}
        source_config = MCMCStrategyConfig.model_validate(config)
        trainer_config = make_trainer_cfg().model_copy(update={"world_size": 2})

        with (
            patch("nurec.utils.trainer.rank_zero_only.rank", 0),
            patch("gsplat.has_reloc", return_value=True),
        ):
            first = MCMCStrategy(source_config, trainer_config, False, TypedModuleDict())
            second = MCMCStrategy(source_config, trainer_config, False, TypedModuleDict())

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
                strategy = MCMCStrategy(
                    MCMCStrategyConfig.model_validate(config),
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
        nodes: TypedModuleDict = TypedModuleDict()
        selected = SimpleNamespace(layer_num_gaussians={})

        with patch("nurec.models.gaussians.strategies.mcmc.SelectedGaussianNodes", return_value=selected) as factory:
            strategy.add_new_gaussians(nodes)

        factory.assert_called_once_with(nodes, ["vehicle_lights"])

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
                side_effect=lambda *_args: operation_order.append("relocate"),
            ) as relocate,
            patch.object(
                strategy,
                "add_new_gaussians",
                side_effect=lambda *_args: operation_order.append("add"),
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
        config = make_gaussian_cfg()
        config.exclude_layer_ids = ["background"]
        nodes = TypedModuleDict(cast(Any, {"background": _StrategyNode(torch.ones(4))}))
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy(config, nodes)

        with (
            patch("nurec.models.gaussians.strategies.mcmc.get_cuda_generator") as get_rng,
            patch("nurec.models.gaussians.strategies.mcmc.gsplat_relocate_gaussians") as operation,
        ):
            strategy.relocate_gaussians(nodes)

        get_rng.assert_not_called()
        operation.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "MCMC policy tests require CUDA")
    def test_relocation_global_noops_do_not_request_rng(self):
        with patch("gsplat.has_reloc", return_value=True):
            strategy = self._make_strategy()

        for densities in (torch.full((3,), 0.5, device="cuda"), torch.zeros(3, device="cuda")):
            node = _StrategyNode(densities)
            nodes = TypedModuleDict(cast(Any, {"background": node}))
            with (
                patch("nurec.models.gaussians.strategies.mcmc.get_cuda_generator") as get_rng,
                patch("nurec.models.gaussians.strategies.mcmc.gsplat_relocate_gaussians") as operation,
            ):
                strategy.relocate_gaussians(nodes)

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

    def test_visibility_transition_preserves_nurec_layer_order_and_deduplication(self):
        class _VisibilityNode(torch.nn.Module):
            def __init__(self, count: int):
                super().__init__()
                self.positions = torch.nn.Parameter(torch.zeros(count, 3))

            @property
            def device(self) -> torch.device:
                return self.positions.device

            def get_num_gaussians(self) -> int:
                return self.positions.shape[0]

        config = make_gaussian_cfg()
        config.relocate.max_invisible_steps = 3
        nodes = TypedModuleDict(
            cast(
                Any,
                {
                    "first": _VisibilityNode(2),
                    "second": _VisibilityNode(1),
                },
            )
        )
        strategy = MCMCStrategy(
            config=MCMCStrategyConfig.model_validate(config),
            trainer_config=make_trainer_cfg(),
            init_from_datasource=False,
            gaussians_nodes=nodes,
        )
        strategy.maybe_initialize_buffers(nodes)
        first_results = object()

        strategy.update_visibility_counters(
            results=first_results,
            visibility=torch.tensor([1.0, 0.0, 0.0]),
            gaussians_nodes=nodes,
        )
        strategy.update_visibility_counters(
            results=first_results,
            visibility=torch.zeros(3),
            gaussians_nodes=nodes,
        )
        strategy.update_visibility_counters(
            results=object(),
            visibility=torch.tensor([0.0, 1.0, 1.0]),
            gaussians_nodes=nodes,
        )

        torch.testing.assert_close(
            strategy.invisible_steps["first"],
            torch.tensor([1, 0], dtype=torch.int32),
        )
        torch.testing.assert_close(
            strategy.invisible_steps["second"],
            torch.tensor([0], dtype=torch.int32),
        )


class TestMCMCPermuteGaussianRows(unittest.TestCase):
    def test_invisible_steps_follows_the_permutation(self):
        # invisible_steps is row-aligned but lives on the strategy, so a reorder inside a node cannot reach it.
        # Left behind, relocation recycles by another Gaussian's history.
        strategy = MCMCStrategy(
            config=MCMCStrategyConfig.model_validate(make_gaussian_cfg()),
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
        self.config = cast(Any, SimpleNamespace(density_activation="sigmoid", scale_activation="exp"))
        self.density_activation = cast(Any, torch.sigmoid)
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
def test_independent_gpu_pool_respects_cap_and_never_borrows_background_donors() -> None:
    """Real GPU pools honor separate caps and cannot repopulate barriers from background."""
    nodes: TypedModuleDict[BaseGaussianModel] = TypedModuleDict(
        {name: _TrackBufferNode(40, "selective_adam", name) for name in ("background", "barriers")}
    )
    config = make_gaussian_cfg()
    config.add.max_n_gaussians = 42
    config.independent_layer_max_n_gaussians = {"barriers": 41}
    strategy = TestMCMCStrategy._make_strategy(config, nodes).to("cuda")
    strategy.add_new_gaussians(nodes)
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
def test_topology_keeps_production_adam_state_and_track_buffer_aligned(
    optimizer_variant: OptimizerVariant,
    operation: str,
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
    strategy = TestMCMCStrategy._make_strategy(config, nodes).to("cuda")

    if operation == "addition":
        strategy.add_new_gaussians(nodes)
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
    def _strategy(config: DictConfig, nodes: TypedModuleDict, *, max_steps: int = 30_000) -> MCMCStrategy:
        trainer_config = make_trainer_cfg().model_copy(update={"max_steps": max_steps})
        return MCMCStrategy(
            config=MCMCStrategyConfig.model_validate(config),
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
    def _make_strategy(self) -> MCMCStrategy:
        with (
            patch("gsplat.has_reloc", return_value=True),
            patch("gsplat.has_3dgs", return_value=True),
        ):
            return MCMCStrategy(
                config=MCMCStrategyConfig.model_validate(make_gaussian_cfg()),
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
