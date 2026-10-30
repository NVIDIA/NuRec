# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch

from torch import nn

from nurec.config.model import GsplatStrategyConfig
from nurec.models.gaussians.strategies.gsplat import GSplatStrategy
from nurec.models.gaussians.strategies.test_utils import make_trainer_cfg
from nurec.models.nn_extensions import TypedModuleDict


class _GaussianNode(torch.nn.Module):
    def __init__(self, count: int = 1) -> None:
        super().__init__()
        self.positions = torch.nn.Parameter(torch.zeros((count, 3)))

    def get_num_gaussians(self) -> int:
        return self.positions.shape[0]


def _make_strategy() -> GSplatStrategy:
    return GSplatStrategy(
        config=GsplatStrategyConfig.model_validate({"name": "gsplat"}),
        trainer_config=make_trainer_cfg(),
        init_from_datasource=False,
        gaussians_nodes=TypedModuleDict(),
    )


class TestGSplatStrategy(unittest.TestCase):
    def test_gsplat_strategy_instantiation(self):
        strategy = _make_strategy()
        self.assertIsInstance(strategy, GSplatStrategy)
        self.assertFalse(strategy.may_mutate_topology_at(1))

    def test_topology_mutation_schedule_ignores_rank_local_layout(self) -> None:
        config = GsplatStrategyConfig.model_validate(
            {
                "name": "gsplat",
                "densify": {"start_iteration": 2, "end_iteration": 12, "frequency": 2},
                "prune": {"start_iteration": 3, "end_iteration": 12, "frequency": 3},
            }
        )
        layouts: tuple[TypedModuleDict[Any], TypedModuleDict[Any]] = (
            TypedModuleDict(cast(Any, {"background": _GaussianNode(1)})),
            TypedModuleDict(cast(Any, {"road": _GaussianNode(7), "vehicle": _GaussianNode(0)})),
        )
        strategies = tuple(
            GSplatStrategy(
                config=config,
                trainer_config=make_trainer_cfg(),
                init_from_datasource=False,
                gaussians_nodes=nodes,
            )
            for nodes in layouts
        )
        steps = (1, 2, 3, 4, 6, 9, 12)

        decisions = tuple(tuple(strategy.may_mutate_topology_at(step) for step in steps) for strategy in strategies)
        self.assertEqual(decisions[0], decisions[1])
        self.assertEqual(decisions[0], (False, False, False, True, True, True, False))

        for strategy in strategies:
            strategy.frozen = True
            self.assertFalse(any(strategy.may_mutate_topology_at(step) for step in steps))

    def test_topology_mutation_schedule_matches_batch_end_gates(self) -> None:
        config = GsplatStrategyConfig.model_validate(
            {
                "name": "gsplat",
                "densify": {"start_iteration": 2, "end_iteration": 12, "frequency": 2},
                "prune": {"start_iteration": 3, "end_iteration": 12, "frequency": 3},
                "reset_density": {"start_iteration": 0, "end_iteration": 12, "frequency": 5},
            }
        )
        background = _GaussianNode()
        road = _GaussianNode()
        nodes: TypedModuleDict = TypedModuleDict(cast(Any, {"background": background, "road": road}))
        strategy = GSplatStrategy(
            config=config,
            trainer_config=make_trainer_cfg(),
            init_from_datasource=False,
            gaussians_nodes=nodes,
        )
        batch = SimpleNamespace(rendering=SimpleNamespace(camera=object(), lidar=None))
        system = SimpleNamespace(
            model=SimpleNamespace(
                collect_gaussian_parameters=lambda *_args, **_kwargs: {
                    "positions": torch.cat((background.positions, road.positions))
                }
            )
        )
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
            patch.object(strategy, "update_gradient_buffers"),
            patch.object(
                strategy,
                "densify_gaussians",
                side_effect=lambda layer_id, _node: operation_order.append(f"densify:{layer_id}"),
            ) as densify,
            patch.object(
                strategy,
                "prune_gaussians_opacity",
                side_effect=lambda layer_id, _node: operation_order.append(f"prune:{layer_id}"),
            ) as prune,
            patch.object(strategy, "reset_density") as reset_density,
        ):
            for step, expected in expected_operations.items():
                with self.subTest(step=step):
                    strategy.update_step_train_batch_end(0, step, cast(Any, batch), system, nodes)

                    actual = (densify.called, prune.called, reset_density.called)
                    self.assertEqual(actual, expected)
                    self.assertEqual(
                        operation_order,
                        [
                            f"{name}:{layer_id}"
                            for layer_id in ("background", "road")
                            for name, called in zip(("densify", "prune"), expected[:2], strict=True)
                            if called
                        ],
                    )
                    self.assertEqual(strategy.may_mutate_topology_at(step), densify.called or prune.called)
                    operation_order.clear()
                    densify.reset_mock()
                    prune.reset_mock()
                    reset_density.reset_mock()

            strategy.frozen = True
            strategy.update_step_train_batch_end(0, 6, cast(Any, batch), system, nodes)
            self.assertFalse(strategy.may_mutate_topology_at(6))
            densify.assert_not_called()
            prune.assert_not_called()
            reset_density.assert_not_called()


class TestGSplatPermuteGaussianRows(unittest.TestCase):
    def test_both_accumulators_follow_the_permutation(self):
        strategy = _make_strategy()
        # Distinct contents per buffer, so permuting the same one twice is caught.
        strategy.densify_grad_norm_accum["background"] = nn.Parameter(
            torch.tensor([[10.0], [20.0], [30.0]]), requires_grad=False
        )
        strategy.densify_grad_norm_denom["background"] = nn.Parameter(
            torch.tensor([[1.0], [2.0], [3.0]]), requires_grad=False
        )

        strategy.permute_gaussian_rows({"background": torch.tensor([2, 0, 1])})

        assert torch.equal(strategy.densify_grad_norm_accum["background"], torch.tensor([[30.0], [10.0], [20.0]]))
        assert torch.equal(strategy.densify_grad_norm_denom["background"], torch.tensor([[3.0], [1.0], [2.0]]))
