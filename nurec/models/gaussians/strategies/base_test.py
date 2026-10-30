# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from typing import cast
from unittest.mock import Mock, patch

import torch

from nurec.config.model import GsplatStrategyConfig
from nurec.models.gaussians.gaussians_model import BaseGaussianModel
from nurec.models.gaussians.strategies.base import BaseGaussianStrategy
from nurec.models.gaussians.strategies.test_utils import make_trainer_cfg
from nurec.models.nn_extensions import TypedModuleDict


class MockGaussianStrategy(BaseGaussianStrategy):
    def _update_step_train_batch_end_impl(
        self,
        epoch: int,
        global_step: int,
        batch,
        system,
        gaussians_nodes,
        **kwargs,
    ) -> None:
        pass


class TestBaseGaussianStrategy(unittest.TestCase):
    def _make_strategy(self, enforce_empty_cache_after_parameter_update: bool = True):
        config = GsplatStrategyConfig.model_validate(
            {
                "name": "gsplat",
                "enforce_empty_cache_after_parameter_update": enforce_empty_cache_after_parameter_update,
            }
        )
        return MockGaussianStrategy(
            config=config,
            trainer_config=make_trainer_cfg(),
            init_from_datasource=False,
            gaussians_nodes=TypedModuleDict(),
        )

    def test_base_class_instantiation(self):
        """Verify the minimal mock strategy still satisfies the abstract base contract."""
        # Use minimal gsplat config
        strategy = self._make_strategy()
        self.assertIsInstance(strategy, BaseGaussianStrategy)

    def test_parameter_update_empty_cache_control_defaults_on(self):
        config = GsplatStrategyConfig.model_validate({"name": "gsplat"})
        self.assertTrue(config.enforce_empty_cache_after_parameter_update)

    def test_unknown_strategy_conservatively_reports_topology_mutation(self):
        strategy = self._make_strategy()
        self.assertTrue(strategy.may_mutate_topology_at(0))
        self.assertTrue(strategy.may_mutate_topology_at(10_000))

        strategy.frozen = True
        self.assertFalse(strategy.may_mutate_topology_at(10_000))

    def test_update_param_with_optimizer_skips_scalar_state(self):
        """Ensure parameter replacement preserves scalar Adam state while updating tensors.

        MCMC relocation rewrites Gaussian parameters and moment buffers per row, but scalar
        optimizer bookkeeping like ``step`` should be copied unchanged.
        """
        strategy = self._make_strategy()
        param = torch.nn.Parameter(torch.arange(6.0).reshape(3, 2))
        optimizer = torch.optim.Adam([{"params": [param], "name": "layer.positions"}], lr=0.1)
        optimizer.state[param] = {
            "step": torch.tensor(3.0),
            "optimizer_bookkeeping": 7,
            "exp_avg": torch.ones_like(param),
            "exp_avg_sq": torch.full_like(param, 2.0),
        }

        class GaussianModelStub:
            def __init__(self):
                self.positions = param
                self.optimizers = [{"optimizer": optimizer}]

            def get_additional_buffers(self):
                return {}

        gaussian_model = GaussianModelStub()
        updated_states: list[tuple[str, str]] = []
        grad_enabled: list[bool] = []

        def update_param_fn(name: str, value: torch.Tensor) -> torch.Tensor:
            assert name == "positions"
            grad_enabled.append(torch.is_grad_enabled())
            return torch.nn.Parameter(value + 10.0)

        def update_optimizer_fn(name: str, key: str, value: torch.Tensor) -> torch.Tensor:
            updated_states.append((name, key))
            grad_enabled.append(torch.is_grad_enabled())
            return value + 1.0

        strategy._update_param_with_optimizer(
            cast(BaseGaussianModel, gaussian_model),
            update_param_fn,
            update_optimizer_fn,
        )

        new_param = gaussian_model.positions
        assert new_param is optimizer.param_groups[0]["params"][0]
        assert torch.equal(new_param, param.detach() + 10.0)
        assert updated_states == [("positions", "exp_avg"), ("positions", "exp_avg_sq")]
        assert torch.equal(optimizer.state[new_param]["step"], torch.tensor(3.0))
        assert optimizer.state[new_param]["optimizer_bookkeeping"] == 7
        assert torch.equal(optimizer.state[new_param]["exp_avg"], torch.full_like(param, 2.0))
        assert torch.equal(optimizer.state[new_param]["exp_avg_sq"], torch.full_like(param, 3.0))
        # The rewrite rebuilds every per-Gaussian tensor of a layer, so it must not build a graph.
        assert not any(grad_enabled), "parameter rewrite ran with grad enabled"

    def test_parameter_update_empty_cache_control(self):
        class GaussianModelStub:
            optimizers = []

            @staticmethod
            def get_additional_buffers():
                return {}

        gaussian_model = cast(BaseGaussianModel, GaussianModelStub())

        for enforce_empty_cache, expected_calls in ((True, 1), (False, 0)):
            with self.subTest(enforce_empty_cache=enforce_empty_cache):
                strategy = self._make_strategy(enforce_empty_cache)
                with patch.object(torch.cuda, "empty_cache") as empty_cache:
                    strategy._update_param_with_optimizer(gaussian_model, Mock(), Mock())
                self.assertEqual(empty_cache.call_count, expected_calls)


class TestPermuteGaussianRows(unittest.TestCase):
    @staticmethod
    def _buffers(**entries: torch.Tensor) -> torch.nn.ParameterDict:
        return torch.nn.ParameterDict(
            {key: torch.nn.Parameter(value, requires_grad=False) for key, value in entries.items()}
        )

    def test_row_aligned_entries_follow_the_permutation(self):
        buffers = self._buffers(background=torch.tensor([[10.0], [20.0], [30.0]]))
        order = torch.tensor([2, 0, 1])

        BaseGaussianStrategy._permute_layer_buffers(buffers, {"background": order})

        assert torch.equal(buffers["background"], torch.tensor([[30.0], [10.0], [20.0]]))

    def test_lazy_entries_are_skipped(self):
        buffers = torch.nn.ParameterDict({"background": torch.nn.UninitializedParameter(requires_grad=False)})

        BaseGaussianStrategy._permute_layer_buffers(buffers, {"background": torch.tensor([1, 0])})

        assert isinstance(buffers["background"], torch.nn.UninitializedParameter)

    def test_entries_whose_leading_dim_does_not_match_are_skipped(self):
        # A buffer that is not row-aligned with this layer is not ours to reorder.
        buffers = self._buffers(background=torch.arange(5.0))

        BaseGaussianStrategy._permute_layer_buffers(buffers, {"background": torch.tensor([2, 0, 1])})

        assert torch.equal(buffers["background"], torch.arange(5.0))
