# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch

from torch import nn

from nurec.config.model import GsplatStrategyConfig
from nurec.models.gaussians.gaussians_model import BaseGaussianModel
from nurec.models.gaussians.strategies.gsplat import GSplatStrategy
from nurec.models.gaussians.strategies.test_utils import make_trainer_cfg
from nurec.models.nn_extensions import TypedModuleDict


class _GaussianNode(torch.nn.Module):
    def __init__(self, count: int = 1) -> None:
        super().__init__()
        self.positions = torch.nn.Parameter(torch.zeros((count, 3)))
        self.config = SimpleNamespace(density_activation="sigmoid", scale_activation="exp")

    def raise_if_publication_failed(self) -> None:
        BaseGaussianModel.raise_if_publication_failed(cast(BaseGaussianModel, self))

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
                side_effect=lambda layer_id, _node, _step: operation_order.append(f"densify:{layer_id}"),
            ) as densify,
            patch.object(
                strategy,
                "prune_gaussians_opacity",
                side_effect=lambda layer_id, _node, _step: operation_order.append(f"prune:{layer_id}"),
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


class _OpsGaussianNode(torch.nn.Module):
    row_ids: torch.Tensor

    def raise_if_publication_failed(self) -> None:
        BaseGaussianModel.raise_if_publication_failed(cast(BaseGaussianModel, self))

    def mark_publication_failed(self) -> None:
        BaseGaussianModel.mark_publication_failed(cast(BaseGaussianModel, self))

    def validate_external_gaussian_topology_edit(self, operation: str) -> None:
        pass

    def _reinstall_grad_hooks(self) -> None:
        self.hook_refresh_count += 1

    def __init__(self, device: str = "cpu", count: int = 3) -> None:
        super().__init__()
        self.config = SimpleNamespace(density_activation="sigmoid", scale_activation="exp")
        self.scale_activation = torch.exp
        self.scale_activation_inv = torch.log
        self.device = torch.device(device)
        self.scene_extent = 1.0
        self.hook_refresh_count = 0
        self.positions = nn.Parameter(torch.arange(count * 3, device=device, dtype=torch.float32).reshape(count, 3))
        self.scales = nn.Parameter(torch.tensor([[0.2, 0.4, 0.8]] * count, device=device).reshape(count, 3).log())
        self.rotations = nn.Parameter(torch.tensor([[0.2, 0.3, 0.4, 0.5]] * count, device=device).reshape(count, 4))
        self.densities = nn.Parameter(torch.logit(torch.linspace(0.1, 0.9, count, device=device).reshape(count, 1)))
        self.register_buffer("row_ids", torch.arange(count, device=device))
        optimizer = torch.optim.Adam(
            [
                {"name": name, "params": [getattr(self, name)]}
                for name in ("positions", "scales", "rotations", "densities")
            ],
            lr=0.001,
        )
        torch.stack([p.square().sum() for p in self.parameters()]).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.optimizers = [{"optimizer": optimizer}]

    def get_num_gaussians(self) -> int:
        return len(self.positions)

    def get_densities(self) -> torch.Tensor:
        return self.densities.sigmoid()

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_scales(self) -> torch.Tensor:
        return self.scales.exp()

    def get_additional_buffers(self) -> dict[str, torch.Tensor]:
        return {"row_ids": self.row_ids}


def _make_ops_strategy(model: _OpsGaussianNode, **config_overrides: Any) -> GSplatStrategy:
    strategy = GSplatStrategy(
        config=GsplatStrategyConfig.model_validate(
            {
                "name": "gsplat",
                "exclude_layer_ids": ["excluded"],
                "enforce_empty_cache_after_parameter_update": False,
                "densify": {"start_iteration": 0, "end_iteration": 10, "frequency": 2},
                "prune": {"start_iteration": 0, "end_iteration": 10, "frequency": 3, "density_threshold": 0.2},
                "reset_density": {"start_iteration": 0, "end_iteration": 10, "frequency": 5, "new_max_density": 0.3},
                **config_overrides,
            }
        ),
        trainer_config=make_trainer_cfg(),
        init_from_datasource=False,
        gaussians_nodes=TypedModuleDict(cast(Any, {"included": model})),
    )
    strategy.densify_grad_norm_accum["included"] = nn.Parameter(
        torch.ones((model.get_num_gaussians(), 1), device=model.device), requires_grad=False
    )
    strategy.densify_grad_norm_denom["included"] = nn.Parameter(
        torch.ones((model.get_num_gaussians(), 1), device=model.device, dtype=torch.int32), requires_grad=False
    )
    return strategy


class TestGSplatOpsIntegration(unittest.TestCase):
    def assert_optimizer_bound(self, model: _OpsGaussianNode) -> None:
        optimizer = model.optimizers[0]["optimizer"]
        for group in optimizer.param_groups:
            parameter = getattr(model, group["name"])
            self.assertIs(group["params"][0], parameter)
            self.assertEqual(optimizer.state[parameter]["exp_avg"].shape, parameter.shape)
        torch.stack([p.square().sum() for p in model.parameters()]).sum().backward()
        optimizer.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_gradient_accumulation_updates_existing_buffers(self) -> None:
        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        self.assertIsNotNone(strategy._gsplat_default)
        self.assertEqual(strategy._gsplat_default.gradient_mode, "3d")
        model.positions.grad = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # Distances must come from rendered positions, not the stored parameter.
        rendered_positions = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [9.0, 9.0, 9.0]])
        rendering_data = SimpleNamespace(rays=torch.tensor([[[[0.0] * 6, [100.0] * 6]]]))
        sums = strategy.densify_grad_norm_accum["included"]
        counts = strategy.densify_grad_norm_denom["included"]
        parameter = model.positions.detach().clone()
        gradient = model.positions.grad.clone()
        for step in (1, 2):
            strategy.update_gradient_buffers(
                "included", model.positions, rendered_positions, cast(Any, rendering_data), step
            )
        self.assertIs(strategy.densify_grad_norm_accum["included"], sums)
        self.assertIs(strategy.densify_grad_norm_denom["included"], counts)
        torch.testing.assert_close(sums, torch.tensor([[11.0], [1.0], [1.0]]))
        torch.testing.assert_close(counts, torch.tensor([[3], [3], [1]], dtype=torch.int32))
        torch.testing.assert_close(model.positions, parameter)
        torch.testing.assert_close(model.positions.grad, gradient)

        restored = _make_ops_strategy(_OpsGaussianNode())
        restored.load_state_dict(strategy.state_dict())
        torch.testing.assert_close(restored.densify_grad_norm_accum["included"], sums)
        torch.testing.assert_close(restored.densify_grad_norm_denom["included"], counts)

    def test_gradient_accumulation_rebinds_after_pruning(self) -> None:
        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        old_sums = strategy.densify_grad_norm_accum["included"]
        strategy.prune_gaussians_opacity("included", cast(BaseGaussianModel, model))
        self.assertEqual(model.get_num_gaussians(), 2)
        model.positions.grad = torch.ones_like(model.positions)
        strategy.update_gradient_buffers(
            "included",
            model.positions,
            torch.ones_like(model.positions),
            cast(Any, SimpleNamespace(rays=torch.zeros(1, 1, 1, 6))),
            1,
        )
        torch.testing.assert_close(strategy.densify_grad_norm_accum["included"], torch.full((2, 1), 2.5))
        torch.testing.assert_close(
            strategy.densify_grad_norm_denom["included"], torch.full((2, 1), 2, dtype=torch.int32)
        )
        torch.testing.assert_close(old_sums, torch.ones(3, 1))

    def test_batch_accumulation_preserves_excluded_layer_and_row_offsets(self) -> None:
        model = _OpsGaussianNode()
        excluded_model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        strategy.densify_grad_norm_accum["excluded"] = nn.Parameter(torch.ones(3, 1), requires_grad=False)
        strategy.densify_grad_norm_denom["excluded"] = nn.Parameter(
            torch.ones(3, 1, dtype=torch.int32), requires_grad=False
        )
        nodes: TypedModuleDict[Any] = TypedModuleDict(cast(Any, {"excluded": excluded_model, "included": model}))
        model.positions.grad = torch.ones_like(model.positions)
        batch = SimpleNamespace(
            rendering=SimpleNamespace(camera=SimpleNamespace(rays=torch.zeros(1, 1, 1, 6)), lidar=None)
        )
        system = SimpleNamespace(
            model=SimpleNamespace(
                collect_gaussian_parameters=lambda *_a, **_kw: {
                    "positions": torch.cat([torch.full((3, 3), 100.0), torch.ones(3, 3)])
                }
            )
        )
        strategy.update_step_train_batch_end(0, 1, cast(Any, batch), system, nodes)
        torch.testing.assert_close(strategy.densify_grad_norm_accum["included"], torch.full((3, 1), 2.5))
        torch.testing.assert_close(
            strategy.densify_grad_norm_denom["included"], torch.full((3, 1), 2, dtype=torch.int32)
        )
        torch.testing.assert_close(strategy.densify_grad_norm_accum["excluded"], torch.ones(3, 1))
        torch.testing.assert_close(strategy.densify_grad_norm_denom["excluded"], torch.ones(3, 1, dtype=torch.int32))

    def test_empty_layer_accumulation(self) -> None:
        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        model.positions = nn.Parameter(torch.empty(0, 3))
        model.positions.grad = torch.empty_like(model.positions)
        strategy.densify_grad_norm_accum["included"] = nn.Parameter(torch.empty(0, 1), requires_grad=False)
        strategy.densify_grad_norm_denom["included"] = nn.Parameter(
            torch.empty(0, 1, dtype=torch.int32), requires_grad=False
        )
        strategy.update_gradient_buffers(
            "included",
            model.positions,
            model.positions,
            cast(Any, SimpleNamespace(rays=torch.zeros(1, 1, 1, 6))),
            1,
        )
        self.assertEqual(strategy.densify_grad_norm_accum["included"].shape, (0, 1))
        self.assertEqual(strategy.densify_grad_norm_denom["included"].shape, (0, 1))

    def test_default_strategy_uses_public_operations(self) -> None:
        import gsplat.strategy as gsplat_api

        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            with self.subTest(device=device):
                model = _OpsGaussianNode(device=device)
                strategy = _make_ops_strategy(model)
                model.positions.grad = torch.ones_like(model.positions)
                with patch.object(gsplat_api, "update", wraps=gsplat_api.update) as public:
                    strategy.update_gradient_buffers(
                        "included",
                        model.positions,
                        torch.ones_like(model.positions),
                        cast(Any, SimpleNamespace(rays=torch.zeros(1, 1, 1, 6, device=device))),
                        1,
                    )
                public.assert_called_once()
                torch.testing.assert_close(
                    strategy.densify_grad_norm_accum["included"], torch.full((3, 1), 2.5, device=device)
                )
                with patch.object(gsplat_api, "prune", wraps=gsplat_api.prune) as public:
                    strategy.prune_gaussians_opacity("included", cast(BaseGaussianModel, model))
                public.assert_called_once()
                torch.testing.assert_close(model.row_ids, torch.tensor([1, 2], device=device))

    def test_exclusion(self) -> None:
        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        self.assertTrue(strategy._layer_is_excluded("excluded"))
        self.assertFalse(strategy._layer_is_excluded("included"))
        nodes: TypedModuleDict[Any] = TypedModuleDict(cast(Any, {"excluded": model}))
        batch = SimpleNamespace(rendering=SimpleNamespace(camera=object(), lidar=None))
        system = SimpleNamespace(
            model=SimpleNamespace(
                collect_gaussian_parameters=lambda *_a, **_kw: {
                    "positions": model.positions,
                }
            )
        )
        with (
            patch.object(strategy, "update_gradient_buffers") as update,
            patch.object(strategy, "densify_gaussians") as densify,
            patch.object(strategy, "prune_gaussians_opacity") as prune,
            patch.object(strategy, "reset_density") as reset,
        ):
            for step in (5, 6):
                strategy.update_step_train_batch_end(0, step, cast(Any, batch), system, nodes)
            for operation in (update, densify, prune, reset):
                operation.assert_not_called()
        self.assert_optimizer_bound(model)

    def test_prune_nan_and_preserve_grouped_optimizer_and_history(self) -> None:
        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        old = model.positions
        history = torch.tensor([[2.0], [3.0], [4.0]])
        strategy.densify_grad_norm_accum["included"].copy_(history)
        strategy.densify_grad_norm_denom["included"].copy_(history.int())
        old_moments = model.optimizers[0]["optimizer"].state[old]["exp_avg"].clone()
        with torch.no_grad():
            model.densities[1] = float("nan")
        strategy.prune_gaussians_opacity("included", cast(BaseGaussianModel, model))
        torch.testing.assert_close(model.row_ids, torch.tensor([2]))
        torch.testing.assert_close(model.optimizers[0]["optimizer"].state[model.positions]["exp_avg"], old_moments[2:])
        self.assertEqual(strategy.densify_grad_norm_accum["included"].shape, (1, 1))
        torch.testing.assert_close(strategy.densify_grad_norm_accum["included"], history[2:])
        torch.testing.assert_close(strategy.densify_grad_norm_denom["included"], history[2:].int())
        self.assertNotIn(old, model.optimizers[0]["optimizer"].state)
        self.assert_optimizer_bound(model)

    def test_reset_independent_cap_and_moments(self) -> None:
        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        old_positions = model.positions
        history = strategy.densify_grad_norm_accum["included"]
        optimizer = model.optimizers[0]["optimizer"]
        step = optimizer.state[model.densities]["step"].clone()
        expected = model.get_densities().clamp(max=0.3)
        strategy.reset_density("included", cast(BaseGaussianModel, model))
        torch.testing.assert_close(model.get_densities(), expected)
        self.assertIs(model.positions, old_positions)
        self.assertIs(strategy.densify_grad_norm_accum["included"], history)
        self.assertEqual(torch.count_nonzero(optimizer.state[model.densities]["exp_avg"]), 0)
        torch.testing.assert_close(optimizer.state[model.densities]["step"], step)
        self.assert_optimizer_bound(model)

    def test_pruning_keeps_expected_rows_when_all_or_none_are_removed(self) -> None:
        for opacity, count in ((0.01, 0), (0.9, 3)):
            with self.subTest(opacity=opacity):
                model = _OpsGaussianNode()
                with torch.no_grad():
                    model.densities.fill_(torch.logit(torch.tensor(opacity)).item())
                expected = {name: value[:count].detach().clone() for name, value in model.state_dict().items()}
                strategy = _make_ops_strategy(model)
                strategy.prune_gaussians_opacity("included", cast(BaseGaussianModel, model))
                self.assertEqual(model.get_num_gaussians(), count)
                for name, value in expected.items():
                    torch.testing.assert_close(getattr(model, name), value)
                for buffers in (strategy.densify_grad_norm_accum, strategy.densify_grad_norm_denom):
                    torch.testing.assert_close(buffers["included"], torch.ones_like(buffers["included"]))
                    self.assertEqual(buffers["included"].shape, (count, 1))
                self.assert_optimizer_bound(model)

    def test_staging_failure_leaves_live_state_unchanged(self) -> None:
        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        optimizer = model.optimizers[0]["optimizer"]
        old_parameters = tuple(model.parameters())
        old_groups = [group["params"][0] for group in optimizer.param_groups]
        old_moments = [optimizer.state[p]["exp_avg"].clone() for p in old_groups]

        def fail(params, optimizers):
            from gsplat.strategy.ops import reset_opa

            reset_opa(params, optimizers, {}, value=0.3)
            raise RuntimeError("staged operation failed")

        with self.assertRaisesRegex(RuntimeError, "staged operation failed"):
            strategy._run_gsplat_operation(cast(BaseGaussianModel, model), fail)
        for before, after in zip(old_parameters, model.parameters(), strict=True):
            self.assertIs(before, after)
        for parameter, group, moments in zip(old_groups, optimizer.param_groups, old_moments, strict=True):
            self.assertIs(group["params"][0], parameter)
            torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], moments)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_split_publishes_nre_state_and_forwards_generator(self) -> None:
        import gsplat.strategy as gsplat_api

        from nurec.utils.cuda_rng import get_cuda_generator

        model = _OpsGaussianNode("cuda")
        child_count = 3
        strategy = _make_ops_strategy(
            model,
            densify={"start_iteration": 0, "end_iteration": 10, "frequency": 2, "split_n_gaussians": child_count},
        )
        self.assertEqual(strategy._gsplat_default.split_n_gaussians, child_count)
        optimizer = model.optimizers[0]["optimizer"]
        old_parameters = {group["name"]: group["params"][0] for group in optimizer.param_groups}
        old_states = {
            name: {key: value.clone() for key, value in optimizer.state[parameter].items()}
            for name, parameter in old_parameters.items()
        }
        generator = get_cuda_generator(torch.device("cuda"))
        initial = generator.get_state()
        default = torch.cuda.get_rng_state()
        operation = gsplat_api.densify

        def check_mapping(*args: Any, **kwargs: Any) -> Any:
            # NRE stores WXYZ quaternions and log-scales, matching gsplat's layout.
            # Check the bridge passes the original fields without conversion.
            for alias, name in (
                ("means", "positions"),
                ("quats", "rotations"),
                ("scales", "scales"),
                ("opacities", "densities"),
            ):
                self.assertIs(args[1][alias], old_parameters[name])
            self.assertIs(kwargs["generator"], generator)
            return operation(*args, **kwargs)

        try:
            sources = torch.tensor([1, 0, 0, 0, 2, 2, 2], device="cuda")
            strategy.densify_grad_norm_accum["included"].copy_(torch.tensor([[1.0], [0.0], [1.0]], device="cuda"))
            with patch.object(gsplat_api, "densify", side_effect=check_mapping) as public:
                strategy.densify_gaussians("included", cast(BaseGaussianModel, model))
            public.assert_called_once()
            torch.testing.assert_close(model.row_ids, sources)
            for name in ("rotations", "densities"):
                torch.testing.assert_close(getattr(model, name), old_parameters[name][sources])
            for group in optimizer.param_groups:
                name = group["name"]
                parameter = group["params"][0]
                self.assertNotIn(old_parameters[name], optimizer.state)
                for key, old_state in old_states[name].items():
                    actual = optimizer.state[parameter][key]
                    if old_state.ndim == 0:
                        torch.testing.assert_close(actual, old_state)
                    else:
                        torch.testing.assert_close(actual[:1], old_state[1:2])
                        self.assertEqual(torch.count_nonzero(actual[1:]), 0)
            for history in (strategy.densify_grad_norm_accum, strategy.densify_grad_norm_denom):
                self.assertEqual(history["included"].shape, (1 + 2 * child_count, 1))
                self.assertEqual(torch.count_nonzero(history["included"]), 0)
            self.assertFalse(torch.equal(generator.get_state(), initial))
            torch.testing.assert_close(torch.cuda.get_rng_state(), default)
            self.assert_optimizer_bound(model)
        finally:
            generator.set_state(initial)

    def test_no_selection_preserves_parameter_and_buffer_identity(self) -> None:
        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        strategy.densify_grad_norm_accum["included"].fill_(0.0001)
        old_history = strategy.densify_grad_norm_accum["included"]
        old = dict(model.named_parameters())
        row_ids = model.row_ids
        strategy.densify_gaussians("included", cast(BaseGaussianModel, model))
        for name, parameter in old.items():
            self.assertIs(getattr(model, name), parameter)
        self.assertIs(model.row_ids, row_ids)
        self.assertEqual(model.hook_refresh_count, 0)
        self.assertEqual(strategy.densify_grad_norm_accum["included"].count_nonzero(), 0)
        self.assertEqual(strategy.densify_grad_norm_denom["included"].count_nonzero(), 0)
        torch.testing.assert_close(old_history, torch.full((3, 1), 0.0001))

    def test_public_calls_follow_independent_schedules_after_growth_ends(self) -> None:
        import gsplat.strategy as gsplat_api

        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(
            model,
            densify={"start_iteration": 2, "end_iteration": 6, "frequency": 4},
            prune={"start_iteration": 6, "end_iteration": 10, "frequency": 4, "density_threshold": 0.2},
            reset_density={"start_iteration": 8, "end_iteration": 12, "frequency": 5, "new_max_density": 0.3},
        )
        nodes: TypedModuleDict[Any] = TypedModuleDict(cast(Any, {"included": model}))
        system = SimpleNamespace(
            model=SimpleNamespace(
                collect_gaussian_parameters=lambda *_a, **_kw: {"positions": model.positions.detach()}
            )
        )
        batch = SimpleNamespace(
            rendering=SimpleNamespace(camera=SimpleNamespace(rays=torch.zeros(1, 1, 1, 6)), lidar=None)
        )
        events: list[tuple[int, str]] = []
        step_now = 0

        def record(name: str, operation: Any) -> Any:
            def call(*args: Any, **kwargs: Any) -> Any:
                events.append((step_now, name))
                return operation(*args, **kwargs)

            return call

        with (
            patch.object(gsplat_api, "update", side_effect=record("update", gsplat_api.update)),
            patch.object(gsplat_api, "densify", side_effect=record("densify", gsplat_api.densify)),
            patch.object(gsplat_api, "prune", side_effect=record("prune", gsplat_api.prune)),
            patch.object(gsplat_api, "reset_opacity", side_effect=record("reset", gsplat_api.reset_opacity)),
        ):
            strategy.densify_grad_norm_accum["included"].zero_()
            for step_now in range(1, 13):
                model.positions.grad = torch.zeros_like(model.positions)
                strategy.update_step_train_batch_end(0, step_now, cast(Any, batch), system, nodes)
        self.assertEqual([x for x in events if x[1] != "update"], [(4, "densify"), (8, "prune"), (10, "reset")])
        self.assertEqual([step for step, name in events if name == "update"], [1, 2, 3, 4, 5])
        self.assertEqual(model.get_num_gaussians(), 2)
        self.assertLessEqual(float(model.get_densities().detach().max()), 0.300001)
        self.assert_optimizer_bound(model)

    def test_prune_and_reset_can_run_without_growth_configuration(self) -> None:
        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model, densify=None)
        nodes: TypedModuleDict[Any] = TypedModuleDict(cast(Any, {"included": model}))
        with patch.object(strategy, "update_gradient_buffers", side_effect=AssertionError("observation not needed")):
            strategy.update_step_train_batch_end(0, 3, cast(Any, None), None, nodes)
            strategy.update_step_train_batch_end(0, 5, cast(Any, None), None, nodes)
        self.assertEqual(model.get_num_gaussians(), 2)
        self.assertLessEqual(float(model.get_densities().detach().max()), 0.300001)

    def test_staged_growth_failure_preserves_history_and_live_state(self) -> None:
        import gsplat.strategy as gsplat_api

        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        old = dict(model.named_parameters())
        history = strategy.densify_grad_norm_accum["included"]
        public = gsplat_api.densify

        def fail(*args: Any, **kwargs: Any) -> Any:
            public(*args, **kwargs)
            raise RuntimeError("staging failed")

        with patch.object(gsplat_api, "densify", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "staging failed"):
                strategy.densify_gaussians("included", cast(BaseGaussianModel, model))
        self.assertIs(strategy.densify_grad_norm_accum["included"], history)
        torch.testing.assert_close(history, torch.ones(3, 1))
        for name, parameter in old.items():
            self.assertIs(getattr(model, name), parameter)
        model.raise_if_publication_failed()
        self.assert_optimizer_bound(model)

    def test_publication_failure_latches_existing_model_guard(self) -> None:
        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        with patch.object(model, "_reinstall_grad_hooks", side_effect=RuntimeError("publication failed")):
            with self.assertRaisesRegex(RuntimeError, "publication failed"):
                strategy.densify_gaussians("included", cast(BaseGaussianModel, model))
        with self.assertRaisesRegex(RuntimeError, "unsafe"):
            model.raise_if_publication_failed()
        with self.assertRaisesRegex(RuntimeError, "unsafe"):
            strategy.prune_gaussians_opacity("included", cast(BaseGaussianModel, model))

    def test_unsupported_storage_rejected_before_gsplat_call(self) -> None:
        import gsplat.strategy as gsplat_api

        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        with (
            patch.object(
                model, "validate_external_gaussian_topology_edit", side_effect=NotImplementedError("fixed-size storage")
            ),
            patch.object(gsplat_api, "densify") as public,
        ):
            with self.assertRaisesRegex(NotImplementedError, "fixed-size"):
                strategy.densify_gaussians("included", cast(BaseGaussianModel, model))
            public.assert_not_called()
        model.raise_if_publication_failed()

    def test_scalar_bookkeeping_and_sharding_survive_publication(self) -> None:
        from nurec.models.base import BaseModel

        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        optimizer = model.optimizers[0]["optimizer"]
        for parameter in model.parameters():
            optimizer.state[parameter]["scalar"] = torch.tensor(11.0)
            optimizer.state[parameter]["tag"] = "unchanged"
            BaseModel.mark_as_sharded(parameter)
        BaseModel.mark_as_sharded(model.row_ids)
        for buffers in (strategy.densify_grad_norm_accum, strategy.densify_grad_norm_denom):
            BaseModel.mark_as_sharded(buffers["included"])
        strategy.densify_gaussians("included", cast(BaseGaussianModel, model))
        for parameter in model.parameters():
            self.assertTrue(BaseModel.is_sharded(parameter))
            self.assertEqual(optimizer.state[parameter]["tag"], "unchanged")
            self.assertEqual(optimizer.state[parameter]["scalar"].item(), 11.0)
        self.assertTrue(BaseModel.is_sharded(model.row_ids))
        for buffers in (strategy.densify_grad_norm_accum, strategy.densify_grad_norm_denom):
            self.assertTrue(BaseModel.is_sharded(buffers["included"]))

    def test_checkpoint_and_row_permutation_after_growth(self) -> None:
        import copy

        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        strategy.densify_gaussians("included", cast(BaseGaussianModel, model))
        count = model.get_num_gaussians()
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(model.optimizers[0]["optimizer"].state_dict())
        strategy_state = copy.deepcopy(strategy.state_dict())
        resumed = _OpsGaussianNode(count=count)
        resumed.load_state_dict(model_state)
        resumed.optimizers[0]["optimizer"].load_state_dict(optimizer_state)
        restored_strategy = _make_ops_strategy(resumed)
        restored_strategy.load_state_dict(strategy_state)
        torch.testing.assert_close(
            restored_strategy.densify_grad_norm_accum["included"], strategy.densify_grad_norm_accum["included"]
        )
        self.assert_optimizer_bound(resumed)
        history = torch.arange(count, dtype=torch.float32).reshape(-1, 1)
        restored_strategy.densify_grad_norm_accum["included"] = nn.Parameter(history.clone(), requires_grad=False)
        order = torch.arange(count - 1, -1, -1)
        restored_strategy.permute_gaussian_rows({"included": order})
        torch.testing.assert_close(restored_strategy.densify_grad_norm_accum["included"], history[order])

    def test_coincident_schedules_use_updated_rows_in_order(self) -> None:
        import gsplat.strategy as gsplat_api

        model = _OpsGaussianNode()
        schedule = {"start_iteration": 0, "end_iteration": 8, "frequency": 2}
        strategy = _make_ops_strategy(
            model,
            densify=schedule,
            prune={**schedule, "density_threshold": 0.2},
            reset_density={**schedule, "new_max_density": 0.3},
        )
        events: list[str] = []

        def record(name: str, operation: Any) -> Any:
            def call(*args: Any, **kwargs: Any) -> Any:
                events.append(name)
                return operation(*args, **kwargs)

            return call

        nodes: TypedModuleDict[Any] = TypedModuleDict(cast(Any, {"included": model}))
        model.positions.grad = torch.ones_like(model.positions)
        batch = SimpleNamespace(
            rendering=SimpleNamespace(camera=SimpleNamespace(rays=torch.zeros(1, 1, 1, 6)), lidar=None)
        )
        system = SimpleNamespace(
            model=SimpleNamespace(
                collect_gaussian_parameters=lambda *_a, **_kw: {"positions": model.positions.detach()}
            )
        )
        with (
            patch.object(gsplat_api, "densify", side_effect=record("densify", gsplat_api.densify)),
            patch.object(gsplat_api, "prune", side_effect=record("prune", gsplat_api.prune)),
            patch.object(gsplat_api, "reset_opacity", side_effect=record("reset", gsplat_api.reset_opacity)),
        ):
            strategy.update_step_train_batch_end(0, 2, cast(Any, batch), system, nodes)
        self.assertEqual(events, ["densify", "prune", "reset"])
        torch.testing.assert_close(model.row_ids, torch.tensor([1, 1, 2, 2]))
        self.assertEqual(strategy.densify_grad_norm_accum["included"].shape, (4, 1))
        self.assertEqual(strategy.densify_grad_norm_denom["included"].count_nonzero(), 0)
        self.assert_optimizer_bound(model)

    def test_empty_layer_densification(self) -> None:
        model = _OpsGaussianNode(count=0)
        strategy = _make_ops_strategy(model)
        strategy.densify_gaussians("included", cast(BaseGaussianModel, model))
        self.assertEqual(model.get_num_gaussians(), 0)
        self.assertEqual(strategy.densify_grad_norm_accum["included"].shape, (0, 1))

    def test_world_size_adjusts_all_schedules_without_mutating_input(self) -> None:
        model = _OpsGaussianNode()
        template = _make_ops_strategy(model)
        trainer = make_trainer_cfg().model_copy(update={"world_size": 2, "relative_schedule": True})
        strategy = GSplatStrategy(
            config=template.config,
            trainer_config=trainer,
            init_from_datasource=False,
            gaussians_nodes=TypedModuleDict(cast(Any, {"included": model})),
        )
        assert strategy.config.densify is not None and template.config.densify is not None
        assert strategy.config.prune is not None and strategy.config.reset_density is not None
        self.assertEqual(strategy.config.densify.end_iteration, 5)
        self.assertEqual(strategy.config.densify.frequency, 1)
        self.assertEqual(strategy.config.prune.frequency, 2)
        self.assertEqual(strategy.config.reset_density.frequency, 3)
        self.assertEqual(template.config.densify.end_iteration, 10)

    def test_missing_geometry_is_rejected_before_publication(self) -> None:
        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        optimizer = model.optimizers[0]["optimizer"]
        optimizer.param_groups = [group for group in optimizer.param_groups if group["name"] != "rotations"]
        original = dict(model.named_parameters())
        with self.assertRaisesRegex(ValueError, "named geometry"):
            strategy.prune_gaussians_opacity("included", cast(BaseGaussianModel, model))
        for name, parameter in original.items():
            self.assertIs(getattr(model, name), parameter)
        model.raise_if_publication_failed()

    def test_legacy_history_checkpoint_continues_with_public_ops(self) -> None:
        model = _OpsGaussianNode()
        strategy = _make_ops_strategy(model)
        # Legacy checkpoints store these same two NRE-owned row buffers.
        checkpoint = {
            "densify_grad_norm_accum.included": torch.tensor([[2.0], [4.0], [6.0]]),
            "densify_grad_norm_denom.included": torch.tensor([[1], [2], [3]], dtype=torch.int32),
        }
        strategy.load_state_dict(checkpoint, strict=True)
        strategy.prune_gaussians_opacity("included", cast(BaseGaussianModel, model))
        torch.testing.assert_close(strategy.densify_grad_norm_accum["included"], torch.tensor([[4.0], [6.0]]))
        torch.testing.assert_close(
            strategy.densify_grad_norm_denom["included"], torch.tensor([[2], [3]], dtype=torch.int32)
        )
        self.assert_optimizer_bound(model)

    def test_unsupported_activations_rejected_for_included_layers(self) -> None:
        for field, value in (("density_activation", "exp"), ("scale_activation", "sigmoid")):
            with self.subTest(field=field):
                model = _OpsGaussianNode()
                setattr(model.config, field, value)
                with self.assertRaisesRegex(ValueError, "sigmoid density and exponential scale"):
                    _make_ops_strategy(model)
                # Excluded layers never enter gsplat's storage-specific operations.
                _make_ops_strategy(model, exclude_layer_ids=["included"])
