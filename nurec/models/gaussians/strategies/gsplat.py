# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging

from collections.abc import Callable
from typing import Any, cast

import torch

from gsplat.scene import Scene
from gsplat.strategy import DefaultStrategy
from torch import nn
from tqdm import tqdm

from nurec.config.model import DensifyConfig, GsplatStrategyConfig, PruneConfig
from nurec.models.base import BaseModel
from nurec.models.gaussians.gaussians_model import BaseGaussianModel
from nurec.models.gaussians.strategies.base import BaseGaussianStrategy
from nurec.models.nn_extensions import TypedModuleDict
from nurec.utils.batch import DataAndRenderingBatch, RenderingData
from nurec.utils.cuda_rng import get_cuda_generator
from nurec.utils.trainer import TrainerConfig, adjust_step_for_world_size


log = logging.getLogger(__name__)


_TopologyMutation = tuple[DensifyConfig | PruneConfig, Callable[[str, BaseGaussianModel, int], None]]


class _LayerScene(Scene[object]):
    """Identify an NRE layer; row metadata is carried by the parameter bridge."""

    def put(self, name: str, component: object) -> None:
        raise NotImplementedError("NRE owns layer components")

    def get(self, component: str) -> object:
        raise KeyError(component)


class GSplatStrategy(BaseGaussianStrategy):
    """
    Implementation of the strategy for densifying and pruning proposed in the 3D Gaussian Splatting paper.

    See: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

    Shared gsplat operations own selection and row updates; NRE owns schedules
    and publication. Growth uses strict > gradient
    thresholds and requires an opacity reset cap strictly between 0 and 1.

    Note that we leverage the positions gradients in 3D instead of the view-space position gradients.
    We therefore rescale the gradient by the distance to the camera as described in the 3D Gaussian Ray Tracing paper.

    See: https://gaussiantracer.github.io/
    """

    config: GsplatStrategyConfig  # Narrow type from parent's StrategyConfigType

    def _topology_mutation_ops(self) -> tuple[_TopologyMutation, ...]:
        """Return topology-changing schedules and operations in execution order."""
        operations: list[_TopologyMutation] = []
        if self.config.densify is not None:
            operations.append((self.config.densify, self.densify_gaussians))
        if self.config.prune is not None:
            operations.append((self.config.prune, self.prune_gaussians_opacity))
        return tuple(operations)

    def _is_topology_mutation_due(self, global_step: int, config: DensifyConfig | PruneConfig) -> bool:
        return self._check_step_condition(
            step=global_step,
            start=config.start_iteration,
            end=config.end_iteration,
            freq=config.frequency,
        )

    def may_mutate_topology_at(self, global_step: int) -> bool:
        """Return whether registered densify/prune work may change row topology."""
        if self.frozen:
            return False
        return any(self._is_topology_mutation_due(global_step, config) for config, _ in self._topology_mutation_ops())

    def __init__(
        self,
        config: GsplatStrategyConfig,
        trainer_config: TrainerConfig,
        init_from_datasource: bool,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
    ) -> None:
        super().__init__(config, trainer_config, init_from_datasource, gaussians_nodes)

        # TODO: Replace with nn.BufferDict once that exists
        self.densify_grad_norm_accum = nn.ParameterDict()  # Accumulation of the norms of the positions gradients
        self.densify_grad_norm_denom = nn.ParameterDict()
        for layer_id, gaussian_model in gaussians_nodes.items():
            self.densify_grad_norm_accum[layer_id] = nn.UninitializedParameter(requires_grad=False)
            self.densify_grad_norm_denom[layer_id] = nn.UninitializedParameter(requires_grad=False)

        self.config = self.config.model_copy(deep=True)
        densify = self.config.densify
        self._gsplat_default = DefaultStrategy(
            gradient_mode="3d",
            exclude_layer_ids=tuple(self.config.exclude_layer_ids),
            clone_grad_threshold=densify.clone_grad_threshold if densify else None,
            split_grad_threshold=densify.split_grad_threshold if densify else None,
            split_n_gaussians=densify.split_n_gaussians if densify else 2,
            grow_scale3d=densify.relative_size_threshold if densify else 0.01,
            prune_nan_opacity=True,
            prune_opa=self.config.prune.density_threshold if self.config.prune else 0.01,
            reset_opa_cap=self.config.reset_density.new_max_density if self.config.reset_density else None,
            prune_scale3d=float("inf"),
        )
        for layer_id, model in gaussians_nodes.items():
            if layer_id in self.config.exclude_layer_ids:
                continue
            if model.config.density_activation != "sigmoid" or model.config.scale_activation != "exp":
                raise ValueError("GSplatStrategy requires sigmoid density and exponential scale activation")
        if self.config.densify is not None:
            self.config.densify.start_iteration = adjust_step_for_world_size(
                trainer_config, self.config.densify.start_iteration
            )
            self.config.densify.end_iteration = adjust_step_for_world_size(
                trainer_config, self.config.densify.end_iteration
            )
            self.config.densify.frequency = adjust_step_for_world_size(trainer_config, self.config.densify.frequency)
            log.info("GSplatStrategy/densify:")
            log.info(f"    |─start_iteration={self.config.densify.start_iteration}")
            log.info(f"    |─end_iteration={self.config.densify.end_iteration}")
            log.info(f"    └─frequency={self.config.densify.frequency}")

        if self.config.prune is not None:
            self.config.prune.start_iteration = adjust_step_for_world_size(
                trainer_config, self.config.prune.start_iteration
            )
            self.config.prune.end_iteration = adjust_step_for_world_size(
                trainer_config, self.config.prune.end_iteration
            )
            self.config.prune.frequency = adjust_step_for_world_size(trainer_config, self.config.prune.frequency)
            log.info("GSplatStrategy/prune:")
            log.info(f"    |─start_iteration={self.config.prune.start_iteration}")
            log.info(f"    |─end_iteration={self.config.prune.end_iteration}")
            log.info(f"    └─frequency={self.config.prune.frequency}")

        if self.config.reset_density is not None:
            self.config.reset_density.start_iteration = adjust_step_for_world_size(
                trainer_config, self.config.reset_density.start_iteration
            )
            self.config.reset_density.end_iteration = adjust_step_for_world_size(
                trainer_config, self.config.reset_density.end_iteration
            )
            self.config.reset_density.frequency = adjust_step_for_world_size(
                trainer_config, self.config.reset_density.frequency
            )
            log.info("GSplatStrategy/reset_density:")
            log.info(f"    |─start_iteration={self.config.reset_density.start_iteration}")
            log.info(f"    |─end_iteration={self.config.reset_density.end_iteration}")
            log.info(f"    └─frequency={self.config.reset_density.frequency}")

    @torch.no_grad()
    def maybe_initialize_buffers(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> None:
        # Accumulation of the norms of the positions gradients
        if self.init_from_datasource:
            for layer_id, gaussian_model in gaussians_nodes.items():
                num_gaussians = gaussian_model.get_num_gaussians()
                self.densify_grad_norm_accum[layer_id] = nn.Parameter(
                    torch.zeros((num_gaussians, 1), dtype=torch.float, device=gaussian_model.device),
                    requires_grad=False,
                )
                self.densify_grad_norm_denom[layer_id] = nn.Parameter(
                    torch.zeros((num_gaussians, 1), dtype=torch.int, device=gaussian_model.device),
                    requires_grad=False,
                )

    @torch.no_grad()
    def update_gradient_buffers(
        self,
        layer_id: str,
        positions_param: torch.Tensor,
        positions: torch.Tensor,
        rendering_data: RenderingData,
        step: int,
    ) -> None:
        """
        Called at training batch end, after backward and the optimizer step:
            - enables us to tap into `gaussian_model.positions.grad`
            - we need the current batch to determine the distance from the camera to the Gaussians that are hit

        Accumulate distance-scaled 3D gradients through the public gsplat API.
        """
        # rays is shaped (b == 1, rows, cols, 6)
        rays_ori = rendering_data.rays.squeeze(0).reshape(-1, 6)[:, :3]

        params_grad = positions_param.grad
        assert params_grad is not None

        from gsplat.strategy import update

        # Rebind views each time: publication and checkpoint loads replace buffers.
        update(
            self._gsplat_default,
            params={"means": cast(nn.Parameter, positions_param)},
            state=self._gsplat_state(layer_id),
            scene=_LayerScene(layer_id),
            position_gradients=params_grad,
            positions=positions,
            reference_origin=rays_ori[0],
        )

    def _update_step_train_batch_end_impl(
        self,
        epoch: int,
        global_step: int,
        batch: DataAndRenderingBatch,
        system,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
        **kwargs,
    ) -> None:
        """Here we perform all the Gaussians-specific updates"""
        do_densify = self.config.densify is not None and global_step < self.config.densify.end_iteration
        if do_densify:
            assert batch.rendering is not None

            if (rendering_camera := batch.rendering.camera) is not None:
                rendering_data = rendering_camera
            elif (rendering_lidar := batch.rendering.lidar) is not None:
                rendering_data = rendering_lidar
            else:
                raise ValueError("Either camera or lidar rendering data must be provided")

            positions = system.model.collect_gaussian_parameters(rendering_data, is_training_batch=True)["positions"]
            offset = 0

        for layer_id, gaussian_model in gaussians_nodes.items():
            gaussian_model.raise_if_publication_failed()
            if do_densify:
                count = gaussian_model.get_num_gaussians()
                current_offset = offset
                offset += count

            if self._layer_is_excluded(layer_id):
                continue

            # Update the gradient buffers
            if do_densify:
                self.update_gradient_buffers(
                    layer_id,
                    gaussian_model.positions,
                    positions[current_offset : current_offset + count],
                    rendering_data,
                    global_step,
                )

            for config, operation in self._topology_mutation_ops():
                if self._is_topology_mutation_due(global_step, config):
                    operation(layer_id, gaussian_model, global_step)

            # Reset the Gaussian density
            if self.config.reset_density is not None and self._check_step_condition(
                step=global_step,
                start=self.config.reset_density.start_iteration,
                end=self.config.reset_density.end_iteration,
                freq=self.config.reset_density.frequency,
            ):
                self.reset_density(layer_id, gaussian_model)
                self.last_reset_epoch = epoch

    def _layer_is_excluded(self, layer_id: str) -> bool:
        return layer_id in self.config.exclude_layer_ids

    def _gsplat_state(self, layer_id: str, scene_scale: float = 1.0, *, stage: bool = False) -> dict[str, Any]:
        state = self._gsplat_default.initialize_state(scene_scale=scene_scale)
        for key, buffers in (("grad2d", self.densify_grad_norm_accum), ("count", self.densify_grad_norm_denom)):
            old = buffers[layer_id]
            if not isinstance(old, nn.UninitializedParameter):
                value = old.view(-1)
                state[key] = value.clone() if stage else value
        return state

    def _publish_gsplat_history(self, layer_id: str, state: dict[str, Any]) -> None:
        for key, buffers in (("grad2d", self.densify_grad_norm_accum), ("count", self.densify_grad_norm_denom)):
            value = state[key]
            if value is not None:
                old = buffers[layer_id]
                replacement = nn.Parameter(value.reshape(-1, 1), requires_grad=False)
                if BaseModel.is_sharded(old):
                    BaseModel.mark_as_sharded(replacement)
                buffers[layer_id] = replacement

    def _run_gsplat_operation(
        self,
        model: BaseGaussianModel,
        operation: Callable[[dict[str, nn.Parameter], dict[str, torch.optim.Optimizer]], object],
        names: list[str] | None = None,
        *,
        history: tuple[str, dict[str, Any]] | None = None,
    ) -> object:
        """Stage per-field optimizer views, then install through NRE's existing updater.

        Views isolate gsplat's one-group-per-field convention from NRE's grouped
        optimizers. Tensor values are shared read-only; replacement state dicts and
        parameter groups are private until the operation succeeds.
        """
        model.raise_if_publication_failed()
        aliases = {"positions": "means", "rotations": "quats", "densities": "opacities"}
        params: dict[str, nn.Parameter] = {}
        optimizers: dict[str, torch.optim.Optimizer] = {}
        for entry in model.optimizers:
            optimizer = entry["optimizer"]
            for group in optimizer.param_groups:
                if "name" not in group:
                    continue
                name = group["name"].split(".")[-1]
                if names is not None and name not in names:
                    continue
                alias = aliases.get(name, name)
                if len(group["params"]) != 1 or alias in params:
                    raise ValueError("gsplat integration requires one unique parameter per named group")
                parameter = group["params"][0]
                params[alias] = parameter
                view = copy.copy(optimizer)
                view.param_groups = [dict(group)]
                view.state = copy.copy(optimizer.state)
                # Adapt only row-shaped state and Adam's step to the gsplat ops
                # layout. Other scalar/non-tensor bookkeeping stays untouched.
                view.state[parameter] = {
                    key: value
                    for key, value in optimizer.state.get(parameter, {}).items()
                    if key == "step" or (isinstance(value, torch.Tensor) and value.ndim > 0)
                }
                optimizers[alias] = view
        for name, buffer in model.get_additional_buffers().items():
            if names is None or name in names:
                alias = aliases.get(name, name)
                if alias in params:
                    raise ValueError(f"buffer duplicates optimizer parameter {name}")
                params[alias] = nn.Parameter(buffer, requires_grad=False)

        if names is None:
            missing = {"means", "quats", "scales", "opacities"} - params.keys()
            if missing:
                raise ValueError(f"gsplat bridge requires named geometry parameters or row buffers: {sorted(missing)}")
        count = model.get_num_gaussians()
        if any(parameter.ndim == 0 or parameter.shape[0] != count for parameter in params.values()):
            raise ValueError("gsplat integration requires Gaussian-row-aligned parameters and buffers")

        for optimizer in optimizers.values():
            for key, value in optimizer.state[optimizer.param_groups[0]["params"][0]].items():
                if key != "step" and value.shape[0] != count:
                    raise ValueError("gsplat optimizer state must be Gaussian-row-aligned")
        originals = dict(params)
        result = operation(params, optimizers)

        def update_param(name: str, old: torch.Tensor) -> torch.Tensor:
            replacement = params[aliases.get(name, name)]
            replacement.requires_grad_(old.requires_grad)
            return replacement if isinstance(old, nn.Parameter) else replacement.detach()

        def update_state(name: str, key: str, old: torch.Tensor) -> torch.Tensor:
            alias = aliases.get(name, name)
            return optimizers[alias].state[params[alias]][key]

        try:
            if any(params[name] is not old for name, old in originals.items()):
                self._update_param_with_optimizer(model, update_param, update_state, names=names)
                model._reinstall_grad_hooks()
            if history is not None:
                self._publish_gsplat_history(*history)
        except Exception:
            # Reuse the model guard already checked by composite checkpoints.
            model.mark_publication_failed()
            raise
        return result

    def reset_density(self, layer_id: str, gaussian_model: BaseGaussianModel) -> None:
        """Periodically reset densities"""
        assert self.config.reset_density is not None
        tqdm.write(f"{layer_id}: Resetting densities")
        strategy = self._gsplat_default
        from gsplat.strategy import reset_opacity

        self._run_gsplat_operation(
            gaussian_model,
            lambda params, optimizers: reset_opacity(strategy, params, optimizers, {}, scene=_LayerScene(layer_id)),
            names=["densities"],
        )

    def densify_gaussians(self, layer_id: str, gaussian_model: BaseGaussianModel, step: int = 0) -> None:
        """Densify gaussians based on the gradient buffers: `densify_grad_norm_accum` and `densify_grad_norm_denom`"""

        assert gaussian_model.get_positions().requires_grad, (
            "Trying to perform split and clone but the positions are not being optimized"
        )
        strategy = self._gsplat_default
        from gsplat.strategy import densify

        gaussian_model.validate_external_gaussian_topology_edit("gsplat densification")
        state = self._gsplat_state(layer_id, gaussian_model.scene_extent, stage=True)
        result = self._run_gsplat_operation(
            gaussian_model,
            lambda params, optimizers: densify(
                strategy,
                params,
                optimizers,
                state,
                step,
                scene=_LayerScene(layer_id),
                generator=get_cuda_generator(gaussian_model.device) if gaussian_model.device.type == "cuda" else None,
                reset_history=True,
            ),
            history=(layer_id, state),
        )
        if self.config.print_stats:
            tqdm.write(f"{layer_id}: gsplat densification: {result}")

    def prune_gaussians_opacity(self, layer_id: str, gaussian_model: BaseGaussianModel, step: int = 0) -> None:
        """Pruning based on opacity logic"""
        assert self.config.prune is not None
        strategy = self._gsplat_default
        from gsplat.strategy import prune

        gaussian_model.validate_external_gaussian_topology_edit("gsplat pruning")
        state = self._gsplat_state(layer_id, gaussian_model.scene_extent, stage=True)
        result = self._run_gsplat_operation(
            gaussian_model,
            lambda params, optimizers: prune(strategy, params, optimizers, state, step, scene=_LayerScene(layer_id)),
            history=(layer_id, state),
        )
        if self.config.print_stats:
            tqdm.write(f"{layer_id}: Pruned {result} gaussians")

    @torch.no_grad()
    def permute_gaussian_rows(self, orders: dict[str, torch.Tensor]) -> None:
        self._permute_layer_buffers(self.densify_grad_norm_accum, orders)
        self._permute_layer_buffers(self.densify_grad_norm_denom, orders)


BaseGaussianStrategy.register_to_gaussian_strategy_factory("gsplat", GSplatStrategy)
