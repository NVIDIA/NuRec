# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from collections.abc import Callable

import torch

from torch import nn
from tqdm import tqdm

from nurec.config.model import DensifyConfig, GsplatStrategyConfig, PruneConfig
from nurec.libs.slang_gaussians.interface import gsplat_strategy as gsplat_slang_strategy  # type: ignore
from nurec.models.gaussians.gaussians_model import BaseGaussianModel
from nurec.models.gaussians.strategies.base import BaseGaussianStrategy
from nurec.models.nn_extensions import TypedModuleDict
from nurec.utils.batch import DataAndRenderingBatch, RenderingData
from nurec.utils.cuda_rng import get_cuda_generator
from nurec.utils.geometry import quat_to_so3_matrix
from nurec.utils.trainer import TrainerConfig, adjust_step_for_world_size


log = logging.getLogger(__name__)


_TopologyMutation = tuple[DensifyConfig | PruneConfig, Callable[[str, BaseGaussianModel], None]]


class GSplatStrategy(BaseGaussianStrategy):
    """
    Implementation of the strategy for densifying and pruning proposed in the 3D Gaussian Splatting paper.

    See: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

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
        Hook that's called after the loss.backward and before the system.optimizer.step:
            - enables us to tap into `gaussian_model.positions.grad`
            - we need the current batch to determine the distance from the camera to the Gaussians that are hit

        Uses GPU-accelerated Slang kernel for efficient gradient accumulation.
        """
        # rays is shaped (b == 1, rows, cols, 6)
        rays_ori = rendering_data.rays.squeeze(0).reshape(-1, 6)[:, :3]

        params_grad = positions_param.grad
        assert params_grad is not None

        # Use Slang kernel for GPU-accelerated gradient buffer update
        # Note: we pass the first ray origin assuming all rays in a batch come from a similar origin
        gsplat_slang_strategy.update_gradient_buffers(
            positions=positions,
            params_grad=params_grad,
            ray_origin=rays_ori[0],
            grad_norm_accum=self.densify_grad_norm_accum[layer_id],
            grad_norm_denom=self.densify_grad_norm_denom[layer_id],
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
        assert self.config.densify is not None
        assert self.config.prune is not None
        assert self.config.reset_density is not None

        do_densify = global_step < self.config.densify.end_iteration
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
            if do_densify:
                count = gaussian_model.get_num_gaussians()
                current_offset = offset
                offset += count

            if layer_id in self.config.exclude_layer_ids:
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
                    operation(layer_id, gaussian_model)

            # Reset the Gaussian density
            if self._check_step_condition(
                step=global_step,
                start=self.config.reset_density.start_iteration,
                end=self.config.reset_density.end_iteration,
                freq=self.config.reset_density.frequency,
            ):
                self.reset_density(layer_id, gaussian_model)
                self.last_reset_epoch = epoch

    def reset_density(self, layer_id: str, gaussian_model: BaseGaussianModel) -> None:
        """Periodically reset densities"""
        assert self.config.reset_density is not None
        tqdm.write(f"{layer_id}: Resetting densities")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            assert name == "densities", "wrong paramaeter passed to update_param_fn"
            assert self.config.reset_density is not None
            densities = torch.clamp(
                param,
                max=gaussian_model.density_activation_inv(
                    torch.tensor(self.config.reset_density.new_max_density)
                ).item(),
            )
            return torch.nn.Parameter(densities)

        def update_optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(v)

        # update the parameters and the state in the optimizers
        self._update_param_with_optimizer(gaussian_model, update_param_fn, update_optimizer_fn, names=["densities"])

    def densify_gaussians(self, layer_id: str, gaussian_model: BaseGaussianModel) -> None:
        """Densify gaussians based on the gradient buffers: `densify_grad_norm_accum` and `densify_grad_norm_denom`"""

        assert gaussian_model.get_positions().requires_grad, (
            "Trying to perform split and clone but the positions are not being optimized"
        )
        densify_grad_norm = self.densify_grad_norm_accum[layer_id] / self.densify_grad_norm_denom[layer_id]
        densify_grad_norm[densify_grad_norm.isnan()] = 0.0

        self.clone_gaussians(layer_id, gaussian_model, densify_grad_norm.squeeze(-1))
        self.split_gaussians(layer_id, gaussian_model, densify_grad_norm.squeeze(-1))

    def clone_gaussians(
        self, layer_id: str, gaussian_model: BaseGaussianModel, densify_grad_norm: torch.Tensor
    ) -> None:
        """Cloning logic and calls densify postfix"""
        assert self.config.densify is not None
        assert densify_grad_norm is not None, "Positional gradients must be available in order to clone the Gaussians"

        # Extract points that satisfy the gradient condition
        mask = torch.where(densify_grad_norm >= self.config.densify.clone_grad_threshold, True, False)

        # If the gaussians are larger they shouldn't be cloned, but rather split
        mask = torch.logical_and(
            mask,
            torch.max(gaussian_model.get_scales(), dim=1).values
            <= self.config.densify.relative_size_threshold * gaussian_model.scene_extent,
        )

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            param_new = torch.cat([param, param[mask]])
            return torch.nn.Parameter(param_new) if isinstance(param, torch.nn.Parameter) else param_new

        def update_optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.cat([v, torch.zeros((int(mask.sum()), *v.shape[1:]), device=gaussian_model.device)])

        self._update_param_with_optimizer(gaussian_model, update_param_fn, update_optimizer_fn)
        self.reset_densification_buffers(layer_id, gaussian_model)

        # stats
        if self.config.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            tqdm.write(f"{layer_id}: Cloned {n_clone} / {n_before} ({n_clone / n_before * 100:.2f}%) gaussians")

    def split_gaussians(
        self, layer_id: str, gaussian_model: BaseGaussianModel, densify_grad_norm: torch.Tensor
    ) -> None:
        """Splitting logic and calls densify postfix"""
        assert self.config.densify is not None
        densify_config = self.config.densify
        n_init_points = gaussian_model.get_positions().shape[0]

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")

        # Here we already have the cloned points in the gaussian_model.positions so only take the points up to size of the initial grad
        padded_grad[: densify_grad_norm.shape[0]] = densify_grad_norm.squeeze(-1)
        mask = torch.where(padded_grad >= densify_config.split_grad_threshold, True, False)
        mask = torch.logical_and(
            mask,
            torch.max(gaussian_model.get_scales(), dim=1).values
            > densify_config.relative_size_threshold * gaussian_model.scene_extent,
        )

        stds = gaussian_model.get_scales()[mask].repeat(densify_config.split_n_gaussians, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        # Sample via the process-wide CUDA Generator (see `nurec/utils/cuda_rng.py`)
        # so the densify hot path does not advance the CUDA default generator's
        # Philox offset -- that advance is incompatible with `torch.cuda.graph(...)`
        # capture. Compose as `randn(generator=) * std + mean` rather than
        # `torch.normal(mean, std, generator=)`: the per-element-mean/std form
        # of torch.normal is not stream-capture-safe in PyTorch 2.7.0+cu128
        # (the version pinned by deps/python/requirements_3_11_x86_64.in:11)
        # and raises "CUDA error: operation not permitted when stream is
        # capturing" inside `torch.cuda.graph`. Reproducible directly via the
        # call below with `mean` and `std` as same-shape tensors -- the
        # internal cuda backend hits a path with a not-yet-stream-capturable
        # op. The composition uses only `torch.randn`, which IS capture-clean
        # (pinned by the contract test
        # `cuda_rng_test::test_per_element_normal_via_randn_under_torch_cuda_graph`).
        # If a future PyTorch upgrade fixes the upstream `torch.normal`
        # behavior, this branch can revert to a direct `torch.normal` call.
        samples = (
            torch.randn(
                stds.shape,
                generator=get_cuda_generator(stds.device),
                device=stds.device,
                dtype=stds.dtype,
            )
            * stds
            + means
        )
        rots = quat_to_so3_matrix(gaussian_model.rotations[mask]).repeat(densify_config.split_n_gaussians, 1, 1)
        offsets = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            repeats = [densify_config.split_n_gaussians] + [1] * (param.dim() - 1)
            if name == "positions":
                p_split = param[mask].repeat(repeats) + offsets  # [2N, 3]
            elif name == "scales":
                p_split = gaussian_model.scale_activation_inv(
                    gaussian_model.scale_activation(param[mask].repeat(repeats))
                    / (0.8 * densify_config.split_n_gaussians)
                )
            else:
                p_split = param[mask].repeat(repeats)

            p_new = torch.cat([param[~mask], p_split])
            if isinstance(param, torch.nn.Parameter):
                p_new = torch.nn.Parameter(p_new)
            return p_new

        def update_optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            v_split = torch.zeros(
                (densify_config.split_n_gaussians * int(mask.sum()), *v.shape[1:]), device=gaussian_model.device
            )
            return torch.cat([v[~mask], v_split])

        self._update_param_with_optimizer(gaussian_model, update_param_fn, update_optimizer_fn)

        self.reset_densification_buffers(layer_id, gaussian_model)

        if self.config.print_stats:
            n_before = mask.shape[0]
            n_split = mask.sum()
            tqdm.write(f"{layer_id}: Splitted {n_split} / {n_before} ({n_split / n_before * 100:.2f}%) gaussians")

    def prune_gaussians_opacity(self, layer_id: str, gaussian_model: BaseGaussianModel) -> None:
        """Pruning based on opacity logic"""
        assert self.config.prune is not None
        valid_mask = gaussian_model.get_densities().squeeze(-1) >= self.config.prune.density_threshold

        if self.config.print_stats:
            n_before = valid_mask.shape[0]
            n_prune = n_before - valid_mask.sum()
            tqdm.write(f"{layer_id}: Density-pruned {n_prune} / {n_before} ({n_prune / n_before * 100:.2f}%) gaussians")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(param[valid_mask]) if isinstance(param, torch.nn.Parameter) else param[valid_mask]

        def update_optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            return v[valid_mask]

        self._update_param_with_optimizer(gaussian_model, update_param_fn, update_optimizer_fn)
        self.prune_densification_buffers(layer_id, valid_mask)

    @torch.no_grad()
    def permute_gaussian_rows(self, orders: dict[str, torch.Tensor]) -> None:
        self._permute_layer_buffers(self.densify_grad_norm_accum, orders)
        self._permute_layer_buffers(self.densify_grad_norm_denom, orders)

    def reset_densification_buffers(self, layer_id: str, gaussian_model: BaseGaussianModel) -> None:
        self.densify_grad_norm_accum[layer_id] = nn.Parameter(
            torch.zeros(
                (gaussian_model.get_positions().shape[0], 1),
                device=gaussian_model.device,
                dtype=self.densify_grad_norm_accum[layer_id].dtype,
            ),
            requires_grad=False,
        )

        self.densify_grad_norm_denom[layer_id] = nn.Parameter(
            torch.zeros(
                (gaussian_model.get_positions().shape[0], 1),
                device=gaussian_model.device,
                dtype=self.densify_grad_norm_denom[layer_id].dtype,
            ),
            requires_grad=False,
        )

    def prune_densification_buffers(self, layer_id: str, valid_mask: torch.Tensor) -> None:
        # Update non-optimizable buffers
        self.densify_grad_norm_accum[layer_id] = nn.Parameter(
            self.densify_grad_norm_accum[layer_id][valid_mask], requires_grad=False
        )
        self.densify_grad_norm_denom[layer_id] = nn.Parameter(
            self.densify_grad_norm_denom[layer_id][valid_mask], requires_grad=False
        )


BaseGaussianStrategy.register_to_gaussian_strategy_factory("gsplat", GSplatStrategy)
