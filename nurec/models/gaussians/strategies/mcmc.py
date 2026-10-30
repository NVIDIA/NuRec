# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

from collections.abc import Callable, Iterator
from typing import Any, Optional

import torch

from torch import nn
from tqdm import tqdm

from nurec.config.model import MCMCStrategyConfig
from nurec.models.base import BaseModel
from nurec.models.gaussians.construction_barriers_model import ConstructionBarriersGaussianModel
from nurec.models.gaussians.gaussians_model import BaseGaussianModel, RigidGaussianModel
from nurec.models.gaussians.strategies.base import BaseGaussianStrategy
from nurec.models.gaussians.strategies.gsplat_mcmc_adapter import (
    add_gaussians as gsplat_add_gaussians,
)
from nurec.models.gaussians.strategies.gsplat_mcmc_adapter import (
    mcmc_dead_mask,
    perturb_positions,
    require_mcmc_capabilities,
    update_invisible_steps,
)
from nurec.models.gaussians.strategies.gsplat_mcmc_adapter import (
    relocate_gaussians as gsplat_relocate_gaussians,
)
from nurec.models.gaussians.strategies.gsplat_mcmc_adapter import (
    relocate_gaussians_grouped as gsplat_relocate_gaussians_grouped,
)
from nurec.models.gaussians.strategies.selected_gaussian_nodes import SelectedGaussianNodes
from nurec.models.nn_extensions import TypedModuleDict
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.cuda_rng import get_cuda_generator
from nurec.utils.profiling import ScopedTimer
from nurec.utils.trainer import TrainerConfig, adjust_gaussian_count_for_world_size, adjust_step_for_world_size


log = logging.getLogger(__name__)


_TopologyMutation = tuple[
    MCMCStrategyConfig.MCMCRelocationConfig | MCMCStrategyConfig.MCMCAdditionConfig,
    Callable[[TypedModuleDict[BaseGaussianModel]], None],
]


class _LazyParameterDict(nn.ParameterDict):
    """ParameterDict that omits unmaterialized checkpoint placeholders on save."""

    def _save_to_state_dict(self, destination: dict[str, Any], prefix: str, keep_vars: bool) -> None:
        # Module.state_dict() recursively calls this method on children. The
        # standard implementation detaches UninitializedParameter and raises.
        for name, parameter in self._parameters.items():
            if parameter is None or isinstance(parameter, nn.UninitializedParameter):
                continue
            destination[prefix + name] = parameter if keep_vars else parameter.detach()
        for name, buffer in self._buffers.items():
            if buffer is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buffer if keep_vars else buffer.detach()


class _RigidScaleGuardCache(nn.Module):
    """Derived rigid scale limits that follow strategy device transfers."""

    def __init__(self, max_rendered_scale_m: torch.Tensor) -> None:
        super().__init__()
        self.max_rendered_scale_m = nn.Buffer(max_rendered_scale_m, persistent=False)


class MCMCStrategy(BaseGaussianStrategy):
    """Densification and prunning strategy that follows the paper:

    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/abs/2404.09591>`_

    MCMC Strategy interprets the training process of placing and optimizing Gaussians
    as a sampling process.

    Specifically, it periodically:
    - Moves "dead" Gaussians (low opacity) to the location of "live" Gaussians (high opacity).
    - Adds covariance dependent noise to the positions of the Gaussians.
    - Introduces new Gaussians sampled based on the opacity distribution.

    """

    config: MCMCStrategyConfig

    def refresh_runtime_scenes(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> None:
        """Bind gsplat owner scenes to materialized Gaussian model layers."""
        for layer_id, node in gaussians_nodes.items():
            if isinstance(node, BaseGaussianModel):
                node.refresh_mcmc_scene(layer_id)

    def _topology_mutation_ops(self) -> tuple[_TopologyMutation, ...]:
        """Return topology-changing schedules and operations in execution order.

        Relocation remains topology-changing even when it preserves the global
        Gaussian count because it may rebuild per-layer or per-owner row layout.
        """
        return (
            (self.config.relocate, self.relocate_gaussians),
            (self.config.add, self.add_new_gaussians),
        )

    def _is_topology_mutation_due(
        self,
        global_step: int,
        config: MCMCStrategyConfig.MCMCRelocationConfig | MCMCStrategyConfig.MCMCAdditionConfig,
    ) -> bool:
        return self._check_step_condition(
            step=global_step,
            start=config.start_iteration,
            end=config.end_iteration,
            freq=config.frequency,
        )

    def may_mutate_topology_at(self, global_step: int) -> bool:
        """Return whether registered relocate/add work may change row topology."""
        if self.frozen:
            return False
        return any(self._is_topology_mutation_due(global_step, config) for config, _ in self._topology_mutation_ops())

    def __init__(
        self,
        config: MCMCStrategyConfig,
        trainer_config: TrainerConfig,
        init_from_datasource: bool,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
    ) -> None:
        """Validate layer policies and initialize rank-adjusted MCMC schedules.

        Cuboid-grouped topology is single-GPU only: reject distributed use
        before any cuboid-local donor decision or row publication can occur.
        """
        super().__init__(config, trainer_config, init_from_datasource, gaussians_nodes)

        self.config = config.model_copy(deep=True)
        grouped = self.config.cuboid_grouped_layer_ids
        if grouped:
            if trainer_config.world_size != 1:
                raise ValueError("Cuboid-grouped MCMC currently requires one GPU")
            if len(set(grouped)) != len(grouped):
                raise ValueError("Duplicate cuboid-grouped MCMC layer")
            if not set(grouped) <= set(self.config.independent_layer_max_n_gaussians):
                raise ValueError("Cuboid-grouped layers require explicit independent growth caps")
            if set(grouped) & (
                set(self.config.add.fixed_count_layer_ids) | set(self.config.relocate.same_track_layer_ids)
            ):
                raise ValueError("Cuboid grouping cannot combine with fixed-count or rigid-track overrides")
        for layer_id, node in gaussians_nodes.items():
            if isinstance(node, ConstructionBarriersGaussianModel) != (layer_id in grouped):
                raise ValueError(
                    "Construction-barrier layers require cuboid-grouped MCMC, and no other layer supports it"
                )
        self._training_end_step = trainer_config.max_steps if trainer_config.max_steps > 0 else None
        self._rigid_scale_guard_cache: TypedModuleDict[_RigidScaleGuardCache] = TypedModuleDict()
        if self.config.binom_n_max <= 0:
            raise ValueError("MCMC relocation requires binom_n_max > 0")
        float32_epsilon = torch.finfo(torch.float32).eps
        if not 0.0 <= self.config.opacity_threshold <= 1.0 - float32_epsilon:
            raise ValueError("MCMC relocation requires opacity_threshold in [0, 1 - float32 epsilon]")
        invalid_quota_layers = set(self.config.relocate.min_gaussians_per_track) - set(
            self.config.relocate.same_track_layer_ids
        )
        if invalid_quota_layers:
            raise ValueError(
                "min_gaussians_per_track keys must also be listed in same_track_layer_ids: "
                f"{sorted(invalid_quota_layers)}"
            )
        self.config.relocate.min_gaussians_per_track = {
            layer_id: max(1, adjust_gaussian_count_for_world_size(trainer_config, minimum))
            for layer_id, minimum in self.config.relocate.min_gaussians_per_track.items()
        }

        # Guards against duplicate visibility counter updates for the same results object
        self._last_visibility_update_id: Optional[int] = None
        self.invisible_steps = _LazyParameterDict()

        self.config.relocate.start_iteration = adjust_step_for_world_size(
            trainer_config,
            self.config.relocate.start_iteration,
        )
        self.config.relocate.end_iteration = adjust_step_for_world_size(
            trainer_config,
            self.config.relocate.end_iteration,
        )
        self.config.relocate.frequency = adjust_step_for_world_size(
            trainer_config,
            self.config.relocate.frequency,
        )
        if self.config.relocate.max_invisible_steps is not None:
            self.config.relocate.max_invisible_steps = adjust_step_for_world_size(
                trainer_config, int(self.config.relocate.max_invisible_steps)
            )
        else:
            self.config.relocate.max_invisible_steps = None
        if self._should_track_visibility():
            # Register the sharded checkpoint names before Gaussian row counts are
            # known. Loading a checkpoint can then materialize each lazy counter.
            for layer_id in gaussians_nodes.keys():
                self.invisible_steps[layer_id] = nn.UninitializedParameter(requires_grad=False)
        log.info("MCMCStrategy/relocate:")
        log.info(f"    |─start_iteration={self.config.relocate.start_iteration}")
        log.info(f"    |─end_iteration={self.config.relocate.end_iteration}")
        log.info(f"    |─frequency={self.config.relocate.frequency}")
        if self.config.relocate.max_invisible_steps is not None:
            log.info(f"    └─max_invisible_steps={self.config.relocate.max_invisible_steps}")
        else:
            log.info("    └─max_invisible_steps=disabled")

        self.config.add.start_iteration = adjust_step_for_world_size(
            trainer_config,
            self.config.add.start_iteration,
        )
        self.config.add.end_iteration = adjust_step_for_world_size(
            trainer_config,
            self.config.add.end_iteration,
        )
        self.config.add.frequency = adjust_step_for_world_size(
            trainer_config,
            self.config.add.frequency,
        )
        self._max_n_gaussians = self.config.add.max_n_gaussians
        self.config.add.max_n_gaussians = adjust_gaussian_count_for_world_size(
            trainer_config,
            self.config.add.max_n_gaussians,
        )
        self.config.independent_layer_max_n_gaussians = {
            layer_id: adjust_gaussian_count_for_world_size(trainer_config, count)
            for layer_id, count in self.config.independent_layer_max_n_gaussians.items()
        }
        log.info("MCMCStrategy/add:")
        log.info(f"    |─start_iteration={self.config.add.start_iteration}")
        log.info(f"    |─end_iteration={self.config.add.end_iteration}")
        log.info(f"    |─frequency={self.config.add.frequency}")
        log.info(f"    └─max_n_gaussians={self.config.add.max_n_gaussians}")

        self.config.perturb.start_iteration = adjust_step_for_world_size(
            trainer_config,
            self.config.perturb.start_iteration,
        )
        self.config.perturb.end_iteration = adjust_step_for_world_size(
            trainer_config,
            self.config.perturb.end_iteration,
        )
        self.config.perturb.frequency = adjust_step_for_world_size(
            trainer_config,
            self.config.perturb.frequency,
        )

        log.info("MCMCStrategy/perturb:")
        log.info(f"    |─start_iteration={self.config.perturb.start_iteration}")
        log.info(f"    |─end_iteration={self.config.perturb.end_iteration}")
        log.info(f"    |─frequency={self.config.perturb.frequency}")
        log.info(f"    └─noise_lr={self.config.perturb.noise_lr.default}")

        self.config.max_scale_guard.start_iteration = adjust_step_for_world_size(
            trainer_config, self.config.max_scale_guard.start_iteration
        )
        self.config.max_scale_guard.end_iteration = adjust_step_for_world_size(
            trainer_config, self.config.max_scale_guard.end_iteration
        )
        self.config.max_scale_guard.frequency = adjust_step_for_world_size(
            trainer_config, self.config.max_scale_guard.frequency
        )
        global_limit_layer_ids = self.config.max_scale_guard.global_limit_layer_ids
        if self.config.max_scale_guard.enabled and global_limit_layer_ids is not None:
            unknown_layer_ids = set(global_limit_layer_ids) - set(gaussians_nodes.keys())
            if unknown_layer_ids:
                raise ValueError(
                    f"MCMC max_scale_guard.global_limit_layer_ids contains unknown layers: {sorted(unknown_layer_ids)}"
                )

        # Precompute the look up table for binomial coefficients (Eq 9 in the MCMC paper)
        self.binoms = nn.Buffer(
            torch.FloatTensor(
                [
                    [math.comb(n, k) if k <= n else 0 for k in range(self.config.binom_n_max)]
                    for n in range(self.config.binom_n_max)
                ],
            ),
            persistent=False,
        )
        require_mcmc_capabilities()

    @torch.no_grad()
    def maybe_initialize_buffers(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> None:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            # This setup-only consistency check runs in all-rank lockstep before
            # training, so it cannot race DDP or Gaussian backward collectives.
            max_n_gaussians = torch.distributed.nn.functional.all_reduce(
                torch.tensor(self.config.add.max_n_gaussians, dtype=torch.int32, device="cuda"),
                op=torch.distributed.ReduceOp.SUM,
            ).item()
            assert max_n_gaussians == self._max_n_gaussians, (
                f"max_n_gaussians={max_n_gaussians} != {self._max_n_gaussians}"
            )

        # Store perturb_noise_lr on CPU to avoid cudaStreamSynchronize when calling .item()
        self.perturb_noise_lr = {
            layer_id: self.config.perturb.noise_lr.layers.get(layer_id, self.config.perturb.noise_lr.default)
            for layer_id in gaussians_nodes.keys()
        }

        for layer_id in self.config.perturb.max_position_offset_m:
            if layer_id in self.config.exclude_layer_ids:
                continue
            if layer_id not in gaussians_nodes:
                continue
            gaussian_model = gaussians_nodes[layer_id]
            if isinstance(gaussian_model.positions, nn.UninitializedParameter):
                continue
            anchor = getattr(gaussian_model, "mcmc_position_anchor", None)
            if anchor is None or isinstance(anchor, nn.UninitializedBuffer):
                raise ValueError(f"{layer_id} must provide an initialized mcmc_position_anchor buffer")
            if anchor.shape != gaussian_model.positions.shape:
                raise ValueError(
                    f"{layer_id}.mcmc_position_anchor shape {tuple(anchor.shape)} "
                    f"does not match positions {tuple(gaussian_model.positions.shape)}"
                )

        if self._should_track_visibility():
            for layer_id, gaussian_model in gaussians_nodes.items():
                if isinstance(gaussian_model.positions, nn.UninitializedParameter):
                    continue
                num_gaussians = gaussian_model.get_num_gaussians()
                self._ensure_invisible_steps_buffer(layer_id, num_gaussians, gaussian_model.device)

        if self.config.max_scale_guard.enabled:
            for layer_id, gaussian_model in gaussians_nodes.items():
                if isinstance(gaussian_model, RigidGaussianModel):
                    self._validate_and_cache_rigid_scale_guard_metadata(layer_id, gaussian_model)

    def _should_track_visibility(self) -> bool:
        return self.config.relocate.max_invisible_steps is not None and self.config.relocate.max_invisible_steps > 0

    def configure_sharded_params_and_buffers(self) -> list[str]:
        names: list[str] = []
        for layer_id, counter in self.invisible_steps.items():
            names.append(f"invisible_steps.{layer_id}")
            if not isinstance(counter, nn.UninitializedParameter):
                BaseModel.mark_as_sharded(counter)
        return names

    def _validate_and_cache_rigid_scale_guard_metadata(
        self,
        layer_id: str,
        gaussian_model: RigidGaussianModel,
    ) -> bool:
        """Validate static rigid metadata once after Gaussian materialization."""
        multiplier = self.config.max_scale_guard.relative_max_scale_to_rigid_cuboid
        if multiplier is None or layer_id in self._rigid_scale_guard_cache:
            return True
        if isinstance(gaussian_model.scales, nn.UninitializedParameter) or isinstance(
            gaussian_model.gaussian_cuboid_ids, nn.UninitializedBuffer
        ):
            return False

        num_gaussians = gaussian_model.scales.shape[0]
        gaussian_cuboid_ids = gaussian_model.gaussian_cuboid_ids
        if gaussian_cuboid_ids.shape != (num_gaussians,):
            raise ValueError(
                f"{layer_id}: gaussian_cuboid_ids shape {tuple(gaussian_cuboid_ids.shape)} "
                f"does not match {num_gaussians} Gaussian rows"
            )

        cuboid_dims = gaussian_model.cuboid_tracks.cuboids_dims
        if cuboid_dims.ndim != 2 or cuboid_dims.shape[1] != 3:
            raise ValueError(f"{layer_id}: expected cuboid dimensions shaped [tracks, 3], got {cuboid_dims.shape}")
        if gaussian_cuboid_ids.numel() > 0:
            min_cuboid_id, max_cuboid_id = torch.aminmax(gaussian_cuboid_ids)
            min_cuboid_id_int = int(min_cuboid_id)
            max_cuboid_id_int = int(max_cuboid_id)
            if min_cuboid_id_int < 0 or max_cuboid_id_int >= cuboid_dims.shape[0]:
                raise ValueError(
                    f"{layer_id}: cuboid ids [{min_cuboid_id_int}, {max_cuboid_id_int}] "
                    f"fall outside {cuboid_dims.shape[0]} tracks"
                )
        if not bool(torch.isfinite(cuboid_dims).all()) or not bool((cuboid_dims > 0).all()):
            raise ValueError(f"{layer_id}: rigid scale guard requires finite positive cuboid dimensions")
        if gaussian_model.track_scales is not None and gaussian_model.track_scales.shape != cuboid_dims.shape:
            raise ValueError(
                f"{layer_id}: track_scales shape {tuple(gaussian_model.track_scales.shape)} "
                f"does not match cuboid dimensions {tuple(cuboid_dims.shape)}"
            )

        self._rigid_scale_guard_cache[layer_id] = _RigidScaleGuardCache(
            cuboid_dims.to(device=gaussian_model.scales.device, dtype=gaussian_model.scales.dtype)
            .amax(dim=-1, keepdim=True)
            .mul(multiplier)
        )
        return True

    @torch.no_grad()
    def permute_gaussian_rows(self, orders: dict[str, torch.Tensor]) -> None:
        self._permute_layer_buffers(self.invisible_steps, orders)

    def _ensure_invisible_steps_buffer(self, layer_id: str, num_gaussians: int, device: torch.device) -> None:
        if layer_id not in self.invisible_steps:
            self.invisible_steps[layer_id] = nn.UninitializedParameter(requires_grad=False)
        buffer = self.invisible_steps[layer_id]
        needs_init = isinstance(buffer, nn.UninitializedParameter) or buffer.numel() != num_gaussians
        if needs_init or buffer.device != device:
            self.invisible_steps[layer_id] = nn.Parameter(
                torch.zeros((num_gaussians,), dtype=torch.int32, device=device),
                requires_grad=False,
            )
        BaseModel.mark_as_sharded(self.invisible_steps[layer_id])

    def _get_gsplat_invisible_steps(
        self,
        selected: SelectedGaussianNodes,
    ) -> torch.nn.ParameterDict | None:
        if not self._should_track_visibility():
            return None

        for layer_id in selected.layer_ids:
            count = selected.layer_num_gaussians[layer_id]
            model = selected.gaussians_nodes[layer_id]
            self._ensure_invisible_steps_buffer(layer_id, count, model.device)
        return self.invisible_steps

    @torch.no_grad()
    def update_visibility_counters(
        self,
        *,
        results,
        visibility: torch.Tensor,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
    ) -> None:
        if not self._should_track_visibility():
            return
        if visibility.numel() == 0:
            return
        results_id = id(results)
        if self._last_visibility_update_id == results_id:
            return
        self._last_visibility_update_id = results_id

        offset = 0
        for layer_id, gaussian_model in gaussians_nodes.items():
            if isinstance(gaussian_model.positions, nn.UninitializedParameter):
                continue
            count = gaussian_model.get_num_gaussians()
            if count == 0:
                continue
            layer_visibility = visibility[offset : offset + count]
            if layer_visibility.numel() != count:
                log.warning(
                    "MCMCStrategy/visibility_counters: visibility length mismatch for layer '%s' (%d != %d)",
                    layer_id,
                    layer_visibility.numel(),
                    count,
                )
                return
            self._ensure_invisible_steps_buffer(layer_id, count, gaussians_nodes[layer_id].device)
            counters = self.invisible_steps[layer_id]
            seen_mask = layer_visibility > 0
            max_invisible_steps = self.config.relocate.max_invisible_steps
            if max_invisible_steps is None:
                raise RuntimeError("MCMC visibility tracking requires max_invisible_steps")
            update_invisible_steps(
                counters,
                seen_mask,
                max_invisible_steps=max_invisible_steps,
            )
            offset += count
        if offset != visibility.numel():
            log.warning(
                "MCMCStrategy/visibility_counters: visibility length mismatch total (%d != %d)",
                offset,
                visibility.numel(),
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
        for gaussian_model in gaussians_nodes.values():
            gaussian_model.raise_if_mcmc_publication_failed()

        for config, operation in self._topology_mutation_ops():
            if self._is_topology_mutation_due(global_step, config):
                operation(gaussians_nodes)

        if self._check_step_condition(
            step=global_step,
            start=self.config.perturb.start_iteration,
            end=self.config.perturb.end_iteration,
            freq=self.config.perturb.frequency,
        ):
            # Applies random perturbation to Gaussians per Gaussian layer.
            for layer_id, gaussian_model in gaussians_nodes.items():
                if layer_id in self.config.exclude_layer_ids:
                    continue
                with ScopedTimer(f"MCMCStrategy/perturb_gaussians.{layer_id}"):
                    self.perturb_gaussians(layer_id, gaussian_model)

        for layer_id, max_offset_m in self.config.perturb.max_position_offset_m.items():
            if layer_id not in self.config.exclude_layer_ids and layer_id in gaussians_nodes:
                self._clamp_position_offset(gaussians_nodes[layer_id], max_offset_m)

        if self._is_max_scale_guard_due(global_step):
            self.enforce_max_scale_guard(gaussians_nodes)

    @staticmethod
    def _protect_track_survivors(
        densities: torch.Tensor,
        track_mask: torch.Tensor,
        dead_mask: torch.Tensor,
        minimum_survivors: int,
    ) -> torch.Tensor:
        """Preserve live donors and revive dead rows only when needed to meet the quota."""

        protected_dead_mask = dead_mask.clone()
        alive_count = int((track_mask & ~dead_mask).sum().item())
        required_count = min(max(minimum_survivors - alive_count, 0), int((track_mask & dead_mask).sum().item()))
        if required_count == 0:
            return protected_dead_mask
        dead_indices = torch.where(track_mask & dead_mask)[0]
        order = torch.argsort(densities[dead_indices], descending=True, stable=True)
        protected_dead_mask[dead_indices[order[:required_count]]] = False
        return protected_dead_mask

    @torch.no_grad()
    def _relocate_same_track_layer(
        self,
        layer_id: str,
        gaussian_model: BaseGaussianModel,
    ) -> int:
        if layer_id in self.config.exclude_layer_ids:
            return 0
        if not isinstance(gaussian_model, RigidGaussianModel):
            raise TypeError(f"same-track MCMC layer '{layer_id}' must be a RigidGaussianModel")

        single_nodes: TypedModuleDict[BaseGaussianModel] = TypedModuleDict()
        single_nodes[layer_id] = gaussian_model
        selected = SelectedGaussianNodes(single_nodes, [])
        if not selected.layer_ids:
            return 0

        densities = selected.get_densities().view(-1)
        if not densities.is_cuda:
            raise RuntimeError("MCMC relocation requires CUDA tensors")
        visibility_state = self._get_gsplat_invisible_steps(selected)
        invisible_steps = None
        max_invisible_steps = None
        if visibility_state is not None and densities.numel() > 0:
            invisible_steps = visibility_state[layer_id].view(-1)
            max_invisible_steps = self.config.relocate.max_invisible_steps

        base_dead_mask = mcmc_dead_mask(
            densities,
            invisible_steps,
            min_opacity=self.config.opacity_threshold,
            max_invisible_steps=max_invisible_steps,
        )

        minimum_survivors = self.config.relocate.min_gaussians_per_track.get(layer_id, 1)
        relocation_groups: list[tuple[torch.Tensor, torch.Tensor]] = []
        for track_id in torch.unique(gaussian_model.gaussian_cuboid_ids):
            track_mask = gaussian_model.gaussian_cuboid_ids == track_id
            dead_mask = self._protect_track_survivors(
                densities,
                track_mask,
                base_dead_mask,
                minimum_survivors,
            )
            dead_mask &= track_mask
            candidate_indices = torch.where(track_mask & ~dead_mask)[0]
            if bool(dead_mask.any().item()) and candidate_indices.numel() > 0:
                relocation_groups.append((dead_mask, candidate_indices))

        if not relocation_groups:
            return 0

        # Sampling remains track-local, while the expensive row-aligned parameter
        # and optimizer-state rewrite is committed only once for the whole layer.
        return gsplat_relocate_gaussians_grouped(
            selected,
            groups=relocation_groups,
            binoms=self.binoms,
            min_opacity=self.config.opacity_threshold,
            generator=get_cuda_generator(densities.device),
            invisible_steps=visibility_state,
        )

    def _maybe_empty_cache_after_relocation(self, relocated: int) -> None:
        if relocated and self._empty_cache_after_parameter_update:
            torch.cuda.empty_cache()

    def _is_max_scale_guard_due(self, step: int) -> bool:
        if not self.config.max_scale_guard.enabled:
            return False
        config = self.config.max_scale_guard
        is_scheduled = self._check_step_condition(
            step=step,
            start=config.start_iteration,
            end=config.end_iteration,
            freq=config.frequency,
        )
        is_training_end = self._training_end_step is not None and step == self._training_end_step
        return is_scheduled or is_training_end

    def _bounded_scale_preactivation(
        self,
        layer_id: str,
        gaussian_model: BaseGaussianModel,
    ) -> torch.Tensor | None:
        """Return bounded raw scales for one layer, preserving valid values exactly."""
        config = self.config.max_scale_guard
        activated_scales = gaussian_model.get_scales()
        max_allowed_rendered_m: torch.Tensor | None = None

        global_layer_ids = config.global_limit_layer_ids
        if global_layer_ids is None or layer_id in global_layer_ids:
            global_limits_m: list[float] = []
            if config.relative_max_scale_to_scene_extent is not None:
                global_limits_m.append(config.relative_max_scale_to_scene_extent * gaussian_model.scene_extent)
            if config.absolute_max_scale_m is not None:
                global_limits_m.append(config.absolute_max_scale_m)
            if global_limits_m:
                max_allowed_rendered_m = activated_scales.new_full((activated_scales.shape[0], 1), min(global_limits_m))

        if config.relative_max_scale_to_rigid_cuboid is not None and isinstance(gaussian_model, RigidGaussianModel):
            if not self._validate_and_cache_rigid_scale_guard_metadata(layer_id, gaussian_model):
                raise ValueError(f"{layer_id}: cannot enforce rigid scale guard before Gaussian initialization")
            gaussian_cuboid_ids = gaussian_model.gaussian_cuboid_ids.long()
            rigid_max_rendered_m = self._rigid_scale_guard_cache[layer_id].max_rendered_scale_m[gaussian_cuboid_ids]
            max_allowed_rendered_m = (
                rigid_max_rendered_m
                if max_allowed_rendered_m is None
                else torch.minimum(max_allowed_rendered_m, rigid_max_rendered_m)
            )

        if max_allowed_rendered_m is None:
            return None

        max_allowed_base_m = max_allowed_rendered_m
        if isinstance(gaussian_model, RigidGaussianModel) and gaussian_model.track_scales is not None:
            owning_track_scales = gaussian_model.scale_activation(gaussian_model.track_scales)[
                gaussian_model.gaussian_cuboid_ids.long()
            ]
            max_allowed_base_m = max_allowed_rendered_m / owning_track_scales

        min_scale_m = 1.0e-8
        max_allowed_base_m = max_allowed_base_m.clamp_min(min_scale_m)
        invalid_or_oversized = (
            (~torch.isfinite(activated_scales))
            | (activated_scales < min_scale_m)
            | (activated_scales > max_allowed_base_m)
        )
        bounded_scales = torch.nan_to_num(
            activated_scales,
            nan=min_scale_m,
            posinf=torch.finfo(activated_scales.dtype).max,
            neginf=min_scale_m,
        ).clamp_min(min_scale_m)
        bounded_scales = torch.minimum(bounded_scales, max_allowed_base_m)
        replacement = gaussian_model.scale_activation_inv(bounded_scales)
        return torch.where(invalid_or_oversized, replacement, gaussian_model.scales)

    @torch.no_grad()
    def enforce_max_scale_guard(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> None:
        """Guarantee finite bounded activated scales without changing row counts."""
        if not self.config.max_scale_guard.enabled:
            return
        for layer_id, gaussian_model in gaussians_nodes.items():
            bounded_scales = self._bounded_scale_preactivation(layer_id, gaussian_model)
            if bounded_scales is not None:
                gaussian_model.scales.copy_(bounded_scales)

    @torch.no_grad()
    def serialization_state_dict_overrides(
        self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]
    ) -> dict[str, torch.Tensor]:
        """Return bounded scale tensors for serialization without changing live parameters."""
        if not self.config.max_scale_guard.enabled:
            return {}
        overrides: dict[str, torch.Tensor] = {}
        for layer_id, gaussian_model in gaussians_nodes.items():
            bounded_scales = self._bounded_scale_preactivation(layer_id, gaussian_model)
            if bounded_scales is not None:
                overrides[f"{layer_id}.scales"] = bounded_scales
        return overrides

    def _population_pools(
        self, gaussians_nodes: TypedModuleDict[BaseGaussianModel], excludes: list[str]
    ) -> Iterator[tuple[SelectedGaussianNodes, int]]:
        """Keep optional independent layers out of the default scene population."""
        independent = self.config.independent_layer_max_n_gaussians
        yield (
            SelectedGaussianNodes(gaussians_nodes, sorted(set(excludes) | set(independent))),
            self.config.add.max_n_gaussians,
        )
        for layer_id, cap in independent.items():
            if layer_id in gaussians_nodes and layer_id not in excludes:
                yield SelectedGaussianNodes(gaussians_nodes, [key for key in gaussians_nodes if key != layer_id]), cap

    @ScopedTimer("MCMCStrategy/relocate_gaussians")
    @torch.no_grad()
    def relocate_gaussians(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> None:
        """Relocate weak rows within the configured track-local or independent donor pools."""
        for gaussian_model in gaussians_nodes.values():
            gaussian_model.raise_if_mcmc_publication_failed()
        track_local_relocated = 0
        for layer_id in self.config.relocate.same_track_layer_ids:
            if layer_id in gaussians_nodes:
                track_local_relocated += self._relocate_same_track_layer(layer_id, gaussians_nodes[layer_id])

        global_excludes = sorted(set(self.config.exclude_layer_ids) | set(self.config.relocate.same_track_layer_ids))
        pool_relocated = 0
        for selected_gaussian_nodes, _ in self._population_pools(gaussians_nodes, global_excludes):
            pool_relocated += self._relocate_pool(selected_gaussian_nodes)
        self._maybe_empty_cache_after_relocation(track_local_relocated + pool_relocated)
        if self.config.print_stats and track_local_relocated:
            tqdm.write(f"Relocated {track_local_relocated} track-local gaussians")

    def _relocate_pool(self, selected_gaussian_nodes: SelectedGaussianNodes) -> int:
        """Relocate one isolated pool, dispatching grouped barriers to cuboid-local donors."""
        if not selected_gaussian_nodes.layer_ids:
            return 0
        if self.config.cuboid_grouped_layer_ids and any(
            layer_id in self.config.cuboid_grouped_layer_ids for layer_id in selected_gaussian_nodes.layer_ids
        ):
            return self._cuboid_topology(selected_gaussian_nodes, relocation=True)

        densities = selected_gaussian_nodes.get_densities().view(-1)
        if not densities.is_cuda:
            raise RuntimeError("MCMC relocation requires CUDA tensors")
        visibility_state = self._get_gsplat_invisible_steps(selected_gaussian_nodes)
        max_invisible_steps = None
        if visibility_state is not None:
            max_invisible_steps = self.config.relocate.max_invisible_steps

        activated_densities_by_owner = {
            layer_id: densities[
                selected_gaussian_nodes.layer_offsets[layer_id] : selected_gaussian_nodes.layer_offsets[layer_id]
                + selected_gaussian_nodes.layer_num_gaussians[layer_id]
            ]
            for layer_id in selected_gaussian_nodes.layer_ids
        }
        original_dead_by_owner = {
            layer_id: mcmc_dead_mask(
                owner_densities,
                None if visibility_state is None else visibility_state[layer_id],
                min_opacity=self.config.opacity_threshold,
                max_invisible_steps=max_invisible_steps,
            )
            for layer_id, owner_densities in activated_densities_by_owner.items()
        }
        n_dead_gaussians = int(torch.stack([dead.sum() for dead in original_dead_by_owner.values()]).sum().item())
        if n_dead_gaussians == 0 or n_dead_gaussians == densities.numel():
            return 0

        effective_dead_by_owner = dict(original_dead_by_owner)
        for layer_id, original_dead in original_dead_by_owner.items():
            effective_dead = original_dead.clone()
            # Keep one deterministic survivor only when this owner is entirely
            # dead. Express the condition on-device to avoid a host sync per
            # owner; torch.argmax returns the lowest index on a tie.
            survivor = torch.argmax(activated_densities_by_owner[layer_id])
            effective_dead[survivor] &= ~original_dead.all()
            effective_dead_by_owner[layer_id] = effective_dead
        source_eligible_by_owner = {
            layer_id: ~original_dead for layer_id, original_dead in original_dead_by_owner.items()
        }

        with ScopedTimer("MCMCStrategy/relocate_gaussians/gsplat_relocate"):
            changed = gsplat_relocate_gaussians(
                selected_gaussian_nodes,
                dead_by_owner=effective_dead_by_owner,
                source_eligible_by_owner=source_eligible_by_owner,
                binoms=self.binoms,
                min_opacity=self.config.opacity_threshold,
                generator=get_cuda_generator(densities.device),
                invisible_steps=visibility_state,
            )
        if not changed:
            log.warning(
                "MCMCStrategy/relocate_gaussians: gsplat returned no relocation for %d dead Gaussians; skipping",
                n_dead_gaussians,
            )
            return 0

        if self.config.print_stats:
            tqdm.write(f"Relocated {n_dead_gaussians} gaussians within pool {selected_gaussian_nodes.layer_ids}")
            gaussians_nodes = selected_gaussian_nodes.gaussians_nodes
            for layer_id, layer_prev_num_gaussians in selected_gaussian_nodes.layer_num_gaussians.items():
                tqdm.write(
                    f"{layer_id}: old count: {layer_prev_num_gaussians}, new count: {gaussians_nodes[layer_id].get_num_gaussians()} "
                    f"({(gaussians_nodes[layer_id].get_num_gaussians() - layer_prev_num_gaussians) / max(1, layer_prev_num_gaussians) * 100:.2f}%)"
                )
        return int(changed)

    @ScopedTimer("MCMCStrategy/add_new_gaussians")
    @torch.no_grad()
    def add_new_gaussians(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> None:
        """Grow each eligible population without sharing independent layers' point budgets."""
        for gaussian_model in gaussians_nodes.values():
            gaussian_model.raise_if_mcmc_publication_failed()
        addition_excludes = sorted(set(self.config.exclude_layer_ids) | set(self.config.add.fixed_count_layer_ids))
        for selected_gaussian_nodes, cap in self._population_pools(gaussians_nodes, addition_excludes):
            self._add_to_pool(selected_gaussian_nodes, cap)

    def _add_to_pool(self, selected_gaussian_nodes: SelectedGaussianNodes, cap: int) -> None:
        """Apply native five-percent growth within this pool's explicit Gaussian cap."""
        if self.config.cuboid_grouped_layer_ids and any(
            layer_id in self.config.cuboid_grouped_layer_ids for layer_id in selected_gaussian_nodes.layer_ids
        ):
            added = self._cuboid_topology(selected_gaussian_nodes, relocation=False)
            if added and self._empty_cache_after_parameter_update:
                torch.cuda.empty_cache()
            return
        # Get the current number of gaussians
        current_num_gaussians = sum(selected_gaussian_nodes.layer_num_gaussians.values())
        target_num_gaussians = min(cap, int(1.05 * current_num_gaussians))
        num_gaussians_to_add = max(0, target_num_gaussians - current_num_gaussians)
        if num_gaussians_to_add == 0:
            return

        densities = selected_gaussian_nodes.get_densities().view(-1)
        if not densities.is_cuda:
            raise RuntimeError("MCMC addition requires CUDA tensors")
        visibility_state = self._get_gsplat_invisible_steps(selected_gaussian_nodes)
        with ScopedTimer("MCMCStrategy/add_new_gaussians/gsplat_add"):
            changed = gsplat_add_gaussians(
                selected_gaussian_nodes,
                count=num_gaussians_to_add,
                binoms=self.binoms,
                min_opacity=self.config.opacity_threshold,
                generator=get_cuda_generator(densities.device),
                invisible_steps=visibility_state,
            )
        if not changed:
            log.warning(
                "MCMCStrategy/add_new_gaussians: gsplat returned no addition for %d requested Gaussians; skipping",
                num_gaussians_to_add,
            )
            return

        if self._empty_cache_after_parameter_update:
            torch.cuda.empty_cache()

        if self.config.print_stats:
            gaussians_nodes = selected_gaussian_nodes.gaussians_nodes
            for layer_id, old_count in selected_gaussian_nodes.layer_num_gaussians.items():
                added = gaussians_nodes[layer_id].get_num_gaussians() - old_count
                tqdm.write(f"{layer_id}: Added {added} ({added / max(1, old_count) * 100:.2f}%) gaussians")

    def _cuboid_topology(self, selected: SelectedGaussianNodes, *, relocation: bool) -> int:
        """Dispatch one validated independent barrier pool to grouped row publication."""
        from gsplat.strategy import MCMCOperation

        from nurec.models.gaussians.strategies.cuboid_mcmc import cuboid_topology

        if len(selected.layer_ids) != 1:
            raise ValueError("Grouped cuboids must have an independent layer pool")
        layer_id = selected.layer_ids[0]
        visibility = self._get_gsplat_invisible_steps(selected)
        changed = cuboid_topology(
            selected,
            operation=MCMCOperation.RELOCATION if relocation else MCMCOperation.ADDITION,
            cap=self.config.independent_layer_max_n_gaussians[layer_id],
            binoms=self.binoms,
            min_opacity=self.config.opacity_threshold,
            generator=get_cuda_generator(selected.gaussians_nodes[layer_id].positions.device),
            invisible_steps=visibility,
            max_invisible_steps=self.config.relocate.max_invisible_steps if visibility is not None else None,
        )
        if self.config.print_stats:
            tqdm.write(f"Cuboid-local {layer_id}: {'relocated' if relocation else 'added'} {changed} Gaussians")
        return changed

    @staticmethod
    @torch.no_grad()
    def _clamp_position_offset(gaussian_model: BaseGaussianModel, max_offset_m: float) -> None:
        anchor = getattr(gaussian_model, "mcmc_position_anchor", None)
        if anchor is None or isinstance(anchor, nn.UninitializedBuffer):
            raise ValueError("Configured MCMC position clamp requires mcmc_position_anchor")
        if anchor.shape != gaussian_model.positions.shape:
            raise ValueError(
                f"MCMC position anchor shape {tuple(anchor.shape)} does not match "
                f"positions {tuple(gaussian_model.positions.shape)}"
            )
        offset = gaussian_model.positions - anchor
        distance = torch.linalg.vector_norm(offset, dim=-1, keepdim=True)
        scale = torch.clamp(max_offset_m / distance.clamp_min(torch.finfo(distance.dtype).eps), max=1.0)
        gaussian_model.positions.copy_(anchor + offset * scale)

    @ScopedTimer("MCMCStrategy/perturb_gaussians")
    @torch.no_grad()
    def perturb_gaussians(self, layer_id: str, gaussian_model: BaseGaussianModel) -> None:
        """Perturb one layer's live positions through gsplat's tensor API."""
        gaussian_model.raise_if_mcmc_publication_failed()
        current_lr = 0.0
        for opp in gaussian_model.optimizers:
            for param_group in opp["optimizer"].param_groups:
                if "name" in param_group and param_group["name"].split(".")[-1] == "positions":
                    current_lr = param_group["lr"]
                    break

        assert current_lr > 0.0, "Current learning rate is not set"
        assert self.perturb_noise_lr is not None, "Perturb noise learning rate is not set"

        # Compute effective learning rate (perturb_noise_lr is already a float, no .item() needed)
        effective_lr = current_lr * self.perturb_noise_lr[layer_id]

        cuboid_dimensions = None
        if (not self.config.perturb.move_outside_of_cuboid) and isinstance(gaussian_model, RigidGaussianModel):
            cuboid_dimensions = gaussian_model.cuboid_tracks.cuboids_dims[gaussian_model.gaussian_cuboid_ids]

        reject_barrier_noise = isinstance(gaussian_model, ConstructionBarriersGaussianModel)
        # Native noise stays in world/NuRec space. Filter local components on a
        # scratch proposal before publishing, never clamp a completed update.
        positions = gaussian_model.positions.clone() if reject_barrier_noise else gaussian_model.positions
        perturb_positions(
            positions=positions,
            quats=gaussian_model.rotations,
            scales=gaussian_model.scales,
            opacities=gaussian_model.densities.squeeze(-1),
            noise_scale=effective_lr,
            generator=get_cuda_generator(gaussian_model.positions.device),
            cuboid_dimensions=cuboid_dimensions,
        )
        if isinstance(gaussian_model, ConstructionBarriersGaussianModel):
            gaussian_model.accept_cuboid_perturbation(positions)


BaseGaussianStrategy.register_to_gaussian_strategy_factory("mcmc", MCMCStrategy)
