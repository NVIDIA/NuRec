# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import math
import struct

from abc import abstractmethod
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import torch

from torch import nn
from tqdm import tqdm

from nurec.config.model import BaseMCMCStrategyConfig, MCMCStrategyConfig, TamingGSMCMCStrategyConfig
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
    update_invisible_steps_segmented,
)
from nurec.models.gaussians.strategies.gsplat_mcmc_adapter import (
    relocate_gaussians as gsplat_relocate_gaussians,
)
from nurec.models.gaussians.strategies.gsplat_mcmc_adapter import (
    relocate_gaussians_and_reindex as gsplat_relocate_gaussians_and_reindex,
)
from nurec.models.gaussians.strategies.gsplat_mcmc_adapter import (
    relocate_gaussians_grouped as gsplat_relocate_gaussians_grouped,
)
from nurec.models.gaussians.strategies.selected_gaussian_nodes import SelectedGaussianNodes
from nurec.models.nn_extensions import TypedModuleDict
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.cuda_rng import get_cuda_generator
from nurec.utils.profiling import ScopedTimer
from nurec.utils.torch_compile import TorchCompile
from nurec.utils.trainer import TrainerConfig, adjust_gaussian_count_for_world_size, adjust_step_for_world_size


log = logging.getLogger(__name__)


_TAMINGGS_VISIBILITY_COUNTER_CAP = 2**30


_TopologyMutation = tuple[
    BaseMCMCStrategyConfig.MCMCRelocationConfig | BaseMCMCStrategyConfig.MCMCAdditionConfig,
    Callable[..., object],
]


def _normalize_taminggs_components(components: torch.Tensor, denominators: torch.Tensor, clip: float) -> torch.Tensor:
    return torch.nan_to_num(components / denominators, nan=0.0, posinf=clip, neginf=0.0).clamp(0.0, clip)


def _score_taminggs_components(
    components: torch.Tensor,
    denominators: torch.Tensor,
    coefficients: tuple[float, ...],
    clip: float,
    minimum: float,
) -> torch.Tensor:
    normalized = _normalize_taminggs_components(components, denominators, clip)
    score = torch.zeros(components.shape[0], device=components.device, dtype=components.dtype)
    for column, coefficient in enumerate(coefficients):
        score = score + coefficient * normalized[:, column]
    return score + minimum


_finish_taminggs_scores = TorchCompile.conditional(fullgraph=True, dynamic=True)(_score_taminggs_components)


def _mask_taminggs_addition(
    score: torch.Tensor, opacity: torch.Tensor, threshold: float, owner_sizes: tuple[int, ...]
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    masked = torch.where(opacity > threshold, score, 0.0)
    masses = []
    offset = 0
    for count in owner_sizes:
        masses.append(masked[offset : offset + count].sum())
        offset += count
    return masked, tuple(masses)


_mask_taminggs_addition_weights = TorchCompile.conditional(fullgraph=True, dynamic=True)(_mask_taminggs_addition)


@TorchCompile.conditional(fullgraph=True, dynamic=True)
def _finish_taminggs_addition_scores(
    components: torch.Tensor,
    denominators: torch.Tensor,
    coefficients: tuple[float, ...],
    clip: float,
    minimum: float,
    opacity: torch.Tensor,
    threshold: float,
    owner_sizes: tuple[int, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    score = _score_taminggs_components(components, denominators, coefficients, clip, minimum)
    return _mask_taminggs_addition(score, opacity, threshold, owner_sizes)


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

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for name, registered_parameter in self._parameters.items():
            saved = state_dict.get(prefix + name)
            if isinstance(registered_parameter, nn.UninitializedParameter) and isinstance(saved, torch.Tensor):
                registered_parameter.materialize(saved.shape, device=saved.device, dtype=saved.dtype)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _RigidScaleGuardCache(nn.Module):
    """Derived rigid scale limits that follow strategy device transfers."""

    def __init__(self, max_rendered_scale_m: torch.Tensor) -> None:
        super().__init__()
        self.max_rendered_scale_m = nn.Buffer(max_rendered_scale_m, persistent=False)


class BaseMCMCStrategy(BaseGaussianStrategy):
    """Shared MCMC topology, visibility-state and perturbation mechanics."""

    config: BaseMCMCStrategyConfig
    # The visibility guard's state (see __init__): class-level defaults as well, so a strategy built without __init__
    # (tests use MCMCStrategy.__new__) reaches the counter update and the step start without missing attributes.
    _last_visibility_update_key: tuple[int, int] | None = None
    _visibility_update_step_count: int = 0

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
        config: BaseMCMCStrategyConfig.MCMCRelocationConfig | BaseMCMCStrategyConfig.MCMCAdditionConfig,
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
        config: BaseMCMCStrategyConfig,
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
        self._validate_variant_config()
        self.config.relocate.min_gaussians_per_track = {
            layer_id: max(1, adjust_gaussian_count_for_world_size(trainer_config, minimum))
            for layer_id, minimum in self.config.relocate.min_gaussians_per_track.items()
        }

        # Updates the visibility counters once per render: the scale and the density loss both hand the step's render
        # result to update_visibility_counters. The key is the result's Python id together with the strategy's own count
        # of training steps started (on_train_step_start; not the trainer's global_step, which the step-start hook does
        # not receive), because CPython hands a freed object's address to the next allocation of its size: keyed on the
        # id alone, the next step's result could carry the last one's id and its counter update would be skipped (about
        # 0.2 percent of the steps).
        self._last_visibility_update_key = None
        self._visibility_update_step_count = 0
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
        self._initialize_variant_state(trainer_config)
        log.info("MCMCStrategy/add:")
        log.info(f"    |─start_iteration={self.config.add.start_iteration}")
        log.info(f"    |─end_iteration={self.config.add.end_iteration}")
        log.info(f"    |─frequency={self.config.add.frequency}")
        log.info(f"    |─max_n_gaussians={self.config.add.max_n_gaussians}")
        log.info(f"    └─strategy={self.config.name}")

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
        collective_device = next(
            (
                parameter.device
                for gaussian_model in gaussians_nodes.values()
                for parameter in gaussian_model.parameters(recurse=False)
                if not isinstance(parameter, nn.UninitializedParameter)
            ),
            self.binoms.device,
        )
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            if torch.distributed.get_backend() == torch.distributed.Backend.NCCL:
                collective_device = torch.device("cuda", torch.cuda.current_device())
            # This setup-only consistency check runs in all-rank lockstep before
            # training, so it cannot race DDP or Gaussian backward collectives.
            max_n_gaussians = torch.distributed.nn.functional.all_reduce(
                torch.tensor(self.config.add.max_n_gaussians, dtype=torch.int32, device=collective_device),
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

        self._initialize_population(gaussians_nodes, collective_device)

        if self.config.max_scale_guard.enabled:
            for layer_id, gaussian_model in gaussians_nodes.items():
                if isinstance(gaussian_model, RigidGaussianModel):
                    self._validate_and_cache_rigid_scale_guard_metadata(layer_id, gaussian_model)

    def _relocation_uses_visibility(self) -> bool:
        return self.config.relocate.max_invisible_steps is not None and self.config.relocate.max_invisible_steps > 0

    def _visibility_counter_cap(self) -> int:
        """Return a saturation bound independent of relocation eligibility."""
        if self._relocation_uses_visibility():
            assert self.config.relocate.max_invisible_steps is not None
            return self.config.relocate.max_invisible_steps
        raise RuntimeError("MCMC visibility counters are not enabled")

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

    def on_train_step_start(self) -> None:
        # A new training step: its render is new to the visibility guard, whatever its id. The step start runs on every
        # training step, also while the strategy is frozen and during controller distillation, which both skip the
        # strategy's batch end.
        self._visibility_update_step_count += 1
        super().on_train_step_start()

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
        # A repeated call for the render already counted in this step (see __init__).
        key = (id(results), self._visibility_update_step_count)
        if key == self._last_visibility_update_key:
            return
        self._last_visibility_update_key = key

        self._apply_visibility_counters(visibility, gaussians_nodes)

    def _apply_visibility_counters(
        self,
        visibility: torch.Tensor,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
    ) -> None:
        if self._strict_visibility():
            expected_rows = sum(
                gaussian_model.get_num_gaussians()
                for gaussian_model in gaussians_nodes.values()
                if not isinstance(gaussian_model.positions, nn.UninitializedParameter)
            )
            if visibility.numel() != expected_rows:
                raise RuntimeError(
                    "MCMCStrategy/visibility_counters: visibility length mismatch "
                    f"({visibility.numel()} != {expected_rows} Gaussian rows)"
                )
        if visibility.numel() == 0:
            return

        # Compare once; node slices share this mask without additional launches.
        seen = visibility > 0
        max_invisible_steps = self._visibility_counter_cap()
        offset = 0
        segments: list[torch.Tensor] = []
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
                if segments:
                    update_invisible_steps_segmented(segments, seen[:offset], max_invisible_steps=max_invisible_steps)
                return
            self._ensure_invisible_steps_buffer(layer_id, count, gaussians_nodes[layer_id].device)
            counters = self.invisible_steps[layer_id]
            segments.append(counters)
            offset += count
        if segments:
            update_invisible_steps_segmented(segments, seen[:offset], max_invisible_steps=max_invisible_steps)
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
            gaussian_model.raise_if_publication_failed()

        due_operations = tuple(
            (config, operation)
            for config, operation in self._topology_mutation_ops()
            if self._is_topology_mutation_due(global_step, config)
        )
        source_weights = self._co_scheduled_source_weights(due_operations, gaussians_nodes)
        carry_scores = source_weights is not None
        for config, operation in due_operations:
            if config is self.config.add:
                operation(gaussians_nodes, global_step=global_step, source_weights_by_owner=source_weights)
            elif config is self.config.relocate and carry_scores:
                reindexed_source_weights = operation(gaussians_nodes, reindex_values_by_owner=source_weights)
                assert isinstance(reindexed_source_weights, dict)
                source_weights = reindexed_source_weights
            else:
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
        if visibility_state is not None and self._relocation_uses_visibility() and densities.numel() > 0:
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
    def relocate_gaussians(
        self,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
        *,
        reindex_values_by_owner: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Keep the proxy in the shared population and native independent pools isolated."""
        for gaussian_model in gaussians_nodes.values():
            gaussian_model.raise_if_publication_failed()
        self._validate_reindex_values(reindex_values_by_owner)
        reindexed_values = None if reindex_values_by_owner is None else dict(reindex_values_by_owner)
        track_local_relocated = 0
        for layer_id in self.config.relocate.same_track_layer_ids:
            if layer_id in gaussians_nodes:
                track_local_relocated += self._relocate_same_track_layer(layer_id, gaussians_nodes[layer_id])
        global_excludes = sorted(set(self.config.exclude_layer_ids) | set(self.config.relocate.same_track_layer_ids))
        pool_relocated = 0
        for pool_index, (selected, _) in enumerate(self._population_pools(gaussians_nodes, global_excludes)):
            if pool_index == 0:
                pool_relocated += self._relocate_shared_pool(selected, reindexed_values)
            else:
                pool_relocated += self._relocate_pool(selected)
        self._maybe_empty_cache_after_relocation(track_local_relocated + pool_relocated)
        if self.config.print_stats and track_local_relocated:
            tqdm.write(f"Relocated {track_local_relocated} track-local gaussians")
        return reindexed_values

    def _addition_selected_nodes(
        self,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
    ) -> SelectedGaussianNodes:
        return SelectedGaussianNodes(gaussians_nodes, sorted(self._addition_excluded_layer_ids()))

    def _addition_excluded_layer_ids(self) -> set[str]:
        return (
            set(self.config.exclude_layer_ids)
            | set(self.config.add.fixed_count_layer_ids)
            | set(self.config.independent_layer_max_n_gaussians)
        )

    @ScopedTimer("MCMCStrategy/add_new_gaussians")
    @torch.no_grad()
    def add_new_gaussians(
        self,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
        *,
        global_step: int | None = None,
        source_weights_by_owner: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Grow independent populations natively without consuming the shared proxy budget."""
        for gaussian_model in gaussians_nodes.values():
            gaussian_model.raise_if_publication_failed()
        self._validate_addition_weights(source_weights_by_owner)
        excludes = sorted(set(self.config.exclude_layer_ids) | set(self.config.add.fixed_count_layer_ids))
        for pool_index, (selected, cap) in enumerate(self._population_pools(gaussians_nodes, excludes)):
            if pool_index == 0:
                self._add_shared_pool(selected, cap, global_step, source_weights_by_owner)
            else:
                self._add_to_pool(selected, cap)

    def _relocate_pool(
        self,
        selected_gaussian_nodes: SelectedGaussianNodes,
        *,
        taminggs_source_weights: Mapping[str, torch.Tensor] | None = None,
        reindex_values_by_owner: dict[str, torch.Tensor] | None = None,
    ) -> int:
        """Relocate one isolated pool with optional shared-population proxy scores."""
        if not selected_gaussian_nodes.layer_ids:
            return 0
        if self.config.cuboid_grouped_layer_ids and any(
            layer_id in self.config.cuboid_grouped_layer_ids for layer_id in selected_gaussian_nodes.layer_ids
        ):
            return self._cuboid_topology(selected_gaussian_nodes, relocation=True)
        gaussians_nodes = selected_gaussian_nodes.gaussians_nodes
        densities = selected_gaussian_nodes.get_densities().view(-1)
        if not densities.is_cuda:
            raise RuntimeError("MCMC relocation requires CUDA tensors")
        visibility_state = self._get_gsplat_invisible_steps(selected_gaussian_nodes)
        max_invisible_steps = None
        if visibility_state is not None and self._relocation_uses_visibility():
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
                None if max_invisible_steps is None or visibility_state is None else visibility_state[layer_id],
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
        taminggs_kwargs: dict[str, Any] = {}
        if taminggs_source_weights is not None:
            taminggs_kwargs["source_weights_by_owner"] = taminggs_source_weights

        with ScopedTimer("MCMCStrategy/relocate_gaussians/gsplat_relocate"):
            if reindex_values_by_owner is None:
                changed = gsplat_relocate_gaussians(
                    selected_gaussian_nodes,
                    dead_by_owner=effective_dead_by_owner,
                    source_eligible_by_owner=source_eligible_by_owner,
                    binoms=self.binoms,
                    min_opacity=self.config.opacity_threshold,
                    generator=get_cuda_generator(densities.device),
                    invisible_steps=visibility_state,
                    **taminggs_kwargs,
                )
                reindexed_values = None
            else:
                source_weights_by_owner = taminggs_kwargs.get("source_weights_by_owner")
                if source_weights_by_owner is None:
                    raise RuntimeError("row reindexing is only supported by TamingGS-inspired proxy relocation")
                changed, reindexed_values = gsplat_relocate_gaussians_and_reindex(
                    selected_gaussian_nodes,
                    dead_by_owner=effective_dead_by_owner,
                    source_eligible_by_owner=source_eligible_by_owner,
                    binoms=self.binoms,
                    min_opacity=self.config.opacity_threshold,
                    generator=get_cuda_generator(densities.device),
                    invisible_steps=visibility_state,
                    source_weights_by_owner=source_weights_by_owner,
                    reindex_values_by_owner=reindex_values_by_owner,
                )
        if not changed:
            log.warning(
                "MCMCStrategy/relocate_gaussians: gsplat returned no relocation for %d dead Gaussians; skipping",
                n_dead_gaussians,
            )
            return 0

        if self.config.print_stats:
            tqdm.write(f"Relocated {n_dead_gaussians} gaussians within pool {selected_gaussian_nodes.layer_ids}")
            for layer_id, layer_prev_num_gaussians in selected_gaussian_nodes.layer_num_gaussians.items():
                tqdm.write(
                    f"{layer_id}: old count: {layer_prev_num_gaussians}, new count: {gaussians_nodes[layer_id].get_num_gaussians()} "
                    f"({(gaussians_nodes[layer_id].get_num_gaussians() - layer_prev_num_gaussians) / max(1, layer_prev_num_gaussians) * 100:.2f}%)"
                )
        if reindex_values_by_owner is not None:
            assert reindexed_values is not None
            reindex_values_by_owner.update(reindexed_values)
        return int(changed)

    def _add_to_pool(
        self,
        selected_gaussian_nodes: SelectedGaussianNodes,
        cap: int,
        *,
        use_taminggs: bool = False,
        global_step: int | None = None,
        source_weights_by_owner: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Grow one pool using either its native cap or the shared proxy schedule."""
        if self.config.cuboid_grouped_layer_ids and any(
            layer_id in self.config.cuboid_grouped_layer_ids for layer_id in selected_gaussian_nodes.layer_ids
        ):
            added = self._cuboid_topology(selected_gaussian_nodes, relocation=False)
            if added and self._empty_cache_after_parameter_update:
                torch.cuda.empty_cache()
            return
        gaussians_nodes = selected_gaussian_nodes.gaussians_nodes
        # Get the current number of gaussians.
        current_num_gaussians = sum(selected_gaussian_nodes.layer_num_gaussians.values())
        target_num_gaussians, taminggs_kwargs = self._addition_target(
            selected_gaussian_nodes, cap, current_num_gaussians, use_taminggs, global_step, source_weights_by_owner
        )
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
                **taminggs_kwargs,
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
            max_invisible_steps=(
                self.config.relocate.max_invisible_steps
                if visibility is not None and self._relocation_uses_visibility()
                else None
            ),
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
        gaussian_model.raise_if_publication_failed()
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

    def _validate_variant_config(self) -> None:
        pass

    def _initialize_variant_state(self, trainer_config: TrainerConfig) -> None:
        pass

    def _initialize_population(
        self, gaussians_nodes: TypedModuleDict[BaseGaussianModel], collective_device: torch.device
    ) -> None:
        pass

    def _addition_target(
        self,
        selected_gaussian_nodes: SelectedGaussianNodes,
        cap: int,
        current_num_gaussians: int,
        use_taminggs: bool,
        global_step: int | None,
        source_weights_by_owner: Mapping[str, torch.Tensor] | None,
    ) -> tuple[int, dict[str, Any]]:
        if source_weights_by_owner is not None or use_taminggs:
            raise ValueError("source_weights_by_owner requires TamingGS-inspired proxy addition")
        return min(cap, int(1.05 * current_num_gaussians)), {}

    @abstractmethod
    def _should_track_visibility(self) -> bool: ...

    @abstractmethod
    def _strict_visibility(self) -> bool: ...

    @abstractmethod
    def _co_scheduled_source_weights(
        self, due_operations: tuple[_TopologyMutation, ...], gaussians_nodes: TypedModuleDict[BaseGaussianModel]
    ) -> dict[str, torch.Tensor] | None: ...

    @abstractmethod
    def _validate_reindex_values(self, values: Mapping[str, torch.Tensor] | None) -> None: ...

    @abstractmethod
    def _relocate_shared_pool(
        self, selected: SelectedGaussianNodes, reindexed_values: dict[str, torch.Tensor] | None
    ) -> int: ...

    @abstractmethod
    def _validate_addition_weights(self, values: Mapping[str, torch.Tensor] | None) -> None: ...

    @abstractmethod
    def _add_shared_pool(
        self,
        selected: SelectedGaussianNodes,
        cap: int,
        global_step: int | None,
        weights: Mapping[str, torch.Tensor] | None,
    ) -> None: ...


class MCMCStrategy(BaseMCMCStrategy):
    """Ordinary opacity-weighted MCMC densification."""

    config: MCMCStrategyConfig

    def _should_track_visibility(self) -> bool:
        return self._relocation_uses_visibility()

    def _strict_visibility(self) -> bool:
        return False

    def _co_scheduled_source_weights(
        self, due_operations: tuple[_TopologyMutation, ...], gaussians_nodes: TypedModuleDict[BaseGaussianModel]
    ) -> dict[str, torch.Tensor] | None:
        return None

    def _validate_reindex_values(self, values: Mapping[str, torch.Tensor] | None) -> None:
        if values is not None:
            raise ValueError("row reindexing requires TamingGS-inspired proxy relocation")

    def _relocate_shared_pool(
        self, selected: SelectedGaussianNodes, reindexed_values: dict[str, torch.Tensor] | None
    ) -> int:
        return self._relocate_pool(selected)

    def _validate_addition_weights(self, values: Mapping[str, torch.Tensor] | None) -> None:
        if values is not None:
            raise ValueError("source_weights_by_owner requires TamingGS-inspired proxy addition")

    def _add_shared_pool(
        self,
        selected: SelectedGaussianNodes,
        cap: int,
        global_step: int | None,
        weights: Mapping[str, torch.Tensor] | None,
    ) -> None:
        self._add_to_pool(selected, cap)


class TamingGSMCMCStrategy(BaseMCMCStrategy):
    """TamingGS-inspired global budget schedule and saliency donor policy.

    An initialization above the growth budget is preserved, with the saved
    initial population serving as the schedule's effective upper bound.
    """

    config: TamingGSMCMCStrategyConfig
    _taminggs_initial_count: torch.Tensor
    _taminggs_initial_counts_by_rank: torch.Tensor
    _taminggs_no_growth_warning_emitted: bool
    _taminggs_training_world_size: int

    def _taminggs_eligible_population_is_materialized(
        self,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
        collective_device: torch.device,
    ) -> bool:
        """Whether the TamingGS-inspired proxy can safely capture its initial population anchor.

        Checkpoint-based validation and resume call ``setup`` before loading the
        state dict. Eligible Gaussian layers are still lazy at that point, and
        the checkpoint will restore the proxy's persistent population anchor along
        with the Gaussian tensors. Leave the sentinel untouched until either
        those tensors are materialized or the checkpoint state is loaded.
        """
        addition_excludes = self._addition_excluded_layer_ids()
        locally_materialized = all(
            layer_id in addition_excludes
            or not isinstance(getattr(gaussian_model, "positions", None), nn.UninitializedParameter)
            for layer_id, gaussian_model in gaussians_nodes.items()
        )
        materialized = torch.tensor(locally_materialized, dtype=torch.int32, device=collective_device)
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            # All ranks must make the same decision before the allocation
            # all-gather below. A rank-local decision could deadlock when
            # deferred initialization materializes eligible layers unevenly.
            torch.distributed.all_reduce(materialized, op=torch.distributed.ReduceOp.MIN)
        return bool(materialized.item())

    def _taminggs_schedule_events(self) -> tuple[int, int, int]:
        """Return the first, last, and number of constructive growth events."""
        add = self.config.add
        first_event = (add.start_iteration // add.frequency + 1) * add.frequency
        effective_end = add.end_iteration - 1
        if self._training_end_step is not None:
            effective_end = min(effective_end, self._training_end_step)
        last_event = effective_end // add.frequency * add.frequency
        if first_event > last_event:
            if not self._taminggs_no_growth_warning_emitted:
                log.warning(
                    "TamingGS has no scheduled addition event before training ends; keeping the initial population: "
                    "start=%d, end=%d, frequency=%d, training_end=%s",
                    add.start_iteration,
                    add.end_iteration,
                    add.frequency,
                    self._training_end_step,
                )
                self._taminggs_no_growth_warning_emitted = True
            return first_event, last_event, 0
        event_count = (last_event - first_event) // add.frequency + 1
        return first_event, last_event, event_count

    def _taminggs_global_target_count(self, global_step: int) -> int:
        """Evaluate the proxy's global integer quadratic ease-out schedule."""
        if int(self._taminggs_initial_count.item()) < 0:
            raise RuntimeError("TamingGS-inspired proxy buffers must be initialized before addition")
        first_event, last_event, event_count = self._taminggs_schedule_events()
        if global_step < first_event or global_step > last_event or global_step % self.config.add.frequency != 0:
            raise ValueError(f"step {global_step} is not a scheduled TamingGS-inspired proxy addition event")

        event_index = (global_step - first_event) // self.config.add.frequency + 1
        initial_count = int(self._taminggs_initial_count.item())
        # Initialization can already exceed the requested growth budget. Keep
        # those rows and use their checkpointed count as the effective ceiling.
        growth = max(0, self._max_n_gaussians - initial_count)
        # q(t) = 1 - (1 - t)^2. Integer ceil division keeps the
        # population monotonic and makes the last scheduled event exact.
        progress_numerator = 2 * event_index * event_count - event_index * event_index
        progress_denominator = event_count * event_count
        scheduled_growth = (growth * progress_numerator + progress_denominator - 1) // progress_denominator
        return initial_count + scheduled_growth

    def _taminggs_local_growth_quota(
        self,
        *,
        global_step: int,
        local_count: int,
        local_score_statistics: Mapping[str, tuple[float, int]],
    ) -> int:
        """Allocate growth by owner mass with stable owner/shard tie-breaks.

        The first Hamilton pass assigns the exact global growth budget to
        lexical owner IDs. A second pass distributes each owner's quota over
        its DDP shards using content-derived keys, so equal fractional
        remainders do not systematically favor a numeric rank. Shard keys move
        with their content when rank assignment changes.
        """
        owner_ids = tuple(sorted(local_score_statistics))
        if not owner_ids:
            raise ValueError("TamingGS-inspired proxy allocation requires at least one eligible owner key")
        if not isinstance(local_count, int) or not 0 <= local_count < 2**52:
            raise ValueError(f"invalid TamingGS-inspired proxy local population: {local_count!r}")
        for owner_id in owner_ids:
            mass, shard_key = local_score_statistics[owner_id]
            if not math.isfinite(mass) or mass < 0.0:
                raise ValueError(f"invalid TamingGS-inspired proxy score mass for {owner_id!r}: {mass}")
            if not isinstance(shard_key, int) or not 0 <= shard_key < 2**52:
                raise ValueError(f"invalid TamingGS-inspired proxy shard key for {owner_id!r}: {shard_key!r}")
        device = self.binoms.device
        flattened_statistics: list[float] = [float(local_count)]
        for owner_id in owner_ids:
            mass, shard_key = local_score_statistics[owner_id]
            flattened_statistics.extend((float(self._taminggs_stable_key(owner_id)), mass, float(shard_key)))
        local_statistics = torch.tensor(flattened_statistics, dtype=torch.float64, device=device)
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            world_size = torch.distributed.get_world_size()
            gathered_statistics = [torch.zeros_like(local_statistics) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_statistics, local_statistics)
            rank = torch.distributed.get_rank()
        else:
            gathered_statistics = [local_statistics]
            rank = 0

        if any(value.shape != local_statistics.shape for value in gathered_statistics):
            raise RuntimeError("TamingGS-inspired proxy gathered malformed rank-local score statistics")
        raw_counts = [float(value[0].item()) for value in gathered_statistics]
        if any(
            not math.isfinite(value) or value < 0.0 or value >= 2**52 or not value.is_integer() for value in raw_counts
        ):
            raise RuntimeError("TamingGS-inspired proxy gathered an invalid rank-local population")
        counts = [int(value) for value in raw_counts]
        if rank >= len(counts) or counts[rank] != local_count:
            raise RuntimeError("TamingGS-inspired proxy gathered an inconsistent rank-local population")
        masses_by_owner: dict[str, list[float]] = {owner_id: [] for owner_id in owner_ids}
        shard_keys_by_owner: dict[str, list[int]] = {owner_id: [] for owner_id in owner_ids}
        for gathered in gathered_statistics:
            for owner_index, owner_id in enumerate(owner_ids):
                offset = 1 + 3 * owner_index
                gathered_owner_key = int(gathered[offset].item())
                if gathered_owner_key != self._taminggs_stable_key(owner_id):
                    raise RuntimeError("TamingGS-inspired proxy owner registry differs across ranks")
                mass = float(gathered[offset + 1].item())
                shard_key_value = float(gathered[offset + 2].item())
                if not math.isfinite(mass) or mass < 0.0:
                    raise RuntimeError(f"TamingGS-inspired proxy gathered invalid score mass for {owner_id!r}")
                if (
                    not math.isfinite(shard_key_value)
                    or not 0.0 <= shard_key_value < 2**52
                    or not shard_key_value.is_integer()
                ):
                    raise RuntimeError(f"TamingGS-inspired proxy gathered invalid shard key for {owner_id!r}")
                masses_by_owner[owner_id].append(mass)
                shard_keys_by_owner[owner_id].append(int(shard_key_value))
        global_current_count = sum(counts)
        global_target_count = self._taminggs_global_target_count(global_step)
        if global_current_count > global_target_count:
            log.warning(
                "TamingGS population exceeds the current schedule target; preserving rows and skipping growth: "
                "current=%d, target=%d",
                global_current_count,
                global_target_count,
            )
            return 0
        growth = global_target_count - global_current_count
        if growth == 0:
            return 0

        owner_masses = {owner_id: math.fsum(masses_by_owner[owner_id]) for owner_id in owner_ids}
        global_mass = math.fsum(owner_masses.values())
        if not math.isfinite(global_mass) or global_mass < 0.0:
            raise RuntimeError("TamingGS-inspired proxy requires finite nonnegative global score mass")
        if global_mass == 0.0:
            log.warning("TamingGS addition has no live donor mass globally; skipping growth")
            return 0
        owner_quotas = self._taminggs_hamilton_quotas(growth, owner_masses)

        quotas_by_rank = [0 for _ in gathered_statistics]
        for owner_id in owner_ids:
            positive_keys = [
                key for key, mass in zip(shard_keys_by_owner[owner_id], masses_by_owner[owner_id]) if mass > 0.0
            ]
            if len(positive_keys) != len(set(positive_keys)):
                raise RuntimeError(f"TamingGS-inspired proxy owner {owner_id!r} has duplicate positive-mass shard keys")
            owner_quota = owner_quotas[owner_id]
            if owner_quota == 0:
                continue
            shard_masses = {
                shard_keys_by_owner[owner_id][shard_rank]: masses_by_owner[owner_id][shard_rank]
                for shard_rank in range(len(gathered_statistics))
                if masses_by_owner[owner_id][shard_rank] > 0.0
            }
            shard_quotas = self._taminggs_hamilton_quotas(owner_quota, shard_masses)
            for shard_rank, shard_key in enumerate(shard_keys_by_owner[owner_id]):
                quotas_by_rank[shard_rank] += shard_quotas.get(shard_key, 0)
        if sum(quotas_by_rank) != growth:
            raise RuntimeError("TamingGS-inspired proxy growth apportionment did not preserve the global target")
        return quotas_by_rank[rank]

    @staticmethod
    def _taminggs_stable_key(value: str) -> int:
        """Map a stable text key to an exactly representable float64 integer."""
        digest = hashlib.blake2b(value.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") & ((1 << 52) - 1)

    @staticmethod
    def _taminggs_hamilton_quotas(total: int, masses: Mapping[Any, float]) -> dict[Any, int]:
        """Hamilton apportionment whose tie-break is the supplied stable key."""
        if total < 0:
            raise ValueError(f"Hamilton total must be nonnegative, got {total}")
        positive = {key: mass for key, mass in masses.items() if mass > 0.0}
        if total == 0:
            return {key: 0 for key in masses}
        denominator = math.fsum(positive.values())
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError("Hamilton apportionment requires positive finite mass")
        raw = {key: total * mass / denominator for key, mass in positive.items()}
        quotas = {key: math.floor(value) for key, value in raw.items()}
        remainder = total - sum(quotas.values())
        order = sorted(positive, key=lambda key: (-(raw[key] - quotas[key]), key))
        for key in order[:remainder]:
            quotas[key] += 1
        return {key: quotas.get(key, 0) for key in masses}

    @classmethod
    def _taminggs_shard_key(
        cls,
        owner_id: str,
        gaussian_model: BaseGaussianModel,
        source_weights: torch.Tensor,
    ) -> int:
        """Build a stable content key without copying a full shard to the host."""
        model_row_count = gaussian_model.get_num_gaussians()
        if source_weights.shape != (model_row_count,):
            raise RuntimeError(
                f"TamingGS-inspired proxy owner {owner_id!r} source weights have shape "
                f"{tuple(source_weights.shape)} for {model_row_count} Gaussian rows"
            )
        row_count = model_row_count
        if row_count == 0:
            return cls._taminggs_stable_key(f"{owner_id}:empty")
        sample_indices = sorted({0, row_count // 2, row_count - 1})
        positions = gaussian_model.positions.detach()[sample_indices].reshape(-1).to(dtype=torch.float64, device="cpu")
        weights = source_weights.detach()[sample_indices].reshape(-1).to(dtype=torch.float64, device="cpu")
        digest = hashlib.blake2b(digest_size=8)
        digest.update(owner_id.encode())
        digest.update(struct.pack("!Q", row_count))
        for value in (*positions.tolist(), *weights.tolist()):
            digest.update(struct.pack("!d", value))
        return int.from_bytes(digest.digest(), "big") & ((1 << 52) - 1)

    def _taminggs_local_score_statistics(
        self,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
        source_weights: Mapping[str, torch.Tensor],
        owner_masses: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, tuple[float, int]]:
        """Return mass and content key for every globally eligible owner."""
        excluded = self._addition_excluded_layer_ids()
        statistics: dict[str, tuple[float, int]] = {}
        for owner_id in sorted(set(gaussians_nodes) - excluded):
            weights = source_weights.get(owner_id)
            if weights is None:
                statistics[owner_id] = (0.0, self._taminggs_stable_key(f"{owner_id}:empty"))
                continue
            statistics[owner_id] = (
                float((owner_masses[owner_id] if owner_masses is not None else weights.detach().sum()).item()),
                self._taminggs_shard_key(owner_id, gaussians_nodes[owner_id], weights),
            )
        return statistics

    def _missing_gradient(self, layer_id: str, component: str, reason: str, positions: torch.Tensor) -> torch.Tensor:
        key = (layer_id, component, reason)
        if key not in self._taminggs_missing_input_warnings:
            log.warning("TamingGS layer %s: enabled %s is unavailable (%s); using zero", layer_id, component, reason)
            self._taminggs_missing_input_warnings.add(key)
        return torch.zeros(positions.shape[0], dtype=torch.float32, device=positions.device)

    def _position_gradient(self, layer_id: str, gaussian_model: BaseGaussianModel) -> torch.Tensor:
        positions = gaussian_model.positions
        gradient = positions.grad
        if gradient is None:
            return self._missing_gradient(layer_id, "current_gradient", "missing gradient", positions)
        if gradient.shape != positions.shape:
            return self._missing_gradient(layer_id, "current_gradient", "shape mismatch", positions)
        return torch.linalg.vector_norm(gradient.detach().to(torch.float32), dim=-1)

    def _position_gradient_ema(self, gaussian_model: BaseGaussianModel, layer_id: str = "unknown") -> torch.Tensor:
        """Read only the position parameter's own full-shaped Adam second moment."""
        positions = gaussian_model.positions
        for optimizer_config in getattr(gaussian_model, "optimizers", None) or []:
            optimizer = optimizer_config["optimizer"]
            for param_group in optimizer.param_groups:
                for parameter in param_group["params"]:
                    if parameter is not positions:
                        continue
                    second_moment = optimizer.state.get(parameter, {}).get("exp_avg_sq")
                    if not isinstance(second_moment, torch.Tensor):
                        return self._missing_gradient(layer_id, "gradient_ema", "missing second moment", positions)
                    if second_moment.shape != positions.shape:
                        return self._missing_gradient(layer_id, "gradient_ema", "shape mismatch", positions)
                    return second_moment.detach().to(torch.float32).mean(dim=-1).clamp_min(0.0).sqrt()
        return self._missing_gradient(layer_id, "gradient_ema", "position parameter not owned", positions)

    def _taminggs_component_medians(
        self,
        components: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather raw components and find global finite-positive medians over all DDP-owned positive rows.

        One padded matrix gather covers every component so normalization is
        independent of rank partitioning without adding one collective pair
        per score term.
        """
        names = tuple(components)
        row_counts = {value.numel() for value in components.values()}
        devices = {value.device for value in components.values()}
        if len(row_counts) != 1 or len(devices) != 1:
            raise RuntimeError("TamingGS-inspired proxy score components must have one row layout and device")
        local_matrix = torch.stack([components[name].reshape(-1).to(torch.float32) for name in names], dim=1)
        global_matrix = local_matrix
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            world_size = torch.distributed.get_world_size()
            local_count = torch.tensor(local_matrix.shape[0], dtype=torch.int64, device=local_matrix.device)
            gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_counts, local_count)
            counts = [int(value.item()) for value in gathered_counts]
            max_count = max(counts)
            if max_count == 0:
                global_matrix = local_matrix
            else:
                padded = local_matrix.new_zeros((max_count, len(names)))
                padded[: local_matrix.shape[0]] = local_matrix
                gathered_matrices = [torch.zeros_like(padded) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_matrices, padded)
                global_matrix = torch.cat(
                    [matrix[:count] for matrix, count in zip(gathered_matrices, counts)],
                    dim=0,
                )

        denominators = []
        for column in range(len(names)):
            component = global_matrix[:, column]
            positive = component[torch.isfinite(component) & (component > 0.0)]
            denominators.append(torch.median(positive) if positive.numel() else component.new_ones(()))
        return local_matrix, torch.stack(denominators)

    def _globally_normalized_taminggs_components(
        self, components: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if not components:
            return {}
        matrix, medians = self._taminggs_component_medians(components)
        normalized = _normalize_taminggs_components(matrix, medians, self.config.add.taminggs.normalization_clip)
        return {name: normalized[:, column] for column, name in enumerate(components)}

    def _normalized_taminggs_component(self, component: torch.Tensor) -> torch.Tensor:
        """Globally median-normalize one positive-only proxy component."""
        return self._globally_normalized_taminggs_components({"component": component})["component"]

    def _taminggs_score_components(
        self, selected: SelectedGaussianNodes
    ) -> tuple[dict[str, torch.Tensor], tuple[float, ...]]:
        config = self.config.add.taminggs
        weights = {
            name: getattr(config, name + "_weight")
            for name in ("current_gradient", "gradient_ema", "opacity", "scale", "visibility")
        }
        components: dict[str, list[torch.Tensor]] = {name: [] for name, weight in weights.items() if weight > 0.0}
        for layer_id in selected.layer_ids:
            model = selected.gaussians_nodes[layer_id]
            row_count = selected.layer_num_gaussians[layer_id]
            for name, values in components.items():
                if name == "current_gradient":
                    value = self._position_gradient(layer_id, model)
                elif name == "gradient_ema":
                    value = self._position_gradient_ema(model, layer_id)
                elif name == "opacity":
                    value = model.get_densities().detach().reshape(-1).to(torch.float32)
                elif name == "scale":
                    value = model.get_scales().detach().to(torch.float32).clamp_min(0.0).prod(dim=-1)
                else:
                    value = torch.zeros(row_count, dtype=torch.float32, device=model.positions.device)
                    if layer_id in self.invisible_steps:
                        counter = self.invisible_steps[layer_id]
                        if not isinstance(counter, nn.UninitializedParameter) and counter.shape == (row_count,):
                            value = torch.reciprocal(1.0 + counter.detach().to(torch.float32).clamp_min(0.0))
                values.append(value)
        device = next(
            (selected.gaussians_nodes[layer_id].positions.device for layer_id in selected.layer_ids), self.binoms.device
        )
        return (
            {
                name: torch.cat(values) if values else torch.empty(0, dtype=torch.float32, device=device)
                for name, values in components.items()
            },
            tuple(weights[name] for name in components),
        )

    def _taminggs_source_weights(self, selected: SelectedGaussianNodes) -> dict[str, torch.Tensor]:
        """Build globally normalized owner-local sampling weights."""
        components, coefficients = self._taminggs_score_components(selected)
        matrix, medians = self._taminggs_component_medians(components)
        config = self.config.add.taminggs
        score = _finish_taminggs_scores(
            matrix,
            medians,
            coefficients,
            config.normalization_clip,
            config.minimum_source_weight,
            enable_torch_compile=matrix.is_cuda,
        )
        return {
            layer_id: score[
                selected.layer_offsets[layer_id] : selected.layer_offsets[layer_id]
                + selected.layer_num_gaussians[layer_id]
            ]
            for layer_id in selected.layer_ids
        }

    def _taminggs_addition_weights(
        self, selected: SelectedGaussianNodes, carried: Mapping[str, torch.Tensor] | None
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        config = self.config.add.taminggs
        sizes = tuple(selected.layer_num_gaussians[layer_id] for layer_id in selected.layer_ids)
        if carried is None:
            components, coefficients = self._taminggs_score_components(selected)
            matrix, medians = self._taminggs_component_medians(components)
            # Reuse the scoring opacity when enabled; eligibility never changes the global median population.
            opacity = components.get("opacity")
            if opacity is None:
                opacity = selected.get_densities().detach().reshape(-1).to(device=matrix.device, dtype=torch.float32)
            score, masses = _finish_taminggs_addition_scores(
                matrix,
                medians,
                coefficients,
                config.normalization_clip,
                config.minimum_source_weight,
                opacity,
                self.config.opacity_threshold,
                sizes,
                enable_torch_compile=matrix.is_cuda,
            )
        else:
            # Relocation can change opacity even when the carried score retains its row correspondence.
            opacity = selected.get_densities().detach().reshape(-1).to(torch.float32)
            weights = (
                torch.cat([carried[layer_id] for layer_id in selected.layer_ids]) if sizes else opacity.new_empty(0)
            )
            score, masses = _mask_taminggs_addition_weights(
                weights,
                opacity,
                self.config.opacity_threshold,
                sizes,
                enable_torch_compile=opacity.is_cuda,
            )
        return (
            {
                layer_id: score[
                    selected.layer_offsets[layer_id] : selected.layer_offsets[layer_id]
                    + selected.layer_num_gaussians[layer_id]
                ]
                for layer_id in selected.layer_ids
            },
            dict(zip(selected.layer_ids, masses)),
        )

    def _validate_variant_config(self) -> None:
        unfixed_track_local_layers = (
            set(self.config.relocate.same_track_layer_ids)
            - set(self.config.exclude_layer_ids)
            - set(self.config.add.fixed_count_layer_ids)
            - set(self.config.independent_layer_max_n_gaussians)
        )
        if unfixed_track_local_layers:
            raise ValueError(
                "TamingGS-inspired proxy co-scheduled saliency requires same-track relocation layers "
                "to have fixed counts: "
                f"{sorted(unfixed_track_local_layers)}"
            )

    def _initialize_variant_state(self, trainer_config: TrainerConfig) -> None:
        self._taminggs_training_world_size = trainer_config.world_size
        self._taminggs_missing_input_warnings: set[tuple[str, str, str]] = set()
        self._taminggs_no_growth_warning_emitted = False
        # Checkpoint the global schedule anchor and each rank's original
        # ownership. Per-row saliency is derived on demand from model and
        # optimizer state rather than retained in another row-sized buffer.
        self.register_buffer("_taminggs_initial_count", torch.tensor(-1, dtype=torch.int64), persistent=True)
        self.register_buffer(
            "_taminggs_initial_counts_by_rank",
            torch.full((trainer_config.world_size,), -1, dtype=torch.int64),
            persistent=True,
        )

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        names = (
            "_taminggs_initial_count",
            "_taminggs_initial_counts_by_rank",
        )
        missing_taminggs_state = [prefix + name for name in names if prefix + name not in state_dict]
        if missing_taminggs_state:
            raise RuntimeError(
                "Cannot enable the TamingGS-inspired MCMC proxy while loading a checkpoint that lacks its "
                "population-allocation state. Start a fresh proxy run; missing keys: "
                f"{missing_taminggs_state}"
            )
        loaded_rank_counts = state_dict[prefix + "_taminggs_initial_counts_by_rank"]
        if not isinstance(loaded_rank_counts, torch.Tensor):
            raise RuntimeError("TamingGS-inspired proxy rank-local population state must be a tensor")
        if loaded_rank_counts.ndim != 1 or loaded_rank_counts.numel() == 0:
            raise RuntimeError("TamingGS-inspired proxy rank-local population state must be a nonempty vector")
        # Validation/export can use a different GPU count. Preserve the saved
        # allocation anchors; enforce training compatibility at the first step.
        self._taminggs_initial_counts_by_rank = self._taminggs_initial_counts_by_rank.new_empty(
            loaded_rank_counts.shape
        )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def on_train_step_start(self) -> None:
        if self._taminggs_initial_counts_by_rank.numel() != self._taminggs_training_world_size:
            raise RuntimeError(
                "Cannot resume the TamingGS-inspired proxy with a different world size: checkpoint has "
                f"{self._taminggs_initial_counts_by_rank.numel()} ranks, current run has "
                f"{self._taminggs_training_world_size}"
            )
        super().on_train_step_start()

    def _initialize_population(
        self, gaussians_nodes: TypedModuleDict[BaseGaussianModel], collective_device: torch.device
    ) -> None:
        if int(self._taminggs_initial_count.item()) < 0 and self._taminggs_eligible_population_is_materialized(
            gaussians_nodes, collective_device
        ):
            selected = self._addition_selected_nodes(gaussians_nodes)
            local_initial_count = sum(selected.layer_num_gaussians.values())
            initial_count = local_initial_count
            initial_counts_by_rank = torch.tensor(
                [local_initial_count],
                dtype=torch.int64,
                device=collective_device,
            )
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                local_count = torch.tensor(local_initial_count, dtype=torch.int64, device=collective_device)
                gathered_counts = [torch.zeros_like(local_count) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gathered_counts, local_count)
                initial_counts_by_rank = torch.stack(gathered_counts)
                initial_count = int(initial_counts_by_rank.sum().item())
            if initial_counts_by_rank.shape != self._taminggs_initial_counts_by_rank.shape:
                raise RuntimeError(
                    "TamingGS-inspired proxy allocation world size changed after strategy construction: "
                    f"expected {self._taminggs_initial_counts_by_rank.numel()}, got {initial_counts_by_rank.numel()}"
                )
            if initial_count > self._max_n_gaussians:
                log.warning(
                    "TamingGS initial eligible population exceeds the configured growth budget; "
                    "preserving initialization with no scheduled population expansion: "
                    "initial=%d, configured_budget=%d, effective_limit=%d",
                    initial_count,
                    self._max_n_gaussians,
                    initial_count,
                )
            if initial_count == 0 and self._max_n_gaussians > 0:
                raise ValueError("TamingGS-inspired proxy addition requires at least one eligible Gaussian donor")
            if initial_count < self._max_n_gaussians:
                self._taminggs_schedule_events()
            self._taminggs_initial_count.fill_(initial_count)
            self._taminggs_initial_counts_by_rank.copy_(initial_counts_by_rank)

    @torch.no_grad()
    def update_prepared_visibility(
        self,
        *,
        visibility: torch.Tensor,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
    ) -> None:
        """Update counters once the rank-symmetric visibility union resolves."""
        if not self._should_track_visibility():
            return
        self._apply_visibility_counters(visibility.detach().reshape(-1).float(), gaussians_nodes)

    def _should_track_visibility(self) -> bool:
        return self._relocation_uses_visibility() or self._strict_visibility()

    def _strict_visibility(self) -> bool:
        return self.config.add.taminggs.visibility_weight > 0.0

    def _visibility_counter_cap(self) -> int:
        if self._relocation_uses_visibility():
            return super()._visibility_counter_cap()
        if self._strict_visibility():
            return _TAMINGGS_VISIBILITY_COUNTER_CAP
        return super()._visibility_counter_cap()

    def _co_scheduled_source_weights(
        self, due_operations: tuple[_TopologyMutation, ...], gaussians_nodes: TypedModuleDict[BaseGaussianModel]
    ) -> dict[str, torch.Tensor] | None:
        if any(config is self.config.relocate for config, _ in due_operations) and any(
            config is self.config.add for config, _ in due_operations
        ):
            return self._taminggs_source_weights(self._addition_selected_nodes(gaussians_nodes))
        return None

    def _validate_reindex_values(self, values: Mapping[str, torch.Tensor] | None) -> None:
        pass

    def _relocate_shared_pool(
        self, selected: SelectedGaussianNodes, reindexed_values: dict[str, torch.Tensor] | None
    ) -> int:
        weights = self._taminggs_source_weights(selected)
        if reindexed_values is not None and set(reindexed_values) - set(selected.layer_ids):
            raise ValueError("reindex values must belong to the shared population")
        return self._relocate_pool(selected, taminggs_source_weights=weights, reindex_values_by_owner=reindexed_values)

    def _validate_addition_weights(self, values: Mapping[str, torch.Tensor] | None) -> None:
        pass

    def _add_shared_pool(
        self,
        selected: SelectedGaussianNodes,
        cap: int,
        global_step: int | None,
        weights: Mapping[str, torch.Tensor] | None,
    ) -> None:
        self._add_to_pool(selected, cap, use_taminggs=True, global_step=global_step, source_weights_by_owner=weights)

    def _addition_target(
        self,
        selected_gaussian_nodes: SelectedGaussianNodes,
        cap: int,
        current_num_gaussians: int,
        use_taminggs: bool,
        global_step: int | None,
        source_weights_by_owner: Mapping[str, torch.Tensor] | None,
    ) -> tuple[int, dict[str, Any]]:
        if not use_taminggs:
            return super()._addition_target(
                selected_gaussian_nodes, cap, current_num_gaussians, False, global_step, source_weights_by_owner
            )
        gaussians_nodes = selected_gaussian_nodes.gaussians_nodes
        taminggs_kwargs: dict[str, Any] = {}
        if global_step is None:
            raise ValueError("TamingGS-inspired proxy addition requires the current global_step")
        source_weights, owner_masses = self._taminggs_addition_weights(selected_gaussian_nodes, source_weights_by_owner)
        local_growth = self._taminggs_local_growth_quota(
            global_step=global_step,
            local_count=current_num_gaussians,
            local_score_statistics=self._taminggs_local_score_statistics(gaussians_nodes, source_weights, owner_masses),
        )
        target_num_gaussians = current_num_gaussians + local_growth
        taminggs_kwargs["source_weights_by_owner"] = source_weights
        return target_num_gaussians, taminggs_kwargs


BaseGaussianStrategy.register_to_gaussian_strategy_factory("mcmc", MCMCStrategy)
BaseGaussianStrategy.register_to_gaussian_strategy_factory("taminggs_mcmc", TamingGSMCMCStrategy)
