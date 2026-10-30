# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Type, Union

import torch

from torch import nn

from nurec.config.model import BaseStrategyConfig
from nurec.models.base import BaseModel
from nurec.models.gaussians.gaussians_model import BaseGaussianModel
from nurec.models.nn_extensions import TypedModuleDict
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.trainer import TrainerConfig


if TYPE_CHECKING:
    from nurec.models.gaussians.gaussians_model import BaseGaussianModel


class BaseGaussianStrategy(nn.Module, ABC):
    """
    Base class for Gaussian Strategies. Strategies are responsible for the densification and pruning of the Gaussians.

    Currently there are two main methods that are called to interact with a Strategy:
        - `update_step_train_batch_end`: called at the end of the train batch, useful for periodic updates to the gaussians (densify, prune, density resets etc.)

    """

    GAUSSIAN_STRATEGY_VARIANTS: dict[str, Type[BaseGaussianStrategy]] = {}

    def __init__(
        self,
        config: BaseStrategyConfig,
        trainer_config: TrainerConfig,
        init_from_datasource: bool,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
    ) -> None:
        super().__init__()
        self.config = config
        self.init_from_datasource = init_from_datasource
        self.last_reset_epoch: Optional[int] = None
        self.frozen: bool = False
        self._empty_cache_after_parameter_update = config.enforce_empty_cache_after_parameter_update

    @torch.no_grad()
    def maybe_initialize_buffers(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> None: ...

    def configure_sharded_params_and_buffers(self) -> list[str]:
        """Return strategy-owned state that is sharded by Gaussian row."""
        return []

    def refresh_runtime_scenes(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> None:
        """Refresh strategy-specific runtime scene bindings after row replacement."""

    @torch.no_grad()
    def serialization_state_dict_overrides(
        self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]
    ) -> dict[str, torch.Tensor]:
        """Return bounded Gaussian state tensors without mutating the live model."""
        return {}

    @abstractmethod
    def _update_step_train_batch_end_impl(
        self,
        epoch: int,
        global_step: int,
        batch: DataAndRenderingBatch,
        system,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
        **kwargs,
    ) -> None: ...

    def update_step_train_batch_end(
        self,
        epoch: int,
        global_step: int,
        batch: DataAndRenderingBatch,
        system,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
        **kwargs,
    ) -> None:
        if self.frozen:
            return
        self._update_step_train_batch_end_impl(epoch, global_step, batch, system, gaussians_nodes, **kwargs)

    def may_mutate_topology_at(self, global_step: int) -> bool:
        """Return whether batch-end work may change per-layer row counts or rebuild row layout.

        Unknown strategies conservatively invalidate cached distributed-gather
        topology. Strategies that opt into generation-plan reuse must override
        this next to the mutation schedule that it describes.
        """
        del global_step
        return not self.frozen

    @staticmethod
    def register_to_gaussian_strategy_factory(name: str, cls: Type[BaseGaussianStrategy]) -> None:
        if name in BaseGaussianStrategy.GAUSSIAN_STRATEGY_VARIANTS:
            raise KeyError(f"{name=} already in GAUSSIAN_STRATEGY_VARIANTS.")
        BaseGaussianStrategy.GAUSSIAN_STRATEGY_VARIANTS[name] = cls

    @staticmethod
    def factory(
        name: str,
        config: BaseStrategyConfig,
        trainer_config: TrainerConfig,
        init_from_datasource: bool,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
    ) -> BaseGaussianStrategy:
        return BaseGaussianStrategy.GAUSSIAN_STRATEGY_VARIANTS[name](
            config, trainer_config, init_from_datasource, gaussians_nodes
        )

    def permute_gaussian_rows(self, orders: dict[str, torch.Tensor]) -> None:
        """Apply per-layer row permutations to the strategy's own row-aligned buffers.

        Strategies keep per-Gaussian accumulators outside the nodes, so a reorder inside a node cannot reach
        them. Leaving one behind leaves an accumulator describing a Gaussian that no longer lives at that row.

        Args:
            orders: Permutation per layer id. Layers absent from the dict were not reordered.
        """

    @staticmethod
    def _permute_layer_buffers(buffers: nn.ParameterDict, orders: dict[str, torch.Tensor]) -> None:
        """Replace each row-aligned entry of a per-layer buffer dict with its permuted copy."""
        for layer_id, order in orders.items():
            if layer_id not in buffers:
                continue
            buffer = buffers[layer_id]
            # Lazy entries hold no rows yet, and a mismatched leading dim is not row-aligned with this layer.
            if isinstance(buffer, nn.UninitializedParameter) or buffer.shape[:1] != order.shape:
                continue
            reordered = nn.Parameter(buffer[order], requires_grad=buffer.requires_grad)
            # These are per-rank shards under DDP even though nothing marks them today, and every sibling
            # helper carries the tag through.
            if BaseModel.is_sharded(buffer):
                BaseModel.mark_as_sharded(reordered)
            buffers[layer_id] = reordered

    @torch.no_grad()
    def _update_param_with_optimizer(
        self,
        gaussian_model: BaseGaussianModel,
        update_param_fn: Callable[[str, torch.Tensor], torch.Tensor],
        update_optimizer_fn: Callable[[str, str, torch.Tensor], torch.Tensor],
        names: Union[list[str], None] = None,
    ) -> None:
        """Update the parameters and the state in the optimizers using the provided lambda functions.

        Args:
            update_param_fn: A function that takes the name of the parameter and the parameter itself,
                and returns the new parameter.
            optimizer_fn: A function that takes the key of the optimizer state and the state value,
                and returns the new state value.
            names: A list of key names to update. If None, update all. Default: None.
        """
        for optim_sched in gaussian_model.optimizers:
            optimizer = optim_sched["optimizer"]
            for i, param_group in enumerate(optimizer.param_groups):
                if "name" in param_group:
                    name = param_group["name"].split(".")[-1]
                    if (names is None) or (name in names):
                        p = param_group["params"][0]
                        p_state = optimizer.state[p]
                        del optimizer.state[p]
                        for key in list(p_state.keys()):
                            v = p_state[key]
                            # Only tensor state with a Gaussian row dimension can be rewritten.
                            # Scalar/non-tensor entries are optimizer-level bookkeeping (for
                            # example Adam's step counter) and must remain unchanged.
                            if not isinstance(v, torch.Tensor) or v.dim() == 0:
                                continue
                            p_state[key] = update_optimizer_fn(name, key, v)
                        p_new = update_param_fn(name, p)
                        # We need to preserve the "sharded" flag
                        if BaseModel.is_sharded(p):
                            BaseModel.mark_as_sharded(p_new)
                        # Set the new parameter in the optimizer
                        optimizer.param_groups[i]["params"] = [p_new]
                        optimizer.state[p_new] = p_state
                        setattr(gaussian_model, name, p_new)

        if self._empty_cache_after_parameter_update:
            torch.cuda.empty_cache()
        # Also update additional non-optimizable buffers
        for name, b in gaussian_model.get_additional_buffers().items():
            if (names is None) or (name in names):
                b_new = update_param_fn(name, b)
                # We need to preserve the "sharded" flag
                if BaseModel.is_sharded(b):
                    BaseModel.mark_as_sharded(b_new)
                # Set the new parameter in the model
                setattr(gaussian_model, name, b_new)

    def _check_step_condition(self, step: int, start: int, end: int, freq: int) -> bool:
        """Checks if an operation should occur for the given step."""
        if step > start and step < end and step % freq == 0:
            return True
        return False
