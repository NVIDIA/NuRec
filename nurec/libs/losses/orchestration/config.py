# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loss configuration and return types.

Uses nurec.config.base_schema.BaseConfigSchema as base (no other nurec imports there, so no circular deps).
Config classes first; loss return/reduction types at the end (so config is a single file for stable import order).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Protocol

import torch

from pydantic import Field, RootModel

from nurec.config.base_schema import BaseConfigSchema
from nurec.libs.loss_aggregation_ops import finalize_accumulated_losses
from nurec.utils.profiling import ScopedTimer


class ReduceConfig(BaseConfigSchema):
    """Configuration for loss reduction functions."""

    name: str = Field(description="Reduction function type")
    quantile: float | None = None


class LambdaSchedulerConfig(BaseConfigSchema):
    """Configuration for lambda scheduler."""

    name: str = Field(description="Lambda scheduler name")
    start: int
    end: int
    update_interval: str
    update_frequency: int
    lambda_end: float


class LossItemConfig(BaseConfigSchema):
    """Configuration for a single loss item."""

    class MaskConfig(BaseConfigSchema):
        mode: str
        value: float

    fn: str = Field(description="Loss function name (l1, mse, ssim, cross_entropy, etc.)")
    lambda_: float
    reduce: ReduceConfig = Field(description="Configuration for reduction function")

    # Optional common fields
    start_step: int | None = None
    lambda_scheduler: LambdaSchedulerConfig | None = None
    visibility_filter: bool = False
    occlusion_aware: bool = False

    mask_semantic_classes: list[str] | None = None
    semantic_lambdas: list[float] | None = None

    confidence_alpha_: float | None = None

    per_image_loss: bool | None = Field(default=None, description="Whether loss is computed per-image or per-ray")

    # SSIM
    window_size: int | None = None
    channel: int | None = None
    mask: MaskConfig | None = None

    # Lambdas for individual loss components (e.g., in PPISP, UnicyclePoseLoss, ...)
    lambdas: dict[str, float] | None = None

    # Road Gaussians
    layer_name: str | None = None
    n_samples: int | None = None
    grid_len: float | None = None
    min: float | None = None
    range: float | None = None
    rotation_lambda: float | None = None

    # BackgroundInTrackGaussian
    layer_names: list[str] | None = None
    density_logits_min: float = -20.0

    # Gaussian_scale, gaussian_density
    layer_lambdas: dict[str, float] | None = None

    # Gaussian Z Scale
    road_z_scale: float | None = None

    # Node Semantic Gaussians
    layer_labels_to_use: dict[str, list[str]] | None = None
    layer_labels_to_exclude: dict[str, list[str]] | None = None

    # Gaussian Flatten
    max_to_median_ratio_threshold: float | None = None
    axes_type: str | None = None

    # Distance and Normal losses
    allow_missing_supervision: bool | None = None
    min_distance: float | None = None
    max_distance: float | None = None
    normalize_by_opacity: bool | None = None

    # bce_clipped, bce_truncated
    clip_opacity: float | None = None

    # LPIPS
    limit_max_resolution: int | None = None

    # Primitive sky distance
    sky_distance: float | None = None

    # Primitive geometry-related (e.g. distance, depth, pointmap) loss
    scale_steps: list[int] | None = None
    scale_lambdas: list[float] | None = None
    use_z_depth: bool = Field(default=False, description="Whether to use z-depth for the geometry loss")

    # Primitive mask loss
    semantic_class: str | None = None

    # Primitive sky cubemap loss
    masked_region: Literal["skip", "average", "smooth"] = Field(
        default="average", description="How to handle masked regions"
    )

    # Primitive velocity loss
    velocity_mask_threshold: float | None = Field(
        default=None,
        description="Threshold for the velocity mask (loss only applied to region whose gt value > threshold). If None, no mask is applied.",
    )


class LossConfig(RootModel[dict[str, LossItemConfig]]):
    """
    Top-level loss configuration.

    Maps loss names to their configurations.
    """

    def __getitem__(self, key: str) -> LossItemConfig:
        """Allow dict-like access: loss_config['rgb']"""
        return self.root[key]

    def items(self):
        """Allow dict.items() usage."""
        return self.root.items()

    def keys(self):
        """Allow dict.keys() usage."""
        return self.root.keys()

    def values(self):
        """Allow dict.values() usage."""
        return self.root.values()

    def get(self, key: str, default=None):
        """Allow dict.get() usage."""
        return self.root.get(key, default)


# -----------------------------------------------------------------------------
# Loss return and reduction types (single file with config for stable order)
# -----------------------------------------------------------------------------


class LossResult(Protocol):
    """The existing surface shared by both training return paths."""

    @property
    def names(self) -> tuple[str, ...]:
        """Names reported this step, including configured zero-valued losses."""
        ...

    def weighted_value_dtype(self, name: str) -> torch.dtype:
        """Report a named weighted value's dtype without materializing the objective."""
        ...

    def compute_total(self) -> torch.Tensor:
        """Return the weighted objective, finalizing lazy results when needed."""
        ...

    def append_loss(self, *, name: str, value: torch.Tensor, weight: float = 1.0) -> None:
        """Append an unweighted scalar once; reject duplicate names and non-scalars."""
        ...

    def rescale_reduced_values(self, scales_by_name: Mapping[str, torch.Tensor]) -> None:
        """Rescale named weighted losses while preserving their autograd graphs."""
        ...

    def backward_roots(self) -> dict[str, list[torch.Tensor]]:
        """Return named autograd roots for opt-in backward profiling."""
        ...

    def log(self, logging_fn: Callable[..., None]) -> None: ...


class RayReduceFn(Protocol):
    """Interface of a reduction function that can reduce to a single value."""

    def __call__(self, value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Reduces inputs to single value."""
        ...


@dataclass(slots=True, kw_only=True)
class LossReturn:
    """
    Unified loss return type.

    - name: name of the loss (e.g. 'rgb_huber_mean')
    - lambda_: coefficient to weight the reduced loss value with
    - value: numerical value of the loss, not reduced; for per-ray losses
      value is only defined for valid rays (e.g. shape (n_valid_cam_rays,))
    - reduce_fn: function to reduce loss value with
    - reduce_mask: optional masked reduction
    """

    name: str
    lambda_: float
    value: torch.Tensor
    reduce_fn: RayReduceFn
    reduce_mask: Optional[torch.Tensor] = None
    _reduced_value: torch.Tensor | None = None

    @property
    def reduced_value(self) -> torch.Tensor:
        assert self._reduced_value is not None
        return self._reduced_value

    def __post_init__(self) -> None:
        assert callable(self.reduce_fn)
        if self._reduced_value is None:
            self._reduced_value = self.reduce()

    def reduce(self) -> torch.Tensor:
        """Performs loss value reduction and lambda-weighting."""
        reduced = self.reduce_fn(self.value, reduce_mask=self.reduce_mask)
        # Scaling a floating-point value by 1.0 is exact, so skip the kernel.
        if self.lambda_ == 1.0 and reduced.is_floating_point():
            return reduced
        return self.lambda_ * reduced

    def rescale_reduced_value(self, scale: torch.Tensor) -> None:
        """Scale the cached reduced value while preserving its autograd graph."""
        self._reduced_value = self.reduced_value * scale


def _check_scalar_loss(name: str, value: torch.Tensor) -> None:
    if value.ndim != 0:
        raise ValueError(f"{name!r} must supply an unweighted scalar")


def _log_weighted_loss(
    logging_fn: Callable[..., None],
    name: str,
    value: torch.Tensor,
    weight: float,
    *,
    batch_size: int | None = None,
    sync_dist: bool = False,
) -> None:
    loss_key = f"loss/{name}"
    weight_key = f"train_params/{name}_lambda"
    if batch_size is None:
        logging_fn(loss_key, value)
        logging_fn(weight_key, weight)
    else:
        logging_fn(loss_key, value, batch_size=batch_size, sync_dist=sync_dist)
        logging_fn(weight_key, weight, batch_size=batch_size)


def _scalar_loss_return(name: str, value: torch.Tensor, weight: float) -> LossReturn:
    _check_scalar_loss(name, value)
    return LossReturn(name=name, value=value, lambda_=weight, reduce_fn=lambda value, **_: value)


@dataclass(slots=True)
class LossAggregatorReturn:
    """Contains all the individual loss returns."""

    loss_returns: dict[str, LossReturn]
    total_value: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.total_value = self.reduce()

    def compute_total(self) -> torch.Tensor:
        """Return the cached eager objective without repeating its reduction."""
        return self.total_value

    def reduce(self) -> torch.Tensor:
        """Performs loss value reduction and lambda-weighting for all loss returns."""
        return torch.sum(torch.stack([lr.reduced_value for lr in self.loss_returns.values()]))

    @property
    def names(self) -> tuple[str, ...]:
        """Names of this step's reported losses."""
        return tuple(self.loss_returns)

    def weighted_value_dtype(self, name: str) -> torch.dtype:
        return self.loss_returns[name].reduced_value.dtype

    def append_loss(self, *, name: str, value: torch.Tensor, weight: float = 1.0) -> None:
        """Append an unweighted scalar and refresh the eagerly reduced objective."""
        if name in self.loss_returns:
            raise ValueError(f"already contains loss {name!r}")
        self.loss_returns[name] = _scalar_loss_return(name, value, weight)
        self.total_value = self.reduce()

    def backward_roots(self) -> dict[str, list[torch.Tensor]]:
        """Expose each legacy loss's weighted autograd root for profiling."""
        return {name: [loss.reduced_value] for name, loss in self.loss_returns.items()}

    def rescale_reduced_values(self, scales_by_name: Mapping[str, torch.Tensor]) -> None:
        """Scale selected reduced losses and refresh the aggregate total."""
        for name, scale in scales_by_name.items():
            self.loss_returns[name].rescale_reduced_value(scale)
        if scales_by_name:
            self.total_value = self.reduce()

    @torch.no_grad()
    def log(self, logging_fn: Callable[..., None]) -> None:
        """Logging of all loss_return values and lambda coeffs."""
        for name, value in self.loss_returns.items():
            _log_weighted_loss(logging_fn, name, value.reduced_value, value.lambda_)


@dataclass(slots=True)
class LossAggregatorBatchReturn:
    """
    Batch version of the LossAggregatorReturn.

    - batch_loss_returns: list of computed losses, one per batch element
    - total_value: total loss for the batch (precomputed)
    - extra_fields: optional batch step data (e.g. for visualizer)
    - appended scalars: shared contributions added once outside the sample mean
    """

    batch_loss_returns: list[LossAggregatorReturn]
    total_value: torch.Tensor = field(init=False)
    extra_fields: dict[str, Any] = field(default_factory=dict)
    _additional_loss_returns: dict[str, LossReturn] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.total_value = self.reduce()

    def compute_total(self) -> torch.Tensor:
        """Return the cached eager objective without repeating its reduction."""
        return self.total_value

    def reduce(self) -> torch.Tensor:
        """Mean over batch to be consistent with multi-node DDP."""
        total = torch.mean(torch.stack([lr.total_value for lr in self.batch_loss_returns]))
        if self._additional_loss_returns:
            total = total + torch.stack([loss.reduced_value for loss in self._additional_loss_returns.values()]).sum()
        return total

    @property
    def names(self) -> tuple[str, ...]:
        """Unique sample loss names followed by shared scalar contributions."""
        return tuple(dict.fromkeys(name for item in self.batch_loss_returns for name in item.names)) + tuple(
            self._additional_loss_returns
        )

    def weighted_value_dtype(self, name: str) -> torch.dtype:
        if name in self._additional_loss_returns:
            return self._additional_loss_returns[name].reduced_value.dtype
        dtypes = [item.weighted_value_dtype(name) for item in self.batch_loss_returns if name in item.loss_returns]
        if not dtypes:
            raise KeyError(name)
        dtype = dtypes[0]
        for other in dtypes[1:]:
            dtype = torch.promote_types(dtype, other)
        return dtype

    def append_loss(self, *, name: str, value: torch.Tensor, weight: float = 1.0) -> None:
        """Append a shared scalar once, outside the mean over sample objectives."""
        if name in self._additional_loss_returns or any(name in item.loss_returns for item in self.batch_loss_returns):
            raise ValueError(f"already contains loss {name!r}")
        self._additional_loss_returns[name] = _scalar_loss_return(name, value, weight)
        self.total_value = self.reduce()

    def rescale_reduced_values(self, scales_by_name: Mapping[str, torch.Tensor]) -> None:
        """Rescale named sample or shared losses and refresh the batch objective."""
        if not scales_by_name:
            return
        for name in scales_by_name:
            if name not in self.names:
                raise KeyError(name)
        for item in self.batch_loss_returns:
            item.rescale_reduced_values(
                {name: scale for name, scale in scales_by_name.items() if name in item.loss_returns}
            )
        for name, loss in self._additional_loss_returns.items():
            if name in scales_by_name:
                loss.rescale_reduced_value(scales_by_name[name])
        self.total_value = self.reduce()

    def backward_roots(self) -> dict[str, list[torch.Tensor]]:
        """Group sample roots by name and include shared scalar contributions once."""
        roots: dict[str, list[torch.Tensor]] = defaultdict(list)
        for item in self.batch_loss_returns:
            for name, tensors in item.backward_roots().items():
                roots[name].extend(tensors)
        for name, loss in self._additional_loss_returns.items():
            roots[name].append(loss.reduced_value)
        return dict(roots)

    @torch.no_grad()
    def log(self, logging_fn: Callable[..., None], sync_dist: bool = False) -> None:
        """Logging of all loss_return values and lambda coeffs."""
        loss_values: dict[str, list[torch.Tensor]] = defaultdict(list)
        loss_lambdas: dict[str, list[float]] = defaultdict(list)

        for loss_returns in self.batch_loss_returns:
            for name, value in loss_returns.loss_returns.items():
                loss_values[name].append(value.reduced_value)
                loss_lambdas[name].append(value.lambda_)

        for name, values in loss_values.items():
            _log_weighted_loss(
                logging_fn,
                name,
                torch.mean(torch.stack(values)),
                sum(loss_lambdas[name]) / len(loss_lambdas[name]),
                batch_size=len(values),
                sync_dist=sync_dist,
            )

        for name, loss in self._additional_loss_returns.items():
            _log_weighted_loss(
                logging_fn,
                name,
                loss.reduced_value,
                loss.lambda_,
                batch_size=len(self.batch_loss_returns),
                sync_dist=sync_dist,
            )


@dataclass(slots=True, kw_only=True)
class AccumulatorLoss:
    """Unweighted producer data; reduction and weighting belong to the new path."""

    name: str
    lambda_: float
    value: torch.Tensor
    reduce_fn: RayReduceFn | None = None
    reduce_mask: torch.Tensor | None = None
    reporting_weight: float | None = None

    def reduce_unweighted(self) -> torch.Tensor:
        if self.reduce_fn is None:
            if self.value.numel() != 1:
                raise ValueError(f"{self.name} must supply a scalar or a reducer")
            return self.value.reshape(())
        return self.reduce_fn(self.value, reduce_mask=self.reduce_mask)


class LossAccumulatorReturn:
    """Lazily pack weighted family slots and independently weighted scalar losses.

    Family producers already apply their outer weights. Independent scalars,
    including PPISP, receive their outer weights in the CUDA finalizer. Every
    entry exposed by ``weighted_values`` therefore has the same weighted meaning.
    Reporting weights are retained separately and never reapplied to family slots.
    """

    def __init__(
        self,
        weighted_vector: torch.Tensor,
        slots_by_name: Mapping[str, int],
        weights_by_name: dict[str, float],
    ) -> None:
        self._weighted_vector = weighted_vector
        self._slots_by_name = dict(slots_by_name)
        self.weights_by_name = weights_by_name
        self._pending_losses: list[AccumulatorLoss] = []
        self._scalar_weights: list[float] = []
        self._finalized: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """Names reported this step, including selected zero-weight family losses."""
        return tuple(self.weights_by_name)

    def weighted_value_dtype(self, name: str) -> torch.dtype:
        """Report the finalizer's dtype without materializing the objective."""
        if name not in self.weights_by_name:
            raise KeyError(name)
        return torch.float32

    def append_loss(self, *, name: str, value: torch.Tensor, weight: float = 1.0) -> None:
        """Append an unweighted scalar without finalizing the CUDA objective."""
        _check_scalar_loss(name, value)
        self.append_producer_loss(AccumulatorLoss(name=name, value=value, lambda_=weight))

    def append_producer_loss(self, loss: AccumulatorLoss) -> None:
        """Keep a producer's reduction lazy and snapshot its outer weight."""
        name = loss.name
        if name in self.weights_by_name:
            raise ValueError(f"accumulator already contains loss {name!r}")
        weight = float(loss.lambda_)
        self._slots_by_name[name] = self._weighted_vector.numel() + len(self._pending_losses)
        self.weights_by_name[name] = weight if loss.reporting_weight is None else float(loss.reporting_weight)
        self._pending_losses.append(loss)
        self._scalar_weights.append(weight)
        self._finalized = None

    def backward_roots(self) -> dict[str, list[torch.Tensor]]:
        # Slots share the finalizer's autograd node, so their backward stays shared.
        return {name: [self.weighted_value(name)] for name in self.names}

    def _finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._finalized is not None:
            return self._finalized
        scalars = []
        for loss in self._pending_losses:
            with ScopedTimer(f"LossAccumulator/{loss.name}/reduce"):
                scalars.append(loss.reduce_unweighted())
        with ScopedTimer("LossAccumulator/reduce"):
            result = finalize_accumulated_losses(self._weighted_vector, scalars, self._scalar_weights)
        # Reads under no_grad/inference_mode must not replace the later training
        # objective with detached outputs, including when only a scalar needs grad.
        if torch.is_grad_enabled() or not (
            self._weighted_vector.requires_grad or any(loss.value.requires_grad for loss in self._pending_losses)
        ):
            self._finalized = result
        return result

    def compute_total(self) -> torch.Tensor:
        """CUDA-reduced training objective, finalized on first use and cached safely."""
        return self._finalize()[0]

    @property
    def weighted_values(self) -> torch.Tensor:
        """Weighted family prefix followed by the independent scalar slots."""
        return self._finalize()[1]

    def weighted_value(self, name: str) -> torch.Tensor:
        """Read one reported slot without constructing a per-loss return object."""
        if name not in self.weights_by_name:
            raise KeyError(name)
        return self.weighted_values[self._slots_by_name[name]]

    def rescale_reduced_values(self, scales_by_name: Mapping[str, torch.Tensor]) -> None:
        """Rescale selected weighted slots while preserving their autograd graph."""
        if not scales_by_name:
            return
        source = self.weighted_values
        values = source.clone()
        for name, scale in scales_by_name.items():
            if name not in self.weights_by_name:
                raise KeyError(name)
            slot = self._slots_by_name[name]
            values[slot] = source[slot] * scale
        # All slots now include their outer weights and component scaling.
        # A subsequently appended scalar remains independently weighted.
        self._weighted_vector = values
        self._pending_losses.clear()
        self._scalar_weights.clear()
        self._finalized = None

    def log(self, logging_fn: Callable[..., None]) -> None:
        """Log weighted values without detaching a pending training objective."""
        values = self.weighted_values
        with torch.no_grad():
            slots = values.unbind(0)
            for name, weight in self.weights_by_name.items():
                _log_weighted_loss(logging_fn, name, slots[self._slots_by_name[name]], weight)
