# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical storage backends for a logically sparse Gaussian-by-time table.

This module deliberately knows nothing about timestamps, cuboids, canonical
positions, or interpolation. Callers convert those concepts into stable row
IDs and integer control columns before constructing a backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from torch import nn


if TYPE_CHECKING:
    from nurec.utils.optim import OptimizerLRSchedulerConfig


@dataclass(frozen=True)
class CompressedDeformationColumn:
    """The stored rows and differentiable values for one logical column."""

    row_indices: torch.Tensor  # [N], stable logical row IDs
    values: torch.Tensor  # [N, value_dim]


class TemporalDeformationStorage(nn.Module, ABC):
    """Storage API for a logically sparse ``G x T`` deformation table.

    A backend only understands integer rows and columns. ``row_start_columns``
    and ``row_end_columns`` describe each row's inclusive stored interval;
    ``row_group_ids`` lets a backend group rows and prevents MCMC from copying
    trajectories between actors.
    """

    row_start_columns: torch.Tensor
    row_end_columns: torch.Tensor
    row_group_ids: torch.Tensor
    _empty_values: torch.Tensor

    def __init__(
        self,
        *,
        num_columns: int,
        row_start_columns: torch.Tensor,
        row_end_columns: torch.Tensor,
        row_group_ids: torch.Tensor,
        value_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if num_columns <= 0:
            raise ValueError("num_columns must be positive")
        if value_dim <= 0:
            raise ValueError("value_dim must be positive")
        if not (
            row_start_columns.ndim == row_end_columns.ndim == row_group_ids.ndim == 1
            and row_start_columns.shape == row_end_columns.shape == row_group_ids.shape
        ):
            raise ValueError("row lifetime and group tensors must be same-length vectors")

        starts = row_start_columns.to(device=device, dtype=torch.int64)
        ends = row_end_columns.to(device=device, dtype=torch.int64)
        groups = row_group_ids.to(device=device, dtype=torch.int64)
        if starts.numel() and (
            bool((starts < 0).any().item())
            or bool((ends >= num_columns).any().item())
            or bool((starts > ends).any().item())
        ):
            raise ValueError("row lifetimes must be non-empty intervals inside the table")

        self.num_columns = int(num_columns)
        self.num_rows = int(starts.numel())
        self.value_dim = int(value_dim)
        self.register_buffer("row_start_columns", starts.clone(), persistent=True)
        self.register_buffer("row_end_columns", ends.clone(), persistent=True)
        self.register_buffer("row_group_ids", groups.clone(), persistent=True)
        self.register_buffer(
            "_empty_values",
            torch.empty((0, self.value_dim), dtype=dtype, device=device),
            persistent=False,
        )

        stored_columns = ends - starts + 1
        self.num_active_slots = int(stored_columns.sum().item())
        self.parameter_bytes = self.num_active_slots * self.value_dim * torch.empty((), dtype=dtype).element_size()

    def empty_column(self) -> CompressedDeformationColumn:
        return CompressedDeformationColumn(
            row_indices=self.row_group_ids.new_empty((0,), dtype=torch.int64),
            values=self._empty_values,
        )

    def active_rows(self, column_index: int) -> torch.Tensor:
        """Return stable rows whose stored interval contains ``column_index``."""

        return torch.where((self.row_start_columns <= column_index) & (column_index <= self.row_end_columns))[0]

    @abstractmethod
    def gather_compressed_column(self, column_index: int) -> CompressedDeformationColumn:
        """Return stable row IDs and differentiable values for one column."""

    def copy_rows(
        self,
        target_row_indices: torch.Tensor,
        source_row_indices: torch.Tensor,
        optimizers: Sequence[OptimizerLRSchedulerConfig],
    ) -> None:
        """Copy complete same-actor trajectories and clear target moments."""

        targets = target_row_indices.to(device=self.row_group_ids.device, dtype=torch.int64)
        sources = source_row_indices.to(device=self.row_group_ids.device, dtype=torch.int64)
        if targets.shape != sources.shape:
            raise ValueError("target and source deformation row IDs must have the same shape")
        same_group = self.row_group_ids[targets] == self.row_group_ids[sources]
        same_lifetime = (self.row_start_columns[targets] == self.row_start_columns[sources]) & (
            self.row_end_columns[targets] == self.row_end_columns[sources]
        )
        if not bool((same_group & same_lifetime).all().item()):
            raise ValueError("deformation-table relocation requires donors from the same actor and lifetime")
        self._copy_rows(targets, sources, optimizers)

    @abstractmethod
    def _copy_rows(
        self,
        target_row_indices: torch.Tensor,
        source_row_indices: torch.Tensor,
        optimizers: Sequence[OptimizerLRSchedulerConfig],
    ) -> None:
        """Backend implementation of :meth:`copy_rows`."""

    @abstractmethod
    def register_frozen_groups(
        self,
        frozen_group_ids: Sequence[int],
        optimizers: Sequence[OptimizerLRSchedulerConfig] = (),
    ) -> None:
        """Freeze group rows in both gradients and optimizer state."""

    @staticmethod
    def _zero_optimizer_slices(
        parameter: nn.Parameter,
        dimension: int,
        indices: torch.Tensor,
        optimizers: Sequence[OptimizerLRSchedulerConfig],
    ) -> None:
        if parameter.grad is not None and parameter.grad.shape == parameter.shape:
            parameter.grad.index_fill_(dimension, indices, 0)
        for optimizer_config in optimizers:
            state = optimizer_config["optimizer"].state.get(parameter)
            if state is None:
                continue
            for value in state.values():
                if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                    value.index_fill_(dimension, indices, 0)


class _ColumnValues(nn.Module):
    """Learnable values for the active rows in one global control column."""

    values: nn.Parameter

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
        allocate_values: bool,
    ) -> None:
        super().__init__()
        if allocate_values:
            self.values = nn.Parameter(torch.zeros(shape, dtype=dtype, device=device))
        else:
            # Artifact loading uses assign=True to install the checkpoint tensor
            # without allocating a duplicate table first.
            self.values = cast(nn.Parameter, nn.UninitializedParameter())


class FrameMajorRaggedStorage(TemporalDeformationStorage):
    """One Parameter per compressed time column.

    This is the compatibility/performance backend: gathering a column returns
    the existing Parameter directly, without concatenation or repacking.
    """

    def __init__(self, *, allocate_values: bool = True, **kwargs) -> None:
        """Build one learnable value block for every non-empty time column."""

        super().__init__(**kwargs)
        count_deltas = torch.zeros(
            self.num_columns + 1,
            device=self.row_start_columns.device,
            dtype=torch.int64,
        )
        ones = torch.ones_like(self.row_start_columns)
        count_deltas.index_add_(0, self.row_start_columns, ones)
        count_deltas.index_add_(0, self.row_end_columns + 1, -ones)

        self.columns = nn.ModuleDict()
        self._frozen_gradient_handles: dict[str, torch.utils.hooks.RemovableHandle] = {}
        dtype = self._empty_values.dtype
        device = self._empty_values.device
        for column_index, active_count in enumerate(count_deltas[:-1].cumsum(0).cpu().tolist()):
            if active_count:
                self.columns[str(column_index)] = _ColumnValues(
                    (int(active_count), self.value_dim),
                    dtype=dtype,
                    device=device,
                    allocate_values=allocate_values,
                )

    def _column(self, key: str) -> _ColumnValues:
        """Return a column narrowed from ``ModuleDict``'s generic module type."""

        return cast(_ColumnValues, self.columns[key])

    def gather_compressed_column(self, column_index: int) -> CompressedDeformationColumn:
        if not 0 <= column_index < self.num_columns:
            raise IndexError(f"deformation column {column_index} is outside [0, {self.num_columns})")
        key = str(column_index)
        if key not in self.columns:
            return self.empty_column()
        return CompressedDeformationColumn(self.active_rows(column_index), self._column(key).values)

    @torch.no_grad()
    def _copy_rows(
        self,
        target_row_indices: torch.Tensor,
        source_row_indices: torch.Tensor,
        optimizers: Sequence[OptimizerLRSchedulerConfig],
    ) -> None:
        for key in self.columns:
            column = self._column(key)
            column_index = int(key)
            pair_is_active = (self.row_start_columns[target_row_indices] <= column_index) & (
                column_index <= self.row_end_columns[target_row_indices]
            )
            if not bool(pair_is_active.any().item()):
                continue
            active_rows = self.active_rows(column_index)
            target_slots = torch.searchsorted(active_rows, target_row_indices[pair_is_active])
            source_slots = torch.searchsorted(active_rows, source_row_indices[pair_is_active])
            column.values.index_copy_(0, target_slots, column.values[source_slots])
            self._zero_optimizer_slices(column.values, 0, target_slots, optimizers)

    def register_frozen_groups(
        self,
        frozen_group_ids: Sequence[int],
        optimizers: Sequence[OptimizerLRSchedulerConfig] = (),
    ) -> None:
        """Freeze selected actors without letting Adam momentum move their rows."""

        # This method may run before and after optimizer construction. Replace
        # old hooks so the active hook always captures the current optimizers.
        for handle in self._frozen_gradient_handles.values():
            handle.remove()
        self._frozen_gradient_handles.clear()

        frozen = torch.as_tensor(frozen_group_ids, device=self.row_group_ids.device, dtype=torch.int64)
        for key in self.columns:
            column = self._column(key)
            if not column.values.requires_grad:
                continue
            rows = self.active_rows(int(key))
            trainable = ~torch.isin(self.row_group_ids[rows], frozen)
            mask = trainable.to(column.values.dtype).unsqueeze(1)
            frozen_slots = torch.where(~trainable)[0]
            if frozen_slots.numel() == 0:
                continue

            self._zero_optimizer_slices(column.values, 0, frozen_slots, optimizers)

            def freeze_rows(
                gradient: torch.Tensor,
                *,
                parameter: nn.Parameter = column.values,
                gradient_mask: torch.Tensor = mask,
                slots: torch.Tensor = frozen_slots,
            ) -> torch.Tensor:
                """Mask frozen gradients and discard any restored optimizer momentum."""

                # Checkpoint restoration can repopulate Adam moments after the
                # freeze is registered. Clear them immediately before every
                # participating optimizer step as well as at registration.
                self._zero_optimizer_slices(parameter, 0, slots, optimizers)
                return gradient * gradient_mask

            self._frozen_gradient_handles[key] = column.values.register_hook(freeze_rows)
