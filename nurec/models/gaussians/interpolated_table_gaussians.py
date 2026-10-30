# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in deformable Gaussian layer backed by interpolated control tables."""

from __future__ import annotations

import logging
import math

from collections.abc import Sequence
from typing import Any, Literal, Optional, Tuple

import torch

from torch import nn

import nurec.models.gaussians.collect as collect

from nurec.config.model import InterpolatedTableDeformableGaussiansLayerConfig
from nurec.config.trainer import TrainerConfig
from nurec.datasets.summary import DataSourceSummary
from nurec.datasets.tracks import CuboidTracks
from nurec.models.gaussians.gaussians_model import BaseGaussianModel, RigidGaussianModel
from nurec.models.gaussians.initializations import BaseInitialization
from nurec.models.gaussians.temporal_deformation_storage import (
    FrameMajorRaggedStorage,
    TemporalDeformationStorage,
)
from nurec.utils.geometry import quat_slerp
from nurec.utils.optim import OptimizerLRSchedulerConfig, configure_optimizers
from nurec.utils.profiling import ScopedTimer
from nurec.utils.trainer import adjust_step_for_world_size


log = logging.getLogger(__name__)


class InterpolatedDeformationTable(nn.Module):
    """A logical ``G x T`` table with position lerp and rotation SLERP.

    The table owns the uniform timestamp-to-column mapping and stable Gaussian
    row identities. Physical parameter layout is delegated to
    :class:`TemporalDeformationStorage`.
    """

    global_row_ids: torch.Tensor
    current_row_for_global_id: torch.Tensor

    def __init__(
        self,
        *,
        reference_positions: torch.Tensor,
        gaussian_cuboid_ids: torch.Tensor,
        track_timestamp_ranges_us: torch.Tensor,
        start_timestamp_us: int,
        end_timestamp_us: int,
        frame_period_us: int,
        interpolation: Literal["nearest", "linear_slerp"],
        storage_backend: Literal["frame_major"],
        allocate_values: bool = True,
    ) -> None:
        super().__init__()
        if frame_period_us <= 0:
            raise ValueError("frame_period_us must be positive")
        if end_timestamp_us < start_timestamp_us:
            raise ValueError("end_timestamp_us must not precede start_timestamp_us")
        if interpolation not in {"nearest", "linear_slerp"}:
            raise ValueError(f"Unsupported deformation interpolation: {interpolation}")
        if track_timestamp_ranges_us.ndim != 2 or track_timestamp_ranges_us.shape[1] != 2:
            raise ValueError("track_timestamp_ranges_us must have shape [num_tracks, 2]")
        if gaussian_cuboid_ids.ndim != 1 or gaussian_cuboid_ids.shape[0] != reference_positions.shape[0]:
            raise ValueError("gaussian_cuboid_ids must have one entry per reference position")

        self.start_timestamp_us = int(start_timestamp_us)
        self.end_timestamp_us = int(end_timestamp_us)
        self.frame_period_us = int(frame_period_us)
        self.interpolation = interpolation
        duration_us = self.end_timestamp_us - self.start_timestamp_us
        num_columns = int(math.ceil(duration_us / self.frame_period_us)) + 1

        ranges = track_timestamp_ranges_us.to(device=gaussian_cuboid_ids.device, dtype=torch.int64)
        track_start_columns = torch.div(
            ranges[:, 0] - self.start_timestamp_us,
            self.frame_period_us,
            rounding_mode="floor",
        ).clamp(0, num_columns - 1)
        track_end_columns = torch.div(
            ranges[:, 1] - self.start_timestamp_us + self.frame_period_us - 1,
            self.frame_period_us,
            rounding_mode="floor",
        ).clamp(0, num_columns - 1)
        stable_group_ids = gaussian_cuboid_ids.to(torch.int64)

        self.storage_backend = storage_backend
        if storage_backend != "frame_major":
            raise ValueError(f"unknown temporal deformation storage backend: {storage_backend}")
        self.storage: TemporalDeformationStorage = FrameMajorRaggedStorage(
            num_columns=num_columns,
            row_start_columns=track_start_columns[stable_group_ids],
            row_end_columns=track_end_columns[stable_group_ids],
            row_group_ids=stable_group_ids,
            value_dim=7,
            dtype=reference_positions.dtype,
            device=reference_positions.device,
            allocate_values=allocate_values,
        )

        stable_ids = torch.arange(self.num_gaussians, device=reference_positions.device, dtype=torch.int64)
        # global_row_ids[current Gaussian row] -> stable table row.
        self.register_buffer("global_row_ids", stable_ids.clone(), persistent=True)
        # Inverse mapping used by full-layer collection.
        self.register_buffer("current_row_for_global_id", stable_ids, persistent=False)

    @property
    def num_frames(self) -> int:
        return self.storage.num_columns

    @property
    def num_gaussians(self) -> int:
        return self.storage.num_rows

    @property
    def num_active_slots(self) -> int:
        return self.storage.num_active_slots

    @property
    def parameter_bytes(self) -> int:
        return self.storage.parameter_bytes

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.current_row_for_global_id.copy_(torch.argsort(self.global_row_ids))

    def frame_index(self, timestamp_us: int) -> int:
        """Map a timestamp to the nearest control column."""

        timestamp = min(max(int(timestamp_us), self.start_timestamp_us), self.end_timestamp_us)
        offset_us = timestamp - self.start_timestamp_us
        # Match row_frame_indices exactly at half-knots and avoid converting
        # absolute microsecond timestamps to floating point.
        index = (offset_us + self.frame_period_us // 2) // self.frame_period_us
        return min(index, self.num_frames - 1)

    def row_frame_indices(self, timestamps_us: torch.Tensor) -> torch.Tensor:
        """Map per-row timestamps to their nearest control columns."""

        timestamps = timestamps_us.to(torch.int64).clamp(self.start_timestamp_us, self.end_timestamp_us)
        offsets = timestamps - self.start_timestamp_us
        # Integer half-up rounding avoids converting absolute microsecond
        # timestamps to floating point.
        return torch.div(
            offsets + self.frame_period_us // 2,
            self.frame_period_us,
            rounding_mode="floor",
        ).clamp(0, self.num_frames - 1)

    def control_frame_pair(self, timestamp_us: int) -> tuple[int, int, float]:
        """Return bounding control columns and their interpolation fraction."""

        timestamp = min(max(int(timestamp_us), self.start_timestamp_us), self.end_timestamp_us)
        offset_us = timestamp - self.start_timestamp_us
        lower = min(offset_us // self.frame_period_us, self.num_frames - 1)
        upper = min(lower + 1, self.num_frames - 1)
        if lower == upper:
            return lower, upper, 0.0
        alpha = (offset_us - lower * self.frame_period_us) / self.frame_period_us
        return lower, upper, float(alpha)

    def row_control_frame_pairs(self, timestamps_us: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return bounding columns and interpolation fractions per output row."""

        timestamps = timestamps_us.to(torch.int64).clamp(self.start_timestamp_us, self.end_timestamp_us)
        offsets = timestamps - self.start_timestamp_us
        lower = torch.div(offsets, self.frame_period_us, rounding_mode="floor").clamp(0, self.num_frames - 1)
        upper = (lower + 1).clamp_max(self.num_frames - 1)
        alpha = (offsets - lower * self.frame_period_us).to(torch.float64) / self.frame_period_us
        alpha = torch.where(lower == upper, torch.zeros_like(alpha), alpha)
        return lower, upper, alpha

    def _dense_residuals_at_frame(
        self,
        frame_index: int,
        gathered_global_row_ids: torch.Tensor,
        reference_positions: torch.Tensor,
        *,
        rows_were_filtered: bool,
    ) -> torch.Tensor:
        """Scatter one compressed column into the collector's current rows."""

        residuals = reference_positions.new_zeros((reference_positions.shape[0], 7))
        column = self.storage.gather_compressed_column(frame_index)
        if column.row_indices.numel() == 0:
            return residuals
        if column.values.shape != (column.row_indices.shape[0], 7):
            raise RuntimeError("deformation storage column row IDs and values do not match")

        if rows_were_filtered:
            output_row_for_global_id = torch.full(
                (self.num_gaussians,),
                -1,
                device=gathered_global_row_ids.device,
                dtype=torch.int64,
            )
            output_row_for_global_id[gathered_global_row_ids.to(torch.int64)] = torch.arange(
                gathered_global_row_ids.shape[0], device=gathered_global_row_ids.device
            )
            output_rows = output_row_for_global_id[column.row_indices]
            keep = output_rows >= 0
            return residuals.index_copy(0, output_rows[keep], column.values[keep])

        output_rows = self.current_row_for_global_id[column.row_indices]
        return residuals.index_copy(0, output_rows, column.values)

    def _dense_residuals_at_row_columns(
        self,
        column_indices: torch.Tensor,
        gathered_global_row_ids: torch.Tensor,
        reference_positions: torch.Tensor,
        *,
        rows_were_filtered: bool,
    ) -> torch.Tensor:
        """Gather the compressed columns requested by individual output rows.

        Resolved actor timestamps normally span only one or two 10 Hz control
        intervals in a rendered frame. Grouping by column retains the sparse
        frame-major hot path and touches only Parameters that contribute.
        """

        if column_indices.shape != (reference_positions.shape[0],):
            raise ValueError("per-row deformation columns must match the collected Gaussian count")
        result = reference_positions.new_zeros((reference_positions.shape[0], 7))
        # ParameterDict lookup requires a host integer. Quantization happens
        # before this synchronization, so one transfer covers every actor.
        used_columns = torch.unique(column_indices.detach()).cpu().tolist()
        for column_index in used_columns:
            column_values = self._dense_residuals_at_frame(
                int(column_index),
                gathered_global_row_ids,
                reference_positions,
                rows_were_filtered=rows_were_filtered,
            )
            result = torch.where((column_indices == column_index).unsqueeze(1), column_values, result)
        return result

    @staticmethod
    def _rotation_residuals_to_quaternions(rotation_residuals: torch.Tensor) -> torch.Tensor:
        """Convert collector ``[xyz, w-1]`` residuals into unit quaternions."""

        quaternions = torch.cat((rotation_residuals[:, :3], rotation_residuals[:, 3:] + 1.0), dim=-1)
        norms = torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
        identity = torch.zeros_like(quaternions)
        identity[:, 3] = 1.0
        return torch.where(norms > 1.0e-12, quaternions / norms.clamp_min(1.0e-12), identity)

    def forward(
        self,
        timestamp_us: int | torch.Tensor,
        gathered_global_row_ids: torch.Tensor,
        reference_positions: torch.Tensor,
        *,
        rows_were_filtered: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return dense position and rotation residuals in collector row order."""

        if isinstance(timestamp_us, torch.Tensor):
            if timestamp_us.dtype != torch.int64 or timestamp_us.device != reference_positions.device:
                raise ValueError("per-row deformation timestamps must be int64 on the Gaussian device")
            if timestamp_us.shape != (reference_positions.shape[0],):
                raise ValueError("per-row deformation timestamps must match the collected Gaussian count")
            if self.interpolation == "nearest":
                values = self._dense_residuals_at_row_columns(
                    self.row_frame_indices(timestamp_us),
                    gathered_global_row_ids,
                    reference_positions,
                    rows_were_filtered=rows_were_filtered,
                )
                return {"positions": values[:, :3], "rotations": values[:, 3:]}

            lower_indices, upper_indices, alpha = self.row_control_frame_pairs(timestamp_us)
            # Exact knots must not touch the following Parameter: keeping its
            # gradient absent preserves frame-major Adam laziness.
            effective_upper_indices = torch.where(alpha == 0.0, lower_indices, upper_indices)
            lower = self._dense_residuals_at_row_columns(
                lower_indices,
                gathered_global_row_ids,
                reference_positions,
                rows_were_filtered=rows_were_filtered,
            )
            upper = self._dense_residuals_at_row_columns(
                effective_upper_indices,
                gathered_global_row_ids,
                reference_positions,
                rows_were_filtered=rows_were_filtered,
            )
            interpolation_steps = alpha.to(dtype=lower.dtype).unsqueeze(1)
            positions = torch.lerp(lower[:, :3], upper[:, :3], interpolation_steps)
            lower_quaternions = self._rotation_residuals_to_quaternions(lower[:, 3:])
            upper_quaternions = self._rotation_residuals_to_quaternions(upper[:, 3:])
            rotations = quat_slerp(lower_quaternions, upper_quaternions, alpha.to(dtype=lower.dtype))
            rotations = torch.cat((rotations[:, :3], rotations[:, 3:] - 1.0), dim=-1)
            return {"positions": positions, "rotations": rotations}

        if self.interpolation == "nearest":
            values = self._dense_residuals_at_frame(
                self.frame_index(timestamp_us),
                gathered_global_row_ids,
                reference_positions,
                rows_were_filtered=rows_were_filtered,
            )
            return {"positions": values[:, :3], "rotations": values[:, 3:]}

        lower_index, upper_index, scalar_alpha = self.control_frame_pair(timestamp_us)
        lower = self._dense_residuals_at_frame(
            lower_index,
            gathered_global_row_ids,
            reference_positions,
            rows_were_filtered=rows_were_filtered,
        )
        # Avoid touching the next Parameter at exact knots so its optimizer
        # state remains lazy until that column actually contributes.
        if lower_index == upper_index or scalar_alpha == 0.0:
            return {"positions": lower[:, :3], "rotations": lower[:, 3:]}

        upper = self._dense_residuals_at_frame(
            upper_index,
            gathered_global_row_ids,
            reference_positions,
            rows_were_filtered=rows_were_filtered,
        )
        positions = torch.lerp(lower[:, :3], upper[:, :3], scalar_alpha)
        lower_quaternions = self._rotation_residuals_to_quaternions(lower[:, 3:])
        upper_quaternions = self._rotation_residuals_to_quaternions(upper[:, 3:])
        steps = lower_quaternions.new_full((lower_quaternions.shape[0],), scalar_alpha)
        rotations = quat_slerp(lower_quaternions, upper_quaternions, steps)
        rotations = torch.cat((rotations[:, :3], rotations[:, 3:] - 1.0), dim=-1)
        return {"positions": positions, "rotations": rotations}

    def position_curvature(
        self,
        timestamp_us: int | torch.Tensor,
        gathered_global_row_ids: torch.Tensor,
        reference_positions: torch.Tensor,
        *,
        rows_were_filtered: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return position second differences and a complete-triplet mask."""

        curvature = reference_positions.new_zeros((reference_positions.shape[0], 3))
        valid = reference_positions.new_zeros((reference_positions.shape[0], 1))
        stable_rows = gathered_global_row_ids.to(torch.int64)
        if isinstance(timestamp_us, torch.Tensor):
            if timestamp_us.dtype != torch.int64 or timestamp_us.device != reference_positions.device:
                raise ValueError("per-row deformation timestamps must be int64 on the Gaussian device")
            center_indices = self.row_frame_indices(timestamp_us)
            if center_indices.shape != (reference_positions.shape[0],):
                raise ValueError("per-row deformation timestamps must match the collected Gaussian count")
            previous = self._dense_residuals_at_row_columns(
                (center_indices - 1).clamp_min(0),
                gathered_global_row_ids,
                reference_positions,
                rows_were_filtered=rows_were_filtered,
            )
            current = self._dense_residuals_at_row_columns(
                center_indices,
                gathered_global_row_ids,
                reference_positions,
                rows_were_filtered=rows_were_filtered,
            )
            following = self._dense_residuals_at_row_columns(
                (center_indices + 1).clamp_max(self.num_frames - 1),
                gathered_global_row_ids,
                reference_positions,
                rows_were_filtered=rows_were_filtered,
            )
            interior = (center_indices > 0) & (center_indices < self.num_frames - 1)
            row_has_previous = self.storage.row_start_columns[stable_rows] <= center_indices - 1
            row_has_following = self.storage.row_end_columns[stable_rows] >= center_indices + 1
        else:
            center_index = self.frame_index(timestamp_us)
            if center_index == 0 or center_index == self.num_frames - 1:
                return curvature, valid
            previous = self._dense_residuals_at_frame(
                center_index - 1,
                gathered_global_row_ids,
                reference_positions,
                rows_were_filtered=rows_were_filtered,
            )
            current = self._dense_residuals_at_frame(
                center_index,
                gathered_global_row_ids,
                reference_positions,
                rows_were_filtered=rows_were_filtered,
            )
            following = self._dense_residuals_at_frame(
                center_index + 1,
                gathered_global_row_ids,
                reference_positions,
                rows_were_filtered=rows_were_filtered,
            )
            interior = torch.ones(reference_positions.shape[0], dtype=torch.bool, device=reference_positions.device)
            row_has_previous = self.storage.row_start_columns[stable_rows] <= center_index - 1
            row_has_following = self.storage.row_end_columns[stable_rows] >= center_index + 1
        curvature = 0.5 * (previous[:, :3] + following[:, :3]) - current[:, :3]
        row_valid = interior & row_has_previous & row_has_following
        return curvature, row_valid.to(reference_positions.dtype).unsqueeze(1)

    @torch.no_grad()
    def apply_mcmc_relocation(
        self,
        source_row_map: torch.Tensor,
        retained_count: int,
        recycled_row_indices: torch.Tensor,
        optimizers: Sequence[OptimizerLRSchedulerConfig],
    ) -> None:
        """Copy donor trajectories into fixed storage slots recycled by MCMC."""

        source_rows = source_row_map.to(device=self.global_row_ids.device, dtype=torch.int64)
        recycled_rows = recycled_row_indices.to(device=self.global_row_ids.device, dtype=torch.int64)
        if source_rows.numel() != self.num_gaussians or source_rows.numel() - retained_count != recycled_rows.numel():
            raise ValueError("deformation-table MCMC relocation must preserve the Gaussian count")

        old_stable_ids = self.global_row_ids.clone()
        donor_stable_ids = old_stable_ids[source_rows[retained_count:]]
        recycled_stable_ids = old_stable_ids[recycled_rows]
        self.storage.copy_rows(recycled_stable_ids, donor_stable_ids, optimizers)

        new_stable_ids = torch.cat(
            (old_stable_ids[source_rows[:retained_count]], recycled_stable_ids),
            dim=0,
        )
        if torch.unique(new_stable_ids).numel() != self.num_gaussians:
            raise RuntimeError("deformation-table relocation produced duplicate stable row IDs")
        self.global_row_ids.copy_(new_stable_ids)
        self.current_row_for_global_id.copy_(torch.argsort(new_stable_ids))

    @torch.no_grad()
    def apply_row_permutation(self, order: torch.Tensor) -> None:
        """Keep stable table identities aligned with a Gaussian row reorder."""

        reordered_ids = self.global_row_ids[order.to(self.global_row_ids.device)]
        self.global_row_ids.copy_(reordered_ids)
        self.current_row_for_global_id.copy_(torch.argsort(reordered_ids))


class InterpolatedTableDeformableGaussianModel(RigidGaussianModel):
    """Deformable actor layer with explicit per-Gaussian temporal knots."""

    config: InterpolatedTableDeformableGaussiansLayerConfig  # type: ignore[assignment]

    def __init__(
        self,
        config: InterpolatedTableDeformableGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: Optional[BaseInitialization],
        precision: Optional[int] = None,
        cuboid_tracks: Optional[CuboidTracks] = None,
        start_end_timestamp_us: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Construct a rigid actor layer whose deformations live in a knot table."""

        super().__init__(
            config,
            trainer_config,
            datasource,
            init_from_datasource,
            initializer,
            precision,
            cuboid_tracks,
            start_end_timestamp_us,
        )
        self.config = self.config.model_copy(
            update={
                "deformation_start_iteration": adjust_step_for_world_size(
                    trainer_config, self.config.deformation_start_iteration
                )
            },
            deep=True,
        )
        self.deformation_table: InterpolatedDeformationTable | None = None
        self.deformation_optimizers: list[OptimizerLRSchedulerConfig] = []
        self._frozen_deformation_track_indices: tuple[int, ...] = ()
        self.use_deformation = False

    @staticmethod
    def _track_timestamp_ranges_us(cuboid_tracks: CuboidTracks) -> torch.Tensor:
        packinfo = cuboid_tracks.tracks_packinfo.to(device="cpu", dtype=torch.int64)
        timestamps = cuboid_tracks.tracks_timestamps_us.to(device="cpu", dtype=torch.int64)
        ranges = []
        for start, count in packinfo.tolist():
            if count <= 0:
                raise ValueError("Deformable tracks must contain at least one timestamp")
            track_timestamps = timestamps[start : start + count]
            ranges.append(torch.stack((track_timestamps[0], track_timestamps[-1])))
        if not ranges:
            return torch.empty((0, 2), dtype=torch.int64, device=cuboid_tracks.device)
        return torch.stack(ranges).to(cuboid_tracks.device)

    def _make_deformation_table(
        self,
        reference_positions: torch.Tensor,
        gaussian_cuboid_ids: torch.Tensor,
        *,
        allocate_values: bool,
    ) -> InterpolatedDeformationTable:
        if self.trainer_config.world_size != 1:
            raise NotImplementedError("interpolated deformation tables currently support one training rank")
        timestamp_ranges = self._track_timestamp_ranges_us(self.cuboid_tracks)
        if self.start_end_timestamp_us is not None:
            start_timestamp_us, end_timestamp_us = self.start_end_timestamp_us
        elif timestamp_ranges.shape[0] > 0:
            start_timestamp_us = int(timestamp_ranges[:, 0].min().item())
            end_timestamp_us = int(timestamp_ranges[:, 1].max().item())
        else:
            # An empty layer still needs a valid timeline. Its origin is
            # immaterial because no Gaussian rows can query the table.
            start_timestamp_us = end_timestamp_us = 0
        return InterpolatedDeformationTable(
            reference_positions=reference_positions,
            gaussian_cuboid_ids=gaussian_cuboid_ids,
            track_timestamp_ranges_us=timestamp_ranges,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            frame_period_us=self.config.table.frame_period_us,
            interpolation=self.config.table.interpolation,
            storage_backend=self.config.table.storage,
            allocate_values=allocate_values,
        )

    def initialize_gaussians_from_datasource(
        self,
        initializer: BaseInitialization,
        datasource: DataSourceSummary,
        cuboid_tracks: Optional[CuboidTracks] = None,
    ) -> None:
        super().initialize_gaussians_from_datasource(initializer, datasource, cuboid_tracks)
        self.deformation_table = self._make_deformation_table(
            self.positions,
            self.gaussian_cuboid_ids,
            allocate_values=True,
        )
        parameter_gib = self.deformation_table.parameter_bytes / (1024**3)
        log.info(
            "InterpolatedDeformationTable: storage=%s gaussians=%d columns=%d active_slots=%d "
            "parameters=%.3f_GiB estimated_parameters_plus_adam=%.3f_GiB",
            self.config.table.storage,
            self.deformation_table.num_gaussians,
            self.deformation_table.num_frames,
            self.deformation_table.num_active_slots,
            parameter_gib,
            3.0 * parameter_gib,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        if self.deformation_table is None:
            positions_key = prefix + "positions"
            cuboid_ids_key = prefix + "gaussian_cuboid_ids"
            if positions_key in state_dict and cuboid_ids_key in state_dict:
                self.deformation_table = self._make_deformation_table(
                    state_dict[positions_key],
                    state_dict[cuboid_ids_key],
                    allocate_values=False,
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

    def get_parameters(self) -> dict[str, torch.Tensor]:
        parameters = super().get_parameters()
        if self.deformation_table is not None:
            if self.get_num_gaussians() != self.deformation_table.num_gaussians:
                raise RuntimeError("interpolated deformation tables require fixed Gaussian counts")
            parameters["deformation_table_global_row_ids"] = self.deformation_table.global_row_ids
        return parameters

    def get_layer_config(self) -> collect.LayerConfigDeformable:
        layer_config = super().get_layer_config()
        assert isinstance(layer_config, collect.LayerConfigSH)
        return collect.LayerConfigDeformable(
            rotation_activation=layer_config.rotation_activation,
            scale_activation=layer_config.scale_activation,
            density_activation=layer_config.density_activation,
            fourier_features_dim=layer_config.fourier_features_dim,
            embed_config=layer_config.embed_config,
        )

    def get_layer_data(
        self,
        context: BaseGaussianModel.CollectionContext,
        gathered_parameters: dict[str, torch.Tensor],
    ) -> collect.LayerDataDeformable:
        base_layer_data = super().get_layer_data(context, gathered_parameters)
        deform_positions = None
        deform_rotations = None

        if self.use_deformation and base_layer_data.tracks_ids.numel() > 0:
            if self.deformation_table is None:
                raise RuntimeError("deformation table was not initialized")
            with ScopedTimer("InterpolatedTableDeformableGaussianModel/table"):
                track_timestamps = context.additional_data["interpolated_track_timestamps"][0]
                timestamp_us = track_timestamps[base_layer_data.tracks_ids]
                rows_were_filtered = context.kept_rows is not None
                deformations = self.deformation_table(
                    timestamp_us,
                    gathered_parameters["deformation_table_global_row_ids"],
                    base_layer_data.positions,
                    rows_were_filtered=rows_were_filtered,
                )
                deform_positions = deformations["positions"].to(base_layer_data.positions)
                deform_rotations = deformations["rotations"].to(base_layer_data.rotations)

                if context.is_training_batch and self.config.table.position_smoothness:
                    smoothness, lifetime_mask = self.deformation_table.position_curvature(
                        timestamp_us,
                        gathered_parameters["deformation_table_global_row_ids"],
                        base_layer_data.positions,
                        rows_were_filtered=rows_were_filtered,
                    )
                    interpolated_mask = context.additional_data["interpolated_mask"][0]
                    track_mask = interpolated_mask[base_layer_data.tracks_ids].to(lifetime_mask.dtype).unsqueeze(1)
                    context.additional_data["deform_smoothness"] = (smoothness, True)
                    context.additional_data["deform_smoothness_mask"] = (lifetime_mask * track_mask, True)

        return collect.LayerDataDeformable(
            positions=base_layer_data.positions,
            rotations=base_layer_data.rotations,
            scales=base_layer_data.scales,
            densities=base_layer_data.densities,
            extra_signal=base_layer_data.extra_signal,
            camera_extra_signal=base_layer_data.camera_extra_signal,
            lidar_extra_signal=base_layer_data.lidar_extra_signal,
            features_albedo=base_layer_data.features_albedo,
            features_specular=base_layer_data.features_specular,
            embed_data=base_layer_data.embed_data,
            poses=base_layer_data.poses,
            keep_mask=base_layer_data.keep_mask,
            tracks_ids=base_layer_data.tracks_ids,
            deform_positions=deform_positions,
            deform_rotations=deform_rotations,
        )

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """Create canonical and table optimizers, then bind frozen-row guards."""

        if self.deformation_table is None:
            raise RuntimeError("cannot configure a deformation-table optimizer before initialization")
        self.deformation_optimizers = configure_optimizers(
            self.config.table,
            self.trainer_config,
            self.deformation_table,
            name_prefix,
        )
        if self._frozen_deformation_track_indices:
            self.deformation_table.storage.register_frozen_groups(
                self._frozen_deformation_track_indices,
                self.deformation_optimizers,
            )
        return super().configure_optimizers(name_prefix) + self.deformation_optimizers

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        if global_step >= self.config.deformation_start_iteration:
            self.use_deformation = True
        return super().update_step_train_batch_start(epoch, global_step, system, **kwargs)

    def get_extra_state(self) -> dict[str, Any] | None:
        state = super().get_extra_state() or {}
        state["use_deformation"] = self.use_deformation
        return state

    def set_extra_state(self, state: dict[str, Any] | None) -> None:
        if state is not None and "use_deformation" in state:
            self.use_deformation = bool(state["use_deformation"])
        super().set_extra_state(state)

    def freeze_tracks(self, frozen_track_indices: Sequence[int]) -> None:
        """Freeze canonical, calibration, and table state for selected actors."""

        super().freeze_tracks(frozen_track_indices)
        self._frozen_deformation_track_indices = tuple(
            sorted(set(self._frozen_deformation_track_indices).union(map(int, frozen_track_indices)))
        )
        if self.deformation_table is not None:
            self.deformation_table.storage.register_frozen_groups(
                self._frozen_deformation_track_indices,
                self.deformation_optimizers,
            )

    def validate_external_gaussian_topology_edit(self, operation: str) -> None:
        """Reject external row edits that cannot update table trajectories."""

        raise NotImplementedError(f"{operation} is not supported by fixed-size deformation tables")

    def apply_mcmc_row_update(
        self,
        source_row_map: torch.Tensor,
        retained_count: int,
        recycled_row_indices: torch.Tensor | None,
    ) -> None:
        if self.deformation_table is None:
            return
        if recycled_row_indices is None:
            raise ValueError("interpolated deformation tables support fixed-count same-track MCMC relocation only")
        self.deformation_table.apply_mcmc_relocation(
            source_row_map,
            retained_count,
            recycled_row_indices,
            self.deformation_optimizers,
        )

    def validate_mcmc_row_update(self, recycled_row_indices: torch.Tensor | None) -> None:
        if self.deformation_table is not None and recycled_row_indices is None:
            raise ValueError("interpolated deformation tables support fixed-count same-track MCMC relocation only")

    def _permute_gaussian_rows(self, order: torch.Tensor) -> None:
        super()._permute_gaussian_rows(order)
        # Some generic Gaussian tests intentionally materialize registered
        # variants without calling their constructors. Treat that synthetic
        # state like a model whose table has not been initialized yet.
        deformation_table = getattr(self, "deformation_table", None)
        if deformation_table is not None:
            deformation_table.apply_row_permutation(order)


BaseGaussianModel.register_to_gaussians_factory(
    "interpolated-table-deformable-gaussians",
    InterpolatedTableDeformableGaussianModel,
)
