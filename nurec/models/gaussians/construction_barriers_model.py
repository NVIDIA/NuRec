# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static barrier Gaussians with persistent track identity and cuboid-local noise bounds."""

from typing import Any

import torch

from torch import nn

from nurec.config.model import ConstructionBarriersGaussiansLayerConfig
from nurec.config.trainer import TrainerConfig
from nurec.datasets.summary import DataSourceSummary
from nurec.datasets.tracks import CuboidTracks
from nurec.models.base import BaseModel
from nurec.models.gaussians.gaussians_model import BaseGaussianModel, SHGaussianModel
from nurec.models.gaussians.initializations import BaseInitialization, ConstructionBarriersInitialization


class ConstructionBarriersGaussianModel(SHGaussianModel):
    """One full-clip static layer; track IDs group topology, not render lifetimes."""

    config: ConstructionBarriersGaussiansLayerConfig
    barrier_object_id: nn.Buffer | nn.UninitializedBuffer

    def __init__(
        self,
        config: ConstructionBarriersGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: BaseInitialization | None,
        precision: int | None = None,
        cuboid_tracks: CuboidTracks | None = None,
        start_end_timestamp_us: tuple[int, int] | None = None,
    ) -> None:
        """Create the SH layer with persistent row identities and source-cuboid metadata."""
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
        self.barrier_object_id = nn.UninitializedBuffer(dtype=torch.int64)  # type: ignore[call-arg]
        self._cuboid_records: list[dict[str, Any]] = []
        self._cuboid_constraint_cache: tuple[torch.Tensor, ...] | None = None

    def initialize_gaussians_from_datasource(
        self,
        initializer: BaseInitialization,
        datasource: DataSourceSummary,
        cuboid_tracks: CuboidTracks | None = None,
    ) -> None:
        """Initialize fixed-per-cuboid seeds and align their IDs with the selected point shard."""
        if not isinstance(initializer, ConstructionBarriersInitialization):
            raise TypeError("Construction barriers require ConstructionBarriersInitialization")
        super().initialize_gaussians_from_datasource(initializer, datasource, cuboid_tracks)
        boxes = initializer.cuboids
        if boxes is None:
            raise ValueError("Barrier initialization did not provide source cuboids")
        rank, world_size = self._get_slice_boundaries()
        ids = boxes.track_ids.repeat_interleave(initializer.config.points_per_cuboid)
        self.barrier_object_id = nn.Buffer(ids[rank::world_size].to(self.device).contiguous())
        self._cuboid_records = [
            {
                "track_id": int(track_id),
                "class_id": int(boxes.class_ids[index]),
                "center_nurec": boxes.centers[index].tolist(),
                "dimensions_m": boxes.dimensions[index].tolist(),
                "rotation_nurec": boxes.rotations[index].tolist(),
            }
            for index, track_id in enumerate(boxes.track_ids.tolist())
        ]

    def _validate_fields(self) -> None:
        """Require one integer track identity per Gaussian and restorable cuboid geometry."""
        super()._validate_fields()
        if self.barrier_object_id.dtype != torch.int64 or self.barrier_object_id.shape != (len(self.positions),):
            raise ValueError("Barrier track identity must be row-aligned int64")
        if not self._cuboid_records:
            raise ValueError("Barrier checkpoint is missing source cuboid metadata")

    def configure_sharded_params_and_buffers(self) -> list[str]:
        """Mark track identities for the same checkpoint sharding as Gaussian rows."""
        configured = super().configure_sharded_params_and_buffers()
        BaseModel.mark_as_sharded(self.barrier_object_id)
        return configured + ["barrier_object_id"]

    def get_additional_buffers(self) -> dict[str, torch.Tensor]:
        """Expose track IDs to native row reordering and donor-state propagation."""
        # Native reorder/topology publication propagates each donor's identity.
        return super().get_additional_buffers() | {"barrier_object_id": self.barrier_object_id}

    def get_extra_state(self) -> dict[str, Any] | None:
        """Include source cuboids so a checkpoint can restore without rereading detections."""
        state = super().get_extra_state() or {}
        state["barrier_cuboid_records"] = self._cuboid_records
        return state

    def set_extra_state(self, state: dict[str, Any] | None) -> None:
        """Restore required cuboid metadata and invalidate the derived geometry cache."""
        super().set_extra_state(state)
        if state is None or "barrier_cuboid_records" not in state:
            raise ValueError("Barrier checkpoint must contain source cuboid metadata")
        self._cuboid_records = state["barrier_cuboid_records"]
        self._cuboid_constraint_cache = None

    @torch.no_grad()
    def accept_cuboid_perturbation(self, proposal: torch.Tensor) -> None:
        """Reject outward noise per local axis, following the native rigid rule.

        Adam steps are not clamped. An already-outside point may move inward,
        but its random perturbation may not take it farther outside. Rejected
        components are zeroed before publication, never snapped onto a face.
        """
        if proposal.shape != self.positions.shape:
            raise ValueError("Cuboid rejection requires a row-aligned position proposal")
        cached = self._cuboid_constraint_cache
        if cached is None or cached[1].device != proposal.device or cached[1].dtype != proposal.dtype:
            records = sorted(self._cuboid_records, key=lambda row: row["track_id"])
            track_ids = torch.tensor([row["track_id"] for row in records], device=proposal.device, dtype=torch.int64)
            centers = proposal.new_tensor([row["center_nurec"] for row in records])
            rotations = proposal.new_tensor([row["rotation_nurec"] for row in records])
            dimensions = proposal.new_tensor([row["dimensions_m"] for row in records])
            if (
                not records
                or len({row["track_id"] for row in records}) != len(records)
                or not bool(
                    torch.isfinite(centers).all()
                    & torch.isfinite(rotations).all()
                    & torch.isfinite(dimensions).all()
                    & (dimensions > 0).all()
                )
                or not bool(torch.isin(self.barrier_object_id, track_ids).all())
            ):
                raise ValueError("Invalid source cuboid geometry/identity for noise rejection")
            self._cuboid_constraint_cache = cached = (track_ids, centers, rotations, dimensions)
        track_ids, centers, rotations, dimensions = cached
        # Resolve live IDs each step: topology/reordering can change row offsets.
        rows = torch.searchsorted(track_ids, self.barrier_object_id)
        rotation = rotations[rows]
        local = ((self.positions - centers[rows]).unsqueeze(-1) * rotation).sum(-2)
        delta_local = ((proposal - self.positions).unsqueeze(-1) * rotation).sum(-2)
        half = dimensions[rows] * (0.5 * self.config.cuboid_perturbation_scale)
        current = local.abs() - half
        proposed = (local + delta_local).abs() - half
        blocked = ((current < 0) & (proposed >= 0)) | ((current >= 0) & (proposed > current))
        filtered_local = torch.where(blocked, 0, delta_local)
        filtered_world = (rotation * filtered_local.unsqueeze(-2)).sum(-1)
        self.positions.copy_(torch.where(blocked.any(-1)[:, None], self.positions + filtered_world, proposal))


BaseGaussianModel.register_to_gaussians_factory("construction-barriers-gaussians", ConstructionBarriersGaussianModel)
