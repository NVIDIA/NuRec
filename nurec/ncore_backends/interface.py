# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contract implemented by NCore dataset backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np

from PIL import Image
from upath import UPath

import ncore.data


MetadataValue = bool | float | int | str | list[float]
JsonObject = Mapping[str, object]


class UnsupportedBackendOperationError(ValueError):
    """Raised when a dataset backend cannot provide an optional operation."""


class DatasetRequest(NamedTuple):
    """Common inputs used to resolve and open an NCore dataset."""

    metadata_path: UPath
    open_consolidated: bool = True
    loader_max_workers: int | None = None
    poses_component_group: str = "default"
    intrinsics_component_group: str = "default"
    masks_component_group: str | None = "default"
    cuboids_component_group: str | None = "default"


@dataclass(frozen=True)
class DatasetMetadata:
    """Format-neutral dataset metadata needed by shared NuRec code."""

    sequence_id: str
    time_range_start_us: int
    time_range_end_us: int
    dataset_paths: tuple[UPath, ...]


@dataclass(frozen=True)
class SequencePartInfo:
    """Identity of one independently processed part of a sequence."""

    base_name: str
    part_index: int
    part_count: int


@dataclass(frozen=True)
class ResolvedDataset:
    """A dataset request matched to the backend that understands it."""

    backend: DatasetBackend
    request: DatasetRequest
    metadata: DatasetMetadata

    def create_sequence_loader(self) -> ncore.data.SequenceLoaderProtocol:
        return self.backend.create_sequence_loader(self.request, self.metadata.dataset_paths)


class DatasetBackend(Protocol):
    """Operations a registered NCore dataset implementation may provide."""

    name: str

    def recognizes(self, manifest: JsonObject) -> bool: ...

    def resolve_metadata(
        self,
        request: DatasetRequest,
        manifest: JsonObject,
    ) -> DatasetMetadata: ...

    def create_sequence_loader(
        self,
        request: DatasetRequest,
        dataset_paths: Sequence[UPath],
    ) -> ncore.data.SequenceLoaderProtocol: ...

    def get_sequence_part_info(self, sequence_loader: ncore.data.SequenceLoaderProtocol) -> SequencePartInfo: ...

    def get_camera_mask_and_resolution(
        self,
        camera_sensor: ncore.data.CameraSensorProtocol,
    ) -> tuple[Image.Image | None, np.ndarray]: ...

    def get_lidar_dynamic_flag(
        self,
        lidar_sensor: ncore.data.LidarSensorProtocol,
        frame_idx: int,
    ) -> np.ndarray | None: ...

    def get_lidar_model_parameters(
        self,
        lidar_sensor: ncore.data.LidarSensorProtocol,
    ) -> ncore.data.ConcreteLidarModelParametersUnion | None: ...

    def get_lidar_model_metadata(
        self,
        lidar_sensor: ncore.data.LidarSensorProtocol,
    ) -> dict[str, MetadataValue]: ...


class ShardDataLoaderProtocol(Protocol):
    """Minimal loader interface used to locate associated auxiliary stores."""

    def get_sequence_id(self) -> str: ...

    def get_shard_paths(self) -> Sequence[Path | UPath]: ...
