# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in backend for the public NCore dataset format."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from PIL import Image
from upath import UPath

import ncore.data
import ncore.data.v4

from nurec.ncore_backends.interface import (
    DatasetMetadata,
    DatasetRequest,
    JsonObject,
    MetadataValue,
    SequencePartInfo,
    UnsupportedBackendOperationError,
)
from nurec.utils.ncore_types import require_concrete_lidar_parameters


class NCoreV4Backend:
    name = "ncore-v4"

    def recognizes(self, manifest: JsonObject) -> bool:
        version = manifest.get("version")
        return isinstance(version, str) and version.startswith("v4")

    def resolve_metadata(self, request: DatasetRequest, manifest: JsonObject) -> DatasetMetadata:
        try:
            timestamp_interval = manifest["sequence_timestamp_interval_us"]
            component_stores = manifest["component_stores"]
            sequence_id = manifest["sequence_id"]
            if not isinstance(timestamp_interval, Mapping) or not isinstance(component_stores, list):
                raise TypeError
            start = timestamp_interval["start"]
            stop = timestamp_interval["stop"]
            if (
                not isinstance(sequence_id, str)
                or isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(stop, bool)
                or not isinstance(stop, int)
            ):
                raise TypeError

            dataset_paths: list[UPath] = []
            for store in component_stores:
                if not isinstance(store, Mapping):
                    raise TypeError
                path = store["path"]
                if not isinstance(path, str):
                    raise TypeError
                dataset_paths.append(request.metadata_path.parent / path)

            return DatasetMetadata(
                sequence_id=sequence_id,
                time_range_start_us=start,
                time_range_end_us=stop,
                dataset_paths=tuple(dataset_paths),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid NCore dataset metadata in {request.metadata_path}") from exc

    def create_sequence_loader(
        self,
        request: DatasetRequest,
        dataset_paths: Sequence[UPath],
    ) -> ncore.data.SequenceLoaderProtocol:
        return ncore.data.v4.SequenceLoaderV4(
            ncore.data.v4.SequenceComponentGroupsReader(
                list(dataset_paths),
                open_consolidated=request.open_consolidated,
            ),
            poses_component_group_name=request.poses_component_group,
            intrinsics_component_group_name=request.intrinsics_component_group,
            masks_component_group_name=request.masks_component_group,
            cuboids_component_group_name=request.cuboids_component_group,
        )

    def get_sequence_part_info(self, sequence_loader: ncore.data.SequenceLoaderProtocol) -> SequencePartInfo:
        if not isinstance(sequence_loader, ncore.data.v4.SequenceLoaderV4):
            raise ValueError(f"Unsupported NCore sequence loader type: {type(sequence_loader)}")
        return SequencePartInfo(base_name=sequence_loader.sequence_id, part_index=0, part_count=1)

    def get_camera_mask_and_resolution(
        self,
        camera_sensor: ncore.data.CameraSensorProtocol,
    ) -> tuple[Image.Image | None, np.ndarray]:
        if not isinstance(camera_sensor, ncore.data.CameraSensorProtocol):
            raise ValueError(f"Unsupported NCore camera sensor type: {type(camera_sensor)}")
        return camera_sensor.get_mask_images().get("ego"), camera_sensor.model_parameters.resolution

    def get_lidar_dynamic_flag(
        self,
        lidar_sensor: ncore.data.LidarSensorProtocol,
        frame_idx: int,
    ) -> np.ndarray | None:
        if not isinstance(lidar_sensor, ncore.data.LidarSensorProtocol):
            raise ValueError(f"Unsupported NCore lidar sensor type: {type(lidar_sensor)}")
        return None

    def get_lidar_model_parameters(
        self,
        lidar_sensor: ncore.data.LidarSensorProtocol,
    ) -> ncore.data.ConcreteLidarModelParametersUnion | None:
        if not isinstance(lidar_sensor, ncore.data.LidarSensorProtocol):
            raise ValueError(f"Unsupported NCore lidar sensor type: {type(lidar_sensor)}")
        parameters = lidar_sensor.model_parameters
        return require_concrete_lidar_parameters(parameters)

    def get_lidar_model_metadata(
        self,
        lidar_sensor: ncore.data.LidarSensorProtocol,
    ) -> dict[str, MetadataValue]:
        raise UnsupportedBackendOperationError("This backend does not provide legacy lidar model metadata")


NCORE_V4_BACKEND = NCoreV4Backend()
