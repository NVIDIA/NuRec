# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in backend for the public NCore dataset format."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from PIL import Image
from upath import UPath

import ncore.data
import ncore.data.v4

from nurec.lidar.firing_patterns.schema import (
    LidarFiringPatternTemplatesData,
)
from nurec.ncore_backends.interface import (
    DatasetMetadata,
    DatasetRequest,
    JsonObject,
    MetadataValue,
    SequencePartInfo,
    UnsupportedBackendOperationError,
)
from nurec.utils.ncore_types import require_concrete_lidar_parameters


FiringPatternLoader = Callable[[DatasetRequest, str], LidarFiringPatternTemplatesData | None]


class NCoreV4Backend:
    """Access standard V4 datasets and optionally adapt native firing-pattern components.

    Args:
        firing_pattern_loader: Optional callable receiving a
            dataset request and sensor ID and returning validated templates.
            ``None`` leaves native firing-pattern discovery unconfigured.
    """

    name = "ncore-v4"

    def __init__(self, firing_pattern_loader: FiringPatternLoader | None = None) -> None:
        """Configure the optional component loader without opening a dataset.

        Args:
            firing_pattern_loader: Callable reading templates for a dataset request and sensor ID,
                or ``None`` to register it later.

        Returns:
            ``None``; stores the loader for explicit template requests.
        """
        self._firing_pattern_loader = firing_pattern_loader

    def register_firing_pattern_loader(self, loader: FiringPatternLoader) -> None:
        """Attach the optional firing-pattern component loader.

        Args:
            loader: Callable receiving a dataset request and sensor ID. It runs only
                when templates are requested, never during normal sequence loading.

        Returns:
            ``None``; retains this callable for explicit template requests. Re-registering the same
            callable is allowed.

        Raises:
            ValueError: If a different pattern loader has already been registered.
        """
        if self._firing_pattern_loader is not None and self._firing_pattern_loader is not loader:
            raise ValueError("A firing-pattern loader is already registered")
        self._firing_pattern_loader = loader

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
        """Open standard V4 component stores without discovering firing patterns.

        Args:
            request: Dataset request controlling consolidated reads and component-group
                names for poses, intrinsics, masks, and cuboids.
            dataset_paths: Component-store locations resolved from the dataset manifest.

        Returns:
            The standard V4 loader. Firing-pattern components are not opened here.

        Raises:
            OSError: If component stores cannot be opened or read.
            ValueError: If the standard V4 component configuration rejects the dataset.
        """
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

    def get_lidar_firing_pattern_templates(
        self,
        request: DatasetRequest,
        sensor_id: str,
    ) -> LidarFiringPatternTemplatesData | None:
        """Read one sensor's optional component without constructing a lidar sensor.

        Args:
            request: Resolved V4 dataset request identifying the sequence manifest.
            sensor_id: Native component instance name, equal to the logical lidar ID.

        Returns:
            Validated templates, or ``None`` when no component loader is configured,
            the component is absent, or observations were insufficient for a cycle.

        Raises:
            UnsupportedBackendOperationError: If the component version is unsupported.
            ValueError: If the component is invalid or explicitly not repeatable.
            OSError: If the manifest or component store cannot be read.
        """
        if self._firing_pattern_loader is None:
            return None
        return self._firing_pattern_loader(request, sensor_id)


NCORE_V4_BACKEND = NCoreV4Backend()
