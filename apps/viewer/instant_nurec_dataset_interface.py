# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from typing import Any, Literal, Optional

import numpy as np
import torch

from apps.viewer.dataset_interface import (
    CameraTrajectoryData,
    CameraTrajectoryId,
    ViewerDatasetInterface,
)
from apps.viewer.usdz_viewer import BaseUSDZLikeDatasetInterface
from nurec.datasets.tracks import CuboidTracks
from nurec.instant_nurec.primitives.base import BaseInstantNuRecPrimitive
from nurec.instant_nurec.primitives.celsius_primitive import CelsiusInstantNuRecPrimitive
from nurec.libs.losses.orchestration.config import LossAggregatorBatchReturn
from nurec.utils.batch import InstantNuRecDataBatch, RenderingBatch
from nurec.utils.misc import unpack_optional
from nurec.utils.types import FrameConversion, HalfClosedInterval, PointCloud


log = logging.getLogger(__name__)


class EmptyDatasetInterface(ViewerDatasetInterface):
    """
    A dummy dataset interface to be used when no datasource is available.
    (e.g. during inference of the an instant_nurec model)
    """

    def get_trajectory_ids(self) -> list[CameraTrajectoryId]:
        return []

    def get_trajectory_data(self, trajectory_id: CameraTrajectoryId) -> CameraTrajectoryData:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support getting trajectory data for {trajectory_id}"
        )

    def distance_nurec_to_distance_world(self, distance: torch.Tensor) -> torch.Tensor:
        return distance

    def get_point_cloud(
        self, color_type: Optional[Literal["camera-rgb", "semantics"]] = None, step_frame: int = 10
    ) -> PointCloud | None:
        return None

    @property
    def is_empty(self) -> bool:
        """
        Return if this is a dummy empty interface. This is typicall true in e.g. instant_nurec when
        the dataset is not yet available (the model hasn't finished its inference).
        """
        return True

    def get_cuboid_tracks(self) -> Optional[CuboidTracks]:
        return None

    @property
    def world_to_nurec(self) -> FrameConversion:
        return FrameConversion(matrix=np.eye(4, dtype=np.float32))

    def supports_get_closest_frame_image(self) -> bool:
        return False


class ViewerInstantNuRecInterface(BaseUSDZLikeDatasetInterface):
    """
    Dataset-viewer compatibility layer that treats a batch as interface.
    This will use the **context** rig trajectory and the corresponding cuboid tracks.
    """

    rendering_batch: RenderingBatch

    def __init__(self, data_batch: InstantNuRecDataBatch, batch_idx: int = 0):
        rig_trajectories = unpack_optional(data_batch.context_rig)[batch_idx]
        if data_batch.cuboid_tracks is not None:
            cuboid_tracks_data_pack = unpack_optional(data_batch.cuboid_tracks)[batch_idx]
            cuboid_tracks = CuboidTracks.Factory.from_pack(cuboid_tracks_data_pack)
        else:
            cuboid_tracks = CuboidTracks.Factory.empty()
        super().__init__(rig_trajectories=rig_trajectories, cuboid_tracks=cuboid_tracks)

        self.data_batch = data_batch.context[batch_idx].data
        self.rendering_batch = unpack_optional(data_batch.context[batch_idx].rendering)

    def get_point_cloud(
        self, color_type: Optional[Literal["camera-rgb", "semantics"]] = None, step_frame: int = 10
    ) -> PointCloud | None:
        # Does not support visualizing point cloud for now.
        return None

    def supports_get_closest_frame_image(self) -> bool:
        return True

    def _get_closest_frame_and_timestamp(
        self, trajectory_id: CameraTrajectoryId, timestamp_us: int
    ) -> tuple[int, torch.Tensor]:
        """
        Find the closest frame to the given timestamp for the specified camera trajectory.

        Args:
            trajectory_id: Camera trajectory identifier
            timestamp_us: Query timestamp in microseconds

        Returns:
            tuple[int, torch.Tensor]: (frame_index, frame_rgb) where frame_rgb has shape (H, W, 3)
        """
        if self.data_batch.camera is None or self.rendering_batch.camera is None:
            raise ValueError("No camera data available for this batch")

        # Get frame timestamps and find the closest frame
        frame_end_timestamps_us = self.rendering_batch.camera.timestamps_startend_us[:, 1]
        diffs = torch.abs(frame_end_timestamps_us - timestamp_us)
        frame_idx = int(torch.argmin(diffs).item())

        # Get the image data for this frame
        height, width = self.rendering_batch.camera.h, self.rendering_batch.camera.w

        # Get the RGB data
        rgb_data = self.data_batch.camera.labels.rgb
        if rgb_data is None:
            raise ValueError("No RGB data available for this batch")

        frame_image = rgb_data[frame_idx].reshape(height, width, 3)

        return frame_idx, frame_image

    def get_trajectory_data(self, trajectory_id: CameraTrajectoryId) -> CameraTrajectoryData:
        """
        Get trajectory data for a specific trajectory.
        Different from the base class, this method will return the trajectory data bounded by the data frame timestamps.
        Args:
            trajectory_id: Camera trajectory identifier

        Returns:
            CameraTrajectoryData: The trajectory data
        """
        frame_end_timestamps_us = unpack_optional(self.rendering_batch.camera).timestamps_startend_us
        return self._get_trajectory_data_bounded(
            trajectory_id,
            HalfClosedInterval(int(frame_end_timestamps_us.min()), int(frame_end_timestamps_us.max())),
        )


def postprocess_instant_nurec_batch_and_primitive(
    batch: InstantNuRecDataBatch, instant_nurec_primitive: BaseInstantNuRecPrimitive
) -> tuple[InstantNuRecDataBatch, BaseInstantNuRecPrimitive]:
    """
    Postprocess instant_nurec batch and primitive for visualization.
    For compatible models, Enable sky mask for visualization by default.
    """
    if isinstance(instant_nurec_primitive, CelsiusInstantNuRecPrimitive):
        instant_nurec_primitive.set_sky_mask_enabled(True)
    return batch, instant_nurec_primitive


def get_instant_nurec_primitive_from_loss_return(
    loss_return: Any, batch_idx: int = 0
) -> BaseInstantNuRecPrimitive | None:
    """
    Extract instant_nurec primitive from loss return for visualization.
    """
    if not isinstance(loss_return, LossAggregatorBatchReturn):
        log.warning(f"Skipping viewer update: {type(loss_return)=} is not a LossAggregatorBatchReturn.")
        return None
    if (primitive_list := loss_return.extra_fields.get("primitives")) is None:
        log.warning(f"Skipping viewer update: {loss_return.extra_fields=} has no field named 'primitives'.")
        return None
    assert isinstance(primitive_list, list)
    primitive = primitive_list[batch_idx]
    if not isinstance(primitive, BaseInstantNuRecPrimitive):
        log.warning(f"Skipping viewer update: {type(primitive)=} is not a renderable instant_nurec primitive.")
        return None
    return primitive
