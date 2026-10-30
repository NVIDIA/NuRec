# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shard Data Manager utility for managing shard loading and 3D label extraction."""

import logging

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from scipy.spatial.transform import Rotation
from upath import UPath

import ncore.data
import ncore.impl.common.transformations as ncore_transformations
import nurec.utils.ncore_utils as ncore_utils

from ncore.sensors import CameraModel, camera_model_from_parameters
from nurec.ncore_backends import DatasetRequest, require_explicit_sequence_input_provider, resolve_dataset
from nurec.utils.types import HalfClosedInterval


logger = logging.getLogger(__name__)


class ShardDataManager:
    """Manages dataset loading, sensor initialization, and 3D label extraction.

    Datasets are opened through the format-agnostic ``ncore.data.SequenceLoaderProtocol``.
    Metadata manifests are resolved by the registered dataset backends, while explicit
    path patterns use an optional provider registered by the composing application.

    Usage:
        # Load a dataset metadata manifest
        manager = ShardDataManager('clipgt-abc.json', 'camera_front')

        # Switch datasets (reuse same instance)
        manager.load('clipgt-def.json')

        # A composing application may also enable explicit sequence paths.
        manager.load('sequence/*')
    """

    def __init__(self, shard_pattern: str, camera_id: str):
        """Initialize shard data manager and load the dataset immediately.

        Args:
            shard_pattern: Explicit sequence-path pattern or path to an NCore dataset
                metadata `.json` file.
            camera_id: ID of the camera to use.
        """
        self.camera_id = camera_id
        self.shard_pattern = shard_pattern
        self._load(shard_pattern)

    def load(self, shard_pattern: str) -> None:
        """Load or reload a dataset.

        Use this to switch to a different dataset without recreating
        the manager instance. Useful for batch processing.

        Args:
            shard_pattern: Explicit sequence-path pattern or path to an NCore dataset
                metadata `.json` file.
        """
        self._load(shard_pattern)
        self.shard_pattern = shard_pattern

    def _load(self, shard_pattern: str) -> None:
        """Internal method to load the dataset.

        Instance state is only committed once loading and validation fully succeed, so a
        failed reload cannot leave the manager with mixed old/new state.

        Args:
            shard_pattern: Explicit sequence-path pattern or path to an NCore dataset
                metadata `.json` file.
        """
        logger.info("Loading dataset from %s...", shard_pattern)

        clip_interval_us = None
        if shard_pattern.endswith(".json"):
            resolved = resolve_dataset(DatasetRequest(metadata_path=UPath(shard_pattern)))
            clip_interval_us = HalfClosedInterval(
                resolved.metadata.time_range_start_us,
                resolved.metadata.time_range_end_us,
            )
            loader = resolved.create_sequence_loader()
        else:
            _, loader, dataset_paths = require_explicit_sequence_input_provider().open_group(
                shard_pattern,
                open_consolidated=True,
                loader_max_workers=None,
            )
            logger.info("  Found %d dataset path(s)", len(dataset_paths))

        camera_ids = list(loader.camera_ids)
        logger.info("  Available cameras: %s", camera_ids)

        if self.camera_id not in camera_ids:
            raise ValueError(f"Camera {self.camera_id} not found in shard")

        camera_sensor: ncore.data.CameraSensorProtocol = loader.get_camera_sensor(self.camera_id)

        # Cuboid labels come from sequence-level track observations (for V3 data the compat
        # loader derives one observation group per lidar frame; lidar-free V4 exports need
        # no lidar sensor at all). Only labeled frames appear in the timeline.
        # Bound label loading to the clip subsection the meta file describes (when present),
        # so metas referencing a sub-range of longer stores do not pull the full sequence.
        observation_interval_us = (
            ncore_transformations.HalfClosedInterval(start=clip_interval_us.start, stop=clip_interval_us.end)
            if clip_interval_us is not None
            else None
        )
        # Grouped by reference_frame_timestamp_us, the time each box pose is expressed at. Exports
        # may set it per frame (VGGT, V3) or per observation (PAI); camera frames are matched to
        # labels per track downstream.
        observations_by_ts: Dict[int, List[ncore.data.CuboidTrackObservation]] = defaultdict(list)
        for observation in loader.get_cuboid_track_observations(timestamp_interval_us=observation_interval_us):
            observations_by_ts[int(observation.reference_frame_timestamp_us)].append(observation)
        if not observations_by_ts:
            # An empty scene is a legitimate clip: keep an empty timeline so downstream
            # produces empty object metrics (previous behavior) instead of no output.
            logger.warning("No cuboid track observations found in %s; object metrics will be empty", shard_pattern)
        observations_by_timestamp_us = dict(sorted(observations_by_ts.items()))
        label_timestamps_us = np.asarray(list(observations_by_timestamp_us.keys()), dtype=np.int64)

        # Bound GT camera frames to the same clip subsection the labels were bounded to:
        # without rendered timestamps, downstream index-based matching would otherwise start
        # at camera frame 0 of the full store and read images outside the sub-clip.
        camera_end_timestamps_us = np.asarray(camera_sensor.get_frames_timestamps_us(), dtype=np.int64)
        if clip_interval_us is not None:
            in_clip = np.nonzero(
                (camera_end_timestamps_us >= int(clip_interval_us.start))
                & (camera_end_timestamps_us < int(clip_interval_us.end))
            )[0]
            if in_clip.size == 0:
                raise ValueError(
                    f"No frames of camera {self.camera_id} fall inside the clip interval "
                    f"[{clip_interval_us.start}, {clip_interval_us.end}) declared by {shard_pattern}"
                )
            camera_frame_index_range = (int(in_clip[0]), int(in_clip[-1]) + 1)
        else:
            camera_frame_index_range = (0, int(camera_end_timestamps_us.shape[0]))

        camera_model: CameraModel = camera_model_from_parameters(camera_sensor.model_parameters, device="cpu")

        # Commit state only after all loading/validation succeeded, so a failed reload
        # cannot leave the manager with mixed old/new state.
        self._sequence_loader = loader
        self.camera_sensor: ncore.data.CameraSensorProtocol = camera_sensor
        self._observations_by_timestamp_us = observations_by_timestamp_us
        self.label_timestamps_us: np.ndarray = label_timestamps_us
        self.camera_frame_index_range: Tuple[int, int] = camera_frame_index_range
        self.camera_model = camera_model

        logger.info(
            "  Successfully loaded dataset (%d cameras, %d label frames)",
            len(camera_ids),
            label_timestamps_us.size,
        )

    @staticmethod
    def box2corners(bbox3) -> np.ndarray:
        """Convert 3D bounding box to 8 corner points.

        Args:
            bbox3: 3D bounding box with center, dimensions, and rotation.

        Returns:
            Array of 8 corner points [8, 3].
        """
        center = bbox3.to_array()[:3]
        dim = bbox3.to_array()[3:6]
        rot = bbox3.to_array()[6:]

        half_dim = dim / 2
        local_corners = np.array(
            [
                [half_dim[0], half_dim[1], half_dim[2]],
                [-half_dim[0], half_dim[1], half_dim[2]],
                [-half_dim[0], -half_dim[1], half_dim[2]],
                [half_dim[0], -half_dim[1], half_dim[2]],
                [half_dim[0], -half_dim[1], -half_dim[2]],
                [-half_dim[0], -half_dim[1], -half_dim[2]],
                [-half_dim[0], half_dim[1], -half_dim[2]],
                [half_dim[0], half_dim[1], -half_dim[2]],
            ]
        )

        rotation_matrix = Rotation.from_euler("XYZ", rot).as_matrix()
        corners = np.array([rotation_matrix @ corner + center for corner in local_corners])
        return corners

    def compile_3d_labels(self) -> Dict[int, Dict[str, dict]]:
        """Compile 3D bounding box labels from all label frames.

        Cuboid observations are grouped by their reference-frame timestamp (for V3 data
        this reproduces the per-lidar-frame grouping), transformed to world coordinates,
        and keyed by position in ``self.label_timestamps_us``.

        Returns:
            Dictionary mapping frame_idx to track data:
            {frame_idx: {track_id: {"vertices": [8,3],
            "class": str, "bbox3": ...}}}
        """
        logger.info("Compiling labels for all frames...")
        box_dict = {}
        pose_graph = self._sequence_loader.pose_graph

        for frame_idx, (timestamp_us, observations) in enumerate(self._observations_by_timestamp_us.items()):
            observations_world = ncore_utils.transform_cuboid_track_observations(
                observations=observations,
                pose_graph=pose_graph,
                target_frame_id="world",
                target_frame_timestamp_us=timestamp_us,
                tqdm_disabled=True,
            )

            frame_boxes = {}
            for observation in observations_world:
                frame_boxes[str(observation.track_id)] = {
                    "vertices": self.box2corners(observation.bbox3),
                    "class": str(observation.class_id),
                    "bbox3": observation.bbox3,
                }

            box_dict[frame_idx] = frame_boxes

        logger.info("Compiled labels for %d frames", len(box_dict))
        return box_dict
