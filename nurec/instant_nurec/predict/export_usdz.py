# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from collections import OrderedDict
from typing import Any, cast

import numpy as np
import torch

from pytorch_lightning.utilities import move_data_to_device
from upath import UPath

import ncore.data
import ncore.impl.common.transformations as ncore_transformations

from apps.av.road_mesh.ground_mesh import DelaunayElevationMeshingAlgorithm
from ncore.data import BBox3
from nurec.config.checkpoint import ArtifactConfig
from nurec.datasets.tracks import CuboidTracks
from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig
from nurec.instant_nurec.primitives.base import BaseInstantNuRecPrimitive
from nurec.instant_nurec.utils.trajectory import transform_rig_trajectories
from nurec.ncore_backends import DatasetRequest, resolve_dataset
from nurec.utils.batch import RigTrajectories
from nurec.utils.io.artifact_cache import ArtifactCache
from nurec.utils.io.checkpoint import reduce_precision_to_fp16, serialize_checkpoint, strip_optimizer_state
from nurec.utils.io.mesh import Mesh, serialize_mesh
from nurec.utils.io.metadata import get_metadata, serialize_metadata
from nurec.utils.io.rig_trajectories import rig_trajectories_time_range, serialize_rig_trajectories
from nurec.utils.io.sequence_tracks import serialize_sequence_tracks
from nurec.utils.misc import to_torch, unpack_optional
from nurec.utils.ncore_types import require_concrete_camera_parameters, require_concrete_lidar_parameters
from nurec.utils.ncore_utils import NVRigMetadata, get_nv_rig_vehicle_bbox
from nurec.utils.types import FrameConversion, HalfClosedInterval, NamedSerialized


logger = logging.getLogger(__name__)


def load_datasource_rig_trajectories(
    dataset_request: DatasetRequest,
    reference_rig_trajectories: RigTrajectories,
    cached_sequence_loader: ncore.data.SequenceLoaderProtocol | None = None,
) -> RigTrajectories:
    """
    Load the rig trajectories from the NCORE data source, and use the reference mainly to
    map the camera unique indices and assign proper unique indices to the new rig trajectories.
    """
    # Create the loader if not provided
    resolved_dataset = resolve_dataset(dataset_request)
    sequence_id = resolved_dataset.metadata.sequence_id
    time_range_us = HalfClosedInterval(
        resolved_dataset.metadata.time_range_start_us,
        resolved_dataset.metadata.time_range_end_us,
    )
    if cached_sequence_loader is None:
        logger.warning(
            "Re-creating sequence loader during USDZ export stage without caching. This could be slow for S3-based datasets."
        )
        sequence_loader = resolved_dataset.create_sequence_loader()
    else:
        sequence_loader = cached_sequence_loader

    # Rig bbox is important for AlpaSim.
    rig_bbox: BBox3 | None = None
    if nv_rig := sequence_loader.generic_meta_data.get("nv-rig", None):
        # parse a NV rig file for it's body-associated bbox
        rig_bbox = BBox3.from_array(get_nv_rig_vehicle_bbox(cast(NVRigMetadata, nv_rig)))
    elif vehicle_bbox := cast(dict | None, sequence_loader.generic_meta_data.get("vehicle-bbox", None)):
        # load bbox from vehicle-bbox field in meta data (available for some datasets, e.g., PAI)
        rig_bbox = BBox3(
            centroid=tuple(vehicle_bbox["centroid"]), dim=tuple(vehicle_bbox["dim"]), rot=tuple(vehicle_bbox["rot"])
        )

    # Get rig_world poses
    rig_world_edge: ncore_transformations.PoseGraphInterpolator.Edge = unpack_optional(
        sequence_loader.pose_graph.get_edge("rig", "world"),
    )
    T_rig_world: np.ndarray = rig_world_edge.T_source_target
    T_rig_world_timestamps_us: np.ndarray = unpack_optional(rig_world_edge.timestamps_us)
    T_rig_world = T_rig_world[poses_range := time_range_us.cover_range(T_rig_world_timestamps_us)]
    T_rig_world_timestamps_us = T_rig_world_timestamps_us[poses_range]
    time_range_us = time_range_us.restricted(T_rig_world_timestamps_us)

    # NuRec uses the @ convention and we follow it here.
    def to_unique_id(sensor_id: str) -> str:
        return f"{sensor_id}@{sequence_id}"

    # Export camera calibration
    camera_calibrations: list[tuple[str, RigTrajectories.CameraCalibration]] = []
    for camera_id in sequence_loader.camera_ids:
        if (ref_cam_calib := reference_rig_trajectories.camera_calibrations.get(camera_id, None)) is not None:
            unique_sensor_idx = ref_cam_calib.unique_sensor_idx
        else:
            logger.warning(
                f"Camera {camera_id} not found in instant_nurec-produced rig trajectories, unique sensor index fallback to 0."
            )
            unique_sensor_idx = 0
        camera_sensor = sequence_loader.get_camera_sensor(camera_id)
        camera_calibrations.append(
            (
                to_unique_id(camera_id),
                RigTrajectories.CameraCalibration(
                    sequence_id=sequence_id,
                    logical_sensor_name=camera_id,
                    unique_sensor_idx=unique_sensor_idx,
                    T_sensor_rig=to_torch(unpack_optional(camera_sensor.T_sensor_rig), device="cpu"),
                    camera_model_parameters=require_concrete_camera_parameters(camera_sensor.model_parameters),
                ),
            )
        )

    # Export lidar calibration
    lidar_calibrations: list[tuple[str, RigTrajectories.LidarCalibration]] = []
    for lidar_id in sequence_loader.lidar_ids:
        if (ref_lidar_calib := reference_rig_trajectories.lidar_calibrations.get(lidar_id, None)) is not None:
            unique_sensor_idx = ref_lidar_calib.unique_sensor_idx
        else:
            logger.warning(
                f"Lidar {lidar_id} not found in instant_nurec-produced rig trajectories, unique sensor index fallback to 0."
            )
            unique_sensor_idx = 0
        lidar_sensor = sequence_loader.get_lidar_sensor(lidar_id)
        lidar_calibrations.append(
            (
                to_unique_id(lidar_id),
                RigTrajectories.LidarCalibration(
                    sequence_id=sequence_id,
                    logical_sensor_name=lidar_id,
                    unique_sensor_idx=unique_sensor_idx,
                    T_sensor_rig=to_torch(unpack_optional(lidar_sensor.T_sensor_rig), device="cpu"),
                    lidar_model_parameters=require_concrete_lidar_parameters(lidar_sensor.model_parameters),
                ),
            )
        )

    # Get the NuRec world_base
    world_world_global_edge: ncore_transformations.PoseGraphInterpolator.Edge = unpack_optional(
        sequence_loader.pose_graph.get_edge("world", "world_global"),
    )

    def get_sensor_frame_timestamps_us(sensor_id: str) -> torch.Tensor:
        sensor = (
            sequence_loader.get_camera_sensor(sensor_id)
            if sensor_id in sequence_loader.camera_ids
            else sequence_loader.get_lidar_sensor(sensor_id)
        )
        cover_range = time_range_us.cover_range(sensor.frames_timestamps_us[:, 1])
        while len(cover_range) and (int(sensor.frames_timestamps_us[cover_range.start, 0]) not in time_range_us):
            cover_range = cover_range[1:]
        return to_torch(sensor.frames_timestamps_us[cover_range], device="cpu", dtype=torch.int64)

    new_rig_trajectories = RigTrajectories(
        T_world_base=to_torch(world_world_global_edge.T_source_target, device="cpu", dtype=torch.float64),
        world_to_nurec=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
        rig_trajectories=[
            RigTrajectories.RigTrajectory(
                sequence_id=sequence_id,
                rig_bbox=rig_bbox,
                T_rig_worlds=to_torch(T_rig_world, device="cpu", dtype=torch.float64),
                T_rig_world_timestamps_us=to_torch(T_rig_world_timestamps_us, device="cpu", dtype=torch.int64),
                cameras_frame_timestamps_us={
                    to_unique_id(camera_id): get_sensor_frame_timestamps_us(camera_id)
                    for camera_id in sequence_loader.camera_ids
                },
                lidars_frame_timestamps_us={
                    to_unique_id(lidar_id): get_sensor_frame_timestamps_us(lidar_id)
                    for lidar_id in sequence_loader.lidar_ids
                },
            )
        ],
        camera_calibrations=OrderedDict(camera_calibrations),
        lidar_calibrations=OrderedDict(lidar_calibrations),
    )

    return new_rig_trajectories


def build_artifact_cache(
    artifact_config: ArtifactConfig,
    primitive: BaseInstantNuRecPrimitive,
    rig_trajectories: RigTrajectories,
    instant_nurec_config: InstantNuRecConfig,
    meta_data: dict[str, Any],
) -> ArtifactCache:
    """Export the USDZ artifact for the given primitive and batch"""
    artifact_cache = ArtifactCache(artifact_config=artifact_config)

    assert "ncore_json_path" in meta_data, f"ncore_json_path key must be provided, only got {meta_data.keys()}"
    ncore_json_path = cast(UPath, meta_data["ncore_json_path"])

    # External tools typically ignore T_world_base, which is considered as relative transform to the raw rig in instant_nurec.
    # Hence we bake it into the rig_trajectories.
    # After this, rig_trajectories.T_world_base is guaranteed to be identity.
    primitive = primitive.rigid_transform(rig_trajectories.T_world_base.to(primitive.device()))
    rig_trajectories = transform_rig_trajectories(rig_trajectories, left_transform=rig_trajectories.T_world_base)

    # Always populate metadata
    # Although we assign scene_id here, rig_trajectories.sequence_id could still be "context-main",
    # one need to set match_datasource to make sure proper sequence_id is set if required.
    metadata = get_metadata(instant_nurec_config, dataset_path=ncore_json_path)
    metadata.scene_id = meta_data["sequence_id"]
    artifact_cache.metadata = serialize_metadata(metadata)

    # Export empty cuboid tracks for now
    if artifact_config.sequence_tracks.enabled:
        artifact_cache.sequence_tracks = serialize_sequence_tracks(
            sequence_id="dummy_chunk_id",
            cuboid_tracks=CuboidTracks.Factory.empty(device=torch.device("cpu")),
            excluded_tracks=None,
            formats=["json"],  # Add usdz format later
        )

    # Add checkpoint
    if artifact_config.checkpoint.enabled:
        checkpoint = primitive.get_checkpoint()
        # Most checkpoints can be loaded in place, but if the dimensions of the buffers are not know in advance, they need to be assigned to the uninitialized tesnors instead.
        checkpoint["load_in_place"] = True
        if artifact_config.checkpoint.strip_optimizer:
            checkpoint = strip_optimizer_state(checkpoint)
        if artifact_config.checkpoint.fp16:
            checkpoint = reduce_precision_to_fp16(checkpoint)
        artifact_cache.checkpoint = serialize_checkpoint(checkpoint)

    # Add config
    if artifact_config.parsed_config.enabled:
        artifact_cache.parsed_config = NamedSerialized.from_config(instant_nurec_config.to_dictconfig())

    # Rig trajectories (make sure we're on CPU first)
    rig_trajectories = move_data_to_device(rig_trajectories, device="cpu")
    if artifact_config.rig_trajectories.enabled:
        if artifact_config.rig_trajectories.match_datasource:
            cached_sequence_loader = meta_data.get("sequence_loader", None)
            dataset_request = meta_data.get("dataset_request")
            if not isinstance(dataset_request, DatasetRequest):
                raise ValueError("dataset_request metadata is required to match datasource rig trajectories")
            rig_trajectories = load_datasource_rig_trajectories(
                dataset_request,
                rig_trajectories,
                cached_sequence_loader=cast(ncore.data.SequenceLoaderProtocol | None, cached_sequence_loader),
            )
        # Need to export both start and end frame timestamps to enable rendering input views with rolling shutter,
        # otherwise there will be a time sync misalignment between rendered and input views.
        artifact_cache.rig_trajectories = serialize_rig_trajectories(
            rig_trajectories,
            usd_timestamp_offset=rig_trajectories_time_range(rig_trajectories).start,
            add_default_cameras=artifact_config.rig_trajectories.add_default_cameras,
        )

    # Simple ground mesh on X-Y plane for now
    if artifact_config.mesh.ground.enabled:
        num_points, ground_scale = 10000, 1000.0
        ground_points = np.concatenate(
            [
                np.random.uniform(-ground_scale / 2.0, ground_scale / 2.0, (num_points, 2)).astype(np.float32),
                np.zeros((num_points, 1), dtype=np.float32),
            ],
            axis=1,
        )
        _, triangles, vertices, _ = DelaunayElevationMeshingAlgorithm(
            enable_downsampling=False,
            mesh_topology="delaunay",
        ).build_mesh_from_points(ground_points)
        artifact_cache.meshes = serialize_mesh(
            mesh=Mesh(vertices=vertices, faces=triangles),
            export_disjoint_meshes=False,
            filename="mesh_ground",
            formats=artifact_config.mesh.ground.formats,
        )

    return artifact_cache
