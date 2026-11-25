# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schedule independent scene preparation jobs during startup."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch

from nurec.config.model import SHGaussiansLayerConfig
from nurec.config.nurec import NuRecConfig
from nurec.config.trainer import TrainerConfig
from nurec.datasets.ncore import NCOREDataSource
from nurec.datasets.summary import DataSourceSummary
from nurec.datasets.tracks import CuboidTracks
from nurec.models.gaussians.initializations import BaseInitialization, CameraGroundMeshRoadInitialization
from nurec.utils.initialization import InitializationResources, current_initialization, prefetch_initialization_ply
from nurec.utils.misc import unpack_optional
from nurec.utils.types import TrackPointCloud


InitializationRequest = tuple[BaseInitialization, DataSourceSummary, CuboidTracks | None]
CameraGroup = list[tuple[BaseInitialization, NCOREDataSource, CuboidTracks]]


def layer_labels(layers: Mapping[str, SHGaussiansLayerConfig], layer_id: str) -> tuple[list[str], list[str]]:
    """Return the labels ``layer_id`` ignores (from its ignored layers) and the labels it uses."""
    layer = layers[layer_id]
    labels_to_ignore = [
        label
        for name in (layer.ignore_classes_from_layers or [])
        for label in unpack_optional(layers[name].class_labels)
    ]
    return labels_to_ignore, list(layer.class_labels or [])


def create_initializer(
    layer: SHGaussiansLayerConfig,
    trainer_config: TrainerConfig,
    labels_to_ignore: list[str],
    labels_to_use: list[str],
) -> BaseInitialization:
    return BaseInitialization.factory(
        layer.initialization.name,
        layer.initialization,
        layer,
        trainer_config,
        labels_to_ignore=labels_to_ignore,
        labels_to_use=labels_to_use,
    )


_CONSTRUCTION_INPUTS = ("initialization config", "layer config", "ignored labels", "used labels")


def _construction_inputs(
    layer: SHGaussiansLayerConfig, labels_to_ignore: list[str], labels_to_use: list[str]
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    # Strings and tuples, so a later mutation of the shared configs cannot change the snapshot.
    return str(layer.initialization), str(layer), tuple(labels_to_ignore), tuple(labels_to_use)


@dataclass
class _PreparedInitializer:
    initializer: BaseInitialization
    inputs: tuple[str, str, tuple[str, ...], tuple[str, ...]]
    consumed: bool = False


def parallel_initialization_eligible(config: NuRecConfig) -> bool:
    """Return whether startup preparation may run on background threads."""
    return (
        config.trainer.world_size == 1
        # Resumed, validation and export runs never initialize from data.
        and config.resume is None
        and config.mode in ("train", "trainval")
        and not config.component_only.enabled
        and any(
            layer.initialization is not None and layer.initialization.name == "camera-ground-mesh-road"
            for layer in config.model.layers.values()
        )
    )


def configure_initialization(resources: InitializationResources, config: NuRecConfig) -> None:
    """Start road and point-cloud preparation as early as each input allows."""
    road_initializers: list[CameraGroundMeshRoadInitialization] = []
    for layer_id, layer in config.model.layers.items():
        initialization = layer.initialization
        if initialization is None:
            continue
        if initialization.name == "accumulated-point-cloud" or (
            initialization.name == "camera-ground-mesh-road" and initialization.get("mesh_path") is None
        ):
            if path := initialization.get("point_cloud_path"):
                prefetch_initialization_ply(resources, str(path))
        if initialization.name == "camera-ground-mesh-road" and initialization.get("mesh_path") is None:
            labels_to_ignore, labels_to_use = layer_labels(config.model.layers, layer_id)
            initializer = cast(
                CameraGroundMeshRoadInitialization,
                create_initializer(layer, config.trainer, labels_to_ignore, labels_to_use),
            )
            resources.prepared_initializers[layer_id] = _PreparedInitializer(
                initializer, _construction_inputs(layer, labels_to_ignore, labels_to_use)
            )
            road_initializers.append(initializer)

    # Runs while sensor data is still loading; the mesh job may read only state set before this call.
    def loading(datasource: NCOREDataSource) -> None:
        for initializer in road_initializers:
            resources.submit(
                initializer.preparation_key("mesh", datasource),
                "init-road",
                initializer._compute_mesh_from_fused_point_cloud,
                datasource,
            )

    # Runs once the datasource is fully loaded.
    def ready(datasource: NCOREDataSource) -> None:
        for initializer in road_initializers:
            resources.submit(
                initializer.preparation_key("state", datasource),
                "init-road",
                initializer._compute_camera_ground_mesh_init_state,
                datasource,
            )

    resources.on_datasource_loading.append(loading)
    resources.on_datasource_ready.append(ready)


def acquire_initializer(
    layer_id: str,
    layer: SHGaussiansLayerConfig,
    trainer_config: TrainerConfig,
    labels_to_ignore: list[str],
    labels_to_use: list[str],
) -> BaseInitialization:
    """Return the initializer prepared for ``layer_id`` at startup, or build one on the caller thread."""
    prepared = _take_prepared_initializer(layer_id, layer, labels_to_ignore, labels_to_use)
    return (
        prepared if prepared is not None else create_initializer(layer, trainer_config, labels_to_ignore, labels_to_use)
    )


def _take_prepared_initializer(
    layer_id: str,
    layer: SHGaussiansLayerConfig,
    labels_to_ignore: list[str],
    labels_to_use: list[str],
) -> BaseInitialization | None:
    """Return the initializer prepared for ``layer_id``.

    - Returns None when no scope is active or the layer was not planned; the caller builds it.
    - Raises RuntimeError when the planned initializer was already taken.
    - Raises RuntimeError when a construction input differs from its snapshot at planning time.
    """
    resources = current_initialization()
    prepared = resources.prepared_initializers.get(layer_id) if resources is not None else None
    if prepared is None:
        return None
    if prepared.consumed:
        raise RuntimeError(f"Layer '{layer_id}' already took its prepared initializer")
    current = _construction_inputs(layer, labels_to_ignore, labels_to_use)
    changed = [name for name, planned, now in zip(_CONSTRUCTION_INPUTS, prepared.inputs, current) if planned != now]
    if changed:
        raise RuntimeError(
            f"Layer '{layer_id}' no longer matches its prepared initializer; changed since planning: {', '.join(changed)}"
        )
    prepared.consumed = True
    return prepared.initializer


def prepare_camera_initializers(requests: list[InitializationRequest]) -> None:
    """Share compatible disjoint track traversals, preserving per-layer order."""
    resources = current_initialization()
    if resources is None:
        return
    groups: dict[tuple[int, tuple[str, ...], int, bool], CameraGroup] = defaultdict(list)
    for initializer, summary, tracks in requests:
        if initializer.config.name != "camera-dynamic-tracks" or tracks is None:
            continue
        datasource = summary.datasource
        if not isinstance(datasource, NCOREDataSource):
            continue
        cameras = initializer.config.get("camera_ids")
        if cameras is None:
            cameras = datasource.train_camera_ids
        if not cameras:
            continue
        key = (
            id(datasource),
            tuple(cameras),
            int(initializer.config.get("step_frame", 1)),
            bool(initializer.config.keep_all_track_poses),
        )
        groups[key].append((initializer, datasource, tracks))
    for key, group in groups.items():
        ids = [track for _, _, tracks in group for track in tracks.tracks_id]
        if len(group) < 2 or len(ids) != len(set(ids)):
            continue
        device = torch.device("cuda", torch.cuda.current_device())
        future = resources.submit(
            ("camera-clouds", key, tuple(id(initializer) for initializer, _, _ in group)),
            "init-clouds",
            _shared_camera_clouds,
            group,
            key[1],
            key[2],
            key[3],
            device,
            torch.cuda.current_stream(device),
        )
        for initializer, _, _ in group:
            initializer.prepared_camera_clouds = future


def _shared_camera_clouds(
    group: CameraGroup,
    cameras: tuple[str, ...],
    step: int,
    keep_all: bool,
    device: torch.device,
    origin: torch.cuda.Stream,
) -> dict[int, list[TrackPointCloud]]:
    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(origin)
        with torch.cuda.stream(stream):
            owners = {track: id(initializer) for initializer, _, tracks in group for track in tracks.tracks_id}
            clouds: dict[int, list[TrackPointCloud]] = {id(initializer): [] for initializer, _, _ in group}
            tracks = CuboidTracks.Ops.concatenate([tracks for _, _, tracks in group]).to_device(device)
            for cloud in group[0][1].get_camera_track_point_clouds(
                camera_ids=list(cameras),
                cuboid_tracks=tracks,
                step_frame=step,
                keep_all_track_poses=keep_all,
                device=device,
            ):
                clouds[owners[cloud.track_id]].append(cloud)
            stream.synchronize()
    return clouds
