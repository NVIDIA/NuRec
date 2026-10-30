# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import uuid

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict

import dataclasses_json
import torch
import yaml

from upath import UPath

from nurec.config.nurec import NuRecConfig
from nurec.config.version import get_version
from nurec.instant_nurec.config.dataset import BaseNCoreInstantNuRecDatasetConfig
from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig
from nurec.utils.media_log import media_backend
from nurec.utils.types import NamedSerialized


@dataclass(kw_only=True)
class Metadata(dataclasses_json.DataClassJsonMixin):
    scene_id: str
    version_string: str
    training_date: str
    dataset_hash: str
    uuid: str
    is_resumable: bool

    @dataclass
    class Sensors(dataclasses_json.DataClassJsonMixin):
        camera_ids: list[str] = field(default_factory=list)
        lidar_ids: list[str] = field(default_factory=list)

    sensors: Sensors

    @dataclass
    class Logger(dataclasses_json.DataClassJsonMixin):
        name: str | None = None
        run_id: str | None = None
        run_url: str | None = None

    logger: Logger

    @dataclass
    class TimeRange(dataclasses_json.DataClassJsonMixin):
        start: int
        end: int

    time_range: TimeRange

    training_step_outputs: Dict[str, float] = field(default_factory=dict)


def calculate_md5(file_path: Path | UPath):
    md5_hash = hashlib.md5()
    with file_path.open("rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def manifest_data_to_sequence_id(manifest_data: Dict) -> str:
    if "sequence_id" in manifest_data:
        # single sequence
        return manifest_data["sequence_id"]
    elif "sequences" in manifest_data:
        # multi-sequence
        return "|".join(manifest_data["sequences"].keys())
    else:
        raise KeyError("dataset.json must contain a sequence ID or multiple sequences!")


def get_logger_metadata(logger_name: str) -> Metadata.Logger:
    logger = Metadata.Logger(name=logger_name)
    if logger_name == "wandb":
        backend = media_backend()
        if backend is not None:
            run = backend.active_run()
            if run is not None:
                logger.run_url, logger.run_id = run

    return logger


def get_time_range(manifest_data: Dict) -> Metadata.TimeRange:
    if "pose-range" in manifest_data:
        # ncore v3 single sequence
        start_timestamp_us = manifest_data["pose-range"]["start-timestamp_us"]
        end_timestamp_us = manifest_data["pose-range"]["end-timestamp_us"]
    elif "sequences" in manifest_data:
        # ncore v3 multi-sequence
        start_timestamps_us = []
        end_timestamps_us = []
        for sequence_manifest in manifest_data["sequences"].values():
            for chunk_manifest in sequence_manifest["chunks"]:
                start_timestamps_us.append(chunk_manifest["start-timestamp_us"])
                end_timestamps_us.append(chunk_manifest["end-timestamp_us"])
        start_timestamp_us = min(start_timestamps_us)
        end_timestamp_us = max(end_timestamps_us)
    elif "sequence_timestamp_interval_us" in manifest_data:
        # ncore v4 single sequence
        start_timestamp_us = manifest_data["sequence_timestamp_interval_us"]["start"]
        end_timestamp_us = manifest_data["sequence_timestamp_interval_us"]["stop"]
    else:
        raise KeyError("dataset.json doesn't match expected format for manifest!")

    return Metadata.TimeRange(start=start_timestamp_us, end=end_timestamp_us)


def get_max_of_each_step_output(outputs: Dict[str, list[torch.Tensor]]) -> dict[str, float]:
    return {key: torch.max(torch.stack(tensors)).item() for key, tensors in outputs.items()}


def get_metadata_nreconfig(
    config: NuRecConfig, dataset_path: Path, training_step_outputs: Dict[str, list[torch.Tensor]]
) -> Metadata:
    assert config.dataset.name == "ncore"

    with open(dataset_path, "r") as file:
        manifest_data = json.load(file)
    metadata = Metadata(
        scene_id=manifest_data_to_sequence_id(manifest_data),
        version_string=repr(config.version),
        training_date=date.today().isoformat(),
        uuid=str(uuid.uuid4()),
        dataset_hash=calculate_md5(dataset_path),
        is_resumable=False,  # TODO: Introduce config flag and connect to this function
        sensors=Metadata.Sensors(
            camera_ids=list(config.dataset.camera_ids),
            lidar_ids=list(config.dataset.lidar_ids),
        ),
        logger=get_logger_metadata(logger_name=config.logger.name),
        time_range=get_time_range(manifest_data),
        training_step_outputs=get_max_of_each_step_output(training_step_outputs),
    )
    return metadata


def get_metadata_instant_nurecconfig(config: InstantNuRecConfig, dataset_path: Path | UPath) -> Metadata:
    """Get metadata from instant_nurec config for given scene id. Exports metadata for the entire clip.

    Can be used for exporting artifacts, which are saved for specific scene_id.
    """

    assert config.dataset.name == "instant_nurec"
    assert config.dataset.predict is not None, "predict dataset config is required"
    assert isinstance(config.dataset.predict, BaseNCoreInstantNuRecDatasetConfig), (
        f"get_metadata_instant_nurecconfig only supports NCore-based dataset configs, got {type(config.dataset.predict)}"
    )
    predict_config = config.dataset.predict

    with dataset_path.open("r") as file:
        manifest_data = json.load(file)

    metadata = Metadata(
        scene_id=manifest_data_to_sequence_id(manifest_data),
        # Use the current version that generates this metadata
        version_string=repr(get_version()),
        # This information is redacted from pretrained checkpoints so we use today as a placeholder
        training_date=date.today().isoformat(),
        uuid=str(uuid.uuid4()),
        dataset_hash=calculate_md5(dataset_path),
        is_resumable=False,
        sensors=Metadata.Sensors(
            camera_ids=[
                camera_id if isinstance(camera_id, str) else camera_id.camera_id
                for camera_id in predict_config.context_camera_ids
            ],
            lidar_ids=[],
        ),
        logger=get_logger_metadata(logger_name=config.logger.name),
        time_range=get_time_range(manifest_data),
        training_step_outputs={},
    )
    return metadata


def get_metadata(
    config: NuRecConfig | InstantNuRecConfig,
    dataset_path: Path | UPath,
    training_step_outputs: Dict[str, list[torch.Tensor]] | None = None,
) -> Metadata:
    if isinstance(config, NuRecConfig):
        assert training_step_outputs is not None
        assert isinstance(dataset_path, Path), "dataset_path must be a Path for NuRecConfig"
        return get_metadata_nreconfig(config, dataset_path, training_step_outputs)
    elif isinstance(config, InstantNuRecConfig):
        return get_metadata_instant_nurecconfig(config, dataset_path)
    else:
        raise ValueError(f"Invalid config type: {type(config)}")


def serialize_metadata(metadata: Metadata, filename: str = "metadata.yaml") -> NamedSerialized:
    return NamedSerialized(filename=filename, serialized=yaml.dump(metadata.to_dict(), default_flow_style=False))
