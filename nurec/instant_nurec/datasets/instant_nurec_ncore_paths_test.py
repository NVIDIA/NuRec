# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from upath import UPath

from nurec.instant_nurec.config.dataset import NCoreInstantNuRecDatasetConfig
from nurec.instant_nurec.datasets import instant_nurec_ncore
from nurec.instant_nurec.datasets.instant_nurec_base import CameraSubsampler
from nurec.instant_nurec.datasets.instant_nurec_ncore import NCoreInstantNuRecDataset
from nurec.instant_nurec.datasets.samplers import FrameBatchSamplerReturn
from nurec.ncore_backends import DatasetRequest
from nurec.utils.misc import list_of_dicts_to_dict_of_lists


def _make_config(**path_config: Any) -> NCoreInstantNuRecDatasetConfig:
    values: dict[str, Any] = {
        "name": "instant_nurec-ncore",
        "aux_data": {
            "enabled": False,
            "enabled_context": False,
            "semantic_segmentation": False,
            "egomask": False,
        },
        "camera_subsampler": {"frame_width": 32, "frame_height": 24},
        "context_camera_ids": [],
        "frame_batch_samplers": {},
        "supervision_camera_ids": [],
        "supervision_frame_batch": {},
        "cuboid_tracks_params": {
            "lidar_id": None,
            "track_min_travel_distance_m": 0.0,
            "track_min_centroid_rig_dist_m": 0.0,
            "track_label_source": "AUTOLABEL",
        },
        **path_config,
    }
    return NCoreInstantNuRecDatasetConfig(**values)


def _local_path_parts(path: Any) -> tuple[str, ...]:
    return tuple(path.parts)


def test_direct_paths_take_precedence_and_resolve_only_relative_entries(tmp_path: Path) -> None:
    base_path = tmp_path / "base"
    absolute_path = tmp_path / "other" / "absolute.json"
    config = _make_config(
        ncore_json_list_path=str(tmp_path / "not-opened.lst"),
        ncore_json_base_path=str(base_path),
        ncore_json_paths=["relative/clip.json", str(absolute_path)],
    )

    dataset = NCoreInstantNuRecDataset(config)

    assert _local_path_parts(dataset.ncore_json_paths[0])[-3:] == ("base", "relative", "clip.json")
    assert _local_path_parts(dataset.ncore_json_paths[1])[-2:] == ("other", "absolute.json")
    assert dataset.ncore_json_paths[1].is_absolute()


def test_list_file_paths_ignore_blank_lines_and_resolve_against_base(tmp_path: Path) -> None:
    base_path = tmp_path / "base"
    absolute_path = tmp_path / "other" / "absolute.json"
    list_path = tmp_path / "dataset.lst"
    list_path.write_text(f"\nrelative/clip.json\n  \n{absolute_path}\n\n")
    config = _make_config(
        ncore_json_list_path=str(list_path),
        ncore_json_base_path=str(base_path),
    )

    dataset = NCoreInstantNuRecDataset(config)

    assert len(dataset.ncore_json_paths) == 2
    assert _local_path_parts(dataset.ncore_json_paths[0])[-3:] == ("base", "relative", "clip.json")
    assert _local_path_parts(dataset.ncore_json_paths[1])[-2:] == ("other", "absolute.json")


def test_list_file_universal_path_bypasses_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    universal_path = "s3@profile://bucket/clip.json"
    parsed_path = tmp_path / "parsed" / "clip.json"
    list_path = tmp_path / "dataset.lst"
    list_path.write_text(f"{universal_path}\n")
    original_parse_universal_path = instant_nurec_ncore.parse_universal_path
    parsed_universal_paths: list[str] = []

    def parse_universal_path(path: str, **kwargs: Any) -> Any:
        if path == universal_path:
            parsed_universal_paths.append(path)
            path = str(parsed_path)
        return original_parse_universal_path(path, **kwargs)

    monkeypatch.setattr(instant_nurec_ncore, "parse_universal_path", parse_universal_path)
    config = _make_config(
        ncore_json_list_path=str(list_path),
        ncore_json_base_path=str(tmp_path / "base"),
    )

    dataset = NCoreInstantNuRecDataset(config)

    assert parsed_universal_paths == [universal_path]
    assert _local_path_parts(dataset.ncore_json_paths[0])[-2:] == ("parsed", "clip.json")


def test_empty_direct_paths_fall_back_to_list_file(tmp_path: Path) -> None:
    list_path = tmp_path / "dataset.lst"
    list_path.write_text("clip.json\n")
    config = _make_config(ncore_json_list_path=str(list_path), ncore_json_paths=[])

    dataset = NCoreInstantNuRecDataset(config)

    assert dataset.ncore_json_paths[0].name == "clip.json"


def test_config_rejects_missing_path_source() -> None:
    with pytest.raises(ValueError, match="Either ncore_json_list_path or a non-empty ncore_json_paths"):
        _make_config()


def test_config_rejects_empty_direct_path_entry() -> None:
    with pytest.raises(ValueError, match=r"ncore_json_paths contains empty paths at indices \[1\]"):
        _make_config(ncore_json_paths=["clip.json", " "])


def test_dataset_request_survives_singleton_metadata_merge(tmp_path: Path) -> None:
    requests = [
        DatasetRequest(metadata_path=UPath(tmp_path / "sequence.json")),
        DatasetRequest(metadata_path=UPath(tmp_path / "sequence.json")),
    ]

    merged = list_of_dicts_to_dict_of_lists(
        [{"dataset_request": request} for request in requests],
        singleton=True,
    )

    assert merged == {"dataset_request": requests[0]}


def test_loader_resolution_reuses_backend_for_external_store_and_cache(tmp_path: Path) -> None:
    dataset = NCoreInstantNuRecDataset.__new__(NCoreInstantNuRecDataset)
    dataset.cache_loaders_and_sensors = True
    dataset._loaders_sensors_cache = {}
    dataset.open_consolidated = True
    dataset.cuboid_loading_max_workers = None
    dataset.poses_component_group = "default"
    dataset.intrinsics_component_group = "default"
    dataset.masks_component_group = "default"
    dataset.cuboids_component_group = "default"
    dataset.aux_data_params = _make_config(ncore_json_paths=["sequence.json"]).aux_data
    dataset.camera_id_mapping = {"main-camera": "archive-main", "external-camera": "archive-external"}
    dataset.lidar_ids = []
    dataset.lidar_id_mapping = {}

    ncore_json_path = UPath(tmp_path / "sequence.json")
    main_dataset_path = UPath(tmp_path / "main.zarr")
    external_dataset_path = UPath(tmp_path / "external.zarr")
    main_camera_id = NCoreInstantNuRecDataset.ExtendedCameraId("main-camera", 0)
    external_camera_id = NCoreInstantNuRecDataset.ExtendedCameraId("external-camera", 1, "external.zarr")

    backend = MagicMock()
    request = MagicMock()
    resolved_dataset = MagicMock(
        backend=backend,
        request=request,
        metadata=SimpleNamespace(dataset_paths=(main_dataset_path,)),
    )
    rig_world_edge = SimpleNamespace(
        T_source_target=np.eye(4, dtype=np.float64)[None],
        timestamps_us=np.asarray([0], dtype=np.int64),
    )
    main_loader = MagicMock(sequence_id="sequence")
    main_loader.pose_graph.get_edge.return_value = rig_world_edge
    external_loader = MagicMock(sequence_id="sequence")
    external_loader.pose_graph.get_edge.return_value = rig_world_edge
    backend.create_sequence_loader.side_effect = [main_loader, external_loader]

    with (
        patch.object(instant_nurec_ncore, "resolve_dataset", return_value=resolved_dataset) as resolve,
        patch.object(instant_nurec_ncore.torch.utils.data, "get_worker_info", return_value=None),
    ):
        result = dataset._get_loaders_and_sensors(ncore_json_path, [main_camera_id, external_camera_id])
        cached_result = dataset._get_loaders_and_sensors(ncore_json_path, [main_camera_id, external_camera_id])

    resolve.assert_called_once()
    assert result.backend is backend
    assert result.request is request
    assert cached_result is result
    assert backend.create_sequence_loader.call_args_list == [
        call(request, (main_dataset_path,)),
        call(request, (external_dataset_path,)),
    ]
    main_loader.get_camera_sensor.assert_called_once_with("archive-main")
    external_loader.get_camera_sensor.assert_called_once_with("archive-external")


@pytest.mark.parametrize("external_path", [None, "other.zarr"])
def test_frame_metadata_keeps_logical_camera_name_not_archive_alias(
    external_path: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    logical_name = "camera_front_wide_120fov"
    dataset = NCoreInstantNuRecDataset.__new__(NCoreInstantNuRecDataset)
    dataset.camera_id_mapping = {logical_name: "camera_frontwide"}
    dataset.lidar_ids = []
    dataset.n_camera_mask_dilation_iterations = 0
    camera_id = NCoreInstantNuRecDataset.ExtendedCameraId(logical_name, 0, external_path)
    camera_sensor = MagicMock()
    camera_sensor.get_frame_timestamp_us.return_value = 100
    camera_sensor.get_frame_image_array.return_value = np.zeros((24, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(instant_nurec_ncore.ncore_utils, "get_camera_sensor_mask", lambda *args, **kwargs: None)

    batch = dataset._load_data_batch(
        FrameBatchSamplerReturn(sampled_sensor_frame_idxs={str(camera_id): [0]}),
        {NCoreInstantNuRecDataset.UniqueFrameId(sensor_id=str(camera_id), frame_idx=0): 7},
        {camera_id: camera_sensor},
        {},
        {},
        {},
        CameraSubsampler(_make_config(ncore_json_paths=["sequence.json"]).camera_subsampler),
        MagicMock(),
    )

    assert batch.camera is not None
    assert batch.camera.meta[0].logical_sensor_id == logical_name
    assert batch.camera.meta[0].unique_sensor_idx == 0
    assert batch.camera.meta[0].unique_frame_idx == 7
