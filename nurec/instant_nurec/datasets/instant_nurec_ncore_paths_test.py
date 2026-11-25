# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from upath import UPath

from nurec.instant_nurec.config.dataset import NCoreInstantNuRecCuboidTracksParamsConfig, NCoreInstantNuRecDatasetConfig
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


def _cuboid_tracks_params(track_extrapolate_timestamps_us: int) -> NCoreInstantNuRecCuboidTracksParamsConfig:
    return NCoreInstantNuRecCuboidTracksParamsConfig(
        lidar_id=None,
        track_min_travel_distance_m=0.0,
        track_min_centroid_rig_dist_m=1.0,
        track_extrapolate_timestamps_us=track_extrapolate_timestamps_us,
        track_label_source="AUTOLABEL",
    )


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


@pytest.mark.parametrize(
    ("cache_enabled", "track_extrapolate_timestamps_us", "expected_compute_calls"),
    [
        (True, -1, 1),
        (False, -1, 2),
        (True, 1_000_000, 2),
    ],
)
def test_cuboid_observation_cache_only_reuses_full_clip_requests(
    cache_enabled: bool,
    track_extrapolate_timestamps_us: int,
    expected_compute_calls: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = NCoreInstantNuRecDataset.__new__(NCoreInstantNuRecDataset)
    dataset.cache_full_clip_cuboid_observations = cache_enabled
    dataset.cuboid_tracks_params = _cuboid_tracks_params(track_extrapolate_timestamps_us)
    dataset._full_clip_cuboid_observations_cache = {}
    sequence_loader = SimpleNamespace(sequence_id="clip-a")
    interval = instant_nurec_ncore.HalfClosedInterval(0, 100)
    compute_calls = 0

    def compute_cuboid_df(loader: object, time_range: object, *, serialize_observation: bool) -> pd.DataFrame:
        nonlocal compute_calls
        compute_calls += 1
        return pd.DataFrame({"call": [compute_calls]})

    monkeypatch.setattr(instant_nurec_ncore, "compute_cuboid_df", compute_cuboid_df)

    first = dataset._get_cuboid_observations(sequence_loader, interval, cache_key="/data/clip-a.json")
    second = dataset._get_cuboid_observations(sequence_loader, interval, cache_key="/data/clip-a.json")

    assert compute_calls == expected_compute_calls
    assert (first is second) is (expected_compute_calls == 1)


@pytest.mark.parametrize(
    ("cache_enabled", "track_extrapolate_timestamps_us", "expected_consolidations"),
    [
        (True, -1, 1),
        (False, -1, 0),
        (True, 1_000_000, 0),
    ],
)
def test_cuboid_consolidation_cache_only_reuses_full_clip_requests(
    cache_enabled: bool,
    track_extrapolate_timestamps_us: int,
    expected_consolidations: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = NCoreInstantNuRecDataset.__new__(NCoreInstantNuRecDataset)
    dataset.cache_full_clip_cuboid_consolidation = cache_enabled
    dataset.cuboid_tracks_params = _cuboid_tracks_params(track_extrapolate_timestamps_us)
    dataset._full_clip_cuboid_consolidation_cache = {}
    sequence_loader = SimpleNamespace(sequence_id="clip-a")
    cuboids_df = pd.DataFrame()
    consolidate_calls = 0

    def consolidate_cuboid_tracks(**kwargs: Any) -> dict[str, dict]:
        nonlocal consolidate_calls
        consolidate_calls += 1
        assert kwargs["T_world_world_base"] is None
        return {"track": {"call": consolidate_calls}}

    monkeypatch.setattr(instant_nurec_ncore, "consolidate_cuboid_tracks", consolidate_cuboid_tracks)

    first = dataset._get_canonical_cuboid_tracks(cuboids_df, sequence_loader, cache_key="/data/clip-a.json")
    second = dataset._get_canonical_cuboid_tracks(cuboids_df, sequence_loader, cache_key="/data/clip-a.json")

    assert consolidate_calls == expected_consolidations
    if expected_consolidations:
        assert first is second
    else:
        assert first is None
        assert second is None


def test_cuboid_observation_cache_distinguishes_manifests_with_same_sequence_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = NCoreInstantNuRecDataset.__new__(NCoreInstantNuRecDataset)
    dataset.cache_full_clip_cuboid_observations = True
    dataset.cuboid_tracks_params = _cuboid_tracks_params(-1)
    dataset._full_clip_cuboid_observations_cache = {}
    loaders = [SimpleNamespace(sequence_id="same-clip"), SimpleNamespace(sequence_id="same-clip")]
    keys = ["/exports/v1/clip.json", "/exports/v2/clip.json"]
    interval = instant_nurec_ncore.HalfClosedInterval(0, 100)
    observations = [pd.DataFrame({"version": [1]}), pd.DataFrame({"version": [2]})]
    compute = MagicMock(side_effect=[observations[0], observations[1], observations[0]])
    monkeypatch.setattr(instant_nurec_ncore, "compute_cuboid_df", compute)

    # A,A and B,B hit independently; returning to A recomputes because only the
    # active manifest is retained, preserving the cache's bounded memory use.
    for expected_calls, index in enumerate([0, 1, 0], start=1):
        for _ in range(2):
            result = dataset._get_cuboid_observations(loaders[index], interval, cache_key=keys[index])
            assert result is observations[index]
            assert compute.call_count == expected_calls
            assert list(dataset._full_clip_cuboid_observations_cache) == [keys[index]]
        assert compute.call_args.args[0] is loaders[index]


def test_cuboid_consolidation_cache_distinguishes_manifests_with_same_sequence_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = NCoreInstantNuRecDataset.__new__(NCoreInstantNuRecDataset)
    dataset.cache_full_clip_cuboid_consolidation = True
    dataset.cuboid_tracks_params = _cuboid_tracks_params(-1)
    dataset._full_clip_cuboid_consolidation_cache = {}
    loaders = [SimpleNamespace(sequence_id="same-clip"), SimpleNamespace(sequence_id="same-clip")]
    keys = ["/exports/v1/clip.json", "/exports/v2/clip.json"]
    observations = [pd.DataFrame({"version": [1]}), pd.DataFrame({"version": [2]})]
    tracks = [{"track": {"version": 1}}, {"track": {"version": 2}}]
    consolidate = MagicMock(side_effect=[tracks[0], tracks[1], tracks[0]])
    monkeypatch.setattr(instant_nurec_ncore, "consolidate_cuboid_tracks", consolidate)

    for expected_calls, index in enumerate([0, 1, 0], start=1):
        for _ in range(2):
            result = dataset._get_canonical_cuboid_tracks(observations[index], loaders[index], cache_key=keys[index])
            assert result is tracks[index]
            assert consolidate.call_count == expected_calls
            assert list(dataset._full_clip_cuboid_consolidation_cache) == [keys[index]]
        assert consolidate.call_args.kwargs["sequence_loader"] is loaders[index]
        assert consolidate.call_args.kwargs["cuboids_df"] is observations[index]


def test_copy_cuboid_tracks_with_world_base_isolates_per_chunk_mutation() -> None:
    canonical: dict[str, dict[str, Any]] = {
        "track": {
            "dimension": np.asarray([1.0, 2.0, 3.0]),
            "label_class": 4,
            "poses": [np.eye(4), np.eye(4)],
            "timestamps_us": [10, 20],
        }
    }
    first_base = np.eye(4)
    first_base[0, 3] = 1.0
    second_base = np.eye(4)
    second_base[1, 3] = 2.0

    first = NCoreInstantNuRecDataset._copy_cuboid_tracks_with_world_base(canonical, first_base)
    second = NCoreInstantNuRecDataset._copy_cuboid_tracks_with_world_base(canonical, second_base)
    first["track"]["poses"].append(np.eye(4))
    first["track"]["timestamps_us"].append(30)

    assert len(canonical["track"]["poses"]) == 2
    assert len(second["track"]["poses"]) == 2
    np.testing.assert_allclose(first["track"]["poses"][0], first_base)
    np.testing.assert_allclose(second["track"]["poses"][0], second_base)


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
