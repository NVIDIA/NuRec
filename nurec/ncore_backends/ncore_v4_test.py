# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from upath import UPath

import ncore.data.v4

from nurec.ncore_backends import (
    DatasetRequest,
    UnsupportedDatasetError,
    registered_backend_names,
    resolve_dataset,
)
from nurec.ncore_backends.ncore_v4 import NCORE_V4_BACKEND


def _write_metadata(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "sequence_id": "public-sequence",
                "sequence_timestamp_interval_us": {"start": 10, "stop": 20},
                "version": "v4.0",
                "component_stores": [{"path": "components.zarr"}],
            }
        )
    )


def test_public_backend_is_registered_by_default(tmp_path: Path) -> None:
    metadata_path = tmp_path / "sequence.json"
    _write_metadata(metadata_path)

    resolved = resolve_dataset(DatasetRequest(metadata_path=UPath(metadata_path)))

    assert registered_backend_names() == ("ncore-v4",)
    assert resolved.backend.name == "ncore-v4"
    assert resolved.metadata.sequence_id == "public-sequence"
    assert resolved.metadata.time_range_start_us == 10
    assert resolved.metadata.time_range_end_us == 20
    assert resolved.metadata.dataset_paths == (UPath(tmp_path / "components.zarr"),)


@pytest.mark.parametrize(
    "metadata",
    [
        {"version": "v5.0", "sequence_timestamp_interval_us": {}, "component_stores": []},
        {"sequence_id": "unversioned"},
    ],
)
def test_public_backend_does_not_claim_other_manifests(tmp_path: Path, metadata: dict[str, object]) -> None:
    metadata_path = tmp_path / "sequence.json"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(UnsupportedDatasetError, match="No registered NCore backend recognized"):
        resolve_dataset(DatasetRequest(metadata_path=UPath(metadata_path)))


@pytest.mark.parametrize("missing_field", ["timestamp", "store_path"])
def test_invalid_public_metadata_raises_value_error(tmp_path: Path, missing_field: str) -> None:
    metadata_path = tmp_path / "sequence.json"
    timestamp_interval = {"start": 10, "stop": 20}
    component_stores = [{"path": "components.zarr"}]
    metadata = {
        "sequence_id": "public-sequence",
        "sequence_timestamp_interval_us": timestamp_interval,
        "version": "v4.0",
        "component_stores": component_stores,
    }
    if missing_field == "timestamp":
        del timestamp_interval["start"]
    else:
        del component_stores[0]["path"]
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="Invalid NCore dataset metadata"):
        resolve_dataset(DatasetRequest(metadata_path=UPath(metadata_path)))


@pytest.mark.parametrize("null_field", ["sequence_timestamp_interval_us", "component_stores"])
def test_null_public_metadata_raises_value_error(tmp_path: Path, null_field: str) -> None:
    metadata_path = tmp_path / "sequence.json"
    metadata: dict[str, object] = {
        "sequence_id": "public-sequence",
        "sequence_timestamp_interval_us": {"start": 10, "stop": 20},
        "version": "v4.0",
        "component_stores": [{"path": "components.zarr"}],
    }
    metadata[null_field] = None
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="Invalid NCore dataset metadata"):
        resolve_dataset(DatasetRequest(metadata_path=UPath(metadata_path)))


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("sequence_id", 123),
        ("sequence_timestamp_interval_us", {"start": "10", "stop": 20}),
        ("component_stores", [{"path": 123}]),
    ],
)
def test_wrongly_typed_public_metadata_raises_value_error(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    metadata_path = tmp_path / "sequence.json"
    metadata: dict[str, object] = {
        "sequence_id": "public-sequence",
        "sequence_timestamp_interval_us": {"start": 10, "stop": 20},
        "version": "v4.0",
        "component_stores": [{"path": "components.zarr"}],
    }
    metadata[field] = invalid_value
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="Invalid NCore dataset metadata"):
        resolve_dataset(DatasetRequest(metadata_path=UPath(metadata_path)))


def test_resolved_dataset_constructs_public_loader(tmp_path: Path) -> None:
    metadata_path = tmp_path / "sequence.json"
    _write_metadata(metadata_path)
    resolved = resolve_dataset(DatasetRequest(metadata_path=UPath(metadata_path), open_consolidated=False))
    component_reader = Mock()
    sequence_loader = Mock()

    with (
        patch.object(
            ncore.data.v4,
            "SequenceComponentGroupsReader",
            return_value=component_reader,
        ) as reader_type,
        patch.object(ncore.data.v4, "SequenceLoaderV4", return_value=sequence_loader) as loader_type,
    ):
        result = resolved.create_sequence_loader()

    assert result is sequence_loader
    reader_type.assert_called_once_with([UPath(tmp_path / "components.zarr")], open_consolidated=False)
    loader_type.assert_called_once_with(
        component_reader,
        poses_component_group_name="default",
        intrinsics_component_group_name="default",
        masks_component_group_name="default",
        cuboids_component_group_name="default",
    )


def test_public_loader_is_one_sequence_part() -> None:
    loader = Mock(spec=ncore.data.v4.SequenceLoaderV4)
    loader.sequence_id = "public-sequence"

    part = NCORE_V4_BACKEND.get_sequence_part_info(loader)

    assert part.base_name == "public-sequence"
    assert part.part_index == 0
    assert part.part_count == 1
