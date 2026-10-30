# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from upath import UPath

from nurec.ncore_backends import (
    AmbiguousDatasetError,
    DatasetBackend,
    DatasetBackendRegistry,
    DatasetMetadata,
    DatasetRequest,
    UnsupportedDatasetError,
    registered_backend_names,
)


def _write_manifest(path: Path) -> None:
    path.write_text(json.dumps({"version": "test"}))


def _backend(name: str, *, recognizes: bool) -> Mock:
    backend = Mock(spec=DatasetBackend)
    backend.name = name
    backend.recognizes.return_value = recognizes
    backend.resolve_metadata.return_value = DatasetMetadata(
        sequence_id="sequence",
        time_range_start_us=0,
        time_range_end_us=1,
        dataset_paths=(),
    )
    return backend


def test_default_registry_contains_only_public_backend() -> None:
    assert registered_backend_names() == ("ncore-v4",)


def test_dataset_request_is_hashable(tmp_path: Path) -> None:
    request = DatasetRequest(metadata_path=UPath(tmp_path / "sequence.json"))

    assert {request} == {request}


def test_registry_get_returns_registered_backend() -> None:
    backend = _backend("test", recognizes=True)
    registry = DatasetBackendRegistry((backend,))

    assert registry.get("test") is backend

    with pytest.raises(UnsupportedDatasetError, match="not registered"):
        registry.get("missing")


def test_registry_resolves_the_only_matching_backend(tmp_path: Path) -> None:
    metadata_path = tmp_path / "sequence.json"
    _write_manifest(metadata_path)
    ignored = _backend("ignored", recognizes=False)
    selected = _backend("selected", recognizes=True)
    registry = DatasetBackendRegistry((ignored, selected))
    request = DatasetRequest(metadata_path=UPath(metadata_path))

    resolved = registry.resolve(request)

    assert resolved.backend is selected
    ignored.resolve_metadata.assert_not_called()
    selected.resolve_metadata.assert_called_once()


def test_registry_reads_the_manifest_once(tmp_path: Path) -> None:
    metadata_path = tmp_path / "sequence.json"
    _write_manifest(metadata_path)
    registry = DatasetBackendRegistry((_backend("selected", recognizes=True),))

    with patch("nurec.ncore_backends.registry.json.load", wraps=json.load) as load:
        registry.resolve(DatasetRequest(metadata_path=UPath(metadata_path)))

    load.assert_called_once()


def test_registry_wraps_manifest_read_errors(tmp_path: Path) -> None:
    metadata_path = UPath(tmp_path / "sequence.json")

    with (
        patch.object(type(metadata_path), "open", side_effect=OSError("read failed")),
        pytest.raises(UnsupportedDatasetError, match="Could not read dataset manifest") as error,
    ):
        DatasetBackendRegistry().resolve(DatasetRequest(metadata_path=metadata_path))

    assert isinstance(error.value.__cause__, OSError)


def test_registry_preserves_missing_manifest_error(tmp_path: Path) -> None:
    metadata_path = UPath(tmp_path / "missing.json")

    with pytest.raises(UnsupportedDatasetError, match="Dataset manifest does not exist"):
        DatasetBackendRegistry().resolve(DatasetRequest(metadata_path=metadata_path))


def test_registry_rejects_an_unrecognized_manifest(tmp_path: Path) -> None:
    metadata_path = tmp_path / "sequence.json"
    _write_manifest(metadata_path)
    registry = DatasetBackendRegistry((_backend("ignored", recognizes=False),))

    with pytest.raises(UnsupportedDatasetError, match="No registered NCore backend recognized"):
        registry.resolve(DatasetRequest(metadata_path=UPath(metadata_path)))


@pytest.mark.parametrize("reverse_order", [False, True])
def test_registry_rejects_ambiguous_manifests_regardless_of_registration_order(
    tmp_path: Path,
    reverse_order: bool,
) -> None:
    metadata_path = tmp_path / "sequence.json"
    _write_manifest(metadata_path)
    backends = [_backend("first", recognizes=True), _backend("second", recognizes=True)]
    if reverse_order:
        backends.reverse()
    registry = DatasetBackendRegistry(backends)

    with pytest.raises(AmbiguousDatasetError, match="first, second"):
        registry.resolve(DatasetRequest(metadata_path=UPath(metadata_path)))


@pytest.mark.parametrize("contents", ["not JSON", "[]"])
def test_registry_rejects_invalid_manifests(tmp_path: Path, contents: str) -> None:
    metadata_path = tmp_path / "sequence.json"
    metadata_path.write_text(contents)

    with pytest.raises(UnsupportedDatasetError, match="manifest"):
        DatasetBackendRegistry().resolve(DatasetRequest(metadata_path=UPath(metadata_path)))
