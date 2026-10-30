# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from pathlib import Path

import pytest

from omegaconf import OmegaConf

from nurec.config.resolvers import _resolve_sam3_aux_dir


def _dataset(tmp_path: Path) -> Path:
    """Use a metadata filename deliberately different from the sequence identity."""
    path = tmp_path / "renamed-dataset.json"
    path.write_text(
        json.dumps(
            {
                "version": "v4.0",
                "sequence_id": "clip-uuid",
                "sequence_timestamp_interval_us": {"start": 0, "stop": 100},
                "component_stores": [{"path": "stores/clip.zarr"}],
            }
        )
    )
    return path


@pytest.mark.parametrize("location", ["clip-uuid/sam3", "sam3", "stores/sam3"])
def test_resolves_ndas_and_standalone_aux_layouts(tmp_path: Path, location: str) -> None:
    dataset = _dataset(tmp_path)
    root = tmp_path / location
    root.mkdir(parents=True)

    assert _resolve_sam3_aux_dir(str(dataset)) == str(root)


def test_absent_aux_does_not_enable_the_optional_layer(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    assert _resolve_sam3_aux_dir(str(dataset)) is None
    assert _resolve_sam3_aux_dir(str(tmp_path / "missing.json")) is None


def test_multiple_aux_roots_require_explicit_selection(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    (tmp_path / "sam3").mkdir()
    (tmp_path / "clip-uuid" / "sam3").mkdir(parents=True)

    with pytest.raises(ValueError, match="Ambiguous SAM3 aux directories"):
        _resolve_sam3_aux_dir(str(dataset))

    assert _resolve_sam3_aux_dir(str(dataset), "/explicit/masks", "/explicit/manifest.json") is None


@pytest.mark.parametrize("artifact_present", [False, True])
def test_saved_config_preserves_discovery_result(tmp_path: Path, artifact_present: bool) -> None:
    """Reloading a resolved config does not rediscover newly added or removed data."""
    dataset = _dataset(tmp_path)
    root = tmp_path / "clip-uuid" / "sam3"
    root.parent.mkdir()
    if artifact_present:
        root.mkdir()
    config = OmegaConf.create({"sam3_aux_dir": "${sam3_aux_dir:" + str(dataset) + "}"})
    OmegaConf.resolve(config)
    saved = OmegaConf.to_yaml(config)
    if artifact_present:
        root.rmdir()
    else:
        root.mkdir()

    restored = OmegaConf.create(saved)
    assert restored.sam3_aux_dir == (str(root) if artifact_present else None)
