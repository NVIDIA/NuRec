# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import zipfile

from pathlib import Path

import yaml

from click.testing import CliRunner

from nurec.artifact import Artifact
from nurec.utils.io.export.external_assets import export_external_assets


def test_export_external_assets_round_trip(tmp_path: Path) -> None:
    input_artifact = tmp_path / "input.usdz"
    with zipfile.ZipFile(input_artifact, "w") as zip_file:
        zip_file.writestr("scene.usda", "#usda 1.0")

    harvester_output = tmp_path / "harvested_assets"
    assets = {
        "8": ("automobile", [4.5, 2.0, 1.8], b"ply-eight"),
        "13": ("pedestrian", [0.7, 0.7, 1.8], b"ply-thirteen"),
        "18": ("construction_vehicle", [7.0, 3.0, 3.5], b"ply-eighteen"),
    }
    metadata_assets = {}
    for track_id, (label_class, cuboids_dims, ply_bytes) in assets.items():
        asset_dir = harvester_output / label_class / track_id
        asset_dir.mkdir(parents=True)
        ply_path = asset_dir / "gaussians.ply"
        ply_path.write_bytes(ply_bytes)
        metadata_assets[track_id] = {
            "label_class": label_class,
            "cuboids_dims": cuboids_dims,
            "ply_file": str(ply_path.relative_to(harvester_output)),
        }
    (harvester_output / "metadata.yaml").write_text(yaml.safe_dump({"assets": metadata_assets}))

    output_artifact = tmp_path / "output.usdz"
    output_edit_file = tmp_path / "edit-assets.json"
    result = CliRunner().invoke(
        export_external_assets,
        [
            "--artifact-path",
            str(input_artifact),
            "--external-assets-dir",
            str(harvester_output),
            "--output-artifact-path",
            str(output_artifact),
            "--output-edit-file",
            str(output_edit_file),
        ],
    )

    assert result.exit_code == 0, result.output
    with zipfile.ZipFile(output_artifact) as zip_file:
        assert zip_file.read("scene.usda") == b"#usda 1.0"
        assert "external_assets/metadata.yaml" in zip_file.namelist()
        for track_id in assets:
            assert f"external_assets/{track_id}/{track_id}.ply" in zip_file.namelist()

    packaged_assets = Artifact(output_artifact).external_assets
    assert set(packaged_assets) == set(assets)
    for track_id, (label_class, cuboids_dims, ply_bytes) in assets.items():
        assert packaged_assets[track_id] == {
            "ply_bytes": ply_bytes,
            "metadata": {"label_class": label_class, "cuboids_dims": cuboids_dims},
        }

    edit_config = json.loads(output_edit_file.read_text())
    assert edit_config["metadata"]["output_artifact_path"] == str(output_artifact)
    assert {entry["track_id"] for entry in edit_config["metadata"]["external_assets_metadata"]} == set(assets)
