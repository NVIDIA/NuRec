# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np
import pytest

from click.testing import CliRunner
from PIL import Image
from python.runfiles import runfiles
from upath import UPath

import nurec.utils.io.export.ncore_tracks as ncore_tracks_module

from nurec.benchmark.gt_from_ncore import export_benchmark_gt_from_ncore
from nurec.ncore_backends import (
    DatasetMetadata,
    DatasetRequest,
)
from nurec.utils.io.export.ego_mask import export_ego_mask
from nurec.utils.io.export.ncore_diagnostic import export_ncore_diagnostic
from nurec.utils.io.export.ncore_tracks import export_ncore_tracks


RUNFILES = runfiles.Create()
SEQUENCE = "clipgt-9048443e-c482-4228-8326-5b3dff3be711"
CAMERA_ID = "camera_front_wide_120fov"
LIDAR_ID = "lidar_gt_top_p128"


def _runfile(repository: str, filename: str) -> Path:
    path = Path(RUNFILES.Rlocation(f"{repository}/{filename}"))
    assert path.is_file(), f"Test data not found at {path}"
    return path


@pytest.fixture
def ncore_v4_input() -> list[str]:
    return ["--dataset-path", str(_runfile("test_data_ncore_v4", f"{SEQUENCE}.json"))]


def test_export_ego_mask_supports_v4(ncore_v4_input: list[str], tmp_path: Path) -> None:
    result = CliRunner().invoke(
        export_ego_mask,
        [
            *ncore_v4_input,
            "--output-dir",
            str(tmp_path),
            "--camera-ids",
            CAMERA_ID,
            "--camera-frame-idx",
            "10",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    with Image.open(tmp_path / "ego-hoods" / f"{CAMERA_ID}.png") as output:
        assert output.mode == "RGBA"
        assert output.size == (960, 540)


def test_export_ncore_tracks_supports_v4(ncore_v4_input: list[str], tmp_path: Path) -> None:
    model_tracks_path = tmp_path / "model_tracks.json"
    model_tracks_path.write_text(
        json.dumps(
            {
                "dummy_chunk_id": {
                    "tracks_data": {
                        "tracks_id": [],
                        "tracks_poses": [],
                        "tracks_timestamps_us": [],
                    },
                    "cuboidtracks_data": {"cuboids_dims": []},
                }
            }
        )
    )

    result = CliRunner().invoke(
        export_ncore_tracks,
        [
            *ncore_v4_input,
            "--model-tracks-json",
            str(model_tracks_path),
            "--output-dir",
            str(tmp_path),
            "--camera-id",
            CAMERA_ID,
            "--lidar-id",
            LIDAR_ID,
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    output_paths = list(tmp_path.glob("sensor_tracks_*.json"))
    assert len(output_paths) == 1
    output = json.loads(output_paths[0].read_text())
    assert set(output) == {CAMERA_ID, LIDAR_ID}
    assert output[CAMERA_ID]
    assert output[LIDAR_ID]


def test_export_ncore_diagnostic_supports_v4(ncore_v4_input: list[str], tmp_path: Path) -> None:
    result = CliRunner().invoke(
        export_ncore_diagnostic,
        [
            *ncore_v4_input,
            "--output-dir",
            str(tmp_path),
            "--meta",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert (tmp_path / "meta.yaml").is_file()


def test_export_ncore_tracks_uses_metadata_time_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pose_edge = SimpleNamespace(timestamps_us=np.array([1_000_000, 20_000_000], dtype=np.uint64))
    loader = SimpleNamespace(
        pose_graph=SimpleNamespace(get_edge=lambda _source, _target: pose_edge),
        get_camera_sensor=lambda _camera_id: object(),
    )
    dataset_requests: list[DatasetRequest] = []
    recorded_ranges: list[tuple[int, int]] = []

    def resolve_dataset(request: DatasetRequest) -> SimpleNamespace:
        dataset_requests.append(request)
        return SimpleNamespace(
            metadata=DatasetMetadata(
                sequence_id="sequence",
                time_range_start_us=4_000_000,
                time_range_end_us=10_000_000,
                dataset_paths=(UPath("component"),),
            ),
            create_sequence_loader=lambda: loader,
        )

    def generate_sensor_tracks(
        _sensor: object,
        sensor_id: str,
        track_dict: dict[str, object],
        _dynamic_tracks: dict[str, object],
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> None:
        recorded_ranges.append((start_timestamp_us, end_timestamp_us))
        track_dict[sensor_id] = {}

    monkeypatch.setattr(ncore_tracks_module, "resolve_dataset", resolve_dataset)
    monkeypatch.setattr(ncore_tracks_module, "generate_sensor_tracks", generate_sensor_tracks)

    result = CliRunner().invoke(
        export_ncore_tracks,
        [
            "--dataset-path",
            "sequence.json",
            "--model-tracks-json",
            "unused.json",
            "--output-dir",
            str(tmp_path),
            "--camera-id",
            CAMERA_ID,
            "--disable-lidars",
            "--seek-offset-sec",
            "2",
            "--duration-sec",
            "3",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert recorded_ranges == [(6_000_000, 9_000_000)]
    assert dataset_requests[0].masks_component_group is None
    assert dataset_requests[0].cuboids_component_group is None
    assert (tmp_path / "sensor_tracks_6000000_9000000.json").exists()


@pytest.mark.parametrize(
    ("command", "required_args"),
    [
        (export_ego_mask, ["--output-dir", "output"]),
        (export_ncore_tracks, ["--model-tracks-json", "model_tracks.json", "--output-dir", "output"]),
        (export_ncore_diagnostic, ["--meta", "--output-dir", "output"]),
    ],
)
def test_public_export_commands_require_dataset_path(
    command: click.Command,
    required_args: list[str],
) -> None:
    runner = CliRunner()
    result = runner.invoke(command, required_args)
    assert result.exit_code != 0
    assert "Exactly one sequence input must be provided" in result.output


@pytest.mark.parametrize(
    "command",
    [export_ego_mask, export_ncore_tracks, export_ncore_diagnostic, export_benchmark_gt_from_ncore],
)
def test_public_export_commands_do_not_expose_raw_shard_input(command: click.Command) -> None:
    result = CliRunner().invoke(command, ["--help"])
    assert result.exit_code == 0
    assert "--shard-file-pattern" not in result.output
