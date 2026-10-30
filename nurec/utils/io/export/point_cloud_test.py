# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import point_cloud_utils as pcu
import pytest

from click.testing import CliRunner
from python.runfiles import runfiles

from nurec.utils.io.export.point_cloud import export_point_cloud


RUNFILES = runfiles.Create()


def _testdata_config() -> str:
    """Absolute runfiles path of the self-contained fixture config (parent-rooted by the parser)."""
    path = Path(RUNFILES.Rlocation("_main/nurec/utils/io/export/testdata/ncore_ds.yaml"))
    if not path.is_file():
        raise AssertionError(f"Test config not found at {path}.")
    return str(path)


@pytest.fixture
def dataset_path_str() -> str:
    path = Path(
        RUNFILES.Rlocation(
            "test_data_ncore_v4/cf5ff7f6-5c82-11ed-806f-00044bf655de_1667597307250262_1667597318349978_1667597307250262_1667597308250262.json"
        ),
    )
    if not path.is_file():
        raise AssertionError(f"Test dataset not found at {path}.")

    # Ensure the path is a *quoted* string for Hydra compatibility with bazel's `~`-separated paths
    return '"{0}"'.format(str(path))


def test_point_cloud_rgb_colors_fused(dataset_path_str: str, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        export_point_cloud,
        [
            f"--config-name={_testdata_config()}",
            f"dataset.path={dataset_path_str}",
            "--output-dir",
            str(tmp_path),
            "--colorizer",
            "rgb",
        ],
        catch_exceptions=False,
    )
    print(result.output)
    assert result.exit_code == 0
    file_path = tmp_path / "colored_point_cloud.ply"
    assert file_path.is_file(), str(file_path) + " not found"


def test_point_cloud_semantic_colors_split_per_frame_per_class(dataset_path_str: str, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        export_point_cloud,
        [
            f"--config-name={_testdata_config()}",
            f"dataset.path={dataset_path_str}",
            "--output-dir",
            str(tmp_path),
            "--colorizer",
            "semantic",
            "--per-frame",
            "--per-class",
            "--frame-step",
            "8",
        ],
        catch_exceptions=False,
    )
    print(result.output)

    assert result.exit_code == 0

    for frame_idx in [0, 8]:
        for class_name in [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "car",
        ]:
            file_path = tmp_path / f"semantic_point_cloud_{frame_idx:04d}_{class_name}.ply"
            assert file_path.is_file(), str(file_path) + " not found"


def test_point_cloud_include_dynamic_points_semantic_world_frame(dataset_path_str: str, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        export_point_cloud,
        [
            f"--config-name={_testdata_config()}",
            f"dataset.path={dataset_path_str}",
            "--output-dir",
            str(tmp_path),
            "--colorizer",
            "semantic",
            "--include-dynamic-points",
            "--coordinate-system",
            "world",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    ply_path = tmp_path / "semantic_point_cloud.ply"
    assert ply_path.is_file()
    assert pcu.load_mesh_v(str(ply_path)).shape[0] > 0


def test_point_cloud_road_colors_split_per_class(dataset_path_str: str, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        export_point_cloud,
        [
            f"--config-name={_testdata_config()}",
            f"dataset.path={dataset_path_str}",
            "--output-dir",
            str(tmp_path),
            "--colorizer",
            "road",
            "--per-class",
        ],
        catch_exceptions=False,
    )
    print(result.output)
    assert result.exit_code == 0
    for class_name in ["road", "nonroad"]:
        file_path = tmp_path / f"segmented_point_cloud_{class_name}.ply"
        assert file_path.is_file(), str(file_path) + " not found"
