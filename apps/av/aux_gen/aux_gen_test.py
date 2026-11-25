# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess

from pathlib import Path

import pytest

from python.runfiles import runfiles

from apps.av.aux_gen.segmentation_test_support import assert_segmentation_archives_match


RUNFILES = runfiles.Create()


def _run_ncore_aux_data(*args: str) -> None:
    """Run one CLI integration case in an isolated CUDA process."""
    binary_path = RUNFILES.Rlocation("_main/apps/av/aux_gen/ncore_aux_data")
    assert binary_path is not None
    result = subprocess.run(
        [binary_path, *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.fixture
def small_dataset_path() -> Path:
    path = Path(
        RUNFILES.Rlocation(
            "test_data_ncore_v4/cf5ff7f6-5c82-11ed-806f-00044bf655de_1667597307250262_1667597318349978_1667597307250262_1667597308250262.json"
        ),
    )
    if not path.exists():
        raise AssertionError(
            f"Test dataset not found. This is an issue with your filesystem/test suite, not the code under test. Missing {path=}"
        )

    return path


def test_ncore_aux_data_shared_segmentation_matches_serial(small_dataset_path: Path, tmp_path: Path) -> None:
    """Compare real-model labels and ego masks on one frame, including spawned workers."""
    # An interior camera frame lies within every LiDAR sensor's timestamp range.
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    for output, mode in ((serial, "--serial-mode"), (parallel, "--parallel-mode")):
        _run_ncore_aux_data(
            f"--dataset-path={small_dataset_path}",
            "--camera-id=camera_front_wide_120fov",
            f"--output-dir={output}",
            mode,
            "--store-meta",
            "--no-seg-logits",
            "sensor-frames",
            "--main-sensor-id=camera_front_wide_120fov",
            "--start-frame=4",
            "--stop-frame=5",
        )
    # Real SAM2 resize kernels can shift boundary pixels slightly on ARM.
    labels, _ = assert_segmentation_archives_match(
        serial, parallel, "camera_front_wide_120fov", False, mask_tolerance=2e-4
    )
    assert len(labels._camera_group("camera_front_wide_120fov")) == 1


def test_ncore_aux_data_no_ego_mask_creates_no_egomask_file(
    small_dataset_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """With --no-ego-mask, no *.aux.egomask.* file should be created (fix for mistaken empty store)."""
    output_root = tmp_path_factory.mktemp("aux_gen_test_no_egomask")
    _run_ncore_aux_data(
        f"--dataset-path={small_dataset_path}",
        "--camera-id=camera_front_wide_120fov",
        f"--output-dir={output_root}",
        "--store-meta",
        "--no-ego-mask",
        "offset",
        "--sequence-duration-sec=0.4",
    )
    egomask_files = list(output_root.rglob("*egomask*"))
    assert not egomask_files, f"Expected no egomask files with --no-ego-mask, found: {egomask_files}"
