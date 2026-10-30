# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import unittest

from pathlib import Path

import numpy as np
import pytest
import torch

from python.runfiles import runfiles

from apps.av.aux_gen.utils import _quantile
from nurec.utils.ncore_utils import EGO_MASK_BASE_GROUP, AuxShardDataLoader, CamAuxDataLoader


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


def test_ncore_aux_data(small_dataset_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Integration test to validate execution of ncore_aux_data cli on small test data set"""

    print("Output root:", output_root := tmp_path_factory.mktemp("aux_gen_test"))

    _run_ncore_aux_data(
        f"--dataset-path={small_dataset_path}",
        "--camera-id=camera_front_wide_120fov",
        f"--output-dir={output_root}",
        "--store-meta",
        "offset",
        "--sequence-duration-sec=0.4",  # restrict to few frames only to terminate more quickly
    )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Shared segmentation GPU integration test requires CUDA")
@pytest.mark.parametrize("seg_logits", [False, True])
def test_ncore_aux_data_shared_segmentation_matches_serial(
    small_dataset_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    seg_logits: bool,
) -> None:
    """Compare shared and serial segmentation and ego-mask archives."""
    output_root = tmp_path_factory.mktemp("aux_gen_test_mask2former_parity")
    serial_output = output_root / "serial"
    parallel_output = output_root / "parallel"
    common_args = [
        f"--dataset-path={small_dataset_path}",
        "--camera-id=camera_front_wide_120fov",
        "--seg-logits" if seg_logits else "--no-seg-logits",
        "--no-lidar-seg-camvis",
        "offset",
        "--sequence-duration-sec=0.2",
    ]
    for output_dir, mode_args in (
        (serial_output, []),
        (parallel_output, ["--parallel-mode"]),
    ):
        _run_ncore_aux_data(
            *common_args[:2],
            f"--output-dir={output_dir}",
            *mode_args,
            *common_args[2:],
        )

    serial_semantic_files = list(serial_output.glob("*.aux.sseg.zarr.itar"))
    parallel_semantic_files = list(parallel_output.glob("*.aux.sseg.zarr.itar"))
    serial_logits_files = list(serial_output.glob("*.aux.sseg-logits.zarr.itar"))
    parallel_logits_files = list(parallel_output.glob("*.aux.sseg-logits.zarr.itar"))
    serial_egomask_files = list(serial_output.glob("*.aux.egomask.zarr.itar"))
    parallel_egomask_files = list(parallel_output.glob("*.aux.egomask.zarr.itar"))
    assert len(serial_semantic_files) == 1
    assert len(parallel_semantic_files) == 1
    assert len(serial_logits_files) == int(seg_logits)
    assert len(parallel_logits_files) == int(seg_logits)
    assert len(serial_egomask_files) == 1
    assert len(parallel_egomask_files) == 1

    camera_id = "camera_front_wide_120fov"
    serial_loader = CamAuxDataLoader(serial_semantic_files[0])
    parallel_loader = CamAuxDataLoader(parallel_semantic_files[0])
    serial_aux_loader = AuxShardDataLoader(
        sequence_id=serial_semantic_files[0].name.partition(".aux.")[0],
        dataset_paths=[serial_semantic_files[0]],
    )
    parallel_aux_loader = AuxShardDataLoader(
        sequence_id=parallel_semantic_files[0].name.partition(".aux.")[0],
        dataset_paths=[parallel_semantic_files[0]],
    )
    assert serial_loader.get_semantic_segmentation_meta(camera_id) == parallel_loader.get_semantic_segmentation_meta(
        camera_id
    )
    serial_timestamps = set(serial_loader._camera_group(camera_id).keys())
    parallel_timestamps = set(parallel_loader._camera_group(camera_id).keys())
    assert serial_timestamps == parallel_timestamps
    assert serial_timestamps
    for timestamp in serial_timestamps:
        np.testing.assert_array_equal(
            np.asarray(serial_loader.get_semantic_segmentation(camera_id, int(timestamp))),
            np.asarray(parallel_loader.get_semantic_segmentation(camera_id, int(timestamp))),
        )

    if seg_logits:
        assert serial_aux_loader.get_semantic_logits_meta(camera_id) == parallel_aux_loader.get_semantic_logits_meta(
            camera_id
        )
        num_classes = len(serial_loader.get_semantic_segmentation_meta(camera_id)["stuff_classes"])
        for timestamp in serial_timestamps:
            serial_logits = serial_aux_loader.get_semantic_logits(camera_id, int(timestamp))
            parallel_logits = parallel_aux_loader.get_semantic_logits(camera_id, int(timestamp))
            assert serial_logits.dtype == np.float16
            assert parallel_logits.dtype == np.float16
            assert serial_logits.shape[-1] == num_classes
            np.testing.assert_array_equal(serial_logits, parallel_logits)

    assert serial_aux_loader.get_egomask_meta(camera_id) == parallel_aux_loader.get_egomask_meta(camera_id)
    serial_egomask_timestamps = set(serial_aux_loader.base_groups[EGO_MASK_BASE_GROUP][0][camera_id].keys())
    parallel_egomask_timestamps = set(parallel_aux_loader.base_groups[EGO_MASK_BASE_GROUP][0][camera_id].keys())
    assert serial_egomask_timestamps == parallel_egomask_timestamps
    assert "0" in serial_egomask_timestamps
    for timestamp in serial_egomask_timestamps:
        serial_egomask = serial_aux_loader.get_egomask(camera_id, int(timestamp))
        parallel_egomask = parallel_aux_loader.get_egomask(camera_id, int(timestamp))
        assert serial_egomask.shape == parallel_egomask.shape
        disagreement_rate = float(np.count_nonzero(serial_egomask != parallel_egomask)) / serial_egomask.size
        # Resize operations can round SAM2 mask boundaries slightly differently on ARM. Semantic labels and logits
        # remain exact above, so tolerate up to 0.02% disagreement in the binary ego-mask boundary.
        assert disagreement_rate <= 2e-4


class TestQuantileFunction(unittest.TestCase):
    def test_torch_quantile(self) -> None:
        a = torch.randn(2, 3)
        x = torch.quantile(a, 0.90, dim=1, keepdim=True)
        y = _quantile(a, 0.90, dim=1, keepdim=True)
        torch.testing.assert_close(x, y)

        a = torch.randn(200, 300)
        x = torch.quantile(a, 0.90)
        y = _quantile(a, 0.90)
        torch.testing.assert_close(x, y)

    def test_torch_quantile_large(self) -> None:
        # If this test fails after upgrading torch, it is probably safe to remove the _quantile function.
        a = torch.randn(4096**2 + 1)
        with self.assertRaises(RuntimeError):
            torch.quantile(a, 0.90)
        _quantile(a, 0.90)
