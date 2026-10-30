# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from pathlib import Path

import numpy as np
import pytest

from click.testing import CliRunner
from python.runfiles import runfiles

from apps.av.aux_gen.ncore_aux_data import cli
from nurec.utils.ncore_utils import CamAuxDataLoader


RUNFILES = runfiles.Create()


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate intersection-over-union between two binary masks.

    Args:
        mask1: First boolean mask.
        mask2: Second boolean mask with the same shape.

    Returns:
        Intersection-over-union, or one when both masks are empty.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 1.0
    return float(iou)


def calculate_mean_class_iou(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """Calculate mean per-class IoU without collapsing semantic labels.

    Args:
        labels1: First ``[H, W]`` semantic class-index image.
        labels2: Second ``[H, W]`` semantic class-index image.

    Returns:
        Mean IoU across every class present in either image.

    Raises:
        ValueError: If inputs are not equally shaped class-index images.
    """
    if labels1.shape != labels2.shape or labels1.ndim != 2:
        raise ValueError(f"Expected equally shaped 2-D labels, got {labels1.shape} and {labels2.shape}")
    classes = np.union1d(labels1, labels2)
    return float(np.mean([calculate_iou(labels1 == class_id, labels2 == class_id) for class_id in classes]))


def compare_segmentation_outputs(trt_dir: Path, torch_dir: Path) -> tuple[float | None, float | None]:
    """Compare archived TensorRT and PyTorch semantic-label outputs.

    Args:
        trt_dir: Directory containing the TensorRT semantic archive.
        torch_dir: Directory containing the PyTorch semantic archive.

    Returns:
        Mean class IoU and mean maximum label difference, or ``(None, None)``
        when no frames are available.
    """
    trt_archives = list(Path(trt_dir).glob("*.aux.sseg.zarr.itar"))
    torch_archives = list(Path(torch_dir).glob("*.aux.sseg.zarr.itar"))
    if len(trt_archives) != 1 or len(torch_archives) != 1:
        raise AssertionError(
            f"Expected one semantic archive per backend, got TRT={len(trt_archives)}, PyTorch={len(torch_archives)}"
        )

    iou_scores = []
    max_diffs = []
    camera_id = "camera_front_wide_120fov"
    trt_loader = CamAuxDataLoader(trt_archives[0])
    torch_loader = CamAuxDataLoader(torch_archives[0])
    trt_timestamps = set(trt_loader._camera_group(camera_id).keys())
    torch_timestamps = set(torch_loader._camera_group(camera_id).keys())
    assert trt_timestamps == torch_timestamps

    for timestamp in sorted(trt_timestamps):
        trt_img = np.asarray(trt_loader.get_semantic_segmentation(camera_id, int(timestamp)))
        torch_img = np.asarray(torch_loader.get_semantic_segmentation(camera_id, int(timestamp)))

        # Calculate maximum difference
        max_diff = np.abs(trt_img.astype(np.float32) - torch_img.astype(np.float32)).max()
        max_diffs.append(max_diff)

        iou_scores.append(calculate_mean_class_iou(trt_img, torch_img))

    # Print results
    if iou_scores:
        mean_iou = float(np.mean(iou_scores))
        mean_max_diff = float(np.mean(max_diffs))
        print("\nResults:")
        print(f"Average IoU: {mean_iou:.4f}")
        print(f"Average Max Difference: {mean_max_diff:.4f}")
        return mean_iou, mean_max_diff
    else:
        print("No valid comparisons made.")
        return None, None


def test_calculate_mean_class_iou_detects_nonzero_class_swaps() -> None:
    first = np.array([[1, 2]], dtype=np.uint8)
    second = np.array([[2, 1]], dtype=np.uint8)

    assert calculate_mean_class_iou(first, second) == 0.0


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


def test_trt_torch_model_comparison(small_dataset_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Integration test to compare TensorRT and PyTorch model outputs"""

    output_root = tmp_path_factory.mktemp("trt_torch_comparison_test")
    trt_output = output_root / "trt_output"
    torch_output = output_root / "torch_output"
    os.makedirs(trt_output, exist_ok=True)
    os.makedirs(torch_output, exist_ok=True)

    camera_id = "camera_front_wide_120fov"  # Use the same camera as in aux_gen_test.py

    print("\nRunning with TensorRT model...")
    trt_result = CliRunner().invoke(
        cli,
        [
            f"--dataset-path={small_dataset_path}",
            f"--camera-id={camera_id}",
            f"--output-dir={trt_output}",
            "--store-meta",
            "offset",
            "--sequence-duration-sec=0.4",  # restrict to few frames only to terminate more quickly
        ],
        catch_exceptions=False,
    )
    assert trt_result.exit_code == 0, f"TensorRT run failed with exit code {trt_result.exit_code}"

    print("\nRunning with PyTorch...")
    torch_result = CliRunner().invoke(
        cli,
        [
            f"--dataset-path={small_dataset_path}",
            f"--camera-id={camera_id}",
            f"--output-dir={torch_output}",
            "--disable-trt",
            "--store-meta",
            "offset",
            "--sequence-duration-sec=0.4",  # restrict to few frames only to terminate more quickly
        ],
        catch_exceptions=False,
    )
    assert torch_result.exit_code == 0, f"PyTorch run failed with exit code {torch_result.exit_code}"

    mean_iou, _ = compare_segmentation_outputs(torch_output, trt_output)

    assert mean_iou is not None, "No segmentation frames were compared"
    assert mean_iou > 0.80, "Mean per-class IoU is too low"
