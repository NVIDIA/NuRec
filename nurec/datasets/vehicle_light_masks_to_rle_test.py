# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

from pathlib import Path

import cv2
import numpy as np

from click.testing import CliRunner, Result

from nurec.datasets.vehicle_light_masks_to_rle import vehicle_light_masks_to_rle


def _write_clip(tmp_path: Path) -> tuple[Path, Path]:
    """Write one camera with one PNG mask and its schema-4 manifest."""
    mask_root = tmp_path / "masks"
    camera_root = mask_root / "camera_front"
    camera_root.mkdir(parents=True)
    image = np.zeros((3, 4), dtype=np.uint8)
    image[1, 2:] = 255
    assert cv2.imwrite(str(camera_root / "000003.png"), image)
    sampled_frames = [
        {
            "camera_id": "camera_front",
            "source_frame_index": 3,
            "source_frame_file": "000003.jpg",
            "active_label_ids": [255],
        }
    ]
    identity_sha256 = hashlib.sha256(
        json.dumps(sampled_frames, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "label_schema": {"classes": [{"label_id": 255, "name": "vehicle_light", "outputs": ["semantic"]}]},
                "source_provenance": {
                    "camera_ids": ["camera_front"],
                    "sampled_frames": sampled_frames,
                    "sampled_frame_identity_count": len(sampled_frames),
                    "sampled_frame_identity_sha256": identity_sha256,
                },
            }
        ),
        encoding="utf-8",
    )
    return mask_root, manifest_path


def _convert(mask_root: Path, manifest_path: Path, *extra_args: str) -> Result:
    return CliRunner().invoke(
        vehicle_light_masks_to_rle,
        ["--mask-dir", str(mask_root), "--manifest", str(manifest_path), *extra_args],
    )


def _mask_format(manifest_path: Path) -> str | None:
    return json.loads(manifest_path.read_text(encoding="utf-8"))["label_schema"].get("mask_format")


def test_cli_converts_the_masks_and_switches_the_manifest(tmp_path: Path) -> None:
    mask_root, manifest_path = _write_clip(tmp_path)

    result = _convert(mask_root, manifest_path)

    assert result.exit_code == 0, result.output
    assert "converted 1 masks" in result.output
    assert _mask_format(manifest_path) == "rle"
    assert (mask_root / "camera_front" / "000003.rle.npz").is_file()


def test_cli_reports_an_already_converted_manifest_without_a_traceback(tmp_path: Path) -> None:
    mask_root, manifest_path = _write_clip(tmp_path)
    assert _convert(mask_root, manifest_path).exit_code == 0

    result = _convert(mask_root, manifest_path)

    assert result.exit_code == 1
    assert "can be converted to RLE" in result.output
    # ClickException exits through SystemExit, any other exception would mean a traceback.
    assert isinstance(result.exception, SystemExit)


def test_cli_replaces_a_leftover_rle_file_only_with_overwrite(tmp_path: Path) -> None:
    mask_root, manifest_path = _write_clip(tmp_path)
    leftover = mask_root / "camera_front" / "000003.rle.npz"
    leftover.write_bytes(b"left behind by an interrupted run")

    result = _convert(mask_root, manifest_path)

    assert result.exit_code == 1
    assert f"RLE mask already exists: {leftover}, rerun with --overwrite" in result.output
    assert isinstance(result.exception, SystemExit)
    assert _mask_format(manifest_path) is None

    result = _convert(mask_root, manifest_path, "--overwrite")

    assert result.exit_code == 0, result.output
    assert _mask_format(manifest_path) == "rle"
