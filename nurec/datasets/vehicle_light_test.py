# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import logging
import zipfile

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch

from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType
from ncore.sensors import camera_model_from_parameters
from nurec.datasets import vehicle_light as vehicle_light_module
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.datasets.vehicle_light import (
    _CameraFrameCache,
    _count_tracks_overlapping_interval,
    _flatten_ray_intersections,
    _FrameKey,
    _load_class_pixel_mask,
    _load_masked_pixel_idxs,
    _load_rle_class_pixel_idxs,
    _read_pixel_mask_rle,
    _rle_to_mask_image,
    _select_frontmost_intersection_rows,
    convert_pixel_masks_to_rle,
    encode_pixel_mask_rle,
    get_masked_camera_track_point_clouds,
    load_source_indexed_pixel_mask_paths,
    save_pixel_mask_rle,
    shared_vehicle_light_frame_cache,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(
    path: Path,
    sampled_frames: list[dict],
    *,
    camera_ids: list[str] | None = None,
    source_dataset_path: Path | None = None,
    mask_format: object = None,
) -> None:
    for frame in sampled_frames:
        frame.setdefault("active_label_ids", [255])
    identity_sha256 = _canonical_sha256(sampled_frames)
    source_provenance: dict[str, object] = {
        "camera_ids": camera_ids or sorted({str(frame["camera_id"]) for frame in sampled_frames}),
        "sampled_frames": sampled_frames,
        "sampled_frame_identity_count": len(sampled_frames),
        "sampled_frame_identity_sha256": identity_sha256,
    }
    if source_dataset_path is not None:
        source_provenance["dataset"] = {
            "filename": source_dataset_path.name,
            "size_bytes": source_dataset_path.stat().st_size,
            "sha256": hashlib.sha256(source_dataset_path.read_bytes()).hexdigest(),
        }
    label_schema: dict[str, object] = {"classes": [{"label_id": 255, "name": "vehicle_light", "outputs": ["semantic"]}]}
    if mask_format is not None:
        label_schema["mask_format"] = mask_format
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "label_schema": label_schema,
                "source_provenance": source_provenance,
            }
        ),
        encoding="utf-8",
    )


def _manifest_fixture(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    mask_root = tmp_path / "masks"
    camera_root = mask_root / "camera_front"
    camera_root.mkdir(parents=True)
    (camera_root / "000003.png").touch()
    sampled_frames = [
        {
            "camera_id": "camera_front",
            "source_frame_index": 3,
            "source_frame_file": "000003.jpg",
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, sampled_frames)
    return mask_root, manifest_path, sampled_frames


@pytest.mark.parametrize(
    "field,value,message",
    [("schema_version", version, "schema_version must be 4") for version in (1, 2, 3)]
    + [("label_schema", {"classes": []}, "one semantic vehicle_light")]
    + [
        (
            "label_schema",
            {"classes": [{"label_id": label_id, "name": "vehicle_light", "outputs": ["semantic"]}]},
            "one semantic vehicle_light",
        )
        for label_id in (0, 256, True)
    ],
)
def test_source_index_manifest_requires_valid_schema_and_light_class(tmp_path: Path, field, value, message) -> None:
    mask_root, manifest_path, _ = _manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])


@pytest.mark.parametrize("light_id,empty", [(255, False), (7, False), (7, True)])
def test_source_index_manifest_maps_exact_camera_frames(tmp_path: Path, light_id: int, empty: bool) -> None:
    mask_root, manifest_path, original = _manifest_fixture(tmp_path)
    frames = (
        []
        if empty
        else [
            {**original[0], "active_label_ids": [light_id]},
            {**original[0], "source_frame_index": 4, "source_frame_file": "000004.jpg", "active_label_ids": [2]},
            {**original[0], "source_frame_index": 5, "source_frame_file": "000005.jpg", "active_label_ids": [3]},
        ]
    )
    if not empty:
        (mask_root / "camera_front" / "000005.png").touch()
    _write_manifest(manifest_path, frames)
    manifest = json.loads(manifest_path.read_text())
    manifest["label_schema"]["classes"] = [
        {"label_id": light_id, "name": "vehicle_light", "outputs": ["semantic"]},
        {"label_id": 2, "name": "cone", "outputs": ["instances"]},
        {"label_id": 3, "name": "pole", "outputs": ["semantic"]},
    ]
    manifest_path.write_text(json.dumps(manifest))
    paths = load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])
    assert paths == {"camera_front": {} if empty else {3: mask_root / "camera_front" / "000003.png"}}


@pytest.mark.parametrize("camera_ids", [["camera_front"], ["camera_front", "camera_front_fisheye"]])
def test_source_index_manifest_supports_flat_single_sampled_camera(tmp_path: Path, camera_ids: list[str]) -> None:
    """An unsampled RGB camera must not turn a valid flat SAM3 layout into a multi-camera layout."""
    mask_root, manifest_path, _ = _manifest_fixture(tmp_path)
    camera_root = mask_root / "camera_front"
    (camera_root / "000003.png").rename(mask_root / "000003.png")
    camera_root.rmdir()

    paths = load_source_indexed_pixel_mask_paths(mask_root, manifest_path, camera_ids)

    assert paths["camera_front"] == {3: mask_root / "000003.png"}
    assert all(not paths[camera_id] for camera_id in camera_ids if camera_id != "camera_front")


@pytest.mark.parametrize("camera_ids", [["camera_front"], ["camera_front", "camera_front_fisheye"]])
def test_source_index_manifest_requires_subdirectories_for_multiple_sampled_cameras(
    tmp_path: Path, camera_ids: list[str]
) -> None:
    """Selecting one camera from a multi-camera artifact must not allow ambiguous flat masks."""
    mask_root, manifest_path, sampled_frames = _manifest_fixture(tmp_path)
    camera_root = mask_root / "camera_front"
    (camera_root / "000003.png").rename(mask_root / "000003.png")
    camera_root.rmdir()
    _write_manifest(manifest_path, [*sampled_frames, {**sampled_frames[0], "camera_id": "camera_rear"}])

    with pytest.raises(FileNotFoundError, match="Multi-camera pixel masks require a subdirectory for camera_front"):
        load_source_indexed_pixel_mask_paths(mask_root, manifest_path, camera_ids)


@pytest.mark.parametrize(
    "camera_ids",
    [
        ["camera_front", "camera_front_fisheye"],
        ["camera_front_fisheye", "camera_front", "camera_left_fisheye"],
        ["camera_front_fisheye", "camera_left_fisheye"],
    ],
)
def test_source_index_manifest_skips_unsampled_cameras(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, camera_ids: list[str]
) -> None:
    mask_root, manifest_path, _ = _manifest_fixture(tmp_path)

    paths = load_source_indexed_pixel_mask_paths(mask_root, manifest_path, camera_ids)

    assert paths == {
        camera_id: {3: mask_root / "camera_front" / "000003.png"} if camera_id == "camera_front" else {}
        for camera_id in camera_ids
    }
    assert "vehicle-light initialization only" in caplog.text
    assert "RGB supervision is unaffected" in caplog.text


@pytest.mark.parametrize("failure", ["missing_mask", "missing_directory", "extra_mask"])
def test_source_index_manifest_still_validates_masks_with_unsampled_cameras(tmp_path: Path, failure: str) -> None:
    mask_root, manifest_path, _ = _manifest_fixture(tmp_path)
    camera_root = mask_root / "camera_front"
    if failure == "extra_mask":
        (camera_root / "000004.png").touch()
    else:
        (camera_root / "000003.png").unlink()
        if failure == "missing_directory":
            camera_root.rmdir()

    with pytest.raises(FileNotFoundError, match="camera_front"):
        load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front", "camera_front_fisheye"])


def test_source_index_manifest_rejects_declared_camera_without_samples(tmp_path: Path) -> None:
    mask_root, manifest_path, sampled_frames = _manifest_fixture(tmp_path)
    _write_manifest(manifest_path, sampled_frames, camera_ids=["camera_front", "camera_rear"])

    with pytest.raises(ValueError, match="must match sampled-frame cameras"):
        load_source_indexed_pixel_mask_paths(
            mask_root, manifest_path, ["camera_front", "camera_rear", "camera_front_fisheye"]
        )


def test_source_index_manifest_rejects_filename_index_mismatch(tmp_path: Path) -> None:
    mask_root = tmp_path / "masks"
    camera_root = mask_root / "camera_front"
    camera_root.mkdir(parents=True)
    (camera_root / "000003.png").touch()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [{"camera_id": "camera_front", "source_frame_index": 3, "source_frame_file": "000004.jpg"}],
    )

    with pytest.raises(ValueError, match="filename/index mismatch"):
        load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        (
            "sampled_frame_identity_count",
            "source_provenance sampled-frame count mismatch",
        ),
        (
            "sampled_frame_identity_sha256",
            "invalid source_provenance sampled_frame_identity_sha256",
        ),
    ],
)
def test_source_index_manifest_requires_identity_fields(
    tmp_path: Path,
    field_name: str,
    message: str,
) -> None:
    mask_root, manifest_path, _ = _manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["source_provenance"][field_name]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])


def test_source_index_manifest_rejects_identity_hash_mismatch(tmp_path: Path) -> None:
    mask_root, manifest_path, _ = _manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_provenance"]["sampled_frame_identity_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source_provenance sampled_frame_identity_sha256 mismatch"):
        load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])


@pytest.mark.parametrize("camera_ids", [["camera_rear"], ["camera_front", "camera_rear"]])
def test_source_index_manifest_requires_sampled_camera_provenance(tmp_path: Path, camera_ids) -> None:
    mask_root, manifest_path, sampled_frames = _manifest_fixture(tmp_path)
    _write_manifest(manifest_path, sampled_frames, camera_ids=camera_ids)

    with pytest.raises(ValueError, match="must match sampled-frame cameras"):
        load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])


def test_source_index_manifest_validates_local_dataset_provenance(tmp_path: Path) -> None:
    mask_root, manifest_path, sampled_frames = _manifest_fixture(tmp_path)
    dataset_path = tmp_path / "clip.json"
    dataset_path.write_text('{"clip": 1}', encoding="utf-8")
    _write_manifest(manifest_path, sampled_frames, source_dataset_path=dataset_path)

    load_source_indexed_pixel_mask_paths(
        mask_root,
        manifest_path,
        ["camera_front"],
        source_dataset_path=dataset_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_provenance"]["dataset"]["size_bytes"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset provenance mismatch"):
        load_source_indexed_pixel_mask_paths(
            mask_root,
            manifest_path,
            ["camera_front"],
            source_dataset_path=dataset_path,
        )


def test_source_index_manifest_rejects_missing_source_dataset(tmp_path: Path) -> None:
    mask_root, manifest_path, _ = _manifest_fixture(tmp_path)
    missing_dataset = tmp_path / "missing.ncore"

    with pytest.raises(FileNotFoundError, match="missing.ncore"):
        load_source_indexed_pixel_mask_paths(
            mask_root,
            manifest_path,
            ["camera_front"],
            source_dataset_path=missing_dataset,
        )


@pytest.mark.parametrize(
    ("mask_format", "suffix"),
    [(None, ".png"), ("png", ".png"), ("rle", ".rle.npz")],
    ids=["field absent", "png", "rle"],
)
def test_source_index_manifest_selects_the_mask_format(
    tmp_path: Path,
    mask_format: str | None,
    suffix: str,
) -> None:
    mask_root, manifest_path, sampled_frames = _manifest_fixture(tmp_path)
    # Both formats sit side by side after a conversion, and each check only sees its own files.
    (mask_root / "camera_front" / "000003.rle.npz").touch()
    _write_manifest(manifest_path, sampled_frames, mask_format=mask_format)

    paths = load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])

    assert paths == {"camera_front": {3: mask_root / "camera_front" / f"000003{suffix}"}}


@pytest.mark.parametrize("mask_format", ["jpeg", ["rle"], 1])
def test_source_index_manifest_rejects_unknown_mask_formats(tmp_path: Path, mask_format: object) -> None:
    mask_root, manifest_path, sampled_frames = _manifest_fixture(tmp_path)
    _write_manifest(manifest_path, sampled_frames, mask_format=mask_format)

    with pytest.raises(ValueError, match="mask_format must be one of"):
        load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])


def test_rle_manifest_requires_the_exact_rle_file_set(tmp_path: Path) -> None:
    mask_root, manifest_path, sampled_frames = _manifest_fixture(tmp_path)
    camera_root = mask_root / "camera_front"
    _write_manifest(manifest_path, sampled_frames, mask_format="rle")

    # The PNG is there, but an RLE manifest must not fall back to it.
    with pytest.raises(FileNotFoundError, match=r"missing=\['000003.rle.npz'\] extra=\[\]"):
        load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])

    (camera_root / "000003.rle.npz").touch()
    (camera_root / "000004.rle.npz").touch()
    with pytest.raises(FileNotFoundError, match=r"missing=\[\] extra=\['000004.rle.npz'\]"):
        load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])


def test_source_index_manifest_rejects_same_size_dataset_content_change(tmp_path: Path) -> None:
    mask_root, manifest_path, sampled_frames = _manifest_fixture(tmp_path)
    dataset_path = tmp_path / "clip.json"
    dataset_path.write_text('{"clip": 1}', encoding="utf-8")
    _write_manifest(manifest_path, sampled_frames, source_dataset_path=dataset_path)

    dataset_path.write_text('{"clip": 2}', encoding="utf-8")

    with pytest.raises(ValueError, match="dataset provenance mismatch"):
        load_source_indexed_pixel_mask_paths(
            mask_root,
            manifest_path,
            ["camera_front"],
            source_dataset_path=dataset_path,
        )


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.zeros((2, 2, 3), dtype=np.uint8), "single-channel"),
        (np.zeros((2, 2), dtype=np.uint16), "uint8 storage"),
        (np.zeros((2, 3), dtype=np.uint8), "aspect ratio"),
    ],
)
def test_class_mask_loader_rejects_ambiguous_encodings(
    tmp_path: Path,
    image: np.ndarray,
    message: str,
) -> None:
    mask_path = tmp_path / "mask.png"
    assert cv2.imwrite(str(mask_path), image)

    with pytest.raises(ValueError, match=message):
        _load_class_pixel_mask(
            mask_path,
            frame_width=2,
            frame_height=2,
            camera_id="camera_front",
            label_id=7,
        )


@pytest.mark.parametrize("value,expected", [(0, False), (255, True), (17, False)])
def test_class_mask_loader_accepts_uniform_maps(tmp_path, value, expected):
    """Uniform class maps preserve requested-label selection and boolean storage."""
    mask_path = tmp_path / "mask.png"
    assert cv2.imwrite(str(mask_path), np.full((2, 2), value, dtype=np.uint8))
    mask = _load_class_pixel_mask(mask_path, frame_width=2, frame_height=2, camera_id="camera_front", label_id=255)
    assert mask.dtype == np.bool_
    assert bool(mask.all()) is expected
    assert bool(mask.any()) is expected


def test_class_mask_loader_selects_only_requested_label_when_resizing(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    image = np.array([[0, 7], [2, 0]], dtype=np.uint8)
    assert cv2.imwrite(str(mask_path), image)

    mask = _load_class_pixel_mask(
        mask_path,
        frame_width=4,
        frame_height=4,
        camera_id="camera_front",
        label_id=7,
    )

    assert mask.shape == (4, 4)
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, np.repeat(np.repeat(image == 7, 2, axis=0), 2, axis=1))


def _write_png_and_rle(directory: Path, image: np.ndarray) -> tuple[Path, Path]:
    png_path = directory / "000000.png"
    rle_path = directory / "000000.rle.npz"
    assert cv2.imwrite(str(png_path), image)
    save_pixel_mask_rle(rle_path, image)
    return png_path, rle_path


@pytest.mark.parametrize(
    ("lit_pixels", "expected_runs"),
    [
        ([], []),
        (list(range(12)), [[0, 12]]),
        ([3, 4, 5], [[3, 3]]),
        ([11], [[11, 1]]),
    ],
    ids=["empty", "full", "run crossing a row", "last pixel"],
)
def test_rle_mask_matches_the_png_loader(
    tmp_path: Path,
    lit_pixels: list[int],
    expected_runs: list[list[int]],
) -> None:
    image = np.zeros(12, dtype=np.uint8)
    image[lit_pixels] = 255
    image = image.reshape(3, 4)
    png_path, rle_path = _write_png_and_rle(tmp_path, image)
    frame: dict[str, Any] = {"frame_width": 4, "frame_height": 3, "camera_id": "camera_front", "label_id": 255}

    rle_idxs = _load_rle_class_pixel_idxs(rle_path, **frame)

    assert encode_pixel_mask_rle(image)["runs"].tolist() == expected_runs
    assert rle_idxs.dtype == np.int64
    assert rle_idxs.tolist() == lit_pixels
    np.testing.assert_array_equal(rle_idxs, np.flatnonzero(_load_class_pixel_mask(png_path, **frame)))
    # The frame loop's entry point returns the same tensor for both formats.
    cpu = torch.device("cpu")
    torch.testing.assert_close(
        _load_masked_pixel_idxs(rle_path, device=cpu, **frame),
        _load_masked_pixel_idxs(png_path, device=cpu, **frame),
        rtol=0,
        atol=0,
    )


def test_rle_encoding_is_lossless_for_class_maps(tmp_path: Path) -> None:
    """Touching runs with different labels stay separate."""
    image = np.array([[0, 1, 1, 2], [2, 2, 0, 3], [3, 0, 0, 7]], dtype=np.uint8)
    rle_path = tmp_path / "000000.rle.npz"
    save_pixel_mask_rle(rle_path, image)

    rle = _read_pixel_mask_rle(rle_path)

    assert rle.shape == (3, 4)
    assert np.stack((rle.starts, rle.lengths), axis=1).tolist() == [[1, 2], [3, 3], [7, 2], [11, 1]]
    assert rle.values.tolist() == [1, 2, 3, 7]
    np.testing.assert_array_equal(_rle_to_mask_image(rle), image)


@pytest.mark.parametrize(
    ("label_id", "expected"),
    [(1, [1, 2]), (2, [3, 4, 5]), (3, [7, 8]), (7, [11]), (255, [])],
)
def test_rle_class_map_selects_only_the_requested_label(tmp_path: Path, label_id: int, expected: list[int]) -> None:
    image = np.array([[0, 1, 1, 2], [2, 2, 0, 3], [3, 0, 0, 7]], dtype=np.uint8)
    png_path, rle_path = _write_png_and_rle(tmp_path, image)
    frame: dict[str, Any] = {"frame_width": 4, "frame_height": 3, "camera_id": "camera_front", "label_id": label_id}

    rle_idxs = _load_rle_class_pixel_idxs(rle_path, **frame)

    assert rle_idxs.tolist() == expected
    np.testing.assert_array_equal(rle_idxs, np.flatnonzero(_load_class_pixel_mask(png_path, **frame)))


def test_rle_mask_resizes_like_the_png_loader(tmp_path: Path) -> None:
    png_path, rle_path = _write_png_and_rle(tmp_path, np.array([[0, 7], [2, 0]], dtype=np.uint8))
    frame: dict[str, Any] = {"frame_width": 4, "frame_height": 4, "camera_id": "camera_front", "label_id": 7}

    rle_idxs = _load_rle_class_pixel_idxs(rle_path, **frame)

    assert rle_idxs.tolist() == np.flatnonzero(_load_class_pixel_mask(png_path, **frame)).tolist()
    assert rle_idxs.tolist() == [2, 3, 6, 7]
    with pytest.raises(ValueError, match="incompatible aspect ratio"):
        _load_rle_class_pixel_idxs(rle_path, frame_width=4, frame_height=3, camera_id="camera_front", label_id=7)


_VALID_RLE_ARRAYS: dict[str, np.ndarray] = {
    "shape": np.array([3, 4], dtype=np.uint32),
    "runs": np.array([[1, 2], [5, 3]], dtype=np.uint32),
    "values": np.array([255, 255], dtype=np.uint8),
}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"runs": np.array([[1, 0], [5, 3]], dtype=np.uint32)}, "zero-length runs"),
        ({"runs": np.array([[1, 5], [5, 3]], dtype=np.uint32)}, "sorted by start and must not overlap"),
        ({"runs": np.array([[5, 3], [1, 2]], dtype=np.uint32)}, "sorted by start and must not overlap"),
        ({"runs": np.array([[1, 2], [10, 3]], dtype=np.uint32)}, r"end at pixel 13, outside its 3x4 shape"),
        ({"runs": np.array([[1, 2], [5, 3]], dtype=np.int64)}, r"runs must be uint32 \[R, 2\]"),
        ({"runs": np.array([[1, 2, 0], [5, 3, 0]], dtype=np.uint32)}, r"runs must be uint32 \[R, 2\]"),
        ({"values": np.array([255], dtype=np.uint8)}, r"values must be uint8 \[2\]"),
        ({"values": np.array([255, 255], dtype=np.int64)}, r"values must be uint8 \[2\]"),
        ({"values": np.array([255, 0], dtype=np.uint8)}, "value 0 must be left implicit"),
        ({"shape": np.array([0, 4], dtype=np.uint32)}, "empty shape"),
        ({"shape": np.array([3, 4, 1], dtype=np.uint32)}, r"shape must be uint32 \[2\]"),
        ({"shape": np.array([3, 4], dtype=np.int64)}, r"shape must be uint32 \[2\]"),
        ({"shape": np.array([1 << 15, (1 << 15) + 1], dtype=np.uint32)}, "is over 1073741824 pixels"),
        ({"values": None}, "must hold exactly"),
        ({"extra": np.zeros(1)}, "must hold exactly"),
    ],
    ids=[
        "zero length",
        "overlap",
        "unsorted",
        "out of bounds",
        "int64 runs",
        "runs width 3",
        "values count",
        "int64 values",
        "unassigned value",
        "empty shape",
        "3D shape",
        "int64 shape",
        "oversized shape",
        "missing values",
        "unknown array",
    ],
)
def test_rle_reader_rejects_corrupt_arrays(tmp_path: Path, overrides: dict, message: str) -> None:
    arrays = {name: array for name, array in {**_VALID_RLE_ARRAYS, **overrides}.items() if array is not None}
    rle_path = tmp_path / "000000.rle.npz"
    with rle_path.open("wb") as stream:
        np.savez(stream, **arrays)

    with pytest.raises(ValueError, match=message):
        _read_pixel_mask_rle(rle_path)


def test_rle_reader_accepts_a_shape_at_the_pixel_cap(tmp_path: Path) -> None:
    rle_path = tmp_path / "000000.rle.npz"
    with rle_path.open("wb") as stream:
        np.savez(stream, **{**_VALID_RLE_ARRAYS, "shape": np.array([1 << 15, 1 << 15], dtype=np.uint32)})

    assert _read_pixel_mask_rle(rle_path).shape == (1 << 15, 1 << 15)


def test_rle_reader_rejects_unreadable_files(tmp_path: Path) -> None:
    rle_path = tmp_path / "000000.rle.npz"
    with rle_path.open("wb") as stream:
        np.save(stream, _VALID_RLE_ARRAYS["runs"])
    with pytest.raises(ValueError, match="must be an .npz archive"):
        _read_pixel_mask_rle(rle_path)

    save_pixel_mask_rle(rle_path, np.full((3, 4), 255, dtype=np.uint8), overwrite=True)
    rle_path.write_bytes(rle_path.read_bytes()[:100])
    with pytest.raises(ValueError, match="not a readable .npz archive"):
        _read_pixel_mask_rle(rle_path)

    rle_path.write_bytes(b"not a mask")
    with pytest.raises(ValueError, match="not a readable .npz archive"):
        _read_pixel_mask_rle(rle_path)

    with zipfile.ZipFile(rle_path, "w") as archive:
        archive.writestr("shape.npy", b"garbage")
        for name in ("runs", "values"):
            buffer = io.BytesIO()
            np.save(buffer, _VALID_RLE_ARRAYS[name])
            archive.writestr(f"{name}.npy", buffer.getvalue())
    with pytest.raises(ValueError, match="shape is not an .npy array"):
        _read_pixel_mask_rle(rle_path)

    with pytest.raises(FileNotFoundError, match="Required pixel mask is missing"):
        _read_pixel_mask_rle(tmp_path / "missing.rle.npz")


def test_rle_writer_refuses_to_clobber_and_rejects_non_uint8_images(tmp_path: Path) -> None:
    rle_path = tmp_path / "000000.rle.npz"
    save_pixel_mask_rle(rle_path, np.zeros((2, 2), dtype=np.uint8))

    with pytest.raises(FileExistsError, match="RLE mask already exists"):
        save_pixel_mask_rle(rle_path, np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="needs a 2D uint8 image"):
        encode_pixel_mask_rle(np.zeros((2, 2), dtype=np.bool_))
    with pytest.raises(ValueError, match="must end with .rle.npz"):
        save_pixel_mask_rle(tmp_path / "000000.npz", np.zeros((2, 2), dtype=np.uint8))


def test_rle_encoder_rejects_masks_over_the_reader_pixel_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vehicle_light_module, "_MAX_RLE_MASK_PIXELS", 12)
    assert encode_pixel_mask_rle(np.full((3, 4), 255, dtype=np.uint8))["shape"].tolist() == [3, 4]

    monkeypatch.setattr(vehicle_light_module, "_MAX_RLE_MASK_PIXELS", 11)
    with pytest.raises(ValueError, match="is empty or over 11 pixels"):
        encode_pixel_mask_rle(np.full((3, 4), 255, dtype=np.uint8))


def test_rle_writer_removes_its_temporary_file_when_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rle_path = tmp_path / "000000.rle.npz"

    def failing_savez(stream, **arrays) -> None:
        stream.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(vehicle_light_module.np, "savez", failing_savez)
    with pytest.raises(OSError, match="disk full"):
        save_pixel_mask_rle(rle_path, np.full((3, 4), 255, dtype=np.uint8))
    assert not rle_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_converter_writes_rle_for_every_semantic_frame_and_marks_the_manifest(tmp_path: Path) -> None:
    mask_root, manifest_path, sampled_frames = _manifest_fixture(tmp_path)
    camera_root = mask_root / "camera_front"
    # Frame 5 holds only another semantic class, but its mask is still part of the file set the loader checks.
    frames = [
        {**sampled_frames[0], "active_label_ids": [255, 3]},
        {**sampled_frames[0], "source_frame_index": 5, "source_frame_file": "000005.jpg", "active_label_ids": [3]},
    ]
    _write_manifest(manifest_path, frames)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["label_schema"]["classes"].append({"label_id": 3, "name": "pole", "outputs": ["semantic"]})
    manifest["label_schema"]["path_template"] = "masks/{camera_id}/{source_frame_index:06d}.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    light_image = np.zeros((3, 4), dtype=np.uint8)
    light_image[1, 2:] = 255
    light_image[0, 0] = 3
    assert cv2.imwrite(str(camera_root / "000003.png"), light_image)
    assert cv2.imwrite(str(camera_root / "000005.png"), np.full((3, 4), 3, dtype=np.uint8))

    assert convert_pixel_masks_to_rle(mask_root, manifest_path) == 2

    # Only mask_format is added, so the identity hash still validates and path_template still names the kept PNGs.
    converted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert converted["label_schema"].pop("mask_format") == "rle"
    assert converted == manifest
    assert (camera_root / "000003.png").is_file()
    assert (camera_root / "000005.rle.npz").is_file()
    paths = load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])
    rle_path = camera_root / "000003.rle.npz"
    assert paths == {"camera_front": {3: rle_path}}
    frame: dict[str, Any] = {"frame_width": 4, "frame_height": 3, "camera_id": "camera_front"}
    assert _load_rle_class_pixel_idxs(rle_path, label_id=255, **frame).tolist() == [6, 7]
    assert _load_rle_class_pixel_idxs(rle_path, label_id=3, **frame).tolist() == [0]
    with pytest.raises(ValueError, match="Only PNG SAM3 mask manifests"):
        convert_pixel_masks_to_rle(mask_root, manifest_path)


def test_converter_rejects_stray_rle_files_and_keeps_the_png_manifest(tmp_path: Path) -> None:
    mask_root, manifest_path, _ = _manifest_fixture(tmp_path)
    camera_root = mask_root / "camera_front"
    assert cv2.imwrite(str(camera_root / "000003.png"), np.full((3, 4), 255, dtype=np.uint8))
    # Left behind by an earlier conversion of a larger frame set, the PNG check never sees it.
    (camera_root / "000009.rle.npz").touch()
    before = manifest_path.read_bytes()

    with pytest.raises(FileNotFoundError, match=r"missing=\[\] extra=\['000009.rle.npz'\]"):
        convert_pixel_masks_to_rle(mask_root, manifest_path)

    assert manifest_path.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))
    paths = load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])
    assert paths == {"camera_front": {3: camera_root / "000003.png"}}


def test_converter_rejects_rle_that_differs_from_the_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mask_root, manifest_path, _ = _manifest_fixture(tmp_path)
    assert cv2.imwrite(str(mask_root / "camera_front" / "000003.png"), np.full((3, 4), 255, dtype=np.uint8))
    before = manifest_path.read_bytes()

    def encode_without_runs(image: np.ndarray) -> dict[str, np.ndarray]:
        return {**encode_pixel_mask_rle(image), "runs": np.zeros((0, 2), np.uint32), "values": np.zeros(0, np.uint8)}

    monkeypatch.setattr(vehicle_light_module, "encode_pixel_mask_rle", encode_without_runs)
    with pytest.raises(RuntimeError, match="does not reproduce"):
        convert_pixel_masks_to_rle(mask_root, manifest_path)
    assert manifest_path.read_bytes() == before


def test_converter_removes_the_temporary_manifest_on_any_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mask_root, manifest_path, _ = _manifest_fixture(tmp_path)
    assert cv2.imwrite(str(mask_root / "camera_front" / "000003.png"), np.full((3, 4), 255, dtype=np.uint8))
    before = manifest_path.read_bytes()
    load_paths = vehicle_light_module._load_semantic_mask_paths

    def interrupted(root: Path, path: Path, camera_ids: list[str]):
        if path.name.endswith(".tmp"):
            raise RuntimeError("interrupted")
        return load_paths(root, path, camera_ids)

    monkeypatch.setattr(vehicle_light_module, "_load_semantic_mask_paths", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        convert_pixel_masks_to_rle(mask_root, manifest_path)
    assert manifest_path.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_converter_failing_partway_keeps_the_png_manifest_and_retries_with_overwrite(tmp_path: Path) -> None:
    mask_root, manifest_path, sampled_frames = _manifest_fixture(tmp_path)
    camera_root = mask_root / "camera_front"
    second_frame = {**sampled_frames[0], "source_frame_index": 5, "source_frame_file": "000005.jpg"}
    _write_manifest(manifest_path, [*sampled_frames, second_frame])
    assert cv2.imwrite(str(camera_root / "000003.png"), np.full((3, 4), 255, dtype=np.uint8))
    # The 3-channel second mask fails after the first frame's RLE file is already written.
    assert cv2.imwrite(str(camera_root / "000005.png"), np.full((3, 4, 3), 255, dtype=np.uint8))
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="must be single-channel"):
        convert_pixel_masks_to_rle(mask_root, manifest_path)

    assert manifest_path.read_bytes() == before
    paths = load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])
    assert paths == {"camera_front": {3: camera_root / "000003.png", 5: camera_root / "000005.png"}}

    assert cv2.imwrite(str(camera_root / "000005.png"), np.full((3, 4), 255, dtype=np.uint8))
    with pytest.raises(FileExistsError, match="000003.rle.npz"):
        convert_pixel_masks_to_rle(mask_root, manifest_path)
    assert convert_pixel_masks_to_rle(mask_root, manifest_path, overwrite=True) == 2
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["label_schema"]["mask_format"] == "rle"


@pytest.mark.parametrize(
    ("interval_start_us", "interval_end_us", "expected_count"),
    [
        (0, 9, 0),
        (10, 10, 1),
        (19, 20, 2),
        (25, 35, 2),
        (41, 50, 0),
    ],
)
def test_count_tracks_overlapping_interval_is_inclusive(
    interval_start_us: int,
    interval_end_us: int,
    expected_count: int,
) -> None:
    count = _count_tracks_overlapping_interval(
        torch.tensor([10, 20, 30]),
        torch.tensor([19, 29, 40]),
        interval_start_us,
        interval_end_us,
    )

    assert count == expected_count


def test_frontmost_ray_intersection_selects_one_visible_track_per_ray() -> None:
    intersection_ts = torch.tensor(
        [
            [8.0, 12.0],
            [3.0, 5.0],
            [4.0, 6.0],
            [2.0, 3.0],
        ]
    )
    intersection_idxs = torch.tensor(
        [
            [0, 4],
            [0, 2],
            [1, 7],
            [3, 1],
        ],
        dtype=torch.int32,
    )

    selected = _select_frontmost_intersection_rows(intersection_ts, intersection_idxs)

    assert selected.tolist() == [1, 2, 3]


def test_flatten_ray_intersections_uses_only_reported_dense_slots() -> None:
    intersection_ts, intersection_idxs = _flatten_ray_intersections(
        torch.tensor([2, 0, 1], dtype=torch.int32),
        torch.tensor([[4, 2], [-1, -1], [7, -1]], dtype=torch.int32),
        torch.tensor(
            [
                [[8.0, 12.0], [3.0, 5.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[4.0, 6.0], [0.0, 0.0]],
            ]
        ),
    )

    assert intersection_ts.tolist() == [[8.0, 12.0], [3.0, 5.0], [4.0, 6.0]]
    assert intersection_idxs.tolist() == [[0, 4], [0, 2], [2, 7]]


def test_frontmost_ray_intersection_breaks_equal_distance_ties_by_track_index() -> None:
    intersection_ts = torch.tensor([[3.0, 5.0], [3.0, 4.0], [-1.0, 2.0]])
    intersection_idxs = torch.tensor([[0, 8], [0, 3], [1, 5]], dtype=torch.int32)

    selected = _select_frontmost_intersection_rows(intersection_ts, intersection_idxs)

    assert selected.tolist() == [1, 2]


def test_frontmost_ray_intersection_handles_no_hits() -> None:
    selected = _select_frontmost_intersection_rows(
        torch.empty((0, 2)),
        torch.empty((0, 2), dtype=torch.int32),
    )

    assert selected.shape == (0,)
    assert selected.dtype == torch.long


def test_masked_initializer_rejects_cpu_device() -> None:
    observations = get_masked_camera_track_point_clouds(
        Mock(),
        ["camera_front"],
        Mock(),
        pixel_mask_dir="unused",
        pixel_mask_manifest_path="unused",
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="requires a CUDA device, got cpu"):
        next(observations)


def test_masked_initializer_without_sampled_cameras_does_not_initialize_datasource(tmp_path: Path) -> None:
    mask_root, manifest_path, sampled_frames = _manifest_fixture(tmp_path)
    dataset_path = tmp_path / "clip.ncore"
    dataset_path.touch()
    _write_manifest(manifest_path, sampled_frames, source_dataset_path=dataset_path)
    camera_ids = ["camera_front_fisheye", "camera_left_fisheye"]
    datasource = Mock(camera_ids=camera_ids)
    datasource.get_path.return_value = dataset_path
    tracks = Mock()

    observations = list(
        get_masked_camera_track_point_clouds(
            datasource,
            camera_ids,
            tracks,
            pixel_mask_dir=str(mask_root),
            pixel_mask_manifest_path=str(manifest_path),
        )
    )

    assert observations == []
    assert camera_ids == ["camera_front_fisheye", "camera_left_fisheye"]
    assert datasource.camera_ids is camera_ids
    datasource._maybe_init_worker.assert_not_called()
    tracks.to_device.assert_not_called()


def _cuda_scene_fixture(tmp_path: Path, device: torch.device) -> dict:
    """Build a four-pixel single-camera scene for the CUDA initializer tests."""
    mask_root, manifest_path, sampled_frames = _manifest_fixture(tmp_path)
    assert cv2.imwrite(str(mask_root / "camera_front" / "000003.png"), np.full((2, 2), 255, dtype=np.uint8))
    dataset_path = tmp_path / "clip.ncore"
    dataset_path.touch()
    _write_manifest(manifest_path, sampled_frames, source_dataset_path=dataset_path)

    camera_params = OpenCVPinholeCameraModelParameters(
        resolution=np.array([2, 2], dtype=np.uint64),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
        principal_point=np.array([1, 1], dtype=np.float32),
        focal_length=np.array([10, 10], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )
    camera_sensor = Mock(model_parameters=camera_params)
    camera_sensor.frames_timestamps_us = np.array([[10, 20]] * 4, dtype=np.int64)
    camera_sensor.get_frames_T_sensor_target.return_value = torch.eye(4, device=device).repeat(2, 1, 1)
    colors = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    camera_sensor.get_frame_image_array.return_value = colors
    datasource = Mock(
        camera_unique_ids={"camera_front": Mock(id="front")},
        camera_sensors={"camera_front": camera_sensor},
        camera_models={"camera_front": camera_model_from_parameters(camera_params)},
        cameras_all_pixels={"camera_front": np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)},
        cameras_all_footprints={"camera_front": np.full(4, 0.01, dtype=np.float32)},
        camera_frame_ranges={"camera_front": [3]},
    )
    datasource.get_path.return_value = dataset_path
    datasource.world_to_nurec.transform_poses.side_effect = lambda poses: poses

    # The nearest cuboid is inactive. Both active tracks bracket the exposure
    # without having a keyframe inside it; all four rays hit both of them.
    poses = [np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0) for _ in range(3)]
    for track_poses, depth in zip(poses, [2, 5, 10], strict=True):
        track_poses[:, 2, 3] = depth
    tracks = CuboidTracks.Factory.from_numpy(
        tracks_id=["inactive", "front", "behind"],
        tracks_poses=poses,
        tracks_timestamps_us=[np.array(interval, dtype=np.int64) for interval in ([40, 50], [0, 30], [0, 30])],
        tracks_label_class=["car"] * 3,
        tracks_flags=[TrackFlags.NONE] * 3,
        cuboids_dims=[np.full(3, 2, dtype=np.float32)] * 3,
        device=torch.device("cpu"),
    )

    return {
        "datasource": datasource,
        "camera_sensor": camera_sensor,
        "tracks": tracks,
        "mask_root": mask_root,
        "manifest_path": manifest_path,
        "colors": colors,
    }


def _run_cuda_initializer(scene: dict, device: torch.device) -> list:
    return list(
        get_masked_camera_track_point_clouds(
            scene["datasource"],
            ["camera_front"],
            scene["tracks"],
            pixel_mask_dir=str(scene["mask_root"]),
            pixel_mask_manifest_path=str(scene["manifest_path"]),
            cuboid_clamp_distance_m=0,
            device=device,
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the CUDA intersection binding")
def test_masked_initializer_assigns_multiple_pixels_through_cuda(tmp_path: Path) -> None:
    """Exercise track transfer, camera rays, the CUDA binding, and frontmost assignment."""
    device = torch.device("cuda")
    scene = _cuda_scene_fixture(tmp_path, device)

    observations = _run_cuda_initializer(scene, device)

    assert len(observations) == 1
    observation = observations[0]
    assert observation.track_id == "front"
    points = observation.point_cloud
    assert points.xyz_end.shape == (4, 3)
    assert torch.isfinite(points.xyz_end).all()
    assert (points.xyz_end.abs() <= 0.99).all()
    torch.testing.assert_close(points.xyz_end[:, 2], torch.full((4,), -0.99, device=device))
    torch.testing.assert_close(points.color, torch.as_tensor(scene["colors"].reshape(4, 3), device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the CUDA intersection binding")
@pytest.mark.parametrize("layout", ["per_camera", "flat"])
def test_masked_initializer_ignores_unsampled_cameras_without_changing_points(tmp_path: Path, layout: str) -> None:
    """Both supported mask layouts preserve covered-camera points and the RGB camera list."""
    device = torch.device("cuda")
    scene = _cuda_scene_fixture(tmp_path, device)
    if layout == "flat":
        camera_root = scene["mask_root"] / "camera_front"
        (camera_root / "000003.png").rename(scene["mask_root"] / "000003.png")
        camera_root.rmdir()
    expected = _run_cuda_initializer(scene, device)
    camera_ids = ["camera_front_fisheye", "camera_front", "camera_left_fisheye"]
    # Only camera_front has sensor metadata in this fixture. Accessing an unsampled camera would fail.
    scene["datasource"].camera_ids = camera_ids

    actual = list(
        get_masked_camera_track_point_clouds(
            scene["datasource"],
            camera_ids,
            scene["tracks"],
            pixel_mask_dir=str(scene["mask_root"]),
            pixel_mask_manifest_path=str(scene["manifest_path"]),
            cuboid_clamp_distance_m=0,
            device=device,
        )
    )

    assert camera_ids == ["camera_front_fisheye", "camera_front", "camera_left_fisheye"]
    assert scene["datasource"].camera_ids is camera_ids
    assert [o.track_id for o in actual] == [o.track_id for o in expected] == ["front"]
    for got, want in zip(actual, expected, strict=True):
        assert got.track_pose_index == want.track_pose_index
        for name in ("xyz_start", "xyz_end", "color", "camera_footprint_scale"):
            torch.testing.assert_close(getattr(got.point_cloud, name), getattr(want.point_cloud, name), rtol=0, atol=0)


def _alternate_track_set() -> CuboidTracks:
    """The same cuboids with only the far one active, so a different track wins."""
    poses = [np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0) for _ in range(3)]
    for track_poses, depth in zip(poses, [2, 5, 10], strict=True):
        track_poses[:, 2, 3] = depth
    return CuboidTracks.Factory.from_numpy(
        tracks_id=["inactive", "front", "behind"],
        tracks_poses=poses,
        tracks_timestamps_us=[np.array(interval, dtype=np.int64) for interval in ([40, 50], [40, 50], [0, 30])],
        tracks_label_class=["car"] * 3,
        tracks_flags=[TrackFlags.NONE] * 3,
        cuboids_dims=[np.full(3, 2, dtype=np.float32)] * 3,
        device=torch.device("cpu"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the CUDA intersection binding")
def test_shared_cache_keeps_two_track_sets_independent(tmp_path: Path) -> None:
    """The two vehicle-light layers differ only in their tracks, so no entry may carry them."""
    device = torch.device("cuda")
    scene = _cuda_scene_fixture(tmp_path, device)
    other_scene = {**scene, "tracks": _alternate_track_set()}
    camera_sensor = scene["camera_sensor"]

    uncached = _run_cuda_initializer(other_scene, device)

    with shared_vehicle_light_frame_cache():
        first = _run_cuda_initializer(scene, device)
        decodes_after_first = camera_sensor.get_frame_image_array.call_count
        cached = _run_cuda_initializer(other_scene, device)

    # Without this the test would also pass if nothing were ever cached.
    assert camera_sensor.get_frame_image_array.call_count == decodes_after_first
    assert [o.track_id for o in first] == ["front"]
    assert [o.track_id for o in cached] == [o.track_id for o in uncached] == ["behind"]
    for got, want in zip(cached, uncached, strict=True):
        for name in ("xyz_start", "xyz_end", "color", "camera_footprint_scale"):
            torch.testing.assert_close(getattr(got.point_cloud, name), getattr(want.point_cloud, name), rtol=0, atol=0)
        assert got.track_pose_index == want.track_pose_index


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the CUDA intersection binding")
def test_shared_cache_keeps_two_datasources_independent(tmp_path: Path) -> None:
    """Datasources that share masks but not camera poses must not share rays."""
    device = torch.device("cuda")
    scene = _cuda_scene_fixture(tmp_path, device)
    shifted = {
        **_cuda_scene_fixture(tmp_path / "shifted", device),
        "mask_root": scene["mask_root"],
        "manifest_path": scene["manifest_path"],
    }
    shifted_poses = torch.eye(4, device=device).repeat(2, 1, 1)
    shifted_poses[:, 0, 3] = 0.25
    shifted["camera_sensor"].get_frames_T_sensor_target.return_value = shifted_poses

    uncached = _run_cuda_initializer(shifted, device)

    with shared_vehicle_light_frame_cache():
        first = _run_cuda_initializer(scene, device)
        cached = _run_cuda_initializer(shifted, device)
        held = vehicle_light_module._active_frame_cache.get()
        assert held is not None
        held_datasources = set(held.datasources.values())

    # Without this the test would also pass if both datasources produced the same rays.
    assert not torch.equal(first[0].point_cloud.xyz_start, uncached[0].point_cloud.xyz_start)
    assert len(cached) == len(uncached) == 1
    for name in ("xyz_start", "xyz_end", "color", "camera_footprint_scale"):
        torch.testing.assert_close(
            getattr(cached[0].point_cloud, name), getattr(uncached[0].point_cloud, name), rtol=0, atol=0
        )
    assert held_datasources == {scene["datasource"], shifted["datasource"]}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the CUDA intersection binding")
def test_shared_frame_cache_reuses_frame_work_without_changing_results(tmp_path: Path) -> None:
    device = torch.device("cuda")
    scene = _cuda_scene_fixture(tmp_path, device)
    camera_sensor = scene["camera_sensor"]

    with shared_vehicle_light_frame_cache():
        first = _run_cuda_initializer(scene, device)
        decodes_after_first = camera_sensor.get_frame_image_array.call_count
        second = _run_cuda_initializer(scene, device)
        assert camera_sensor.get_frame_image_array.call_count == decodes_after_first

    assert decodes_after_first == 1
    assert len(first) == len(second) == 1
    for field_name in ("xyz_start", "xyz_end", "color", "camera_footprint_scale"):
        torch.testing.assert_close(
            getattr(first[0].point_cloud, field_name),
            getattr(second[0].point_cloud, field_name),
            rtol=0,
            atol=0,
        )
    assert first[0].track_pose_index == second[0].track_pose_index


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the CUDA intersection binding")
def test_frame_work_repeats_without_a_cache_scope(tmp_path: Path) -> None:
    device = torch.device("cuda")
    scene = _cuda_scene_fixture(tmp_path, device)

    _run_cuda_initializer(scene, device)
    _run_cuda_initializer(scene, device)

    assert scene["camera_sensor"].get_frame_image_array.call_count == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the CUDA intersection binding")
@pytest.mark.parametrize("lit_pixels", [[0, 1, 3], []], ids=["three pixels", "empty"])
def test_rle_masks_yield_the_same_point_clouds_as_png_masks(
    tmp_path: Path,
    lit_pixels: list[int],
    caplog: pytest.LogCaptureFixture,
) -> None:
    device = torch.device("cuda")
    scene = _cuda_scene_fixture(tmp_path, device)
    image = np.zeros(4, dtype=np.uint8)
    # Pixel 2 holds another class, which neither format may load as a vehicle light.
    image[2] = 3
    image[lit_pixels] = 255
    assert cv2.imwrite(str(scene["mask_root"] / "camera_front" / "000003.png"), image.reshape(2, 2))

    def own_logs() -> list[str]:
        # caplog sits on the root logger, so drop records from other libraries.
        return [record.getMessage() for record in caplog.records if record.name == vehicle_light_module.__name__]

    with caplog.at_level(logging.INFO, logger=vehicle_light_module.__name__):
        from_png = _run_cuda_initializer(scene, device)
        png_logs = own_logs()
        caplog.clear()
        assert convert_pixel_masks_to_rle(scene["mask_root"], scene["manifest_path"]) == 1
        from_rle = _run_cuda_initializer(scene, device)
        rle_logs = own_logs()

    # The summary line carries total_mask_pixels, so equal logs mean equal pixel totals and empty-mask skips.
    assert rle_logs == png_logs
    assert any(f"/{len(lit_pixels)} SAM3" in message for message in png_logs) == bool(lit_pixels)
    assert len(from_rle) == len(from_png) == (1 if lit_pixels else 0)
    for got, want in zip(from_rle, from_png, strict=True):
        assert got.track_id == want.track_id
        assert got.track_pose_index == want.track_pose_index
        for name in ("xyz_start", "xyz_end", "color", "camera_footprint_scale"):
            torch.testing.assert_close(getattr(got.point_cloud, name), getattr(want.point_cloud, name), rtol=0, atol=0)


def test_cache_scope_is_released_when_the_body_raises() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with shared_vehicle_light_frame_cache(byte_budget=1024):
            assert vehicle_light_module._active_frame_cache.get() is not None
            raise RuntimeError("boom")

    assert vehicle_light_module._active_frame_cache.get() is None


def test_nested_cache_scope_leaves_the_outer_cache_in_place() -> None:
    with shared_vehicle_light_frame_cache(byte_budget=1024):
        outer = vehicle_light_module._active_frame_cache.get()
        with shared_vehicle_light_frame_cache(byte_budget=1):
            assert vehicle_light_module._active_frame_cache.get() is outer
        assert vehicle_light_module._active_frame_cache.get() is outer

    assert vehicle_light_module._active_frame_cache.get() is None


def test_cache_scope_empties_entries_so_a_live_generator_cannot_hold_them() -> None:
    held: _CameraFrameCache | None = None
    with shared_vehicle_light_frame_cache(byte_budget=1024):
        held = vehicle_light_module._active_frame_cache.get()
        assert held is not None
        held.put_pixel_idxs(_FrameKey(0, "camera_front", 3, "mask.png", "cpu"), torch.arange(4))
        assert held.cached_bytes > 0

    assert held.pixel_idxs == {}
    assert held.cached_bytes == 0
    held.put_pixel_idxs(_FrameKey(0, "camera_front", 4, "mask.png", "cpu"), torch.arange(4))
    assert held.pixel_idxs == {}
    assert held.cached_bytes == 0


def test_cache_scope_without_cuda_defaults_to_a_zero_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with shared_vehicle_light_frame_cache():
        cache = vehicle_light_module._active_frame_cache.get()
        assert cache is not None
        cache.put_pixel_idxs(_FrameKey(0, "camera_front", 3, "mask.png", "cpu"), torch.arange(4))
        assert cache.byte_budget == 0
        assert cache.cached_bytes == 0


def test_cache_budget_is_measured_when_the_cache_is_first_used(monkeypatch: pytest.MonkeyPatch) -> None:
    free_bytes = [10_000]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (free_bytes[0], 20_000))
    with shared_vehicle_light_frame_cache():
        cache = vehicle_light_module._active_frame_cache.get()
        assert cache is not None
        # Earlier layers take most of the device before the vehicle-light layers start caching.
        free_bytes[0] = 400
        cache.put_pixel_idxs(_FrameKey(0, "camera_front", 3, "mask.png", "cpu"), torch.arange(4))
        assert cache.byte_budget == 40


def test_zero_budget_cache_refuses_every_entry() -> None:
    cache = _CameraFrameCache(byte_budget=0)
    key = _FrameKey(0, "camera_front", 3, "mask.png", "cpu")

    cache.put_pixel_idxs(key, torch.arange(4))
    cache.put_pixel_idxs(key, torch.empty(0, dtype=torch.long))

    assert cache.get_pixel_idxs(key) is None
    assert cache.cached_bytes == 0
    assert cache.over_budget == 2


def test_over_budget_entries_do_not_block_smaller_later_ones() -> None:
    cache = _CameraFrameCache(byte_budget=64)
    small = _FrameKey(0, "camera_front", 1, "a.png", "cpu")
    large = _FrameKey(0, "camera_front", 2, "b.png", "cpu")
    later = _FrameKey(0, "camera_front", 3, "c.png", "cpu")

    cache.put_pixel_idxs(small, torch.arange(4))
    cache.put_pixel_idxs(large, torch.arange(64))
    cache.put_pixel_idxs(later, torch.arange(2))

    assert cache.get_pixel_idxs(small) is not None
    assert cache.get_pixel_idxs(large) is None
    assert cache.get_pixel_idxs(later) is not None
    assert cache.over_budget == 1


def test_frame_cache_separates_class_labels():
    cache = _CameraFrameCache(byte_budget=64)
    key = _FrameKey(0, "camera", 0, "labels.png", "cpu")
    cache.put_pixel_idxs(key, torch.tensor([1]))
    assert cache.get_pixel_idxs(key).tolist() == [1]
    assert cache.get_pixel_idxs(key._replace(label_id=2)) is None
