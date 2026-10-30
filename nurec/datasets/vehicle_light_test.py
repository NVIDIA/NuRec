# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch

from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType
from ncore.sensors import camera_model_from_parameters
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.datasets.vehicle_light import (
    _count_tracks_overlapping_interval,
    _flatten_ray_intersections,
    _load_binary_pixel_mask,
    _select_frontmost_intersection_rows,
    get_masked_camera_track_point_clouds,
    load_source_indexed_pixel_mask_paths,
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
) -> None:
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
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "sampled_frame_identity_count": len(sampled_frames),
                "sampled_frame_identity_sha256": identity_sha256,
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


def test_source_index_manifest_requires_schema_version_three(tmp_path: Path) -> None:
    mask_root, manifest_path, _ = _manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version must be 3"):
        load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])


def test_source_index_manifest_maps_exact_camera_frames(tmp_path: Path) -> None:
    mask_root, manifest_path, _ = _manifest_fixture(tmp_path)

    paths = load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])

    assert paths == {"camera_front": {3: mask_root / "camera_front" / "000003.png"}}


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
    ("container_name", "field_name", "message"),
    [
        ("top-level", "sampled_frame_identity_count", "invalid sampled_frame_identity_count"),
        ("top-level", "sampled_frame_identity_sha256", "invalid top-level sampled_frame_identity_sha256"),
        (
            "source_provenance",
            "sampled_frame_identity_count",
            "source_provenance sampled-frame count mismatch",
        ),
        (
            "source_provenance",
            "sampled_frame_identity_sha256",
            "invalid source_provenance sampled_frame_identity_sha256",
        ),
    ],
)
def test_source_index_manifest_requires_identity_fields(
    tmp_path: Path,
    container_name: str,
    field_name: str,
    message: str,
) -> None:
    mask_root, manifest_path, _ = _manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    container = manifest if container_name == "top-level" else manifest["source_provenance"]
    del container[field_name]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])


@pytest.mark.parametrize("hash_owner", ["top-level", "source_provenance"])
def test_source_index_manifest_rejects_identity_hash_mismatch(
    tmp_path: Path,
    hash_owner: str,
) -> None:
    mask_root, manifest_path, _ = _manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hash_owner == "top-level":
        manifest["sampled_frame_identity_sha256"] = "0" * 64
    else:
        manifest["source_provenance"]["sampled_frame_identity_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{hash_owner} sampled_frame_identity_sha256 mismatch"):
        load_source_indexed_pixel_mask_paths(mask_root, manifest_path, ["camera_front"])


def test_source_index_manifest_requires_requested_camera_in_provenance(tmp_path: Path) -> None:
    mask_root, manifest_path, sampled_frames = _manifest_fixture(tmp_path)
    _write_manifest(manifest_path, sampled_frames, camera_ids=["camera_rear"])

    with pytest.raises(ValueError, match="missing requested cameras"):
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
        (np.array([[0, 1], [255, 0]], dtype=np.uint8), "binary values"),
    ],
)
def test_binary_mask_loader_rejects_ambiguous_encodings(
    tmp_path: Path,
    image: np.ndarray,
    message: str,
) -> None:
    mask_path = tmp_path / "mask.png"
    assert cv2.imwrite(str(mask_path), image)

    with pytest.raises(ValueError, match=message):
        _load_binary_pixel_mask(
            mask_path,
            frame_width=2,
            frame_height=2,
            camera_id="camera_front",
        )


def test_binary_mask_loader_preserves_binary_values_when_resizing(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    image = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    assert cv2.imwrite(str(mask_path), image)

    mask = _load_binary_pixel_mask(
        mask_path,
        frame_width=4,
        frame_height=4,
        camera_id="camera_front",
    )

    assert mask.shape == (4, 4)
    assert mask.dtype == np.bool_
    assert set(np.unique(mask)) == {False, True}


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the CUDA intersection binding")
def test_masked_initializer_assigns_multiple_pixels_through_cuda(tmp_path: Path) -> None:
    """Exercise track transfer, camera rays, the CUDA binding, and frontmost assignment."""
    device = torch.device("cuda")
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

    observations = list(
        get_masked_camera_track_point_clouds(
            datasource,
            ["camera_front"],
            tracks,
            pixel_mask_dir=str(mask_root),
            pixel_mask_manifest_path=str(manifest_path),
            cuboid_clamp_distance_m=0,
            device=device,
        )
    )

    assert len(observations) == 1
    observation = observations[0]
    assert observation.track_id == "front"
    points = observation.point_cloud
    assert points.xyz_end.shape == (4, 3)
    assert torch.isfinite(points.xyz_end).all()
    assert (points.xyz_end.abs() <= 0.99).all()
    torch.testing.assert_close(points.xyz_end[:, 2], torch.full((4,), -0.99, device=device))
    torch.testing.assert_close(points.color, torch.as_tensor(colors.reshape(4, 3), device=device))
