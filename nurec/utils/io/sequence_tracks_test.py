# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import zipfile

from pathlib import Path

import lietorch as lt
import numpy as np
import point_cloud_utils as pcu
import torch

from nurec.artifact.artifact import Artifact
from nurec.datasets.tracks import CuboidTracks, TracksData
from nurec.utils.cuboid import cuboid_corners_local
from nurec.utils.io.sequence_tracks import (
    generate_track_colors,
    save_cuboids_ply,
    serialize_sequence_tracks,
)
from nurec.utils.types import CuboidTracksData, TrackFlags


def create_test_cuboid_tracks_with_suffixes() -> CuboidTracks:
    """Create a test CuboidTracks object with track IDs that have @suffixes."""
    # Create test data with track IDs that have @suffixes
    tracks_id = [
        "15@scene:obstacles:autolabels:v2",
        "349@scene:obstacles:autolabels:v2",
        "42@scene:obstacles:autolabels:v2",
        "100@scene:obstacles:autolabels:v2",
    ]

    n_tracks = len(tracks_id)
    n_poses_per_track = 3

    # Create poses data (tquat format: [tx, ty, tz, qx, qy, qz, qw])
    tracks_poses_tquat = torch.zeros(n_tracks * n_poses_per_track, 7, dtype=torch.float32)
    tracks_timestamps_us = torch.zeros(n_tracks * n_poses_per_track, dtype=torch.int64)
    tracks_packinfo = torch.zeros(n_tracks, 2, dtype=torch.int64)

    # Fill in the data
    for i, track_id in enumerate(tracks_id):
        start_idx = i * n_poses_per_track
        end_idx = start_idx + n_poses_per_track

        # Set poses (identity poses for simplicity)
        tracks_poses_tquat[start_idx:end_idx, 6] = 1.0  # qw = 1 for identity quaternion

        # Set timestamps (increasing over time)
        tracks_timestamps_us[start_idx:end_idx] = torch.arange(i * 1000, i * 1000 + n_poses_per_track * 100, 100)

        # Set packinfo
        tracks_packinfo[i, 0] = start_idx
        tracks_packinfo[i, 1] = n_poses_per_track

    # Convert to SE3 format
    tracks_poses = lt.SE3(tracks_poses_tquat)

    # Create dimensions (cuboid sizes)
    cuboids_dims = torch.ones(n_tracks, 3, dtype=torch.float32) * 2.0  # 2x2x2 meter cubes

    tracks_data = TracksData(
        tracks_id=tracks_id,
        tracks_poses=tracks_poses,
        tracks_timestamps_us=tracks_timestamps_us,
        tracks_packinfo=tracks_packinfo,
        max_track_n_poses=n_poses_per_track,
        tracks_label_class=["obstacle"] * n_tracks,
        tracks_flags=torch.zeros(n_tracks, dtype=torch.int32),
        max_positive_timestamp_gap_us=torch.full((n_tracks,), 100, dtype=torch.int64),
    )

    cuboidtracks_data = CuboidTracksData(cuboids_dims=cuboids_dims)

    return CuboidTracks(tracks_data=tracks_data, cuboidtracks_data=cuboidtracks_data)


def test_sequence_tracks_json_track_id_cleaning() -> None:
    """Test that JSON serialization cleans track IDs by removing @suffixes."""
    # Create test cuboid tracks with @suffixes
    cuboid_tracks = create_test_cuboid_tracks_with_suffixes()

    # Serialize to JSON format
    sequence_tracks_artifacts = serialize_sequence_tracks(
        sequence_id="test_sequence",
        cuboid_tracks=cuboid_tracks,
        excluded_tracks=None,
        filename="sequence_tracks",
        formats=["json"],
        usd_timestamp_offset_us=0,
    )

    # Parse the JSON to check cleaned track IDs
    json_artifact = sequence_tracks_artifacts[0]
    assert hasattr(json_artifact, "serialized"), "Expected NamedSerialized artifact"
    sequence_tracks_data = json.loads(json_artifact.serialized)
    cleaned_track_ids = sequence_tracks_data["test_sequence"]["tracks_data"]["tracks_id"]

    # Verify all track IDs are cleaned (no @suffixes)
    expected_cleaned_ids = ["15", "349", "42", "100"]

    assert len(cleaned_track_ids) == len(expected_cleaned_ids), (
        f"Expected {len(expected_cleaned_ids)} track IDs, got {len(cleaned_track_ids)}"
    )

    for i, (actual_id, expected_id) in enumerate(zip(cleaned_track_ids, expected_cleaned_ids)):
        assert actual_id == expected_id, f"Track ID {i}: expected '{expected_id}', got '{actual_id}'"
        assert "@" not in actual_id, f"Track ID '{actual_id}' still contains '@' suffix"


def test_sequence_tracks_usda_track_id_cleaning() -> None:
    """Test that USDA serialization cleans track IDs by removing @suffixes."""
    # Create test cuboid tracks with @suffixes
    cuboid_tracks = create_test_cuboid_tracks_with_suffixes()

    # Serialize to USDA format
    sequence_tracks_artifacts = serialize_sequence_tracks(
        sequence_id="test_sequence",
        cuboid_tracks=cuboid_tracks,
        excluded_tracks=None,
        filename="sequence_tracks",
        formats=["usda"],
        usd_timestamp_offset_us=0,
    )

    # Get the USDA content
    usda_artifact = sequence_tracks_artifacts[0]
    assert hasattr(usda_artifact, "stage"), "Expected NamedUSDStage artifact"
    usda_content = usda_artifact.stage.ExportToString()

    # Check that the USDA content doesn't contain @suffixes
    for original_id in ["15@scene:obstacles:autolabels:v2", "349@scene:obstacles:autolabels:v2"]:
        assert original_id not in usda_content, (
            f"USDA file still contains original track ID with @suffix: {original_id}"
        )

    # Check that cleaned IDs are present
    for cleaned_id in ["15", "349", "42", "100"]:
        assert cleaned_id in usda_content, f"USDA file missing cleaned track ID: {cleaned_id}"


def test_usdz_track_id_cleaning() -> None:
    """Test that USDZ serialized items have cleaned track IDs (no @suffixes)."""
    # Create test cuboid tracks with @suffixes
    cuboid_tracks = create_test_cuboid_tracks_with_suffixes()

    # Serialize to USDZ format (which includes sequence_tracks.json)
    sequence_tracks_artifacts = serialize_sequence_tracks(
        sequence_id="test_sequence",
        cuboid_tracks=cuboid_tracks,
        excluded_tracks=None,
        filename="sequence_tracks",
        formats=["json", "usda"],
        usd_timestamp_offset_us=0,
    )

    # Create a temporary USDZ file
    with tempfile.NamedTemporaryFile(suffix=".usdz", delete=False) as tmp_file:
        usdz_path = Path(tmp_file.name)

    try:
        # Create a simple USDZ by zipping the artifacts
        with zipfile.ZipFile(usdz_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
            for artifact in sequence_tracks_artifacts:
                if hasattr(artifact, "serialized") and artifact.serialized:
                    # Add JSON file
                    zip_file.writestr(artifact.filename, artifact.serialized)
                elif hasattr(artifact, "stage") and artifact.stage:
                    # Add USDA file
                    usda_content = artifact.stage.ExportToString()
                    zip_file.writestr(artifact.filename, usda_content)

        # Load the USDZ artifact
        usdz_artifact = Artifact(usdz_path)

        # Get the sequence tracks from the USDZ
        sequence_tracks = usdz_artifact.sequence_tracks

        # Verify the structure
        assert "test_sequence" in sequence_tracks, "Expected 'test_sequence' key in sequence tracks"
        tracks_data = sequence_tracks["test_sequence"]["tracks_data"]
        assert "tracks_id" in tracks_data, "Expected 'tracks_id' in tracks_data"

        # Get the track IDs from the USDZ
        track_ids_from_usdz = tracks_data["tracks_id"]

        # Verify that all track IDs are cleaned (no @suffixes)
        expected_cleaned_ids = ["15", "349", "42", "100"]

        assert len(track_ids_from_usdz) == len(expected_cleaned_ids), (
            f"Expected {len(expected_cleaned_ids)} track IDs, got {len(track_ids_from_usdz)}"
        )

        for i, (actual_id, expected_id) in enumerate(zip(track_ids_from_usdz, expected_cleaned_ids)):
            assert actual_id == expected_id, f"Track ID {i}: expected '{expected_id}', got '{actual_id}'"
            assert "@" not in actual_id, f"Track ID '{actual_id}' still contains '@' suffix"

        # Also verify the USDA file if it exists
        with zipfile.ZipFile(usdz_path, "r") as zip_file:
            if "sequence_tracks.usda" in zip_file.namelist():
                usda_content = zip_file.read("sequence_tracks.usda").decode("utf-8")

                # Check that the USDA content doesn't contain @suffixes
                for original_id in ["15@scene:obstacles:autolabels:v2", "349@scene:obstacles:autolabels:v2"]:
                    assert original_id not in usda_content, (
                        f"USDA file still contains original track ID with @suffix: {original_id}"
                    )

                # Check that cleaned IDs are present
                for cleaned_id in ["15", "349", "42", "100"]:
                    assert cleaned_id in usda_content, f"USDA file missing cleaned track ID: {cleaned_id}"

    finally:
        # Clean up the temporary file
        usdz_path.unlink()


def test_original_vs_cleaned_track_ids() -> None:
    """Test that we can verify the difference between original and cleaned track IDs."""
    # Create test data
    cuboid_tracks = create_test_cuboid_tracks_with_suffixes()

    # Get original track IDs
    original_track_ids = cuboid_tracks.tracks_id

    # Serialize and check what gets written to JSON
    sequence_tracks_artifacts = serialize_sequence_tracks(
        sequence_id="test_sequence",
        cuboid_tracks=cuboid_tracks,
        excluded_tracks=None,
        filename="sequence_tracks",
        formats=["json"],
        usd_timestamp_offset_us=0,
    )

    # Parse the JSON to see the cleaned track IDs
    json_artifact = sequence_tracks_artifacts[0]
    assert hasattr(json_artifact, "serialized"), "Expected NamedSerialized artifact"
    sequence_tracks_data = json.loads(json_artifact.serialized)
    cleaned_track_ids = sequence_tracks_data["test_sequence"]["tracks_data"]["tracks_id"]

    # Verify cleaning worked
    for original, cleaned in zip(original_track_ids, cleaned_track_ids):
        expected_cleaned = original.split("@")[0]
        assert cleaned == expected_cleaned, f"Expected '{expected_cleaned}', got '{cleaned}' for original '{original}'"


def test_generate_track_colors_deterministic_and_shape() -> None:
    ids = ["track_a", "track_b"]
    colors1 = generate_track_colors(ids)
    colors2 = generate_track_colors(ids)
    np.testing.assert_array_equal(colors1["track_a"], colors2["track_a"])
    np.testing.assert_array_equal(colors1["track_b"], colors2["track_b"])
    assert colors1["track_a"].shape == (3,)
    assert colors1["track_a"].dtype == np.uint8


def _make_cuboid_tracks_simple(n_tracks: int = 1, dynamic: bool = True) -> CuboidTracks:
    """Build n_tracks cuboid tracks, each with 2 poses at the same position.

    Track i has both poses at (i, 0, 0) — tracks are spaced 1 m apart along x, but each
    individual track is stationary (both poses share the same world position). All tracks
    have dims [2.0, 1.0, 1.5] m and are flagged DYNAMIC or not based on the `dynamic` arg.
    No other flags are set.
    """
    cpu = torch.device("cpu")
    flags = TrackFlags.DYNAMIC if dynamic else TrackFlags(0)
    tracks_id = [f"track_{i}" for i in range(n_tracks)]
    poses = []
    for i in range(n_tracks):
        T = np.eye(4, dtype=np.float32)[np.newaxis].repeat(2, axis=0)
        T[:, 0, 3] = float(i)
        poses.append(T)
    timestamps = [np.array([0, 1], dtype=np.int64) for _ in range(n_tracks)]
    label_class = ["vehicle"] * n_tracks
    flags_list = [flags] * n_tracks
    dims = [np.array([2.0, 1.0, 1.5], dtype=np.float32) for _ in range(n_tracks)]
    return CuboidTracks.Factory.from_numpy(tracks_id, poses, timestamps, label_class, flags_list, dims, device=cpu)


def test_save_cuboids_ply_writes_file() -> None:
    # 2 tracks × 2 poses × 8 corners = 32 vertices; × 12 triangles = 48 faces.
    n_tracks, n_poses = 2, 2
    cuboid_tracks = _make_cuboid_tracks_simple(n_tracks=n_tracks, dynamic=True)
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "cuboids.ply")
        save_cuboids_ply(cuboid_tracks, out)
        assert Path(out).is_file()
        loaded = pcu.TriangleMesh()
        loaded.load(out)
        assert len(loaded.vertex_data.positions) == n_tracks * n_poses * 8
        assert len(loaded.face_data.vertex_ids) == n_tracks * n_poses * 12


def _make_cuboid_tracks_mixed(n_dynamic: int = 1, n_static: int = 1) -> CuboidTracks:
    """Build CuboidTracks with a mix of dynamic and static tracks.

    Produces n_dynamic + n_static tracks, each with 2 identical poses. Tracks are spaced
    10 m apart along the x-axis so their exported vertices are distinguishable. The first
    n_dynamic tracks are flagged TrackFlags.DYNAMIC; the remaining n_static tracks carry no
    flags (TrackFlags(0)). No other flags are set. All tracks use dims [2.0, 1.0, 1.5] m.
    """
    cpu = torch.device("cpu")
    tracks_id = [f"dyn_{i}" for i in range(n_dynamic)] + [f"sta_{i}" for i in range(n_static)]
    flags_list = [TrackFlags.DYNAMIC] * n_dynamic + [TrackFlags(0)] * n_static
    poses = [np.eye(4, dtype=np.float32)[np.newaxis].repeat(2, axis=0) for _ in tracks_id]
    for track_idx, track_poses in enumerate(poses):
        track_poses[:, 0, 3] = float(track_idx * 10)
    timestamps = [np.array([0, 1], dtype=np.int64) for _ in tracks_id]
    label_class = ["vehicle"] * len(tracks_id)
    dims = [np.array([2.0, 1.0, 1.5], dtype=np.float32) for _ in tracks_id]
    return CuboidTracks.Factory.from_numpy(tracks_id, poses, timestamps, label_class, flags_list, dims, device=cpu)


def test_save_cuboids_ply_require_flags_filter() -> None:
    cuboid_tracks = _make_cuboid_tracks_mixed(n_dynamic=1, n_static=1)
    corners = cuboid_corners_local(np.array([2.0, 1.0, 1.5], dtype=np.float32))
    expected_dynamic_vertices = np.tile(corners, (2, 1))
    expected_static_vertices = np.tile(corners + np.array([10.0, 0.0, 0.0], dtype=np.float32), (2, 1))
    with tempfile.TemporaryDirectory() as d:
        out_dynamic = str(Path(d) / "dynamic.ply")
        out_all = str(Path(d) / "all.ply")
        save_cuboids_ply(cuboid_tracks, out_dynamic, require_flags=TrackFlags.DYNAMIC)
        save_cuboids_ply(cuboid_tracks, out_all)

        loaded_dynamic = pcu.TriangleMesh()
        loaded_dynamic.load(out_dynamic)
        # 1 dynamic track x 2 poses per track x 8 corners & 12 faces per pose
        assert len(loaded_dynamic.vertex_data.positions) == 16
        assert len(loaded_dynamic.face_data.vertex_ids) == 24
        np.testing.assert_allclose(loaded_dynamic.vertex_data.positions, expected_dynamic_vertices)

        loaded_all = pcu.TriangleMesh()
        loaded_all.load(out_all)
        # 2 tracks (1 dynamic, 1 static) x 2 poses per track x 8 corners & 12 faces per pose
        assert len(loaded_all.vertex_data.positions) == 32
        assert len(loaded_all.face_data.vertex_ids) == 48
        np.testing.assert_allclose(
            loaded_all.vertex_data.positions,
            np.concatenate([expected_dynamic_vertices, expected_static_vertices]),
        )


def test_save_cuboids_ply_exclude_flags_filter() -> None:
    cuboid_tracks = _make_cuboid_tracks_mixed(n_dynamic=1, n_static=1)
    corners = cuboid_corners_local(np.array([2.0, 1.0, 1.5], dtype=np.float32))
    expected_static_vertices = np.tile(corners + np.array([10.0, 0.0, 0.0], dtype=np.float32), (2, 1))
    with tempfile.TemporaryDirectory() as d:
        out_static = str(Path(d) / "static.ply")
        save_cuboids_ply(cuboid_tracks, out_static, exclude_flags=TrackFlags.DYNAMIC)
        loaded_static = pcu.TriangleMesh()
        loaded_static.load(out_static)
        # 1 static track x 2 poses per track x 8 corners & 12 faces per pose
        assert len(loaded_static.vertex_data.positions) == 16
        assert len(loaded_static.face_data.vertex_ids) == 24
        np.testing.assert_allclose(loaded_static.vertex_data.positions, expected_static_vertices)


def test_save_cuboids_ply_external_colors() -> None:
    cuboid_tracks = _make_cuboid_tracks_simple(n_tracks=1, dynamic=True)
    colors = {cuboid_tracks.tracks_id[0]: np.array([12, 34, 56], dtype=np.uint8)}
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "cuboids.ply")
        save_cuboids_ply(cuboid_tracks, out, colors_per_track=colors)
        loaded = pcu.TriangleMesh()
        loaded.load(out)
        # 1 track x 2 poses per track x 8 corners & 12 faces per pose
        assert len(loaded.vertex_data.positions) == 16
        assert len(loaded.face_data.vertex_ids) == 24
        assert loaded.vertex_data.colors is not None
        expected_colors = np.tile(colors[cuboid_tracks.tracks_id[0]], (16, 1))
        np.testing.assert_array_equal(loaded.vertex_data.colors[:, :3], expected_colors)
