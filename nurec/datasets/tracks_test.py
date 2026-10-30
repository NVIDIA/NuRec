# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib

from collections.abc import Iterator

import lietorch as lt
import numpy as np
import pytest
import torch

from scipy.spatial.transform import Rotation

from nurec.datasets.summary import DataSourceSummary
from nurec.datasets.track_split import TrackSplitMetadata, TrackSplitResolver
from nurec.datasets.tracks import CuboidTracks, TrackFlags, Tracks
from nurec.utils.tests import CommonTestCase
from nurec.utils.types import FrameConversion


def _make_cadence_test_tracks(device: torch.device) -> CuboidTracks:
    timestamps = [
        np.array([10, 10, 5, 12, 8, 30], dtype=np.int64),
        np.array([100, 90, 90, 95], dtype=np.int64),
    ]
    return CuboidTracks.Factory.from_numpy(
        tracks_id=["mixed", "decreasing"],
        tracks_poses=[
            np.repeat(np.eye(4, dtype=np.float32)[None], len(track_timestamps), axis=0)
            for track_timestamps in timestamps
        ],
        tracks_timestamps_us=timestamps,
        tracks_label_class=["mixed_label", "decreasing_label"],
        tracks_flags=[TrackFlags.NONE, TrackFlags.NONE],
        cuboids_dims=[np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
        device=device,
    )


def test_track_cadence_uses_maximum_strictly_positive_same_track_gap() -> None:
    tracks = _make_cadence_test_tracks(torch.device("cpu"))

    # Cross-track 30 -> 100 is deliberately larger and must not contribute.
    torch.testing.assert_close(
        tracks.max_positive_timestamp_gap_us,
        torch.tensor([22, 5], dtype=torch.int64),
        rtol=0,
        atol=0,
    )


def test_empty_track_min_distance_mask_does_not_poison_cuda() -> None:
    empty_tracks = CuboidTracks.Factory.empty(device=torch.device("cuda"))

    mask = CuboidTracks.Ops.mask_by_min_distance(empty_tracks, min_distance_m=5.0)

    np.testing.assert_array_equal(mask, np.empty((0,), dtype=bool))
    probe = torch.arange(1, device="cuda")
    torch.cuda.synchronize()
    torch.testing.assert_close(probe, torch.zeros(1, dtype=torch.int64, device="cuda"))


def test_track_cadence_freeze_recomputes_from_replacement_endpoints() -> None:
    tracks = _make_cadence_test_tracks(torch.device("cpu"))
    frozen = CuboidTracks.Ops.freeze(
        tracks,
        min_timestamps_us=torch.tensor([100, 900], dtype=torch.int64),
        max_timestamps_us=torch.tensor([500, 800], dtype=torch.int64),
        tracks_poses_start=tracks.tracks_poses[tracks.tracks_packinfo[:, 0].long()],
    )

    torch.testing.assert_close(
        frozen.max_positive_timestamp_gap_us,
        torch.tensor([400, 0], dtype=torch.int64),
        rtol=0,
        atol=0,
    )


def test_subset_carries_cadence_and_pose_order() -> None:
    device = torch.device("cuda")
    tracks = _make_cadence_test_tracks(device)
    tracks.tracks_poses.data[:, 0] = torch.arange(tracks.tracks_poses.shape[0], dtype=torch.float32, device=device)
    expected_poses = torch.cat((tracks.tracks_poses.data[6:10], tracks.tracks_poses.data[:6])).clone()

    subset = CuboidTracks.Ops.subset_from_indices(tracks, [1, 0])

    torch.testing.assert_close(
        subset.tracks_packinfo,
        torch.tensor([[0, 4], [4, 6]], dtype=torch.int32, device=device),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        subset.tracks_timestamps_us,
        torch.tensor([100, 90, 90, 95, 10, 10, 5, 12, 8, 30], dtype=torch.int64, device=device),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        subset.max_positive_timestamp_gap_us,
        torch.tensor([5, 22], dtype=torch.int64, device=device),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(subset.tracks_poses.data, expected_poses, rtol=0, atol=0)


def test_subset_from_mask_normalizes_tensor_indices_for_metadata() -> None:
    device = torch.device("cuda")
    tracks = _make_cadence_test_tracks(device)

    subset = CuboidTracks.Ops.subset_from_mask(
        tracks,
        torch.tensor([False, True], dtype=torch.bool, device=device),
    )

    assert subset.tracks_id == ["decreasing"]
    assert subset.tracks_label_class == ["decreasing_label"]
    torch.testing.assert_close(
        subset.max_positive_timestamp_gap_us,
        torch.tensor([5], dtype=torch.int64, device=device),
        rtol=0,
        atol=0,
    )


@contextlib.contextmanager
def _float32_matmul_precision(precision: str) -> Iterator[None]:
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _make_world_scale_tracks(device: torch.device) -> tuple[CuboidTracks, np.ndarray]:
    rng = np.random.default_rng(0)
    # World-scale coordinates: TF32's 10-bit mantissa rounds a float32 matmul of these to
    # millimetre-to-centimetre error, while float32 storage alone stays below 1e-4 m.
    positions = rng.uniform(-400.0, 400.0, size=(64, 3))
    matrices = np.repeat(np.eye(4, dtype=np.float32)[None], len(positions), axis=0)
    matrices[:, :3, :3] = Rotation.random(len(positions), rng=rng).as_matrix().astype(np.float32)
    matrices[:, :3, 3] = positions.astype(np.float32)
    tracks = CuboidTracks.Factory.from_numpy(
        tracks_id=["world_scale"],
        tracks_poses=[matrices],
        tracks_timestamps_us=[np.arange(len(positions), dtype=np.int64)],
        tracks_label_class=["vehicle"],
        tracks_flags=[TrackFlags.NONE],
        cuboids_dims=[np.ones(3, dtype=np.float32)],
        device=device,
    )
    return tracks, matrices


def _assert_poses_match(transformed: CuboidTracks, reference: np.ndarray) -> None:
    # The 1e-4 m position bound holds while coordinates stay inside this binade; float32
    # storage resolution alone would breach it past ~2048 m.
    assert np.abs(reference[:, :3, 3]).max() < 1024.0

    # The composition widens to float64, so the poses have to arrive back at storage dtype.
    assert transformed.tracks_poses.dtype == torch.float32

    result = transformed.tracks_poses.matrix().cpu().numpy().astype(np.float64)
    position_error = np.abs(result[:, :3, 3] - reference[:, :3, 3]).max()
    assert position_error < 1e-4, f"pose positions off by {position_error:.6f} m"
    # Rotation entries are O(1); the float32 round trip alone stays near 3e-7.
    rotation_error = np.abs(result[:, :3, :3] - reference[:, :3, :3]).max()
    assert rotation_error < 1e-5, f"pose rotations off by {rotation_error:.8f}"


def test_frame_conversion_keeps_pose_precision_under_tf32() -> None:
    device = torch.device("cuda")
    tracks, matrices = _make_world_scale_tracks(device)
    conversion = FrameConversion.from_origin_scale_axis(
        target_origin=np.array([123.456, -78.9, 4.2], dtype=np.float32),
        # A scale other than 1 makes the cuboid-dims division observable: S's diagonal is
        # 1/target_scale, so unit dims come back scaled by target_scale.
        target_scale=0.5,
        # A permuted axis order puts off-diagonal entries in T, so an error in the rotation
        # half of the composition cannot hide behind a diagonal operand.
        target_axis=[1, 2, 0],
    )

    with _float32_matmul_precision("medium"):
        transformed = CuboidTracks.Ops.transform_with_frame_conversion(tracks, conversion, None)

    T, S = conversion.get_transformation_matrices()
    reference = T.astype(np.float64) @ matrices.astype(np.float64) @ S.astype(np.float64)
    _assert_poses_match(transformed, reference)
    torch.testing.assert_close(
        transformed.cuboids_dims,
        torch.full((1, 3), 0.5, device=device),
    )


def test_local_to_common_transform_keeps_pose_precision_under_tf32() -> None:
    device = torch.device("cuda")
    tracks, matrices = _make_world_scale_tracks(device)
    # A rotation, not just a translation, so the rotation half of this matmul is exercised.
    T_local_np = np.eye(4, dtype=np.float32)
    T_local_np[:3, :3] = Rotation.from_euler("xyz", [20.0, -35.0, 50.0], degrees=True).as_matrix()
    T_local_np[:3, 3] = np.array([250.0, -180.0, 30.0], dtype=np.float32)

    with _float32_matmul_precision("medium"):
        transformed = CuboidTracks.Ops.transform_with_frame_conversion(tracks, None, T_local_np)

    reference = T_local_np.astype(np.float64) @ matrices.astype(np.float64)
    _assert_poses_match(transformed, reference)


def test_clone_override_replaces_poses_and_clones_every_other_field() -> None:
    tracks = _make_cadence_test_tracks(torch.device("cpu"))
    replacement = lt.SE3(tracks.tracks_poses.vec().flip(0).contiguous())

    cloned = CuboidTracks.Ops.clone(tracks, tracks_poses=replacement)

    # The override is handed through as-is; cloning poses the caller overwrites
    # would be paid on every call.
    assert cloned.tracks_poses is replacement
    # Every other field is a copy: equal values in fresh storage.
    for name in (
        "tracks_packinfo",
        "tracks_timestamps_us",
        "tracks_flags",
        "max_positive_timestamp_gap_us",
        "cuboids_dims",
    ):
        source = getattr(tracks, name)
        copied = getattr(cloned, name)
        assert copied is not source, name
        torch.testing.assert_close(copied, source, rtol=0, atol=0)
    assert cloned.tracks_id == tracks.tracks_id
    assert cloned.tracks_id is not tracks.tracks_id
    assert cloned.tracks_label_class == tracks.tracks_label_class
    assert cloned.tracks_label_class is not tracks.tracks_label_class
    assert cloned.max_track_n_poses == tracks.max_track_n_poses


@pytest.mark.parametrize(
    "replacement_factory",
    [
        pytest.param(lambda poses: lt.SE3(poses.data[:-1].clone()), id="shape"),
        pytest.param(lambda poses: lt.SE3(poses.data.to(device="meta")), id="device"),
        pytest.param(lambda poses: lt.SE3(poses.data.double()), id="dtype"),
    ],
)
def test_clone_override_rejects_incompatible_poses(replacement_factory) -> None:
    tracks = _make_cadence_test_tracks(torch.device("cpu"))

    with pytest.raises(ValueError, match="must match the source shape, device, and dtype"):
        CuboidTracks.Ops.clone(tracks, tracks_poses=replacement_factory(tracks.tracks_poses))


class SimpleCuboidTracksTest(CommonTestCase):
    def setUp(self) -> None:
        # Create simple track "scene"

        #           ┌──────┐     ┌──────┐
        #           │      │     │      │
        #           │   ───┼───► │      │
        #           │      │     │      │
        #           └──────┘     └──────┘
        #
        #
        #           ┌──────┐
        #           │      │
        #           │      │
        #           │      │
        #           └──────┘
        #
        #    ▲         ▲      ▲
        #    │   ▲     │      │
        #    │   │     │      │
        #    │   └──►  │      │

        tracks_id: list[str] = []
        tracks_poses: list[np.ndarray] = []
        tracks_timestamps_us: list[np.ndarray] = []
        tracks_label_class: list[str] = []
        tracks_flags: list[TrackFlags] = []
        cuboids_dims: list[np.ndarray] = []

        # 1m cube cuboid static track at (1, 1.5, 0)
        tracks_id.append("static")
        pose0 = np.eye(4, 4, dtype=np.float32)
        pose0[:3, 3] = [1, 1.5, 0]
        poses = np.stack([pose0, pose0])
        tracks_poses.append(poses)
        tracks_timestamps_us.append(np.array([0, 1000000], np.int64))  # at time 0sec -> 1sec
        tracks_label_class.append("automobile")
        tracks_flags.append(TrackFlags.NONE)
        cuboids_dims.append(np.array([1, 1, 1], dtype=np.float32))

        # 1m cube cuboid dynamic track between (1, 3, 0) and (3, 3, 0)
        tracks_id.append("dynamic")
        pose0 = np.eye(4, 4, dtype=np.float32)
        pose0[:3, 3] = [1, 3, 0]
        pose1 = np.eye(4, 4, dtype=np.float32)
        pose1[:3, 3] = [3, 3, 0]
        poses = np.stack([pose0, pose1])
        tracks_poses.append(poses)
        tracks_timestamps_us.append(np.array([0, 1000000], np.int64))  # at time 0sec -> 1sec

        tracks_label_class.append("pedestrian")
        tracks_flags.append(TrackFlags.DYNAMIC)
        cuboids_dims.append(np.array([1, 1, 1], dtype=np.float32))

        self.tracks = CuboidTracks.Factory.from_numpy(
            tracks_id, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, cuboids_dims=cuboids_dims
        )

        # Create three rays intersecting none, two, and one track
        self.rays_o = torch.tensor([[-1, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=torch.float32, device=torch.device("cuda"))
        self.rays_d = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.float32, device=torch.device("cuda"))
        self.rays_timestamps_us = torch.tensor([0, 0, 1000000 // 2], dtype=torch.int64, device=torch.device("cuda"))


class RandomCuboidTracksTest(CommonTestCase):
    def setUp(self) -> None:
        # Create a random track "scene".
        # The tracks are randomly distributed within [-10, 10]^3, with random rotation.
        # Each track is of very large size 10x10x10 to maximize the chance of intersection.
        # The rays are originating from the origin (with a random delta), and pointing randomly.
        # Timestamp is from 0 to 10000.

        tracks_id: list[str] = []
        tracks_poses: list[np.ndarray] = []
        tracks_timestamps_us: list[np.ndarray] = []
        tracks_label_class: list[str] = []
        tracks_flags: list[TrackFlags] = []
        cuboids_dims: list[np.ndarray] = []

        rng = np.random.RandomState(0)

        for idx in range(100):
            tracks_id.append(f"track_{idx}")
            poses = np.repeat(np.eye(4, 4, dtype=np.float32)[None, :, :], 10, axis=0)
            poses[:, :3, :3] = Rotation.random(10, random_state=rng).as_matrix()
            poses[:, :3, 3] = rng.uniform(-10, 10, (10, 3))
            tracks_poses.append(poses)
            tracks_timestamps_us.append(np.linspace(0, 10000, 10, dtype=np.int64))
            tracks_label_class.append("unknown")
            track_flag = TrackFlags.NONE
            for value in TrackFlags.__members__:
                if rng.rand() < 0.5:
                    track_flag |= TrackFlags[value]
            tracks_flags.append(track_flag)
            cuboids_dims.append(np.array([10, 10, 10], dtype=np.float32))

        self.tracks = CuboidTracks.Factory.from_numpy(
            tracks_id, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, cuboids_dims=cuboids_dims
        )

        # Create random rays with origin close to the center of the scene and random direction
        rays_o = rng.uniform(-1, 1, (1000, 3)).astype(np.float32)
        rays_d = rng.randn(1000, 3).astype(np.float32)
        rays_d /= np.linalg.norm(rays_d, axis=1)[:, None]
        rays_timestamps_us = rng.randint(0, 10000, 1000).astype(np.int64)

        self.rays_o = torch.from_numpy(rays_o).cuda()
        self.rays_d = torch.from_numpy(rays_d).cuda()
        self.rays_timestamps_us = torch.from_numpy(rays_timestamps_us).cuda()

        # Create an empty track "scene".
        self.empty_tracks = CuboidTracks.Factory.empty()

    def test_serialization(self):
        def check(value: CuboidTracks | Tracks, ref: CuboidTracks | Tracks):
            self.assertEqual(value.tracks_id, ref.tracks_id)
            self._compareTensor(value.tracks_packinfo, ref.tracks_packinfo)
            self._compareTensor(value.tracks_poses.data, ref.tracks_poses.data)
            self._compareTensor(value.tracks_timestamps_us, ref.tracks_timestamps_us)
            self._compareTensor(value.max_positive_timestamp_gap_us, ref.max_positive_timestamp_gap_us)
            self._compareTensor(value.tracks_flags, ref.tracks_flags)
            self.assertEqual(value.max_track_n_poses, ref.max_track_n_poses)
            self.assertEqual(value.tracks_label_class, ref.tracks_label_class)

            if isinstance(value, CuboidTracks) and isinstance(ref, CuboidTracks):
                self._compareTensor(value.cuboids_dims, ref.cuboids_dims)

        test_cases = [self.tracks, self.empty_tracks]
        for tracks in test_cases:
            with self.subTest(tracks=tracks):
                # Serialize and deserialize CuboidTracks
                cuboidtracks_from_dict = CuboidTracks.from_dict(tracks.to_dict())
                cuboidtracks_from_json = CuboidTracks.from_json(tracks.to_json())

                # Check that the data is the same
                check(cuboidtracks_from_dict, tracks)
                check(cuboidtracks_from_json, tracks)

                # Serialize and deserialize Tracks
                base_tracks = Tracks(tracks_data=tracks.tracks_data)
                tracks_from_dict = Tracks.from_dict(base_tracks.to_dict())
                tracks_from_json = Tracks.from_json(base_tracks.to_json())

                # Check that the data is the same
                check(tracks_from_dict, base_tracks)
                check(tracks_from_json, base_tracks)


class BinarySearchIntersectionTest(CommonTestCase):
    def setUp(self):
        # Create a test track with known timestamps for testing binary search behavior
        tracks_id = ["test_track"]
        tracks_poses = [np.eye(4, dtype=np.float32)[None, :, :].repeat(4, axis=0)]
        tracks_timestamps_us = [np.array([100000, 200000, 300000, 400000], dtype=np.int64)]
        tracks_label_class = ["test"]
        tracks_flags = [TrackFlags.NONE]
        cuboids_dims = [np.array([1.0, 1.0, 1.0], dtype=np.float32)]

        self.tracks = CuboidTracks.Factory.from_numpy(
            tracks_id, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, cuboids_dims
        )

        # Create test rays that will intersect the cuboid at different timestamps
        self.rays_o = torch.tensor(
            [
                [0.0, -1.0, 0.0],  # Ray before first timestamp
                [0.0, -1.0, 0.0],  # Ray at first timestamp
                [0.0, -1.0, 0.0],  # Ray between timestamps
                [0.0, -1.0, 0.0],  # Ray at last timestamp
                [0.0, -1.0, 0.0],  # Ray after last timestamp
            ],
            device=torch.device("cuda"),
            dtype=torch.float32,
        )

        self.rays_d = torch.tensor([[0.0, 1.0, 0.0]], device=torch.device("cuda"), dtype=torch.float32).repeat(5, 1)

        self.rays_timestamps_us = torch.tensor(
            [
                50000,  # Before first timestamp
                100000,  # Exactly at first timestamp
                250000,  # Between timestamps
                400000,  # At last timestamp
                450000,  # After last timestamp
            ],
            device=torch.device("cuda"),
            dtype=torch.int64,
        )

    def test_ray_intersection_binary_search(self):
        # Test regular ray intersection
        intersections_cnt = self.tracks.ray_intersection(
            self.rays_o, self.rays_d, self.rays_timestamps_us, with_intersections_ts=False
        ).intersections_cnt

        # Verify intersection counts
        expected_counts = torch.tensor([0, 1, 1, 1, 0], device=torch.device("cuda"), dtype=torch.int32)
        self._compareTensor(intersections_cnt, expected_counts)

    def test_rolling_shutter_intersection_binary_search(self):
        # Test rolling shutter intersection
        pixel_idxs = torch.tensor([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], device="cuda:0", dtype=torch.int16)

        camera_rays = self.rays_d

        # Camera poses at start and end of frame
        camera_poses = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Start pose
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # End pose
            ],
            device="cuda:0",
            dtype=torch.float32,
        )

        # Frame timestamps spanning our test timestamps
        camera_timestamp_start_us = 50000
        camera_timestamp_end_us = 450000

        w, h = 5, 1  # Simple 5x1 image
        shutter_type = 2  # ROLLING_LEFT_TO_RIGHT

        intersections_cnt = self.tracks.ray_rolling_shutter_intersection(
            pixel_idxs,
            camera_rays,
            camera_poses,
            camera_timestamp_start_us,
            camera_timestamp_end_us,
            w,
            h,
            shutter_type,
        ).intersections_cnt

        # Verify intersection counts
        expected_counts = torch.tensor([0, 1, 1, 1, 0], device="cuda:0", dtype=torch.int32)
        self._compareTensor(intersections_cnt, expected_counts)

    def test_edge_cases(self):
        # Create an edge case track with only one timestamp
        with pytest.raises(AssertionError, match="Tracks: require at least two poses per track"):
            CuboidTracks.Factory.from_numpy(
                tracks_id=["single"],
                tracks_poses=[np.eye(4, dtype=np.float32)[None, :, :]],
                tracks_timestamps_us=[np.array([200000], dtype=np.int64)],
                tracks_label_class=["test"],
                tracks_flags=[TrackFlags.NONE],
                cuboids_dims=[np.array([1.0, 1.0, 1.0], dtype=np.float32)],
            )

        # Test rays at different timestamps
        rays_o = torch.tensor([[0.0, -1.0, 0.0]], device="cuda:0", dtype=torch.float32).repeat(3, 1)
        rays_d = torch.tensor([[0.0, 1.0, 0.0]], device="cuda:0", dtype=torch.float32).repeat(3, 1)
        rays_timestamps_us = torch.tensor(
            [
                50000,  # Before timestamp
                400000,  # At last timestamp
                500000,  # After timestamp
            ],
            device="cuda:0",
            dtype=torch.int64,
        )

        # Test intersection
        intersections_cnt = self.tracks.ray_intersection(
            rays_o, rays_d, rays_timestamps_us, with_intersections_ts=False
        ).intersections_cnt

        # Should only intersect at exact timestamp
        expected_counts = torch.tensor([0, 1, 0], device="cuda:0", dtype=torch.int32)
        self._compareTensor(intersections_cnt, expected_counts)


class CleanTrackIdsTest(CommonTestCase):
    """Test suite for CuboidTracks.Ops.clean_track_ids functionality."""

    def setUp(self):
        # Create a track with IDs that have @suffixes (as seen in NCORE data)
        self.original_track_ids = [
            "15@scene:obstacles:autolabels:v2",
            "349@scene:obstacles:autolabels:v2",
            "42@scene:obstacles:autolabels:v2",
        ]
        self.expected_cleaned_ids = ["15", "349", "42"]

        tracks_poses: list[np.ndarray] = []
        tracks_timestamps_us: list[np.ndarray] = []
        tracks_label_class: list[str] = []
        tracks_flags: list[TrackFlags] = []
        cuboids_dims: list[np.ndarray] = []

        for i, track_id in enumerate(self.original_track_ids):
            # Create simple identity poses
            pose0 = np.eye(4, 4, dtype=np.float32)
            pose0[:3, 3] = [float(i), 0, 0]
            pose1 = pose0.copy()
            pose1[:3, 3] = [float(i) + 1, 0, 0]
            tracks_poses.append(np.stack([pose0, pose1]))
            tracks_timestamps_us.append(np.array([0, 1000000], np.int64))
            tracks_label_class.append("vehicle")
            tracks_flags.append(TrackFlags.DYNAMIC)
            cuboids_dims.append(np.array([2, 1, 1], dtype=np.float32))

        self.tracks = CuboidTracks.Factory.from_numpy(
            self.original_track_ids,
            tracks_poses,
            tracks_timestamps_us,
            tracks_label_class,
            tracks_flags,
            cuboids_dims=cuboids_dims,
        )

    def test_clean_track_ids_removes_suffixes(self):
        """Test that clean_track_ids removes @suffixes from track IDs."""
        cleaned_tracks = CuboidTracks.Ops.clean_track_ids(self.tracks, DataSourceSummary._clean_track_id_str)

        # Verify track IDs are cleaned
        self.assertEqual(cleaned_tracks.tracks_id, self.expected_cleaned_ids)

        # Verify original tracks are unchanged (immutability)
        self.assertEqual(self.tracks.tracks_id, self.original_track_ids)

    def test_clean_track_ids_preserves_other_data(self):
        """Test that clean_track_ids preserves all other track data."""
        cleaned_tracks = CuboidTracks.Ops.clean_track_ids(self.tracks, DataSourceSummary._clean_track_id_str)

        # Verify other data is preserved
        self._compareTensor(cleaned_tracks.tracks_packinfo, self.tracks.tracks_packinfo)
        self._compareTensor(cleaned_tracks.tracks_poses.data, self.tracks.tracks_poses.data)
        self._compareTensor(cleaned_tracks.tracks_timestamps_us, self.tracks.tracks_timestamps_us)
        self._compareTensor(cleaned_tracks.tracks_flags, self.tracks.tracks_flags)
        self.assertEqual(cleaned_tracks.max_track_n_poses, self.tracks.max_track_n_poses)
        self.assertEqual(cleaned_tracks.tracks_label_class, self.tracks.tracks_label_class)
        self._compareTensor(cleaned_tracks.cuboids_dims, self.tracks.cuboids_dims)

    def test_clean_track_ids_with_no_suffix(self):
        """Test that clean_track_ids works correctly when IDs have no suffixes."""
        # Create tracks with already clean IDs
        clean_ids = ["track_1", "track_2"]
        tracks_poses = [
            np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]),
            np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]),
        ]
        tracks_timestamps_us = [np.array([0, 1000], np.int64), np.array([0, 1000], np.int64)]
        tracks_label_class = ["car", "car"]
        tracks_flags = [TrackFlags.NONE, TrackFlags.NONE]
        cuboids_dims = [np.array([1, 1, 1], dtype=np.float32), np.array([1, 1, 1], dtype=np.float32)]

        clean_tracks = CuboidTracks.Factory.from_numpy(
            clean_ids, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, cuboids_dims=cuboids_dims
        )

        # Apply cleaning - should be a no-op
        result = CuboidTracks.Ops.clean_track_ids(clean_tracks, DataSourceSummary._clean_track_id_str)
        self.assertEqual(result.tracks_id, clean_ids)

    def test_clean_track_ids_with_empty_tracks(self):
        """Test that clean_track_ids works correctly with empty tracks."""
        empty_tracks = CuboidTracks.Factory.empty()
        result = CuboidTracks.Ops.clean_track_ids(empty_tracks, DataSourceSummary._clean_track_id_str)
        self.assertEqual(result.tracks_id, [])
        self.assertEqual(result.n_tracks, 0)

    def test_clean_track_ids_subset_lookup_after_cleaning(self):
        """Test that subset_from_tracks_id works correctly after cleaning track IDs.

        This tests the real-world scenario where cleaned track IDs are used to lookup
        tracks in a CuboidTracks that has also been cleaned.
        """
        cleaned_tracks = CuboidTracks.Ops.clean_track_ids(self.tracks, DataSourceSummary._clean_track_id_str)

        # This should work without raising ValueError
        subset = CuboidTracks.Ops.subset_from_tracks_id(cleaned_tracks, ["15", "42"])

        self.assertEqual(subset.tracks_id, ["15", "42"])
        self.assertEqual(subset.n_tracks, 2)


class VisibleTrackSplitOpsTest(CommonTestCase):
    @staticmethod
    def _make_tracks() -> CuboidTracks:
        timestamps_us = np.arange(0, 7000000, 1000000, dtype=np.int64)
        person_poses = []
        vehicle_poses = []
        for idx in range(len(timestamps_us)):
            person_pose = np.eye(4, dtype=np.float32)
            person_pose[0, 3] = float(idx)
            person_poses.append(person_pose)

            vehicle_pose = np.eye(4, dtype=np.float32)
            vehicle_pose[1, 3] = 10.0
            vehicle_poses.append(vehicle_pose)

        return CuboidTracks.Factory.from_numpy(
            tracks_id=["000021@scene:obstacles:autolabels", "vehicle@scene:obstacles:autolabels"],
            tracks_poses=[np.stack(person_poses), np.stack(vehicle_poses)],
            tracks_timestamps_us=[timestamps_us, timestamps_us],
            tracks_label_class=["person", "vehicle"],
            tracks_flags=[TrackFlags.DYNAMIC, TrackFlags.NONE],
            cuboids_dims=[
                np.array([1.0, 1.0, 2.0], dtype=np.float32),
                np.array([4.0, 2.0, 2.0], dtype=np.float32),
            ],
        )

    @staticmethod
    def _track_timestamps_us(cuboid_tracks: CuboidTracks, track_idx: int) -> list[int]:
        start_idx, n_poses = cuboid_tracks.tracks_packinfo[track_idx].detach().cpu().numpy().tolist()
        return cuboid_tracks.tracks_timestamps_us[start_idx : start_idx + n_poses].detach().cpu().numpy().tolist()

    def test_split_by_visible_intervals_records_metadata_and_preserves_source_suffix(self):
        tracks = self._make_tracks()
        split_children = []

        split_tracks = CuboidTracks.Ops.split_by_visible_intervals(
            tracks,
            visible_intervals_us=[
                np.array([[1000000, 5000000]], dtype=np.int64),
                np.array([[0, 6000000]], dtype=np.int64),
            ],
            max_visible_duration_us=2000000,
            split_mask=np.array([True, False], dtype=bool),
            split_children_out=split_children,
        )

        self.assertEqual(
            split_tracks.tracks_id,
            [
                "000021__split00@scene:obstacles:autolabels",
                "000021__split01@scene:obstacles:autolabels",
                "000021__split02@scene:obstacles:autolabels",
                "000021__split03@scene:obstacles:autolabels",
                "vehicle@scene:obstacles:autolabels",
            ],
        )
        self.assertEqual([child.parent_track_id for child in split_children], ["000021@scene:obstacles:autolabels"] * 4)
        self.assertEqual([child.child_track_id for child in split_children], split_tracks.tracks_id[:4])
        self.assertEqual(
            [(child.start_timestamp_us, child.end_timestamp_us) for child in split_children],
            [(0, 1000000), (1000000, 3000000), (3000000, 5000000), (5000000, 6000000)],
        )
        self.assertEqual(split_tracks.tracks_label_class, ["person", "person", "person", "person", "vehicle"])

    def test_split_by_visible_intervals_merges_short_visibility_gaps(self):
        tracks = self._make_tracks()
        split_children = []

        split_tracks = CuboidTracks.Ops.split_by_visible_intervals(
            tracks,
            visible_intervals_us=[
                np.array([[1000000, 2000000], [2400000, 4500000]], dtype=np.int64),
                np.array([[0, 6000000]], dtype=np.int64),
            ],
            max_visible_duration_us=2000000,
            split_mask=np.array([True, False], dtype=bool),
            merge_gap_us=500000,
            split_children_out=split_children,
        )

        self.assertEqual(
            [(child.start_timestamp_us, child.end_timestamp_us) for child in split_children],
            [(0, 1000000), (1000000, 3000000), (3000000, 4500000), (4500000, 6000000)],
        )
        self.assertEqual(self._track_timestamps_us(split_tracks, 1), [1000000, 2000000, 3000000])

    def test_split_by_visible_intervals_merges_only_short_edge_invisible_segments(self):
        tracks = self._make_tracks()
        split_children = []

        CuboidTracks.Ops.split_by_visible_intervals(
            tracks,
            visible_intervals_us=[
                np.array([[100000, 2000000], [2100000, 5900000]], dtype=np.int64),
                np.array([[0, 6000000]], dtype=np.int64),
            ],
            max_visible_duration_us=2000000,
            split_mask=np.array([True, False], dtype=bool),
            min_child_visible_duration_us=500000,
            split_children_out=split_children,
        )

        self.assertEqual(
            [(child.start_timestamp_us, child.end_timestamp_us) for child in split_children],
            [
                (0, 2000000),
                (2000000, 2100000),
                (2100000, 4100000),
                (4100000, 6000000),
            ],
        )

    def test_merge_split_children_restores_parent_facing_tracks(self):
        tracks = self._make_tracks()
        split_children = []
        split_tracks = CuboidTracks.Ops.split_by_visible_intervals(
            tracks,
            visible_intervals_us=[
                np.array([[1000000, 5000000]], dtype=np.int64),
                np.array([[0, 6000000]], dtype=np.int64),
            ],
            max_visible_duration_us=2000000,
            split_mask=np.array([True, False], dtype=bool),
            split_children_out=split_children,
        )

        merged_tracks = CuboidTracks.Ops.merge_split_children(split_tracks, TrackSplitMetadata(children=split_children))

        self.assertEqual(
            merged_tracks.tracks_id,
            ["000021@scene:obstacles:autolabels", "vehicle@scene:obstacles:autolabels"],
        )
        self.assertEqual(self._track_timestamps_us(merged_tracks, 0), list(range(0, 7000000, 1000000)))

    def test_track_split_resolver_maps_parent_and_child_ids(self):
        metadata = TrackSplitMetadata.from_dict(
            {
                "children": [
                    {
                        "parent_track_id": "000021",
                        "child_track_id": "000021__split00",
                        "start_timestamp_us": 0,
                        "end_timestamp_us": 1000000,
                        "child_index": 0,
                    },
                    {
                        "parent_track_id": "000021",
                        "child_track_id": "000021__split01",
                        "start_timestamp_us": 1000000,
                        "end_timestamp_us": 3000000,
                        "child_index": 1,
                    },
                ]
            }
        )
        resolver = TrackSplitResolver(metadata)

        self.assertFalse(TrackSplitResolver(None).has_splits)
        self.assertEqual(resolver.to_parent_track_id("000021__split01"), "000021")
        self.assertEqual(resolver.child_ids_for_parent("unsplit"), ["unsplit"])
        self.assertEqual(
            resolver.child_ids_overlapping_interval("000021", 900000, 1100000), ["000021__split00", "000021__split01"]
        )
        self.assertEqual(resolver.active_child_id("000021", 1000000), "000021__split01")


class InterpolateTracksPosesTest(CommonTestCase):
    def setUp(self):
        # Create two tracks: one with translation only, one with translation and rotation
        tracks_id = ["track_translation", "track_rotation"]

        # Track 1: translation along x-axis
        base_pose = np.eye(4, dtype=np.float32)
        poses1 = []
        for x in range(4):
            pose = base_pose.copy()
            pose[0, 3] = float(x)
            poses1.append(pose)
        poses1 = np.stack(poses1)

        # Track 2: rotation about z-axis (0, 90, 180, 270 degrees)
        poses2 = []
        for angle_deg in [0, 90, 180, 270]:
            pose = base_pose.copy()
            rot = Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
            pose[:3, :3] = rot
            poses2.append(pose)
        poses2 = np.stack(poses2)

        tracks_poses = [poses1, poses2]
        tracks_timestamps_us = [
            np.array([100000, 200000, 300000, 400000], dtype=np.int64),
            np.array([100000, 200000, 300000, 400000], dtype=np.int64),
        ]
        tracks_label_class = ["translation", "rotation"]
        tracks_flags = [TrackFlags.NONE, TrackFlags.NONE]
        cuboids_dims = [np.array([1.0, 1.0, 1.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)]

        self.tracks = CuboidTracks.Factory.from_numpy(
            tracks_id, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, cuboids_dims
        )

    def test_interpolate_tracks_poses_translation(self):
        # Test translation track at boundaries and interpolation
        track_idx = torch.tensor([0], dtype=torch.int32, device="cuda")
        timestamps = torch.tensor([100000, 250000, 400000], dtype=torch.int64, device="cuda")
        # 100000: at first pose, 250000: halfway (should be x=1.5), 400000: last pose
        expected_x = [0.0, 1.5, 3.0]
        poses = self.tracks.interpolate_tracks_poses(timestamps, track_idx.repeat(len(timestamps)))
        for i, x in enumerate(expected_x):
            self._compareTensor(
                poses[i].vec(), torch.tensor([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda", dtype=torch.float32)
            )

    def test_interpolate_tracks_poses_ex_translation(self):
        # Test translation track with out of bounds timestamps
        track_idx = torch.tensor([0], dtype=torch.int32, device="cuda")
        timestamps = torch.tensor([-100000, 100000, 250000, 400000, 10000000], dtype=torch.int64, device="cuda")
        # -100000 is out of bounds,: at first pose, 250000: halfway (should be x=1.5), 400000: last pose, 10000000: out of bounds
        expected_x = [0.0, 0.0, 1.5, 3.0, 0.0]
        expected_mask = [False, True, True, True, False]
        poses, mask = self.tracks.interpolate_tracks_poses_ex(
            timestamps,
            track_idx.repeat(len(timestamps)),
        )

        for i, x in enumerate(expected_x):
            if expected_mask[i]:
                self._compareTensor(
                    poses[i].vec(), torch.tensor([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda", dtype=torch.float32)
                )

            assert mask[i].item() == expected_mask[i]

    def test_interpolate_tracks_poses_rotation(self):
        # Test rotation track at boundaries and interpolation
        track_idx = torch.tensor([1], dtype=torch.int32, device="cuda")
        timestamps = torch.tensor([100000, 200000, 300000, 400000, 250000], dtype=torch.int64, device="cuda")
        # 100000: 0 deg, 200000: 90 deg, 300000: 180 deg, 400000: 270 deg, 250000: halfway (135 deg)

        expected_angles = [0, 90, 180, 270, 135]
        poses = self.tracks.interpolate_tracks_poses(timestamps, track_idx.repeat(len(timestamps)))
        for i, angle in enumerate(expected_angles):
            # Extract quaternion from pose
            quat = poses[i].vec()[3:].detach().cpu().numpy()
            # Convert to rotation matrix and get angle
            rot = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
            z_angle = rot.as_euler("zxy", degrees=True)[0]
            # Normalize angle to [0, 360)
            z_angle = (z_angle + 360) % 360
            expected = angle % 360
            # Allow a small tolerance due to quaternion interpolation
            self.assertTrue(np.isclose(z_angle, expected, atol=1.0), f"Expected {expected}, got {z_angle}")

    def test_interpolate_tracks_poses_multiple(self):
        # Test batch interpolation for both tracks at the same timestamp
        timestamps = torch.tensor([150000, 350000], dtype=torch.int64, device="cuda")
        tracks_idx = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        poses = self.tracks.interpolate_tracks_poses(timestamps, tracks_idx)
        # Track 0: x = 0.5, Track 1: rotation = 225 deg
        self._compareTensor(
            poses[0].vec(), torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda", dtype=torch.float32)
        )

        # For rotation, check angle
        quat = poses[1].vec()[3:].detach().cpu().numpy()
        rot = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
        z_angle = rot.as_euler("zxy", degrees=True)[0]
        z_angle = (z_angle + 360) % 360
        self.assertTrue(np.isclose(z_angle, 225, atol=1.0), f"Expected 225, got {z_angle}")

    def test_interpolate_tracks_poses_zero_gap(self):
        # Create a track with two poses at the same timestamp
        tracks_id = ["zero_gap_track"]
        base_pose = np.eye(4, dtype=np.float32)
        pose1 = base_pose.copy()
        pose1[0, 3] = 1.0
        pose2 = base_pose.copy()
        pose2[0, 3] = 2.0
        tracks_poses = [np.stack([pose1, pose2])]
        tracks_timestamps_us = [np.array([100000, 100000], dtype=np.int64)]  # zero gap
        tracks_label_class = ["zero_gap"]
        tracks_flags = [TrackFlags.NONE]
        cuboids_dims = [np.array([1.0, 1.0, 1.0], dtype=np.float32)]

        tracks = CuboidTracks.Factory.from_numpy(
            tracks_id, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, cuboids_dims
        )
        # Interpolate at the timestamp
        timestamps = torch.tensor([100000], dtype=torch.int64, device="cuda")
        track_idx = torch.tensor([0], dtype=torch.int32, device="cuda")
        poses = tracks.interpolate_tracks_poses(timestamps, track_idx)
        # Should return the first pose (alpha=0)
        self._compareTensor(
            poses[0].vec(), torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda", dtype=torch.float32)
        )
