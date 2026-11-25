# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap

import torch

from nurec.libs.vren.interface import vren  # type: ignore
from nurec.utils.tests import CommonTestCase


class PointsInTracksTest(CommonTestCase):
    def setUp(self) -> None:
        # Create a simple scene with 2 tracks
        #
        #     t=0:                  t=1000000:
        #
        #     Track 1    Track 2    Track 1    Track 2
        #    ┌───┐      ┌───┐       ┌───┐      ┌───┐
        #    │   │      │   │       │   │      │   │
        #    │p0 │      │   │       │p1 │      │   │
        #    │   │      │   │       │   │      │   │
        #    └───┘      └───┘       └───┘      └───┘
        #    (0,0,0)    (0,1,0)     (1,0,0)    (0,2,0)
        #
        #   p0 = (0,0,0)  - inside Track 1 at t=0
        #   p1 = (1,0,0)  - inside Track 1 at t=1000000
        #   p2 = (0,1.5,0)- inside Track 2 at t=500000 (interpolated)
        #   p3 = (3,0,0)  - outside all tracks at all times

        device = torch.device("cuda")

        # Two tracks, each with 2 poses
        self.tracks_packinfo = torch.tensor(
            [
                [0, 2],  # Track 1: poses 0-1
                [2, 2],
            ],  # Track 2: poses 2-3
            dtype=torch.int32,
            device=device,
        )

        # Track poses: [tx, ty, tz, qw, qx, qy, qz]
        self.tracks_poses = torch.tensor(
            [
                # Track 1 poses
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # t=0
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # t=1000000
                # Track 2 poses
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # t=0
                [0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # t=1000000
            ],
            device=device,
        )

        self.tracks_timestamps_us = torch.tensor([0, 1000000, 0, 1000000], dtype=torch.long, device=device)

        # Both tracks are 1x1x1 cubes
        self.cuboids_dims = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            device=device,
        )

        # Test points
        self.points = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # p0
                [1.0, 0.0, 0.0],  # p1
                [0.0, 1.5, 0.0],  # p2
                [3.0, 0.0, 0.0],  # p3
            ],
            device=device,
        )

    def test_point_cuboidtracks_intersection_single_timestamp(self):
        # Test at t=0
        result = vren.point_cuboidtracks_intersection(
            self.points,
            torch.tensor([0], dtype=torch.long, device=self.points.device),
            self.tracks_packinfo,
            self.tracks_poses,
            self.tracks_timestamps_us,
            self.cuboids_dims,
            2,
            True,
        )

        expected = torch.tensor(
            [
                [True, False],  # p0 inside Track 1
                [False, False],  # p1 outside all
                [False, True],  # p2 inside Track 2
                [False, False],  # p3 outside all
            ],
            device=self.points.device,
        )

        self._compareTensor(result, expected)

    def test_point_cuboidtracks_intersection_single_timestamp_dense_mask(self):
        result_dense_mask = vren.point_cuboidtracks_intersection(
            self.points,
            torch.tensor([0], dtype=torch.long, device=self.points.device),
            self.tracks_packinfo,
            self.tracks_poses,
            self.tracks_timestamps_us,
            self.cuboids_dims,
            2,
            True,
        )

        expected = torch.tensor(
            [
                [True, False],  # p0 inside Track 1
                [False, False],  # p1 outside all
                [False, True],  # p2 inside Track 2
                [False, False],  # p3 outside all
            ],
            device=self.points.device,
        )

        self._compareTensor(result_dense_mask, expected)

        result_sparse_mask = vren.point_cuboidtracks_intersection(
            self.points,
            torch.tensor([0], dtype=torch.long, device=self.points.device),
            self.tracks_packinfo,
            self.tracks_poses,
            self.tracks_timestamps_us,
            self.cuboids_dims,
            2,
            False,
        )

        assert result_sparse_mask.shape == torch.Size([4, 1])
        self._compareTensor(result_sparse_mask, torch.any(expected, dim=1, keepdims=True))

    def test_point_cuboidtracks_intersection_interpolated(self):
        # Test at t=400000 (interpolated)
        result = vren.point_cuboidtracks_intersection(
            self.points,
            torch.tensor([400000], dtype=torch.long, device=self.points.device),
            self.tracks_packinfo,
            self.tracks_poses,
            self.tracks_timestamps_us,
            self.cuboids_dims,
            2,
            True,
        )

        expected = torch.tensor(
            [
                [True, False],  # p0 = (0,0,0) inside Track 1 at t=400000 (almost halfway between 0,0,0 and 1,0,0)
                [False, False],  # p1 = (1,0,0) inside Track 1 at t=400000 (almost halfway between 0,0,0 and 1,0,0)
                [False, True],  # p2 = (0,1.5,0) inside Track 2 at t=400000 (almost halfway between 0,1,0 and 0,2,0)
                [False, False],  # p3 = (3,0,0) outside all tracks
            ],
            device=self.points.device,
        )

        self._compareTensor(result, expected)

    def test_point_cuboidtracks_intersection_per_point_timestamps(self):
        # Test with different timestamp for each point
        timestamps = torch.tensor([0, 1000000, 500000, 0], dtype=torch.long, device=self.points.device)

        result = vren.point_cuboidtracks_intersection(
            self.points,
            timestamps,
            self.tracks_packinfo,
            self.tracks_poses,
            self.tracks_timestamps_us,
            self.cuboids_dims,
            2,
            True,
        )

        expected = torch.tensor(
            [
                [True, False],  # p0 at t=0: inside Track 1
                [True, False],  # p1 at t=1000000: inside Track 1
                [False, True],  # p2 at t=500000: inside Track 2
                [False, False],  # p3 at t=0: outside all
            ],
            device=self.points.device,
        )

        self._compareTensor(result, expected)

    def test_point_cuboidtracks_intersection_edge_cases(self):
        # Test empty points
        empty_result = vren.point_cuboidtracks_intersection(
            torch.zeros((0, 3), device=self.points.device),
            torch.zeros((1,), dtype=torch.long, device=self.points.device),
            self.tracks_packinfo,
            self.tracks_poses,
            self.tracks_timestamps_us,
            self.cuboids_dims,
            2,
            True,
        )

        self._compareTensor(empty_result.shape, torch.Size([0, 2]))

        # Test single-pose track (should return all False)
        tracks_packinfo = torch.tensor([[0, 1]], dtype=torch.int32, device=self.points.device)
        tracks_poses = torch.zeros((1, 7), device=self.points.device)
        tracks_poses[0, 3] = 1.0  # Set quaternion w to 1
        tracks_timestamps_us = torch.zeros((1,), dtype=torch.long, device=self.points.device)

        single_pose_result = vren.point_cuboidtracks_intersection(
            self.points,
            torch.zeros((1,), dtype=torch.long, device=self.points.device),
            tracks_packinfo,
            tracks_poses,
            tracks_timestamps_us,
            self.cuboids_dims[:1],
            1,
            True,
        )

        self._compareTensor(single_pose_result, torch.zeros((4, 1), dtype=torch.bool, device=self.points.device))


class CameraTrackIntervalsTest(CommonTestCase):
    def _assert_pack_range_rejected(self, bad_call: str) -> None:
        script = (
            textwrap.dedent(
                """
            import torch
            from nurec.libs.vren.interface import vren

            def run(max_poses, start=0):
                one = lambda *shape: torch.ones(shape, device="cuda")
                poses = torch.zeros((1, 2, 7), device="cuda")
                poses[:, :, 6] = 1
                vren.camera_track_intervals(
                    torch.zeros((1, 2), dtype=torch.int16, device="cuda"),
                    torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
                    poses,
                    torch.tensor([[0, 10]], device="cuda"),
                    torch.zeros(1, device="cuda"),
                    torch.tensor([[start, 2]], dtype=torch.int32, device="cuda"),
                    one(2, 7),
                    torch.tensor([0, 10], device="cuda"),
                    one(1, 3),
                    width=1, height=1, shutter=5, max_poses=max_poses, max_hits=1,
                )
                torch.cuda.synchronize()

            run(2)
            print("ok", flush=True)
            """
            )
            + bad_call
        )
        # A failed device assert leaves the CUDA context unusable, so the call runs in a child process.
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
        self.assertEqual(result.stdout.strip(), "ok", result.stderr)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("track pack range exceeds max_poses", result.stderr)

    def test_pack_range_beyond_max_poses_is_rejected(self):
        self._assert_pack_range_rejected("run(1)\n")

    def test_pack_range_start_near_int32_limit_is_rejected(self):
        # start + count would wrap past the int32 maximum.
        self._assert_pack_range_rejected(f"run(2, start={2**31 - 2})\n")

    def test_batched_intervals_match_legacy_hit_reduction(self):
        width, height = 12, 10
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        pixels = torch.stack((xx, yy), -1).reshape(-1, 2).to(device="cuda", dtype=torch.int16) * 2
        rays = torch.stack(((xx.float() - 6) / 8, (yy.float() - 5) / 8, torch.ones_like(xx)), -1)
        rays = torch.nn.functional.normalize(rays.reshape(-1, 3), dim=-1).cuda()
        relative_times = torch.linspace(0, 1, len(pixels), device="cuda")
        poses = torch.zeros((3, 2, 7), device="cuda")
        poses[:, :, 6] = 1
        poses[:, 1, 0] = 0.1
        packinfo = torch.tensor([[0, 2], [2, 2]], dtype=torch.int32, device="cuda")
        track_poses = torch.tensor(
            [[0, 0, 6, 0, 0, 0, 1], [0.2, 0, 6, 0, 0, 0, 1], [2, 0, 8, 0, 0, 0, 1], [2, 0.2, 8, 0, 0, 0, 1]],
            dtype=torch.float32,
            device="cuda",
        )
        dims = torch.tensor([[3, 3, 3], [2, 2, 2]], dtype=torch.float32, device="cuda")
        for base in (-500_000, 1_700_000_000_000_000):
            timestamps = torch.tensor(
                [[base + i * 200_000, base + i * 200_000 + 33_333] for i in range(3)],
                dtype=torch.int64,
                device="cuda",
            )
            track_times = torch.tensor([base - 300_000, base + 1_000_000] * 2, device="cuda")
            for shutter in range(1, 6):
                with self.subTest(base=base, shutter=shutter):
                    actual = vren.camera_track_intervals(
                        pixels,
                        rays,
                        poses,
                        timestamps,
                        relative_times,
                        packinfo,
                        track_poses,
                        track_times,
                        dims,
                        width=width,
                        height=height,
                        shutter=shutter,
                        max_poses=2,
                        max_hits=2,
                    )
                    expected = torch.full_like(actual, -1)
                    expected[:, :, 0] = torch.iinfo(torch.int64).max
                    for frame in range(3):
                        start, end = timestamps[frame].tolist()
                        _, ids = vren.ray_cuboidtracks_rolling_shutter_intersection(
                            pixels,
                            rays,
                            poses[frame],
                            start,
                            end,
                            width,
                            height,
                            shutter,
                            packinfo,
                            track_poses,
                            track_times,
                            dims,
                            2,
                            2,
                            False,
                        )
                        report_times = start + (relative_times * float(end - start)).to(torch.int64)
                        for track in range(2):
                            hit = (ids == track).any(1)
                            if hit.any():
                                expected[frame, track, 0] = report_times[hit].min()
                                expected[frame, track, 1] = report_times[hit].max().clamp_min(-1)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with self.assertRaisesRegex(RuntimeError, "truncating"):
                vren.camera_track_intervals(
                    pixels,
                    rays,
                    poses,
                    timestamps,
                    relative_times,
                    packinfo,
                    track_poses,
                    track_times,
                    dims,
                    width=width,
                    height=height,
                    shutter=1,
                    max_poses=2,
                    max_hits=1,
                )
