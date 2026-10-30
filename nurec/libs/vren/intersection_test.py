# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
