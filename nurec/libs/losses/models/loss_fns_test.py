# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from nurec.libs.losses.models.loss_fns import total_variation_spatial
from nurec.libs.losses.models.utils import get_mask_semantic
from nurec.utils.batch import CameraFrameLabels
from nurec.utils.types import RayFlags


class TestTotalVariation(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.num_channels = 7
        self.xyz_size = 3
        self.shape = (self.batch_size, self.num_channels, self.xyz_size, self.xyz_size, self.xyz_size)

    def test_shape(self):
        x = torch.rand(self.shape)
        tv_loss = total_variation_spatial(x)
        assert tv_loss.shape == (self.batch_size,)

    def test_constant(self):
        x = torch.ones(self.shape)
        tv_loss = torch.mean(total_variation_spatial(x))
        assert tv_loss.item() == 0.0

    def test_size_one(self):
        x = torch.rand(self.batch_size, self.num_channels, 1, 1, 1)
        tv_loss = torch.mean(total_variation_spatial(x))
        assert tv_loss.item() == 0.0

    def test_random_tensor(self):
        x = torch.rand(self.shape)
        tv_loss = torch.mean(total_variation_spatial(x))
        assert tv_loss.item() >= 0.0

    def test_wrap_around(self):
        x = torch.rand(self.shape)
        tv_loss = total_variation_spatial(x, wrap_around=True)

        expected_tv_z = (x - torch.roll(x, shifts=-1, dims=2)).square().mean(dim=(1, 2, 3, 4))
        expected_tv_y = (x - torch.roll(x, shifts=-1, dims=3)).square().mean(dim=(1, 2, 3, 4))
        expected_tv_x = (x - torch.roll(x, shifts=-1, dims=4)).square().mean(dim=(1, 2, 3, 4))
        expected_tv_loss = expected_tv_z + expected_tv_y + expected_tv_x

        assert torch.allclose(tv_loss, expected_tv_loss)

    def test_manual_variation(self):
        self.xyz_size = 2
        x = torch.zeros(self.batch_size, self.num_channels, self.xyz_size, self.xyz_size, self.xyz_size)
        x[:, :, 0, 0, 0] = 1.0
        x[:, :, 1, 1, 1] = 1.0

        # Manually calculate the expected total variation.
        # Since the "1"s are diagonally opposite of each other, the TV will accumulate in all three
        # dimensions, by two units in each dimension.
        # Note that both the number of channels and batch size cancel out in the normalized result.
        number_of_deltas_per_dim = (self.xyz_size - 1) * self.xyz_size * self.xyz_size
        expected_tv = torch.tensor([3 * 2 / number_of_deltas_per_dim])

        tv_loss = total_variation_spatial(x)
        print(tv_loss)
        for row in tv_loss:
            assert torch.isclose(row, expected_tv)


class TestMaskSemantic(unittest.TestCase):
    def setUp(self):
        # Create a CameraFrameLabels object with some flags set
        BATCH_SIZE, WIDTH, HEIGHT = 2, 64, 64

        # Set some random synthetic flags
        self.is_sky = torch.rand(BATCH_SIZE, WIDTH, HEIGHT, 1) < 0.5
        self.is_road = torch.rand(BATCH_SIZE, WIDTH, HEIGHT, 1) < 0.5
        self.is_vehicle = torch.rand(BATCH_SIZE, WIDTH, HEIGHT, 1) < 0.5
        flags = torch.zeros(BATCH_SIZE, WIDTH, HEIGHT, 1, dtype=torch.int32)
        flags[self.is_sky] |= RayFlags.SKY_SEMANTIC
        flags[self.is_road] |= RayFlags.ROAD_SEMANTIC
        flags[self.is_vehicle] |= RayFlags.VEHICLE_SEMANTIC

        # Build the CameraFrameLabels object
        self.rays_meta = CameraFrameLabels(flags=flags)

    def test_single_operation(self):
        # Test that the mask semantic function returns the correct mask
        sky_semantic = get_mask_semantic(self.rays_meta, "sky")
        road_semantic = get_mask_semantic(self.rays_meta, "road")
        vehicle_semantic = get_mask_semantic(self.rays_meta, "vehicle")

        assert (sky_semantic == self.is_sky).all()
        assert (road_semantic == self.is_road).all()
        assert (vehicle_semantic == self.is_vehicle).all()

    def test_compound_operation(self):
        mask_semantic = get_mask_semantic(self.rays_meta, "sky | road")
        assert (mask_semantic == (self.is_sky | self.is_road)).all()

        mask_semantic = get_mask_semantic(self.rays_meta, "sky & road")
        assert (mask_semantic == (self.is_sky & self.is_road)).all()

        mask_semantic = get_mask_semantic(self.rays_meta, "sky & (~road | vehicle)")
        assert (mask_semantic == (self.is_sky & (~self.is_road | self.is_vehicle))).all()
