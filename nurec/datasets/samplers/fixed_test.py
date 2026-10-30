# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from unittest.mock import Mock

import numpy as np

from omegaconf import DictConfig

from nurec.datasets.samplers.base import BaseFrameSampler, SkipFrameSampler
from nurec.datasets.samplers.fixed import FixedFrameSampler


class TestFixedFrameSampler(unittest.TestCase):
    def _sampler(self, frame_index: int) -> FixedFrameSampler:
        return FixedFrameSampler(config=DictConfig({"frame_index": frame_index}), dataset=Mock())

    def test_always_samples_the_configured_local_index(self):
        sampler = self._sampler(2)
        rng = np.random.default_rng(0)
        for frame_range in [range(5), range(10, 16)]:
            result = sampler.sample_frame(rng, batch_idx=0, frame_range=frame_range, unique_sensor_id="cam")
            self.assertEqual(result.sampled_frame_idx, frame_range[2])

    def test_out_of_range_index_raises(self):
        sampler = self._sampler(7)
        with self.assertRaises(IndexError):
            sampler.sample_frame(np.random.default_rng(0), 0, range(3), "cam")

    def test_registered_in_factory(self):
        self.assertIs(BaseFrameSampler.FRAME_SAMPLER_VARIANTS["fixed"], FixedFrameSampler)


class TestTrainingFrameMasks(unittest.TestCase):
    """training_frame_mask must mirror each sampler's reachable frame set."""

    def test_fixed_mask_is_one_hot(self):
        mask = FixedFrameSampler.training_frame_mask(DictConfig({"frame_index": 3}), n_frames=5)
        np.testing.assert_array_equal(mask, [False, False, False, True, False])

    def test_fixed_mask_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            FixedFrameSampler.training_frame_mask(DictConfig({"frame_index": 5}), n_frames=3)

    def test_fixed_mask_negative_index_raises(self):
        # __init__ asserts frame_index >= 0; the mask must reject it too,
        # not silently wrap to the last frame.
        with self.assertRaises(ValueError):
            FixedFrameSampler.training_frame_mask(DictConfig({"frame_index": -1}), n_frames=5)

    def test_base_default_reaches_every_frame(self):
        mask = BaseFrameSampler.training_frame_mask(DictConfig({}), n_frames=4)
        np.testing.assert_array_equal(mask, [True, True, True, True])

    def test_skip_reaches_no_frame(self):
        mask = SkipFrameSampler.training_frame_mask(DictConfig({}), n_frames=4)
        np.testing.assert_array_equal(mask, [False, False, False, False])


class TestGroupTrainingFrameMask(unittest.TestCase):
    """The group-level mask mirrors BaseBatchSampler's substitution rules."""

    def test_zero_ratio_substitutes_skip(self):
        group = {"ratio_camera_samples": 0, "camera_frame_sampler": {"name": "uniform"}}
        mask = BaseFrameSampler.group_training_frame_mask(group, "camera", 3)
        np.testing.assert_array_equal(mask, [False, False, False])

    def test_named_sampler_mask_is_used(self):
        group = {
            "ratio_camera_samples": 1,
            "camera_frame_sampler": {"name": "holdout", "include_frames_start": 0, "include_every_n_frames": 2},
        }
        mask = BaseFrameSampler.group_training_frame_mask(group, "camera", 4)
        np.testing.assert_array_equal(mask, [True, False, True, False])

    def test_lidar_shares_timestamp_camera_sampler(self):
        group = {
            "ratio_camera_samples": 1,
            "ratio_lidar_samples": 1,
            "camera_frame_sampler": {"name": "timestamp"},
            "lidar_frame_sampler": {"name": "fixed", "frame_index": 0},
        }
        mask = BaseFrameSampler.group_training_frame_mask(group, "lidar", 3)
        np.testing.assert_array_equal(mask, [True, True, True])

    def test_lidar_shares_skipped_timestamp_camera_sampler(self):
        # Zero camera ratio substitutes SkipFrameSampler for the camera —
        # the timestamp-sharing rule hands that Skip to the lidar side.
        group = {
            "ratio_camera_samples": 0,
            "ratio_lidar_samples": 1,
            "camera_frame_sampler": {"name": "timestamp"},
            "lidar_frame_sampler": {"name": "uniform"},
        }
        mask = BaseFrameSampler.group_training_frame_mask(group, "lidar", 3)
        np.testing.assert_array_equal(mask, [False, False, False])

    def test_missing_sampler_config_is_conservative(self):
        mask = BaseFrameSampler.group_training_frame_mask({"ratio_camera_samples": 1}, "camera", 2)
        np.testing.assert_array_equal(mask, [True, True])


if __name__ == "__main__":
    unittest.main()
