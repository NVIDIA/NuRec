# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np

from omegaconf import DictConfig

from nurec.datasets.samplers.uniform import UniformSensorSampler


def _sampler(*, sample_all_sensors: bool) -> UniformSensorSampler:
    return UniformSensorSampler(DictConfig({"sample_all_sensors": sample_all_sensors}), MagicMock())


def test_single_sensor_does_not_advance_rng() -> None:
    rng = np.random.default_rng(42)
    reference_rng = np.random.default_rng(42)

    result = _sampler(sample_all_sensors=False).sample_sensor(rng, batch_idx=0, sensor_ids=["lidar_top"])

    assert result.sampled_sensor_ids == ["lidar_top"]
    assert rng.random() == reference_rng.random()


def test_multiple_sensors_are_sampled_across_batches() -> None:
    rng = np.random.default_rng(42)
    sampler = _sampler(sample_all_sensors=False)
    sensor_ids = ["lidar_top", "lidar_front"]

    sampled_ids = {
        sampler.sample_sensor(rng, batch_idx=batch_idx, sensor_ids=sensor_ids).sampled_sensor_ids[0]
        for batch_idx in range(20)
    }

    assert sampled_ids == set(sensor_ids)


def test_sample_all_sensors_returns_every_sensor() -> None:
    sensor_ids = ["lidar_top", "lidar_front"]

    result = _sampler(sample_all_sensors=True).sample_sensor(
        np.random.default_rng(42), batch_idx=0, sensor_ids=sensor_ids
    )

    assert result.sampled_sensor_ids == sensor_ids
