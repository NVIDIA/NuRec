# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PerSensor resolver helpers (base_schema)."""

import pytest

from omegaconf import OmegaConf

from nurec.config.base_schema import resolve_per_sensor, validate_per_sensor_coverage


def test_resolve_scalar_broadcasts():
    # A non-dict value applies to every sensor id.
    assert resolve_per_sensor(1, "lidar_a") == 1
    assert resolve_per_sensor("unstructured", "anything") == "unstructured"
    assert resolve_per_sensor(None, "lidar_a") is None
    assert resolve_per_sensor([-25.0, 15.0], "lidar_a") == [-25.0, 15.0]


def test_resolve_dict_by_id():
    value = {"lidar_a": 1, "lidar_b": 2}
    assert resolve_per_sensor(value, "lidar_a") == 1
    assert resolve_per_sensor(value, "lidar_b") == 2


def test_resolve_dict_missing_without_default_raises():
    with pytest.raises(ValueError, match=r"no entry for sensor 'lidar_b'"):
        resolve_per_sensor({"lidar_a": 1}, "lidar_b")


def test_resolve_dict_missing_with_explicit_default_returns_it():
    # An explicit default= is returned on a miss (used by optional per-sensor fields
    # such as the FOV overrides, which fall back to None = "use the model FOV").
    assert resolve_per_sensor({"lidar_a": [1.0, 2.0]}, "lidar_b", default=None) is None


def test_resolve_dict_with_none_sensor_id_raises():
    # sensor_id=None against a dict is a miss (there is no None key): with no
    # default it must raise (mentioning 'None'), documenting that None is only
    # meaningful for scalar broadcast.
    with pytest.raises(ValueError, match=r"no entry for sensor 'None'"):
        resolve_per_sensor({"lidar_a": 1}, None)


def test_resolve_dict_with_none_sensor_id_uses_default():
    # ... unless an explicit default= is supplied.
    assert resolve_per_sensor({"lidar_a": 1}, None, default=7) == 7


def test_resolve_dictconfig_is_looked_up_not_broadcast():
    # An OmegaConf DictConfig (which does NOT subclass dict) must be treated as a
    # per-sensor dict, not silently broadcast as a scalar.
    value = OmegaConf.create({"lidar_a": 1, "lidar_b": 2})
    assert resolve_per_sensor(value, "lidar_a") == 1
    assert resolve_per_sensor(value, "lidar_b") == 2
    with pytest.raises(ValueError, match=r"no entry for sensor 'lidar_c'"):
        resolve_per_sensor(value, "lidar_c")


def test_validate_coverage_scalar_always_ok():
    validate_per_sensor_coverage(3, ["lidar_a", "lidar_b"])  # no raise


def test_validate_coverage_dict_all_ids_ok():
    validate_per_sensor_coverage({"lidar_a": 1, "lidar_b": 2}, ["lidar_a", "lidar_b"])


def test_validate_coverage_dictconfig_all_ids_ok():
    # A DictConfig with exactly the active ids validates like a plain dict.
    value = OmegaConf.create({"lidar_a": 1, "lidar_b": 2})
    validate_per_sensor_coverage(value, ["lidar_a", "lidar_b"])


def test_validate_coverage_dict_missing_id_raises():
    with pytest.raises(ValueError, match=r"missing entries for \['lidar_b'\]"):
        validate_per_sensor_coverage({"lidar_a": 1}, ["lidar_a", "lidar_b"], field_name="dataset.foo")


def test_validate_coverage_dict_extra_key_raises():
    # An unknown/typo'd sensor key must be rejected (it would never be applied and
    # a missing real sensor could go unnoticed).
    with pytest.raises(ValueError, match=r"unknown sensor keys \['ldar_a'\]"):
        validate_per_sensor_coverage({"lidar_a": 1, "ldar_a": 2}, ["lidar_a"], field_name="dataset.foo")
