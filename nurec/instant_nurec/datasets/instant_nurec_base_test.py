# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

from typing import Literal, NoReturn
from unittest.mock import MagicMock

import numpy as np
import pytest

import nurec.instant_nurec.datasets.registry as registry_module

from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType
from nurec.instant_nurec.config.dataset import CameraSubsamplerConfig, InstantNuRecMixedDatasetConfig
from nurec.instant_nurec.config.dataset import TestIndexInstantNuRecDatasetConfig as IndexDatasetConfig
from nurec.instant_nurec.datasets.instant_nurec_base import (
    CameraSubsampler,
    InstantNuRecDataError,
    MixedInstantNuRecDataset,
)
from nurec.instant_nurec.datasets.instant_nurec_base import TestIndexInstantNuRecDataset as IndexDataset
from nurec.instant_nurec.datasets.registry import make


def _sample_rng(dataset: IndexDataset, batch_idx: int) -> int:
    return int(dataset._get_rng(batch_idx).integers(0, 2**31))


def _expected_sample(*, epoch: int, batch_idx: int, seed: int) -> int:
    digest = hashlib.sha256(f"{epoch}_{batch_idx}_{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return int(rng.integers(0, 2**31))


def test_rng_uses_lightning_environment_when_seed_is_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PL_GLOBAL_SEED", "17")
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)
    dataset = IndexDataset(config)
    dataset.set_rng_epoch(3)

    assert _sample_rng(dataset, 5) == _expected_sample(epoch=3, batch_idx=5, seed=17)


def test_omitted_seed_preserves_dynamic_environment_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)
    dataset = IndexDataset(config)

    monkeypatch.setenv("PL_GLOBAL_SEED", "1")
    first = _sample_rng(dataset, 0)
    monkeypatch.setenv("PL_GLOBAL_SEED", "2")

    assert _sample_rng(dataset, 0) != first


def test_omitted_seed_requires_lightning_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)
    dataset = IndexDataset(config)

    with pytest.raises(AssertionError, match="Please call pl.seed_everything"):
        _sample_rng(dataset, 0)


def test_explicit_seed_does_not_depend_on_lightning_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)
    dataset = IndexDataset(config, seed=0)

    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    first = _sample_rng(dataset, 0)
    monkeypatch.setenv("PL_GLOBAL_SEED", "1")
    assert _sample_rng(dataset, 0) == first


def test_registry_preserves_environment_fallback_when_seed_is_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PL_GLOBAL_SEED", "7")
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)

    dataset = make(config.name, config, "train")

    assert isinstance(dataset, IndexDataset)
    assert _sample_rng(dataset, 0) == _expected_sample(epoch=-1, batch_idx=0, seed=7)


def test_registry_omits_keyword_for_legacy_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)
    calls: list[tuple[IndexDatasetConfig, str]] = []

    def legacy_constructor(dataset_config: IndexDatasetConfig, split: str) -> IndexDataset:
        calls.append((dataset_config, split))
        return IndexDataset(dataset_config, split)

    monkeypatch.setitem(registry_module.datasets, "legacy", legacy_constructor)

    omitted = registry_module.make("legacy", config, "train")
    explicit_none = registry_module.make("legacy", config, "train", seed=None)
    with pytest.raises(TypeError, match="unexpected keyword argument 'seed'"):
        registry_module.make("legacy", config, "train", seed=1)

    assert isinstance(omitted, IndexDataset)
    assert isinstance(explicit_none, IndexDataset)
    assert calls == [(config, "train"), (config, "train")]


def test_registry_forwards_explicit_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)

    zero = make(config.name, config, "train", seed=0)
    other = make(config.name, config, "train", seed=12)

    assert isinstance(zero, IndexDataset)
    assert isinstance(other, IndexDataset)
    assert _sample_rng(zero, 0) == _expected_sample(epoch=-1, batch_idx=0, seed=0)
    assert _sample_rng(other, 0) == _expected_sample(epoch=-1, batch_idx=0, seed=12)
    assert _sample_rng(zero, 0) != _sample_rng(other, 0)


def test_registry_rejects_constructor_that_drops_explicit_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PL_GLOBAL_SEED", "7")
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)

    def drops_seed(
        dataset_config: IndexDatasetConfig,
        split: str,
        *,
        seed: int | None = None,
    ) -> IndexDataset:
        return IndexDataset(dataset_config, split)

    monkeypatch.setitem(registry_module.datasets, "drops-seed", drops_seed)

    with pytest.raises(RuntimeError, match="accepted seed=12 but did not forward"):
        registry_module.make("drops-seed", config, "train", seed=12)


def test_mixed_dataset_forwards_explicit_seed_to_components() -> None:
    component_config = IndexDatasetConfig(name="instant_nurec-test-index", size=2)
    config = InstantNuRecMixedDatasetConfig(
        name="instant_nurec-mixed",
        mixture={
            "component": InstantNuRecMixedDatasetConfig.MixtureComponentConfig(
                sample_ratio=1.0,
                config=component_config,
            )
        },
    )

    dataset = MixedInstantNuRecDataset(config, seed=23)
    component = dataset.datasets[0].dataset

    assert isinstance(component, IndexDataset)
    assert _sample_rng(component, 0) == _sample_rng(IndexDataset(component_config, seed=23), 0)


@pytest.mark.parametrize("dataset_size", [1, 3, 10, 12])
def test_failed_dataset_retries_are_bounded_even_when_rng_repeats(
    dataset_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = IndexDataset(IndexDatasetConfig(name="instant_nurec-test-index", size=dataset_size), seed=0)
    attempted: list[int] = []
    draw_count = 0
    rng = MagicMock()

    def draw_zero(low: int, high: int) -> int:
        nonlocal draw_count
        draw_count += 1
        # Fail quickly, rather than hanging the test, if a rejection loop returns.
        assert draw_count <= dataset_size, "Retry selection did not terminate"
        assert low == 0 and high > 0
        return 0

    def fail(batch_idx: int, rng: np.random.Generator) -> NoReturn:
        attempted.append(batch_idx)
        raise InstantNuRecDataError("No synchronized context available")

    rng.integers.side_effect = draw_zero
    monkeypatch.setattr(dataset, "_get_rng", lambda batch_idx: rng)
    monkeypatch.setattr(dataset, "getitem_allow_exceptions", fail)
    expected_attempts = min(10, dataset_size)

    with pytest.raises(InstantNuRecDataError, match=f"tried out {expected_attempts} attempts") as error:
        dataset[0]

    assert isinstance(error.value.__cause__, InstantNuRecDataError)
    assert error.value.__cause__.message == "No synchronized context available"
    assert attempted == list(range(expected_attempts))
    assert draw_count == expected_attempts - 1


@pytest.fixture
def subsampler() -> CameraSubsampler:
    return CameraSubsampler(CameraSubsamplerConfig(frame_width=3, frame_height=3))


def _camera_parameters(
    resolution: tuple[int, int] = (8, 5),
) -> OpenCVPinholeCameraModelParameters:
    width, height = resolution
    return OpenCVPinholeCameraModelParameters(
        resolution=np.array(resolution, dtype=np.uint64),
        focal_length=np.array([width / 2, height], dtype=np.float32),
        principal_point=np.array([width / 2, height / 2], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
    )


def test_compute_pixel_rect_characterization(subsampler: CameraSubsampler) -> None:
    pixel_rect, resized_pre_crop_resolution = subsampler._compute_pixel_rect(8, 5)

    assert resized_pre_crop_resolution == (5, 3)
    assert (pixel_rect.width, pixel_rect.height) == (3, 3)
    assert (pixel_rect.i, pixel_rect.j) == (1, 0)
    assert pixel_rect.subsample_factor == pytest.approx(5 / 3)
    assert pixel_rect.image_transform.image_domain_scale == 0.6
    assert pixel_rect.image_transform.resized_pre_crop_resolution == (5, 3)


def test_apply_camera_parameters_characterization(subsampler: CameraSubsampler) -> None:
    transformed = subsampler.apply_camera_parameters(_camera_parameters())

    assert isinstance(transformed, OpenCVPinholeCameraModelParameters)
    np.testing.assert_array_equal(transformed.resolution, [3, 3])
    np.testing.assert_allclose(transformed.focal_length, [2.4, 3.0])
    np.testing.assert_allclose(transformed.principal_point, [1.4, 1.5])


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "bilinear",
            [
                [2.7499974, 4.4999957, 6.2499943],
                [18.749983, 20.999958, 22.249979],
                [34.749966, 36.499966, 38.24996],
            ],
        ),
        (
            "nearest-min",
            [
                [2.0, 4.0, 5.0],
                [10.0, 12.0, 13.0],
                [26.0, 28.0, 29.0],
            ],
        ),
    ],
)
def test_apply_depth_data_characterization(
    subsampler: CameraSubsampler,
    mode: Literal["bilinear", "nearest-min"],
    expected: list[list[float]],
) -> None:
    depth = np.arange(1, 41, dtype=np.float32).reshape(5, 8)
    depth[2, 3] = 0.0

    resized = subsampler.apply_depth_data(depth, mode=mode)

    np.testing.assert_allclose(resized, expected, rtol=1e-5, atol=1e-5)


def test_apply_frame_data_read_only_2d_float_characterization(subsampler: CameraSubsampler) -> None:
    frame = np.arange(5 * 8, dtype=np.float32).reshape(5, 8)
    frame.flags.writeable = False

    resized = subsampler.apply_frame_data(frame)

    np.testing.assert_allclose(
        resized,
        [
            [7.7942863, 9.634285, 11.380954],
            [22.08, 23.919998, 25.666668],
            [32.08, 33.92, 35.666668],
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    assert resized.dtype == np.float32


def test_apply_frame_data_writable_3d_integer_characterization(subsampler: CameraSubsampler) -> None:
    frame = np.arange(5 * 8 * 3, dtype=np.uint8).reshape(5, 8, 3)

    resized = subsampler.apply_frame_data(frame)

    np.testing.assert_array_equal(
        resized,
        [
            [[3, 4, 5], [9, 10, 11], [12, 13, 14]],
            [[27, 28, 29], [33, 34, 35], [36, 37, 38]],
            [[75, 76, 77], [81, 82, 83], [84, 85, 86]],
        ],
    )
    assert resized.dtype == np.uint8


def test_apply_frame_data_rejects_invalid_rank(subsampler: CameraSubsampler) -> None:
    with pytest.raises(AssertionError):
        subsampler.apply_frame_data(np.zeros((1, 5, 8, 3), dtype=np.uint8))


def test_half_tie_resolution_uses_exact_canonical_resized_pre_crop_size() -> None:
    subsampler = CameraSubsampler(CameraSubsamplerConfig(frame_width=3, frame_height=2))
    frame = np.zeros((35, 14, 3), dtype=np.uint8)
    depth = np.ones((35, 14), dtype=np.float32)

    pixel_rect, resized_pre_crop_resolution = subsampler._compute_pixel_rect(14, 35)

    assert resized_pre_crop_resolution == (3, 8)
    assert pixel_rect.downsampled_resolution() == (3, 8)
    assert subsampler.apply_frame_data(frame).shape == (2, 3, 3)
    assert subsampler.apply_depth_data(depth, mode="bilinear").shape == (2, 3)


def test_apply_timestamps_characterization(subsampler: CameraSubsampler) -> None:
    poses = np.stack([np.eye(4), np.eye(4)])

    cropped_poses, start_timestamp, end_timestamp = subsampler.apply_T_sensor_startend_timestamps(
        poses,
        100,
        200,
        _camera_parameters((5, 8)),
    )

    np.testing.assert_array_equal(cropped_poses, poses)
    assert start_timestamp == 128
    assert end_timestamp == 185
