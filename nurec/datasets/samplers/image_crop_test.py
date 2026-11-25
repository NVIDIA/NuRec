# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from omegaconf import OmegaConf
from pydantic import ValidationError

from nurec.datasets.samplers.image_crop import ImageCropCameraPixelSampler
from nurec.utils.batch import RectSubsampled


_UNSET = object()


def _config(
    *,
    crop_type: str = "full_image",
    subsample: int | None | object = _UNSET,
    resolution: dict | list[dict] | None | object = _UNSET,
    roi_lt: list[float] | None = None,
    roi_rb: list[float] | None = None,
    crop_ratio: float | None = None,
):
    if subsample is _UNSET and resolution is _UNSET:
        subsample = None
        resolution = {"mode": "relative", "subsample": 2}
    elif subsample is _UNSET:
        subsample = None
    elif resolution is _UNSET:
        resolution = None
    return OmegaConf.create(
        {
            "crop_type": crop_type,
            "subsample": subsample,
            "resolution": resolution,
            "roi_lt": roi_lt,
            "roi_rb": roi_rb,
            "crop_ratio": crop_ratio,
        }
    )


def _dataset(
    camera_resolutions: dict[str, tuple[int, int]] | None = None,
    *,
    unique_camera_ids: dict[str, str] | None = None,
) -> Mock:
    camera_resolutions = camera_resolutions or {"front": (1920, 1080)}
    unique_camera_ids = unique_camera_ids or {
        camera_id: camera_id if len(camera_resolutions) > 1 else "camera" for camera_id in camera_resolutions
    }
    dataset = Mock()
    dataset.camera_ids = list(camera_resolutions)
    dataset.datasource = SimpleNamespace(
        camera_models={
            camera_id: SimpleNamespace(resolution=np.asarray(resolution))
            for camera_id, resolution in camera_resolutions.items()
        },
        camera_unique_ids={
            camera_id: SimpleNamespace(id=unique_camera_ids[camera_id]) for camera_id in camera_resolutions
        },
    )
    return dataset


def _sample(
    sampler: ImageCropCameraPixelSampler,
    *,
    seed: int = 0,
    unique_camera_id: str = "camera",
    source_resolution: tuple[int, int] = (1920, 1080),
) -> RectSubsampled:
    result = sampler.sample_camera_pixels(
        np.random.default_rng(seed),
        batch_idx=0,
        frame_range=range(1),
        n_frame_pixel_samples=0,
        frame_all_pixels=np.empty((0, 2)),
        frame_valid_pixels_mask=np.ones(source_resolution[::-1], dtype=bool),
        unique_camera_id=unique_camera_id,
        camera_frame_idx=0,
    )
    assert isinstance(result.sampled_pixels, RectSubsampled)
    return result.sampled_pixels


def test_full_image_crop_characterization() -> None:
    sampled = _sample(ImageCropCameraPixelSampler(_config(), _dataset()))

    assert sampled.to_json() == {
        "source_resolution": (1920, 1080),
        "resized_pre_crop_resolution": (960, 540),
        "output_resolution": (960, 540),
        "image_domain_offset": (0, 0),
        "image_domain_scale": 0.5,
    }


def test_runtime_sampler_requires_canonical_resolution() -> None:
    legacy_config = _config(subsample=2)
    del legacy_config["resolution"]

    with pytest.raises(ValueError, match="must specify either relative or absolute"):
        ImageCropCameraPixelSampler(legacy_config, _dataset())


def test_runtime_sampler_uses_normalized_resolution_only() -> None:
    config = OmegaConf.create(
        {
            "crop_type": "full_image",
            "subsample": 3,
            "resolution": {
                "mode": "relative",
                "subsample": 2,
            },
            "roi_lt": None,
            "roi_rb": None,
            "crop_ratio": None,
        }
    )

    sampler = ImageCropCameraPixelSampler(config, _dataset({"front": (1920, 1080)}))

    sampled = _sample(sampler)
    assert (sampled.width, sampled.height) == (960, 540)


def test_unshipped_camera_group_list_is_rejected() -> None:
    config = _config(resolution=[{"camera_ids": ["front"], "width": 960}])

    with pytest.raises(ValidationError):
        ImageCropCameraPixelSampler(config, _dataset())


@pytest.mark.parametrize(
    ("resolution", "expected_size", "expected_factor"),
    [
        ({"width": 960, "keep_aspect_ratio": True}, (960, 540), 2.0),
        ({"width": None, "height": 540, "keep_aspect_ratio": True}, (960, 540), 2.0),
        (
            {"width": 800, "height": 600, "keep_aspect_ratio": False},
            (800, 600),
            (2.4, 1.8),
        ),
    ],
)
def test_full_image_crop_supports_absolute_resolution(
    resolution: dict,
    expected_size: tuple[int, int],
    expected_factor: float | tuple[float, float],
) -> None:
    sampled = _sample(
        ImageCropCameraPixelSampler(
            _config(
                resolution={
                    "mode": "absolute",
                    "camera_groups": [{"camera_ids": ["front"], **resolution}],
                }
            ),
            _dataset(),
        )
    )

    assert (sampled.width, sampled.height) == expected_size
    assert sampled.subsample_factor == pytest.approx(expected_factor)
    assert sampled.image_transform.source_resolution == (1920, 1080)
    assert sampled.image_transform.resized_pre_crop_resolution == expected_size


def test_image_crop_ignores_legacy_scalar_after_schema_normalization() -> None:
    sampler = ImageCropCameraPixelSampler(
        _config(
            subsample=2,
            resolution={
                "mode": "absolute",
                "camera_groups": [{"camera_ids": ["front"], "width": 960}],
            },
        ),
        _dataset(),
    )
    assert _sample(sampler).image_transform.source_resolution == (1920, 1080)


def test_absolute_anisotropic_crop_applies_exact_requested_resolution() -> None:
    sampled = _sample(
        ImageCropCameraPixelSampler(
            _config(
                resolution={
                    "mode": "absolute",
                    "camera_groups": [
                        {
                            "camera_ids": ["front"],
                            "width": 800,
                            "height": 600,
                            "keep_aspect_ratio": False,
                        }
                    ],
                },
            ),
            _dataset(),
        )
    )

    resized = sampled.apply_to_array(
        np.zeros((1080, 1920, 1), dtype=np.uint8),
        interpolation=0,
    )

    assert resized.shape == (600, 800, 1)


def test_absolute_keep_aspect_crop_supports_rounded_dimension() -> None:
    sampled = _sample(
        ImageCropCameraPixelSampler(
            _config(
                resolution={
                    "mode": "absolute",
                    "camera_groups": [{"camera_ids": ["front"], "width": 959}],
                }
            ),
            _dataset(),
        )
    )

    resized = sampled.apply_to_array(
        np.zeros((1080, 1920, 1), dtype=np.uint8),
        interpolation=0,
    )

    assert sampled.image_transform.source_resolution == (1920, 1080)
    assert resized.shape == (539, 959, 1)


def test_preset_roi_characterization() -> None:
    sampler = ImageCropCameraPixelSampler(
        _config(crop_type="preset_roi", roi_lt=[0.25, 0.2], roi_rb=[0.75, 0.8]),
        _dataset(),
    )

    sampled = _sample(sampler)

    assert (sampled.i, sampled.j, sampled.width, sampled.height) == (240, 108, 480, 324)


def test_random_crop_characterization() -> None:
    sampler = ImageCropCameraPixelSampler(_config(crop_type="random_crop", crop_ratio=0.5), _dataset())

    sampled = _sample(sampler, seed=7)

    assert (sampled.i, sampled.j, sampled.width, sampled.height) == (430, 168, 480, 270)


@pytest.mark.parametrize(
    "config",
    [
        _config(crop_type="preset_roi", roi_lt=[0.0, 0.0], roi_rb=None),
        _config(crop_type="preset_roi", roi_lt=[0.0, 0.0], roi_rb=[0.5]),
        _config(crop_type="preset_roi", roi_lt=[0.0], roi_rb=[0.5, 0.5]),
        _config(crop_type="preset_roi", roi_lt=None, roi_rb=[0.5, 0.5]),
        _config(crop_type="preset_roi", roi_lt=[0.0, 0.0], roi_rb=[1.1, 0.5]),
        _config(crop_type="preset_roi", roi_lt=[0.0, 0.0], roi_rb=[-0.1, 0.5]),
        _config(crop_type="preset_roi", roi_lt=[-0.1, 0.0], roi_rb=[0.5, 0.5]),
        _config(crop_type="preset_roi", roi_lt=[0.0, 1.1], roi_rb=[0.5, 0.5]),
        _config(crop_type="random_crop", crop_ratio=None),
        _config(crop_type="random_crop", crop_ratio=-0.1),
        _config(crop_type="random_crop", crop_ratio=1.1),
    ],
)
def test_invalid_crop_configuration_is_rejected(config) -> None:
    with pytest.raises(AssertionError):
        ImageCropCameraPixelSampler(config, _dataset())


def test_unknown_crop_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported crop type"):
        ImageCropCameraPixelSampler(_config(crop_type="unknown"), _dataset())


@pytest.mark.parametrize("subsample", [7, 16])
def test_subsample_must_divide_source_resolution(subsample: int) -> None:
    with pytest.raises(AssertionError, match=rf"Subsample factor {subsample} invalid"):
        ImageCropCameraPixelSampler(_config(resolution={"mode": "relative", "subsample": subsample}), _dataset())


def test_unreachable_unknown_crop_type_remains_defensive() -> None:
    sampler = ImageCropCameraPixelSampler(_config(), _dataset())
    sampler.crop_type = "unknown"

    with pytest.raises(ValueError, match="Unsupported crop type"):
        _sample(sampler)


def test_absolute_camera_groups_resolve_each_active_camera() -> None:
    sampler = ImageCropCameraPixelSampler(
        _config(
            resolution={
                "mode": "absolute",
                "camera_groups": [
                    {"camera_ids": ["front"], "width": 960},
                    {"camera_ids": ["rear"], "width": None, "height": 360},
                ],
            },
        ),
        _dataset(
            {"front": (1920, 1080), "rear": (1280, 720)},
            unique_camera_ids={"front": "front-unique", "rear": "rear-unique"},
        ),
    )

    front = _sample(sampler, unique_camera_id="front-unique")
    rear = _sample(
        sampler,
        unique_camera_id="rear-unique",
        source_resolution=(1280, 720),
    )

    assert (front.width, front.height) == (960, 540)
    assert (rear.width, rear.height) == (640, 360)


def test_absolute_camera_group_rejects_mixed_source_resolutions() -> None:
    with pytest.raises(ValueError, match="split cameras with different source resolutions"):
        ImageCropCameraPixelSampler(
            _config(
                resolution={
                    "mode": "absolute",
                    "camera_groups": [{"camera_ids": ["front", "rear"], "width": 640}],
                },
            ),
            _dataset({"front": (1920, 1080), "rear": (1280, 720)}),
        )


def test_fixed_sampled_pixels_uses_precomputed_camera_transform() -> None:
    sampler = ImageCropCameraPixelSampler(_config(), _dataset())

    sampled = sampler.fixed_sampled_pixels(
        camera_id="front",
        original_width=1920,
        original_height=1080,
    )

    assert sampled is not None
    assert (sampled.width, sampled.height) == (960, 540)


def test_fixed_sampled_pixels_rejects_non_full_image_crop() -> None:
    sampler = ImageCropCameraPixelSampler(
        _config(crop_type="random_crop", crop_ratio=0.5),
        _dataset(),
    )

    assert (
        sampler.fixed_sampled_pixels(
            camera_id="front",
            original_width=1920,
            original_height=1080,
        )
        is None
    )
