# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from dataclasses import asdict

import cv2
import numpy as np
import pytest

from nurec.utils.image_resolution import ImageResolutionTransform, ResolutionRounding, resize_image_array


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_transform_rejects_invalid_geometric_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ImageResolutionTransform((8, 6), (4, 3), (4, 3), scale)


def test_anisotropic_transform_json_round_trip_is_hashable() -> None:
    transform = ImageResolutionTransform.from_subsample((8, 6), (2.0, 3.0))
    restored = ImageResolutionTransform(**json.loads(json.dumps(asdict(transform))))
    assert restored == transform
    assert hash(restored) == hash(transform)


@pytest.mark.parametrize("rounding", ["round", "floor"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("source", [(3840, 2160), (10, 7), (10, 9)])
def test_absolute_and_relative_share_rounding_and_uniform_geometry(
    rounding: ResolutionRounding, axis: int, source: tuple[int, int]
) -> None:
    requested = source[axis] // 2
    relative = ImageResolutionTransform.from_subsample(source, source[axis] / requested, rounding=rounding)
    absolute = ImageResolutionTransform.from_absolute(
        source,
        width=requested if axis == 0 else None,
        height=requested if axis == 1 else None,
        keep_aspect_ratio=True,
        rounding=rounding,
    )
    assert absolute == relative


def test_height_fit_preserves_legacy_truncation_and_uniform_scale() -> None:
    transform = ImageResolutionTransform.fit_height((3840, 2160), 1000)
    assert transform.output_resolution == (1777, 1000)
    assert transform.image_domain_scale == 1000 / 2160


@pytest.mark.parametrize("factor", [49, 93])
def test_integer_stride_tolerates_reciprocal_roundoff(factor: int) -> None:
    transform = ImageResolutionTransform.from_subsample((factor * 2, factor), factor)
    yy, xx = np.indices((factor, factor * 2))
    pixels, half_offset = transform.sample_integer_stride_pixel_grid(np.stack((xx, yy), axis=-1))
    offset = (factor - 1) // 2
    np.testing.assert_array_equal(pixels, [[offset, offset], [factor + offset, offset]])
    assert not half_offset


def test_plain_array_resize_rejects_missing_channel_axis() -> None:
    with pytest.raises(AssertionError, match="Array must be 3D"):
        resize_image_array(np.zeros((4, 3)), (2, 2), cv2.INTER_NEAREST)


def test_crop_corners_preserve_float32_shutter_scanline_selection() -> None:
    transform = ImageResolutionTransform.from_subsample((70, 70), 3.5, output_resolution=(17, 17))
    corners = transform.rect_points_lb()
    expected = 0.5 + 3.5 * np.array([[0, 0], [17 - 1 / 3.5, 17 - 1 / 3.5]], dtype=np.float32)
    np.testing.assert_array_equal(corners, expected)
    assert np.floor(corners[1, 1]) == 58


@pytest.mark.parametrize(
    "source_resolution",
    [(0, 6), (8, 0)],
)
def test_transform_requires_positive_source_resolution(source_resolution: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="source_resolution"):
        ImageResolutionTransform(
            source_resolution=source_resolution,
            resized_pre_crop_resolution=(4, 3),
            output_resolution=(3, 2),
            image_domain_scale=0.5,
        )


@pytest.mark.parametrize(
    "resized_pre_crop_resolution",
    [(0, 3), (4, 0)],
)
def test_transform_requires_positive_resized_pre_crop_resolution(
    resized_pre_crop_resolution: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="resized_pre_crop_resolution"):
        ImageResolutionTransform(
            source_resolution=(8, 6),
            resized_pre_crop_resolution=resized_pre_crop_resolution,
            output_resolution=(3, 2),
            image_domain_scale=0.5,
        )


@pytest.mark.parametrize(
    "output_resolution",
    [(0, 2), (3, 0)],
)
def test_transform_requires_positive_output_resolution(output_resolution: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="output_resolution"):
        ImageResolutionTransform(
            source_resolution=(8, 6),
            resized_pre_crop_resolution=(4, 3),
            output_resolution=output_resolution,
            image_domain_scale=0.5,
        )


@pytest.mark.parametrize("image_domain_offset", [(-1, 0), (0, -1)])
def test_transform_requires_non_negative_crop_offset(image_domain_offset: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="image_domain_offset must be non-negative"):
        ImageResolutionTransform(
            source_resolution=(8, 6),
            resized_pre_crop_resolution=(4, 3),
            output_resolution=(3, 2),
            image_domain_scale=0.5,
            image_domain_offset=image_domain_offset,
        )


@pytest.mark.parametrize("output_resolution", [(4, 2), (3, 3)])
def test_transform_requires_crop_inside_resized_pre_crop_image(output_resolution: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="exceeds resized pre-crop resolution"):
        ImageResolutionTransform(
            source_resolution=(8, 6),
            resized_pre_crop_resolution=(4, 3),
            output_resolution=output_resolution,
            image_domain_scale=0.5,
            image_domain_offset=(1, 1),
        )


def test_transform_requires_scale_to_match_resized_pre_crop_resolution() -> None:
    with pytest.raises(ValueError, match="image_domain_scale does not match"):
        ImageResolutionTransform(
            source_resolution=(3840, 2160),
            resized_pre_crop_resolution=(1000, 565),
            output_resolution=(1000, 565),
            image_domain_scale=1000 / 3840,
        )


def test_from_scalar_subsample_uses_full_resized_pre_crop_image_by_default() -> None:
    transform = ImageResolutionTransform.from_subsample((8, 6), 2.0)

    assert transform.source_resolution == (8, 6)
    assert transform.resized_pre_crop_resolution == (4, 3)
    assert transform.output_resolution == (4, 3)
    assert transform.image_domain_scale == 0.5
    assert transform.subsample_factor == 2.0


def test_from_anisotropic_subsample_preserves_scale_crop_and_geometry() -> None:
    transform = ImageResolutionTransform.from_subsample(
        (8, 6),
        (2.0, 3.0),
        output_resolution=(2, 1),
        image_domain_offset=(1, 1),
    )

    assert transform.resized_pre_crop_resolution == (4, 2)
    assert transform.output_resolution == (2, 1)
    assert transform.image_domain_scale == (0.5, 1.0 / 3.0)
    assert transform.subsample_factor == (2.0, 3.0)
    np.testing.assert_array_equal(
        transform.rect_points_lb(),
        [[2.5, 3.5], [5.5, 5.5]],
    )


@pytest.mark.parametrize("subsample_factor", [0.0, (1.0, 0.0)])
def test_from_subsample_requires_positive_factors(subsample_factor) -> None:
    with pytest.raises(AssertionError, match="Invalid subsample_factor"):
        ImageResolutionTransform.from_subsample((8, 6), subsample_factor)


@pytest.mark.parametrize("subsample_factor", [3.0, 4.0])
def test_from_subsample_can_require_divisible_source(subsample_factor: float) -> None:
    with pytest.raises(AssertionError, match="Subsample factor"):
        ImageResolutionTransform.from_subsample(
            (8, 6),
            subsample_factor,
            require_divisible=True,
        )


def test_from_subsample_accepts_divisible_source() -> None:
    transform = ImageResolutionTransform.from_subsample(
        (8, 6),
        2.0,
        require_divisible=True,
    )

    assert transform.resized_pre_crop_resolution == (4, 3)


def test_from_subsample_supports_legacy_floor_rounding() -> None:
    transform = ImageResolutionTransform.from_subsample(
        (8, 6),
        2.5,
        rounding="floor",
    )

    assert transform.resized_pre_crop_resolution == (3, 2)
    assert transform.image_domain_scale == 1 / 2.5


def test_from_subsample_rejects_unknown_rounding() -> None:
    with pytest.raises(AssertionError, match="Unsupported resolution rounding"):
        ImageResolutionTransform.from_subsample((8, 6), 2.0, rounding="ceiling")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source_resolution", "absolute_resolution"),
    [
        ((8, 1), {"width": 1, "height": None}),
        ((1, 8), {"width": None, "height": 1}),
    ],
)
def test_from_absolute_rejects_resolution_with_zero_pixel_derived_axis(
    source_resolution: tuple[int, int],
    absolute_resolution: dict,
) -> None:
    with pytest.raises(AssertionError, match="must be positive"):
        ImageResolutionTransform.from_absolute(
            source_resolution,
            **absolute_resolution,
            keep_aspect_ratio=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "expected_resolution", "expected_scale"),
    [
        ({"width": 4, "height": None, "keep_aspect_ratio": True}, (4, 3), 0.5),
        ({"width": None, "height": 3, "keep_aspect_ratio": True}, (4, 3), 0.5),
        ({"width": 3, "height": 2, "keep_aspect_ratio": False}, (3, 2), (0.375, 1.0 / 3.0)),
    ],
)
def test_from_absolute_supported_forms(
    kwargs: dict,
    expected_resolution: tuple[int, int],
    expected_scale: float | tuple[float, float],
) -> None:
    transform = ImageResolutionTransform.from_absolute((8, 6), **kwargs)

    assert transform.resized_pre_crop_resolution == expected_resolution
    assert transform.output_resolution == expected_resolution
    assert transform.image_domain_scale == expected_scale


def test_from_absolute_aspect_ratio_preserves_uniform_scale_after_rounding() -> None:
    transform = ImageResolutionTransform.from_absolute(
        (8, 5),
        width=3,
        height=None,
        keep_aspect_ratio=True,
    )

    assert transform.output_resolution == (3, 2)
    assert transform.image_domain_scale == 3 / 8
    reconstructed = ImageResolutionTransform.from_subsample((8, 5), transform.subsample_factor)
    assert reconstructed.output_resolution == transform.output_resolution


def test_from_absolute_rounded_resolution_preserves_legacy_uniform_intrinsics() -> None:
    transform = ImageResolutionTransform.from_absolute(
        (3840, 2160),
        width=1000,
        height=None,
        keep_aspect_ratio=True,
    )
    parameters = _FakeCameraParameters((3840, 2160))

    transform.apply_to_camera_parameters(parameters)

    assert transform.output_resolution == (1000, 562)
    assert transform.image_domain_scale == 1000 / 3840
    assert parameters.transform_args == {
        "image_domain_scale": 1000 / 3840,
        "new_resolution": (1000, 562),
    }
    np.testing.assert_allclose(
        transform.rect_points_lb(),
        [[0.5, 0.5], [3839.5, 2157.58]],
    )


@pytest.mark.parametrize(
    ("source_resolution", "kwargs", "expected_resolution"),
    [
        ((4, 2), {"width": 3, "height": None}, (3, 2)),
        ((2, 4), {"width": None, "height": 3}, (2, 3)),
        ((10, 45), {"width": 7, "height": None}, (7, 32)),
        ((45, 10), {"width": None, "height": 7}, (32, 7)),
    ],
)
def test_from_absolute_aspect_ratio_reuses_subsample_rounding(
    source_resolution: tuple[int, int],
    kwargs: dict,
    expected_resolution: tuple[int, int],
) -> None:
    transform = ImageResolutionTransform.from_absolute(
        source_resolution,
        **kwargs,
        keep_aspect_ratio=True,
    )

    assert transform.output_resolution == expected_resolution


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"width": 0, "height": 2, "keep_aspect_ratio": False}, "width must be positive"),
        ({"width": 3, "height": 0, "keep_aspect_ratio": False}, "height must be positive"),
        (
            {"width": 4, "height": 3, "keep_aspect_ratio": True},
            "Exactly one of width or height",
        ),
        ({"width": None, "height": None, "keep_aspect_ratio": True}, "Exactly one of width or height"),
        ({"width": 4, "height": None, "keep_aspect_ratio": False}, "Both width and height"),
    ],
)
def test_from_absolute_rejects_unsupported_forms(kwargs: dict, message: str) -> None:
    with pytest.raises(AssertionError, match=message):
        ImageResolutionTransform.from_absolute((8, 6), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 16, "height": None, "keep_aspect_ratio": True},
        {"width": 8, "height": 12, "keep_aspect_ratio": False},
    ],
)
def test_from_absolute_rejects_upsampling_by_default(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="allow_upsampling=true"):
        ImageResolutionTransform.from_absolute((8, 6), **kwargs)


def test_from_absolute_allows_explicit_upsampling() -> None:
    transform = ImageResolutionTransform.from_absolute(
        (8, 6),
        width=16,
        height=9,
        keep_aspect_ratio=False,
        allow_upsampling=True,
    )

    assert transform.output_resolution == (16, 9)
    assert transform.image_domain_scale == (2.0, 1.5)


@pytest.mark.parametrize(
    ("requested_resolution", "expected_resolution", "expected_scale"),
    [
        ((4, 1), (4, 3), 0.5),
        ((1, 3), (4, 3), 0.5),
    ],
)
def test_fit_keep_aspect_ratio_characterization(
    requested_resolution: tuple[int, int],
    expected_resolution: tuple[int, int],
    expected_scale: float,
) -> None:
    transform = ImageResolutionTransform.fit_keep_aspect_ratio(
        (8, 6),
        requested_resolution,
    )

    assert transform.resized_pre_crop_resolution == expected_resolution
    assert transform.output_resolution == expected_resolution
    assert transform.image_domain_scale == expected_scale


@pytest.mark.parametrize(
    ("source_resolution", "expected_resized_pre_crop_resolution", "expected_offset"),
    [
        ((8, 4), (6, 3), (1, 0)),
        ((4, 8), (3, 6), (0, 1)),
        ((8, 5), (5, 3), (1, 0)),
    ],
)
def test_cover_and_center_crop(
    source_resolution: tuple[int, int],
    expected_resized_pre_crop_resolution: tuple[int, int],
    expected_offset: tuple[int, int],
) -> None:
    transform = ImageResolutionTransform.cover_and_center_crop(source_resolution, (3, 3))

    assert transform.resized_pre_crop_resolution == expected_resized_pre_crop_resolution
    assert transform.output_resolution == (3, 3)
    assert transform.image_domain_offset == expected_offset


def test_cover_and_center_crop_preserves_uniform_scale_after_rounding() -> None:
    transform = ImageResolutionTransform.cover_and_center_crop((8, 5), (3, 3))

    assert transform.image_domain_scale == 3 / 5


def _crop_transform() -> ImageResolutionTransform:
    return ImageResolutionTransform.from_subsample(
        (8, 6),
        2.0,
        output_resolution=(3, 2),
        image_domain_offset=(1, 1),
    )


def test_crop_array_characterization() -> None:
    resized_pre_crop = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4, 1)

    cropped = _crop_transform().crop_array(resized_pre_crop)

    np.testing.assert_array_equal(cropped[..., 0], [[5, 6, 7], [9, 10, 11]])


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.zeros((3, 4)), "Array must be 3D"),
        (np.zeros((2, 4, 1)), r"Array height \(2\) does not match resized pre-crop height \(3\)"),
        (np.zeros((3, 3, 1)), r"Array width \(3\) does not match resized pre-crop width \(4\)"),
    ],
)
def test_crop_array_validates_resized_pre_crop_shape(array: np.ndarray, message: str) -> None:
    with pytest.raises(AssertionError, match=message):
        _crop_transform().crop_array(array)


@pytest.mark.parametrize(
    ("subsample_factor", "expected_pixels", "expected_half_pixel_offset"),
    [
        (2.0, [[2, 0], [4, 0], [2, 2], [4, 2]], True),
        (3.0, [[1, 1], [4, 1]], False),
    ],
)
def test_sample_integer_stride_pixel_grid(
    subsample_factor: float,
    expected_pixels: list[list[int]],
    expected_half_pixel_offset: bool,
) -> None:
    source_resolution = (6, 4) if subsample_factor == 2.0 else (6, 3)
    transform = ImageResolutionTransform.from_subsample(
        source_resolution,
        subsample_factor,
        output_resolution=(2, 2 if subsample_factor == 2.0 else 1),
        image_domain_offset=(1 if subsample_factor == 2.0 else 0, 0),
    )
    width, height = source_resolution
    yy, xx = np.indices((height, width))
    camera_all_pixels = np.stack((xx, yy), axis=-1).reshape(-1, 2)

    pixels, half_pixel_offset = transform.sample_integer_stride_pixel_grid(camera_all_pixels)

    np.testing.assert_array_equal(pixels, expected_pixels)
    assert half_pixel_offset is expected_half_pixel_offset


def test_sample_integer_stride_pixel_grid_rejects_anisotropic_subsampling() -> None:
    transform = ImageResolutionTransform.from_subsample((6, 4), (2.0, 2.0))

    with pytest.raises(AssertionError, match="only supports isotropic"):
        transform.sample_integer_stride_pixel_grid(np.zeros((24, 2)))


def test_sample_integer_stride_pixel_grid_rejects_rounded_non_integer_subsampling() -> None:
    transform = ImageResolutionTransform.from_subsample((6, 4), 1.5)

    with pytest.raises(AssertionError, match="must be an integer"):
        transform.sample_integer_stride_pixel_grid(np.zeros((24, 2)))


def test_sample_integer_stride_pixel_grid_rejects_isotropic_non_integer_subsampling() -> None:
    transform = ImageResolutionTransform.from_subsample((6, 3), 1.5)

    with pytest.raises(AssertionError, match="must be an integer"):
        transform.sample_integer_stride_pixel_grid(np.zeros((18, 2)))


def test_apply_to_array_without_resize() -> None:
    transform = ImageResolutionTransform.from_subsample(
        (4, 3),
        1.0,
        output_resolution=(2, 2),
        image_domain_offset=(1, 1),
    )
    image = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4, 1)

    result = transform.apply_to_array(image, interpolation=cv2.INTER_NEAREST)

    np.testing.assert_array_equal(result[..., 0], [[5, 6], [9, 10]])


def test_apply_to_array_cv2_resize_characterization() -> None:
    image = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8, 1)

    result = _crop_transform().apply_to_array(image, interpolation=cv2.INTER_NEAREST)

    np.testing.assert_array_equal(result[..., 0], [[18, 20, 22], [34, 36, 38]])


def test_apply_anisotropic_resize_to_array_and_camera_parameters() -> None:
    transform = ImageResolutionTransform.from_subsample((8, 6), (2.0, 3.0))
    image = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8, 1)
    parameters = _FakeCameraParameters((8, 6))

    resized = transform.apply_to_array(image, interpolation=cv2.INTER_NEAREST)
    transform.apply_to_camera_parameters(parameters)

    np.testing.assert_array_equal(
        resized[..., 0],
        [[0, 2, 4, 6], [24, 26, 28, 30]],
    )
    assert parameters.transform_args == {
        "image_domain_scale": (0.5, 1.0 / 3.0),
        "new_resolution": (4, 2),
    }


def test_apply_to_array_pil_resize_characterization() -> None:
    image = np.full((6, 8, 3), [17, 31, 47], dtype=np.uint8)

    result = _crop_transform().apply_to_array(
        image,
        interpolation=None,
        use_pil_lanczos=True,
    )

    np.testing.assert_array_equal(
        result,
        np.broadcast_to(np.array([17, 31, 47], dtype=np.uint8), (2, 3, 3)),
    )


def test_apply_to_array_requires_interpolation_for_cv2() -> None:
    with pytest.raises(AssertionError, match="interpolation is required"):
        _crop_transform().apply_to_array(
            np.zeros((6, 8, 1)),
            interpolation=None,
        )


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.zeros((6, 8)), "Array must be 3D"),
        (np.zeros((5, 8, 1)), "Image height"),
        (np.zeros((6, 7, 1)), "Image width"),
    ],
)
def test_apply_to_array_validates_source_shape(array: np.ndarray, message: str) -> None:
    with pytest.raises(AssertionError, match=message):
        _crop_transform().apply_to_array(array, interpolation=cv2.INTER_NEAREST)


class _FakeCameraParameters:
    def __init__(
        self,
        resolution: tuple[int, int],
        *,
        returned_resolution: tuple[int, int] | None = None,
    ) -> None:
        self.resolution = np.asarray(resolution, dtype=np.uint64)
        self.returned_resolution = returned_resolution
        self.transform_args = None

    def transform(self, **kwargs):
        self.transform_args = kwargs
        return _FakeCameraParameters(self.returned_resolution or kwargs["new_resolution"])


def test_apply_to_camera_parameters_omits_zero_crop_offset() -> None:
    parameters = _FakeCameraParameters((8, 6))
    transform = ImageResolutionTransform.from_subsample((8, 6), 2.0)

    transformed = transform.apply_to_camera_parameters(parameters)

    assert parameters.transform_args == {
        "image_domain_scale": 0.5,
        "new_resolution": (4, 3),
    }
    np.testing.assert_array_equal(transformed.resolution, [4, 3])


def test_apply_to_camera_parameters_passes_nonzero_crop_offset() -> None:
    parameters = _FakeCameraParameters((8, 6))
    transform = ImageResolutionTransform.from_subsample(
        (8, 6),
        2.0,
        output_resolution=(2, 2),
        image_domain_offset=(1, 1),
    )

    transform.apply_to_camera_parameters(parameters)

    assert parameters.transform_args is not None
    assert parameters.transform_args["image_domain_offset"] == (1, 1)


def test_apply_to_camera_parameters_validates_source_resolution() -> None:
    transform = ImageResolutionTransform.from_subsample((8, 6), 2.0)

    with pytest.raises(AssertionError, match="Camera resolution"):
        transform.apply_to_camera_parameters(_FakeCameraParameters((7, 6)))


def test_apply_to_camera_parameters_validates_result_resolution() -> None:
    transform = ImageResolutionTransform.from_subsample((8, 6), 2.0)

    with pytest.raises(AssertionError):
        transform.apply_to_camera_parameters(
            _FakeCameraParameters((8, 6), returned_resolution=(2, 2)),
        )
