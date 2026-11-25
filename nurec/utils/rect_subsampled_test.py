# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from typing import Any, cast
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from ncore.data import (
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from nurec.utils.batch import RectSubsampled
from nurec.utils.image_resolution import ImageResolutionTransform


def _rect(**overrides) -> RectSubsampled:
    values: dict[str, Any] = {
        "original_width": 8,
        "original_height": 6,
        "width": 3,
        "height": 2,
        "i": 1,
        "j": 1,
        "subsample_factor": 2.0,
    }
    values.update(overrides)
    return RectSubsampled.from_subsample(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"subsample_factor": 0.0},
        {"i": -1},
        {"j": -1},
        {"width": 0},
        {"height": 0},
    ],
)
def test_invalid_rect_is_rejected(overrides) -> None:
    with pytest.raises((AssertionError, ValueError)):
        _rect(**overrides)


def test_rect_value_semantics_and_resolution() -> None:
    rect = _rect()

    assert rect.downsampled_resolution() == (4, 3)
    assert rect == _rect()
    assert rect != _rect(i=0)
    assert rect != object()
    assert hash(rect) == hash(_rect())
    assert rect.to_json()["image_domain_scale"] == 0.5
    assert "resolution_mode" not in rect.to_json()
    assert "resized_pre_crop_resolution_override" not in rect.to_json()
    torch.testing.assert_close(
        rect.rect_points_lb,
        torch.tensor([[2.5, 2.5], [7.5, 5.5]]),
    )
    torch.testing.assert_close(rect.resolution, torch.tensor([8.0, 6.0]))
    torch.testing.assert_close(rect.rect_points_lb_cpu, rect.rect_points_lb)
    torch.testing.assert_close(rect.resolution_cpu, rect.resolution)


def test_rect_preserves_fractional_shutter_scanline_boundary() -> None:
    rect = _rect(original_width=70, original_height=70, width=17, height=17, i=0, j=0, subsample_factor=3.5)
    expected = 0.5 + 3.5 * np.array([[0, 0], [17 - 1 / 3.5, 17 - 1 / 3.5]], dtype=np.float32)
    np.testing.assert_array_equal(rect.rect_points_lb_cpu.numpy(), expected)
    assert np.floor(rect.rect_points_lb_cpu[1, 1].item()) == 58


def test_array_operations_preserve_integer_factor_with_reciprocal_roundoff() -> None:
    rect = _rect(original_width=98, original_height=49, width=2, height=1, i=0, j=0, subsample_factor=49.0)
    image = np.arange(49 * 98, dtype=np.float32).reshape(49, 98, 1)
    resized = rect.apply_to_array(image, interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(resized, image[::49, ::49])
    np.testing.assert_array_equal(rect.crop_array(resized), resized)


def test_from_image_resolution_transform_preserves_absolute_transform() -> None:
    transform = ImageResolutionTransform.cover_and_center_crop((14, 35), (3, 2))

    rect = RectSubsampled.from_image_resolution_transform(transform)

    assert rect.image_transform == transform
    assert rect.downsampled_resolution() == (3, 8)
    assert rect.to_json()["resized_pre_crop_resolution"] == (3, 8)
    assert "resolution_mode" not in rect.to_json()


def test_exact_resized_pre_crop_resolution_survives_json_round_trip() -> None:
    rect = RectSubsampled.from_image_resolution_transform(
        ImageResolutionTransform.cover_and_center_crop((14, 35), (3, 2))
    )

    restored = RectSubsampled.from_image_resolution_transform(
        ImageResolutionTransform(**json.loads(json.dumps(rect.to_json())))
    )

    assert restored == rect
    assert hash(restored) == hash(rect)
    assert restored.downsampled_resolution() == (3, 8)


def test_from_image_resolution_transform_supports_relative_crop_override() -> None:
    transform = ImageResolutionTransform.from_subsample((8, 6), 2.0)

    rect = RectSubsampled.from_image_resolution_transform(
        transform,
        output_resolution=(3, 2),
        image_domain_offset=(1, 1),
    )

    assert (rect.width, rect.height) == (3, 2)
    assert (rect.i, rect.j) == (1, 1)
    assert rect.image_transform.resized_pre_crop_resolution == (4, 3)


def test_pixel_coordinates_are_downsampled_image_pixel_centers() -> None:
    coordinates = _rect().pixel_coordinates()

    assert coordinates.shape == (2, 3, 2)
    torch.testing.assert_close(coordinates[0, 0], torch.tensor([1.5, 1.5]))
    torch.testing.assert_close(coordinates[-1, -1], torch.tensor([3.5, 2.5]))


def test_relative_and_absolute_rectangles_have_identical_value_semantics() -> None:
    relative = RectSubsampled.from_subsample(
        original_width=8, original_height=5, width=3, height=2, subsample_factor=8 / 3
    )
    absolute = RectSubsampled.from_image_resolution_transform(
        ImageResolutionTransform.from_absolute((8, 5), width=3, height=None, keep_aspect_ratio=True)
    )
    assert relative == absolute
    assert hash(relative) == hash(absolute)
    assert relative.to_json() == absolute.to_json()
    np.testing.assert_array_equal(relative.crop_array(np.zeros((2, 3, 1))), absolute.crop_array(np.zeros((2, 3, 1))))


def test_apply_to_array_without_resize() -> None:
    rect = _rect(
        original_width=4,
        original_height=3,
        width=2,
        height=2,
        subsample_factor=1.0,
    )
    image = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4, 1)

    result = rect.apply_to_array(image, interpolation=cv2.INTER_NEAREST)

    np.testing.assert_array_equal(result[..., 0], [[5, 6], [9, 10]])


def test_apply_to_array_cv2_resizes_and_preserves_single_channel_shape() -> None:
    image = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8, 1)

    result = _rect().apply_to_array(
        image,
        interpolation=cv2.INTER_NEAREST,
    )

    np.testing.assert_array_equal(
        result[..., 0],
        [
            [18, 20, 22],
            [34, 36, 38],
        ],
    )


def test_apply_to_array_supports_absolute_anisotropic_scale() -> None:
    rect = RectSubsampled.from_image_resolution_transform(
        ImageResolutionTransform.from_absolute((8, 6), width=3, height=2, keep_aspect_ratio=False)
    )
    image = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8, 1)

    result = rect.apply_to_array(image, interpolation=cv2.INTER_NEAREST)

    assert rect.downsampled_resolution() == (3, 2)
    np.testing.assert_array_equal(result[..., 0], [[0, 2, 5], [24, 26, 29]])


def test_apply_to_array_accepts_absolute_scale_with_float_roundoff() -> None:
    rect = RectSubsampled.from_image_resolution_transform(
        ImageResolutionTransform.from_absolute(
            (13, 15), width=23, height=11, keep_aspect_ratio=False, allow_upsampling=True
        )
    )

    result = rect.apply_to_array(
        np.zeros((15, 13, 1), dtype=np.uint8),
        interpolation=cv2.INTER_NEAREST,
    )

    assert result.shape == (11, 23, 1)


def test_apply_to_array_supports_rounded_absolute_keep_aspect_scale() -> None:
    rect = RectSubsampled.from_image_resolution_transform(
        ImageResolutionTransform.from_absolute((8, 5), width=3, height=None, keep_aspect_ratio=True)
    )

    result = rect.apply_to_array(
        np.zeros((5, 8, 1), dtype=np.uint8),
        interpolation=cv2.INTER_NEAREST,
    )

    assert result.shape == (2, 3, 1)


def test_apply_to_array_pil_lanczos_resizes_rgb() -> None:
    image = np.full((6, 8, 3), [17, 31, 47], dtype=np.uint8)

    result = _rect().apply_to_array(
        image,
        interpolation=None,
        use_pil_lanczos=True,
    )

    np.testing.assert_array_equal(
        result,
        np.broadcast_to(np.array([17, 31, 47], dtype=np.uint8), (2, 3, 3)),
    )


def test_apply_to_array_requires_interpolation_for_cv2() -> None:
    image = np.zeros((6, 8, 1), dtype=np.uint8)

    with pytest.raises(AssertionError, match="interpolation is required"):
        _rect().apply_to_array(image, interpolation=None)


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
        _rect().apply_to_array(array, interpolation=cv2.INTER_NEAREST)


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
        _rect().crop_array(array)


def test_crop_array_validates_resolved_shape_without_a_config_mode_bypass() -> None:
    rect = _rect(subsample_factor=2.5, i=0, j=0)
    assert rect.image_transform.resized_pre_crop_resolution == (3, 2)
    with pytest.raises(AssertionError, match="Array height"):
        rect.crop_array(np.zeros((3, 4, 1)))


def test_rect_rejects_region_past_resized_pre_crop_width() -> None:
    with pytest.raises(ValueError, match="exceeds resized pre-crop resolution"):
        _rect(i=2, width=3)


def test_rect_rejects_region_past_resized_pre_crop_height() -> None:
    with pytest.raises(ValueError, match="exceeds resized pre-crop resolution"):
        _rect(j=2, height=2)


def _camera_parameters(parameter_type):
    common = {
        "resolution": np.array([8, 6], dtype=np.uint64),
        "shutter_type": ShutterType.GLOBAL,
        "principal_point": np.array([4.0, 3.0], dtype=np.float32),
    }
    if parameter_type is FThetaCameraModelParameters:
        return parameter_type(
            **common,
            reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
            pixeldist_to_angle_poly=np.zeros(6, dtype=np.float32),
            angle_to_pixeldist_poly=np.array([0.0, 4.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            max_angle=1.0,
            linear_cde=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
    if parameter_type is OpenCVFisheyeCameraModelParameters:
        return parameter_type(
            **common,
            focal_length=np.array([4.0, 5.0], dtype=np.float32),
            radial_coeffs=np.zeros(4, dtype=np.float32),
            max_angle=1.0,
        )
    return parameter_type(
        **common,
        focal_length=np.array([4.0, 5.0], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )


class _Camera:
    device = torch.device("cpu")

    def __init__(self, parameters) -> None:
        self._parameters = parameters
        self.resolution = torch.as_tensor(parameters.resolution)

    def get_parameters(self):
        return self._parameters


@pytest.mark.parametrize(
    ("parameter_type", "expected_principal_point"),
    [
        (FThetaCameraModelParameters, [0.75, 0.25]),
        (OpenCVFisheyeCameraModelParameters, [1.0, 0.5]),
        (OpenCVPinholeCameraModelParameters, [1.0, 0.5]),
    ],
)
def test_apply_to_camera_model_scales_offsets_and_crops_intrinsics(
    parameter_type,
    expected_principal_point: list[float],
) -> None:
    camera = cast(Any, _Camera(_camera_parameters(parameter_type)))

    with patch("nurec.utils.batch.camera_model_from_parameters", side_effect=lambda parameters, **_: parameters):
        transformed_parameters = cast(Any, _rect().apply_to_camera_model(camera))

    np.testing.assert_array_equal(transformed_parameters.resolution, [3, 2])
    np.testing.assert_allclose(transformed_parameters.principal_point, expected_principal_point)
    assert isinstance(transformed_parameters, parameter_type)
    if hasattr(transformed_parameters, "focal_length"):
        np.testing.assert_allclose(transformed_parameters.focal_length, [2.0, 2.5])


def test_apply_to_camera_model_supports_absolute_anisotropic_scale() -> None:
    camera = cast(Any, _Camera(_camera_parameters(OpenCVPinholeCameraModelParameters)))
    rect = RectSubsampled.from_image_resolution_transform(
        ImageResolutionTransform.from_absolute((8, 6), width=3, height=2, keep_aspect_ratio=False)
    )

    with patch("nurec.utils.batch.camera_model_from_parameters", side_effect=lambda parameters, **_: parameters):
        transformed_parameters = cast(Any, rect.apply_to_camera_model(camera))

    np.testing.assert_array_equal(transformed_parameters.resolution, [3, 2])
    np.testing.assert_allclose(transformed_parameters.focal_length, [1.5, 5.0 / 3.0])
    np.testing.assert_allclose(transformed_parameters.principal_point, [1.5, 1.0])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"original_width": 7}, "Image width"),
        ({"original_height": 5, "j": 0}, "Image height"),
    ],
)
def test_apply_to_camera_model_validates_source_resolution(overrides, message: str) -> None:
    camera = cast(Any, _Camera(_camera_parameters(OpenCVPinholeCameraModelParameters)))

    with pytest.raises(AssertionError, match=message):
        _rect(**overrides).apply_to_camera_model(camera)


def test_apply_to_camera_model_rejects_unsupported_parameters() -> None:
    class _UnsupportedParameters:
        @staticmethod
        def type() -> str:
            return "unsupported"

        @staticmethod
        def to_json() -> dict:
            return {}

    class _UnsupportedCamera:
        resolution = torch.tensor([8, 6])
        device = torch.device("cpu")

        @staticmethod
        def get_parameters() -> _UnsupportedParameters:
            return _UnsupportedParameters()

    with pytest.raises(Exception, match="Unsupported camera model type"):
        _rect().apply_to_camera_model(_UnsupportedCamera())  # type: ignore[arg-type]
