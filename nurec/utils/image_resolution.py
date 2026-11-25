# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self, TypeVar

import cv2
import numpy as np

from PIL import Image as PILImage


Resolution = tuple[int, int]
Scale2D = float | tuple[float, float]
ResolutionRounding = Literal["round", "floor"]


class TransformableCameraParameters(Protocol):  # pragma: no cover - typing-only protocol
    resolution: Any

    def transform(
        self,
        *,
        image_domain_scale: Scale2D,
        image_domain_offset: tuple[int, int] = (0, 0),
        new_resolution: Resolution,
    ) -> Self: ...


_TransformableCameraParametersT = TypeVar(
    "_TransformableCameraParametersT",
    bound=TransformableCameraParameters,
)


def _components(value: Scale2D) -> tuple[float, float]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def resize_image_array(
    array: np.ndarray,
    resolution: Resolution,
    interpolation: int | None,
    *,
    use_pil_lanczos: bool = False,
) -> np.ndarray:
    """Resize an HWC array to exact dimensions, preserving its channel axis."""
    assert array.ndim == 3, f"Array must be 3D shape, but got {array.ndim}D of shape {array.shape}"
    height, width, channels = array.shape
    if (width, height) == resolution:
        return array
    if use_pil_lanczos:
        array = np.asarray(PILImage.fromarray(array).resize(resolution, PILImage.Resampling.LANCZOS))
    else:
        assert interpolation is not None, "interpolation is required when use_pil_lanczos is False"
        array = cv2.resize(array, resolution, interpolation=interpolation)
    return array.reshape(resolution[1], resolution[0], channels)


@dataclass(frozen=True, slots=True)
class ImageResolutionTransform:
    """Resize and optional crop in the camera image domain.

    Resizing is applied before cropping. Resolutions use ``(width, height)``
    ordering; the crop offset uses ``(x, y)`` ordering in the resized pre-crop domain.
    """

    source_resolution: Resolution
    resized_pre_crop_resolution: Resolution
    output_resolution: Resolution
    image_domain_scale: Scale2D
    image_domain_offset: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        """Validate the resize and crop geometry."""
        # JSON transports tuples as lists; normalize them once for stable value
        # semantics and hashing, independent of the configuration spelling.
        for name in ("source_resolution", "resized_pre_crop_resolution", "output_resolution", "image_domain_offset"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if isinstance(self.image_domain_scale, list):
            object.__setattr__(self, "image_domain_scale", tuple(self.image_domain_scale))
        resolutions = {
            "source_resolution": self.source_resolution,
            "resized_pre_crop_resolution": self.resized_pre_crop_resolution,
            "output_resolution": self.output_resolution,
        }
        for name, (width, height) in resolutions.items():
            if width <= 0 or height <= 0:
                raise ValueError(f"{name} must contain positive dimensions, got {width}x{height}")

        offset_x, offset_y = self.image_domain_offset
        if offset_x < 0 or offset_y < 0:
            raise ValueError(f"image_domain_offset must be non-negative, got {self.image_domain_offset}")

        resized_pre_crop_width, resized_pre_crop_height = self.resized_pre_crop_resolution
        output_width, output_height = self.output_resolution
        if offset_x + output_width > resized_pre_crop_width or offset_y + output_height > resized_pre_crop_height:
            raise ValueError(
                f"Crop at offset {self.image_domain_offset} with output resolution "
                f"{output_width}x{output_height} exceeds resized pre-crop resolution "
                f"{resized_pre_crop_width}x{resized_pre_crop_height}"
            )

        # Keep the geometric scale, rather than deriving a slightly anisotropic
        # scale from rounded dimensions. Legacy consumers round or truncate the
        # raster extent; neither operation changes the camera's uniform scale.
        for source, resized, scale in zip(
            self.source_resolution, self.resized_pre_crop_resolution, _components(self.image_domain_scale)
        ):
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("image_domain_scale must be finite and positive")
            if abs(source * scale - resized) >= 1.0 + 1e-9:
                raise ValueError("image_domain_scale does not match the resized pre-crop resolution within rounding")

    @classmethod
    def from_subsample(
        cls,
        source_resolution: Resolution,
        subsample_factor: Scale2D,
        *,
        output_resolution: Resolution | None = None,
        image_domain_offset: tuple[int, int] = (0, 0),
        require_divisible: bool = False,
        rounding: ResolutionRounding = "round",
    ) -> ImageResolutionTransform:
        """Build a transform from an input-relative subsampling factor."""
        source_width, source_height = source_resolution
        factor_x, factor_y = _components(subsample_factor)
        assert factor_x > 0.0 and factor_y > 0.0, "Invalid subsample_factor value"
        assert not require_divisible or (source_width % factor_x == 0 and source_height % factor_y == 0), (
            f"Subsample factor {subsample_factor} invalid: resolution {source_width}x{source_height} must be divisible"
        )

        assert rounding in ("round", "floor"), f"Unsupported resolution rounding mode {rounding}"
        resolve_dimension = round if rounding == "round" else int
        resized_pre_crop_resolution = (
            resolve_dimension(source_width / factor_x),
            resolve_dimension(source_height / factor_y),
        )
        assert resized_pre_crop_resolution[0] > 0 and resized_pre_crop_resolution[1] > 0, (
            "Resized pre-crop resolution must be positive"
        )
        image_domain_scale: Scale2D = (
            (1.0 / factor_x, 1.0 / factor_y) if isinstance(subsample_factor, tuple) else 1.0 / factor_x
        )

        return cls(
            source_resolution=source_resolution,
            resized_pre_crop_resolution=resized_pre_crop_resolution,
            output_resolution=output_resolution or resized_pre_crop_resolution,
            image_domain_scale=image_domain_scale,
            image_domain_offset=image_domain_offset,
        )

    @classmethod
    def from_absolute(
        cls,
        source_resolution: Resolution,
        *,
        width: int | None,
        height: int | None,
        keep_aspect_ratio: bool,
        allow_upsampling: bool = False,
        rounding: ResolutionRounding = "round",
    ) -> ImageResolutionTransform:
        """Build a transform from one of the supported absolute-resolution forms."""
        source_width, source_height = source_resolution
        assert width is None or width > 0, "width must be positive"
        assert height is None or height > 0, "height must be positive"

        if keep_aspect_ratio:
            assert (width is None) != (height is None), (
                "Exactly one of width or height is required when keep_aspect_ratio is true"
            )
            if width is not None:
                factor = source_width / width
            else:
                assert height is not None
                factor = source_height / height
            relative_transform = cls.from_subsample(source_resolution, factor, rounding=rounding)
            # Do not reconstruct the explicitly requested dimension via a float
            # reciprocal: floor rounding can otherwise lose a pixel there.
            output_resolution = (
                width if width is not None else relative_transform.output_resolution[0],
                height if height is not None else relative_transform.output_resolution[1],
            )
            image_domain_scale = relative_transform.image_domain_scale
        else:
            assert width is not None and height is not None, (
                "Both width and height are required when keep_aspect_ratio is false"
            )
            output_resolution = (width, height)
            image_domain_scale = (width / source_width, height / source_height)

        if not allow_upsampling and (output_resolution[0] > source_width or output_resolution[1] > source_height):
            raise ValueError(
                f"Absolute resolution {output_resolution[0]}x{output_resolution[1]} would upsample "
                f"the source resolution {source_width}x{source_height}; set allow_upsampling=true to allow it"
            )

        return cls(
            source_resolution=source_resolution,
            resized_pre_crop_resolution=output_resolution,
            output_resolution=output_resolution,
            image_domain_scale=image_domain_scale,
        )

    @classmethod
    def fit_keep_aspect_ratio(
        cls,
        source_resolution: Resolution,
        requested_resolution: Resolution,
    ) -> ImageResolutionTransform:
        """Legacy render fit: cover the request, truncating the unrequested axis.

        Unlike subsampling's default nearest rounding, rendering historically
        truncates the raster extent without changing the uniform camera scale.
        """
        source_width, source_height = source_resolution
        requested_width, requested_height = requested_resolution
        aspect_ratio = source_width / source_height
        if (requested_width / source_width) > (requested_height / source_height):
            output_width = requested_width
            output_height = int(output_width / aspect_ratio)
            scale = output_width / source_width
        else:
            output_height = requested_height
            output_width = int(output_height * aspect_ratio)
            scale = output_height / source_height

        output_resolution = (output_width, output_height)
        return cls(
            source_resolution=source_resolution,
            resized_pre_crop_resolution=output_resolution,
            output_resolution=output_resolution,
            image_domain_scale=scale,
        )

    @classmethod
    def fit_height(cls, source_resolution: Resolution, height: int) -> ImageResolutionTransform:
        """Legacy height-only rendering: keep uniform scale and truncate width."""
        scale = height / source_resolution[1]
        resolution = (int(source_resolution[0] * scale), height)
        return cls(source_resolution, resolution, resolution, scale)

    @classmethod
    def cover_and_center_crop(
        cls,
        source_resolution: Resolution,
        output_resolution: Resolution,
    ) -> ImageResolutionTransform:
        """Preserve aspect ratio while covering and center-cropping an exact output."""
        source_width, source_height = source_resolution
        output_width, output_height = output_resolution
        scale = max(output_width / source_width, output_height / source_height)
        resized_pre_crop_width = round(source_width * scale)
        resized_pre_crop_height = round(source_height * scale)
        resized_pre_crop_resolution = (resized_pre_crop_width, resized_pre_crop_height)
        assert resized_pre_crop_width >= output_width and resized_pre_crop_height >= output_height
        image_domain_offset = (
            (resized_pre_crop_width - output_width) // 2,
            (resized_pre_crop_height - output_height) // 2,
        )
        return cls(
            source_resolution=source_resolution,
            resized_pre_crop_resolution=resized_pre_crop_resolution,
            output_resolution=output_resolution,
            image_domain_scale=scale,
            image_domain_offset=image_domain_offset,
        )

    @property
    def subsample_factor(self) -> Scale2D:
        scale_x, scale_y = _components(self.image_domain_scale)
        if isinstance(self.image_domain_scale, tuple):
            return (1.0 / scale_x, 1.0 / scale_y)
        return 1.0 / scale_x

    def rect_points_lb(self) -> np.ndarray:
        """Return crop corner pixel centers in the original image domain."""
        factor_x, factor_y = _components(self.subsample_factor)
        offset_x, offset_y = self.image_domain_offset
        output_width, output_height = self.output_resolution
        # Preserve the original float32 operation order: shutter interpolation
        # floors these endpoints, so algebraic rearrangement can select a
        # different scanline at a floating-point boundary.
        points = np.array(
            [
                [offset_x, offset_y],
                [
                    offset_x + (output_width - 1.0 / factor_x),
                    offset_y + (output_height - 1.0 / factor_y),
                ],
            ],
            dtype=np.float32,
        )
        return 0.5 + np.array([factor_x, factor_y], dtype=np.float32) * points

    def crop_array(self, array: np.ndarray) -> np.ndarray:
        """Crop an already-resized pre-crop ``HWC`` array to the output domain."""
        assert array.ndim == 3, f"Array must be 3D shape, but got {array.ndim}D of shape {array.shape}"
        height, width, _ = array.shape
        resized_pre_crop_width, resized_pre_crop_height = self.resized_pre_crop_resolution
        assert height == resized_pre_crop_height, (
            f"Array height ({height}) does not match resized pre-crop height ({resized_pre_crop_height})"
        )
        assert width == resized_pre_crop_width, (
            f"Array width ({width}) does not match resized pre-crop width ({resized_pre_crop_width})"
        )

        offset_x, offset_y = self.image_domain_offset
        output_width, output_height = self.output_resolution
        right = offset_x + output_width
        bottom = offset_y + output_height
        return array[offset_y:bottom, offset_x:right, :]

    def sample_integer_stride_pixel_grid(self, camera_all_pixels: np.ndarray) -> tuple[np.ndarray, bool]:
        """Select source pixel centers for a legacy integer-stride rectangle."""
        subsample_factor = self.subsample_factor
        assert not isinstance(subsample_factor, tuple), "Pixel-grid sampling only supports isotropic subsampling"
        n_subsample = round(subsample_factor)
        assert n_subsample >= 1 and math.isclose(n_subsample, subsample_factor, rel_tol=1e-12), (
            f"Subsample factor {subsample_factor} in the sampler_sampled_pixels must be an integer for now."
        )

        source_width, source_height = self.source_resolution
        frame_all_pixels_subsampled = camera_all_pixels.reshape(source_height, source_width, 2)[
            ::n_subsample, ::n_subsample
        ]
        offset_x, offset_y = self.image_domain_offset
        output_width, output_height = self.output_resolution
        pixels_subsampled_offset = (n_subsample - 1) // 2
        pixels = (
            frame_all_pixels_subsampled[
                offset_y : offset_y + output_height,
                offset_x : offset_x + output_width,
            ].reshape(-1, 2)
            + pixels_subsampled_offset
        )
        return pixels, n_subsample % 2 == 0

    def apply_to_array(
        self,
        array: np.ndarray,
        interpolation: int | None,
        *,
        use_pil_lanczos: bool = False,
    ) -> np.ndarray:
        """Resize a source ``HWC`` array and crop it to the output domain."""
        assert array.ndim == 3, f"Array must be 3D shape, but got {array.ndim}D of shape {array.shape}"
        height, width, _ = array.shape
        source_width, source_height = self.source_resolution
        assert height == source_height, f"Image height ({height}) does not match the original height ({source_height})"
        assert width == source_width, f"Image width ({width}) does not match the original width ({source_width})"

        array = resize_image_array(
            array, self.resized_pre_crop_resolution, interpolation, use_pil_lanczos=use_pil_lanczos
        )
        return self.crop_array(array)

    def apply_to_camera_parameters(
        self,
        camera_parameters: _TransformableCameraParametersT,
    ) -> _TransformableCameraParametersT:
        """Apply the resize/crop transform to camera model parameters."""
        source_resolution = tuple(int(value) for value in camera_parameters.resolution.tolist())
        assert source_resolution == self.source_resolution, "Camera resolution does not match the source resolution"

        transform_args: dict[str, Any] = {
            "image_domain_scale": self.image_domain_scale,
            "new_resolution": self.output_resolution,
        }
        if self.image_domain_offset != (0, 0):
            transform_args["image_domain_offset"] = self.image_domain_offset
        transformed = camera_parameters.transform(**transform_args)
        transformed_resolution = tuple(int(value) for value in transformed.resolution.tolist())
        assert transformed_resolution == self.output_resolution
        return transformed
