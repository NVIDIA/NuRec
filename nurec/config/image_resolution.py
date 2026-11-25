# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import ConfigDict, TypeAdapter, model_validator

from nurec.config.base_schema import BaseConfigSchema, Field
from nurec.utils.image_resolution import ImageResolutionTransform, Resolution, ResolutionRounding


ResolutionDimension = Annotated[int, Field(gt=0)] | Literal["original"] | None


class RelativeImageResolutionConfig(BaseConfigSchema):
    """Scale both image dimensions by a relative divisor."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["relative"] = "relative"
    subsample: int = Field(ge=1)

    def resolve(
        self,
        source_resolution: Resolution,
        *,
        require_divisible: bool = False,
        rounding: ResolutionRounding = "round",
    ) -> ImageResolutionTransform:
        return ImageResolutionTransform.from_subsample(
            source_resolution,
            self.subsample,
            require_divisible=require_divisible,
            rounding=rounding,
        )


class _AbsoluteImageResolutionConfig(BaseConfigSchema):
    """Shared absolute-resolution leaf options."""

    model_config = ConfigDict(extra="forbid")

    width: ResolutionDimension = "original"
    height: ResolutionDimension = None
    keep_aspect_ratio: bool = True
    allow_upsampling: bool = False

    @model_validator(mode="after")
    def validate_resolution_selection(self) -> _AbsoluteImageResolutionConfig:
        has_width = self.width is not None
        has_height = self.height is not None
        if self.keep_aspect_ratio:
            if has_width == has_height:
                raise ValueError("Exactly one of width or height is required when keep_aspect_ratio is true")
        elif not (has_width and has_height):
            raise ValueError("Both width and height are required when keep_aspect_ratio is false")
        return self

    def resolve(
        self,
        source_resolution: Resolution,
        *,
        rounding: ResolutionRounding = "round",
    ) -> ImageResolutionTransform:
        width = source_resolution[0] if self.width == "original" else self.width
        height = source_resolution[1] if self.height == "original" else self.height
        return ImageResolutionTransform.from_absolute(
            source_resolution,
            width=width,
            height=height,
            keep_aspect_ratio=self.keep_aspect_ratio,
            allow_upsampling=self.allow_upsampling,
            rounding=rounding,
        )


class AbsoluteImageResolutionConfig(_AbsoluteImageResolutionConfig):
    """Select one absolute image resolution."""

    mode: Literal["absolute"] = "absolute"


ImageResolutionConfig: TypeAlias = Annotated[
    RelativeImageResolutionConfig | AbsoluteImageResolutionConfig,
    Field(discriminator="mode"),
]


class CameraImageResolutionGroupConfig(_AbsoluteImageResolutionConfig):
    """Absolute resolution shared by an explicit group of cameras."""

    camera_ids: Annotated[list[str], Field(min_length=1)]


CameraImageResolutionGroupsConfig: TypeAlias = Annotated[
    list[CameraImageResolutionGroupConfig],
    Field(min_length=1),
]


class AbsoluteCameraImageResolutionConfig(BaseConfigSchema):
    """Absolute resolutions partitioned into explicit camera groups."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["absolute"] = "absolute"
    camera_groups: CameraImageResolutionGroupsConfig


CameraImageResolutionConfig: TypeAlias = Annotated[
    RelativeImageResolutionConfig | AbsoluteCameraImageResolutionConfig,
    Field(discriminator="mode"),
]

_IMAGE_RESOLUTION_ADAPTER: TypeAdapter[ImageResolutionConfig] = TypeAdapter(ImageResolutionConfig)
_CAMERA_IMAGE_RESOLUTION_ADAPTER: TypeAdapter[CameraImageResolutionConfig] = TypeAdapter(CameraImageResolutionConfig)

LEGACY_SUBSAMPLE_OVERRIDE_PATHS = {
    "dataset.n_val_image_subsample": "dataset.n_val_image_resolution",
    "dataset.n_train_sequential_image_subsample": "dataset.n_train_sequential_image_resolution",
    "dataset.samplers.batch_sampler.camera_pixel_sampler.subsample": (
        "dataset.samplers.batch_sampler.camera_pixel_sampler.resolution"
    ),
    "fixed_camera_render.subsample": "fixed_camera_render.resolution",
}


def warn_legacy_resolution(field_name: str, replacement: str) -> None:
    """Log deprecations even when the application filters Python warnings."""
    logging.getLogger(__name__).warning("%s is deprecated; use %s", field_name, replacement)


def parse_camera_image_resolution_config(
    resolution: Any,
    *,
    legacy_subsample: int | None,
    field_name: str,
) -> CameraImageResolutionConfig:
    """Parse camera resolution, accepting a prior subsample setting."""
    if (
        resolution is not None
        and not isinstance(resolution, (Mapping, BaseConfigSchema))
        and hasattr(resolution, "items")
    ):
        resolution = dict(resolution)
    legacy_subsample_field = (
        f"{field_name.removesuffix('_resolution')}_subsample"
        if field_name.endswith("_resolution")
        else f"{field_name.removesuffix('.resolution')}.subsample"
    )
    if legacy_subsample is None:
        if resolution is None:
            raise ValueError(f"{field_name} must specify either relative or absolute resolution")
        return _CAMERA_IMAGE_RESOLUTION_ADAPTER.validate_python(resolution)

    legacy_resolution = RelativeImageResolutionConfig(subsample=legacy_subsample)
    if resolution is None:
        warn_legacy_resolution(legacy_subsample_field, field_name)
        return legacy_resolution

    canonical_resolution = _CAMERA_IMAGE_RESOLUTION_ADAPTER.validate_python(resolution)
    if not isinstance(canonical_resolution, RelativeImageResolutionConfig):
        raise ValueError(f"{field_name} cannot combine an absolute resolution with its legacy subsample setting")
    warn_legacy_resolution(legacy_subsample_field, field_name)
    return legacy_resolution


def parse_image_resolution_config(
    resolution: Any,
    *,
    legacy_subsample: int | None,
    field_name: str,
    default_subsample: int,
) -> ImageResolutionConfig:
    """Parse fixed-render resolution, accepting its prior subsample setting."""
    if (
        resolution is not None
        and not isinstance(resolution, (Mapping, BaseConfigSchema))
        and hasattr(resolution, "items")
    ):
        resolution = dict(resolution)
    if resolution is None:
        if legacy_subsample is None:
            return RelativeImageResolutionConfig(subsample=default_subsample)
        warn_legacy_resolution(f"{field_name}.subsample", f"{field_name}.resolution.subsample")
        return RelativeImageResolutionConfig(subsample=legacy_subsample)

    canonical_resolution = _IMAGE_RESOLUTION_ADAPTER.validate_python(resolution)
    if legacy_subsample is None:
        return canonical_resolution
    if not isinstance(canonical_resolution, RelativeImageResolutionConfig):
        raise ValueError(f"{field_name}.resolution cannot combine an absolute resolution with {field_name}.subsample")
    warn_legacy_resolution(f"{field_name}.subsample", f"{field_name}.resolution.subsample")
    return RelativeImageResolutionConfig(subsample=legacy_subsample)


def resolve_camera_image_resolution_transforms(
    image_resolution: CameraImageResolutionConfig,
    *,
    active_camera_ids: Iterable[str],
    source_resolutions: Mapping[str, Resolution],
    field_name: str,
    require_divisible: bool = False,
    rounding: ResolutionRounding = "round",
) -> dict[str, ImageResolutionTransform]:
    """Validate and resolve a relative broadcast or strict absolute camera groups."""
    camera_ids = list(active_camera_ids)
    missing_data_ids = [camera_id for camera_id in camera_ids if camera_id not in source_resolutions]
    if missing_data_ids:
        raise ValueError(
            f"{field_name}: active cameras are not present in the data: {missing_data_ids}; "
            f"available cameras are {sorted(source_resolutions)}"
        )

    if isinstance(image_resolution, RelativeImageResolutionConfig):
        return {
            camera_id: image_resolution.resolve(
                source_resolutions[camera_id],
                require_divisible=require_divisible,
                rounding=rounding,
            )
            for camera_id in camera_ids
        }

    resolution_by_camera: dict[str, CameraImageResolutionGroupConfig] = {}
    duplicate_camera_ids: list[str] = []
    for group in image_resolution.camera_groups:
        for camera_id in group.camera_ids:
            if camera_id in resolution_by_camera:
                if camera_id not in duplicate_camera_ids:
                    duplicate_camera_ids.append(camera_id)
            else:
                resolution_by_camera[camera_id] = group
    if duplicate_camera_ids:
        raise ValueError(
            f"{field_name}: cameras may appear in exactly one resolution group; "
            f"duplicate cameras are {duplicate_camera_ids}"
        )

    active_camera_id_set = set(camera_ids)
    configured_camera_ids = list(resolution_by_camera)
    missing_camera_ids = [camera_id for camera_id in camera_ids if camera_id not in resolution_by_camera]
    extra_camera_ids = [camera_id for camera_id in configured_camera_ids if camera_id not in active_camera_id_set]
    if missing_camera_ids or extra_camera_ids:
        problems = []
        if missing_camera_ids:
            problems.append(f"missing cameras {missing_camera_ids}")
        if extra_camera_ids:
            problems.append(f"unknown cameras {extra_camera_ids}")
        raise ValueError(
            f"{field_name}: resolution groups must match the active camera ids exactly "
            f"(active={sorted(active_camera_id_set)}, configured={sorted(configured_camera_ids)}): "
            f"{'; '.join(problems)}"
        )

    for group_index, group in enumerate(image_resolution.camera_groups):
        group_source_resolutions = {camera_id: source_resolutions[camera_id] for camera_id in group.camera_ids}
        if len(set(group_source_resolutions.values())) > 1:
            raise ValueError(
                f"{field_name}: resolution group {group_index} contains cameras with different source "
                f"resolutions: {group_source_resolutions}; split cameras with different source resolutions "
                "into separate groups"
            )

    return {
        camera_id: resolution_by_camera[camera_id].resolve(
            source_resolutions[camera_id],
            rounding=rounding,
        )
        for camera_id in camera_ids
    }
