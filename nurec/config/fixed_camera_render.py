# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import model_validator

from nurec.config.base_schema import BaseConfigSchema, Field
from nurec.config.image_resolution import (
    ImageResolutionConfig,
    RelativeImageResolutionConfig,
    parse_image_resolution_config,
)


class FixedCameraRenderConfig(BaseConfigSchema):
    enabled: bool = Field(
        default=False,
        description="Whether to render a fixed view periodically during training.",
    )
    camera_id: str = Field(
        default="camera_front_wide_120fov",
        description="Camera ID to render. Must be present in the train-sequential split.",
    )
    frame_idxs: list[int] = Field(
        default_factory=lambda: [300],
        min_length=1,
        description=(
            "Frame indices (as found in dataset.camera_frames[camera_id]) to render. "
            "All listed frames are rendered at each scheduled tick. Each frame gets "
            "its own output sub-directory and one MP4 per configured view."
        ),
    )
    views: list[str] = Field(
        default_factory=lambda: ["rgb", "residual"],
        min_length=1,
        description=(
            "Which derived images to save per scheduled tick. Currently supported: "
            "'rgb' (predicted color), 'residual' (|pred-gt|, turbo colormap, "
            "ego-masked). Unknown names log a warning and are skipped. Each view "
            "produces its own step_NNNNNNNN_<view>.jpg files and a "
            "frame_NNNN_<view>.mp4 at training end."
        ),
    )
    every_n_steps: int = Field(
        default=100,
        ge=1,
        description="Render every N training iterations (global_step).",
    )
    start_step: int = Field(
        default=0,
        ge=0,
        description=(
            "First global_step at which to render. Schedule fires at start_step, start_step + every_n_steps, ..."
        ),
    )
    resolution: ImageResolutionConfig = Field(
        default_factory=lambda: RelativeImageResolutionConfig(subsample=2),
        description="Resolution of the fixed-camera render, independent of train-sequential and validation resolution.",
    )
    video_fps: int = Field(
        default=10,
        ge=1,
        description="FPS for the assembled MP4.",
    )
    jpeg_quality: int = Field(
        default=90,
        ge=1,
        le=100,
        description="JPEG encoder quality (1-100) passed to imageio.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_resolution(cls, values: Any) -> Any:
        if not isinstance(values, Mapping) and not hasattr(values, "items"):
            return values
        normalized = dict(values)
        unsupported_fields = {"width", "height", "keep_aspect_ratio", "allow_upsampling"} & normalized.keys()
        if unsupported_fields:
            raise ValueError(
                f"Unsupported flat fixed-camera resolution fields: {sorted(unsupported_fields)}; "
                "set these under fixed_camera_render.resolution"
            )
        normalized["resolution"] = parse_image_resolution_config(
            normalized.get("resolution"),
            legacy_subsample=normalized.pop("subsample", None),
            field_name="fixed_camera_render",
            default_subsample=2,
        )
        return normalized
