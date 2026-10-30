# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nurec.config.base_schema import BaseConfigSchema, Field


class ViewerConfig(BaseConfigSchema):
    class RaySubsamplingStep(BaseConfigSchema):
        high_resolution: float = Field(default=4.0, description="Render every nth ray in high resolution mode.")
        low_resolution: float = Field(default=16.0, description="Render every nth ray in low resolution mode.")

    enabled: bool = Field(default=False, description="Whether to start the viewer")

    host: str = Field(default="127.0.0.1", description="Host to serve on")
    port: int = Field(default=8080, description="Port to serve on")

    remain_after_trainer_loop: bool = Field(
        default=False,
        description="Whether to keep the viewer running after the trainer loop ends (blocking the process from exiting).",
    )

    force_update_scene: bool = Field(
        default=False,
        description="In instant_nurec systems, by default we cache the current scene the user is looking at even if the system has processed new scenes. This flag forces the viewer to always update to the newest processed scene.",
    )

    ray_subsampling_step: RaySubsamplingStep = Field(
        default_factory=RaySubsamplingStep, description="Render every nth ray (for performance)."
    )

    n_lidar_frames_displayed: int = Field(default=15, description="Number of lidar frames to display.")
    max_n_visible_frusta: int = Field(
        default=20,
        description=(
            "Maximum number of camera frusta to display. "
            "These are expensive for the in-browser renderer and should be kept low. "
            "All frusta are subsampled to match this argument (showing every nth frustum)."
        ),
    )
