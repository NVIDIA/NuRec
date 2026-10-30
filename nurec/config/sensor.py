# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from nurec.config.base_schema import BaseConfigSchema, Field


class SpinningLidarConfig(BaseConfigSchema):
    n_vertical_bins: int
    frequency: int
    horizontal_fov: int
    n_horizontal_bins: int
    sync_angle_deg: int
    azimuths_deg_offset: list[float]
    inclinations_deg: list[float]


class LidarModelsConfig(BaseConfigSchema):
    HESAI_Pandar128: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for Hesai Pandar128 lidar model. It will be loaded from the Lidar intrinsic json file.",
    )
    HESAI_AT128: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for Hesai AT128 lidar model. It will be loaded from the Lidar intrinsic json file.",
    )
    spinning: dict[Literal["HESAI-AT128"] | Literal["HESAI-Pandar128"], SpinningLidarConfig]


class SensorConfig(BaseConfigSchema):
    lidar_models: LidarModelsConfig
