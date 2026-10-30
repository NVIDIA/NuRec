# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nurec.config.base_schema import BaseConfigSchema, Field


class GroundMeshExtensionToCuboidsConfig(BaseConfigSchema):
    """Settings for extending the ground mesh to cover cuboid track positions."""

    enabled: bool = Field(
        default=False,
        description="Extend the ground mesh to cuboid track positions, to cover areas with no nearby scanned ground points.",
    )
    base_margin: float = Field(
        default=0.5,
        ge=0,
        description="Margin in meters added to each cuboid's base rectangle before sampling ground points.",
    )
    exclude_points_radius: float = Field(
        default=1.0,
        ge=0,
        description="2D radius in meters; a cuboid ground point is excluded if its k-th nearest scanned ground point is within this radius.",
    )
    exclude_points_knns: int = Field(
        default=1,
        gt=0,
        description="Number of nearest scanned ground point neighbours queried per cuboid ground point for the coverage check.",
    )
    cuboid_point_merge_cell_size: float = Field(
        default=0.5,
        gt=0,
        description="XY grid cell size in meters used to merge cuboid ground samples before meshing.",
    )
