# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from nurec.config.base_schema import BaseConfigSchema


class AssetHarvestingConfig(BaseConfigSchema):
    """Main configuration for asset harvesting pipeline"""

    class NCoreParserConfig(BaseConfigSchema):
        """Configuration for ncore parser stage of asset harvesting pipeline"""

        target_resolution: int
        num_lidar_ref_frames: int
        cam_pose_flip: List[int]
        max_threads: int
        occ_rate_threshold: float
        crop_min_area_ratio: float
        mask_exceed_threshold: float
        min_instance_pixels: int
        mask_overlap_threshold: float
        camera_ids: List[str]
        segmentation_ckpt: str = ""

    ncore_parser: NCoreParserConfig

    class ModelRepoConfig(BaseConfigSchema):
        """Hugging Face repo containing the Asset Harvester model bundle."""

        repo_id: str
        revision: str

    model_repo: ModelRepoConfig
