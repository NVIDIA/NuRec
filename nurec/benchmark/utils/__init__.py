# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility modules for benchmark operations."""

from nurec.benchmark.utils.bbox_projector import BBoxProjector
from nurec.benchmark.utils.object_data_loader import (
    ObjectDataLoader,
    load_camera_offset_json,
    normalize_track_id,
)
from nurec.benchmark.utils.object_level_iq_metric_visualization import (
    visualize_occlusion_impact,
    visualize_tracked_objects,
)
from nurec.benchmark.utils.occlusion_utils import (
    DEFAULT_PROXY_CONFIG,
    OcclusionStatistics,
    ProxyConfig,
    compute_occlusion_ratio,
    compute_occlusion_ratios_for_frame,
    should_skip_object,
)
from nurec.benchmark.utils.shard_data_manager import ShardDataManager


__all__ = [
    "BBoxProjector",
    "DEFAULT_PROXY_CONFIG",
    "ObjectDataLoader",
    "OcclusionStatistics",
    "ProxyConfig",
    "ShardDataManager",
    "compute_occlusion_ratio",
    "compute_occlusion_ratios_for_frame",
    "load_camera_offset_json",
    "normalize_track_id",
    "should_skip_object",
    "visualize_occlusion_impact",
    "visualize_tracked_objects",
]
