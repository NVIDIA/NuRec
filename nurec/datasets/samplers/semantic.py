# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import omegaconf

from nurec.datasets.samplers.base import (
    BaseLidarPointSampler,
    LidarPointSamplerReturn,
)
from nurec.utils.profiling import ScopedTimer, TimingTag


if TYPE_CHECKING:
    from nurec.datasets.ncore import NCORETrainDataset


class SemanticLidarPointSampler(BaseLidarPointSampler):
    """An LidarPointSampler which implements the ray sampling based on the semantic segmentation mask"""

    def __init__(self, config: omegaconf.dictconfig.DictConfig, dataset: NCORETrainDataset) -> None:
        self.semantic_classes_lidar_frame_masks = dataset.get_datasource().get_semantic_classes_frame_masks(
            class_names=config.class_names, camera_semantics=False, lidar_semantics=True
        )

    @ScopedTimer("SemanticLidarPointSampler.sample_lidar_points", TimingTag.DATALOADER)
    def sample_lidar_points(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        n_frame_point_samples: int,
        frame_valid_points_mask: npt.NDArray,
        unique_lidar_id: str,
        lidar_frame_idx: int,
    ) -> LidarPointSamplerReturn:
        """Samples valid lidar points based on semantic segmentation maps"""

        # Sample from *valid* points only
        frame_valid_point_idxs = np.where(frame_valid_points_mask)[0]

        # the number of valid points is a lower bound for the number of independent points we can produce
        # (this supports zero valid points in the limit)
        n_frame_point_samples = min(len(frame_valid_point_idxs), n_frame_point_samples)

        valid_semantic_mask = self.semantic_classes_lidar_frame_masks[unique_lidar_id][lidar_frame_idx].unpacked()
        masked_valid_semantic_mask = valid_semantic_mask[frame_valid_points_mask]
        n_frame_point_samples = min(masked_valid_semantic_mask.sum(), n_frame_point_samples)

        return LidarPointSamplerReturn(
            sampled_point_idxs=rng.choice(
                frame_valid_point_idxs[masked_valid_semantic_mask],
                size=n_frame_point_samples,
                replace=False,
                shuffle=False,
            )
        )


BaseLidarPointSampler.register_to_lidar_point_sampler_factory("semantic", SemanticLidarPointSampler)
