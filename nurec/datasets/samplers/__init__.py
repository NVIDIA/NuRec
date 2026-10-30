# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nurec.datasets.samplers.base as base
import nurec.datasets.samplers.fixed as fixed
import nurec.datasets.samplers.holdout as holdout
import nurec.datasets.samplers.image_crop as image_crop
import nurec.datasets.samplers.lidar_crop as lidar_crop
import nurec.datasets.samplers.semantic as semantic
import nurec.datasets.samplers.timestamp as timestamp
import nurec.datasets.samplers.uniform as uniform
import nurec.datasets.samplers.weighted as weighted

from nurec.datasets.samplers.base import (
    BaseBatchSampler,
    BaseCameraPixelSampler,
    BaseFrameSampler,
    BaseLidarPointSampler,
    BaseSensorSampler,
    DefaultBatchSampler,
    SkipCameraPixelSampler,
    SkipFrameSampler,
    SkipLidarPointSampler,
    SkipSensorSampler,
)
from nurec.datasets.samplers.fixed import FixedFrameSampler
from nurec.datasets.samplers.holdout import (
    CoverageFrameSampler,
    CoverageSensorSampler,
    HoldOutFrameSampler,
)
from nurec.datasets.samplers.image_crop import ImageCropCameraPixelSampler
from nurec.datasets.samplers.lidar_crop import LidarPointCloudCropSampler
from nurec.datasets.samplers.semantic import SemanticLidarPointSampler
from nurec.datasets.samplers.timestamp import TimestampFrameSampler
from nurec.datasets.samplers.uniform import (
    UniformFrameSampler,
    UniformLidarPointSampler,
    UniformSensorSampler,
)
from nurec.datasets.samplers.weighted import WeightedSensorSampler


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "base",
    "fixed",
    "holdout",
    "image_crop",
    "lidar_crop",
    "semantic",
    "timestamp",
    "uniform",
    "weighted",
    "BaseSensorSampler",
    "BaseFrameSampler",
    "BaseCameraPixelSampler",
    "BaseLidarPointSampler",
    "SkipSensorSampler",
    "SkipFrameSampler",
    "SkipCameraPixelSampler",
    "SkipLidarPointSampler",
    "BaseBatchSampler",
    "DefaultBatchSampler",
    "ImageCropCameraPixelSampler",
    "FixedFrameSampler",
    "SemanticLidarPointSampler",
    "TimestampFrameSampler",
    "UniformSensorSampler",
    "UniformFrameSampler",
    "UniformLidarPointSampler",
    "WeightedSensorSampler",
    "HoldOutFrameSampler",
    "CoverageFrameSampler",
    "CoverageSensorSampler",
    "LidarPointCloudCropSampler",
]
