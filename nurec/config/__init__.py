# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nurec.config.base_schema import BaseConfigSchema
from nurec.config.checkpoint import ArtifactConfig, CheckpointConfig
from nurec.config.datamodule import BaseDatamoduleConfig, SODatamoduleConfig
from nurec.config.dataset import (
    CuboidTracksConfig,
    LidarDynamicPointsConfig,
    NCoreDatasetConfig,
    ValidLidarPointsCuboidTrackConfig,
    ValidPixelsCuboidTrackConfig,
    ValidPixelsFrameMaskConfig,
    ValidPixelsSceneFlowConfig,
    ValidPixelsTrafficLightConfig,
)
from nurec.config.logger import DummyLoggerConfig, TensorboardLoggerConfig, WandbLoggerConfig
from nurec.config.prober import ProberConfig
from nurec.config.scopedtimer import ScopedTimerConfig, VerbosityLevel
from nurec.config.sensor import LidarModelsConfig, SensorConfig
from nurec.config.systems import (
    GaussiansSystemConfig,
)
from nurec.config.trainer import TrainerConfig
from nurec.config.version import Version
from nurec.config.viewer import ViewerConfig


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "ArtifactConfig",
    "BaseConfigSchema",
    "BaseDatamoduleConfig",
    "CheckpointConfig",
    "CuboidTracksConfig",
    "DummyLoggerConfig",
    "GaussiansSystemConfig",
    "LidarDynamicPointsConfig",
    "LidarModelsConfig",
    "NCoreDatasetConfig",
    "ProberConfig",
    "ScopedTimerConfig",
    "SensorConfig",
    "SODatamoduleConfig",
    "TensorboardLoggerConfig",
    "TrainerConfig",
    "ValidLidarPointsCuboidTrackConfig",
    "ValidPixelsCuboidTrackConfig",
    "ValidPixelsFrameMaskConfig",
    "ValidPixelsSceneFlowConfig",
    "ValidPixelsTrafficLightConfig",
    "VerbosityLevel",
    "Version",
    "ViewerConfig",
    "WandbLoggerConfig",
]
