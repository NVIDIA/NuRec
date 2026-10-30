# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nurec.datasets.ncore as ncore
import nurec.datasets.samplers as samplers

from nurec.datasets.base import BaseDataset, BaseDataSource, RigTrajectoriesProvider
from nurec.datasets.ncore import NCOREDataset, NCOREDataSource, NCORESequentialDataset, NCORETrainDataset
from nurec.datasets.registry import make
from nurec.datasets.tracks import CuboidTracks, TrackFlags


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "ncore",
    "samplers",
    "make",
    "NCOREDataset",
    "NCOREDataSource",
    "NCORESequentialDataset",
    "NCORETrainDataset",
    "TrackFlags",
    "CuboidTracks",
    "BaseDataset",
    "BaseDataSource",
    "RigTrajectoriesProvider",
]
