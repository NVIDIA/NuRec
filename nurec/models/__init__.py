# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nurec.models.background as background
import nurec.models.calib as calib
import nurec.models.feature_volume as feature_volume
import nurec.models.object_feature_volume as object_feature_volume
import nurec.models.tracks_calib as tracks_calib

from nurec.models.background import (
    BackgroundColor,
    BaseBackground,
    ExtraSignalSkyMlpBackground,
    SkipBackground,
    SkyMlpBackground,
)
from nurec.models.composite import CompositeModel
from nurec.models.custom_modules import (
    _TruncBCE,
    _TruncExp,
)
from nurec.models.feature_volume import BaseFeatureVolume, TcnnMultiLevelEncoding
from nurec.models.input_embedding import (
    BaseInputEmbedding,
    HolisticRemapTimeInputEmbedding,
    HolisticStepTimeInputEmbedding,
    IndividualRemapTimeInputEmbedding,
    IndividualStepTimeInputEmbedding,
    SkipInputEmbedding,
    WeightedInstanceInputEmbedding,
)
from nurec.models.nn_extensions import LieGroupParameter
from nurec.models.object_feature_volume import (
    BaseObjectFeatureVolume,
    HashGridObjectFeatureVolume,
)
from nurec.models.post_processing import (
    BasePostProcessing,
    BilateralGridPerCamera,
    BilateralGridPerFrame,
    DinoV2PEPostProcessing,
    PPISPLegacyPostProcessing,
    PPISPPostProcessing,
)
from nurec.models.tracks_calib import (
    BaseTracksCalib,
    DirectTracksCalib,
    ElevationAwareUnicycleTracksCalib,
    SkipTracksCalib,
    StaticTracksCalib,
)


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "background",
    "calib",
    "feature_volume",
    "object_feature_volume",
    "tracks_calib",
    "BaseBackground",
    "SkyMlpBackground",
    "ExtraSignalSkyMlpBackground",
    "BackgroundColor",
    "SkipBackground",
    "BilateralGridPerFrame",
    "BilateralGridPerCamera",
    "PPISPLegacyPostProcessing",
    "PPISPPostProcessing",
    "LieGroupParameter",
    "_TruncExp",
    "_TruncBCE",
    "TcnnMultiLevelEncoding",
    "BaseFeatureVolume",
    "BaseInputEmbedding",
    "SkipInputEmbedding",
    "WeightedInstanceInputEmbedding",
    "HolisticRemapTimeInputEmbedding",
    "HolisticStepTimeInputEmbedding",
    "IndividualRemapTimeInputEmbedding",
    "IndividualStepTimeInputEmbedding",
    "BaseObjectFeatureVolume",
    "HashGridObjectFeatureVolume",
    "BaseTracksCalib",
    "DirectTracksCalib",
    "StaticTracksCalib",
    "ElevationAwareUnicycleTracksCalib",
    "SkipTracksCalib",
    "BasePostProcessing",
    "DinoV2PEPostProcessing",
    "CompositeModel",
]
