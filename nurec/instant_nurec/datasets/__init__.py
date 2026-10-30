# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nurec.instant_nurec.datasets.datamodule import InstantNuRecPredictDataModule
from nurec.instant_nurec.datasets.instant_nurec_base import (
    BaseInstantNuRecDataset,
    BaseInstantNuRecIndexableDataset,
    BaseInstantNuRecIterableDataset,
    DummyInstantNuRecDataset,
    MixedInstantNuRecDataset,
)
from nurec.instant_nurec.datasets.instant_nurec_ncore import NCoreInstantNuRecDataset
from nurec.instant_nurec.datasets.instant_nurec_ncore_websocket import WebSocketNCoreInstantNuRecDataset
from nurec.instant_nurec.datasets.registry import make, register


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "make",
    "register",
    "InstantNuRecPredictDataModule",
    "BaseInstantNuRecDataset",
    "BaseInstantNuRecIndexableDataset",
    "BaseInstantNuRecIterableDataset",
    "WebSocketNCoreInstantNuRecDataset",
    "NCoreInstantNuRecDataset",
    "DummyInstantNuRecDataset",
    "MixedInstantNuRecDataset",
]
