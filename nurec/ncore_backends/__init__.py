# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCore dataset backend registration and selection."""

from nurec.ncore_backends.explicit_inputs import (
    ExplicitSequenceInputProvider,
    register_explicit_sequence_input_provider,
    require_explicit_sequence_input_provider,
)
from nurec.ncore_backends.interface import (
    DatasetBackend,
    DatasetMetadata,
    DatasetRequest,
    JsonObject,
    MetadataValue,
    ResolvedDataset,
    SequencePartInfo,
    ShardDataLoaderProtocol,
    UnsupportedBackendOperationError,
)
from nurec.ncore_backends.ncore_v4 import NCORE_V4_BACKEND
from nurec.ncore_backends.registry import (
    AmbiguousDatasetError,
    DatasetBackendRegistry,
    UnsupportedDatasetError,
    get_backend,
    register_backend,
    registered_backend_names,
    resolve_dataset,
)


register_backend(NCORE_V4_BACKEND)

__all__ = [
    "AmbiguousDatasetError",
    "DatasetBackend",
    "DatasetBackendRegistry",
    "DatasetMetadata",
    "DatasetRequest",
    "ExplicitSequenceInputProvider",
    "JsonObject",
    "MetadataValue",
    "ResolvedDataset",
    "SequencePartInfo",
    "ShardDataLoaderProtocol",
    "UnsupportedBackendOperationError",
    "UnsupportedDatasetError",
    "get_backend",
    "register_backend",
    "register_explicit_sequence_input_provider",
    "registered_backend_names",
    "resolve_dataset",
    "require_explicit_sequence_input_provider",
]
