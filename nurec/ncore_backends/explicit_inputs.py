# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional providers for sequence inputs that do not use a manifest."""

from __future__ import annotations

from typing import Protocol

from upath import UPath

import ncore.data

from nurec.ncore_backends.interface import DatasetBackend, UnsupportedBackendOperationError


class ExplicitSequenceInputProvider(Protocol):
    """Open sequence paths supplied without a metadata manifest."""

    def open_inputs(
        self,
        path_pattern: str,
        *,
        open_consolidated: bool,
        loader_max_workers: int | None,
    ) -> list[tuple[DatasetBackend, ncore.data.SequenceLoaderProtocol, UPath]]: ...

    def reopen_input(
        self,
        path: UPath,
        *,
        open_consolidated: bool,
        loader_max_workers: int | None,
    ) -> tuple[DatasetBackend, ncore.data.SequenceLoaderProtocol]: ...

    def open_group(
        self,
        path_pattern: str,
        *,
        open_consolidated: bool,
        loader_max_workers: int | None,
    ) -> tuple[DatasetBackend, ncore.data.SequenceLoaderProtocol, tuple[UPath, ...]]: ...


_provider: ExplicitSequenceInputProvider | None = None


def register_explicit_sequence_input_provider(provider: ExplicitSequenceInputProvider) -> None:
    """Register the optional explicit-input provider for this process."""

    global _provider
    if _provider is not None and _provider is not provider:
        raise ValueError("An explicit sequence input provider is already registered")
    _provider = provider


def require_explicit_sequence_input_provider() -> ExplicitSequenceInputProvider:
    """Return the registered provider or explain that this input is unavailable."""

    if _provider is None:
        raise UnsupportedBackendOperationError(
            "This entrypoint does not support explicit sequence path patterns; "
            "use --dataset-path with a dataset manifest"
        )
    return _provider
