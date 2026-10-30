# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry for built-in and optional NCore dataset backends."""

from __future__ import annotations

import json

from collections.abc import Sequence

from nurec.ncore_backends.interface import DatasetBackend, DatasetRequest, JsonObject, ResolvedDataset


class UnsupportedDatasetError(ValueError):
    """Raised when no registered backend recognizes a dataset."""


class AmbiguousDatasetError(ValueError):
    """Raised when more than one registered backend recognizes a dataset."""


def _read_manifest(request: DatasetRequest) -> JsonObject:
    try:
        with request.metadata_path.open("r") as manifest_file:
            manifest: object = json.load(manifest_file)
    except FileNotFoundError as exc:
        raise UnsupportedDatasetError(f"Dataset manifest does not exist: {request.metadata_path}") from exc
    except OSError as exc:
        raise UnsupportedDatasetError(f"Could not read dataset manifest: {request.metadata_path}") from exc
    except (TypeError, ValueError) as exc:
        raise UnsupportedDatasetError(f"Dataset manifest is not valid JSON: {request.metadata_path}") from exc

    if not isinstance(manifest, dict) or not all(isinstance(key, str) for key in manifest):
        raise UnsupportedDatasetError(f"Dataset manifest must be a JSON object: {request.metadata_path}")

    return manifest


class DatasetBackendRegistry:
    """Ordered collection of available dataset backends."""

    def __init__(self, backends: Sequence[DatasetBackend] = ()) -> None:
        self._backends: dict[str, DatasetBackend] = {}
        for backend in backends:
            self.register(backend)

    def register(self, backend: DatasetBackend) -> None:
        if (existing := self._backends.get(backend.name)) is not None:
            if existing is backend:
                return
            raise ValueError(f"A dataset backend named {backend.name!r} is already registered")
        self._backends[backend.name] = backend

    def names(self) -> tuple[str, ...]:
        return tuple(self._backends)

    def get(self, name: str) -> DatasetBackend:
        try:
            return self._backends[name]
        except KeyError as exc:
            raise UnsupportedDatasetError(f"Dataset backend {name!r} is not registered") from exc

    def resolve(self, request: DatasetRequest) -> ResolvedDataset:
        manifest = _read_manifest(request)
        matches = [backend for backend in self._backends.values() if backend.recognizes(manifest)]
        if not matches:
            raise UnsupportedDatasetError(f"No registered NCore backend recognized {request.metadata_path}")
        if len(matches) > 1:
            names = ", ".join(sorted(backend.name for backend in matches))
            raise AmbiguousDatasetError(f"Multiple NCore backends recognized {request.metadata_path}: {names}")

        backend = matches[0]
        metadata = backend.resolve_metadata(request, manifest)
        return ResolvedDataset(backend=backend, request=request, metadata=metadata)


_DEFAULT_REGISTRY = DatasetBackendRegistry()


def register_backend(backend: DatasetBackend) -> None:
    _DEFAULT_REGISTRY.register(backend)


def registered_backend_names() -> tuple[str, ...]:
    return _DEFAULT_REGISTRY.names()


def get_backend(name: str) -> DatasetBackend:
    return _DEFAULT_REGISTRY.get(name)


def resolve_dataset(request: DatasetRequest) -> ResolvedDataset:
    return _DEFAULT_REGISTRY.resolve(request)
