# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable keys for the on-disk scene cache and loaded-backend cache."""

from __future__ import annotations

import hashlib

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, unquote_plus, urlsplit, urlunsplit


SCENE_URL_HEADER = "x-nre-scene-url"
SCENE_ID_HEADER = "x-nre-scene-id"

# Authentication parameters change whenever an object-store URL is signed. They
# do not identify the object and must not create a fresh cache entry. Parameters
# such as versionId are intentionally retained.
_AUTH_QUERY_PARAMETER_PREFIXES = ("x-amz-", "x-goog-")
_AUTH_QUERY_PARAMETERS = {"temp_url_expires", "temp_url_sig"}


def _is_auth_query_parameter(name: str) -> bool:
    normalized = name.lower()
    return normalized in _AUTH_QUERY_PARAMETERS or normalized.startswith(_AUTH_QUERY_PARAMETER_PREFIXES)


def _remove_auth_query_parameters(query: str) -> str:
    """Remove signing fields without rewriting content-bearing query data."""
    retained_components = []
    for component in query.split("&"):
        encoded_name = component.partition("=")[0]
        if not _is_auth_query_parameter(unquote_plus(encoded_name)):
            retained_components.append(component)
    return "&".join(retained_components)


def canonicalize_scene_url(scene_url: str) -> str:
    """Return the stable object location without ephemeral URL credentials."""
    stripped = scene_url.strip()
    if not stripped:
        raise ValueError("Scene URL must not be empty")

    if Path(stripped).is_absolute():
        return Path(stripped).resolve().as_uri()

    parsed = urlsplit(stripped)
    if parsed.scheme.lower() == "file":
        if not stripped.lower().startswith("file://"):
            raise ValueError("Local scene URLs must use an absolute path or file:// URI")
        if parsed.netloc.lower() not in ("", "localhost"):
            raise ValueError(f"Non-local file URI authority is not supported: {parsed.netloc!r}")
        return Path(unquote(parsed.path)).resolve().as_uri()

    if parsed.scheme.lower() not in ("http", "https") or parsed.hostname is None:
        raise ValueError("Scene URL must use http://, https://, file://, or an absolute local path")

    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Scene URLs containing userinfo are not supported; use a public or presigned URL")

    query = _remove_auth_query_parameters(parsed.query)
    hostname = (parsed.hostname or "").lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"

    return urlunsplit(
        (
            parsed.scheme.lower(),
            netloc,
            parsed.path,
            query,
            "",
        )
    )


@dataclass(frozen=True)
class SceneCacheKey:
    """Logical key for a scene source in the on-disk cache and registry."""

    scene_id: str
    canonical_url: str | None = None

    @classmethod
    def from_url(cls, scene_id: str, scene_url: str) -> SceneCacheKey:
        return cls(scene_id=scene_id, canonical_url=canonicalize_scene_url(scene_url))

    @property
    def key_hash(self) -> str:
        """Return a stable hash of the scene ID and canonical artifact URL, if present."""
        canonical_url = self.canonical_url or "local"
        material = f"{self.scene_id}\0{canonical_url}".encode()
        return f"scene-{hashlib.sha256(material).hexdigest()}"


@dataclass(frozen=True)
class BackendCacheKey:
    """Logical key for a loaded backend created from exact artifact bytes.

    The scene and backend caches evict entries independently. A scene file may
    therefore be evicted and downloaded again while a backend created from its
    previous bytes remains in memory. Including the content SHA prevents that
    stale backend from being reused if the same source unexpectedly returns
    different bytes.
    """

    scene_key: SceneCacheKey
    content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_sha256", self.content_sha256.lower())

    @property
    def key_hash(self) -> str:
        """Return a stable hash for a backend loaded from the artifact's exact bytes."""
        material = f"{self.scene_key.key_hash}\0{self.content_sha256}".encode()
        return f"backend-{hashlib.sha256(material).hexdigest()}"


def require_scene_id(scene_id: str | None) -> str:
    """Validate the scene ID required whenever a scene URL is supplied."""
    if not scene_id:
        raise ValueError(f"Missing required metadata: {SCENE_ID_HEADER}")
    return scene_id
