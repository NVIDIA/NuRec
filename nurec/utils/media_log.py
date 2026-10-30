# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extension point for experiment trackers that need media types of their own.

TensorBoard goes through its own public API for the media it supports and needs nothing
here; MP4 logging is skipped for it, because ``add_video`` wants a decoded NCHW tensor.
To support a tracker that wraps media in tracker-specific objects, implement ``MediaBackend``
and call ``register_media_backend`` from an entrypoint that ships the tracker's package.
"""

from __future__ import annotations

import io

from typing import TYPE_CHECKING, Mapping, Optional, Protocol, runtime_checkable


if TYPE_CHECKING:
    import numpy as np

    from lightning_fabric.loggers.logger import Logger


@runtime_checkable
class MediaBackend(Protocol):
    """Media calls a tracker implements in terms of its own types."""

    def flush_media(
        self,
        logger: Logger,
        prefix: str,
        images: Mapping[str, io.BytesIO],
        videos: Mapping[str, tuple[io.BytesIO, str]],
        media_step: Optional[int],
    ) -> None:
        """Log a step's worth of images and videos.

        Images arrive JPEG-encoded so the caller keeps control of the encoding quality;
        each video is its bytes and its container format. A non-None ``media_step`` is the
        logical x-axis value: log the media against that step and commit it there, so images,
        videos and the step stay aligned.
        """

    def log_image(self, logger: Logger, key: str, image: np.ndarray, step: Optional[int]) -> None:
        """Log a single image."""

    def log_video(self, logger: Logger, key: str, path: str, fmt: Optional[str] = None) -> None:
        """Log a single video from a file on disk; ``fmt`` overrides format inference."""

    def finalize_preempted_run(self) -> None:
        """Mark the active run preempted and close it, if the tracker has one open."""

    def active_run(self) -> Optional[tuple[str, str]]:
        """Return ``(url, id)`` of the tracker's active run, or None when there is none."""


_backend: Optional[MediaBackend] = None


def register_media_backend(backend: MediaBackend) -> None:
    """Install the backend. Called by an entrypoint that ships the tracker's package."""
    global _backend
    _backend = backend


def media_backend() -> Optional[MediaBackend]:
    """Return the registered backend, or None when none is registered."""
    return _backend
