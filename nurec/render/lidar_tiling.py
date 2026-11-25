# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Renderer tiling and per-template GPU caches for recorded firing patterns."""

from __future__ import annotations

import logging
import math
import threading
import weakref

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from nurec.lidar.firing_patterns.templates import LidarFiringPatternTemplate


if TYPE_CHECKING:
    import torch

    from nurec.utils.batch import UnstructuredLidarTilingData
    from nurec.utils.unstructured_lidar import UnstructuredFov


log = logging.getLogger(__name__)

# Margin for limited-FOV pre-culling.
_FOV_MARGIN_RAD = np.deg2rad(0.5)


@dataclass
class _TilingCache:
    template: weakref.ReferenceType[LidarFiringPatternTemplate]
    by_device: dict[str, UnstructuredLidarTilingData] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


# Key by identity: templates contain NumPy arrays and are not hashable. Weak
# references preserve the original cache lifetime without retaining scene data.
_tiling_caches: dict[int, _TilingCache] = {}
_tiling_caches_lock = threading.RLock()


def _tiling_cache_for(template: LidarFiringPatternTemplate) -> _TilingCache:
    key = id(template)

    def discard(reference: weakref.ReferenceType[LidarFiringPatternTemplate]) -> None:
        with _tiling_caches_lock:
            cached = _tiling_caches.get(key)
            if cached is not None and cached.template is reference:
                del _tiling_caches[key]

    with _tiling_caches_lock:
        cache = _tiling_caches.get(key)
        if cache is None or cache.template() is not template:
            cache = _TilingCache(weakref.ref(template, discard))
            _tiling_caches[key] = cache
        return cache


def firing_pattern_fov(firing_pattern: LidarFiringPatternTemplate) -> "UnstructuredFov":
    """Return tiling bounds derived from every direction in this template.

    Args:
        firing_pattern: Validated non-empty template with sensor-frame unit directions.

    Returns:
        Unstructured angular bounds in radians, expanded by a 0.5-degree margin
        for limited coverage, or full periodic azimuth bounds for a spinning lidar.

    Raises:
        ValueError: If limited azimuth coverage crosses the +/-180-degree seam
            and cannot be represented by linear azimuth bins, or bounds are invalid.
    """

    from nurec.utils.unstructured_lidar import PERIODIC_AZ_SPAN_RAD, fov_from_bounds

    directions = firing_pattern.directions_sensor.astype(np.float64)
    azimuth = np.arctan2(directions[:, 1], directions[:, 0])
    elevation = np.arcsin(
        np.clip(
            directions[:, 2] / np.clip(np.linalg.norm(directions, axis=1), 1e-9, None),
            -1.0,
            1.0,
        )
    )
    sorted_azimuth = np.sort(azimuth)
    internal_gaps = np.diff(sorted_azimuth)
    wrap_gap = float(2.0 * math.pi - (sorted_azimuth[-1] - sorted_azimuth[0]))
    largest_internal_gap = float(internal_gaps.max()) if internal_gaps.size else 0.0
    circular_span = 2.0 * math.pi - max(largest_internal_gap, wrap_gap)

    if circular_span >= PERIODIC_AZ_SPAN_RAD:
        azimuth_low, azimuth_high = -math.pi, math.pi
    elif largest_internal_gap > wrap_gap + 1e-6:
        raise ValueError(
            "the firing-pattern template's limited azimuth coverage crosses the "
            "+/-180 degree seam, which unstructured linear azimuth binning cannot represent"
        )
    else:
        azimuth_low = max(float(sorted_azimuth[0]) - _FOV_MARGIN_RAD, -math.pi)
        azimuth_high = min(float(sorted_azimuth[-1]) + _FOV_MARGIN_RAD, math.pi)

    return fov_from_bounds(
        azimuth_low,
        azimuth_high,
        float(elevation.min()) - _FOV_MARGIN_RAD,
        float(elevation.max()) + _FOV_MARGIN_RAD,
        source="the firing-pattern template's measured directions",
    )


def build_unstructured_tiling(
    firing_pattern: LidarFiringPatternTemplate,
    frame_start_us: int,
    device: "torch.device",
) -> "UnstructuredLidarTilingData":
    """Build and cache gsplat tiling for one reusable template and device.

    Args:
        firing_pattern: Validated template whose ray order determines tile indexing.
        frame_start_us: Physical start timestamp in microseconds, applied to the
            returned metadata without changing cached relative firing times.
        device: Target torch device for tiling tensors; use a concrete device index
            when multiple GPUs are active.

    Returns:
        Tiling data sharing tensors cached by template identity and device, with
        a fresh metadata mapping for this frame's start and final firing time.

    Raises:
        ValueError: If the template's field of view or tiling inputs are invalid.
        RuntimeError: If tensor creation or transfer to the requested device fails.
    """

    import dataclasses

    import torch

    from nurec.utils.unstructured_lidar import build_tiling

    cache_key = str(device)
    cache = _tiling_cache_for(firing_pattern)
    with cache.lock:
        cached = cache.by_device.get(cache_key)
        if cached is None:
            cached = build_tiling(
                torch.from_numpy(firing_pattern.directions_sensor),
                torch.from_numpy(firing_pattern.ray_timestamps_us(0)),
                firing_pattern_fov(firing_pattern),
                # Match ray motion even when the first firing is after frame start.
                frame_start_us=0,
                frame_end_us=firing_pattern.span_us,
            ).to(device)
            cache.by_device[cache_key] = cached
            log.info(
                "Built unstructured tiling for a %d-ray template firing over %.1f ms on %s (cached).",
                firing_pattern.n_rays,
                firing_pattern.span_us / 1000.0,
                device,
            )

    meta = dict(cached.meta)
    meta["frame_start_us"] = float(frame_start_us)
    meta["frame_end_us"] = float(frame_start_us + firing_pattern.span_us)
    return dataclasses.replace(cached, meta=meta)
