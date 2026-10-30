# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serializing measured lidar firing patterns into the artifact.

A lidar with no parametric sensor model cannot have its rays synthesised from a model, so
serving replays the directions it was observed to fire. The pattern is measured here, at
export, and carried in the artifact as ``firing_patterns/<logical_lidar_id>.npz`` -- the
layout ``Artifact.firing_patterns`` reads and ``LidarFiringPattern.load`` parses.
"""

from __future__ import annotations

import io
import logging

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from nurec.artifact.artifact import FIRING_PATTERNS_DIR
from nurec.utils.lidar_pattern_recovery import describe_pattern, recover_firing_pattern
from nurec.utils.types import ArtifactContents, NamedSerialized, RigTrajectories


if TYPE_CHECKING:
    from nurec.utils.io.artifact_cache import ArtifactCache


log = logging.getLogger(__name__)


def serialize_firing_patterns(
    rig_trajectories: RigTrajectories,
    lidar_sensors: Mapping[str, object],
) -> ArtifactContents:
    """A pattern per lidar that has no sensor model, keyed by logical id.

    ``lidar_sensors`` is the datasource's sensors, keyed by logical lidar id
    (``NCOREDataSource.lidar_sensors``).

    Lidars that carry model parameters are skipped: serving synthesises their rays from the
    model, which covers the sensor's whole grid rather than one capture's firings.
    """
    res: ArtifactContents = []
    seen: set[str] = set()
    failures: list[str] = []
    for calibration in rig_trajectories.lidar_calibrations.values():
        # Calibrations are keyed by unique sensor id, the datasource's sensors by logical name.
        logical_id = calibration.logical_sensor_name
        if calibration.lidar_model_parameters is not None:
            continue
        if logical_id in seen:
            # The member name is the logical id, so a second one would write a duplicate zip
            # entry and leave which pattern serves that lidar up to the reader.
            failures.append(
                f"two lidars share logical id '{logical_id}', so both would write "
                f"{FIRING_PATTERNS_DIR}/{logical_id}.npz"
            )
            continue
        seen.add(logical_id)
        sensor = lidar_sensors.get(logical_id)
        if sensor is None:
            failures.append(f"no datasource sensor is available for lidar '{logical_id}'")
            continue
        if "/" in logical_id:
            failures.append(f"lidar id '{logical_id}' contains '/' and cannot name an artifact member")
            continue

        try:
            directions, offsets, frame_duration_us = recover_firing_pattern(sensor)
        except ValueError as error:
            failures.append(f"could not recover a firing pattern for lidar '{logical_id}': {error}")
            continue

        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            directions_sensor=directions,
            timestamp_offsets_us=offsets,
            frame_duration_us=np.asarray(frame_duration_us, dtype=np.int64),
        )
        res.append(NamedSerialized(filename=f"{FIRING_PATTERNS_DIR}/{logical_id}.npz", serialized=buffer.getvalue()))
        log.info(f"Firing pattern for lidar '{logical_id}': {describe_pattern(directions, offsets, frame_duration_us)}")

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise ValueError(f"Could not serialize all required model-free lidar firing patterns:\n{details}")

    return res


def populate_firing_patterns(
    artifact_cache: ArtifactCache,
    rig_trajectories: RigTrajectories,
    lidar_sensors: Mapping[str, object],
) -> None:
    """Populate enabled firing patterns exactly once for any artifact producer."""
    if not artifact_cache.artifact_config.firing_patterns.enabled:
        return
    if artifact_cache.firing_patterns is None:
        artifact_cache.firing_patterns = serialize_firing_patterns(rig_trajectories, lidar_sensors)
