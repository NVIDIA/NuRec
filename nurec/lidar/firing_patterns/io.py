# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and write lidar firing-pattern artifacts."""

from __future__ import annotations

import io
import logging

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import numpy as np

import ncore.data

from nurec.artifact.artifact import FIRING_PATTERNS_DIR
from nurec.lidar.firing_patterns.recovery import choose_frame, describe_pattern, recover_firing_pattern
from nurec.lidar.firing_patterns.schema import (
    DIRECTIONS_ARRAY_NAME,
    PACKET_RAY_OFFSETS_ARRAY_NAME,
    SCHEMA_VERSION,
    TEMPLATE_PACKET_OFFSETS_ARRAY_NAME,
    TEMPLATE_RAY_OFFSETS_ARRAY_NAME,
    TIMESTAMP_OFFSETS_ARRAY_NAME,
    LidarFiringPatternTemplatesData,
)
from nurec.lidar.firing_patterns.templates import LidarFiringPatternTemplate, LidarFiringPatternTemplateSet
from nurec.ncore_backends.interface import UnsupportedBackendOperationError
from nurec.utils.types import ArtifactContents, NamedSerialized, RigTrajectories


if TYPE_CHECKING:
    from nurec.ncore_backends.interface import ResolvedDataset
    from nurec.utils.io.artifact_cache import ArtifactCache


log = logging.getLogger(__name__)


def _recover_firing_pattern_templates(
    sensor: ncore.data.LidarSensorProtocol,
) -> tuple[LidarFiringPatternTemplatesData, str]:
    """Build a packetless template set from one dense observed frame."""

    frame_index = choose_frame(sensor)
    directions, timestamp_offsets_us, frame_period_us = recover_firing_pattern(
        sensor,
        frame=frame_index,
    )
    frame_timestamps_us = np.asarray(sensor.frames_timestamps_us, dtype=np.int64)
    reference_frame_start_us = int(frame_timestamps_us[frame_index, 0])
    ray_count = len(directions)
    templates = LidarFiringPatternTemplatesData(
        directions_sensor=directions,
        timestamp_offsets_us=timestamp_offsets_us,
        template_ray_offsets=np.asarray([0, ray_count], dtype=np.int64),
        packet_ray_offsets=None,
        template_packet_offsets=None,
        reference_frame_start_us=reference_frame_start_us,
        frame_period_us=frame_period_us,
        frame_start_tolerance_us=0,
    )
    return templates, describe_pattern(directions, timestamp_offsets_us, frame_period_us)


def serialize_firing_patterns(
    rig_trajectories: RigTrajectories,
    lidar_sensors: Mapping[str, ncore.data.LidarSensorProtocol],
    dataset: "ResolvedDataset",
    *,
    include_modeled_lidars: bool = False,
) -> ArtifactContents:
    """Preserve native templates and recover missing patterns for model-free lidars.

    Args:
        rig_trajectories: Calibrated rig data defining logical lidar IDs and
            whether each sensor has a parametric model.
        lidar_sensors: Datasource sensor objects keyed by logical lidar ID,
            retaining dense frame arrays for any observed-frame recovery.
        dataset: Explicit resolved dataset supplying the native backend and manifest request.
        include_modeled_lidars: Opt in to native schedules for sensors with intrinsics.
            By default those sensors retain their established parametric rendering path
            and their optional components are never read. Model-free sensors always
            require native templates or a recoverable observed frame.

    Returns:
        Serialized artifact members named ``firing_patterns/<logical_id>.npz``,
        one per exported sensor. Modeled sensors receive a member only when explicitly
        enabled and native templates are available.

    Raises:
        ValueError: Aggregates unsupported model-free export, missing sensors,
            invalid native templates, filename collisions, and failed recovery.
    """
    res: ArtifactContents = []
    seen: set[str] = set()
    failures: list[str] = []
    for calibration in rig_trajectories.lidar_calibrations.values():
        logical_id = calibration.logical_sensor_name
        has_model = calibration.lidar_model_parameters is not None
        if has_model and not include_modeled_lidars:
            continue

        try:
            templates = dataset.backend.get_lidar_firing_pattern_templates(dataset.request, logical_id)
        except UnsupportedBackendOperationError as error:
            if not has_model:
                failures.append(f"firing-pattern export is unsupported for lidar '{logical_id}': {error}")
            else:
                log.warning("Using intrinsics for lidar '%s': %s", logical_id, error)
            continue
        except ValueError as error:
            failures.append(f"invalid dataset firing-pattern templates for lidar '{logical_id}': {error}")
            continue

        # Opted-in modeled sensors still use intrinsics when native templates are absent.
        if templates is None and has_model:
            continue
        if logical_id in seen:
            failures.append(
                f"two lidars share logical id '{logical_id}', so both would write "
                f"{FIRING_PATTERNS_DIR}/{logical_id}.npz"
            )
            continue
        seen.add(logical_id)
        if "/" in logical_id:
            failures.append(f"lidar id '{logical_id}' contains '/' and cannot name an artifact member")
            continue

        if templates is not None:
            source = "dataset metadata"
            packet_count = templates.packet_count
            packet_description = "packet layout unavailable" if packet_count is None else f"{packet_count} packets"
            description = (
                f"{templates.template_count} templates, {len(templates.directions_sensor)} rays, {packet_description}"
            )
        else:
            sensor = lidar_sensors.get(logical_id)
            if sensor is None:
                failures.append(f"no datasource sensor is available for lidar '{logical_id}'")
                continue
            try:
                templates, description = _recover_firing_pattern_templates(sensor)
            except ValueError as error:
                failures.append(
                    f"dataset has no firing-pattern templates for lidar '{logical_id}', "
                    f"and observed-frame recovery failed: {error}"
                )
                continue
            source = "observed dense frame (packet boundaries unavailable)"

        buffer = io.BytesIO()
        np.savez_compressed(buffer, **templates.artifact_arrays())
        res.append(NamedSerialized(filename=f"{FIRING_PATTERNS_DIR}/{logical_id}.npz", serialized=buffer.getvalue()))
        log.info("Firing pattern for lidar '%s' from %s: %s", logical_id, source, description)

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise ValueError(f"Could not serialize all required lidar firing patterns:\n{details}")

    return res


def populate_firing_patterns(
    artifact_cache: ArtifactCache,
    rig_trajectories: RigTrajectories,
    lidar_sensors: Mapping[str, ncore.data.LidarSensorProtocol],
    dataset: "ResolvedDataset",
) -> None:
    """Populate enabled firing patterns exactly once for any artifact producer.

    Args:
        artifact_cache: Mutable export cache whose configuration enables or
            disables firing-pattern output. An existing populated entry is reused.
        rig_trajectories: Calibrations describing the logical sensors to export.
        lidar_sensors: Datasource sensors keyed by logical lidar ID.
        dataset: Explicit resolved dataset supplying the native backend and manifest request.

    Returns:
        ``None``; updates ``artifact_cache.firing_patterns`` when enabled and unset.

    Raises:
        ValueError: If ``serialize_firing_patterns`` cannot export required patterns.
    """
    if not artifact_cache.artifact_config.firing_patterns.enabled:
        return
    if artifact_cache.firing_patterns is None:
        artifact_cache.firing_patterns = serialize_firing_patterns(
            rig_trajectories,
            lidar_sensors,
            dataset,
            include_modeled_lidars=artifact_cache.artifact_config.firing_patterns.include_modeled_lidars,
        )


def load_firing_patterns(source: Path | BinaryIO) -> "LidarFiringPatternTemplateSet":
    """Load versioned or legacy firing-pattern data from NPZ.

    Args:
        source: NPZ filesystem path or seekable binary file containing either
            the versioned template schema or the supported legacy single pattern.

    Returns:
        Validated runtime templates and their physical frame grid. The NPZ archive
        is closed after loading; callers retain ownership of a supplied file object.

    Raises:
        ValueError: If the schema version, required fields, scalar or array types,
            geometry, packet layout, or runtime timing constraints are invalid.
        OSError: If the source cannot be opened or read.
    """

    core_names = {
        "schema_version",
        "reference_frame_start_us",
        "frame_period_us",
        "frame_start_tolerance_us",
        DIRECTIONS_ARRAY_NAME,
        TIMESTAMP_OFFSETS_ARRAY_NAME,
        TEMPLATE_RAY_OFFSETS_ARRAY_NAME,
    }
    with np.load(source) as stored:
        stored_names = set(stored.files)

        def scalar_int(name: str) -> int:
            value = np.asarray(stored[name])
            if value.shape != () or value.dtype != np.int64:
                raise ValueError(f"{source}: {name} must be an int64 scalar")
            return int(value)

        if "schema_version" not in stored_names:
            legacy_names = {
                "directions_sensor",
                "timestamp_offsets_us",
                "frame_duration_us",
            }
            missing_legacy_names = legacy_names - stored_names
            if missing_legacy_names:
                raise ValueError(f"{source}: legacy firing pattern is missing {sorted(missing_legacy_names)}")
            directions = np.ascontiguousarray(stored["directions_sensor"], dtype=np.float32)
            timestamp_offsets_us = np.ascontiguousarray(stored["timestamp_offsets_us"], dtype=np.int64)
            norms = np.linalg.norm(directions, axis=1)
            if not np.isfinite(directions).all() or not np.isfinite(norms).all():
                raise ValueError(f"{source}: directions_sensor must contain finite vectors")
            if np.any(norms <= 1e-9):
                raise ValueError(f"{source}: directions_sensor must not contain zero-length vectors")
            if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-4):
                log.warning("%s: normalizing non-unit legacy ray directions", source)
                directions = (directions / norms[:, None]).astype(np.float32)
            return LidarFiringPatternTemplateSet(
                templates=(
                    LidarFiringPatternTemplate(
                        directions_sensor=directions,
                        timestamp_offsets_us=timestamp_offsets_us,
                    ),
                ),
                reference_frame_start_us=0,
                frame_period_us=scalar_int("frame_duration_us"),
                frame_start_tolerance_us=0,
            )

        missing_names = core_names - stored_names
        if missing_names:
            raise ValueError(f"{source}: firing-pattern templates are missing {sorted(missing_names)}")

        schema_version = scalar_int("schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"{source}: unsupported firing-pattern schema_version {schema_version}; expected {SCHEMA_VERSION}"
            )
        packet_names = {PACKET_RAY_OFFSETS_ARRAY_NAME, TEMPLATE_PACKET_OFFSETS_ARRAY_NAME}
        present_packet_names = packet_names & stored_names
        if present_packet_names and present_packet_names != packet_names:
            missing_packet_names = packet_names - present_packet_names
            raise ValueError(f"{source}: packet layout is partial; missing {sorted(missing_packet_names)}")

        return LidarFiringPatternTemplateSet.from_data(
            LidarFiringPatternTemplatesData(
                directions_sensor=np.array(stored[DIRECTIONS_ARRAY_NAME], copy=True, order="C"),
                timestamp_offsets_us=np.array(stored[TIMESTAMP_OFFSETS_ARRAY_NAME], copy=True, order="C"),
                template_ray_offsets=np.array(stored[TEMPLATE_RAY_OFFSETS_ARRAY_NAME], copy=True, order="C"),
                packet_ray_offsets=(
                    np.array(stored[PACKET_RAY_OFFSETS_ARRAY_NAME], copy=True, order="C")
                    if present_packet_names
                    else None
                ),
                template_packet_offsets=(
                    np.array(stored[TEMPLATE_PACKET_OFFSETS_ARRAY_NAME], copy=True, order="C")
                    if present_packet_names
                    else None
                ),
                reference_frame_start_us=scalar_int("reference_frame_start_us"),
                frame_period_us=scalar_int("frame_period_us"),
                frame_start_tolerance_us=scalar_int("frame_start_tolerance_us"),
            )
        )
