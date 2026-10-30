# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate NCore V4 dataset quality via quantitative metrics.

Computes per-sensor and per-sequence metrics for poses, cameras, LiDARs,
cuboids, and cross-sensor consistency.  Results can be exported to JSON,
YAML, or CSV, and a human-readable summary table is logged to stderr by
default.

Follows a structured per-sensor metrics pattern.
"""

from __future__ import annotations

import csv
import json
import logging
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np

from scipy.spatial.transform import Rotation

from ncore.data import (
    CuboidTrackObservation,
    RowOffsetStructuredSpinningLidarModelParameters,
)
from ncore.impl.common.transformations import (
    is_within_3d_bboxes,
    transform_point_cloud,
)
from ncore.sensors import RowOffsetStructuredSpinningLidarModel
from ncore_validation.utils import (
    NcoreV4Context,
    cuboid_bbox_to_rig,
    get_lidar_model_parameters_first,
    open_context,
)
from ncore_validation.validate_helpers import (
    MAX_POSE_ANGULAR_RATE_DEG_S,
    MIN_EGO_DISPLACEMENT_FOR_CLOCK_CHECK_M,
)


log = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_SEQUENCE_SENSOR_ID = "__sequence__"


# ---------------------------------------------------------------------------
# MetricsStore -- collection, aggregation, and multi-format export
# ---------------------------------------------------------------------------


class MetricsStore:
    """Collects metric values and provides aggregation and export methods."""

    @dataclass
    class Item:
        """A single metric measurement."""

        sensor_id: str
        component: str
        metric_name: str
        value: float
        frame_idx: int | None = None

    def __init__(self) -> None:
        """Initialize an empty metrics store."""
        self._items: list[MetricsStore.Item] = []
        self._descriptions: dict[tuple[str, str], str] = {}

    def add(
        self,
        sensor_id: str,
        component: str,
        metric_name: str,
        value: float,
        frame_idx: int | None = None,
        description: str = "",
    ) -> None:
        """Record a single metric measurement.

        Args:
            sensor_id: Sensor instance id, or ``_SEQUENCE_SENSOR_ID`` for
                sequence-level metrics.
            component: Logical grouping (e.g. ``"poses"``, ``"camera"``,
                ``"lidar"``).
            metric_name: Short snake_case name for the metric.
            value: Scalar measurement value.
            frame_idx: Optional frame index for per-frame metrics.
            description: Human-readable description of what this metric
                measures.  Only needs to be provided once per
                ``(component, metric_name)`` pair; subsequent calls with
                the same key may omit it.
        """
        self._items.append(
            MetricsStore.Item(
                sensor_id=sensor_id,
                component=component,
                metric_name=metric_name,
                value=value,
                frame_idx=frame_idx,
            )
        )
        if description:
            self._descriptions[(component, metric_name)] = description

    # -- aggregation --------------------------------------------------------

    def aggregate(
        self,
    ) -> dict[str, dict[str, dict[str, dict[str, float | int | str]]]]:
        """Aggregate items into ``component -> sensor_id -> metric_name -> stats``.

        Single-valued metrics produce ``{"value": x}``.  Multi-valued metrics
        produce ``{"avg", "min", "max", "std", "cnt"}``.  A ``"description"``
        key is included when one has been registered.

        Returns:
            Nested dict of ``component -> sensor_id -> metric_name -> stats``.
        """
        tree: dict[str, dict[str, dict[str, list[float]]]] = {}
        for item in self._items:
            tree.setdefault(item.component, {}).setdefault(item.sensor_id, {}).setdefault(item.metric_name, []).append(
                item.value
            )

        result: dict[str, dict[str, dict[str, dict[str, float | int | str]]]] = {}
        for component, sensors in tree.items():
            for sensor_id, metrics in sensors.items():
                for metric_name, vals in metrics.items():
                    if len(vals) == 1:
                        entry: dict[str, float | int | str] = {
                            "value": round(float(vals[0]), 6),
                        }
                    else:
                        arr = np.array(vals, dtype=np.float64)
                        entry = {
                            "avg": round(float(arr.mean()), 6),
                            "min": round(float(arr.min()), 6),
                            "max": round(float(arr.max()), 6),
                            "std": round(float(arr.std()), 6),
                            "cnt": len(vals),
                        }
                    description = self._descriptions.get((component, metric_name))
                    if description:
                        entry["description"] = description
                    result.setdefault(component, {}).setdefault(sensor_id, {})[metric_name] = entry
        return result

    # -- export -------------------------------------------------------------

    def to_dict(self, sequence_id: str) -> dict[str, Any]:
        """Serialize the store to a JSON-serializable dictionary.

        Args:
            sequence_id: Identifier for the evaluated sequence.

        Returns:
            Dictionary with ``schema_version``, ``sequence_id``, and
            ``metrics`` (the aggregated metric tree).
        """
        return {
            "schema_version": _SCHEMA_VERSION,
            "sequence_id": sequence_id,
            "metrics": self.aggregate(),
        }

    def save_json(self, path: Path, sequence_id: str) -> None:
        """Write aggregated metrics to a JSON file.

        Args:
            path: Output file path (parent directories are created by the caller).
            sequence_id: Identifier for the evaluated sequence.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(sequence_id), f, indent=2)

    def save_yaml(self, path: Path, sequence_id: str) -> None:
        """Write aggregated metrics to a YAML file.

        Requires ``pyyaml``; exits with an error message if not installed.

        Args:
            path: Output file path.
            sequence_id: Identifier for the evaluated sequence.
        """
        try:
            import yaml  # optional dependency
        except ImportError:
            log.error("pyyaml is required for YAML output (pip install pyyaml)")
            sys.exit(1)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(sequence_id), f, default_flow_style=False, sort_keys=False)

    def save_csv(self, path: Path) -> None:
        """Write aggregated metrics to a CSV file.

        Columns: ``component``, ``sensor_id``, ``metric``, ``value`` (for
        single-valued), ``avg``/``min``/``max``/``std``/``cnt`` (for
        multi-valued), and ``description``.

        Args:
            path: Output file path.
        """
        aggregated = self.aggregate()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "component",
                    "sensor_id",
                    "metric",
                    "value",
                    "avg",
                    "min",
                    "max",
                    "std",
                    "cnt",
                    "description",
                ]
            )
            for component in sorted(aggregated):
                for sensor_id in sorted(aggregated[component]):
                    for metric_name in sorted(aggregated[component][sensor_id]):
                        stats = aggregated[component][sensor_id][metric_name]
                        description = stats.get("description", "")
                        if "value" in stats:
                            writer.writerow(
                                [
                                    component,
                                    sensor_id,
                                    metric_name,
                                    stats["value"],
                                    "",
                                    "",
                                    "",
                                    "",
                                    1,
                                    description,
                                ]
                            )
                        else:
                            writer.writerow(
                                [
                                    component,
                                    sensor_id,
                                    metric_name,
                                    "",
                                    stats["avg"],
                                    stats["min"],
                                    stats["max"],
                                    stats["std"],
                                    stats["cnt"],
                                    description,
                                ]
                            )

    def print_summary(self) -> None:
        """Log a human-readable summary table to stderr.

        Single-valued metrics are shown as a single number; multi-valued
        metrics display ``avg``, ``min``, ``max``, ``std``, and ``n``.
        """
        aggregated = self.aggregate()
        if not aggregated:
            log.info("  (no metrics collected)")
            return
        name_width = max(
            len(metric_name)
            for sensors in aggregated.values()
            for metrics in sensors.values()
            for metric_name in metrics
        )
        name_width = max(name_width, 6)
        for component in sorted(aggregated):
            log.info(f"\n  {component}:")
            for sensor_id in sorted(aggregated[component]):
                label = sensor_id if sensor_id != _SEQUENCE_SENSOR_ID else "(sequence)"
                log.info(f"    {label}:")
                for metric_name in sorted(aggregated[component][sensor_id]):
                    stats = aggregated[component][sensor_id][metric_name]
                    if "value" in stats:
                        log.info(f"      {metric_name:<{name_width}}  {stats['value']:.4f}")
                    else:
                        log.info(
                            f"      {metric_name:<{name_width}}  avg={stats['avg']:>10.4f}  "
                            f"min={stats['min']:>10.4f}  max={stats['max']:>10.4f}  "
                            f"std={stats['std']:>8.4f}  n={stats['cnt']}"
                        )


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------

# -- Poses ------------------------------------------------------------------


def compute_pose_metrics(ctx: NcoreV4Context, store: MetricsStore) -> None:
    """Compute pose-related metrics from the rig→world dynamic pose.

    Metrics emitted (component ``"poses"``, sensor ``__sequence__``):

    * **rate_hz** -- pose sample rate.
    * **jitter_us** -- standard deviation of inter-pose timestamp deltas.
    * **temporal_coverage_pct** -- pose time span as a percentage of the
      sequence interval.
    * **total_displacement_m** -- cumulative ego travel distance.
    * **max_angular_rate_deg_s** -- peak rotation rate between consecutive
      poses (degrees per second).
    * **rig_forward_alignment** -- median per-segment dot product of the
      rig's +x axis with the instantaneous travel direction.  Robust to
      turning sequences (1.0 = correct NCore convention).

    Only the first valid poses reader is used.

    Args:
        ctx: Open NCore V4 sequence context.
        store: Metrics store to append measurements to.
    """
    for pose_reader in ctx.poses_readers.values():
        try:
            T_rig_worlds, ts = pose_reader.get_dynamic_pose("rig", "world")
        except KeyError:
            continue

        if ts.size < 2:
            continue

        sensor_id = _SEQUENCE_SENSOR_ID
        component = "poses"

        duration_us = float(ts[-1] - ts[0])
        duration_s = duration_us / 1e6
        if duration_s > 0:
            store.add(
                sensor_id,
                component,
                "rate_hz",
                len(ts) / duration_s,
                description="Rig pose sample rate",
            )

        deltas = np.diff(ts.astype(np.float64))
        store.add(
            sensor_id,
            component,
            "jitter_us",
            float(np.std(deltas)),
            description="Std-dev of inter-pose timestamp deltas",
        )

        interval_us = float(ctx.interval.stop - ctx.interval.start)
        if interval_us > 0:
            store.add(
                sensor_id,
                component,
                "temporal_coverage_pct",
                duration_us / interval_us * 100.0,
                description="Pose time span as % of sequence interval",
            )

        translations = T_rig_worlds[:, :3, 3].astype(np.float64)
        displacements = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        store.add(
            sensor_id,
            component,
            "total_displacement_m",
            float(displacements.sum()),
            description="Total ego travel distance",
        )

        if duration_s > 0 and len(T_rig_worlds) >= 2:
            # Compute angular velocity from consecutive rotation matrices.
            max_ang_rate = 0.0
            for i in range(len(T_rig_worlds) - 1):
                dt = float(ts[i + 1] - ts[i]) / 1e6
                if dt <= 0:
                    continue
                r_rel = T_rig_worlds[i + 1, :3, :3].astype(np.float64) @ T_rig_worlds[i, :3, :3].astype(np.float64).T
                angle_rad = float(Rotation.from_matrix(r_rel).magnitude())
                rate = np.degrees(angle_rad) / dt
                max_ang_rate = max(max_ang_rate, rate)
            angular_rate_desc = "Peak rotation rate between consecutive poses"
            if max_ang_rate > MAX_POSE_ANGULAR_RATE_DEG_S:
                angular_rate_desc += (
                    f" [WARNING: {max_ang_rate:.0f} deg/s exceeds "
                    f"{MAX_POSE_ANGULAR_RATE_DEG_S:.0f} deg/s physical limit -- "
                    f"likely corrupted pose entries]"
                )
            store.add(
                sensor_id,
                component,
                "max_angular_rate_deg_s",
                max_ang_rate,
                description=angular_rate_desc,
            )

        # Rig forward alignment: median per-segment dot(rig +x, instantaneous travel).
        # Robust to sequences with significant turns.
        if len(T_rig_worlds) >= 2:
            positions = T_rig_worlds[:, :3, 3].astype(np.float64)
            segments = positions[1:] - positions[:-1]
            norms = np.linalg.norm(segments, axis=1)
            moving = norms > 0.01
            if np.any(moving):
                rig_x_axes = T_rig_worlds[:-1, :3, 0].astype(np.float64)[moving]
                seg_dirs = segments[moving] / norms[moving, np.newaxis]
                dots = np.sum(rig_x_axes * seg_dirs, axis=1)
                alignment = float(np.median(dots))
                store.add(
                    sensor_id,
                    component,
                    "rig_forward_alignment",
                    alignment,
                    description="Median per-segment dot(rig +x, travel direction); 1.0 = correct convention",
                )

        break  # use first valid poses reader


# -- Camera -----------------------------------------------------------------


def compute_camera_metrics(ctx: NcoreV4Context, store: MetricsStore) -> None:
    """Compute camera-related metrics.

    Sequence-level metrics (component ``"camera"``, sensor ``__sequence__``):

    * **count** -- number of camera sensors.
    * **frame_balance** -- ratio of min to max frame count across cameras
      (1.0 means all cameras have the same number of frames).
    * **mask_presence** -- whether a masks component exists (1/0).

    Per-camera metrics (sensor = camera id):

    * **frame_rate_hz** -- capture rate derived from first/last timestamp.
    * **resolution_w** / **resolution_h** -- image dimensions from intrinsics.
    * **distortion_coeff_order** -- highest non-zero radial distortion
      coefficient index (indicates lens model complexity).

    Args:
        ctx: Open NCore V4 sequence context.
        store: Metrics store to append measurements to.
    """
    component = "camera"
    store.add(
        _SEQUENCE_SENSOR_ID,
        component,
        "count",
        float(len(ctx.camera_readers)),
        description="Number of camera sensors",
    )

    frame_counts: dict[str, int] = {}
    for camera_id, cam_reader in ctx.camera_readers.items():
        ts = cam_reader.frames_timestamps_us
        n_frames = ts.shape[0] if ts.size > 0 else 0
        frame_counts[camera_id] = n_frames

        if n_frames >= 2:
            duration_s = float(ts[-1, 1] - ts[0, 0]) / 1e6
            if duration_s > 0:
                store.add(
                    camera_id,
                    component,
                    "frame_rate_hz",
                    n_frames / duration_s,
                    description="Camera capture rate",
                )

        # Resolution from intrinsics
        for intrinsics_reader in ctx.intrinsics_readers.values():
            try:
                cam_params = intrinsics_reader.get_camera_model_parameters(camera_id)
                if cam_params is not None:
                    resolution = np.asarray(cam_params.resolution)
                    store.add(
                        camera_id,
                        component,
                        "resolution_w",
                        float(resolution[0]),
                        description="Image width in pixels",
                    )
                    store.add(
                        camera_id,
                        component,
                        "resolution_h",
                        float(resolution[1]),
                        description="Image height in pixels",
                    )
                    # Count non-zero distortion coefficients to assess model complexity.
                    if hasattr(cam_params, "radial_coeffs") and cam_params.radial_coeffs is not None:
                        coeffs = np.asarray(cam_params.radial_coeffs)
                        nonzero_indices = np.flatnonzero(np.abs(coeffs) > 1e-12)
                        highest_nonzero = int(nonzero_indices[-1]) + 1 if nonzero_indices.size > 0 else 0
                        store.add(
                            camera_id,
                            component,
                            "distortion_coeff_order",
                            float(highest_nonzero),
                            description="Highest non-zero radial distortion coefficient index",
                        )
                    break
            except (KeyError, AttributeError):
                continue

    if frame_counts:
        counts = list(frame_counts.values())
        if min(counts) > 0:
            store.add(
                _SEQUENCE_SENSOR_ID,
                component,
                "frame_balance",
                float(min(counts)) / float(max(counts)),
                description="Min/max frame count ratio across cameras (1.0 = balanced)",
            )
        else:
            store.add(
                _SEQUENCE_SENSOR_ID,
                component,
                "frame_balance",
                0.0,
                description="Min/max frame count ratio across cameras (1.0 = balanced)",
            )

    store.add(
        _SEQUENCE_SENSOR_ID,
        component,
        "mask_presence",
        1.0 if ctx.masks_readers else 0.0,
        description="Whether camera masks are present (1/0)",
    )


# -- LiDAR ------------------------------------------------------------------


def compute_lidar_metrics(ctx: NcoreV4Context, store: MetricsStore) -> None:
    """Compute LiDAR-related metrics (component ``"lidar"``, sensor = lidar id).

    Per-frame metrics (aggregated across all frames):

    * **points_per_frame** -- raw point count per sweep.
    * **distance_min_m** / **distance_max_m** / **distance_mean_m** --
      range return statistics per frame.
    * **intensity_completeness** -- fraction of points with non-zero
      intensity per frame.
    * **model_element_oob_frac** -- fraction of rays whose ``model_element``
      row or column index exceeds the grid dimensions (only emitted when
      out-of-bounds rays exist, indicating a converter bug).
    * **model_angular_error_max_deg** / **model_angular_error_mean_deg** --
      angular error between stored ray directions and directions predicted
      by the spinning LiDAR model for the corresponding ``model_element``.
      NuRec viability requires max error <= 0.5 deg.

    Sequence-level metric:

    * **element_coverage_pct** -- percentage of the spinning grid's
      ``(row, col)`` cells observed across all frames.

    Args:
        ctx: Open NCore V4 sequence context.
        store: Metrics store to append measurements to.
    """
    component = "lidar"

    for lidar_id, lidar_reader in ctx.lidar_readers.items():
        ts = lidar_reader.frames_timestamps_us
        if ts.size == 0:
            continue

        params = get_lidar_model_parameters_first(ctx.intrinsics_readers, lidar_id)
        has_spinning = isinstance(params, RowOffsetStructuredSpinningLidarModelParameters)
        lidar_model: RowOffsetStructuredSpinningLidarModel | None = None
        if has_spinning:
            lidar_model = RowOffsetStructuredSpinningLidarModel(params, device="cpu")

        unique_elements: set[tuple[int, int]] = set()

        for frame_idx in range(ts.shape[0]):
            frame_end_ts = int(ts[frame_idx, 1])
            try:
                direction = lidar_reader.get_frame_ray_bundle_data(frame_end_ts, "direction")
            except Exception:
                log.debug(
                    "Skipping frame: get_frame_ray_bundle_data(direction) failed "
                    "(component=%s, lidar_id=%s, frame_idx=%d, frame_end_ts=%d)",
                    component,
                    lidar_id,
                    frame_idx,
                    frame_end_ts,
                    exc_info=True,
                )
                continue
            n_pts = direction.shape[0]
            if n_pts == 0:
                continue

            store.add(
                lidar_id,
                component,
                "points_per_frame",
                float(n_pts),
                frame_idx=frame_idx,
                description="Number of LiDAR points per frame",
            )

            # Distance stats
            try:
                dist = lidar_reader.get_frame_ray_bundle_return_data(frame_end_ts, "distance_m", return_index=0)
                valid_dist = dist[np.isfinite(dist)]
                if valid_dist.size > 0:
                    store.add(
                        lidar_id,
                        component,
                        "distance_min_m",
                        float(valid_dist.min()),
                        frame_idx=frame_idx,
                        description="Minimum range return per frame",
                    )
                    store.add(
                        lidar_id,
                        component,
                        "distance_max_m",
                        float(valid_dist.max()),
                        frame_idx=frame_idx,
                        description="Maximum range return per frame",
                    )
                    store.add(
                        lidar_id,
                        component,
                        "distance_mean_m",
                        float(valid_dist.mean()),
                        frame_idx=frame_idx,
                        description="Mean range return per frame",
                    )
            except Exception:
                log.debug(
                    "Skipping distance metrics: "
                    "get_frame_ray_bundle_return_data(distance_m) failed "
                    "(component=%s, lidar_id=%s, frame_idx=%d, frame_end_ts=%d)",
                    component,
                    lidar_id,
                    frame_idx,
                    frame_end_ts,
                    exc_info=True,
                )

            # Intensity completeness
            try:
                intensity = lidar_reader.get_frame_ray_bundle_return_data(frame_end_ts, "intensity", return_index=0)
                if intensity.size > 0:
                    nonzero_frac = float(np.count_nonzero(intensity)) / intensity.size
                    store.add(
                        lidar_id,
                        component,
                        "intensity_completeness",
                        nonzero_frac,
                        frame_idx=frame_idx,
                        description="Fraction of points with non-zero intensity",
                    )
            except Exception:
                log.debug(
                    "Skipping intensity metric: "
                    "get_frame_ray_bundle_return_data(intensity) failed "
                    "(component=%s, lidar_id=%s, frame_idx=%d, frame_end_ts=%d)",
                    component,
                    lidar_id,
                    frame_idx,
                    frame_end_ts,
                    exc_info=True,
                )

            # Model accuracy (angular error between stored and model-predicted directions)
            if has_spinning:
                try:
                    if not lidar_reader.has_frame_ray_bundle_data(frame_end_ts, "model_element"):
                        continue
                    model_element = lidar_reader.get_frame_ray_bundle_data(frame_end_ts, "model_element")
                except Exception:
                    log.debug(
                        "Skipping model-element metrics: "
                        "get_frame_ray_bundle_data(model_element) failed "
                        "(component=%s, lidar_id=%s, frame_idx=%d, "
                        "frame_end_ts=%d)",
                        component,
                        lidar_id,
                        frame_idx,
                        frame_end_ts,
                        exc_info=True,
                    )
                    continue
                if model_element.shape[0] == 0:
                    continue
                n = min(direction.shape[0], model_element.shape[0])
                dir_f64 = direction[:n].astype(np.float64)
                norms = np.linalg.norm(dir_f64, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                dir_f64 = dir_f64 / norms
                model_element = model_element[:n]

                in_bounds = (model_element[:, 0] < params.n_rows) & (model_element[:, 1] < params.n_columns)
                n_oob = int((~in_bounds).sum())
                if n_oob > 0:
                    store.add(
                        lidar_id,
                        component,
                        "model_element_oob_frac",
                        n_oob / model_element.shape[0],
                        frame_idx=frame_idx,
                        description="Fraction of rays with out-of-bounds model_element indices",
                    )
                    dir_f64 = dir_f64[in_bounds]
                    model_element = model_element[in_bounds]
                if model_element.shape[0] == 0:
                    continue

                # Cast to int64: elements_to_sensor_rays internally converts to
                # torch.long but its uint16→int16 guard rejects values > 32767.
                model_dir = (
                    lidar_model.elements_to_sensor_rays(model_element.astype(np.int64)).cpu().numpy().astype(np.float64)
                )
                dot = np.clip(np.einsum("ij,ij->i", dir_f64, model_dir), -1.0, 1.0)
                errors_deg = np.degrees(np.arccos(dot))

                store.add(
                    lidar_id,
                    component,
                    "model_angular_error_max_deg",
                    float(errors_deg.max()),
                    frame_idx=frame_idx,
                    description="Max angular error between stored and model-predicted directions",
                )
                store.add(
                    lidar_id,
                    component,
                    "model_angular_error_mean_deg",
                    float(errors_deg.mean()),
                    frame_idx=frame_idx,
                    description="Mean angular error between stored and model-predicted directions",
                )

                for row, col in model_element:
                    unique_elements.add((int(row), int(col)))

        # Element coverage across all frames
        if has_spinning and unique_elements:
            total_cells = params.n_rows * params.n_columns
            coverage_pct = len(unique_elements) / total_cells * 100.0
            store.add(
                lidar_id,
                component,
                "element_coverage_pct",
                coverage_pct,
                description="% of spinning grid cells observed across all frames",
            )


# -- Cuboids ----------------------------------------------------------------


def compute_cuboid_metrics(ctx: NcoreV4Context, store: MetricsStore) -> None:
    """Compute cuboid annotation metrics (component ``"cuboids"``, sensor ``__sequence__``).

    * **track_count** -- number of unique object tracks.
    * **observation_density** -- mean number of observations per track.
    * **single_obs_ratio** -- fraction of tracks seen in only one frame
      (high values suggest incomplete tracking).
    * **class_diversity** -- number of distinct object class labels.
    * **lidar_points_per_cuboid** -- LiDAR points falling inside each
      cuboid bounding box (sampled over up to 200 observations and 10
      LiDAR frames).
    * **lidar_density_pts_per_m3** -- LiDAR point density normalized by
      cuboid volume.

    Skipped entirely when no cuboid readers are present.

    Args:
        ctx: Open NCore V4 sequence context.
        store: Metrics store to append measurements to.
    """
    if not ctx.cuboid_readers:
        return
    component = "cuboids"

    all_observations: list[CuboidTrackObservation] = []
    track_obs_count: dict[str, int] = {}
    class_names: set[str] = set()

    for component_instance_name, cuboid_reader in ctx.cuboid_readers.items():
        try:
            for obs in cuboid_reader.get_observations():
                all_observations.append(obs)
                track_id = str(getattr(obs, "track_id", "unknown"))
                track_obs_count[track_id] = track_obs_count.get(track_id, 0) + 1
                class_name = getattr(obs, "class_id", None) or getattr(obs, "class_name", None)
                if class_name:
                    class_names.add(str(class_name))
        except Exception:
            log.debug(
                "Skipping cuboid reader observations (component=%s, component_instance_name=%s)",
                component,
                component_instance_name,
                exc_info=True,
            )
            continue

    n_tracks = len(track_obs_count)
    store.add(
        _SEQUENCE_SENSOR_ID,
        component,
        "track_count",
        float(n_tracks),
        description="Number of unique cuboid tracks",
    )

    if n_tracks > 0:
        obs_counts = list(track_obs_count.values())
        store.add(
            _SEQUENCE_SENSOR_ID,
            component,
            "observation_density",
            float(np.mean(obs_counts)),
            description="Mean observations per track",
        )
        single_obs = sum(1 for count in obs_counts if count == 1)
        store.add(
            _SEQUENCE_SENSOR_ID,
            component,
            "single_obs_ratio",
            float(single_obs) / n_tracks,
            description="Fraction of tracks with only one observation",
        )

    store.add(
        _SEQUENCE_SENSOR_ID,
        component,
        "class_diversity",
        float(len(class_names)),
        description="Number of distinct object classes",
    )

    # LiDAR point density inside cuboid bounding boxes.
    # For each observation with a known pose and dimensions, count how many
    # LiDAR points from the closest-in-time frame fall inside the box.
    if not (ctx.lidar_readers and ctx.poses_readers and all_observations):
        return

    _compute_cuboid_lidar_density(ctx, store, all_observations, component)


def _compute_cuboid_lidar_density(
    ctx: NcoreV4Context,
    store: MetricsStore,
    observations: list[CuboidTrackObservation],
    comp: str,
) -> None:
    """Compute LiDAR point density inside cuboid bounding boxes.

    Builds a cache of up to 10 LiDAR frames (evenly spaced across the
    sequence) transformed into the rig frame.  For each cuboid observation
    (capped at 200 for runtime), transforms the cuboid's BBox3 into the
    rig frame (handling ``"rig"``, sensor, and ``"world"`` reference frames),
    then counts how many LiDAR points fall inside each oriented
    bounding box.

    Emits ``lidar_points_per_cuboid`` and ``lidar_density_pts_per_m3``
    into *store* under the given *comp* component.

    Args:
        ctx: Open NCore V4 sequence context.
        store: Metrics store to append measurements to.
        observations: List of ``CuboidTrackObservation`` instances.
        comp: Component name to use in the store (typically ``"cuboids"``).
    """
    # Build a cache of LiDAR points in rig frame per frame timestamp.
    # Points from multiple sensors at the same timestamp are concatenated.
    lidar_cache: dict[int, list[np.ndarray]] = {}

    for lidar_id, lidar_reader in ctx.lidar_readers.items():
        ts = lidar_reader.frames_timestamps_us
        if ts.size == 0:
            continue
        n_frames = ts.shape[0]
        indices = np.linspace(0, n_frames - 1, min(10, n_frames), dtype=int)
        for idx in indices:
            frame_end_ts = int(ts[idx, 1])
            try:
                direction = lidar_reader.get_frame_ray_bundle_data(frame_end_ts, "direction")
                distance = lidar_reader.get_frame_ray_bundle_return_data(frame_end_ts, "distance_m", return_index=0)
                if direction.shape[0] == 0 or distance.size == 0:
                    continue
                xyz_sensor = direction * distance[:, np.newaxis]
                lidar_to_rig = None
                for pose_reader in ctx.poses_readers.values():
                    try:
                        lidar_to_rig = pose_reader.get_static_pose(lidar_id, "rig")
                        break
                    except KeyError:
                        continue
                if lidar_to_rig is None:
                    continue
                xyz_rig = transform_point_cloud(xyz_sensor, lidar_to_rig)
                valid = np.isfinite(xyz_rig).all(axis=1)
                if valid.any():
                    lidar_cache.setdefault(frame_end_ts, []).append(xyz_rig[valid])
            except Exception:
                log.debug(
                    "Skipping LiDAR frame for cuboid-density cache "
                    "(component=%s, lidar_id=%s, frame_idx=%d, frame_end_ts=%d)",
                    comp,
                    lidar_id,
                    int(idx),
                    frame_end_ts,
                    exc_info=True,
                )

    # Concatenate per-timestamp point lists into single arrays.
    merged_cache: dict[int, np.ndarray] = {t: np.concatenate(arrs) for t, arrs in lidar_cache.items()}

    if not merged_cache:
        return

    sorted_lidar_ts = np.array(sorted(merged_cache.keys()), dtype=np.int64)

    # Look up rig→world dynamic pose for world-frame cuboid transforms.
    T_rig_worlds: np.ndarray | None = None
    rig_ts: np.ndarray | None = None
    for pose_reader in ctx.poses_readers.values():
        try:
            T_rig_worlds, rig_ts = pose_reader.get_dynamic_pose("rig", "world")
            break
        except KeyError:
            continue

    points_per_cuboid: list[float] = []
    density_per_cuboid: list[float] = []

    for obs in observations[:200]:  # cap to keep runtime bounded
        obs_ts = getattr(obs, "timestamp_us", None)
        if obs_ts is None:
            continue
        # Pick the nearest LiDAR timestamp, not just the first one >= obs_ts:
        # searchsorted returns the insertion point, so compare the candidates on
        # either side and keep whichever is closer in absolute time.  Cast obs_ts
        # to a native int so the subtraction can't wrap on an unsigned scalar.
        obs_ts_int = int(obs_ts)
        i = int(np.searchsorted(sorted_lidar_ts, obs_ts_int))
        left = max(i - 1, 0)
        right = min(i, len(sorted_lidar_ts) - 1)
        idx = (
            left
            if abs(int(sorted_lidar_ts[left]) - obs_ts_int) <= abs(int(sorted_lidar_ts[right]) - obs_ts_int)
            else right
        )
        closest_ts = int(sorted_lidar_ts[idx])
        if closest_ts not in merged_cache:
            continue
        pts_rig = merged_cache[closest_ts]

        bbox_rig = cuboid_bbox_to_rig(obs, ctx, T_rig_worlds, rig_ts)
        if bbox_rig is None:
            continue

        try:
            inside = is_within_3d_bboxes(pts_rig.astype(np.float64), bbox_rig.reshape(1, 9))[:, 0]
            n_inside = int(inside.sum())
            points_per_cuboid.append(float(n_inside))
            dims = bbox_rig[3:6]
            vol = float(dims[0] * dims[1] * dims[2])
            if vol > 0:
                density_per_cuboid.append(n_inside / vol)
        except Exception:
            log.debug(
                "Skipping is_within_3d_bboxes for cuboid (component=%s, obs_ts=%s, closest_lidar_ts=%d)",
                comp,
                obs_ts,
                closest_ts,
                exc_info=True,
            )
            continue

    if points_per_cuboid:
        for val in points_per_cuboid:
            store.add(
                _SEQUENCE_SENSOR_ID,
                comp,
                "lidar_points_per_cuboid",
                val,
                description="LiDAR points inside cuboid bounding box",
            )
    if density_per_cuboid:
        for val in density_per_cuboid:
            store.add(
                _SEQUENCE_SENSOR_ID,
                comp,
                "lidar_density_pts_per_m3",
                val,
                description="LiDAR point density inside cuboid (points/m^3)",
            )


# -- Cross-Sensor -----------------------------------------------------------


def compute_cross_sensor_metrics(ctx: NcoreV4Context, store: MetricsStore) -> None:
    """Cross-sensor consistency metrics (component ``"cross_sensor"``).

    * **temporal_overlap_pct** -- camera-LiDAR temporal overlap as a
      percentage of their union time span (sensor ``__sequence__``).
    * **clock_consistency_ratio** -- average ratio of camera-derived
      displacement to ego displacement; ~1.0 indicates good clock sync
      (sensor ``__sequence__``).  Only emitted when ego displacement
      exceeds ``MIN_EGO_DISPLACEMENT_FOR_CLOCK_CHECK_M``.

    Args:
        ctx: Open NCore V4 sequence context.
        store: Metrics store to append measurements to.
    """
    component = "cross_sensor"
    sensor_id = _SEQUENCE_SENSOR_ID

    cam_min, cam_max = None, None
    for cam_reader in ctx.camera_readers.values():
        ts = cam_reader.frames_timestamps_us
        if ts.size > 0:
            ts_start, ts_end = int(ts[:, 0].min()), int(ts[:, 1].max())
            cam_min = ts_start if cam_min is None else min(cam_min, ts_start)
            cam_max = ts_end if cam_max is None else max(cam_max, ts_end)

    lidar_min, lidar_max = None, None
    for lidar_reader in ctx.lidar_readers.values():
        ts = lidar_reader.frames_timestamps_us
        if ts.size > 0:
            ts_start, ts_end = int(ts[:, 0].min()), int(ts[:, 1].max())
            lidar_min = ts_start if lidar_min is None else min(lidar_min, ts_start)
            lidar_max = ts_end if lidar_max is None else max(lidar_max, ts_end)

    if cam_min is not None and lidar_min is not None:
        overlap_start = max(cam_min, lidar_min)
        overlap_end = min(cam_max, lidar_max)
        union_start = min(cam_min, lidar_min)
        union_end = max(cam_max, lidar_max)
        union_dur = union_end - union_start
        if union_dur > 0:
            overlap_dur = max(0, overlap_end - overlap_start)
            store.add(
                sensor_id,
                component,
                "temporal_overlap_pct",
                overlap_dur / union_dur * 100.0,
                description="Camera-LiDAR temporal overlap as % of union",
            )

    # Clock consistency: estimate camera-ego drift via displacement comparison.
    if ctx.camera_readers and ctx.poses_readers:
        T_rig_worlds = None
        rig_ts = None
        for pose_reader in ctx.poses_readers.values():
            try:
                T_rig_worlds, rig_ts = pose_reader.get_dynamic_pose("rig", "world")
                break
            except KeyError:
                continue

        if T_rig_worlds is not None and rig_ts.size >= 2:
            ego_displacement = float(np.linalg.norm(T_rig_worlds[-1, :3, 3] - T_rig_worlds[0, :3, 3]))
            if ego_displacement > MIN_EGO_DISPLACEMENT_FOR_CLOCK_CHECK_M:
                displacements: list[float] = []
                for camera_id, cam_reader in ctx.camera_readers.items():
                    cam_ts = cam_reader.frames_timestamps_us
                    if cam_ts.size == 0 or cam_ts[0, 0] == cam_ts[-1, 0]:
                        continue
                    cam_to_rig = None
                    for pose_reader in ctx.poses_readers.values():
                        try:
                            cam_to_rig = pose_reader.get_static_pose(camera_id, "rig")
                            break
                        except KeyError:
                            continue
                    if cam_to_rig is None:
                        continue
                    idx_first = int(
                        np.clip(
                            np.searchsorted(rig_ts, int(cam_ts[0, 0])),
                            0,
                            len(rig_ts) - 1,
                        )
                    )
                    idx_last = int(
                        np.clip(
                            np.searchsorted(rig_ts, int(cam_ts[-1, 0])),
                            0,
                            len(rig_ts) - 1,
                        )
                    )
                    pos_first = (T_rig_worlds[idx_first] @ cam_to_rig)[:3, 3]
                    pos_last = (T_rig_worlds[idx_last] @ cam_to_rig)[:3, 3]
                    displacements.append(float(np.linalg.norm(pos_last - pos_first)))

                if displacements:
                    # Ratio of camera displacement to ego displacement; ~1.0 means good sync.
                    ratios = [displacement / ego_displacement for displacement in displacements]
                    avg_ratio = float(np.mean(ratios))
                    store.add(
                        sensor_id,
                        component,
                        "clock_consistency_ratio",
                        avg_ratio,
                        description="Camera-to-ego displacement ratio (~1.0 = good clock sync)",
                    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--component-group",
    "component_groups",
    multiple=True,
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path(s) to V4 component group(s) or sequence meta JSON",
)
@click.option(
    "--open-consolidated/--no-open-consolidated",
    default=True,
    help="Pre-load consolidated zarr metadata",
)
@click.option(
    "--output-json",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Write metrics to JSON file",
)
@click.option(
    "--output-yaml",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Write metrics to YAML file",
)
@click.option(
    "--output-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Write metrics to CSV file",
)
@click.option("--quiet", is_flag=True, default=False, help="Suppress terminal summary")
def main(
    component_groups: tuple[Path, ...],
    open_consolidated: bool,
    output_json: Path | None,
    output_yaml: Path | None,
    output_csv: Path | None,
    quiet: bool,
) -> None:
    """Evaluate NCore V4 dataset quality via quantitative metrics.

    Opens the sequence, runs all metric computation functions in order,
    then exports results to the requested formats.

    Args:
        component_groups: Paths to NCore V4 component-group directories
            (or a single consolidated directory).
        open_consolidated: If ``True``, treat the first component-group
            path as a consolidated shard directory.
        output_json: If set, write aggregated metrics as JSON.
        output_yaml: If set, write aggregated metrics as YAML.
        output_csv: If set, write aggregated metrics as CSV.
        quiet: Suppress the terminal summary table.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        ctx = open_context(component_groups, open_consolidated=open_consolidated)
    except Exception as e:
        log.error(f"Failed to open sequence: {e}")
        sys.exit(1)

    store = MetricsStore()

    compute_pose_metrics(ctx, store)
    compute_camera_metrics(ctx, store)
    compute_lidar_metrics(ctx, store)
    compute_cuboid_metrics(ctx, store)
    compute_cross_sensor_metrics(ctx, store)

    sequence_id = ctx.reader.sequence_id or "unknown"

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        store.save_json(output_json, sequence_id)
        log.info(f"Metrics written to {output_json}")

    if output_yaml:
        output_yaml.parent.mkdir(parents=True, exist_ok=True)
        store.save_yaml(output_yaml, sequence_id)
        log.info(f"Metrics written to {output_yaml}")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        store.save_csv(output_csv)
        log.info(f"Metrics written to {output_csv}")

    if not quiet:
        log.info(f"\nMetrics for sequence: {sequence_id}")
        store.print_summary()
        log.info("")


if __name__ == "__main__":
    main()
