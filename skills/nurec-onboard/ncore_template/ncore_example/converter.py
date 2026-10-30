# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example (skeleton) NCore V4 converter and CLI entry point.

This module is both the converter class (``ExampleConverter``) and the
``bazel run //ncore_example:convert`` CLI binary. Following NCore's per-converter
convention (``tools/data_converter/<dataset>/converter.py`` as both the class
and the CLI), this single file is the only Python source the agent customizes;
``bazel run`` invokes the ``cli()`` Click group declared at the bottom.

Start from this file when adding your own format. It demonstrates the converter
contract: get_sequence_ids, from_config, convert_sequence, using
SequenceComponentGroupsWriter and V4 component writers. Input is minimal (empty
root_dir or a single placeholder sequence); output is **structurally** valid V4
with all major component types (Poses, Intrinsics, Masks, Camera, Lidar, Radar,
Cuboids) but **intentionally semantically invalid** in several places so the
``validate.py`` Data-Quality tier has something to flag and so users can see
what realistic warnings look like:

* All static sensor-to-rig poses are identity (no real calibration loaded) --
  triggers ``warn_identity_sensor_translations`` and three
  ``warn_camera_optical_axis_tilt`` warnings (cameras end up looking straight
  up because no camera-to-NCore basis change is applied).
* The placeholder spinning LiDAR has its stored ``direction`` derived from
  the same model used to compute it, with ``row_azimuth_offsets_rad = 0`` --
  triggers ``warn_lidar_direction_collapsed_to_model`` (the LiDAR-smearing
  failure mode).
* The fisheye intrinsics use a placeholder ``max_angle`` that doesn't match
  the distortion polynomial -- triggers
  ``warn_camera_fisheye_max_angle_deviation``.
* ``row_elevations_rad`` and ``column_azimuths_rad`` are uniform synthetic
  grids -- triggers the matching ``warn_lidar_*_uniform`` warnings.

Real converters that load actual calibration and per-frame data should not
produce any of these warnings on a healthy sequence.  The skeleton's purpose
is to be a working pipeline scaffold, **not** a reference for ground-truth
output.
"""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

import click
import numpy as np

from ncore.data import (
    BBox3,
    CuboidTrackObservation,
    FThetaCameraModelParameters,
    LabelSource,
    OpenCVFisheyeCameraModelParameters,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)
from ncore.data.v4 import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)
from ncore.data_converter import FileBasedDataConverter, FileBasedDataConverterConfig
from ncore.impl.common.transformations import (  # noqa: F401 -- re-exported for real converters
    HalfClosedInterval,
    PoseInterpolator,
    se3_inverse,
    time_bounds,
)  # internal API; no public equivalent yet
from ncore.impl.data.v4.types import ComponentGroupAssignments  # internal API; no public equivalent yet

from ncore_example.utils import (  # noqa: F401 -- re-exported for real converters
    bin_per_ray_to_columns,
    build_lidar_model_from_per_frame_azimuth,
    build_pinhole_model_from_opencv_K_D,
    check_cuboid_track_timestamps,
    cuboid_track_sample_timestamps,
    sanitize_sequence_id,
    solve_static_extrinsic_mean,
    verify_camera_model_projection,
)


_PLACEHOLDER_JPEG = b"\xff\xd8\xff\xd9"
"""Minimal JPEG (SOI + EOI) used by the skeleton converter.

In a real converter, read actual image files::

    with open(image_path, "rb") as f:
        jpeg_bytes = f.read()
"""

# --- Spinning LiDAR model parameters (shared by intrinsics + frame writers) ---
#
# FILL IN: Replace ALL values below with real values from your sensor datasheet
# or dataset calibration.  Four silent correctness traps:
#
#   spinning_direction: "cw" or "ccw" viewed from above (rig +Z down). Wrong
#     value mirrors Y and causes a Z-flip in NuRec.  Most automotive LiDARs
#     (Velodyne, Ouster, Hesai) are "cw"; the placeholder is "ccw".
#   row_elevations_rad: real per-beam inclination from calibration, descending
#     order (highest beam first).  Never use linspace -- all spinning LiDARs
#     have non-uniform beam spacing.
#     See https://nvidia.github.io/ncore/data/sensor_models.html#lidar-models
#   column_azimuths_rad: per-column azimuth at frame start.  NCore asserts
#     relative_angle ordering is strictly increasing in the spin direction.
#     Derive from sensor data when per-column pose/heading is available.
#   row_azimuth_offsets_rad: per-beam azimuth offset on top of column azimuth.
#     Zero for most sensors; check datasheet (notably Ouster has non-zero).
#
# Model-frame choice: the native-to-rig yaw must live in EXACTLY ONE of
# (T_lidar_rig.R, column_azimuths_rad).  Splitting or dropping it leaves a
# constant rotation residual on every ray.
# Enforced by ``check_lidar_forward_ray_in_rig`` in
# ``ncore_validation/validate_lidar.py``.

_SPINNING_LIDAR_PARAMS = RowOffsetStructuredSpinningLidarModelParameters(
    spinning_frequency_hz=10.0,  # FILL IN: sensor spin rate (Hz)
    spinning_direction="ccw",  # FILL IN: verify from sensor datasheet; most automotive LiDARs are "cw"
    n_rows=4,  # FILL IN: number of laser beams (channels)
    n_columns=360,  # FILL IN: columns per revolution
    # SKELETON: row_elevations_rad / column_azimuths_rad / row_azimuth_offsets_rad
    # below are intentionally synthetic (uniform grids with zero per-beam offset).
    # validate.py fires warn_lidar_row_elevations_uniform,
    # warn_lidar_column_azimuths_uniform, and (combined with the model-derived
    # direction stream emitted by _populate_lidar_frame) also
    # warn_lidar_direction_collapsed_to_model. Real converters replace these
    # with the sensor's per-beam calibration.
    row_elevations_rad=np.linspace(
        np.deg2rad(15), np.deg2rad(-15), 4, dtype=np.float32
    ),  # FILL IN: use real per-beam angles (non-uniform); descending order
    column_azimuths_rad=np.linspace(
        0, 2 * np.pi, 360, endpoint=False, dtype=np.float32
    ),  # FILL IN: derive from sensor data if per-column pose is available
    row_azimuth_offsets_rad=np.zeros(
        4, dtype=np.float32
    ),  # FILL IN: per-row azimuth offsets from sensor datasheet (often called beam_azimuth_angles)
)
"""Placeholder spinning LiDAR intrinsics (``lidar_top``)."""


@dataclass(kw_only=True, slots=True)
class ExampleConverterConfig(FileBasedDataConverterConfig):
    """Configuration for the example (skeleton) converter.

    Extends FileBasedDataConverterConfig with store type, component group
    profile, and whether to write sequence meta JSON after conversion.

    Attributes:
        store_type: Output store type ("itar" or "directory").
        component_group_profile: Component group layout ("default",
            "separate-sensors", or "separate-all").
        store_sequence_meta: Whether to write {sequence_id}.json after conversion.
        seek_sec: Seconds to skip from the first pose timestamp (None = no seek).
        duration_sec: Maximum duration in seconds from the (possibly seeked) start
            (None = no limit).
    """

    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True
    seek_sec: float | None = None
    duration_sec: float | None = None


class ExampleConverter(FileBasedDataConverter):
    """
    Skeleton converter that writes minimal valid NCore V4 output.

    Use this as a reference when implementing your own converter.
    """

    def __init__(self, config: ExampleConverterConfig) -> None:
        """Initialize the converter from an example config.

        ``super().__init__(config)`` populates ``self.root_dir`` (from
        ``FileBasedDataConverter``) and ``self.output_dir`` (from
        ``BaseDataConverter``) as ``UPath`` objects, so use pathlib-style
        operators (``self.output_dir / sequence_id``) rather than string
        concatenation. Copy any other config fields you need onto ``self``
        below. NCore's reference converters at
        ``ncore/tools/data_converter/{kitti,waymo,colmap}/converter.py`` follow
        the same per-field copy pattern. The base class owns
        ``self._active_camera_ids`` and ``self._active_lidar_ids`` for
        ``--camera-id`` / ``--lidar-id`` scope filters; do not assign those
        attributes directly. Use ``get_active_camera_ids()`` and
        ``get_active_lidar_ids()`` when filtering the dataset's native sensor
        lists.

        Args:
            config: ExampleConverterConfig with root_dir, output_dir, store_type,
                component_group_profile, store_sequence_meta, and base sensor options.
        """
        super().__init__(config)
        self.store_type: Literal["itar", "directory"] = config.store_type
        self.component_group_profile: Literal["default", "separate-sensors", "separate-all"] = (
            config.component_group_profile
        )
        self.store_sequence_meta: bool = config.store_sequence_meta
        self.seek_sec: float | None = config.seek_sec
        self.duration_sec: float | None = config.duration_sec
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_sequence_ids(config: ExampleConverterConfig) -> list[str]:
        """Return sequence IDs to convert.

        The skeleton converter generates synthetic data, so the sequence id is
        fixed and ``config.root_dir`` is ignored.  Real converters discover
        sequences from ``config.root_dir`` -- e.g. by listing subdirectories,
        reading a manifest CSV/JSON, or parsing a dataset index -- and return
        the list of sequence identifiers passed to :meth:`convert_sequence`.

        Args:
            config: Converter config; unused by the skeleton, but real
                implementations should derive sequence ids from
                ``config.root_dir``.

        Returns:
            List of sequence identifier strings (the skeleton always returns
            ``["example_sequence_001"]``).
        """
        # --- FILL IN: Return the list of sequence IDs your dataset exposes. ---
        # Real converters scan config.root_dir (subdirectories, manifest, dataset
        # index, ...), pass each source name through sanitize_sequence_id(), and
        # return the sanitized IDs. Each returned ID is passed to convert_sequence().
        return ["example_sequence_001"]

    @staticmethod
    def from_config(config: ExampleConverterConfig) -> ExampleConverter:
        """Build an ExampleConverter instance from the given config.

        Args:
            config: ExampleConverterConfig instance.

        Returns:
            ExampleConverter ready to run convert_sequence.
        """
        # --- FILL IN: Return an instance of your converter. ---
        # Use this to pass through config and optionally do one-time setup
        # (e.g. load calibration, open a dataset index) that all sequences will use.
        #
        # TIP: NCore provides shared utilities for real converters in
        # ncore.impl.common.transformations: se3_inverse, transform_point_cloud,
        # transform_bbox, PoseInterpolator, MotionCompensator, time_bounds.
        return ExampleConverter(config)

    def convert_sequence(self, sequence_id: str) -> None:
        """Write valid NCore V4 output for one sequence using placeholder data.

        Demonstrates all V4 component types: Poses (dynamic rig->world,
        static camera->rig, optional static world->world_global), Intrinsics, Masks,
        Camera frames, LiDAR frames, Radar frames, and Cuboids. Uses synthetic
        data so the example runs without any real sensor input.

        Args:
            sequence_id: Identifier for this sequence (used for output path and meta).
        """
        self.logger.info("Converting sequence %s", sequence_id)

        # --- Sequence time range (required by NCore / NuRec). ---
        # FILL IN: timestamps come from _compute_sequence_interval (modality
        # intersection).  from_start_end treats `end` as inclusive (stop = end+1);
        # do NOT pass end+1 yourself.  Dynamic poses must exactly span the interval.
        t_start_us, t_end_us = self._compute_sequence_interval(sequence_id)
        sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(t_start_us, t_end_us)

        # --- Which sensors to export. ---
        # get_active_* respects --no-cameras, --camera-id, etc. Pass the full
        # list of sensor IDs your format has; the method returns the filtered list.
        # FILL IN: Replace these with your dataset's actual sensor IDs.
        camera_ids = self.get_active_camera_ids(
            ["pinhole_camera", "fisheye_camera", "ftheta_camera"]
        )  # FILL IN: real camera IDs
        lidar_ids = self.get_active_lidar_ids(["lidar_top"])  # FILL IN: real LiDAR IDs
        radar_ids = self.get_active_radar_ids(["radar_front"])  # FILL IN: real radar IDs (or remove if no radar)

        # --- Component group layout (required so NuRec finds components). ---
        # This defines how Poses, Cameras, Lidars, etc. are grouped in the store.
        # Use the same profile (e.g. separate-sensors) as in this example unless
        # you have a reason to change it.
        component_groups = ComponentGroupAssignments.create(
            camera_ids=camera_ids,
            lidar_ids=lidar_ids,
            radar_ids=list(radar_ids),
            point_clouds_ids=[],  # non-empty if ingesting pre-computed geometry (SfM, MVS)
            camera_labels_ids=[],  # non-empty if writing per-camera semantic or instance labels
            profile=self.component_group_profile,
        )

        # --- Create the NCore V4 writer. ---
        # One writer per sequence. output_dir/sequence_id will hold the store
        # (itar file or directory). sequence_id and sequence_timestamp_interval_us
        # are written into the store and must match what NuRec expects.
        output_path = self.output_dir / sequence_id
        store_writer = SequenceComponentGroupsWriter(
            output_dir_path=output_path,
            store_base_name=sequence_id,
            sequence_id=sequence_id,
            sequence_timestamp_interval_us=sequence_timestamp_interval_us,
            store_type=self.store_type,
            generic_meta_data={"source": "ncore_example"},  # FILL IN: update source tag
        )

        # --- Register component writers and write data. ---
        # Order: Poses -> Intrinsics -> Masks -> Camera / Lidar / Radar -> Cuboids.
        self._write_poses(store_writer, component_groups, camera_ids, lidar_ids, radar_ids, t_start_us, t_end_us)
        self._write_intrinsics(store_writer, component_groups, camera_ids, lidar_ids)
        self._write_masks(store_writer, component_groups, camera_ids)
        self._write_cameras(store_writer, component_groups, camera_ids, t_start_us, t_end_us)
        self._write_lidars(store_writer, component_groups, lidar_ids, t_start_us, t_end_us)
        self._write_radars(store_writer, component_groups, radar_ids, t_start_us)
        self._write_cuboids(store_writer, component_groups, t_start_us, t_end_us)

        # --- Finalize the store. ---
        # Call finalize() once all components are written. No further writes
        # to this sequence are allowed. You get back the paths to the store.
        ncore_4_paths = store_writer.finalize()

        # --- Write sequence meta JSON (required for NuRec and ncore_vis). ---
        # NuRec and the NCore viewer need a {sequence_id}.json file that
        # describes the sequence and where to find the component groups. Write
        # it from the reader's get_sequence_meta() after finalize().
        if self.store_sequence_meta:
            reader = SequenceComponentGroupsReader(ncore_4_paths)
            sequence_meta_path = output_path / f"{reader.sequence_id}.json"
            with sequence_meta_path.open("w") as f:
                json.dump(reader.get_sequence_meta().to_dict(), f, indent=2)
            self.logger.info("Wrote sequence meta %s", sequence_meta_path)

    # ------------------------------------------------------------------
    # Per-component write methods
    # ------------------------------------------------------------------

    def _compute_sequence_interval(self, sequence_id: str) -> tuple[int, int]:
        """Return ``(t_start_us, t_end_us)`` from the **intersection** of modality timestamps.

        Take ``max(starts)`` and ``min(ends)`` across poses, per-sensor
        frames, and cuboids.  NCore requires every stored timestamp to fall
        inside ``sequence_timestamp_interval_us``, so the intersection (not
        union) is the correct rule.

        FILL IN: collect ``(modality_min, modality_max)`` from every
        required modality, then intersect.  Optionally apply ``--seek-sec``
        / ``--duration-sec`` via ``ncore.impl.common.transformations.time_bounds``.
        """
        t_start_us, t_end_us = 0, 100_000  # FILL IN: replace with real per-modality intersection logic

        # FILL IN: apply --seek-sec / --duration-sec restriction after computing
        # the raw intersection above. Pass the full pose timestamp list (not just
        # [t_start_us, t_end_us]) so time_bounds can seek from the true first pose.
        if self.seek_sec is not None or self.duration_sec is not None:
            t_start_us, t_end_us = time_bounds([t_start_us, t_end_us], self.seek_sec, self.duration_sec)

        return t_start_us, t_end_us

    def _write_poses(
        self,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        camera_ids: list[str],
        lidar_ids: list[str],
        radar_ids: list[str],
        t_start_us: int,
        t_end_us: int,
    ) -> None:
        """Write Poses component: rig->world dynamic + static transforms.

        FILL IN: Decode poses from your format and write them.

        NCore expects rig->world as a *dynamic* pose (multiple 4x4 matrices
        with timestamps_us). The world->world_global *static* pose is optional
        as of NuRec 26.04 (defaults to identity when absent) but recommended for
        a global CRS; see inline comment below for details.

        Pose trajectory density:
            NuRec interpolates rig->world at every per-ray LiDAR timestamp during
            motion compensation.  A sparse trajectory (e.g. one pose per LiDAR frame)
            causes rowing artifacts -- horizontal stripes where adjacent sweep columns
            snap to a staircase-interpolated pose.

            Combine **every** available pose source into a single trajectory:
              1. Per-camera-image poses (primary) -- iterate ALL cameras, not just
                 one.  N cameras x M frames yields N x M waypoints.
              2. Frame-level / LiDAR-level poses -- if the dataset provides a separate
                 pose per frame at a distinct timestamp, include it.
              3. IMU / GPS / odometry -- if available at higher frequency, include too.
            Concatenate all sources, deduplicate by timestamp (np.unique), and sort.
            Waypoint spacing should be smaller than the LiDAR sweep duration
            (e.g. <50 ms for a 10 Hz LiDAR).

        Rig frame: +X forward, +Y left, +Z up.  Timestamps in microseconds.

        Constraint: timestamps_us[0] must equal the sequence interval start
        and timestamps_us[-1] must equal the inclusive end (interval.stop - 1).
        If these don't match, the writer raises an AssertionError.

        Args:
            store_writer: V4 sequence writer to register the PosesComponent on.
            component_groups: Component group assignments for the store layout.
            camera_ids: Camera sensor IDs needing camera->rig static poses.
            lidar_ids: LiDAR sensor IDs needing lidar->rig static poses.
            radar_ids: Radar sensor IDs needing radar->rig static poses.
            t_start_us: Sequence start timestamp in microseconds.
            t_end_us: Sequence end timestamp in microseconds (inclusive).
        """
        poses_writer = store_writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.poses_component_group,
            generic_meta_data={  # FILL IN: update calibration and egomotion source tags
                "calibration_type": "example:calib",
                "egomotion_type": "example:egomotion",
            },
        )

        # FILL IN: Collect rig->world poses from EVERY available source:
        #   1. Per-camera-image poses (primary -- iterate ALL cameras, not just one)
        #   2. Frame-level / LiDAR-level poses (if at distinct timestamps)
        #   3. IMU / GPS / odometry poses (if available at higher frequency)
        # Concatenate all poses + timestamps, deduplicate by timestamp
        # (np.unique), sort, and store.
        #
        # TIME-RANGE FILTERING (--seek-sec / --duration-sec):
        # Filter the pose arrays to [t_start_us, t_end_us] BEFORE storing.
        # Example:
        #   selected = (timestamps >= t_start_us) & (timestamps <= t_end_us)
        #   poses = poses[selected]
        #   timestamps = timestamps[selected]
        # At least two poses must remain after filtering.
        #
        # IMPORTANT: All pose computation (collection, concatenation,
        # deduplication, sorting, re-referencing, extrapolation) must stay in
        # float64. Cast to float32 only as the very last step before storing.
        # Use the canonical re-referencing pattern (preserves absolute frame-0
        # pose in world->world_global):
        #
        #   from ncore.impl.common.transformations import se3_inverse
        #   T_world_world_global = poses_f64[0].copy()   # float64, store below
        #   poses_f64 = se3_inverse(T_world_world_global)[None] @ poses_f64
        #   poses_f32 = poses_f64.astype(np.float32)
        #   # then: poses_writer.store_static_pose(
        #   #     source_frame_id="world",
        #   #     target_frame_id="world_global",
        #   #     pose=T_world_world_global,   # float64
        #   # )
        #
        # As of NuRec 26.04 the world->world_global edge is OPTIONAL: when
        # absent NuRec defaults it to identity float64. Storing it is still
        # recommended for datasets with a global CRS because NuRec reads it for
        # scene extent, geo-location, and USDZ export. The simpler
        # `T_ref_inv = inv(poses[0])` variant with world->world_global = identity
        # re-references the dynamic trajectory identically but throws away the
        # absolute frame-0 pose, which is lossy for datasets with a global CRS
        # (ECEF/UTM/map-aligned).
        #
        # Without re-referencing at all, raw global coordinates (GPS/UTM/ENU at
        # 10+ km) lose precision in float32, causing motion-compensation
        # artifacts (diagonal streaking).
        T_rig_world_start = np.eye(4, dtype=np.float32)
        T_rig_world_end = np.eye(4, dtype=np.float32)
        T_rig_world_end[0, 3] = 10.0  # 10 m forward along +x
        T_rig_worlds = np.stack([T_rig_world_start, T_rig_world_end])
        timestamps_us = np.array([t_start_us, t_end_us], dtype=np.uint64)
        poses_writer.store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            poses=T_rig_worlds,
            timestamps_us=timestamps_us,
        )
        # Store world->world_global static pose. As of NuRec 26.04 this edge is
        # OPTIONAL: when absent NuRec defaults it to identity float64. Storing it
        # is still recommended for datasets with a global CRS because NuRec reads
        # it for scene extent, geo-location, and USDZ export. Replace the identity
        # below with the actual world->world_global transform (4x4 SE3) if your
        # dataset provides one (e.g. ECEF); omit this call entirely for plain
        # reconstruction with no global frame.
        # dtype MUST be float64 for geo-referenced precision.
        poses_writer.store_static_pose(
            source_frame_id="world",
            target_frame_id="world_global",
            pose=np.eye(4, dtype=np.float64),  # FILL IN: replace with actual transform if global CRS exists
        )

        # FILL IN: Store a static camera->rig pose for each camera.
        # NuRec needs camera->rig to project points into images. Use your
        # dataset's extrinsic calibration (4x4 SE3: camera frame -> rig frame).
        #
        # ---- Camera frame convention (READ BEFORE FILLING IN) ----
        #
        # NCore's canonical camera frame (docs/data/conventions.rst):
        #   +Z  principal / optical axis (points OUT of the lens)
        #   +X  points RIGHT in the image
        #   +Y  points DOWN in the image
        # This matches OpenCV's camera convention.
        #
        # Your dataset's native camera frame may differ. Make the convention
        # decision EXPLICIT and document it. Two paths:
        #
        #   * Native is already OpenCV/NCore -> no basis change; document
        #     the fact inline and cite the dataset docs.
        #   * Native differs (e.g. +X forward, +Y left, +Z up) -> apply an
        #     explicit 3x3 basis change before storing T_camera_rig.
        #
        # Recipe for a non-NCore native convention (pick the columns that map
        # the NATIVE axes onto the NCore axes; each column of R_ncore_native
        # is the NATIVE-frame coordinates of the NCore axis):
        #
        #   # Example: native is (+X forward, +Y left, +Z up), NCore is
        #   # (+Z forward, +X right, +Y down):
        #   R_ncore_native = np.array([
        #       # col 0: NCore +X (right)  = native -Y  (left is +Y -> right is -Y)
        #       # col 1: NCore +Y (down)   = native -Z  (up is +Z -> down is -Z)
        #       # col 2: NCore +Z (fwd)    = native +X  (fwd is +X)
        #       [ 0, 0, 1],
        #       [-1, 0, 0],
        #       [ 0,-1, 0],
        #   ], dtype=np.float64)
        #   basis_change = np.eye(4, dtype=np.float64)
        #   basis_change[:3, :3] = R_ncore_native
        #   T_camera_rig_ncore = T_camera_rig_native @ basis_change
        #
        # If your dataset is already OpenCV convention, skip the basis change
        # and record "native is OpenCV/NCore; no basis change" in the
        # camera_frame_convention justification.
        #
        # dtype MUST be float32. NuRec's pose processing does not promote
        # input dtypes, so a float64 sensor->rig pose collides with the
        # float32 rig trajectory, raising `RuntimeError: double != float`.
        # The only pose that stays float64 is world->world_global.
        for camera_id in camera_ids:
            # SKELETON: identity is intentionally wrong. Identity leaves the
            # camera looking up rig +Z (no source-to-NCore basis change applied),
            # so validate.py will fire warn_camera_optical_axis_tilt for every
            # camera in the example output. Real converters replace this with
            # one of the two options below.
            #
            # Option A: Dataset provides a direct extrinsic (e.g. from calibration file):
            T_camera_rig = np.eye(4, dtype=np.float32)
            # Option B: Dataset only provides per-frame world poses for each sensor.
            # Solve the static extrinsic from the trajectory:
            #   T_camera_rig = solve_static_extrinsic_mean(
            #       rig_interpolator=rig_interpolator,
            #       sensor_world_poses=camera_world_poses,  # [N, 4, 4] float64
            #       sensor_timestamps_us=camera_timestamps_us,
            #   ).astype(np.float32)
            poses_writer.store_static_pose(
                source_frame_id=camera_id,
                target_frame_id="rig",
                pose=T_camera_rig,
            )

        # FILL IN: Store a static T_lidar_rig for each LiDAR.
        # If the LiDAR is the rig reference sensor (its poses define the rig
        # trajectory), this is identity. Otherwise use the LiDAR-to-rig
        # extrinsic calibration. The pose graph needs this edge for motion
        # compensation and LiDAR-to-camera projection. T_camera_rig and
        # T_lidar_rig must encode the correct physical sensor-to-sensor
        # relationship. Verify alignment with ncore_vis /
        # ncore_project_pc_to_img.
        #
        # IMPORTANT: T_lidar_rig.R is the OTHER half of the spinning model-frame
        # choice.  Split the native-to-rig yaw between T_lidar_rig.R and
        # column_azimuths_rad according to Option A (model frame = rig frame)
        # OR Option B (model frame = sensor native frame) -- see the extended
        # comment block above ``_SPINNING_LIDAR_PARAMS``.  DO NOT carry the yaw
        # in both places or in neither.
        #
        # dtype MUST be float32 (same reason as T_camera_rig above).
        for lidar_id in lidar_ids:
            # SKELETON: identity is intentionally wrong; combined with the
            # camera identities above this is what triggers
            # warn_identity_sensor_translations in the example output. Real
            # converters write the LiDAR-to-rig extrinsic from calibration.
            T_lidar_rig = np.eye(4, dtype=np.float32)
            poses_writer.store_static_pose(
                source_frame_id=lidar_id,
                target_frame_id="rig",
                pose=T_lidar_rig,
            )

        # FILL IN: Store a static T_radar_rig for each radar.
        # Same principle as camera and lidar: use the radar-to-rig extrinsic
        # calibration from your dataset.
        #
        # dtype MUST be float32 (same reason as T_camera_rig above).
        for radar_id in radar_ids:
            # SKELETON: identity is intentionally wrong (no calibration loaded);
            # real converters write the radar-to-rig extrinsic from calibration.
            T_radar_rig = np.eye(4, dtype=np.float32)
            poses_writer.store_static_pose(
                source_frame_id=radar_id,
                target_frame_id="rig",
                pose=T_radar_rig,
            )

    def _write_intrinsics(
        self,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        camera_ids: list[str],
        lidar_ids: list[str],
    ) -> None:
        """Write Intrinsics component: camera + LiDAR model parameters.

        Registers one IntrinsicsComponent.Writer and delegates to
        _write_camera_intrinsics() and _write_lidar_intrinsics().

        For vehicles with windshield refraction, NCore provides
        BivariateWindshieldModelParameters as an external distortion model.
        This is specialized and omitted here.

        Args:
            store_writer: V4 sequence writer to register the IntrinsicsComponent on.
            component_groups: Component group assignments for the store layout.
            camera_ids: Camera sensor IDs to store camera intrinsics for.
            lidar_ids: LiDAR sensor IDs to store LiDAR intrinsics for.
        """
        intrinsics_writer = store_writer.register_component_writer(
            IntrinsicsComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.intrinsics_component_group,
        )
        self._write_camera_intrinsics(intrinsics_writer, camera_ids)
        self._write_lidar_intrinsics(intrinsics_writer, lidar_ids)

    def _write_camera_intrinsics(
        self,
        intrinsics_writer: IntrinsicsComponent.Writer,
        camera_ids: list[str],
    ) -> None:
        """Store per-camera intrinsics (model parameters).

        FILL IN: Use the camera model that matches your dataset's cameras.

        NCore supports three camera models (all from ncore.data):
        OpenCVPinholeCameraModelParameters, OpenCVFisheyeCameraModelParameters,
        and FThetaCameraModelParameters. One example of each is shown below.

        Each camera model includes a ShutterType. Most datasets use GLOBAL
        (instantaneous capture) or ROLLING_TOP_TO_BOTTOM. All variants:
        GLOBAL, ROLLING_TOP_TO_BOTTOM, ROLLING_LEFT_TO_RIGHT,
        ROLLING_BOTTOM_TO_TOP, ROLLING_RIGHT_TO_LEFT.

        Args:
            intrinsics_writer: Already-registered IntrinsicsComponent.Writer.
            camera_ids: Camera sensor IDs to store intrinsics for.
        """
        # FILL IN: Build each camera's parameters from the dataset calibration,
        # then immediately verify them against source-derived projection samples
        # before storing. The expected image points below are skeleton examples;
        # real converters must compute them from the source camera model, not by
        # projecting through NCore. If calibration is sequence-invariant, move
        # this build-and-verify block to converter setup and cache the verified
        # parameters instead of repeating it in every convert_sequence() call.
        pinhole_K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float64)
        pinhole_D = np.zeros(5)
        pinhole_resolution = (1920, 1080)
        pinhole_params = build_pinhole_model_from_opencv_K_D(
            K=pinhole_K,
            D=pinhole_D,
            resolution_wh=pinhole_resolution,
            shutter_type=ShutterType.GLOBAL,
        )
        # FILL IN: Choose probe samples that span center-to-corner / image-edge
        # radii for each camera. Distortion-model mistakes are often invisible
        # near the principal point and largest near the image boundary. These are
        # NCore image points (pixel center = index + 0.5), the convention
        # verify_camera_model_projection compares against.
        pinhole_probe_image_points = np.array(
            [
                [pinhole_K[0, 2], pinhole_K[1, 2]],
                [0.5, 0.5],
                [pinhole_resolution[0] - 0.5, 0.5],
                [0.5, pinhole_resolution[1] - 0.5],
                [pinhole_resolution[0] - 0.5, pinhole_resolution[1] - 0.5],
            ],
            dtype=np.float32,
        )
        # FILL IN: Compute expected image points from the source pinhole model.
        # The skeleton source camera is undistorted OpenCV pinhole, so
        # K-projection is the independent reference. If the source has
        # distortion, apply its documented distortion formula here before
        # calling the verifier.
        pinhole_probe_rays = np.column_stack(
            [
                (pinhole_probe_image_points[:, 0] - pinhole_K[0, 2]) / pinhole_K[0, 0],
                (pinhole_probe_image_points[:, 1] - pinhole_K[1, 2]) / pinhole_K[1, 1],
                np.ones(pinhole_probe_image_points.shape[0], dtype=np.float32),
            ]
        ).astype(np.float32)
        verify_camera_model_projection(
            pinhole_params,
            pinhole_probe_rays,
            pinhole_probe_image_points,
        )

        # SKELETON: max_angle is intentionally inconsistent with the
        # zeroed radial_coeffs + this focal_length, so validate.py fires
        # warn_camera_fisheye_max_angle_deviation in the example output.
        # Real converters compute max_angle from the calibrated radial
        # polynomial via OpenCVFisheyeCameraModelParameters.compute_max_angle().
        fisheye_params = OpenCVFisheyeCameraModelParameters(
            resolution=np.array([1920, 1080], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([960.0, 540.0], dtype=np.float32),
            focal_length=np.array([500.0, 500.0], dtype=np.float32),
            radial_coeffs=np.zeros(4, dtype=np.float32),
            max_angle=np.pi / 2.0,
        )
        # FILL IN: Use source-frame camera rays that exercise the calibrated
        # FOV. These samples are shared by the fisheye and FTheta skeleton
        # examples only because both use simple angle-domain placeholder models.
        angle_probe_rays = np.array(
            [
                [0.0, 0.0, 1.0],
                [-0.8, -0.45, 1.0],
                [0.8, -0.45, 1.0],
                [-0.8, 0.45, 1.0],
                [0.8, 0.45, 1.0],
            ],
            dtype=np.float32,
        )
        ray_xy = angle_probe_rays[:, :2].astype(np.float64)
        ray_xy_norm = np.linalg.norm(ray_xy, axis=1)
        theta = np.arctan2(ray_xy_norm, angle_probe_rays[:, 2].astype(np.float64))
        unit_xy = np.divide(ray_xy, ray_xy_norm[:, None], out=np.zeros_like(ray_xy), where=ray_xy_norm[:, None] > 0.0)
        # FILL IN: Compute expected image points from the source fisheye
        # calibration. This skeleton has zero radial_coeffs, so OpenCV-fisheye's
        # delta(theta) polynomial reduces to delta(theta)=theta. Real converters
        # must evaluate their source KB/OpenCV-fisheye polynomial here. The
        # OpenCV-fisheye model does not shift principal_point, so these image
        # points use principal_point directly (no +0.5).
        fisheye_radius = theta  # zero radial_coeffs: delta(theta) = theta
        fisheye_probe_image_points = np.column_stack(
            [
                fisheye_params.focal_length[0] * fisheye_radius * unit_xy[:, 0] + fisheye_params.principal_point[0],
                fisheye_params.focal_length[1] * fisheye_radius * unit_xy[:, 1] + fisheye_params.principal_point[1],
            ]
        ).astype(np.float32)
        verify_camera_model_projection(
            fisheye_params,
            angle_probe_rays,
            fisheye_probe_image_points,
        )

        ftheta_params = FThetaCameraModelParameters(
            resolution=np.array([1920, 1080], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([960.0, 540.0], dtype=np.float32),
            reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
            pixeldist_to_angle_poly=np.array([0.0, 1.0 / 500.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            angle_to_pixeldist_poly=np.array([0.0, 500.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            max_angle=np.pi / 2.0,
        )
        # FILL IN: Compute expected image points from the source FTheta
        # calibration. The skeleton uses a linear angle->pixel-distance
        # polynomial (r = 500 * theta). NCore's FThetaCameraModel shifts the
        # stored principal_point by +0.5 internally, so its image points are
        # expressed about (principal_point + 0.5); the +0.5 here matches that.
        # Mirror it only when your source convention matches this skeleton.
        ftheta_radius = 500.0 * theta
        ftheta_probe_image_points = np.column_stack(
            [
                ftheta_radius * unit_xy[:, 0] + ftheta_params.principal_point[0] + 0.5,
                ftheta_radius * unit_xy[:, 1] + ftheta_params.principal_point[1] + 0.5,
            ]
        ).astype(np.float32)
        verify_camera_model_projection(
            ftheta_params,
            angle_probe_rays,
            ftheta_probe_image_points,
        )

        # FILL IN: Replace these placeholder IDs and parameters with every
        # camera the source dataset exposes.
        camera_intrinsics = {
            "pinhole_camera": pinhole_params,
            "fisheye_camera": fisheye_params,
            "ftheta_camera": ftheta_params,
        }
        for camera_id in camera_ids:
            intrinsics_writer.store_camera_intrinsics(
                camera_id=camera_id,
                camera_model_parameters=camera_intrinsics[camera_id],
            )

    def _write_lidar_intrinsics(
        self,
        intrinsics_writer: IntrinsicsComponent.Writer,
        lidar_ids: list[str],
    ) -> None:
        """Store per-LiDAR intrinsics for structured spinning sensors.

        Populate parameters directly from the dataset's per-beam calibration.
        Do NOT use ``np.linspace`` for ``row_elevations_rad`` (all spinning
        LiDARs have non-uniform beam spacing) or ``row_azimuth_offsets_rad``.
        ``column_azimuths_rad`` may be uniform (constant-spin sensors) but
        prefer real per-column data when available.  Default
        ``row_azimuth_offsets_rad`` to zeros when the calibration has no
        per-beam offset table.  See the ``_SPINNING_LIDAR_PARAMS`` comment
        block for the full parameter and model-frame guidance.

        Args:
            intrinsics_writer: Already-registered IntrinsicsComponent.Writer.
            lidar_ids: LiDAR sensor IDs to store intrinsics for.
        """
        for lidar_id in lidar_ids:
            intrinsics_writer.store_lidar_intrinsics(
                lidar_id=lidar_id,
                lidar_model_parameters=_SPINNING_LIDAR_PARAMS,
            )

    def _write_masks(
        self,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        camera_ids: list[str],
    ) -> None:
        """Write Masks component (per-camera static masks).

        NuRec expects this component to exist. The NuRec data quality guide
        **requires** ego-vehicle masks per camera for best reconstruction
        quality -- without them, ego-vehicle pixels (hood, bumper, roof rack)
        leak into the reconstruction. Pass an empty dict only as a fallback
        when the dataset provides no masks and they cannot be generated.

        Real masks are PIL images keyed by name, e.g.::

            from PIL import Image
            sky_mask = Image.open("sky_mask.png")
            masks_writer.store_camera_masks(camera_id, {"sky": sky_mask, "ego": ego_mask})

        Args:
            store_writer: V4 sequence writer to register the MasksComponent on.
            component_groups: Component group assignments for the store layout.
            camera_ids: Camera sensor IDs to store masks for.
        """
        masks_writer = store_writer.register_component_writer(
            MasksComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.masks_component_group,
        )
        for camera_id in camera_ids:
            # FILL IN: Replace {} with {"mask_name": pil_image, ...} if your
            # dataset provides static camera masks (e.g. sky, ego-vehicle).
            masks_writer.store_camera_masks(camera_id=camera_id, mask_images={})

    def _write_cameras(
        self,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        camera_ids: list[str],
        t_start_us: int,
        t_end_us: int,
    ) -> None:
        """Write Camera frames (images).

        Register one CameraSensorComponent.Writer per camera, then call
        store_frame() for every frame. ``frame_timestamps_us`` is
        ``[exposure_start, exposure_end]`` **per camera per frame** (not a
        single global timestamp). Rolling-shutter cameras need real per-camera
        timing; using a shared frame timestamp causes incorrect motion-blur
        compensation in NuRec.  Global-shutter: start == end is valid.

        Constraints: (1) end timestamps must be unique per writer, (2) both
        timestamps must fall within the sequence interval.

        Args:
            store_writer: V4 sequence writer to register CameraSensorComponents on.
            component_groups: Component group assignments for the store layout.
            camera_ids: Camera sensor IDs to write frames for.
            t_start_us: Timestamp (microseconds) used for placeholder frame data.
            t_end_us: Sequence end timestamp (inclusive); the placeholder second
                frame is clamped to it so ``--seek-sec`` / ``--duration-sec``
                cannot push it past the interval.
        """
        # FILL IN: use real per-frame timestamps. Clamp to t_end_us so a short
        # --duration-sec window keeps the placeholder frame inside the interval.
        t_second_us = min(t_start_us + 100_000, t_end_us)
        for camera_id in camera_ids:
            camera_writer = store_writer.register_component_writer(
                CameraSensorComponent.Writer,
                component_instance_name=camera_id,
                group_name=component_groups.camera_component_groups[camera_id],
            )
            # FILL IN: Loop over your actual frames and read real image bytes.
            # TIME-RANGE FILTERING (--seek-sec / --duration-sec):
            # Skip frames whose exposure interval falls outside the sequence
            # interval. Example:
            #   if frame_start_us < t_start_us:
            #       continue
            #   if frame_end_us >= sequence_timestamp_interval_us.stop:
            #       break
            camera_writer.store_frame(
                image_binary_data=_PLACEHOLDER_JPEG,
                image_format="jpeg",
                frame_timestamps_us=np.array([t_start_us, t_start_us], dtype=np.uint64),
                generic_data={},
                generic_meta_data={},
            )
            # Skip the second frame on a zero-width interval, where t_second_us
            # collapses to t_start_us and would duplicate the first frame's
            # end-of-frame timestamp (which the store rejects as non-unique).
            if t_second_us > t_start_us:
                camera_writer.store_frame(
                    image_binary_data=_PLACEHOLDER_JPEG,
                    image_format="jpeg",
                    frame_timestamps_us=np.array([t_second_us, t_second_us], dtype=np.uint64),
                    generic_data={},
                    generic_meta_data={},
                )

    def _write_lidars(
        self,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        lidar_ids: list[str],
        t_start_us: int,
        t_end_us: int,
    ) -> None:
        """Write LiDAR frames (point clouds).

        Register one LidarSensorComponent.Writer per LiDAR sensor, then call
        store_frame() for every sweep. Key arrays::

            direction    -- [N, 3] float32 unit-norm rays in sensor frame
            timestamp_us -- [N] uint64 per-ray timestamps (see below)
            distance_m   -- [R, N] float32 metric distances (R = number of returns)
            intensity    -- [R, N] float32 normalized to [0, 1]
            model_element -- [N, 2] uint16 (row, col) or None

        Same timestamp constraints as camera frames: end timestamp must be
        unique per frame, within the sequence interval, and per-point
        timestamps must be within [frame_start, frame_end].

        **Per-ray timestamps**: NuRec uses per-ray ``timestamp_us`` for motion
        compensation — always supply real per-ray capture times (e.g.
        column-based for spinning LiDARs).

        **Sweep start convention**: ``frame_timestamps_us[0]`` must be the
        sweep **start** (the physical time the first ray fires), NOT the sweep
        center.  Per-ray timestamps span ``[frame_ts, frame_ts + sweep_duration]``.
        Treating the source frame timestamp as the sweep center and subtracting
        half the sweep duration shifts every per-ray timestamp by ~50 ms
        (for a 10 Hz LiDAR), causing severe motion-compensation blur.  The
        ``check_lidar_camera_sweep_alignment`` validation check detects this
        by comparing LiDAR sweep midpoints against camera exposure midpoints.

        Args:
            store_writer: V4 sequence writer to register LidarSensorComponents on.
            component_groups: Component group assignments for the store layout.
            lidar_ids: LiDAR sensor IDs to write frames for.
            t_start_us: Timestamp (microseconds) used for placeholder frame data.
            t_end_us: Sequence end timestamp (inclusive); bounds the placeholder
                sweep so per-ray timestamps stay inside the interval.
        """
        for lidar_id in lidar_ids:
            lidar_writer = store_writer.register_component_writer(
                LidarSensorComponent.Writer,
                component_instance_name=lidar_id,
                group_name=component_groups.lidar_component_groups[lidar_id],
            )
            # FILL IN: Loop over your actual sweeps and read real point cloud data.
            # TIME-RANGE FILTERING (--seek-sec / --duration-sec):
            # Skip sweeps whose [sweep_start, sweep_end) falls outside the
            # sequence interval. Example:
            #   if frame_start_us < t_start_us:
            #       continue
            #   if frame_end_us >= sequence_timestamp_interval_us.stop:
            #       break
            self._write_spinning_lidar_frame(lidar_writer, t_start_us, t_end_us)

    def _write_spinning_lidar_frame(
        self,
        lidar_writer: LidarSensorComponent.Writer,
        t_start_us: int,
        t_end_us: int,
    ) -> None:
        """Write a placeholder frame for a spinning LiDAR.

        FILL IN: Replace with real per-sweep point-cloud data.

        Direction vectors must be unit-norm in **sensor coordinates**.
        Use ``xyz / norm(xyz)`` for sensor-frame XYZ, or derive from
        beam model for range images.  World-frame XYZ must be
        decompensated first via ``MotionCompensator``.

        ``model_element`` (``[N, 2] uint16 (row, col)``) must come from
        per-ray metadata, not synthesized by binning angles.

        Args:
            lidar_writer: Already-registered LidarSensorComponent.Writer.
            t_start_us: Timestamp (microseconds) for the placeholder frame.
            t_end_us: Sequence end timestamp (inclusive); the synthetic sweep is
                clamped to it so per-ray timestamps stay inside the interval.
        """
        # FILL IN: Loop over your actual sweeps and read real point clouds.
        # Direction vectors MUST be unit-norm (L2 norm ~= 1.0) in sensor coordinates.
        # If your data provides Cartesian XYZ in sensor frame, normalize:
        #   direction = xyz / norm(xyz).
        # If your data provides world-frame or ego-compensated XYZ, you MUST
        # decompensate first -- see the docstring above.
        # For spinning LiDARs, supply per-ray model_element (row, col) indices
        # consistent with _write_lidar_intrinsics; see NCore V4 LiDAR format.
        params = _SPINNING_LIDAR_PARAMS
        n_half = 5
        fwd_cols = np.arange(n_half, dtype=np.uint16)
        bwd_cols = np.array([params.n_columns // 2 + i for i in range(n_half)], dtype=np.uint16)
        # Use top row (positive elevation) for forward rays and bottom row
        # (negative elevation) for backward rays so the rig-frame AABB spans
        # the origin in all three axes.
        rows = np.concatenate([np.zeros(n_half, dtype=np.uint16), np.full(n_half, params.n_rows - 1, dtype=np.uint16)])
        cols = np.concatenate([fwd_cols, bwd_cols])
        model_element = np.column_stack([rows, cols])

        # FILL IN: use ``direction = xyz / norm(xyz)`` from real sensor-frame XYZ.
        # XYZ-derived direction is recommended regardless of which
        # row_azimuth_offsets_rad strategy you choose (sensor-native + preserve,
        # sensor-native + encode-in-column, synthetic-uniform + encode-in-column).
        # NCore's Waymo and Hesai P128 reference converters both use this approach.
        # Real converters: ``direction = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)``
        # where ``xyz`` is the per-ray sensor-frame Cartesian return.
        #
        # SKELETON: the synthesis below derives the stored direction from the
        # same model parameters used downstream to recompute it. Combined with
        # the placeholder zero ``row_azimuth_offsets_rad`` in
        # ``_SPINNING_LIDAR_PARAMS`` this is exactly the LiDAR-smearing
        # failure mode validate.py's ``warn_lidar_direction_collapsed_to_model``
        # detects, and is intentional in the skeleton so the warning is visible
        # in the example output. Real converters write XYZ-derived directions.
        cos_e = np.cos(params.row_elevations_rad[rows])
        az = params.column_azimuths_rad[cols] + params.row_azimuth_offsets_rad[rows]
        synth_xyz = np.stack([cos_e * np.cos(az), cos_e * np.sin(az), np.sin(params.row_elevations_rad[rows])], axis=-1)
        direction = (synth_xyz / np.linalg.norm(synth_xyz, axis=1, keepdims=True)).astype(np.float32)

        n_points = direction.shape[0]
        distances = np.ones(n_points, dtype=np.float32) * 10.0
        distances[:n_half] = 5.0  # forward rays at 5 m land inside the example cuboid

        # Per-ray timestamps: column index maps linearly to time within the sweep.
        # 100 ms is typical for a 10 Hz spinning LiDAR, but clamp to the sequence
        # interval so a short --seek-sec / --duration-sec window cannot push the
        # frame end (or any per-ray timestamp) past t_end_us.
        sweep_duration_us = min(100_000, t_end_us - t_start_us)
        frame_end_us = t_start_us + sweep_duration_us
        col_frac = cols.astype(np.float64) / max(params.n_columns - 1, 1)
        timestamp_us = (t_start_us + col_frac * sweep_duration_us).astype(np.uint64)

        # Defensive frame-window clamp (three-way reconciliation).
        #
        # NCore V4 asserts every per-ray ``timestamp_us`` falls inside
        # ``[frame_timestamps_us[0], frame_timestamps_us[1]]``. When per-ray
        # timestamps are computed from a nominal frame start plus per-column
        # offsets (or come from the sensor with per-column rounding, overlap
        # rays beyond the nominal column count, etc.), individual rays can land
        # a few microseconds outside the nominal sweep window. Reconcile three
        # bounds in this exact order:
        #
        #   1. Nominal frame window from ``frame_start + sweep_duration``.
        #   2. Clamp to the sequence interval ``[t_start_us, t_end_us]`` so the
        #      writer accepts the frame at all.
        #   3. WIDEN (not clip) to cover ``per_ray_ts.min()/max()`` so no ray is
        #      rejected.
        #
        # Example (uncomment and plug in real ``t_start_us``/``t_end_us``
        # from ``_compute_sequence_interval`` once you replace the placeholder):
        #
        #     frame_start = max(int(fr_ts), int(t_start_us_seq))
        #     frame_end = min(int(fr_ts + sweep_duration_us - 1), int(t_end_us_seq))
        #     frame_start = min(frame_start, int(timestamp_us.min()))
        #     frame_end = max(frame_end, int(timestamp_us.max()))
        lidar_writer.store_frame(
            direction=direction,
            timestamp_us=timestamp_us,
            model_element=model_element,
            distance_m=distances.reshape(1, -1),
            intensity=np.ones((1, n_points), dtype=np.float32) * 0.5,
            frame_timestamps_us=np.array([t_start_us, frame_end_us], dtype=np.uint64),
            generic_data={},
            generic_meta_data={},
        )

    def _write_radars(
        self,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        radar_ids: list[str],
        t_start_us: int,
    ) -> None:
        """Write Radar frames (not required for NuRec).

        RadarSensorComponent is structurally similar to LiDAR but without
        intensity or model_element. Key arrays::

            direction    -- [N, 3] float32 unit-norm rays in sensor frame
            timestamp_us -- [N] uint64 per-ray timestamps
            distance_m   -- [R, N] float32 metric distances (R = number of returns)

        Same timestamp constraints as LiDAR / camera frames apply.
        Radar data is optional and not used by NuRec; include it only if your
        dataset provides radar measurements and you want them in the shard.

        Args:
            store_writer: V4 sequence writer to register RadarSensorComponents on.
            component_groups: Component group assignments for the store layout.
            radar_ids: Radar sensor IDs to write frames for.
            t_start_us: Timestamp (microseconds) used for placeholder frame data.
        """
        for radar_id in radar_ids:
            radar_writer = store_writer.register_component_writer(
                RadarSensorComponent.Writer,
                component_instance_name=radar_id,
                group_name=component_groups.radar_component_groups[radar_id],
            )
            # FILL IN: Loop over your actual radar sweeps.
            # TIME-RANGE FILTERING: same pattern as LiDAR -- skip sweeps
            # outside [t_start_us, sequence_timestamp_interval_us.stop).
            n_points = 5
            direction = np.zeros((n_points, 3), dtype=np.float32)
            direction[:, 0] = 1.0  # all rays point forward (+X)
            radar_writer.store_frame(
                direction=direction,
                timestamp_us=np.full(n_points, t_start_us, dtype=np.uint64),
                distance_m=np.ones((1, n_points), dtype=np.float32) * 20.0,
                frame_timestamps_us=np.array([t_start_us, t_start_us], dtype=np.uint64),
                generic_data={},
                generic_meta_data={},
            )

    def _write_cuboids(
        self,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        t_start_us: int,
        t_end_us: int,
    ) -> None:
        """Write Cuboids component (3-D bounding boxes).

        Register one CuboidsComponent.Writer, then call store_observations()
        with all CuboidTrackObservation objects. Each observation needs:
        track_id, class_id, timestamp, reference_frame_id (must match a known
        pose frame), reference_frame_timestamp_us, and a BBox3 (centroid +
        dimensions + XYZ Euler rotation in radians).

        Key traps:
          - Timestamps must fall within the sequence interval.  Pose
            interpolation handles non-exact matches, but rig/world-frame
            cuboids should have a pose waypoint at the label timestamp to
            avoid speed-proportional spatial offsets.
          - World re-referencing: if ``_write_poses`` applies a non-identity
            ``T_world_world_global``, cuboids must use the same ``world``.
            Use ``reference_frame_id="world_global"`` with raw centroids, or
            pre-transform centroids by ``se3_inverse(T_world_world_global)``.
            Caught by ``check_cuboid_rig_proximity`` in
            ``ncore_validation/validate_cuboids.py``.
          - Centroid must be the geometric center (midpoint of all axes).
            Some datasets use bottom-center: add ``dim_z / 2`` to Z.

        Args:
            store_writer: V4 sequence writer to register the CuboidsComponent on.
            component_groups: Component group assignments for the store layout.
            t_start_us: Timestamp (microseconds) used for placeholder cuboid data.
        """
        cuboids_writer = store_writer.register_component_writer(
            CuboidsComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.cuboid_track_observations_component_group,
        )
        # FILL IN: Build observations from your annotation data.
        # The three observations below demonstrate all reference_frame_id
        # variants: rig frame, sensor frame, and world frame.  Because the
        # example uses identity for all sensor→rig poses and rig→world at
        # t=0 is also identity, all three resolve to the same rig-frame
        # position.
        #
        # PER-TRACK TIMESTAMPS: select each track's timestamps against the
        # stored sensor coverage, interpolate poses at any added boundary
        # timestamp, and check every track before storing; the number checked
        # is recorded in the build summary. See conventions.md "Tracks lost to
        # interval clipping". Example:
        #   sample_us = cuboid_track_sample_timestamps(source_us, coverage_lo, coverage_hi)
        #   ...build this track's observations at sample_us...
        #   check_cuboid_track_timestamps(track_id, stored_us, source_us, coverage_lo, coverage_hi)
        cuboids_writer.store_observations(
            [
                CuboidTrackObservation(
                    track_id="example-track-001",
                    class_id="car",
                    timestamp_us=t_start_us,
                    reference_frame_id="rig",
                    reference_frame_timestamp_us=t_start_us,
                    bbox3=BBox3(
                        centroid=(5.0, 0.0, 0.0),
                        dim=(4.5, 2.0, 3.0),
                        rot=(0.0, 0.0, 0.0),
                    ),
                    source=LabelSource.EXTERNAL,
                ),
                CuboidTrackObservation(
                    track_id="example-track-002",
                    class_id="car",
                    timestamp_us=t_start_us,
                    reference_frame_id="lidar_top",
                    reference_frame_timestamp_us=t_start_us,
                    bbox3=BBox3(
                        centroid=(5.0, 0.0, 0.0),
                        dim=(4.5, 2.0, 3.0),
                        rot=(0.0, 0.0, 0.0),
                    ),
                    source=LabelSource.EXTERNAL,
                ),
                CuboidTrackObservation(
                    track_id="example-track-003",
                    class_id="car",
                    timestamp_us=t_start_us,
                    reference_frame_id="world",
                    reference_frame_timestamp_us=t_start_us,
                    bbox3=BBox3(
                        centroid=(5.0, 0.0, 0.0),
                        dim=(4.5, 2.0, 3.0),
                        rot=(0.0, 0.0, 0.0),
                    ),
                    source=LabelSource.EXTERNAL,
                ),
                CuboidTrackObservation(
                    track_id="example-track-001",
                    class_id="car",
                    timestamp_us=t_end_us,
                    reference_frame_id="rig",
                    reference_frame_timestamp_us=t_end_us,
                    bbox3=BBox3(
                        centroid=(15.0, 0.0, 0.0),
                        dim=(4.5, 2.0, 3.0),
                        rot=(0.0, 0.0, 0.0),
                    ),
                    source=LabelSource.EXTERNAL,
                ),
            ]
        )


# ----------------------------------------------------------------------------
# CLI entry point.
# Following NCore's tools/data_converter/<dataset>/converter.py convention,
# the same module that defines the converter class also exposes the click
# group that ``bazel run //ncore_example:convert`` invokes.
# ----------------------------------------------------------------------------


def _ctx_to_config(ctx: click.Context, **subcommand_kwargs) -> dict:
    """Build a config dict from the Click group context and subcommand kwargs.

    Merges shared options (root_dir, output_dir, sensor flags, etc.) with
    subcommand-specific options. Empty sensor-ID tuples are replaced with None
    (meaning "all") for dataclass compatibility.

    Args:
        ctx: Click context whose obj holds the group options.
        **subcommand_kwargs: Overrides/additions from the subcommand (e.g. store_type).

    Returns:
        Dict suitable for constructing a converter Config (e.g. ExampleConverterConfig).
    """
    obj: SimpleNamespace = ctx.obj
    config_dict = vars(obj).copy()
    config_dict.update(subcommand_kwargs)
    for key in ("camera_ids", "lidar_ids", "radar_ids"):
        if key in config_dict and config_dict[key] is not None and len(config_dict[key]) == 0:
            config_dict[key] = None
    return config_dict


@click.group()
@click.option(
    "--root-dir",
    type=click.Path(exists=False, resolve_path=True),
    default=".",
    help="Path to the raw data sequences (required for file-based converters)",
)
@click.option(
    "--output-dir",
    type=click.Path(resolve_path=True),
    required=True,
    help="Path where the converted NCore V4 data will be saved",
)
@click.option("--verbose", is_flag=True, default=False, help="Enable debug logging")
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Start debugpy remote debugging session",
)
@click.option(
    "--debug-port",
    type=int,
    default=5678,
    help="Port for debugpy to listen on",
)
@click.option(
    "--seek-sec",
    default=None,
    type=click.FloatRange(min=0.0),
    help="Time to skip from the first pose timestamp (in seconds)",
)
@click.option(
    "--duration-sec",
    default=None,
    type=click.FloatRange(min=0.0, min_open=True),
    help="Restrict total duration of the conversion (in seconds); must be > 0",
)
@click.option(
    "--no-cameras",
    is_flag=True,
    default=False,
    help="Disable exporting any camera sensor",
)
@click.option(
    "--camera-id",
    "camera_ids",
    multiple=True,
    type=str,
    default=None,
    help="Cameras to export (repeatable; default: all)",
)
@click.option(
    "--no-lidars",
    is_flag=True,
    default=False,
    help="Disable exporting any lidar sensor",
)
@click.option(
    "--lidar-id",
    "lidar_ids",
    multiple=True,
    type=str,
    default=None,
    help="Lidars to export (repeatable; default: all)",
)
@click.option(
    "--no-radars",
    is_flag=True,
    default=False,
    help="Disable exporting any radar sensor",
)
@click.option(
    "--radar-id",
    "radar_ids",
    multiple=True,
    type=str,
    default=None,
    help="Radars to export (repeatable; default: all)",
)
@click.pass_context
def cli(
    ctx: click.Context,
    *_: object,
    **kwargs: object,
) -> None:
    """Convert raw sensor data to NCore V4 format.

    Use subcommands (e.g. example) to select the source format. Shared options
    include --root-dir, --output-dir, time restriction (--seek-sec, --duration-sec),
    sensor selection (--no-cameras, --camera-id, etc.), and --verbose / --debug /
    --debug-port.

    Args:
        ctx: Click context; receives group options in ctx.obj.
        *_: Unused positional args from Click.
        **kwargs: Group options (root_dir, output_dir, verbose, debug, etc.).
    """
    ctx.obj = SimpleNamespace(**kwargs)
    logging.basicConfig(
        level=logging.DEBUG if ctx.obj.verbose else logging.INFO,
        format="<%(asctime)s|%(levelname)s|%(filename)s:%(lineno)d|%(name)s> %(message)s",
    )
    if getattr(ctx.obj, "debug", False):
        try:
            import debugpy

            debugpy.listen(port=getattr(ctx.obj, "debug_port", 5678))
            logging.getLogger(__name__).info("debugpy listening for client")
        except ImportError:
            logging.getLogger(__name__).warning("debugpy not installed; --debug has no effect (pip install debugpy)")


@cli.command("example")
@click.option(
    "--store-type",
    type=click.Choice(["itar", "directory"], case_sensitive=False),
    default="itar",
    show_default=True,
    help="Output store type (itar or directory)",
)
@click.option(
    "--profile",
    "component_group_profile",
    type=click.Choice(["default", "separate-sensors", "separate-all"], case_sensitive=False),
    default="separate-sensors",
    show_default=True,
    help="Component group layout profile",
)
@click.option(
    "--sequence-meta/--no-sequence-meta",
    "store_sequence_meta",
    default=True,
    help="Write sequence meta JSON for NuRec / ncore_vis",
)
@click.pass_context
def example_cmd(
    ctx: click.Context,
    store_type: str,
    component_group_profile: str,
    store_sequence_meta: bool,
) -> None:
    """Run the example (skeleton) converter.

    Writes minimal valid NCore V4 output (one sequence, rig poses, optional
    sequence meta) to the output dir. Uses shared --root-dir and --output-dir.

    Args:
        ctx: Click context with shared group options.
        store_type: Output store type ("itar" or "directory").
        component_group_profile: Component group layout ("default", "separate-sensors", "separate-all").
        store_sequence_meta: Whether to write {sequence_id}.json after conversion.
    """
    config = ExampleConverterConfig(
        **_ctx_to_config(
            ctx,
            store_type=store_type,
            component_group_profile=component_group_profile,
            store_sequence_meta=store_sequence_meta,
        )
    )
    ExampleConverter.convert(config)


if __name__ == "__main__":
    cli()
