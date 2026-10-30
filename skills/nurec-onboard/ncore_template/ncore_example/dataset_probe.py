#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-conversion dataset probe script skeleton.

Every converter built from this template should fill in the marked
sections of this file with dataset-specific exploration logic. The
script runs BEFORE the converter is written and produces the evidence
base for converter parameter decisions. After the agent's package
rename (``ncore_example/`` → ``ncore_<dataset>/``), this file lives at
``ncore_<dataset>/dataset_probe.py``; no file rename is required.

Running this before writing converter code is strongly recommended
(this is a template best practice, not an NCore library requirement).
Reasoning from ``data.description`` alone instead of measuring leads to
silent failures at every critical parameter (extrinsic translation,
cross-sensor time base, cuboid centroid convention, etc.); a probe
script that dumps per-sensor Z-offsets, cross-sensor timestamp deltas,
and cuboid ground clearance catches them before any conversion code is
written.

Design rules
------------

1. **Execution mode follows ``dependency_strategy``** (see
   ``nurec-onboard/steps/build-ncore-converter/SKILL.md`` for the per-strategy
   probe invocations). Keep each invocation path working:

   * **Bazel-resident**: the template ships a ``probe`` ``py_binary``
     target -- do not add a second one.
   * **External SDK env**: the shebang + ``argparse`` interface below
     makes the file runnable as a plain script (preserve both);
     depend only on the dataset SDK (or raw file parsers) plus numpy
     so the probe imports from a minimal external env without
     Bazel-graph packages.

2. **Writes a ``<dataset>_meta.json`` summary at the workdir root.**
   The JSON schema is open but every key written there can be cited by
   converter parameter justifications via its path (e.g. ``"see
   <dataset>_meta.json: lidar.z_offset_m=<value>"``).

3. **Covers every metric that drives a justification.** At minimum:

   * Every sensor calibration: intrinsics, extrinsics, model class,
     rolling-shutter direction if applicable. Print the full 4x4
     matrices -- do not summarize.
   * Timestamp semantics for every sensor: sample one representative
     frame and print the per-frame timestamp alongside the per-ray /
     per-point / per-pixel timestamp min-max. Compute and print
     cross-sensor midpoint deltas (camera exposure center vs LiDAR
     sweep center) aggregated across the sequence.
   * Per-sensor Z-offset histogram (for detecting mount-height vs
     ground-plane issues).
   * Sweep duration statistics (min/max/mean across the sequence) for
     every spinning LiDAR.
   * Cuboid ground-clearance histogram: ``mean(centroid.z - dim_z/2)``
     over vehicle-class cuboids -- distinguishes geometric-center from
     bottom-center conventions without relying on docs.
   * Cuboid dimension-axis ordering sanity check: for a sample of
     cuboids, compare ``dim_x/dim_y`` ratio against expected aspect
     ratios for the dataset's annotated object classes (e.g. ~2.3 for
     passenger cars in automotive datasets) to detect x/y swaps.
   * One example of every label type dumped verbatim (raw field names
     preserved).

4. **Run it twice.** Once before writing the converter
   and once after the first conversion passes; comparing the two JSONs
   catches silent raw-data drift.

Replace every ``FILL IN`` marker below with dataset-specific logic and
delete this module-level comment block before checking the script in.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any

import numpy as np  # noqa: F401 -- used by explore_* implementations


def explore_calibrations(root: Path) -> dict[str, Any]:
    """Enumerate every sensor calibration.

    FILL IN: load calibration files, return a dict of the form:
        {
          "cameras": {"<id>": {"intrinsics": ..., "T_camera_rig": ..., "model": ...}, ...},
          "lidars":  {"<id>": {"T_lidar_rig": ..., "rows": ..., "cols": ..., "spinning_direction": ...}, ...},
          "radars":  {"<id>": {...}, ...},
        }
    Print the full 4x4 extrinsic matrices to stdout (do not summarize).
    """
    raise NotImplementedError("FILL IN: explore_calibrations")


def explore_timestamps(root: Path) -> dict[str, Any]:
    """Measure timestamp semantics for every sensor.

    FILL IN: for one representative sequence, return a dict containing:
      * per-sensor per-frame timestamp (start / end / unit)
      * per-point / per-ray / per-pixel timestamp min-max vs per-frame
        (tells you whether frame ts is sweep_start / center / end)
      * cross-sensor midpoint deltas:
        camera_exposure_center_us - lidar_sweep_center_us, aggregated
        median across the sequence (this is the same statistic that
        ``check_lidar_camera_sweep_alignment`` in
        ``ncore_validation/validate_cross_sensor.py`` uses)
      * LiDAR sweep duration statistics (min/max/mean)
    """
    raise NotImplementedError("FILL IN: explore_timestamps")


def explore_cuboids(root: Path) -> dict[str, Any]:
    """Dump one example of every label type + key geometric statistics.

    FILL IN: return a dict with:
      * ground_clearance_histogram: distribution of
        (centroid.z - dim_z/2) for vehicle-class cuboids. Used to
        distinguish geometric-center (~0 m) from bottom-center
        (~-dim_z/2) conventions.
      * dim_xy_ratio_histogram: distribution of dim_x/dim_y for
        cuboids; compare against expected aspect ratios for the
        dataset's object classes (e.g. ~2.3 for cars). A bimodal or
        mean < 1 indicates the dim.x/dim.y axes are swapped.
      * one_example_per_label_type: raw dumps (preserve field names).
      * centroid_convention_probe: summary of the measurement that
        supports the chosen `cuboid_centroid_convention` justification.
    If the dataset has no cuboids, return {"has_cuboids": false} and
    stop; record has_annotations: false in inferred_parameters.
    """
    raise NotImplementedError("FILL IN: explore_cuboids")


def explore_lidar_points(root: Path) -> dict[str, Any]:
    """Enumerate LiDAR point fields and probe mount height.

    FILL IN: for each spinning LiDAR, return a dict with:
      * raw field list (every column in the raw point record, NOT just
        XYZ -- flag any field whose shape matches the ray grid
        [H, W, ...], [H], or [W] AND whose values look like poses or
        per-ray timestamps; these are per-ray motion metadata and
        indicate the converter must use the intrinsic beam model).
      * motion_compensation_state: sensor_frame vs world_frame vs
        ego_compensated, determined by checking whether point extents
        are consistent across frames (world-frame would translate).
      * z_offset_probe: median Z of ground-plane returns (filter by
        radius and negative Z) in the sensor frame. This is the
        translation component of `T_lidar_rig` when the dataset does
        not publish a translation; missing it produces a cuboid Z
        offset equal to the sensor mount height.
    """
    raise NotImplementedError("FILL IN: explore_lidar_points")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Pre-conversion dataset probe -- fill in dataset-specific logic for your raw data layout.")
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Path to the raw dataset root (e.g. /input-dataset).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=("Output JSON summary path. Convention: ``<dataset>_meta.json`` at the workdir root."),
    )
    args = parser.parse_args()

    if not args.root_dir.is_dir():
        raise SystemExit(f"--root-dir does not exist: {args.root_dir}")

    summary: dict[str, Any] = {
        "root_dir": str(args.root_dir),
        "calibrations": explore_calibrations(args.root_dir),
        "timestamps": explore_timestamps(args.root_dir),
        "cuboids": explore_cuboids(args.root_dir),
        "lidar_points": explore_lidar_points(args.root_dir),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"Wrote reconnaissance summary to {args.output}")


if __name__ == "__main__":
    main()
