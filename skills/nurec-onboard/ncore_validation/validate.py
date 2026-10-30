# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate NCore V4 sequence data.

Opens the sequence with SequenceComponentGroupsReader and runs three
tiers of checks: V4 Structure, NuRec Readiness, and Data Quality.
See README.md "Validation tiers" for the full list.

Exits 0 on success, non-zero on failure. A typed
``validate_result.yaml`` sidecar is always written (default path
``./validate_result.yaml``), including on early crashes and on
unexpected exceptions inside any check, so downstream tooling can read
structured pass/fail data without grepping stdout.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback

from importlib import metadata
from pathlib import Path

import click
import yaml

from validators._validation_base import VALIDATE_RESULT_PROGRAM, VALIDATE_RESULT_SCHEMA_VERSION

from ncore_validation.utils import NcoreV4Context, open_context
from ncore_validation.validate_cameras import (
    check_camera_frame_ts_order,
    warn_all_cameras_global_shutter,
    warn_camera_fisheye_max_angle_deviation,
    warn_camera_frame_count_mismatch_if_multi_cam,
    warn_camera_optical_axis_azimuth,
    warn_camera_optical_axis_tilt,
    warn_camera_timestamps_all_identical,
)
from ncore_validation.validate_cross_sensor import (
    check_camera_lidar_time_overlap,
    check_cameras_inside_lidar_bbox,
    check_lidar_camera_sweep_alignment,
    warn_camera_follows_ego_if_moving,
    warn_lidar_centroid_drift,
    warn_sequence_envelope_coverage,
    warn_sparse_pose_trajectory,
    warn_sweep_timestamps_centered,
)
from ncore_validation.validate_cuboids import (
    check_cuboid_duplicate_observation_timestamps,
    check_cuboid_lidar_overlap,
    check_cuboid_obb_inlier_ratio,
    check_cuboid_pose_gap,
    check_cuboid_rig_proximity,
    check_cuboid_timestamps_in_interval,
    check_cuboid_ts_overlaps_sensor_range,
    warn_cuboid_cross_source_timestamp_collisions,
    warn_cuboid_ground_clearance,
)
from ncore_validation.validate_helpers import (
    FAIL_ICON,
    PASS_ICON,
    FailedCheck,
    ValidateResult,
    _group,
    _section,
    _warn,
    dump_validate_result,
    start_collection,
    stop_collection,
    utc_now_iso,
)
from ncore_validation.validate_lidar import (
    check_lidar_data_quality_group,
    warn_low_intensity_dynamic_range,
)
from ncore_validation.validate_nurec import run_nurec_readiness_checks
from ncore_validation.validate_poses import (
    check_pose_dtypes,
    check_pose_se3_valid,
    check_pose_translation_magnitude,
    check_rig_forward_alignment,
    check_static_sensor_rig_translations,
    warn_identity_sensor_translations,
    warn_pose_angular_rate,
    warn_rig_forward_not_x,
)
from ncore_validation.validate_structure import print_verbose_summary, run_v4_structure_checks


def _ncore_version() -> str:
    """Return the ``nvidia-ncore`` package version, or a fallback string.

    Returns:
        The installed ``nvidia-ncore`` version string, or
        ``"unknown"`` when :func:`importlib.metadata.version` cannot
        find the distribution (e.g. running from a development tree
        without the package installed). Logs a warning on the fallback
        path so a misconfigured workspace surfaces in the operator log
        rather than landing silently as the literal ``"unknown"`` in
        every sidecar.
    """
    try:
        return metadata.version("nvidia-ncore")
    except metadata.PackageNotFoundError:
        log.warning(
            "nvidia-ncore distribution metadata not found; validate_result.yaml will record program_version='unknown'"
        )
        return "unknown"


log = logging.getLogger("validate")


def _exit_code_from_interrupt(exc: SystemExit | KeyboardInterrupt) -> int:
    """Return the process exit code to record for an interrupting exception.

    Args:
        exc: Exception that interrupted validation.

    Returns:
        Non-zero process exit code to use in the result sidecar.
    """
    if isinstance(exc, KeyboardInterrupt):
        return 130
    if isinstance(exc, SystemExit):
        code = exc.code
        if isinstance(code, int) and code != 0:
            return code
    return 1


# ---------------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------------


def _lidar_frames_subsampled(ctx: NcoreV4Context, sample_frames: int | None) -> bool:
    """Report whether ``--sample-frames`` actually thinned any LiDAR's frames.

    Mirrors the cutoffs in ``_sample_frame_indices``, which treats a value that
    is absent, non-positive, or at least the sensor's frame count as a full
    sweep.  Keying off the flag alone would skip the cuboid-LiDAR checks on a
    run that asked for more frames than the shard has and so kept them all.
    """
    if sample_frames is None or sample_frames <= 0:
        return False
    return any(len(reader.frames_timestamps_us) > sample_frames for reader in ctx.lidar_readers.values())


def run_data_quality_checks(
    ctx: NcoreV4Context,
    sample_frames: int | None = None,
) -> None:
    """Run the **Data Quality** tier (poses, lidar, cameras, cross-sensor checks).

    Args:
        ctx: NCore V4 context.
        sample_frames: Forwarded to :func:`check_lidar_data_quality_group`;
            limits per-frame LiDAR checks to ``sample_frames`` evenly-spaced
            frames per sensor.  Cheap structural checks always run on the
            full set.  ``None`` runs every frame.
    """
    _section("Data Quality")
    lidar_frames_subsampled = _lidar_frames_subsampled(ctx, sample_frames)
    if lidar_frames_subsampled:
        _warn(
            f"--sample-frames={sample_frames} is active; LiDAR per-frame checks "
            f"will see only a subsample.  Rerun without --sample-frames before "
            f"treating the sequence as converged."
        )

    # --- Poses (single-sensor) ---
    check_static_sensor_rig_translations(ctx)
    warn_identity_sensor_translations(ctx)
    check_pose_dtypes(ctx)
    check_pose_se3_valid(ctx)
    check_pose_translation_magnitude(ctx)
    check_rig_forward_alignment(ctx)
    warn_pose_angular_rate(ctx)
    warn_rig_forward_not_x(ctx)

    # --- LiDAR (single-sensor) ---
    lidar_pts_rig = check_lidar_data_quality_group(ctx, sample_frames=sample_frames)
    warn_low_intensity_dynamic_range(ctx)
    warn_lidar_centroid_drift(ctx)

    # --- Cameras (single-sensor) ---
    check_camera_frame_ts_order(ctx)
    warn_camera_timestamps_all_identical(ctx)
    warn_all_cameras_global_shutter(ctx)
    warn_camera_frame_count_mismatch_if_multi_cam(ctx)
    warn_camera_fisheye_max_angle_deviation(ctx)
    warn_camera_optical_axis_tilt(ctx)
    warn_camera_optical_axis_azimuth(ctx)

    # --- Cross-Sensor (checks first, then warnings) ---
    _group("Cross-Sensor")
    check_cameras_inside_lidar_bbox(ctx, lidar_pts_rig=lidar_pts_rig)
    check_camera_lidar_time_overlap(ctx)
    check_lidar_camera_sweep_alignment(ctx)
    warn_sequence_envelope_coverage(ctx)
    check_cuboid_ts_overlaps_sensor_range(ctx)
    check_cuboid_timestamps_in_interval(ctx)
    check_cuboid_duplicate_observation_timestamps(ctx)
    warn_cuboid_cross_source_timestamp_collisions(ctx)
    check_cuboid_pose_gap(ctx)
    check_cuboid_rig_proximity(ctx)
    if lidar_frames_subsampled:
        _warn(
            "cuboid-LiDAR overlap, OBB inlier, and ground-clearance checks skipped "
            "because --sample-frames does not provide continuous LiDAR coverage"
        )
    else:
        check_cuboid_lidar_overlap(ctx, lidar_pts_rig=lidar_pts_rig)
        check_cuboid_obb_inlier_ratio(ctx, lidar_pts_rig=lidar_pts_rig)
        warn_cuboid_ground_clearance(ctx, lidar_pts_rig=lidar_pts_rig)
    warn_sparse_pose_trajectory(ctx)
    warn_sweep_timestamps_centered(ctx)
    warn_camera_follows_ego_if_moving(ctx)


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
@click.option("--verbose", is_flag=True, default=False, help="Print summary")
@click.option(
    "--sample-frames",
    type=int,
    default=None,
    help=(
        "If set, evaluate only N evenly-spaced frames per LiDAR in the "
        "iteratively-expensive checks (check_lidar_frame_data and "
        "check_lidar_model_accuracy). Intended for fast iteration cycles "
        "during converter development; rerun without --sample-frames "
        "before treating the sequence as converged. Cheap structural "
        "checks always run on the full set."
    ),
)
@click.option(
    "--output-result",
    "output_result",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("validate_result.yaml"),
    show_default=True,
    help=(
        "Path for the typed validate_result.yaml sidecar. Always written "
        "(including on early crashes) so downstream tooling can read "
        "structured pass/fail/warning data without grepping stdout. "
        "Relative paths resolve against $PWD; under `bazel run` that is "
        "the user's invocation directory, not the runfiles tree."
    ),
)
def main(
    component_groups: tuple[Path, ...],
    open_consolidated: bool,
    verbose: bool,
    sample_frames: int | None,
    output_result: Path,
) -> None:
    """Validate NCore V4 sequence: open stores and run all checks.

    Exits 0 on success, non-zero on failure. Always writes a
    ``validate_result.yaml`` sidecar at ``output_result`` regardless of
    pass/fail outcome.

    Args:
        component_groups: Paths to ``{sequence_id}.json`` meta files or store roots.
        open_consolidated: Open a single consolidated zarr store instead of per-component stores.
        verbose: Log sequence metadata and opened component instances before checks.
        sample_frames: When set, limit per-frame LiDAR checks to N
            evenly-spaced frames per sensor (fast-iteration mode).
        output_result: Destination path for the ``validate_result.yaml``
            sidecar.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    started_at = utc_now_iso()
    started_perf = time.monotonic()
    exit_code = 0
    collector = start_collection()
    ctx: NcoreV4Context | None = None
    try:
        try:
            ctx = open_context(component_groups, open_consolidated=open_consolidated)
        except Exception as e:
            log.error(f"{FAIL_ICON} Validation failed: could not open sequence")
            log.error(f"  {type(e).__name__}: {e}")
            collector.failed_checks.append(
                FailedCheck(
                    tier="V4 Structure",
                    group=None,
                    name="open sequence",
                    message=f"could not open sequence: {type(e).__name__}: {e}",
                )
            )
            exit_code = 1

        # Record unexpected check exceptions in the sidecar as validation failures.
        if exit_code == 0 and ctx is not None:
            try:
                if verbose:
                    print_verbose_summary(ctx)

                run_v4_structure_checks(ctx)
                run_nurec_readiness_checks(ctx)
                run_data_quality_checks(ctx, sample_frames=sample_frames)
            except Exception as e:
                log.error(f"\n{FAIL_ICON} Validation failed: unexpected exception during checks")
                log.error(f"  {type(e).__name__}: {e}")
                log.error(traceback.format_exc())
                collector.failed_checks.append(
                    FailedCheck(
                        tier="Internal",
                        group=None,
                        name="unexpected exception",
                        message=f"unexpected exception during checks: {type(e).__name__}: {e}",
                    )
                )
                exit_code = 1

        if exit_code == 0 and ctx is not None and ctx.errors:
            log.error(f"\n{FAIL_ICON} Validation failed:")
            for msg in ctx.errors:
                log.error(f"  - {msg}")
            exit_code = 1
    except (SystemExit, KeyboardInterrupt) as e:
        # Record interrupts in the sidecar and exit non-zero.
        exit_code = _exit_code_from_interrupt(e)
        collector.failed_checks.append(
            FailedCheck(
                tier="Internal",
                group=None,
                name="interrupted",
                message=f"validation interrupted: {type(e).__name__}: {e}",
            )
        )
    finally:
        ended_at = utc_now_iso()
        duration_sec = time.monotonic() - started_perf
        result = ValidateResult(
            schema_version=VALIDATE_RESULT_SCHEMA_VERSION,
            program=VALIDATE_RESULT_PROGRAM,
            program_version=_ncore_version(),
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=duration_sec,
            exit_code=exit_code,
            passed=(exit_code == 0),
            component_groups=[str(p) for p in component_groups],
            sample_frames=sample_frames,
            failed_checks=list(collector.failed_checks),
            warnings=list(collector.warnings),
        )
        try:
            dump_validate_result(output_result, result)
        except (OSError, yaml.YAMLError) as e:
            # OSError: filesystem failures (permission, missing parent, EBADF, ...).
            # yaml.YAMLError: defensive net for unrepresentable field types if
            # ValidateResult ever grows a non-YAML-safe attribute. Both are
            # caught here so a serialization failure cannot bubble out of
            # `finally` and replace the `sys.exit(exit_code)` below it.
            log.error(f"{FAIL_ICON} Failed to write {output_result}: {e}")
            if exit_code == 0:
                exit_code = 1
            # Remove any stale sidecar from a prior run so a downstream reader
            # cannot mistake last run's `passed: true` for this run's outcome.
            # ``os.replace`` is atomic, so on failure the prior file is still
            # in place untouched. A second failure here is logged but must not
            # mask the dump_validate_result failure above or change exit_code.
            try:
                Path(output_result).unlink()
            except FileNotFoundError:
                pass
            except OSError as rm_err:
                log.error(f"  (also failed to remove stale {output_result}: {rm_err})")
        else:
            # Only print the success banner when the sidecar write has also
            # succeeded; otherwise stdout would say "Validation passed" while
            # exit_code is being flipped to 1 by the dump-failure branch above.
            if exit_code == 0:
                log.info(f"\n{PASS_ICON} Validation passed.")
        stop_collection()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
