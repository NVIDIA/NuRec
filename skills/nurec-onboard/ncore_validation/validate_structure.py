# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V4 Structure tier checks."""

from __future__ import annotations

import logging

import numpy as np

from ncore_validation.utils import NcoreV4Context
from ncore_validation.validate_helpers import _check, _group, _section, _warn


log = logging.getLogger("validate")


def print_verbose_summary(ctx: NcoreV4Context) -> None:
    """Log sequence metadata and which component instances were opened."""
    log.info("")
    log.info(f"  sequence_id:                    {ctx.reader.sequence_id}")
    log.info(f"  version:                        {ctx.meta.version}")
    log.info(f"  sequence_timestamp_interval_us: [{ctx.interval.start}, {ctx.interval.stop})")
    log.info("")
    _component_types: list[tuple[str, dict]] = [
        ("Poses", ctx.poses_readers),
        ("Cameras", ctx.camera_readers),
        ("Intrinsics", ctx.intrinsics_readers),
        ("Masks", ctx.masks_readers),
        ("LiDAR", ctx.lidar_readers),
        ("Radar", ctx.radar_readers),
        ("Cuboids", ctx.cuboid_readers),
    ]
    for type_name, readers_dict in _component_types:
        instances = sorted(readers_dict.keys()) if readers_dict else []
        if instances:
            log.info(f"  {type_name}: {', '.join(instances)}")
        else:
            log.info(f"  {type_name}: (none)")


def check_sequence_identity(ctx: NcoreV4Context) -> None:
    """Validate sequence id, V4 version string, and timestamp interval bounds."""
    _group("Sequence")
    _check(
        "sequence_id non-empty",
        bool((ctx.reader.sequence_id or "").strip()),
        ctx.errors,
        "sequence_id is empty",
    )
    _check(
        "version is v4",
        ctx.meta.version == "v4",
        ctx.errors,
        f"version is '{ctx.meta.version}', expected 'v4'",
    )
    _check(
        "sequence_timestamp_interval_us (start < stop)",
        ctx.interval.start < ctx.interval.stop,
        ctx.errors,
        "sequence_timestamp_interval_us has start >= stop",
    )


def check_poses_rig_world_in_interval(ctx: NcoreV4Context) -> None:
    """Require every rig->world dynamic trajectory to lie in the interval.

    Downstream checks use the first pose reader that has rig->world, so any
    invalid trajectory fails here even when another reader has a valid one.
    """
    _group("Poses")
    has_valid_poses = False
    pose_issues: list[str] = []
    if ctx.poses_readers:
        for component_instance_name, pose_reader in ctx.poses_readers.items():
            try:
                _poses, timestamps_us = pose_reader.get_dynamic_pose("rig", "world")
            except KeyError:
                continue
            if timestamps_us.size == 0:
                pose_issues.append(f"'{component_instance_name}': no timestamps")
            elif not ((timestamps_us >= ctx.interval.start) & (timestamps_us < ctx.interval.stop)).all():
                pose_issues.append(f"'{component_instance_name}': timestamps outside interval")
            else:
                has_valid_poses = True
    error_detail = "; ".join(pose_issues) if pose_issues else "no rig→world dynamic pose found"
    _check(
        "rig→world dynamic in interval",
        has_valid_poses and not pose_issues,
        ctx.errors,
        f"Poses rig→world: {error_detail}",
    )


def check_pose_timestamps_strictly_increasing(ctx: NcoreV4Context) -> None:
    """Fail when dynamic pose timestamps are not strictly increasing.

    NCore's writer asserts ``np.all(np.diff(timestamps_us) > 0)`` for every
    dynamic pose trajectory.  Non-monotonic timestamps would break NuRec's
    pose interpolation.
    """
    if not ctx.poses_readers:
        return
    bad: list[str] = []
    for component_instance_name, pose_reader in ctx.poses_readers.items():
        try:
            dynamic_poses = list(pose_reader.get_dynamic_poses())
        except (KeyError, ValueError, OSError) as e:
            log.debug(
                "check_pose_timestamps: cannot read dynamic poses from %s: %s",
                component_instance_name,
                e,
            )
            continue
        for (source, target), (_poses, ts) in dynamic_poses:
            if ts.size < 2:
                continue
            diffs = np.diff(ts.astype(np.int64))
            if not np.all(diffs > 0):
                n_bad = int(np.sum(diffs <= 0))
                bad.append(f"{component_instance_name}:{source}→{target} ({n_bad} non-increasing gaps)")
    _check(
        "dynamic pose timestamps strictly increasing",
        not bad,
        ctx.errors,
        f"dynamic pose timestamps not strictly increasing: {'; '.join(bad)}",
    )


def warn_pose_coverage(ctx: NcoreV4Context) -> None:
    """Warn when dynamic pose timestamps do not span the full sequence interval.

    NCore's writer defaults to ``require_sequence_time_coverage=True``, which
    asserts ``timestamps[0] == interval.start`` and
    ``timestamps[-1] == interval.stop - 1``.  Since this flag *can* be
    disabled, a mismatch is a warning rather than a hard error.
    """
    if not ctx.poses_readers:
        return
    for component_instance_name, pose_reader in ctx.poses_readers.items():
        try:
            _poses, ts = pose_reader.get_dynamic_pose("rig", "world")
        except KeyError:
            continue
        if ts.size == 0:
            continue
        first, last = int(ts[0]), int(ts[-1])
        expected_first = ctx.interval.start
        expected_last = ctx.interval.stop - 1
        if first != expected_first or last != expected_last:
            _warn(
                f"Pose timestamps do not cover the full sequence interval "
                f"for {component_instance_name}:rig→world. "
                f"First pose ts={first} (expected {expected_first}), "
                f"last pose ts={last} (expected {expected_last}). "
                f"NCore's writer requires this by default "
                f"(require_sequence_time_coverage=True)."
            )


def run_v4_structure_checks(ctx: NcoreV4Context) -> None:
    """Run the **V4 Structure** tier (sequence metadata and poses sanity)."""
    _section("V4 Structure")
    check_sequence_identity(ctx)
    check_poses_rig_world_in_interval(ctx)
    check_pose_timestamps_strictly_increasing(ctx)
    warn_pose_coverage(ctx)
