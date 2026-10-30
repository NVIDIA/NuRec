# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Carry a reference scene's map and geo-registration into a scene package.

A reconstruction from a raw PAI clip has no map and no global pose: its USDZ carries no
``map.xodr`` and its ``T_world_base`` is the identity. A reference USDZ of the same clip (any
scene package that has them) supplies the OpenDRIVE map, the pose record and the ClipGT layers. The
ECEF ``T_world_base`` that consumers use to place the map in the scene's frame is solved from the
pose record.

The record is the map-localised ego trajectory. The scene's own trajectory is paired with it in time
and ``T_world_base`` is the rigid transform that fits the scene's poses to the record's best. The
residual of that fit gates the result. The scene's own timestamps are kept: the renderer and the tracks index by
them.
"""

from __future__ import annotations

import json
import os
import zipfile

from pathlib import Path
from typing import Any, BinaryIO

import click
import numpy as np


COPIED_MEMBERS = ("map.xodr", "pose_record.json")
COPIED_PREFIXES = ("clipgt/",)
RIG_TRAJECTORIES = "rig_trajectories.json"
DATASOURCE_SUMMARY = "datasource_summary.json"
# Each pose is fitted as its origin and the points this far along its x and y axes: positions alone
# leave the roll about a straight road free.
AXIS_POINT_M = 10.0
# A pose-record alignment is trusted only when the runner-up offset fits clearly worse: at least this
# many times the best squared step error, and at least this much more (a 3 us RMS disagreement).
# Evenly spaced timestamps fit every offset equally, and the wrong one misplaces the map by whole
# frames of travel, which a rigid fit absorbs on a straight road.
MIN_ALIGNMENT_CONTRAST = 10.0
MIN_ALIGNMENT_GAP_US2 = 10.0
# Residual of the fit beyond which the scene's trajectory is not the recorded drive's. The rotation
# residual includes a constant attitude offset between the scene's rig frame and the record's.
MAX_MEDIAN_POSITION_M = 1.0
MAX_P95_POSITION_M = 3.0
MAX_MEDIAN_ROTATION_DEG = 3.0


def _open_reference(reference: str) -> BinaryIO:
    """A seekable handle on a local path or an ``hf://`` path; only the members read are fetched."""
    if reference.startswith("hf://"):
        from huggingface_hub import HfFileSystem  # reads HF_TOKEN from the environment

        return HfFileSystem().open(reference, "rb", block_size=1 << 16)
    return open(reference, "rb")


def _ego(rig: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    trajectory = rig["rig_trajectories"][0]
    return np.asarray(trajectory["T_rig_worlds"], dtype=np.float64), np.asarray(
        trajectory["T_rig_world_timestamps_us"], dtype=np.int64
    )


def _pose_differences(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    position = np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1)
    relative = np.einsum("nji,njk->nik", b[:, :3, :3], a[:, :3, :3])
    rotation = np.degrees(np.arccos(((np.trace(relative, axis1=1, axis2=2) - 1) / 2).clip(-1, 1)))
    return {
        "median_position_m": float(np.median(position)),
        "p95_position_m": float(np.percentile(position, 95)),
        "max_position_m": float(position.max()),
        "median_rotation_deg": float(np.median(rotation)),
        "max_rotation_deg": float(rotation.max()),
    }


def _ecef_pose(record: dict[str, Any]) -> np.ndarray:
    """ECEF pose of one pose-record sample: WGS84 position, axis-angle orientation in local ENU."""
    lla, axis_angle = record["alignment_world_pose"]["lat_lng_alt"], record["alignment_world_pose"]["axis_angle"]
    lat, lon = np.radians(lla["latitude"]), np.radians(lla["longitude"])
    a, e2 = 6378137.0, 6.69437999014e-3
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    h = lla["altitude"]
    position = [
        (n + h) * np.cos(lat) * np.cos(lon),
        (n + h) * np.cos(lat) * np.sin(lon),
        (n * (1 - e2) + h) * np.sin(lat),
    ]
    ecef_from_enu = np.array(
        [
            [-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
            [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
            [0.0, np.cos(lat), np.sin(lat)],
        ]
    )
    # The record's axis scaled by its angle is taken as the rotation vector, as the producer does.
    rotvec = np.radians(axis_angle["angle"]) * np.array([axis_angle["qx"], axis_angle["qy"], axis_angle["qz"]])
    theta = np.linalg.norm(rotvec)
    k = np.array([[0, -rotvec[2], rotvec[1]], [rotvec[2], 0, -rotvec[0]], [-rotvec[1], rotvec[0], 0]]) / max(
        theta, 1e-12
    )
    rotation = np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = ecef_from_enu @ rotation, position
    return pose


def _axis_points(poses: np.ndarray) -> np.ndarray:
    origins = poses[:, :3, 3]
    return np.concatenate([origins, origins + AXIS_POINT_M * poses[:, :3, 0], origins + AXIS_POINT_M * poses[:, :3, 1]])


def _rigid_fit(poses: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """The rigid ``T`` (4x4) that best maps ``poses`` onto ``targets``, both ``(N, 4, 4)`` rig-to-frame poses.

    Each pose contributes its origin and the points ``AXIS_POINT_M`` along its x and y axes, so the
    fit weighs orientation as well as position; ``T`` minimises the summed squared distances between
    the transformed and the target points (Kabsch).
    """
    a, b = _axis_points(poses), _axis_points(targets)
    a_mean, b_mean = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - a_mean).T @ (b - b_mean))
    rotation = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, b_mean - rotation @ a_mean
    return transform


def world_base_from_pose_record(scene_rig: dict[str, Any], pose_record: dict[str, Any]) -> tuple[list, dict[str, Any]]:
    """``T_world_base`` fitting the scene's trajectory to the pose record, and the residual of the fit.

    The two time bases are matched on the spacing of consecutive timestamps: the rig's interior
    intervals slide along the record's, and the offset with the least squared difference pairs rig
    pose ``i`` with record ``i + offset``. ``T_world_base`` is the rigid transform that best maps the
    paired interior poses onto the record's. Raises ``ValueError`` when the record cannot cover the
    scene or no offset clearly wins.
    """
    poses, timestamps = _ego(scene_rig)
    records = pose_record.get("record", [])
    record_timestamps = np.array([int(r["timestamp_microseconds"]) for r in records], dtype=np.int64)
    rig_steps, record_steps = np.diff(timestamps)[1:-1], np.diff(record_timestamps)[1:-1]
    if len(rig_steps) < 1 or len(record_steps) < len(rig_steps):
        raise ValueError(f"{len(records)} pose records cannot cover {len(timestamps)} rig poses")
    errors = [
        np.mean((rig_steps - record_steps[o : o + len(rig_steps)]) ** 2)
        for o in range(len(record_steps) - len(rig_steps) + 1)
    ]
    order = np.argsort(errors)
    offset = int(order[0])
    best, runner_up = float(errors[order[0]]), float(errors[order[1]]) if len(errors) > 1 else float("inf")
    if runner_up < MIN_ALIGNMENT_CONTRAST * best or runner_up - best < MIN_ALIGNMENT_GAP_US2:
        raise ValueError(
            f"timestamps do not pin the pose-record alignment (best offset {best:.3g} us^2, runner-up {runner_up:.3g} us^2)"
        )
    rig_indices = np.arange(1, len(rig_steps) + 2)
    ecef = np.stack([_ecef_pose(records[i + offset]) for i in rig_indices])
    world_base = _rigid_fit(poses[rig_indices], ecef)
    report = {
        "time_offset_us": int(record_timestamps[rig_indices[0] + offset] - timestamps[rig_indices[0]]),
        "matched_fraction": len(rig_indices) / len(timestamps),
        "alignment_error_us2": best,
        "runner_up_error_us2": runner_up,
    }
    report.update(_pose_differences(world_base @ poses[rig_indices], ecef))
    return world_base.tolist(), report


def _reference_members(archive: zipfile.ZipFile) -> list[str]:
    names = archive.namelist()
    return [n for n in names if n in COPIED_MEMBERS or n.startswith(COPIED_PREFIXES)]


def _skip(reason: str) -> dict[str, Any]:
    click.echo(f"WARNING: no reference map carried over: {reason}", err=True)
    return {"status": "skipped", "reason": reason}


def inject(
    scene: Path,
    reference: str,
    output: Path,
    *,
    max_median_position_m: float = MAX_MEDIAN_POSITION_M,
    max_p95_position_m: float = MAX_P95_POSITION_M,
    max_median_rotation_deg: float = MAX_MEDIAN_ROTATION_DEG,
    min_matched_fraction: float = 0.9,
    warn_median_position_m: float = 0.5,
) -> dict[str, Any]:
    """Write ``output`` as ``scene`` plus the reference's map members and ``T_world_base``.

    Args:
        scene: scene package (USDZ) to place.
        reference: USDZ of the same clip with ``map.xodr`` and ``pose_record.json``, as a local or
            ``hf://`` path; may be ``scene`` itself to re-place it.
        output: path to write; must differ from both inputs.
        max_median_position_m, max_p95_position_m: fit position residual limits, metres.
        max_median_rotation_deg: fit rotation residual limit, degrees.
        min_matched_fraction: least fraction of the scene's poses paired with the record.
        warn_median_position_m: median position residual above which a warning is printed.

    Residuals are over the paired interior poses after the fit. Returns the report also written to
    ``<output stem>.reference_map.json``. A missing reference, or one without ``map.xodr`` or a pose
    record covering the scene, is skipped with a warning and writes no scene. Raises
    ``click.ClickException`` when ``output`` is an input, and when the fit exceeds a limit, in which
    case no output is left behind.
    """
    inputs = [scene] + ([Path(reference)] if reference and not reference.startswith("hf://") else [])
    for path in inputs:
        if output.resolve() == path.resolve() or (output.exists() and path.exists() and output.samefile(path)):
            raise click.ClickException(f"--output {output} is an input ({path}); write to another path")
    report: dict[str, Any] = {"scene": str(scene), "reference": reference, "output": str(output)}
    report_path = output.with_name(output.name.removesuffix(".usdz") + ".reference_map.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    def finish(extra: dict[str, Any]) -> dict[str, Any]:
        report.update(extra)
        if report["status"] != "injected":
            # A reused work directory must not present an earlier run's scene as this run's.
            output.unlink(missing_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        return report

    if not reference:
        return finish(_skip("no reference given"))
    try:
        handle = _open_reference(reference)
    except FileNotFoundError:
        return finish(_skip(f"{reference} not found"))

    try:
        ref = zipfile.ZipFile(handle)
    except zipfile.BadZipFile:
        handle.close()
        return finish(_skip(f"{reference} is not a readable USDZ"))

    with handle, ref, zipfile.ZipFile(scene) as src:
        copied = _reference_members(ref)
        if "map.xodr" not in copied:
            return finish(_skip(f"{reference} has no map.xodr"))
        if "pose_record.json" not in copied:
            return finish(_skip(f"{reference} has no pose_record.json"))
        scene_rig = json.loads(src.read(RIG_TRAJECTORIES))
        report.update(world_base_source="pose_record.json", previous_T_world_base=scene_rig.get("T_world_base"))
        try:
            world_base, agreement = world_base_from_pose_record(scene_rig, json.loads(ref.read("pose_record.json")))
        except ValueError as error:
            return finish(_skip(f"{reference}: {error}"))
        thresholds = {
            "max_median_position_m": max_median_position_m,
            "max_p95_position_m": max_p95_position_m,
            "max_median_rotation_deg": max_median_rotation_deg,
            "min_matched_fraction": min_matched_fraction,
        }
        report.update(agreement=agreement, thresholds=thresholds)
        if (
            agreement["matched_fraction"] < min_matched_fraction
            or agreement["median_position_m"] > max_median_position_m
            or agreement["p95_position_m"] > max_p95_position_m
            or agreement["median_rotation_deg"] > max_median_rotation_deg
        ):
            finish({"status": "failed"})
            raise click.ClickException(
                f"ego poses of {scene} do not fit the pose record of {reference} within {thresholds}: {agreement}"
            )
        if agreement["median_position_m"] > warn_median_position_m:
            median = agreement["median_position_m"]
            click.echo(f"WARNING: ego poses fit the pose record to {median:.2f} m median", err=True)

        scene_rig["T_world_base"] = world_base
        replaced = {RIG_TRAJECTORIES: json.dumps(scene_rig).encode()}
        if DATASOURCE_SUMMARY in src.namelist():
            summary = json.loads(src.read(DATASOURCE_SUMMARY))
            if "T_world_base" in (summary.get("rig_trajectories") or {}):
                summary["rig_trajectories"]["T_world_base"] = world_base
                replaced[DATASOURCE_SUMMARY] = json.dumps(summary).encode()

        tmp = output.with_name(output.name + ".tmp")
        try:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as out:
                for info in src.infolist():
                    if info.filename in copied:
                        continue
                    if info.filename in replaced:
                        out.writestr(info, replaced[info.filename], compress_type=zipfile.ZIP_STORED)
                    else:
                        with src.open(info) as fin, out.open(info, "w") as fout:
                            while chunk := fin.read(1 << 24):
                                fout.write(chunk)
                for name in copied:
                    out.writestr(zipfile.ZipInfo(name, ref.getinfo(name).date_time), ref.read(name))
            os.replace(tmp, output)
        finally:
            tmp.unlink(missing_ok=True)

    return finish({"status": "injected", "copied": copied, "replaced": sorted(replaced)})


@click.command("inject-reference-map")
@click.option("--scene-package", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--reference",
    default="",
    help="Reference USDZ of the same clip: a local path or an hf:// path. Empty or missing skips with a warning.",
)
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--max-median-position-m", default=MAX_MEDIAN_POSITION_M, show_default=True)
@click.option("--max-p95-position-m", default=MAX_P95_POSITION_M, show_default=True)
@click.option("--max-median-rotation-deg", default=MAX_MEDIAN_ROTATION_DEG, show_default=True)
def cli(
    scene_package: Path,
    reference: str,
    output: Path,
    max_median_position_m: float,
    max_p95_position_m: float,
    max_median_rotation_deg: float,
) -> None:
    """Carry a reference USDZ's map.xodr, pose record and ClipGT layers into a scene package, placed by the record."""
    report = inject(
        scene_package,
        reference,
        output,
        max_median_position_m=max_median_position_m,
        max_p95_position_m=max_p95_position_m,
        max_median_rotation_deg=max_median_rotation_deg,
    )
    click.echo(json.dumps({k: report[k] for k in ("status", "reason", "agreement") if k in report}))


if __name__ == "__main__":
    cli()
