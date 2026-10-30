# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reduce one clip's metric files to the benchmark's published row.

The chain writes four metric files in four shapes; the published benchmark reports one row per
scene. Which key feeds which published metric is a decision that changes the numbers -- the render
rate differs by ~7% between the CUDA-event and wall-clock timings, against a 5% regression
tolerance -- so it is made here, beside the spec, rather than in whatever assembles the package.

A missing value is always paired with a reason, so an absent metric is reportable rather than
silently blank.
"""

from __future__ import annotations

import json
import math
import os
import re

from typing import Any, Optional

import click
import yaml


# The published render rate is the CUDA-event measurement. Wall clock is carried beside it, never
# substituted: mixing the two across releases reads as a regression that did not happen.
RENDER_RATE_SOURCE = "cuda_ms"
RENDER_RATE_FALLBACK = "render_time_ms"
TIMING_METHOD = {"cuda_ms": "cuda_event", "render_time_ms": "wall_clock"}

# The PRD reports object quality for vehicles, vulnerable road users and other objects. The grouping
# changes the published numbers, so it is pinned here; any class not listed counts as "other".
VEHICLE_CLASSES = frozenset(
    {"automobile", "car", "heavy_truck", "truck", "bus", "trailer", "train", "train_or_tram_car", "other_vehicle"}
)
VRU_CLASSES = frozenset(
    {
        "pedestrian",
        "person",
        "person_group",
        "stroller",
        "rider",
        "bicycle",
        "bicycle_with_rider",
        "cyclist",
        "cycle",
        "motorcycle",
        "motorcycle_with_rider",
    }
)
DRIVER_VERSION_FILE = "/proc/driver/nvidia/version"


def _load(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    with open(path) as handle:
        return yaml.safe_load(handle)


def _aggregated(doc: Optional[dict], key: str) -> Optional[float]:
    value = ((doc or {}).get("aggregated_metrics") or {}).get(key)
    return value.get("value") if isinstance(value, dict) else None


def _cameras(doc: Optional[dict]) -> set[str]:
    cameras: set[str] = set()
    for sensors in (((doc or {}).get("metrics") or {}).get("per_sequence") or {}).values():
        cameras |= set((sensors or {}).get("per_camera") or {})
    return cameras


def _object_result(doc: dict, metric: str) -> dict:
    for entry in ((doc.get("metrics") or {}).get(metric) or {}).get("aggregated_results") or []:
        if entry.get("method") == "mean":
            return entry.get("result") or {}
    return {}


def _group(name: str) -> str:
    name = name.lower()
    if name in VEHICLE_CLASSES:
        return "vehicle"
    return "vru" if name in VRU_CLASSES else "other"


def _weighted(per_class: dict, names: list[str], key: str) -> Optional[float]:
    """Mean over classes weighted by the values behind each class mean, the evaluator's own weighting.

    The weight is the class's `<metric>_count` of finite values, or `num_objects` where the evaluator
    reports no count. A class with no finite value drops out.
    """
    count_key = key.removesuffix("_mean") + "_count"
    weights = {n: per_class[n].get(count_key, per_class[n].get("num_objects")) or 0 for n in names}
    used = [n for n in names if weights[n]]
    if not used or any(key not in per_class[n] for n in used):
        return None
    weights_used = [weights[n] for n in used]
    return sum(per_class[n][key] * w for n, w in zip(used, weights_used)) / sum(weights_used)


def gpu_driver_version(text: Optional[str]) -> Optional[str]:
    """Kernel driver version from the first line of /proc/driver/nvidia/version."""
    match = re.search(r"\s(\d+\.\d+(?:\.\d+)*)(?:\s|$)", (text or "").splitlines()[0] if text else "")
    return match.group(1) if match else None


def camera_count(cameras: str) -> int:
    """Number of cameras in a Hydra list string such as "[a,b,c]"."""
    return len([c for c in cameras.strip().strip("[]").split(",") if c.strip()])


def _put(row: dict[str, Any], field: str, value: Any, reason: str) -> None:
    """Write a published metric, or the reason it has no value."""
    if isinstance(value, float) and not math.isfinite(value):
        # An aggregate over zero finite samples is NaN; published as absent, since NaN is not JSON.
        row[field + "_absent_reason"] = "%s is not finite (%r)" % (field, value)
    elif value is None:
        row[field + "_absent_reason"] = reason
    else:
        row[field] = value


def build_row(
    validate_metrics: Optional[dict],
    rgb_metrics: Optional[dict],
    render_metrics: Optional[dict],
    manifest: Optional[dict] = None,
    clip_id: Optional[str] = None,
    object_metrics: Optional[dict] = None,
    expected_camera_count: int = 7,
    driver_version: Optional[str] = None,
) -> dict[str, Any]:
    """Map the chain's metric files onto the published row."""
    row: dict[str, Any] = {"clip_id": clip_id, "expected_camera_count": expected_camera_count}
    _put(row, "gpu_driver_version", driver_version, "no " + DRIVER_VERSION_FILE)

    if validate_metrics is None:
        row["fidelity_absent_reason"] = "validate wrote no metrics file"
    else:
        # The published "LiDAR return agreement" is raydrop accuracy; raydrop IoU is a different
        # quantity and is carried beside it rather than under the same name.
        for field, key in (
            ("camera_psnr_%dcam" % expected_camera_count, "test/psnr"),
            ("lidar_distance_mae", "test/depth_mae"),
            ("lidar_return_agreement", "test/raydrop_accuracy"),
            ("lidar_raydrop_iou", "test/raydrop_IoU"),
        ):
            _put(row, field, _aggregated(validate_metrics, key), "validate metrics carry no " + key)
        row["metrics_schema_version"] = validate_metrics.get("schema_version")
        row["hardware"] = validate_metrics.get("hardware")
        _put(
            row,
            "nre_version",
            (validate_metrics.get("program_version") or {}).get("version_string"),
            "validate metrics carry no program_version",
        )
        cameras = _cameras(validate_metrics)
        row["camera_count"] = len(cameras)
        if cameras and len(cameras) < expected_camera_count:
            # Silent on its own: a camera whose validation rays were all invalid vanishes here
            # while the renders still look correct.
            row["warnings"] = ["camera_dropout:%d/%d" % (len(cameras), expected_camera_count)]

    if rgb_metrics is None:
        row["rgb_absent_reason"] = "render metrics wrote no file"
    else:
        masked = ((rgb_metrics.get("cross_sequence_metrics") or {}).get("static_camera_mask")) or {}
        for source, published in (("psnr", "camera_psnr"), ("ssim", "camera_ssim"), ("lpips", "camera_lpips")):
            _put(row, published, (masked.get(source) or {}).get("avg"), "render metrics carry no %s average" % source)
        row["rgb_frame_count"] = (masked.get("psnr") or {}).get("cnt")
        row["render_camera_count"] = 1

    if render_metrics is None:
        row["render_rate_absent_reason"] = "render command wrote no metrics file"
    else:
        source = RENDER_RATE_SOURCE
        milliseconds = _aggregated(render_metrics, source)
        if milliseconds is None or not math.isfinite(milliseconds):
            source = RENDER_RATE_FALLBACK
            milliseconds = _aggregated(render_metrics, source)
        if milliseconds is not None and math.isfinite(milliseconds) and milliseconds > 0:
            row["single_camera_render_rate_fps"] = 1000.0 / milliseconds
            row["render_rate_timing_method"] = TIMING_METHOD[source]
            row["render_ms_mean"] = milliseconds
        else:
            row["render_rate_absent_reason"] = "render metrics carry no timing metric"

    if object_metrics is None:
        row["object_level_absent_reason"] = "object-level metrics wrote no file"
    else:
        perceptual = _object_result(object_metrics, "perceptual")
        semantic = _object_result(object_metrics, "semantic")
        for field, result, key in (
            ("object_perceptual_score", perceptual, "object_level_perceptual"),
            ("object_semantic_score", semantic, "object_level_semantic"),
        ):
            _put(row, field, (result.get("values") or {}).get(key), "object metrics carry no " + key)
        per_class = (perceptual.get("metadata") or {}).get("per_class") or {}
        semantic_per_class = (semantic.get("metadata") or {}).get("per_class") or {}
        row["object_count"] = sum(c.get("num_objects") or 0 for c in per_class.values())
        row["object_track_count"] = len((perceptual.get("metadata") or {}).get("per_track") or {})
        for group in ("vehicle", "vru", "other"):
            names = [n for n in per_class if _group(n) == group]
            count = sum(per_class[n].get("num_objects") or 0 for n in names)
            row["object_count_" + group] = count
            for field, classes, key in (
                ("object_perceptual_", per_class, "perceptual_score_mean"),
                ("object_semantic_", semantic_per_class, "semantic_raw_mean"),
                ("object_psnr_", per_class, "psnr_mean"),
            ):
                reason = "no %s objects scored" % group if not count else "object metrics carry no " + key
                present = [n for n in names if n in classes]
                value = _weighted(classes, present, key) if count and len(present) == len(names) else None
                _put(row, field + group, value, reason)

    # Runners update the manifest after each task, so the benchmark-row task can publish the
    # completed reconstruction phase timings. The row still supports callers without a manifest.
    if manifest is None:
        row["reconstruction_seconds_absent_reason"] = "runner manifest not supplied"
    else:
        _put(
            row,
            "reconstruction_seconds",
            manifest.get("reconstruction_seconds"),
            "runner manifest carries no reconstruction_seconds",
        )
        for phase, seconds in (manifest.get("phase_seconds") or {}).items():
            row["phase_seconds_" + phase] = seconds

    return row


@click.command("benchmark-row")
@click.option(
    "--validate-metrics",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="metrics.yaml written by the validate task.",
)
@click.option(
    "--rgb-metrics",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="rendering_metrics.yaml written by the render-metrics task.",
)
@click.option(
    "--render-metrics",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="metrics.yaml written by the render task.",
)
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Runner manifest, for the reconstruction-time rows.",
)
@click.option(
    "--object-metrics",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="metrics.yaml written by the object-level render-metrics task.",
)
@click.option(
    "--cameras",
    required=True,
    help="Trained and validated camera set, as the spec's Hydra list, e.g. [a,b,c].",
)
@click.option("--clip-id", default=None, help="Clip the row describes.")
@click.option("--output-file", type=click.Path(dir_okay=False), required=True)
def cli(validate_metrics, rgb_metrics, render_metrics, manifest, object_metrics, cameras, clip_id, output_file):
    """Reduce one clip's metric files to the benchmark's published row."""
    expected = camera_count(cameras)
    if not expected:
        raise click.BadParameter("names no camera", param_hint="--cameras")
    driver = None
    if os.path.exists(DRIVER_VERSION_FILE):
        with open(DRIVER_VERSION_FILE) as handle:
            driver = gpu_driver_version(handle.read())
    row = build_row(
        _load(validate_metrics),
        _load(rgb_metrics),
        _load(render_metrics),
        _load(manifest),
        clip_id,
        object_metrics=_load(object_metrics),
        expected_camera_count=expected,
        driver_version=driver,
    )
    with open(output_file, "w") as handle:
        json.dump(row, handle, indent=2, sort_keys=True, allow_nan=False)
    click.echo("wrote %s" % output_file)


if __name__ == "__main__":
    cli()
