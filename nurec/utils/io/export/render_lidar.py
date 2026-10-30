# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render native LiDAR frames with optional timing reports and point-cloud saving."""

from __future__ import annotations

import json
import logging
import re
import shutil
import time

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from ncore.impl.common.transformations import PoseInterpolator
from nurec.render import PoseRange, RenderableModel
from nurec.render.render import LidarFrame, LidarRayBundle
from nurec.utils.cudaeventtimer import CudaEventTimer
from nurec.utils.geometry import se3_matrix_to_tquat
from nurec.utils.io.export.render_helpers import select_frame_indices, validate_sensor_output_names
from nurec.utils.io.export.render_lidar_ply import save_lidar_ply
from nurec.utils.io.export.render_nsight import NsightCapture
from nurec.utils.lidar_model import LidarModelBundle
from nurec.utils.metrics import MetricsCollector
from nurec.utils.types import RigTrajectories


log = logging.getLogger(__name__)


@dataclass
class LidarRenderSample:
    """Keep a rendered cloud and its identity; valid applies to the CUDA timing."""

    rendered: LidarFrame
    frame: int
    iteration: int
    cuda_ms: float
    start_us: int
    end_us: int
    valid: bool = True


def save_ply(rendered: LidarFrame, path: Path, *, copy_label: str = "nurec.render.lidar_copy") -> None:
    """Download after timing, then use the same intensity-colored PLY writer as gRPC."""
    with torch.cuda.nvtx.range(copy_label):
        positions = rendered.point_positions.detach().cpu().numpy()
        intensities = None if rendered.point_intensities is None else rendered.point_intensities.detach().cpu().numpy()
    save_lidar_ply(path, positions, intensities)


def render_lidar_frames(
    renderable: RenderableModel,
    rigs: RigTrajectories,
    *,
    output_dir: Path,
    frame_step: int,
    warmup_frames: int,
    iterations: int,
    save_modes: set[str],
    frame_naming: str,
    metrics_collector: MetricsCollector | None,
    frame_index: str = "all",
    nsight_capture: NsightCapture | None = None,
) -> None:
    """Use the selected trajectory's first calibrated LiDAR at its native ray grid."""
    if not rigs.rig_trajectories:
        raise ValueError("Artifact contains no rig trajectories")
    trajectory = rigs.rig_trajectories[0]
    for unique_id in sorted(trajectory.lidars_frame_timestamps_us):
        calibration = rigs.lidar_calibrations.get(unique_id)
        if calibration is None:
            continue
        parameters = calibration.lidar_model_parameters
        timestamps = trajectory.lidars_frame_timestamps_us[unique_id].detach().cpu().numpy()
        if parameters is not None and timestamps.ndim == 2 and timestamps.shape[1] == 2 and len(timestamps):
            break
    else:
        raise ValueError("Selected trajectory contains no renderable LiDAR")

    frame_indices = select_frame_indices(frame_index, len(timestamps), frame_step)
    columns, rows = int(parameters.n_columns), int(parameters.n_rows)
    if columns <= 0 or rows <= 0:
        raise ValueError(f"LiDAR has invalid native ray grid: {columns}x{rows}")
    lidar_id = calibration.logical_sensor_name
    sensor_name = re.sub(r"[^A-Za-z0-9_.-]", "_", lidar_id)
    validate_sensor_output_names([sensor_name], sensor_type="LiDAR")
    report = metrics_collector is not None
    save_outputs = "none" not in save_modes
    record_sample = report or save_outputs
    sensor_dir = output_dir / "lidar" / sensor_name
    if save_outputs:
        sensor_dir.mkdir(parents=True, exist_ok=True)
    extra_dir = output_dir / "lidar-extra" / sensor_name
    if save_outputs and save_modes:
        extra_dir.mkdir(parents=True, exist_ok=True)
    if metrics_collector is not None:
        for metric in ("render_time_ms", "cuda_ms"):
            metrics_collector.set_metric_metadata(metric, lidar_id=lidar_id, ray_grid=[columns, rows])
    log.info("LiDAR %s: %dx%d native rays, %d available frames", lidar_id, columns, rows, len(timestamps))

    # Only CPU poses are assembled here; native rendering owns uploads and caches.
    interpolator = PoseInterpolator(
        poses=trajectory.T_rig_worlds.detach().cpu().numpy(),
        timestamps=trajectory.T_rig_world_timestamps_us.detach().cpu().numpy(),
    )
    sensor_to_rig = calibration.T_sensor_rig.detach().cpu().numpy()
    offsets = trajectory.lidars_linear_start_frame_indices
    minimum: LidarRenderSample | None = None
    maximum: LidarRenderSample | None = None
    output_timestamps: list[dict[str, str | int | float]] = []
    extra_timestamps: list[dict[str, str | int | float]] = []
    cuda_timer = CudaEventTimer("nurec.render.lidar.forward")
    nsight_capture = nsight_capture or NsightCapture(None)

    def save_sample(
        sample: LidarRenderSample, selection: str = "", *, extra: bool = False, copy_from: Path | None = None
    ) -> Path:
        """Keep canonical frame numbering independent of additional profiling clouds."""
        manifest = extra_timestamps if extra else output_timestamps
        output_index = len(manifest)
        if extra:
            stem = f"render-{columns}x{rows}-frame-{sample.frame}-iteration-{sample.iteration}"
            if selection:
                stem += f"-{selection}"
        else:
            match frame_naming:
                case "frame-end-timestamp":
                    stem = f"{sample.end_us:020d}"
                case "contiguous-output-index":
                    stem = f"{output_index:06d}"
                case _:
                    raise ValueError(f"Invalid frame naming scheme: {frame_naming}")
        file_name = f"{stem}.ply"
        path = (extra_dir if extra else sensor_dir) / file_name
        if copy_from is not None:
            # The final iteration may also be an extra: reuse its completed PLY.
            shutil.copyfile(copy_from, path)
        else:
            save_ply(
                sample.rendered,
                path,
                copy_label=(
                    f"nurec.render.lidar_copy(selection={selection or ('index' if extra else 'last')}, "
                    f"frame={sample.frame}, iteration={sample.iteration})"
                ),
            )
        manifest.append(
            {
                "file_name": file_name,
                "render_frame_idx": output_index,
                "source_frame_idx": sample.frame,
                "iteration": sample.iteration,
                "cuda_ms": sample.cuda_ms,
                "valid": sample.valid,
                "frame_start_timestamp_us": sample.start_us,
                "frame_end_timestamp_us": sample.end_us,
            }
        )
        return path

    with (
        torch.cuda.device(renderable.device),
        torch.inference_mode(),
        nsight_capture.sensor(lidar_id, sensor_dir=output_dir / "nsight" / sensor_name, lidar=True),
    ):
        # This native constructor initializes the sensor once, outside measurement.
        lidar_model = LidarModelBundle(parameters)
        neighbor_source = renderable.default_lidar_neighbor_source()
        for frame in frame_indices:
            start, end = map(int, timestamps[frame])
            if start < 0:
                start = end  # Negative starts denote instantaneous scans in older artifacts.
            poses = interpolator.interpolate_to_timestamps(np.asarray([start, end], dtype=np.int64))
            lidar_to_world = PoseRange(
                se3_matrix_to_tquat(np.asarray(poses[0] @ sensor_to_rig, dtype=np.float32)),
                se3_matrix_to_tquat(np.asarray(poses[1] @ sensor_to_rig, dtype=np.float32)),
                start,
                end,
            )
            unique_frame_idx = None if offsets is None else int(offsets[unique_id]) + frame

            def render_frame() -> LidarFrame:
                """Include native uploads, GPU ray generation, rendering and final filtering."""
                ray_bundle = LidarRayBundle.build(
                    lidar_model=lidar_model,
                    lidar_to_world=lidar_to_world,
                    world_to_nurec=rigs.world_to_nurec,
                    unique_sensor_idx=calibration.unique_sensor_idx,
                    unique_frame_idx=unique_frame_idx,
                )
                try:
                    return renderable.render_lidar_frame_from_ray_bundle(ray_bundle, neighbor_source=neighbor_source)
                except TypeError as error:
                    if "Lidar rendering does not support models" not in str(error):
                        raise
                    raise ValueError("This artifact model does not support LiDAR rendering in NuRec") from error

            # Index each frame while keeping this sensor's capture active.
            with nsight_capture.frame(lidar_id, frame, lidar=True):
                if frame == frame_indices.start:
                    for warmup_iteration in range(1, warmup_frames + 1):
                        # Match measured event boundaries without publishing warmup samples.
                        with cuda_timer(
                            renderable.device,
                            label=f"nurec.render.lidar.warmup(frame={frame}, iteration={warmup_iteration})",
                            record_sample=False,
                        ):
                            warmup = render_frame()
                        del warmup

                # The frame range contains measured iterations only, after warm-up finishes.
                with torch.cuda.nvtx.range(f"nurec.render.lidar.frame(lidar={lidar_id}, frame={frame})"):
                    for iteration in range(1, iterations + 1):
                        if report:
                            with cuda_timer(renderable.device):
                                start_time = time.perf_counter()
                                rendered = render_frame()
                                render_time_ms = (time.perf_counter() - start_time) * 1000
                        else:
                            with cuda_timer(renderable.device, record_sample=record_sample):
                                rendered = render_frame()
                        if not record_sample:
                            del rendered
                            continue
                        cuda_ms = cuda_timer.milliseconds[-1]
                        cuda_valid = cuda_timer.valid[-1]
                        if metrics_collector is not None:
                            metrics_collector.record_frame(
                                sequence_id=trajectory.sequence_id or "",
                                sensor_id=lidar_id,
                                frame=frame,
                                iteration=iteration,
                                is_lidar=True,
                                metric_validity={"cuda_ms": cuda_valid},
                                render_time_ms=render_time_ms,
                                cuda_ms=cuda_ms,
                            )
                            log.info(
                                "LiDAR %s frame %d iteration %d: render_time_ms=%.3f cuda_ms=%.3f valid=%s",
                                lidar_id,
                                frame,
                                iteration,
                                render_time_ms,
                                cuda_ms,
                                cuda_valid,
                            )
                        if not save_outputs:
                            del rendered
                            continue
                        sample = LidarRenderSample(rendered, frame, iteration, cuda_ms, start, end, valid=cuda_valid)
                        is_minimum = (
                            cuda_valid and "min" in save_modes and (minimum is None or cuda_ms < minimum.cuda_ms)
                        )
                        is_maximum = (
                            cuda_valid and "max" in save_modes and (maximum is None or cuda_ms > maximum.cuda_ms)
                        )
                        if is_minimum or is_maximum:
                            # Reuse a pinned snapshot only when the other extremum does not retain it.
                            previous = minimum if is_minimum else maximum
                            if minimum is maximum:
                                previous = None
                            selection = ",".join(
                                mode for mode, improved in (("min", is_minimum), ("max", is_maximum)) if improved
                            )
                            with torch.cuda.nvtx.range(
                                f"nurec.render.lidar_copy(selection={selection}, frame={frame}, iteration={iteration})"
                            ):
                                tensors: dict[str, torch.Tensor | None] = {}
                                for name in ("point_positions", "point_intensities"):
                                    source = getattr(rendered, name)
                                    destination = None if previous is None else getattr(previous.rendered, name)
                                    if source is None:
                                        tensors[name] = None
                                        continue
                                    if (
                                        destination is None
                                        or destination.shape != source.shape
                                        or destination.dtype != source.dtype
                                    ):
                                        destination = torch.empty_like(source, device="cpu", pin_memory=True)
                                    # Blocking copies finish outside both rendering clocks.
                                    destination.copy_(source)
                                    tensors[name] = destination
                                positions = tensors["point_positions"]
                                assert positions is not None
                                snapshot = replace(
                                    rendered, point_positions=positions, point_intensities=tensors["point_intensities"]
                                )
                            retained = replace(sample, rendered=snapshot)
                            if is_minimum:
                                minimum = retained
                            if is_maximum:
                                maximum = retained
                            del previous
                        # A timing extremum already has a complete CPU snapshot.
                        saved_sample = retained if is_minimum or is_maximum else sample
                        canonical_path = save_sample(saved_sample) if iteration == iterations else None
                        if "all-iterations" in save_modes or "last-iteration" in save_modes and iteration == iterations:
                            save_sample(saved_sample, extra=True, copy_from=canonical_path)
                        del sample, saved_sample, rendered

        if minimum is not None:
            save_sample(minimum, "min", extra=True)
        if maximum is not None:
            save_sample(maximum, "max", extra=True)
    if save_outputs:
        (sensor_dir / "timestamps.json").write_text(json.dumps(output_timestamps, indent=2) + "\n")
        if save_modes:
            (extra_dir / "timestamps.json").write_text(json.dumps(extra_timestamps, indent=2) + "\n")
