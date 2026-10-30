# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch an external Nsight performance capture of the render command."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


NSIGHT_OUTPUT_DIR_ENV = "_NUREC_RENDER_NSYS_OUTPUT_DIR"
NSIGHT_STAGING_DIR_ENV = "_NUREC_RENDER_NSYS_STAGING_DIR"
NSIGHT_CAPTURE_MANIFEST = "nsight_captures.json"
NSIGHT_REPORT = "performance.nsys-rep"


class NsightCapture:
    """Keep one capture active across all selected frames of each sensor."""

    def __init__(self, output_dir: Path | None) -> None:
        self.output_dir = output_dir
        staging_dir = os.environ.get(NSIGHT_STAGING_DIR_ENV)
        # staging_dir is None exactly when output_dir is; deriving it here keeps that true by
        # construction, and lets the containment check below see a Path rather than Path | None.
        self.staging_dir: Path | None = None
        if output_dir is not None:
            self.staging_dir = Path(staging_dir) if staging_dir else output_dir / ".nsight"
            self.staging_dir.resolve().relative_to((output_dir / ".nsight").resolve())
        self.records: list[dict[str, Any]] = []
        self.capture_index = 0
        self.active = False
        self._sensor: tuple[str, str] | None = None
        self._sensor_traces: dict[tuple[str, str], str] = {}

    def start(self) -> None:
        """Start before initialization, or resume before the next sensor's warm-up."""
        if self.staging_dir is None or self.active:
            return
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        self.active = True
        self.capture_index += 1

    def stop(self) -> None:
        """Finish an active capture, including when initialization or rendering fails."""
        if self.active:
            try:
                torch.cuda.profiler.stop()
            finally:
                self.active = False

    @contextmanager
    def sensor(self, sensor_id: str, *, sensor_dir: Path, lidar: bool = False) -> Iterator[None]:
        """Export once after a sensor's warm-up and every selected frame/iteration."""
        output_dir = self.output_dir
        staging_dir = self.staging_dir
        if output_dir is None or staging_dir is None:
            yield
            return
        if self._sensor is not None:
            raise RuntimeError("Nsight sensor captures cannot be nested")
        sensor = ("lidar" if lidar else "camera", sensor_id)
        relative_dir = sensor_dir.resolve().relative_to(output_dir.resolve())
        if relative_dir == Path("."):
            raise ValueError("Nsight sensor reports require a sensor subdirectory")
        trace = (relative_dir / NSIGHT_REPORT).as_posix()
        if sensor in self._sensor_traces:
            raise RuntimeError(f"Nsight sensor {sensor_id!r} was already captured")
        if trace in self._sensor_traces.values():
            raise RuntimeError("Nsight sensors must have distinct report directories")
        # The first sensor reuses the capture already active during initialization.
        self.start()
        self._sensor = sensor
        self._sensor_traces[sensor] = trace
        complete = False
        try:
            yield
            complete = True
        finally:
            try:
                self.stop()
            except BaseException:
                # Export may have completed before Ctrl+C or a profiler error arrived.
                complete = False
                raise
            finally:
                self._sensor = None
                path = staging_dir / f"performance-{self.capture_index}.nsys-rep"
                if path.is_file() and path.stat().st_size:
                    destination = output_dir / trace
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(destination)
                    # Keep %n from reusing this index until all sensors have finished.
                    path.touch()
                    self.write_manifest(final=False)
                elif complete:
                    raise RuntimeError(f"Nsight did not produce the capture: {path}")

    @contextmanager
    def frame(self, sensor_id: str, frame: int, *, lidar: bool = False) -> Iterator[None]:
        """Index one frame without interrupting the sensor's continuous capture."""
        if self.output_dir is None:
            yield
            return
        sensor = ("lidar" if lidar else "camera", sensor_id)
        trace = self._sensor_traces.get(sensor)
        if not self.active or self._sensor != sensor or trace is None:
            raise RuntimeError("Nsight frames require an active matching sensor capture")
        complete = False
        try:
            yield
            complete = True
        finally:
            record = {sensor[0]: sensor_id, "frame": frame, "trace": trace}
            if not complete:
                record["complete"] = False
            self.records.append(record)

    def write_manifest(self, *, final: bool = True) -> None:
        """Persist frame locations; at shutdown remove only empty export reservations."""
        if self.output_dir is None:
            return
        manifest = self.output_dir / NSIGHT_CAPTURE_MANIFEST
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.records, indent=2) + "\n")
        temporary.replace(manifest)
        staging = self.staging_dir
        if final and staging is not None and staging.is_dir():
            for path in staging.glob("performance-*.nsys-rep"):
                if path.stat().st_size == 0:
                    path.unlink()
            # Preserve unassigned reports, including initialization-only failures.
            if not any(staging.iterdir()):
                staging.rmdir()
                staging_root = self.output_dir / ".nsight"
                if staging != staging_root and staging_root.is_dir() and not any(staging_root.iterdir()):
                    staging_root.rmdir()


def run_nsight_profile(command: list[str], output_dir: Path, *, expect_metrics: bool = True) -> None:
    """Capture one continuous range per sensor, then validate reports and expected metrics."""
    nsys = shutil.which("nsys")
    if nsys is None:
        raise RuntimeError("--nsight requires NVIDIA Nsight Systems (nsys) on PATH.")
    output_dir = output_dir.resolve()
    staging_root = output_dir / ".nsight"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix="capture-", dir=staging_root))
    environment = dict(os.environ)
    environment[NSIGHT_OUTPUT_DIR_ENV] = str(output_dir)
    environment[NSIGHT_STAGING_DIR_ENV] = str(staging_dir)
    # Bazel configures imports through sys.path; carry them into the child.
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    arguments = [
        nsys,
        "profile",
        "--trace=cuda,nvtx",
        "--cuda-event-trace=true",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=repeat:sync",
        "--force-overwrite=true",
        f"--output={staging_dir / 'performance-%n'}",
        *command,
    ]
    log_path = output_dir / "performance-capture.log"
    with log_path.open("w") as log:
        completed = subprocess.run(arguments, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        message = f"Nsight performance profiling failed; see {log_path}."
        # Surface the last Click validation error without requiring a log search.
        error = next(
            (
                line
                for line in reversed(log_path.read_text(errors="replace").splitlines())
                if line.startswith("Error: ")
            ),
            None,
        )
        if error is not None:
            message += f"\n{error}"
        raise RuntimeError(message)
    manifest_path = output_dir / NSIGHT_CAPTURE_MANIFEST
    if not manifest_path.is_file() or not manifest_path.stat().st_size:
        raise RuntimeError(f"Nsight performance profiling did not produce {manifest_path}; see {log_path}.")
    try:
        captures = json.loads(manifest_path.read_text())
        if not isinstance(captures, list) or not captures:
            raise ValueError("expected at least one capture")
        sensor_traces: dict[tuple[str, str], Path] = {}
        trace_sensors: dict[Path, tuple[str, str]] = {}
        for capture in captures:
            if type(capture["frame"]) is not int or capture["frame"] < 0:
                raise ValueError("capture frame indices must be nonnegative integers")
            sensor_types = [kind for kind in ("camera", "lidar") if kind in capture]
            if len(sensor_types) != 1:
                raise ValueError("each frame must identify exactly one camera or lidar")
            kind = sensor_types[0]
            sensor_id = capture[kind]
            if not isinstance(sensor_id, str) or not sensor_id:
                raise ValueError("capture sensor identifiers must be nonempty strings")
            sensor = (kind, sensor_id)
            trace = Path(capture["trace"])
            if trace.is_absolute() or ".." in trace.parts or trace.name != NSIGHT_REPORT or trace.parent == Path("."):
                raise ValueError("capture report paths must be relative sensor performance.nsys-rep paths")
            resolved_trace = (output_dir / trace).resolve()
            resolved_trace.relative_to(output_dir)
            if sensor in sensor_traces and sensor_traces[sensor] != resolved_trace:
                raise ValueError("all frames of a sensor must reference the same report")
            if resolved_trace in trace_sensors and trace_sensors[resolved_trace] != sensor:
                raise ValueError("different sensors must not share a capture report")
            sensor_traces[sensor] = resolved_trace
            trace_sensors[resolved_trace] = sensor
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"Invalid Nsight capture manifest {manifest_path}: {error}; see {log_path}.") from error
    required_outputs = list(trace_sensors)
    if expect_metrics:
        required_outputs.append(output_dir / "metrics.yaml")
    for path in required_outputs:
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"Nsight performance profiling did not produce {path}; see {log_path}.")
