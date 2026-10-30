# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiDAR export unit tests and process-isolated native GPU integration coverage."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import point_cloud_utils as pcu
import pytest
import torch

from ncore.data import RowOffsetStructuredSpinningLidarModelParameters
from ncore.impl.common.transformations import PoseInterpolator
from nurec.render.render import LidarFrame, LidarRayBundle
from nurec.utils.geometry import tquat_to_se3_matrix
from nurec.utils.io.export import render_lidar
from nurec.utils.types import FrameConversion, RigTrajectories


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Use reusable CPU output buffers to expose incorrect snapshot lifetimes."""
    state = SimpleNamespace(
        active=False,
        record_sample=False,
        report=True,
        timers=[],
        timer_calls=[],
        calls=[],
        builds=[],
        events=[],
        poses=[],
        cuda_values=[],
        error=None,
        models=0,
        copy_ranges=[],
        copy_range=None,
        warmup_ranges=[],
        pinned_buffers=[],
        pinned_copies=[],
        native_downloads=[],
        point_counts=None,
        frame_range=None,
        frame_ranges=[],
        captured=[],
        capture_completed=[],
        sensor_captures=[],
        capture_enabled=False,
    )
    parameters = SimpleNamespace(n_columns=3, n_rows=2)
    calibration = SimpleNamespace(
        logical_sensor_name="lidar_front",
        unique_sensor_idx=9,
        lidar_model_parameters=parameters,
        T_sensor_rig=torch.eye(4),
    )
    calibration.T_sensor_rig[0, 3] = 2
    trajectory = SimpleNamespace(
        sequence_id="sequence",
        lidars_frame_timestamps_us={"lidar@sequence": torch.tensor([[-1, 10], [11, 20], [21, 30]])},
        lidars_linear_start_frame_indices={"lidar@sequence": 100},
        T_rig_worlds=torch.eye(4).repeat(2, 1, 1),
        T_rig_world_timestamps_us=torch.tensor([0, 30]),
    )
    rigs: Any = SimpleNamespace(
        rig_trajectories=[trajectory],
        lidar_calibrations={"lidar@sequence": calibration},
        world_to_nurec=object(),
    )
    output = LidarFrame(torch.zeros((2, 3)), torch.zeros(2))

    class Timer:
        def __init__(self, label: str) -> None:
            assert label == "nurec.render.lidar.forward"
            self.milliseconds: list[float] = []
            self.valid: list[bool] = []
            state.timers.append(self)

        @contextmanager
        def __call__(self, device: torch.device, *, label: str | None = None, record_sample: bool = True):
            assert not state.active
            state.timer_calls.append((label, record_sample))
            with copy_range(label) if label is not None else nullcontext():
                state.active = True
                state.record_sample = record_sample
                state.events.append("start")
                try:
                    yield
                    if record_sample:
                        milliseconds = state.cuda_values[len(self.milliseconds)]
                        self.milliseconds.append(0.0 if milliseconds is None else milliseconds)
                        self.valid.append(milliseconds is not None)
                finally:
                    state.events.append("stop")
                    state.active = False
                    state.record_sample = False

    class Interpolator:
        def __init__(self, **kwargs: Any) -> None:
            assert not state.active

        def interpolate_to_timestamps(self, timestamps: np.ndarray) -> np.ndarray:
            assert not state.active
            state.poses.append(timestamps.copy())
            poses = np.tile(np.eye(4), (2, 1, 1))
            poses[:, 0, 3] = timestamps
            return poses

    def build(**kwargs: Any) -> dict:
        state.builds.append({**kwargs, "timed": state.active, "record_sample": state.record_sample})
        state.events.append("build")
        return kwargs

    def native_render(bundle: dict, **kwargs: Any) -> LidarFrame:
        assert kwargs == {"neighbor_source": "angle"}
        if state.capture_enabled:
            assert capture.active
            assert capture.frame_index is not None, "a capturing render must have a frame index"
            assert bundle["unique_frame_idx"] == 100 + capture.frame_index
        assert state.active, "Warmup and measured renders must both use CUDA event timing"
        measured = state.frame_range is not None
        if not measured:
            assert not state.record_sample, "Warmup renders must not publish timing samples"
        state.calls.append(measured)
        state.events.append("render")
        if state.error is not None:
            raise TypeError(state.error)
        if state.point_counts is not None:
            count = state.point_counts[len(state.calls) - 1]
            if len(output.point_positions) != count:
                output.point_positions = torch.empty((count, 3))
                output.point_intensities = torch.empty(count)
        output.point_positions.fill_(len(state.calls))
        if output.point_intensities is not None:
            output.point_intensities.fill_(len(state.calls) + 0.5)
        return output

    def synchronize(device: torch.device) -> None:
        assert not state.active and state.frame_range is None
        if state.capture_enabled:
            assert capture.active
        state.events.append("synchronize")

    def clock() -> float:
        assert state.active and state.frame_range is not None and state.report
        state.events.append("clock")
        return float(state.events.count("clock")) * 0.01

    def model_bundle(values: Any) -> Any:
        assert not state.active and values is parameters
        state.models += 1
        return values

    # Both snapshots and PLY downloads must occur after the stop event.
    original_cpu, original_to = torch.Tensor.cpu, torch.Tensor.to
    original_copy, original_empty_like = torch.Tensor.copy_, torch.empty_like

    @contextmanager
    def copy_range(label: str):
        if label.startswith("nurec.render.lidar.frame("):
            before = len(state.calls)
            assert not state.active and state.frame_range is None
            state.frame_range = label
            try:
                yield
            finally:
                assert not state.active
                assert all(state.calls[before:])
                state.frame_range = None
                state.frame_ranges.append((label, before, state.calls[before:]))
            return
        if label.startswith("nurec.render.lidar.warmup("):
            before = len(state.calls)
            assert not state.active and state.frame_range is None
            yield
            assert state.frame_range is None
            assert len(state.calls) == before + 1 and state.calls[-1] is False
            state.warmup_ranges.append((label, before, len(state.calls)))
            return
        assert not state.active and state.copy_range is None
        record = {"label": label, "after_render": len(state.calls), "operations": []}
        state.copy_ranges.append(record)
        state.copy_range = record
        try:
            yield
        finally:
            assert not state.active
            state.copy_range = None

    def record_native_download(tensor: torch.Tensor) -> None:
        # CPU tensors emulate native CUDA buffers; shared storage identifies their downloads.
        for name in ("point_positions", "point_intensities"):
            native = getattr(output, name)
            if native is not None and tensor.data_ptr() == native.data_ptr():
                if state.capture_enabled:
                    assert capture.active
                state.native_downloads.append((len(state.calls), name))

    def cpu(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        assert not state.active
        if state.copy_range is not None:
            state.copy_range["operations"].append("cpu")
            record_native_download(tensor)
        return original_cpu(tensor, *args, **kwargs)

    def to(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        assert not state.active
        if args and str(args[0]) == "cpu":
            assert state.copy_range is not None, "Retained point clouds need an NVTX copy label"
            state.copy_range["operations"].append("to_cpu")
        return original_to(tensor, *args, **kwargs)

    def empty_like(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if kwargs.get("pin_memory"):
            assert not state.active and state.copy_range is not None
            assert str(kwargs.get("device")) == "cpu"
            result = original_empty_like(tensor, *args, **{**kwargs, "pin_memory": False})
            state.pinned_buffers.append(result)
            return result
        return original_empty_like(tensor, *args, **kwargs)

    def copy_(tensor: torch.Tensor, source: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        for index, buffer in enumerate(state.pinned_buffers):
            if tensor is buffer:
                assert not state.active and state.copy_range is not None
                assert not non_blocking, "Snapshot copies must finish before buffers can be reused"
                state.copy_range["operations"].append("copy_cpu")
                state.pinned_copies.append((index, len(state.calls)))
                record_native_download(source)
                break
        return original_copy(tensor, source, non_blocking=non_blocking)

    monkeypatch.setattr(render_lidar, "CudaEventTimer", Timer)
    monkeypatch.setattr(render_lidar, "PoseInterpolator", Interpolator)
    monkeypatch.setattr(render_lidar, "LidarModelBundle", model_bundle)
    monkeypatch.setattr(render_lidar.LidarRayBundle, "build", build)
    monkeypatch.setattr(render_lidar.torch.cuda.nvtx, "range", copy_range)
    monkeypatch.setattr(render_lidar.torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(render_lidar.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(render_lidar.time, "perf_counter", clock)
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    monkeypatch.setattr(torch, "empty_like", empty_like)
    renderable: Any = SimpleNamespace(
        device=torch.device("cuda:0"),
        default_lidar_neighbor_source=lambda: "angle",
        render_lidar_frame_from_ray_bundle=native_render,
    )
    metrics = Mock()

    class Capture:
        active = False
        frame_index: int | None = None
        sensor_id: str | None = None

        def start(self) -> None:
            if self.active:
                return
            assert not state.active
            state.capture_enabled = True
            self.active = True
            state.events.append("capture-start")

        def stop(self) -> None:
            if not self.active:
                return
            assert not state.active and self.frame_index is None
            state.events.append("capture-stop")
            self.active = False

        @contextmanager
        def sensor(self, sensor_id: str, *, sensor_dir: Path, lidar: bool = False):
            assert sensor_dir == tmp_path / "nsight" / "lidar_front"
            assert lidar and self.sensor_id is None and not state.active
            self.start()  # Inherit the first capture started before model initialization.
            self.sensor_id = sensor_id
            before = len(state.calls)
            try:
                yield
            finally:
                assert self.active and self.frame_index is None and not state.active
                self.stop()
                state.sensor_captures.append((sensor_id, lidar, before, len(state.calls)))
                state.events.append("capture-export")
                self.sensor_id = None

        @contextmanager
        def frame(self, sensor_id: str, frame: int, *, lidar: bool = False):
            assert isinstance(frame, int) and lidar
            assert self.active and sensor_id == self.sensor_id
            assert not state.active and self.frame_index is None
            assert state.events.count("clock") % 2 == 0
            self.frame_index = frame
            before = len(state.calls)
            complete = False
            try:
                yield
                assert state.events.count("clock") % 2 == 0
                complete = True
            finally:
                assert self.active and not state.active and state.frame_range is None
                state.captured.append((sensor_id, frame, lidar, before, len(state.calls)))
                state.capture_completed.append(complete)
                self.frame_index = None

    capture = Capture()

    def run(
        *,
        save_modes: set[str] | None = None,
        report: bool = True,
        warmup_frames: int = 0,
        iterations: int = 2,
        frame_step: int = 2,
        frame_index: str = "all",
        nsight_capture: Any = None,
        frame_naming: str = "contiguous-output-index",
        cuda_values: tuple[float | None, ...] = (4.0, 1.0, 6.0, 3.0),
    ) -> None:
        state.cuda_values = cuda_values
        state.report = report
        # Simulate the outer CLI's pre-initialization start and idempotent Click
        # finalizer. The sensor context stops/exports after all its frames and saves.
        if nsight_capture is not None:
            nsight_capture.start()
        try:
            render_lidar.render_lidar_frames(
                renderable,
                rigs,
                output_dir=tmp_path,
                frame_step=frame_step,
                frame_index=frame_index,
                nsight_capture=nsight_capture,
                warmup_frames=warmup_frames,
                iterations=iterations,
                save_modes=save_modes or set(),
                frame_naming=frame_naming,
                metrics_collector=metrics if report else None,
            )
        finally:
            if nsight_capture is not None:
                nsight_capture.stop()

    return SimpleNamespace(
        run=run,
        state=state,
        capture=capture,
        rigs=rigs,
        trajectory=trajectory,
        calibration=calibration,
        output=output,
        metrics=metrics,
        root=tmp_path,
        sensor_dir=tmp_path / "lidar" / "lidar_front",
        extra_dir=tmp_path / "lidar-extra" / "lidar_front",
    )


def read_ply(path: Path) -> tuple[str, np.ndarray]:
    header, payload = path.read_bytes().split(b"end_header\n", 1)
    text = header.decode("ascii")
    if "element vertex" not in text:
        assert not payload
        return text, np.empty((0, 3))
    mesh = pcu.load_triangle_mesh(str(path))
    vertices = mesh.vertex_data.positions
    if mesh.vertex_data.colors is not None:
        vertices = np.column_stack((vertices, mesh.vertex_data.colors))
    return text, vertices


def test_warmup_stride_grid_and_paired_clocks(harness: SimpleNamespace) -> None:
    harness.run(warmup_frames=2)
    assert harness.state.calls == [False, False, True, True, True, True]
    assert harness.state.warmup_ranges == [
        ("nurec.render.lidar.warmup(frame=0, iteration=1)", 0, 1),
        ("nurec.render.lidar.warmup(frame=0, iteration=2)", 1, 2),
    ]
    assert all(build["timed"] for build in harness.state.builds)
    assert [build["record_sample"] for build in harness.state.builds] == harness.state.calls
    assert [build["unique_frame_idx"] for build in harness.state.builds] == [100, 100, 100, 100, 102, 102]
    assert all(build["unique_sensor_idx"] == 9 for build in harness.state.builds)
    assert all(build["world_to_nurec"] is harness.rigs.world_to_nurec for build in harness.state.builds)
    assert harness.state.events[:8] == ["start", "build", "render", "stop"] * 2
    assert harness.state.events[8:14] == ["start", "clock", "build", "render", "clock", "stop"]
    assert "synchronize" not in harness.state.events
    assert harness.state.timer_calls == [
        ("nurec.render.lidar.warmup(frame=0, iteration=1)", False),
        ("nurec.render.lidar.warmup(frame=0, iteration=2)", False),
        *[(None, True)] * 4,
    ]
    np.testing.assert_array_equal(harness.state.poses, [[10, 10], [21, 30]])
    pose = harness.state.builds[-1]["lidar_to_world"]
    assert pose.start_pose_tquat_sensor_world[0].item() == 23
    assert pose.end_pose_tquat_sensor_world[0].item() == 32
    records = [call.kwargs for call in harness.metrics.record_frame.call_args_list]
    assert [(row["frame"], row["iteration"]) for row in records] == [(0, 1), (0, 2), (2, 1), (2, 2)]
    assert [row["cuda_ms"] for row in records] == [4.0, 1.0, 6.0, 3.0]
    assert [row["render_time_ms"] for row in records] == pytest.approx([10.0] * 4)
    assert all(row["sensor_id"] == "lidar_front" and row["sequence_id"] == "sequence" for row in records)
    assert all(row["is_lidar"] is True for row in records)
    for call in harness.metrics.set_metric_metadata.call_args_list:
        assert call.kwargs == {"lidar_id": "lidar_front", "ray_grid": [3, 2]}
    manifest = json.loads((harness.sensor_dir / "timestamps.json").read_text())
    assert [row["file_name"] for row in manifest] == ["000000.ply", "000001.ply"]
    assert [row["source_frame_idx"] for row in manifest] == [0, 2]
    assert [row["iteration"] for row in manifest] == [2, 2]
    for row, value in zip(manifest, (4, 6), strict=True):
        _, vertices = read_ply(harness.sensor_dir / row["file_name"])
        np.testing.assert_array_equal(vertices[:, :3], np.full((2, 3), value))
    assert not harness.extra_dir.exists()


@pytest.mark.parametrize("save_mode", ["all-iterations", "last-iteration"])
@pytest.mark.parametrize(
    "cuda_values,expected_valid",
    [
        ((None, 1.0, None, 3.0), [(0, 2, 1.0), (2, 2, 3.0)]),
        ((4.0, None, None, None), [(0, 1, 4.0)]),
        ((None, None, None, None), []),
    ],
)
def test_invalid_timings_preserve_metrics_and_clouds_but_cannot_select_extrema(
    harness: SimpleNamespace,
    save_mode: str,
    cuda_values: tuple[float | None, ...],
    expected_valid: list[tuple[int, int, float]],
) -> None:
    harness.run(cuda_values=cuda_values, save_modes={save_mode, "min", "max"})

    assert harness.state.calls == [True] * 4
    identities = [(0, 1), (0, 2), (2, 1), (2, 2)]
    expected = [
        (frame, iteration, 0.0 if value is None else value, value is not None)
        for (frame, iteration), value in zip(identities, cuda_values, strict=True)
    ]
    records = [call.kwargs for call in harness.metrics.record_frame.call_args_list]
    assert [
        (row["frame"], row["iteration"], row["cuda_ms"], row["metric_validity"]["cuda_ms"]) for row in records
    ] == expected
    assert [row["render_time_ms"] for row in records] == pytest.approx([10.0] * 4)
    canonical = json.loads((harness.sensor_dir / "timestamps.json").read_text())
    assert [row["file_name"] for row in canonical] == ["000000.ply", "000001.ply"]
    assert [(row["source_frame_idx"], row["iteration"], row["cuda_ms"], row["valid"]) for row in canonical] == [
        row for row in expected if row[1] == 2
    ]
    saved = json.loads((harness.extra_dir / "timestamps.json").read_text())
    indexed = expected if save_mode == "all-iterations" else [row for row in expected if row[1] == 2]
    extrema = (
        [(*min(expected_valid, key=lambda row: row[2]), True), (*max(expected_valid, key=lambda row: row[2]), True)]
        if expected_valid
        else []
    )
    assert [
        (row["source_frame_idx"], row["iteration"], row["cuda_ms"], row["valid"]) for row in saved
    ] == indexed + extrema
    assert len(list(harness.sensor_dir.glob("*.ply"))) == len(canonical)
    assert len(list(harness.extra_dir.glob("*.ply"))) == len(saved)
    for row in canonical:
        _, vertices = read_ply(harness.sensor_dir / row["file_name"])
        native_call = row["source_frame_idx"] + row["iteration"]
        np.testing.assert_array_equal(vertices[:, :3], np.full((2, 3), native_call))
    for row in saved:
        _, vertices = read_ply(harness.extra_dir / row["file_name"])
        native_call = row["source_frame_idx"] + row["iteration"]
        np.testing.assert_array_equal(vertices[:, :3], np.full((2, 3), native_call))


@pytest.mark.parametrize("frame_offset", [None, 40])
def test_custom_lidar_only_trajectory_passes_interpolated_poses_and_calibration_to_ray_builder(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, frame_offset: int | None
) -> None:
    # Use the native CPU interpolator; a rotated rig makes the extrinsic composition order observable.
    monkeypatch.setattr(render_lidar, "PoseInterpolator", PoseInterpolator)
    rig_poses = torch.tensor(
        [
            [[0, -1, 0, 2], [1, 0, 0, 10], [0, 0, 1, 1], [0, 0, 0, 1]],
            [[0, -1, 0, 8], [1, 0, 0, 10], [0, 0, 1, 1], [0, 0, 0, 1]],
        ],
        dtype=torch.float64,
    )
    metadata: dict[str, Any] = {
        "sequence_id": "custom-sequence",
        "rig_bbox": None,
        "cameras_frame_timestamps_us": {},
        "lidars_frame_timestamps_us": {"lidar@sequence": [[1000, 1250], [1250, 1750]]},
        "T_rig_worlds": rig_poses.tolist(),
        "T_rig_world_timestamps_us": [1000, 2000],
    }
    if frame_offset is not None:
        metadata["lidars_linear_start_frame_indices"] = {"lidar@sequence": frame_offset}
    harness.rigs.rig_trajectories = [RigTrajectories.RigTrajectory.from_dict(metadata)]
    harness.rigs.camera_calibrations = {}
    harness.rigs.world_to_nurec = FrameConversion(
        matrix=np.array([[1, 0, 0, 20], [0, 1, 0, -30], [0, 0, 1, 40], [0, 0, 0, 1]], dtype=np.float64)
    )
    harness.calibration.T_sensor_rig = torch.tensor(
        [[-1, 0, 0, 2], [0, -1, 0, 1], [0, 0, 1, 0.5], [0, 0, 0, 1]], dtype=torch.float32
    )

    harness.run(frame_index="1", iterations=1, cuda_values=(3.0,))

    assert len(harness.state.builds) == 1
    build = harness.state.builds[0]
    assert build["lidar_model"] is harness.calibration.lidar_model_parameters
    assert build["world_to_nurec"] is harness.rigs.world_to_nurec
    assert build["unique_sensor_idx"] == harness.calibration.unique_sensor_idx
    assert build["unique_frame_idx"] == (None if frame_offset is None else frame_offset + 1)
    pose = build["lidar_to_world"]
    assert (pose.start_timestamp_us, pose.end_timestamp_us) == (1250, 1750)
    for tquat, x in ((pose.start_pose_tquat_sensor_world, 2.5), (pose.end_pose_tquat_sensor_world, 5.5)):
        expected = torch.tensor([[0, 1, 0, x], [-1, 0, 0, 12], [0, 0, 1, 1.5], [0, 0, 0, 1]], dtype=torch.float32)
        torch.testing.assert_close(tquat_to_se3_matrix(tquat), expected)
    assert harness.metrics.record_frame.call_args.kwargs["sequence_id"] == "custom-sequence"


@pytest.mark.parametrize(
    "modes,expected",
    [
        ({"last-iteration"}, [(0, 2, "", 2), (2, 2, "", 4)]),
        ({"all-iterations"}, [(0, 1, "", 1), (0, 2, "", 2), (2, 1, "", 3), (2, 2, "", 4)]),
        ({"min"}, [(0, 2, "-min", 2)]),
        ({"max"}, [(2, 1, "-max", 3)]),
        ({"min", "max"}, [(0, 2, "-min", 2), (2, 1, "-max", 3)]),
        (
            {"last-iteration", "all-iterations", "min", "max"},
            [(0, 1, "", 1), (0, 2, "", 2), (2, 1, "", 3), (2, 2, "", 4), (0, 2, "-min", 2), (2, 1, "-max", 3)],
        ),
    ],
)
def test_save_modes_preserve_actual_clouds(harness: SimpleNamespace, modes: set[str], expected: list[tuple]) -> None:
    harness.run(save_modes=modes)
    records = json.loads((harness.extra_dir / "timestamps.json").read_text())
    assert len(list(harness.extra_dir.glob("*.ply"))) == len(expected)
    canonical = json.loads((harness.sensor_dir / "timestamps.json").read_text())
    assert [
        (row["file_name"], row["render_frame_idx"], row["source_frame_idx"], row["iteration"]) for row in canonical
    ] == [
        ("000000.ply", 0, 0, 2),
        ("000001.ply", 1, 2, 2),
    ]
    assert len(list(harness.sensor_dir.glob("*.ply"))) == 2
    for row, value in zip(canonical, (2, 4), strict=True):
        _, vertices = read_ply(harness.sensor_dir / row["file_name"])
        np.testing.assert_array_equal(vertices[:, :3], np.full((2, 3), value))
    assert len(harness.state.calls) == 4  # Saving extrema never renders again.
    for record, (frame, iteration, label, value) in zip(records, expected, strict=True):
        name = f"render-3x2-frame-{frame}-iteration-{iteration}{label}.ply"
        assert record["file_name"] == name
        assert (record["source_frame_idx"], record["iteration"]) == (frame, iteration)
        header, vertices = read_ply(harness.extra_dir / name)
        assert "element vertex 2" in header
        assert "property uchar red" in header
        assert "property float intensity" not in header
        np.testing.assert_array_equal(vertices[:, :3], np.full((2, 3), value))
        np.testing.assert_array_equal(vertices[:, 3:], np.tile([0, 0, 255, 255], (2, 1)))


def test_copy_ranges_distinguish_cloud_candidates_and_final_exports(harness: SimpleNamespace) -> None:
    harness.run(save_modes={"min", "max"})
    assert harness.state.copy_ranges == [
        {
            "label": "nurec.render.lidar_copy(selection=min,max, frame=0, iteration=1)",
            "after_render": 1,
            "operations": ["copy_cpu", "copy_cpu"],
        },
        {
            "label": "nurec.render.lidar_copy(selection=min, frame=0, iteration=2)",
            "after_render": 2,
            "operations": ["copy_cpu", "copy_cpu"],
        },
        {
            "label": "nurec.render.lidar_copy(selection=last, frame=0, iteration=2)",
            "after_render": 2,
            "operations": ["cpu", "cpu"],
        },
        {
            "label": "nurec.render.lidar_copy(selection=max, frame=2, iteration=1)",
            "after_render": 3,
            "operations": ["copy_cpu", "copy_cpu"],
        },
        {
            "label": "nurec.render.lidar_copy(selection=last, frame=2, iteration=2)",
            "after_render": 4,
            "operations": ["cpu", "cpu"],
        },
        {
            "label": "nurec.render.lidar_copy(selection=min, frame=0, iteration=2)",
            "after_render": 4,
            "operations": ["cpu", "cpu"],
        },
        {
            "label": "nurec.render.lidar_copy(selection=max, frame=2, iteration=1)",
            "after_render": 4,
            "operations": ["cpu", "cpu"],
        },
    ]


def test_ordinary_cloud_copy_ranges_include_frame_and_iteration_identity(harness: SimpleNamespace) -> None:
    harness.run(save_modes={"all-iterations"})
    assert [entry["label"] for entry in harness.state.copy_ranges] == [
        f"nurec.render.lidar_copy(selection={selection}, frame={frame}, iteration={iteration})"
        for frame in (0, 2)
        for iteration, selection in ((1, "index"), (2, "last"))
    ]
    assert [entry["after_render"] for entry in harness.state.copy_ranges] == [1, 2, 3, 4]
    assert all(entry["operations"] == ["cpu", "cpu"] for entry in harness.state.copy_ranges)


@pytest.mark.parametrize("ordinary", ["last-iteration", "all-iterations"])
def test_combined_saves_download_each_native_cloud_once(harness: SimpleNamespace, ordinary: str) -> None:
    harness.run(save_modes={ordinary, "min", "max"})
    assert harness.state.native_downloads == [
        (call, name) for call in (1, 2, 3, 4) for name in ("point_positions", "point_intensities")
    ]
    assert harness.metrics.record_frame.call_count == 4
    expected = [(0, 2, 2), (2, 2, 4)] if ordinary == "last-iteration" else [(0, 1, 1), (0, 2, 2), (2, 1, 3), (2, 2, 4)]
    for frame, iteration, value in expected:
        _, vertices = read_ply(harness.extra_dir / f"render-3x2-frame-{frame}-iteration-{iteration}.ply")
        np.testing.assert_array_equal(vertices[:, :3], np.full((2, 3), value))
        np.testing.assert_array_equal(vertices[:, 3:], np.tile([0, 0, 255, 255], (2, 1)))
    for selection, frame, iteration, value in (("min", 0, 2, 2), ("max", 2, 1, 3)):
        _, vertices = read_ply(harness.extra_dir / f"render-3x2-frame-{frame}-iteration-{iteration}-{selection}.ply")
        np.testing.assert_array_equal(vertices[:, :3], np.full((2, 3), value))


def test_pinned_extrema_reuse_buffers_without_overwriting_shared_clouds(harness: SimpleNamespace) -> None:
    harness.run(save_modes={"min", "max"}, frame_index="0", iterations=6, cuda_values=(5, 4, 3, 6, 7, 2))
    assert len(harness.state.pinned_buffers) == 4  # XYZ and intensity for each retained extremum.
    assert harness.state.pinned_copies == [
        (buffer, call)
        for pair, call in (((0, 1), 1), ((2, 3), 2), ((2, 3), 3), ((0, 1), 4), ((0, 1), 5), ((2, 3), 6))
        for buffer in pair
    ]
    for selection, iteration in (("min", 6), ("max", 5)):
        _, vertices = read_ply(harness.extra_dir / f"render-3x2-frame-0-iteration-{iteration}-{selection}.ply")
        assert np.all(vertices[:, :3] == iteration)
        np.testing.assert_array_equal(vertices[:, 3:], np.tile([0, 0, 255, 255], (len(vertices), 1)))


def test_pinned_cloud_buffers_reallocate_only_when_point_count_changes(harness: SimpleNamespace) -> None:
    harness.state.point_counts = (2, 2, 3, 3)
    harness.run(save_modes={"min"}, frame_index="0", iterations=4, cuda_values=(4, 3, 2, 1))
    assert [tuple(buffer.shape) for buffer in harness.state.pinned_buffers] == [(2, 3), (2,), (3, 3), (3,)]
    assert harness.state.pinned_copies == [
        (buffer, call) for pair, call in (((0, 1), 1), ((0, 1), 2), ((2, 3), 3), ((2, 3), 4)) for buffer in pair
    ]
    _, vertices = read_ply(harness.extra_dir / "render-3x2-frame-0-iteration-4-min.ply")
    assert vertices.shape == (3, 7)
    assert np.all(vertices[:, :3] == 4)
    np.testing.assert_array_equal(vertices[:, 3:], np.tile([0, 0, 255, 255], (3, 1)))


def test_ties_keep_first_and_missing_offsets_stay_optional(harness: SimpleNamespace) -> None:
    harness.trajectory.lidars_linear_start_frame_indices = None
    harness.run(save_modes={"min", "max"}, cuda_values=(2.0,) * 4)
    assert len(harness.state.pinned_buffers) == 2
    assert harness.state.pinned_copies == [(0, 1), (1, 1)]
    assert all(build["unique_frame_idx"] is None for build in harness.state.builds)
    for selection in ("min", "max"):
        _, vertices = read_ply(harness.extra_dir / f"render-3x2-frame-0-iteration-1-{selection}.ply")
        assert np.all(vertices[:, :3] == 1)


@pytest.mark.parametrize(
    "naming,expected",
    [
        ("frame-end-timestamp", "00000000000000000010.ply"),
        ("contiguous-output-index", "000000.ply"),
    ],
)
def test_zero_warmup_one_capture_and_legacy_names(harness: SimpleNamespace, naming: str, expected: str) -> None:
    harness.run(
        save_modes={"last-iteration"},
        iterations=1,
        frame_step=100,
        frame_naming=naming,
        cuda_values=(3.0,),
    )
    assert harness.state.calls == [True]
    assert not harness.state.warmup_ranges
    assert len(harness.metrics.record_frame.call_args_list) == 1
    assert (harness.sensor_dir / expected).is_file()
    assert (harness.extra_dir / "render-3x2-frame-0-iteration-1.ply").read_bytes() == (
        harness.sensor_dir / expected
    ).read_bytes()
    assert harness.state.native_downloads == [(1, "point_positions"), (1, "point_intensities")]


@pytest.mark.parametrize("nsight", [False, True])
def test_save_none_keeps_timings_and_captures_without_output_copies_or_files(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, nsight: bool
) -> None:
    save_ply = Mock()
    monkeypatch.setattr(render_lidar, "save_ply", save_ply)
    harness.run(
        save_modes={"none"},
        frame_index="1-2",
        frame_step=1,
        warmup_frames=2,
        nsight_capture=harness.capture if nsight else None,
    )

    assert harness.state.calls == [False, False, True, True, True, True]
    assert len(harness.state.warmup_ranges) == 2
    assert all(build["timed"] for build in harness.state.builds)
    records = [call.kwargs for call in harness.metrics.record_frame.call_args_list]
    assert [(row["frame"], row["iteration"], row["cuda_ms"]) for row in records] == [
        (1, 1, 4.0),
        (1, 2, 1.0),
        (2, 1, 6.0),
        (2, 2, 3.0),
    ]
    if nsight:
        assert harness.state.captured == [("lidar_front", 1, True, 0, 4), ("lidar_front", 2, True, 4, 6)]
    assert not harness.capture.active
    save_ply.assert_not_called()
    assert not harness.state.copy_ranges
    assert not harness.state.native_downloads
    assert not harness.state.pinned_buffers
    assert not harness.state.pinned_copies
    # The capture fixture and metrics mock do not write files; render products and
    # timestamp manifests must not create anything themselves in this mode.
    assert list(harness.root.rglob("*")) == []


def test_no_report_no_save_preserves_timed_capture_without_reporting_or_samples(
    harness: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger=render_lidar.__name__)
    harness.run(
        report=False,
        save_modes={"none"},
        frame_index="1-2",
        frame_step=1,
        warmup_frames=2,
        nsight_capture=harness.capture,
        cuda_values=(),
    )

    assert harness.state.calls == [False, False, True, True, True, True]
    assert harness.state.captured == [("lidar_front", 1, True, 0, 4), ("lidar_front", 2, True, 4, 6)]
    assert harness.state.timer_calls == [
        ("nurec.render.lidar.warmup(frame=1, iteration=1)", False),
        ("nurec.render.lidar.warmup(frame=1, iteration=2)", False),
        *[(None, False)] * 4,
    ]
    assert all(build["timed"] for build in harness.state.builds)
    assert harness.state.events.count("start") == harness.state.events.count("stop") == 6
    assert "clock" not in harness.state.events
    assert len(harness.state.timers) == 1
    assert harness.state.timers[0].milliseconds == harness.state.timers[0].valid == []
    assert not harness.metrics.mock_calls
    assert "render_time_ms=" not in caplog.text
    assert not harness.state.copy_ranges
    assert not harness.state.native_downloads
    assert not harness.state.pinned_buffers
    assert not harness.state.pinned_copies
    assert not harness.capture.active
    assert list(harness.root.rglob("*")) == []


def test_no_report_preserves_saved_timing_metadata_and_actual_extrema(
    harness: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger=render_lidar.__name__)
    harness.run(report=False, save_modes={"min", "max"}, warmup_frames=2)

    assert "clock" not in harness.state.events
    assert not harness.metrics.mock_calls
    assert "render_time_ms=" not in caplog.text
    assert harness.state.timers[0].milliseconds == [4.0, 1.0, 6.0, 3.0]
    canonical = json.loads((harness.sensor_dir / "timestamps.json").read_text())
    assert [(row["source_frame_idx"], row["iteration"], row["cuda_ms"]) for row in canonical] == [
        (0, 2, 1.0),
        (2, 2, 3.0),
    ]
    extras = json.loads((harness.extra_dir / "timestamps.json").read_text())
    assert [(row["source_frame_idx"], row["iteration"], row["cuda_ms"]) for row in extras] == [
        (0, 2, 1.0),
        (2, 1, 6.0),
    ]
    for row, expected_position in zip(extras, (4, 5), strict=True):
        _, vertices = read_ply(harness.extra_dir / row["file_name"])
        assert np.all(vertices[:, :3] == expected_position)


def test_contiguous_primary_indices_are_independent_of_extra_clouds(harness: SimpleNamespace) -> None:
    harness.run(save_modes={"all-iterations", "min", "max"}, frame_naming="contiguous-output-index")

    canonical = json.loads((harness.sensor_dir / "timestamps.json").read_text())
    extras = json.loads((harness.extra_dir / "timestamps.json").read_text())
    assert [row["file_name"] for row in canonical] == ["000000.ply", "000001.ply"]
    assert [row["render_frame_idx"] for row in canonical] == [0, 1]
    assert [row["file_name"] for row in extras] == [
        "render-3x2-frame-0-iteration-1.ply",
        "render-3x2-frame-0-iteration-2.ply",
        "render-3x2-frame-2-iteration-1.ply",
        "render-3x2-frame-2-iteration-2.ply",
        "render-3x2-frame-0-iteration-2-min.ply",
        "render-3x2-frame-2-iteration-1-max.ply",
    ]
    assert [row["render_frame_idx"] for row in extras] == list(range(6))
    for primary, extra in (
        ("000000.ply", "render-3x2-frame-0-iteration-2.ply"),
        ("000001.ply", "render-3x2-frame-2-iteration-2.ply"),
    ):
        assert (harness.sensor_dir / primary).read_bytes() == (harness.extra_dir / extra).read_bytes()
    assert harness.metrics.record_frame.call_count == 4


def test_invalid_last_iteration_still_saves_canonical_cloud_without_extras(harness: SimpleNamespace) -> None:
    harness.run(cuda_values=(1.0, None, 2.0, None))
    canonical = json.loads((harness.sensor_dir / "timestamps.json").read_text())
    assert [row["file_name"] for row in canonical] == ["000000.ply", "000001.ply"]
    assert all(row["iteration"] == 2 and row["valid"] is False for row in canonical)
    for row, value in zip(canonical, (2, 4), strict=True):
        _, vertices = read_ply(harness.sensor_dir / row["file_name"])
        np.testing.assert_array_equal(vertices[:, :3], np.full((2, 3), value))
    assert not harness.extra_dir.exists()
    assert harness.metrics.record_frame.call_count == 4


@pytest.mark.parametrize("empty", [False, True])
def test_ply_without_intensity_preserves_xyz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty: bool) -> None:
    monkeypatch.setattr(render_lidar.torch.cuda.nvtx, "range", lambda label: nullcontext())
    positions = torch.empty((0, 3)) if empty else torch.tensor([[1.25, -2.5, 0.0], [0.5, 10, -9]])
    path = tmp_path / "cloud.ply"
    render_lidar.save_ply(LidarFrame(positions), path)
    header, vertices = read_ply(path)
    assert "intensity" not in header
    assert "property uchar red" not in header
    np.testing.assert_array_equal(vertices, positions.numpy())


def test_sensor_output_name_is_sanitized_without_changing_metric_identity(harness: SimpleNamespace) -> None:
    harness.calibration.logical_sensor_name = "lidar/front"

    harness.run(frame_index="0", iterations=1, save_modes={"last-iteration"}, cuda_values=(3.0,))

    for subdir in ("lidar", "lidar-extra"):
        sensor_dir = harness.root / subdir / "lidar_front"
        assert len(list(sensor_dir.glob("*.ply"))) == 1
        assert (sensor_dir / "timestamps.json").is_file()
        assert {path.name for path in (harness.root / subdir).iterdir()} == {"lidar_front"}
    assert harness.metrics.record_frame.call_args.kwargs["sensor_id"] == "lidar/front"


@pytest.mark.parametrize("sensor_name", ["", ".", ".."])
def test_invalid_sensor_name_fails_before_output_or_rendering(harness: SimpleNamespace, sensor_name: str) -> None:
    harness.calibration.logical_sensor_name = sensor_name

    with pytest.raises(ValueError, match="Invalid LiDAR sensor name"):
        harness.run(save_modes={"last-iteration"})

    assert not any(harness.root.iterdir())
    assert harness.state.models == 0
    assert not harness.state.builds
    assert not harness.state.calls
    harness.metrics.set_metric_metadata.assert_not_called()
    harness.metrics.record_frame.assert_not_called()


@pytest.mark.parametrize("columns, rows", [(0, 2), (-1, 2), (3, 0), (3, -1)])
def test_invalid_native_ray_grid_fails_before_rendering(harness: SimpleNamespace, columns: int, rows: int) -> None:
    parameters = harness.calibration.lidar_model_parameters
    parameters.n_columns = columns
    parameters.n_rows = rows

    with pytest.raises(ValueError, match=f"invalid native ray grid: {columns}x{rows}"):
        harness.run()

    assert harness.state.models == 0
    assert not harness.state.builds
    assert not harness.state.calls
    assert not any(harness.root.iterdir())
    harness.metrics.set_metric_metadata.assert_not_called()
    harness.metrics.record_frame.assert_not_called()


@pytest.mark.parametrize("missing", ["trajectory", "sensor", "calibration", "missing_calibration", "frames"])
def test_missing_lidar_fails_before_rendering(harness: SimpleNamespace, missing: str) -> None:
    if missing == "trajectory":
        harness.rigs.rig_trajectories = []
    elif missing == "sensor":
        harness.trajectory.lidars_frame_timestamps_us = {}
    elif missing == "missing_calibration":
        harness.rigs.lidar_calibrations.clear()
    elif missing == "calibration":
        harness.calibration.lidar_model_parameters = None
    else:
        harness.trajectory.lidars_frame_timestamps_us["lidar@sequence"] = torch.empty((0, 2), dtype=torch.int64)
    with pytest.raises(ValueError, match="no rig trajectories|no renderable LiDAR"):
        harness.run()
    assert not harness.state.calls
    harness.metrics.record_frame.assert_not_called()


def test_unusable_first_sensor_is_skipped(harness: SimpleNamespace) -> None:
    harness.trajectory.lidars_frame_timestamps_us["a"] = torch.empty((0, 2), dtype=torch.int64)
    harness.rigs.lidar_calibrations["a"] = SimpleNamespace(lidar_model_parameters=None)
    harness.run(iterations=1, frame_step=100, cuda_values=(3.0,))
    assert len(harness.state.calls) == 1


def test_unsupported_model_has_actionable_error(harness: SimpleNamespace) -> None:
    harness.state.error = "Lidar rendering does not support models of type Kelvin."
    with pytest.raises(ValueError, match="does not support LiDAR rendering in NuRec"):
        harness.run()
    harness.metrics.record_frame.assert_not_called()


def test_other_native_errors_are_preserved(harness: SimpleNamespace) -> None:
    harness.state.error = "invalid native request"
    with pytest.raises(TypeError, match="invalid native request"):
        harness.run()


def test_nonzero_single_frame_ignores_stride_and_saves_its_index(harness: SimpleNamespace) -> None:
    harness.run(frame_index="2", frame_step=100, iterations=1, save_modes={"last-iteration"}, cuda_values=(3.0,))
    assert harness.state.calls == [True]
    assert [build["unique_frame_idx"] for build in harness.state.builds] == [102]
    np.testing.assert_array_equal(harness.state.poses, [[21, 30]])
    record = harness.metrics.record_frame.call_args.kwargs
    assert (record["frame"], record["iteration"]) == (2, 1)
    manifest = json.loads((harness.sensor_dir / "timestamps.json").read_text())
    assert len(manifest) == 1
    assert manifest[0]["source_frame_idx"] == 2
    assert manifest[0]["file_name"] == "000000.ply"
    extra_manifest = json.loads((harness.extra_dir / "timestamps.json").read_text())
    assert extra_manifest[0]["file_name"] == "render-3x2-frame-2-iteration-1.ply"


def test_nonzero_range_anchors_stride_and_warmup_to_first_selected_frame(harness: SimpleNamespace) -> None:
    harness.trajectory.lidars_frame_timestamps_us["lidar@sequence"] = torch.tensor(
        [[0, 10], [11, 20], [21, 30], [31, 40], [41, 50]]
    )
    harness.run(
        frame_index="1-4",
        frame_step=2,
        warmup_frames=2,
        save_modes={"last-iteration", "min", "max"},
    )
    assert harness.state.calls == [False, False, True, True, True, True]
    assert [build["unique_frame_idx"] for build in harness.state.builds] == [101, 101, 101, 101, 103, 103]
    assert [entry[0] for entry in harness.state.warmup_ranges] == [
        "nurec.render.lidar.warmup(frame=1, iteration=1)",
        "nurec.render.lidar.warmup(frame=1, iteration=2)",
    ]
    records = [call.kwargs for call in harness.metrics.record_frame.call_args_list]
    assert [(row["frame"], row["iteration"]) for row in records] == [(1, 1), (1, 2), (3, 1), (3, 2)]
    assert harness.state.events[:8] == ["start", "build", "render", "stop"] * 2
    assert "synchronize" not in harness.state.events
    np.testing.assert_array_equal(harness.state.poses, [[11, 20], [31, 40]])
    manifest = json.loads((harness.sensor_dir / "timestamps.json").read_text())
    assert [entry["source_frame_idx"] for entry in manifest] == [1, 3]
    assert [entry["file_name"] for entry in manifest] == ["000000.ply", "000001.ply"]
    extra_manifest = json.loads((harness.extra_dir / "timestamps.json").read_text())
    assert [entry["source_frame_idx"] for entry in extra_manifest] == [1, 3, 1, 3]
    for frame, iteration, selection, value in ((1, 2, "min", 4), (3, 1, "max", 5)):
        _, vertices = read_ply(harness.extra_dir / f"render-3x2-frame-{frame}-iteration-{iteration}-{selection}.ply")
        assert np.all(vertices[:, :3] == value)


@pytest.mark.parametrize("selection", ["3", "1-3", "2-1", "-1", "invalid"])
def test_invalid_selection_fails_before_sensor_initialization(harness: SimpleNamespace, selection: str) -> None:
    with pytest.raises(ValueError):
        harness.run(frame_index=selection, frame_step=100, save_modes={"last-iteration"})
    assert harness.state.models == 0
    assert not harness.state.builds
    assert not harness.state.calls
    assert not harness.state.poses
    harness.metrics.record_frame.assert_not_called()
    assert not (harness.root / "lidar").exists()


@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("warmup_frames", [0, 2])
def test_nsight_sensor_capture_contains_all_iterations_and_final_exports(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, save: bool, warmup_frames: int
) -> None:
    original_save = render_lidar.save_ply

    def save_cloud(rendered: LidarFrame, path: Path, *, copy_label: str = "nurec.render.lidar_copy") -> None:
        assert not harness.state.active
        # Continuous capture also includes final exports from retained CPU snapshots.
        assert harness.capture.active
        assert path.parent in (harness.sensor_dir, harness.extra_dir)
        harness.state.events.append("save")
        original_save(rendered, path, copy_label=copy_label)

    monkeypatch.setattr(render_lidar, "save_ply", save_cloud)
    harness.run(
        frame_index="1-2",
        frame_step=1,
        warmup_frames=warmup_frames,
        save_modes={"last-iteration", "min", "max"} if save else set(),
        nsight_capture=harness.capture,
    )
    assert harness.state.captured == [
        ("lidar_front", 1, True, 0, warmup_frames + 2),
        ("lidar_front", 2, True, warmup_frames + 2, warmup_frames + 4),
    ]
    assert not harness.capture.active
    assert harness.state.calls == [False] * warmup_frames + [True] * 4
    events = harness.state.events
    before_measurement = ["capture-start"] + ["start", "build", "render", "stop"] * warmup_frames
    assert "synchronize" not in events
    assert events[: len(before_measurement)] == before_measurement
    assert events[-2:] == ["capture-stop", "capture-export"]
    assert events.count("capture-export") == 1
    assert harness.state.sensor_captures == [("lidar_front", True, 0, warmup_frames + 4)]
    assert events.count("capture-start") == events.count("capture-stop") == 1
    assert harness.state.capture_completed == [True, True]
    assert events.count("start") == events.count("stop") == warmup_frames + 4
    assert harness.state.frame_ranges == [
        ("nurec.render.lidar.frame(lidar=lidar_front, frame=1)", warmup_frames, [True, True]),
        ("nurec.render.lidar.frame(lidar=lidar_front, frame=2)", warmup_frames + 2, [True, True]),
    ]
    assert len(harness.state.warmup_ranges) == warmup_frames
    records = [call.kwargs for call in harness.metrics.record_frame.call_args_list]
    assert [(row["frame"], row["iteration"]) for row in records] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    expected_saves = 4 if save else 2
    assert events.count("save") == expected_saves
    assert len(list(harness.sensor_dir.glob("*.ply"))) == 2
    assert len(list(harness.extra_dir.glob("*.ply"))) == (4 if save else 0)
    assert harness.state.models == 1


def test_nsight_frame_capture_handles_zero_warmup_and_one_selected_iteration(harness: SimpleNamespace) -> None:
    harness.run(frame_index="2", frame_step=100, iterations=1, cuda_values=(3.0,), nsight_capture=harness.capture)
    assert harness.state.captured == [("lidar_front", 2, True, 0, 1)]
    assert harness.state.calls == [True]
    assert harness.state.frame_ranges == [("nurec.render.lidar.frame(lidar=lidar_front, frame=2)", 0, [True])]
    assert not harness.state.warmup_ranges
    assert "synchronize" not in harness.state.events
    assert not harness.capture.active
    assert harness.metrics.record_frame.call_count == 1


@pytest.mark.parametrize("warmup_frames", [0, 1])
def test_nsight_failed_frame_is_recorded_before_sensor_capture_finalization(
    harness: SimpleNamespace, warmup_frames: int
) -> None:
    harness.state.error = "native rendering failed"
    with pytest.raises(TypeError, match="native rendering failed"):
        harness.run(frame_index="1-2", frame_step=1, warmup_frames=warmup_frames, nsight_capture=harness.capture)
    assert harness.state.captured == [("lidar_front", 1, True, 0, 1)]
    assert harness.state.capture_completed == [False]
    assert harness.state.events[0] == "capture-start"
    assert harness.state.events[-2:] == ["capture-stop", "capture-export"]
    assert harness.state.sensor_captures == [("lidar_front", True, 0, 1)]
    assert not harness.capture.active and not harness.state.active
    assert harness.state.frame_range is None
    harness.metrics.record_frame.assert_not_called()


def _native_lidar_bundle_worker(output_dir: Path) -> None:
    parameters = RowOffsetStructuredSpinningLidarModelParameters(
        spinning_frequency_hz=10.0,
        spinning_direction="ccw",
        n_rows=2,
        n_columns=4,
        row_elevations_rad=np.array([0.1, -0.1], dtype=np.float32),
        column_azimuths_rad=np.linspace(-0.5, 0.5, 4).astype(np.float32),
        row_azimuth_offsets_rad=np.zeros(2, dtype=np.float32),
    )
    sensor_id = "lidar_front@sequence"
    sensor_pose = torch.eye(4, dtype=torch.float32)
    sensor_pose[0, 3] = 2.0
    calibration = RigTrajectories.LidarCalibration(
        sequence_id="sequence",
        logical_sensor_name="lidar_front",
        unique_sensor_idx=9,
        T_sensor_rig=sensor_pose,
        lidar_model_parameters=parameters,
    )
    trajectory = RigTrajectories.RigTrajectory(
        sequence_id="sequence",
        rig_bbox=None,
        cameras_frame_timestamps_us={},
        lidars_frame_timestamps_us={sensor_id: torch.tensor([[0, 100_000], [100_000, 200_000]], dtype=torch.int64)},
        lidars_linear_start_frame_indices={sensor_id: 40},
        T_rig_worlds=torch.eye(4, dtype=torch.float64).repeat(2, 1, 1),
        T_rig_world_timestamps_us=torch.tensor([0, 200_000], dtype=torch.int64),
    )
    rigs = RigTrajectories(
        T_world_base=torch.eye(4, dtype=torch.float64),
        world_to_nurec=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
        rig_trajectories=[trajectory],
        camera_calibrations=OrderedDict(),
        lidar_calibrations=OrderedDict([(sensor_id, calibration)]),
    )
    received: list[LidarRayBundle] = []

    def native_render(bundle: LidarRayBundle, *, neighbor_source: str) -> LidarFrame:
        assert isinstance(bundle, LidarRayBundle)
        assert neighbor_source == "angle"
        received.append(bundle)
        return LidarFrame(bundle.rendering_data.rays.reshape(-1, 6)[:, :3])

    # Only the final scene renderer is substituted; calibration, model construction,
    # pose interpolation, CUDA ray generation, and timing use the real implementations.
    renderable: Any = SimpleNamespace(
        device=torch.device("cuda:0"),
        default_lidar_neighbor_source=lambda: "angle",
        render_lidar_frame_from_ray_bundle=native_render,
    )
    render_lidar.render_lidar_frames(
        renderable,
        rigs,
        output_dir=output_dir,
        frame_index="1",
        frame_step=1,
        warmup_frames=0,
        iterations=1,
        save_modes={"none"},
        frame_naming="frame-end-timestamp",
        metrics_collector=None,
    )

    assert len(received) == 1
    bundle = received[0]
    data = bundle.rendering_data
    assert data.rays.is_cuda
    assert data.rays.shape == (1, 2, 4, 6)
    assert torch.isfinite(data.rays).all()
    assert bundle.model_elements.is_cuda
    assert bundle.model_elements.shape == (8, 2)
    assert data.sensor_model_parameters[0] is parameters
    assert bundle.frame_meta.unique_sensor_idx == 9
    assert bundle.frame_meta.unique_frame_idx == 41
    torch.testing.assert_close(data.timestamps_startend_us_cpu, torch.tensor([[100_000, 200_000]]))
    torch.testing.assert_close(
        data.rays[..., :3], torch.tensor([2.0, 0.0, 0.0], device=data.rays.device).expand(1, 2, 4, 3)
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(data.rays[..., 3:], dim=-1), torch.ones((1, 2, 4), device=data.rays.device)
    )


def test_native_lidar_bundle_reaches_renderer_in_isolated_process(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Native LiDAR bundle integration test requires a GPU")
    # Preserve Bazel's import paths without inheriting any in-process CUDA mocks.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    completed = subprocess.run(
        [sys.executable, __file__, "--native-lidar-bundle-worker", str(tmp_path)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--native-lidar-bundle-worker":
    _native_lidar_bundle_worker(Path(sys.argv[2]))
