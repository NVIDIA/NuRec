# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the continuous Nsight launcher without running Nsight or rendering."""

import json
import os
import subprocess
import sys

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nurec.utils.io.export.render_nsight import (
    NSIGHT_CAPTURE_MANIFEST,
    NSIGHT_OUTPUT_DIR_ENV,
    NSIGHT_REPORT,
    NSIGHT_STAGING_DIR_ENV,
    NsightCapture,
    run_nsight_profile,
)


CudaProfilerMocks = tuple[MagicMock, MagicMock, MagicMock]


@pytest.fixture
def cuda_profiler() -> Iterator[CudaProfilerMocks]:
    with (
        patch("nurec.utils.io.export.render_nsight.torch.cuda.synchronize") as synchronize,
        patch("nurec.utils.io.export.render_nsight.torch.cuda.profiler.start") as start,
        patch("nurec.utils.io.export.render_nsight.torch.cuda.profiler.stop") as stop,
    ):
        yield synchronize, start, stop


def staging_of(capture: NsightCapture) -> Path:
    """The capture's staging directory, which exists whenever it was given an output directory.

    `staging_dir` is `Path | None` because a capture with no output directory is a no-op. Every
    caller of this helper uses an enabled capture, and most read it from inside a nested callback
    where narrowing the attribute would not survive.
    """
    staging = capture.staging_dir
    assert staging is not None
    return staging


@pytest.mark.parametrize("expect_metrics", [True, False])
def test_capture_preserves_command_imports_and_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expect_metrics: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = Path("run with spaces")
    output_dir.mkdir()
    command = [
        "/python path/python",
        "/entry point/run.py",
        "render",
        "--nsight",
        "--artifact-path",
        "/scene dir/model.usdz",
        "--save",
        "min,max",
        "--output-dir",
        "/parent dir",
    ]
    original_command = command.copy()
    monkeypatch.setenv("NUREC_TEST_PARENT_VALUE", "preserved")
    imports = ["/runfiles root", "/dependency,one", ""]
    completed: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(command, 0)

    def capture(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert Path(kwargs["env"][NSIGHT_STAGING_DIR_ENV]).is_dir()
        kwargs["stdout"].write("captured stdout and stderr\n")
        for camera in ("front", "rear"):
            sensor_dir = output_dir / "color" / camera
            sensor_dir.mkdir(parents=True)
            (sensor_dir / NSIGHT_REPORT).write_bytes(b"sensor trace")
        if expect_metrics:
            (output_dir / "metrics.yaml").write_text("frames: []\n")
        (output_dir / NSIGHT_CAPTURE_MANIFEST).write_text(
            json.dumps(
                [
                    {"camera": camera, "frame": 4, "trace": f"color/{camera}/{NSIGHT_REPORT}"}
                    for camera in ("front", "rear")
                ]
            )
        )
        return completed

    with (
        patch("nurec.utils.io.export.render_nsight.shutil.which", return_value="/nsight path/nsys"),
        patch("nurec.utils.io.export.render_nsight.subprocess.run", side_effect=capture) as run,
        patch.object(sys, "path", imports),
    ):
        if expect_metrics:
            run_nsight_profile(command, output_dir)
        else:
            run_nsight_profile(command, output_dir, expect_metrics=False)
    assert (output_dir / "metrics.yaml").exists() is expect_metrics
    arguments = run.call_args.args[0]
    staging_dir = Path(run.call_args.kwargs["env"][NSIGHT_STAGING_DIR_ENV])
    assert staging_dir.parent == output_dir.resolve() / ".nsight"
    assert arguments == [
        "/nsight path/nsys",
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
    assert command == original_command
    environment = run.call_args.kwargs["env"]
    assert environment[NSIGHT_OUTPUT_DIR_ENV] == str(output_dir.resolve())
    assert environment["PYTHONPATH"] == os.pathsep.join(imports)
    assert environment["NUREC_TEST_PARENT_VALUE"] == "preserved"
    assert run.call_args.kwargs["stderr"] == subprocess.STDOUT
    assert run.call_args.kwargs["check"] is False
    assert (output_dir / "performance-capture.log").read_text() == "captured stdout and stderr\n"


def test_missing_nsight_reports_requirement(tmp_path: Path) -> None:
    with (
        patch("nurec.utils.io.export.render_nsight.shutil.which", return_value=None),
        patch("nurec.utils.io.export.render_nsight.subprocess.run") as run,
        pytest.raises(RuntimeError, match="--nsight requires NVIDIA Nsight Systems"),
    ):
        run_nsight_profile(["render"], tmp_path)
    run.assert_not_called()


@pytest.mark.parametrize("log_text", ["profiler failed\n", "Error: first\nnoise\nError: final validation\n"])
def test_failed_child_reports_log_and_last_validation_error(tmp_path: Path, log_text: str) -> None:
    def capture(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        kwargs["stdout"].write(log_text)
        return subprocess.CompletedProcess(arguments, 2)

    with (
        patch("nurec.utils.io.export.render_nsight.shutil.which", return_value="/nsys"),
        patch("nurec.utils.io.export.render_nsight.subprocess.run", side_effect=capture),
        pytest.raises(RuntimeError) as error,
    ):
        run_nsight_profile(["render"], tmp_path)
    assert str(tmp_path / "performance-capture.log") in str(error.value)
    assert "Error: first" not in str(error.value)
    if "Error:" in log_text:
        assert str(error.value).endswith("Error: final validation")


@pytest.mark.parametrize(
    ("missing_name", "expect_metrics"),
    [
        (f"color/front/{NSIGHT_REPORT}", True),
        (f"color/rear/{NSIGHT_REPORT}", True),
        (NSIGHT_CAPTURE_MANIFEST, True),
        ("metrics.yaml", True),
        (f"color/front/{NSIGHT_REPORT}", False),
        (f"color/rear/{NSIGHT_REPORT}", False),
        (NSIGHT_CAPTURE_MANIFEST, False),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_empty_capture_outputs_fail(
    tmp_path: Path, missing_name: str, empty: bool, expect_metrics: bool
) -> None:
    manifest = json.dumps(
        [{"camera": camera, "frame": 4, "trace": f"color/{camera}/{NSIGHT_REPORT}"} for camera in ("front", "rear")]
    )
    # Old frame reports and sensor-local metrics cannot replace the required outputs.
    sensor_dir = tmp_path / "color" / "front"
    sensor_dir.mkdir(parents=True)
    (sensor_dir / "metrics.yaml").write_text("unrelated report")
    (sensor_dir / "frame-4.nsys-rep").write_bytes(b"old per-frame trace")
    for name in (
        f"color/front/{NSIGHT_REPORT}",
        f"color/rear/{NSIGHT_REPORT}",
        NSIGHT_CAPTURE_MANIFEST,
        "metrics.yaml",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name != missing_name:
            path.write_text(manifest if name == NSIGHT_CAPTURE_MANIFEST else "nonempty")
        elif empty:
            path.touch()
    with (
        patch("nurec.utils.io.export.render_nsight.shutil.which", return_value="/nsys"),
        patch(
            "nurec.utils.io.export.render_nsight.subprocess.run",
            return_value=subprocess.CompletedProcess(["render"], 0),
        ),
        pytest.raises(RuntimeError) as error,
    ):
        run_nsight_profile(["render"], tmp_path, expect_metrics=expect_metrics)
    assert str(tmp_path / missing_name) in str(error.value)
    assert str(tmp_path / "performance-capture.log") in str(error.value)


def test_sensor_capture_spans_all_frames_and_iterations_before_export(
    tmp_path: Path, cuda_profiler: CudaProfilerMocks
) -> None:
    capture = NsightCapture(tmp_path)
    calls: list[str | tuple[str, int, int]] = []
    expected: list[str | tuple[str, int, int]] = ["sync", "start", "model initialization"]
    sensors = [("front", False), ("rear", False), ("pandar128", True)]

    def export() -> None:
        calls.append("stop")
        # Nsight's %n picks the first unused filename, including empty reservations.
        index = next(
            index for index in range(1, 5) if not (staging_of(capture) / f"performance-{index}.nsys-rep").exists()
        )
        assert index == capture.capture_index
        (staging_of(capture) / f"performance-{index}.nsys-rep").write_text(f"trace {index}")

    synchronize, start, stop = cuda_profiler
    synchronize.side_effect = lambda: calls.append("sync")
    start.side_effect = lambda: calls.append("start")
    stop.side_effect = export
    with patch.object(capture, "write_manifest", wraps=capture.write_manifest) as write_manifest:
        capture.stop()
        capture.start()
        capture.start()
        calls.append("model initialization")
        for index, (sensor_id, lidar) in enumerate(sensors, 1):
            sensor_dir = tmp_path / ("lidar" if lidar else "color") / sensor_id
            if index > 1:
                expected.extend(["sync", "start"])
            with capture.sensor(sensor_id, sensor_dir=sensor_dir, lidar=lidar):
                capture.start()
                calls.append("warmup")
                expected.append("warmup")
                for frame in (4, 9):
                    with capture.frame(sensor_id, frame, lidar=lidar):
                        for iteration in (1, 2, 3):
                            calls.append((sensor_id, frame, iteration))
                            expected.append((sensor_id, frame, iteration))
                    assert capture.active
                    assert calls == expected
                    assert stop.call_count == index - 1
                    assert write_manifest.call_count == index - 1
                    assert not (sensor_dir / NSIGHT_REPORT).exists()
                    assert not (staging_of(capture) / f"performance-{index}.nsys-rep").exists()
                assert not sensor_dir.exists()
            expected.append("stop")
            assert calls == expected
            assert not capture.active
            assert stop.call_count == index
            assert write_manifest.call_count == index
            assert write_manifest.call_args.kwargs == {"final": False}
            assert (sensor_dir / NSIGHT_REPORT).read_text() == f"trace {index}"
            assert (staging_of(capture) / f"performance-{index}.nsys-rep").read_bytes() == b""
            assert json.loads((tmp_path / NSIGHT_CAPTURE_MANIFEST).read_text()) == capture.records
        capture.stop()
        capture.write_manifest()
    assert capture.capture_index == 3
    assert capture.records == [
        {
            "lidar" if lidar else "camera": sensor_id,
            "frame": frame,
            "trace": f"{'lidar' if lidar else 'color'}/{sensor_id}/{NSIGHT_REPORT}",
        }
        for sensor_id, lidar in sensors
        for frame in (4, 9)
    ]
    assert not (tmp_path / ".nsight").exists()


def test_launcher_isolates_staging_from_preserved_previous_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cuda_profiler: CudaProfilerMocks
) -> None:
    synchronize, start, stop = cuda_profiler
    staging_root = tmp_path / ".nsight"
    staging_root.mkdir()
    previous = staging_root / "performance-1.nsys-rep"
    previous.write_bytes(b"previous unassigned trace")
    created_staging: list[Path] = []

    def run_child(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        stage = Path(kwargs["env"][NSIGHT_STAGING_DIR_ENV])
        created_staging.append(stage)
        assert stage.parent == staging_root
        assert stage.is_dir()
        with monkeypatch.context() as child_env:
            child_env.setenv(NSIGHT_STAGING_DIR_ENV, str(stage))
            capture = NsightCapture(tmp_path)
            stop.side_effect = lambda: (stage / f"performance-{capture.capture_index}.nsys-rep").write_bytes(
                b"current trace"
            )
            capture.start()
            with capture.sensor("front", sensor_dir=tmp_path / "color" / "front"):
                for frame in (0, 5):
                    with capture.frame("front", frame):
                        pass
            assert capture.capture_index == 1
            capture.write_manifest()
        assert not stage.exists()
        assert previous.read_bytes() == b"previous unassigned trace"
        return subprocess.CompletedProcess(arguments, 0)

    with (
        patch("nurec.utils.io.export.render_nsight.shutil.which", return_value="/nsys"),
        patch("nurec.utils.io.export.render_nsight.subprocess.run", side_effect=run_child),
    ):
        for _ in range(2):
            run_nsight_profile(["render", "--no-report"], tmp_path, expect_metrics=False)
    assert created_staging[0] != created_staging[1]
    assert synchronize.call_count == start.call_count == 2
    assert list(staging_root.iterdir()) == [previous]
    assert (tmp_path / "color" / "front" / NSIGHT_REPORT).read_bytes() == b"current trace"


def test_disabled_capture_is_a_noop(tmp_path: Path, cuda_profiler: CudaProfilerMocks) -> None:
    capture = NsightCapture(None)
    synchronize, start, stop = cuda_profiler
    capture.start()
    with capture.sensor("front", sensor_dir=tmp_path / "color" / "front"):
        for frame in (4, 9):
            with capture.frame("front", frame):
                pass
    with capture.sensor("pandar128", sensor_dir=tmp_path / "lidar" / "pandar128", lidar=True):
        with capture.frame("pandar128", 12, lidar=True):
            pass
    capture.stop()
    capture.write_manifest()
    synchronize.assert_not_called()
    start.assert_not_called()
    stop.assert_not_called()
    assert not capture.active
    assert capture.records == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("lidar", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("exported", [False, True])
def test_interrupted_sensor_preserves_frame_records_and_partial_report(
    tmp_path: Path, lidar: bool, error_type: type[BaseException], exported: bool, cuda_profiler: CudaProfilerMocks
) -> None:
    capture = NsightCapture(tmp_path)
    sensor_dir = tmp_path / ("lidar/pandar128" if lidar else "color/front")
    sensor_id = "pandar128" if lidar else "front"
    error = error_type("render interrupted")

    def export() -> None:
        if exported:
            (staging_of(capture) / "performance-1.nsys-rep").write_bytes(b"partial trace")

    _, _, stop = cuda_profiler
    stop.side_effect = export
    with pytest.raises(error_type) as caught:
        with capture.sensor(sensor_id, sensor_dir=sensor_dir, lidar=lidar):
            with capture.frame(sensor_id, 4, lidar=lidar):
                pass
            with capture.frame(sensor_id, 9, lidar=lidar):
                stop.assert_not_called()
                raise error
    assert caught.value is error
    assert not capture.active
    capture.stop()
    capture.write_manifest()
    stop.assert_called_once_with()
    trace = (sensor_dir.relative_to(tmp_path) / NSIGHT_REPORT).as_posix()
    assert json.loads((tmp_path / NSIGHT_CAPTURE_MANIFEST).read_text()) == [
        {"lidar" if lidar else "camera": sensor_id, "frame": 4, "trace": trace},
        {"lidar" if lidar else "camera": sensor_id, "frame": 9, "trace": trace, "complete": False},
    ]
    if exported:
        assert (sensor_dir / NSIGHT_REPORT).read_bytes() == b"partial trace"
    else:
        assert not sensor_dir.exists()
    assert not (tmp_path / ".nsight").exists()


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, RuntimeError])
@pytest.mark.parametrize("exported", [False, True])
def test_failed_sensor_export_retains_original_error_and_any_report(
    tmp_path: Path, error_type: type[BaseException], exported: bool, cuda_profiler: CudaProfilerMocks
) -> None:
    capture = NsightCapture(tmp_path)
    sensor_dir = tmp_path / "color" / "front"
    error = error_type("stop interrupted")

    def export() -> None:
        if exported:
            (staging_of(capture) / "performance-1.nsys-rep").write_bytes(b"exported before interruption")
        raise error

    _, _, stop = cuda_profiler
    stop.side_effect = export
    with pytest.raises(error_type) as caught:
        with capture.sensor("front", sensor_dir=sensor_dir):
            with capture.frame("front", 4):
                pass
    assert caught.value is error
    assert not capture.active
    capture.stop()
    capture.write_manifest()
    stop.assert_called_once_with()
    assert json.loads((tmp_path / NSIGHT_CAPTURE_MANIFEST).read_text()) == [
        {"camera": "front", "frame": 4, "trace": f"color/front/{NSIGHT_REPORT}"}
    ]
    if exported:
        assert (sensor_dir / NSIGHT_REPORT).read_bytes() == b"exported before interruption"
    else:
        assert not sensor_dir.exists()


@pytest.mark.parametrize("empty", [False, True])
def test_missing_report_is_detected_at_sensor_exit_not_between_frames(
    tmp_path: Path, empty: bool, cuda_profiler: CudaProfilerMocks
) -> None:
    capture = NsightCapture(tmp_path)

    def export() -> None:
        if empty:
            (staging_of(capture) / "performance-1.nsys-rep").touch()

    _, _, stop = cuda_profiler
    stop.side_effect = export
    with pytest.raises(RuntimeError, match="performance-1.nsys-rep"):
        with capture.sensor("front", sensor_dir=tmp_path / "color" / "front"):
            for frame in (4, 9):
                with capture.frame("front", frame):
                    pass
                stop.assert_not_called()
    assert not capture.active
    assert len(capture.records) == 2


def test_final_cleanup_preserves_unassigned_initialization_report(
    tmp_path: Path, cuda_profiler: CudaProfilerMocks
) -> None:
    capture = NsightCapture(tmp_path)
    _, _, stop = cuda_profiler
    stop.side_effect = lambda: (staging_of(capture) / "performance-1.nsys-rep").write_bytes(b"initialization trace")
    capture.start()
    capture.stop()
    (staging_of(capture) / "performance-2.nsys-rep").touch()
    capture.write_manifest()
    assert (staging_of(capture) / "performance-1.nsys-rep").read_bytes() == b"initialization trace"
    assert not (staging_of(capture) / "performance-2.nsys-rep").exists()
    assert json.loads((tmp_path / NSIGHT_CAPTURE_MANIFEST).read_text()) == []


@pytest.mark.parametrize("misuse", ["no sensor", "wrong sensor", "wrong type", "nested sensor", "stopped sensor"])
def test_frame_requires_matching_active_sensor_and_sensor_scopes_cannot_nest(
    tmp_path: Path, misuse: str, cuda_profiler: CudaProfilerMocks
) -> None:
    capture = NsightCapture(tmp_path)
    sensor_dir = tmp_path / "color" / "front"
    _, _, stop = cuda_profiler
    stop.side_effect = lambda: (staging_of(capture) / "performance-1.nsys-rep").write_bytes(b"trace")
    if misuse == "no sensor":
        with pytest.raises(RuntimeError, match="active matching sensor"):
            with capture.frame("front", 4):
                pass
    else:
        with capture.sensor("front", sensor_dir=sensor_dir):
            with pytest.raises(RuntimeError):
                if misuse == "nested sensor":
                    with capture.sensor("rear", sensor_dir=tmp_path / "color" / "rear"):
                        pass
                else:
                    if misuse == "stopped sensor":
                        capture.stop()
                    with capture.frame(
                        "rear" if misuse == "wrong sensor" else "front", 4, lidar=misuse == "wrong type"
                    ):
                        pass
    assert capture.records == []


@pytest.mark.parametrize("same_sensor", [False, True])
def test_reopened_sensors_and_shared_destinations_are_rejected(
    tmp_path: Path, same_sensor: bool, cuda_profiler: CudaProfilerMocks
) -> None:
    capture = NsightCapture(tmp_path)
    sensor_dir = tmp_path / "color" / "front"
    _, start, stop = cuda_profiler
    stop.side_effect = lambda: (staging_of(capture) / "performance-1.nsys-rep").write_bytes(b"trace")
    with capture.sensor("front", sensor_dir=sensor_dir):
        with capture.frame("front", 4):
            pass
    with pytest.raises(RuntimeError):
        with capture.sensor("front" if same_sensor else "rear", sensor_dir=sensor_dir):
            pass
    start.assert_called_once_with()
    assert (sensor_dir / NSIGHT_REPORT).read_bytes() == b"trace"


@pytest.mark.parametrize("destination", ["root", "outside"])
def test_sensor_report_directory_must_be_inside_output_subdirectory(tmp_path: Path, destination: str) -> None:
    capture = NsightCapture(tmp_path)
    sensor_dir = tmp_path if destination == "root" else tmp_path.parent / "outside"
    with (
        patch("nurec.utils.io.export.render_nsight.torch.cuda.profiler.start") as start,
        pytest.raises(ValueError),
    ):
        with capture.sensor("front", sensor_dir=sensor_dir):
            pass
    start.assert_not_called()


@pytest.mark.parametrize(
    "manifest",
    [
        "not JSON",
        "{}",
        "[]",
        "[null]",
        "[1]",
        "[[]]",
        *[
            json.dumps([{"camera": "front", "frame": 4, "trace": trace}])
            for trace in (
                "../performance.nsys-rep",
                "/performance.nsys-rep",
                "performance.nsys-rep",
                "color/front/frame-4.nsys-rep",
                None,
            )
        ],
        *[
            json.dumps([{"camera": "front", "frame": frame, "trace": "color/front/performance.nsys-rep"}])
            for frame in (-1, "4", 4.0, True, None)
        ],
        '[{"camera": "front", "frame": 4}]',
        '[{"camera": "front", "trace": "color/front/performance.nsys-rep"}]',
        '[{"frame": 4, "trace": "color/front/performance.nsys-rep"}]',
        '[{"camera": "front", "lidar": "front", "frame": 4, "trace": "color/front/performance.nsys-rep"}]',
        '[{"camera": [], "frame": 4, "trace": "color/front/performance.nsys-rep"}]',
        '[{"camera": "", "frame": 4, "trace": "color/front/performance.nsys-rep"}]',
        json.dumps(
            [
                {"camera": "front", "frame": 4, "trace": "color/front/performance.nsys-rep"},
                {"camera": "front", "frame": 9, "trace": "color/rear/performance.nsys-rep"},
            ]
        ),
        json.dumps(
            [
                {"camera": "front", "frame": 4, "trace": "color/front/performance.nsys-rep"},
                {"camera": "rear", "frame": 4, "trace": "color/front/performance.nsys-rep"},
            ]
        ),
        json.dumps(
            [
                {"camera": "front", "frame": 4, "trace": "color/front/performance.nsys-rep"},
                {"lidar": "front", "frame": 4, "trace": "color/front/performance.nsys-rep"},
            ]
        ),
    ],
)
def test_invalid_capture_manifest_fails(tmp_path: Path, manifest: str) -> None:
    (tmp_path / NSIGHT_CAPTURE_MANIFEST).write_text(manifest)
    with (
        patch("nurec.utils.io.export.render_nsight.shutil.which", return_value="/nsys"),
        patch(
            "nurec.utils.io.export.render_nsight.subprocess.run",
            return_value=subprocess.CompletedProcess(["render"], 0),
        ),
        pytest.raises(RuntimeError, match="Invalid Nsight capture manifest"),
    ):
        run_nsight_profile(["render"], tmp_path)
