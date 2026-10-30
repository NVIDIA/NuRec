# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nurec/utils/io/export/render.py."""

import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

from collections import OrderedDict
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import torch

from click.testing import CliRunner
from PIL import Image

from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType
from nurec.render import ActorsSnapshot, CameraFrame, PoseRange
from nurec.render.utils import transform_intrinsics_to_resolution
from nurec.utils.io.export.render import (
    _apply_precomputed_bbox_overlay,
    _compute_bbox_overlay,
    _save_semantics_legend,
    export_videos_and_json,
)
from nurec.utils.io.export.render_helpers import CameraOutputSubdir
from nurec.utils.visualize import build_semantics_class_color_lut


_W, _H = 64, 48
_GREEN = (0, 255, 0)


class TestSaveSemanticsLegend(unittest.TestCase):
    def test_legend_file_created_with_class_names(self):
        lut = build_semantics_class_color_lut(3)
        with tempfile.TemporaryDirectory() as tmp:
            _save_semantics_legend(tmp, lut, {0: "bg", 1: "car", 2: "ped"})
            legend_path = os.path.join(tmp, CameraOutputSubdir.SEMANTICS_COLORIZED, "semantic_classes.png")
            self.assertTrue(os.path.exists(legend_path), f"Expected legend at {legend_path}")

    def test_legend_file_created_without_class_names(self):
        lut = build_semantics_class_color_lut(5)
        with tempfile.TemporaryDirectory() as tmp:
            _save_semantics_legend(tmp, lut, None)
            legend_path = os.path.join(tmp, CameraOutputSubdir.SEMANTICS_COLORIZED, "semantic_classes.png")
            self.assertTrue(os.path.exists(legend_path))

    def test_legend_file_created_with_sparse_class_names(self):
        """Sparse id->name mapping: unnamed ids within the LUT fall back to numeric labels."""
        lut = build_semantics_class_color_lut(5)
        with tempfile.TemporaryDirectory() as tmp:
            _save_semantics_legend(tmp, lut, {0: "bg", 4: "ped"})
            legend_path = os.path.join(tmp, CameraOutputSubdir.SEMANTICS_COLORIZED, "semantic_classes.png")
            self.assertTrue(os.path.exists(legend_path))

    def test_legend_raises_when_named_id_exceeds_lut(self):
        """A class name with id >= color LUT size must raise (LUT cannot color that class)."""
        lut = build_semantics_class_color_lut(3)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _save_semantics_legend(tmp, lut, {0: "bg", 5: "car"})

    def test_legend_file_is_readable_png(self):
        from PIL import Image

        lut = build_semantics_class_color_lut(2)
        with tempfile.TemporaryDirectory() as tmp:
            _save_semantics_legend(tmp, lut, {0: "a", 1: "b"})
            legend_path = os.path.join(tmp, CameraOutputSubdir.SEMANTICS_COLORIZED, "semantic_classes.png")
            img = Image.open(legend_path)
            self.assertEqual(img.mode, "RGB")
            self.assertGreater(img.width, 0)
            self.assertGreater(img.height, 0)


def _pose_range() -> PoseRange:
    identity_tquat = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return PoseRange(
        start_pose_tquat_sensor_world=identity_tquat,
        end_pose_tquat_sensor_world=identity_tquat,
        start_timestamp_us=0,
        end_timestamp_us=100,
    )


def _one_actor_snapshot() -> ActorsSnapshot:
    actor_poses = torch.tensor(
        [[[0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    return ActorsSnapshot(
        actor_ids=["a"],
        actor_poses=actor_poses,
        cuboid_dims=torch.tensor([[2.0, 2.0, 2.0]]),
    )


class TestComputeBboxOverlay(unittest.TestCase):
    """Tests for _compute_bbox_overlay."""

    def test_none_mode_returns_none(self):
        result = _compute_bbox_overlay(
            render_bboxes="none",
            overlay_actors_snapshot=_one_actor_snapshot(),
            camera_model_parameters=MagicMock(),
            camera_to_world=_pose_range(),
            image_width=_W,
            image_height=_H,
            bbox_color=_GREEN,
        )
        self.assertIsNone(result)

    def test_no_snapshot_returns_none(self):
        result = _compute_bbox_overlay(
            render_bboxes="model",
            overlay_actors_snapshot=None,
            camera_model_parameters=MagicMock(),
            camera_to_world=_pose_range(),
            image_width=_W,
            image_height=_H,
            bbox_color=_GREEN,
        )
        self.assertIsNone(result)

    def test_empty_snapshot_returns_none(self):
        result = _compute_bbox_overlay(
            render_bboxes="model",
            overlay_actors_snapshot=ActorsSnapshot.empty(),
            camera_model_parameters=MagicMock(),
            camera_to_world=_pose_range(),
            image_width=_W,
            image_height=_H,
            bbox_color=_GREEN,
        )
        self.assertIsNone(result)

    @patch("nurec.utils.io.export.render.overlay_bounding_boxes")
    @patch("nurec.utils.io.export.render.transform_intrinsics_to_resolution")
    def test_with_actor_returns_drawn_and_mask(self, mock_transform, mock_overlay):
        """With an actor present, returns (drawn, mask) derived from overlay_bounding_boxes output."""
        drawn = np.zeros((_H, _W, 3), dtype=np.uint8)
        drawn[10:20, 10:20] = _GREEN
        mock_overlay.return_value = drawn
        mock_transform.return_value = MagicMock()

        result = _compute_bbox_overlay(
            render_bboxes="artifact",
            overlay_actors_snapshot=_one_actor_snapshot(),
            camera_model_parameters=MagicMock(),
            camera_to_world=_pose_range(),
            image_width=_W,
            image_height=_H,
            bbox_color=_GREEN,
        )

        self.assertIsNotNone(result)
        drawn_out, mask_out = result
        np.testing.assert_array_equal(drawn_out, drawn)
        np.testing.assert_array_equal(mask_out, drawn.any(axis=-1))

    @patch("nurec.utils.io.export.render.overlay_bounding_boxes")
    @patch("nurec.utils.io.export.render.transform_intrinsics_to_resolution")
    def test_blank_overlay_produces_all_false_mask(self, mock_transform, mock_overlay):
        """overlay_bounding_boxes returning all-zeros produces a mask that is entirely False."""
        mock_overlay.return_value = np.zeros((_H, _W, 3), dtype=np.uint8)
        mock_transform.return_value = MagicMock()

        result = _compute_bbox_overlay(
            render_bboxes="model",
            overlay_actors_snapshot=_one_actor_snapshot(),
            camera_model_parameters=MagicMock(),
            camera_to_world=_pose_range(),
            image_width=_W,
            image_height=_H,
            bbox_color=_GREEN,
        )

        self.assertIsNotNone(result)
        _, mask_out = result
        self.assertFalse(mask_out.any())


class TestApplyPrecomputedBboxOverlay(unittest.TestCase):
    """Tests for _apply_precomputed_bbox_overlay."""

    def test_none_overlay_returns_copy_unchanged(self):
        img = np.zeros((_H, _W, 3), dtype=np.uint8)
        img[5, 5] = (100, 100, 100)
        result = _apply_precomputed_bbox_overlay(img, None)
        np.testing.assert_array_equal(result, img)
        self.assertIsNot(result, img)

    def test_overlay_applied_at_mask_positions(self):
        img = np.zeros((_H, _W, 3), dtype=np.uint8)
        drawn = np.zeros((_H, _W, 3), dtype=np.uint8)
        drawn[10, 10] = _GREEN
        mask = drawn.any(axis=-1)

        result = _apply_precomputed_bbox_overlay(img, (drawn, mask))

        self.assertEqual(tuple(result[10, 10]), _GREEN)
        np.testing.assert_array_equal(result[~mask], img[~mask])

    def test_does_not_mutate_input(self):
        img = np.zeros((_H, _W, 3), dtype=np.uint8)
        drawn = np.zeros((_H, _W, 3), dtype=np.uint8)
        drawn[5, 5] = _GREEN
        mask = drawn.any(axis=-1)

        _ = _apply_precomputed_bbox_overlay(img, (drawn, mask))

        np.testing.assert_array_equal(img, np.zeros((_H, _W, 3), dtype=np.uint8))


class TestRenderBenchmarkCli(unittest.TestCase):
    """Exercise the real CLI loop with deterministic timings and a reused framebuffer."""

    def run_render(
        self,
        options=(),
        *,
        cameras=("cam0",),
        frame_count=2,
        durations=None,
        wall_durations=None,
        output_root=None,
        environment=None,
        nsight_error=None,
        lidar_error=None,
        load_error=None,
        render_error=None,
        trajectory_count=1,
        invalid_offsets=False,
        use_default_training_views=False,
        output_error=None,
        rig_parser_error=None,
        custom_loader_error=None,
        malformed_rigs=False,
        native_resolutions=None,
        custom_rigs=None,
    ):
        module = importlib.import_module("nurec.utils.io.export.render")
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        artifact_path = root / "scene.usdz"
        artifact_path.touch()  # Loading is mocked; no real artifact or GPU is needed.
        output = root / "output" if output_root is None else output_root
        calls = []
        trajectory_calls = []
        render_arguments = []
        validations = []
        saved = []
        copy_ranges = []
        warmup_ranges = []
        frame_ranges = []
        frame_state = SimpleNamespace(active=None)
        pinned_buffers = []
        pinned_copies = []
        copy_state = SimpleNamespace(active=None)
        synchronizations = []
        wall_clock = MagicMock()
        timing = SimpleNamespace(active=False, record_sample=False)
        timer_calls = []
        capture_state = SimpleNamespace(active=None, sensor=None, trace=None, started=False, lifecycle=[])
        capture_windows = []
        sensor_capture_windows = []
        duration_values = iter(durations if durations is not None else range(1, 100))

        class Timer:
            def __init__(self):
                self.milliseconds = []
                self.valid = []

            @contextmanager
            def __call__(self, device, *, label=None, record_sample=True):
                timer_calls.append((label, record_sample))
                with copy_range(label) if label is not None else nullcontext():
                    timing.active = True
                    timing.record_sample = record_sample
                    try:
                        yield
                    finally:
                        timing.active = False
                        timing.record_sample = False
                    if record_sample:
                        duration = next(duration_values)
                        self.milliseconds.append(0.0 if duration is None else float(duration))
                        self.valid.append(duration is not None)

        timer = Timer()
        original_validate = module.validate_camera_output

        def validate_output(camera_frame, resolution, render_options):
            self.assertFalse(timing.active, "Output validation belongs outside CUDA timing")
            self.assertEqual(wall_clock.call_count % 2, 0, "Output validation belongs outside wall timing")
            validations.append((len(calls), resolution))
            # Native rendering is simulated on CPU; preserve real output/shape validation
            # while advertising the device that production rendering must return.
            fields = {}
            for name in ("color_image", "distance_image", "semantic_image"):
                tensor = getattr(camera_frame, name)
                fields[name] = (
                    None
                    if tensor is None
                    else SimpleNamespace(shape=tensor.shape, device=torch.device("cuda"), ndim=tensor.ndim)
                )
            original_validate(SimpleNamespace(**fields), resolution, render_options)

        artifact = SimpleNamespace(
            semantic_class_names=None,
            semantic_class_colors=None,
            scene_id="test-scene",
            rig_trajectories={},
            custom_rig_trajectories={},
            load_custom_rig_trajectories=MagicMock(side_effect=custom_loader_error),
            parsed_config={"dataset": {"samplers": {}}, "model": {"renderer": {"name": "fixture-renderer"}}},
        )
        trajectories = []
        calibrations = OrderedDict()
        sensor_identities = {}
        for trajectory in range(trajectory_count):
            # Main supports one sequence; additional rig trajectories reuse its calibrated sensors.
            sequence = "test-sequence" if trajectory_count == 1 else "test-sequence-0"
            timestamps = {}
            starts = {}
            for unique_index, camera in enumerate(cameras):
                unique_id = f"{camera}@{sequence}"
                sensor_identities.setdefault(unique_index, (trajectory, camera))
                native_width, native_height = (native_resolutions or {}).get(camera, (4, 2))
                calibrations[unique_id] = SimpleNamespace(
                    camera_model_parameters=OpenCVPinholeCameraModelParameters(
                        resolution=np.array([native_width, native_height], dtype=np.uint64),
                        shutter_type=ShutterType.GLOBAL,
                        external_distortion_parameters=None,
                        principal_point=np.array([(native_width - 1) / 2, (native_height - 1) / 2], dtype=np.float32),
                        focal_length=np.array([native_width, native_width], dtype=np.float32),
                        radial_coeffs=np.zeros(6, dtype=np.float32),
                        tangential_coeffs=np.zeros(2, dtype=np.float32),
                        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
                    ),
                    unique_sensor_idx=unique_index,
                    logical_sensor_name=camera,
                    sequence_id=sequence,
                )
                start_us = trajectory * 10_000
                timestamps[unique_id] = torch.tensor(
                    [[start_us + frame * 100, start_us + frame * 100 + 99] for frame in range(frame_count)]
                )
                starts[unique_id] = unique_index * frame_count
            trajectories.append(
                module.RigTrajectories.RigTrajectory(
                    sequence_id=sequence,
                    rig_bbox=None,
                    cameras_linear_start_frame_indices=starts,
                    cameras_frame_timestamps_us=timestamps,
                    lidars_frame_timestamps_us={},
                    T_rig_worlds=torch.eye(4).repeat(2, 1, 1),
                    T_rig_world_timestamps_us=torch.tensor(
                        [trajectory * 10_000, trajectory * 10_000 + max(frame_count * 100, 1)]
                    ),
                )
            )
        if invalid_offsets:
            trajectories[-1].cameras_linear_start_frame_indices = None
        rigs = module.RigTrajectories(
            T_world_base=torch.eye(4),
            world_to_nurec=MagicMock(),
            rig_trajectories=trajectories,
            camera_calibrations=calibrations,
            lidar_calibrations=OrderedDict(),
        )

        original_parse = module.RigTrajectories.from_dict

        def parse_rigs(data):
            custom = data is artifact.custom_rig_trajectories
            if custom == ("--custom-rig-trajectory" in options):
                if rig_parser_error is not None:
                    raise rig_parser_error
                if malformed_rigs:
                    return original_parse(data)
            return custom_rigs if custom and custom_rigs is not None else rigs

        rig_parser = MagicMock(side_effect=parse_rigs)

        def camera_provider(selected_rigs, *args, **kwargs):
            trajectory = selected_rigs.rig_trajectories[0]
            provider = MagicMock()
            provider.get_available_camera_ids.return_value = list(cameras)
            provider.get_num_camera_frames.return_value = frame_count
            provider.get_unique_sensor_id.side_effect = lambda camera: f"{camera}@{trajectory.sequence_id}"
            provider.get_camera_calibration.side_effect = lambda camera: selected_rigs.camera_calibrations[
                f"{camera}@{trajectory.sequence_id}"
            ]
            provider.get_unique_frame_index.side_effect = lambda camera, frame: module.get_unique_frame_index(
                camera, frame, trajectory.cameras_linear_start_frame_indices, trajectory.sequence_id
            )
            provider.get_camera_view_pose.side_effect = lambda camera, frame: replace(
                _pose_range(), start_timestamp_us=frame * 100, end_timestamp_us=frame * 100 + 99
            )
            return provider

        camera_provider_factory = MagicMock(side_effect=camera_provider)

        def sensor_provider(*, camera_calibrations, **kwargs):
            sequence = next(iter(camera_calibrations.values())).sequence_id
            index = next(i for i, trajectory in enumerate(trajectories) if trajectory.sequence_id == sequence)
            provider = camera_provider(
                SimpleNamespace(rig_trajectories=[trajectories[index]], camera_calibrations=camera_calibrations)
            )
            provider.get_unique_frame_index.side_effect = lambda camera, frame: module.get_unique_frame_index(
                camera, frame, kwargs["cameras_start_frame_indices"], sequence
            )
            return provider

        sensor_provider_factory = MagicMock(side_effect=sensor_provider)
        original_mapping = module.build_training_frame_mapping

        def build_mapping(config, trajectory):
            load.assert_called_once()
            return original_mapping(config, trajectory)

        training_mapping = MagicMock(side_effect=build_mapping)
        framebuffer = torch.empty((2, 4, 3))

        def native_render(**kwargs):
            trajectory, camera = sensor_identities[kwargs["unique_sensor_idx"]]
            frame = kwargs["frame_start_us"] // 100
            if nsight_factory.call_args.args[0] is not None:
                self.assertTrue(capture_state.started, "The continuous capture must cover every render")
                self.assertEqual(capture_state.active[0], camera)
                self.assertEqual(frame, capture_state.active[1])
            self.assertTrue(timing.active, "Warmup and measured renders must both use CUDA event timing")
            measured = frame_state.active is not None
            if measured:
                self.assertEqual(frame_state.active, f"nurec.render.frame(camera={camera}, frame={frame})")
            else:
                self.assertFalse(timing.record_sample, "Warmups must not publish timing samples")
            calls.append((camera, frame, measured))
            trajectory_calls.append((trajectory, camera, frame, measured))
            render_arguments.append(kwargs)
            if render_error is not None:
                raise render_error
            # Use the same CPU intrinsics normalization as the real native renderer.
            intrinsics = transform_intrinsics_to_resolution(kwargs["camera_intrinsics"], kwargs["resolution"])
            # Later renders overwrite the same storage: retained extrema must be copied.
            width, height = intrinsics.resolution.astype(int).tolist()
            framebuffer.resize_(height, width, 3).fill_(len(calls) / 32)
            if output_error == "missing":
                return CameraFrame()
            if output_error == "wrong-size":
                return CameraFrame(color_image=framebuffer[:1])
            return CameraFrame(color_image=framebuffer)

        renderable = SimpleNamespace(
            device=torch.device("cpu"),
            render_camera_frame=native_render,
            model_class_name=lambda: "MockRenderable",
            _model=object(),
            get_camera_trajectories=MagicMock(return_value={key: object() for key in calibrations}),
        )
        metrics = MagicMock()
        collectors = []

        def make_metrics(**kwargs):
            collector = metrics if not collectors else MagicMock()
            collector.metadata_values = {}
            collector.set_metric_metadata.side_effect = lambda name, **metadata: collector.metadata_values.setdefault(
                name, {}
            ).update(metadata)
            collectors.append(collector)
            return collector

        metrics_factory = MagicMock(side_effect=make_metrics)
        timer_factory = MagicMock(return_value=timer)
        nsight_helper = MagicMock(side_effect=nsight_error)
        nsight_capture = MagicMock()
        nsight_capture.records = []
        nsight_factory = MagicMock(return_value=nsight_capture)
        lidar_helper = MagicMock(side_effect=lidar_error)
        executor = MagicMock()
        executor.__enter__.return_value = executor

        def start_capture():
            self.assertFalse(timing.active, "Profiler startup belongs outside render timing")
            self.assertEqual(wall_clock.call_count % 2, 0)
            if nsight_factory.call_args.args[0] is not None and not capture_state.started:
                capture_state.started = True
                capture_state.lifecycle.append("capture-start")

        def stop_capture():
            self.assertFalse(timing.active, "Profiler completion belongs outside render timing")
            if capture_state.started:
                capture_state.started = False
                capture_state.lifecycle.append("capture-stop")

        def load_model(*args, **kwargs):
            if kwargs.get("before_model_load") is not None:
                kwargs["before_model_load"]()
            if nsight_factory.call_args.args[0] is not None:
                self.assertTrue(capture_state.started, "Nsight must start before model initialization")
            capture_state.lifecycle.append("load-model")
            if load_error is not None:
                raise load_error
            return renderable

        @contextmanager
        def capture_sensor(camera_id, *, sensor_dir: Path, lidar: bool = False):
            self.assertIsInstance(sensor_dir, Path)
            if nsight_factory.call_args.args[0] is None:
                yield
                return
            self.assertIsNone(capture_state.sensor)
            self.assertFalse(timing.active)
            nsight_capture.start()
            start = len(calls)
            capture_state.sensor = camera_id
            capture_state.trace = (
                (sensor_dir / "performance.nsys-rep").relative_to(nsight_factory.call_args.args[0]).as_posix()
            )
            try:
                yield
            finally:
                self.assertIsNone(capture_state.active)
                self.assertFalse(timing.active)
                sensor_capture_windows.append((camera_id, start, len(calls)))
                nsight_capture.stop()
                capture_state.sensor = None
                capture_state.trace = None
                nsight_capture.write_manifest(final=False)

        @contextmanager
        def capture_frame(camera_id, frame: int, *, lidar: bool = False):
            self.assertIsInstance(frame, int)
            if nsight_factory.call_args.args[0] is None:
                yield
                return
            self.assertFalse(timing.active, "Capture setup belongs outside render timing")
            self.assertIsNone(capture_state.active)
            self.assertEqual(capture_state.sensor, camera_id)
            self.assertTrue(capture_state.started)
            stops = nsight_capture.stop.call_count
            manifests = nsight_capture.write_manifest.call_count
            start = len(calls)
            capture_state.active = (camera_id, frame)
            complete = False
            try:
                yield
                self.assertEqual(wall_clock.call_count % 2, 0)
                complete = True
            finally:
                self.assertFalse(timing.active, "Capture completion belongs outside render timing")
                capture_state.active = None
                self.assertTrue(capture_state.started, "Frame boundaries must keep the camera capture active")
                self.assertEqual(nsight_capture.stop.call_count, stops, "Frames must not export a capture")
                self.assertEqual(nsight_capture.write_manifest.call_count, manifests)
                renders = calls[start:]
                self.assertTrue(renders)
                self.assertTrue(all(camera == camera_id and index == frame for camera, index, _ in renders))
                capture_windows.append((camera_id, frame, start, renders))
                record = {
                    "lidar" if lidar else "camera": camera_id,
                    "frame": frame,
                    "trace": capture_state.trace,
                }
                if not complete:
                    record["complete"] = False
                nsight_capture.records.append(record)

        def write_manifest(*, final=True):
            self.assertFalse(capture_state.started, "Stop the profiler before finalizing capture locations")
            self.assertIsNone(capture_state.active)
            self.assertFalse(timing.active)
            if nsight_factory.call_args.args[0] is not None:
                capture_state.lifecycle.append("write-manifest" if final else "write-sensor-manifest")

        nsight_capture.start.side_effect = start_capture
        nsight_capture.stop.side_effect = stop_capture
        nsight_capture.sensor.side_effect = capture_sensor
        nsight_capture.frame.side_effect = capture_frame
        nsight_capture.write_manifest.side_effect = write_manifest

        def submit(function, *args):
            future = Future()
            try:
                future.set_result(function(*args))
            except Exception as error:
                future.set_exception(error)
            return future

        executor.submit.side_effect = submit
        original_cpu = torch.Tensor.cpu
        original_to = torch.Tensor.to
        original_copy = torch.Tensor.copy_
        original_empty_like = torch.empty_like
        original_save = module._save_rendered_frame_to_disk

        @contextmanager
        def copy_range(label):
            if label.startswith("nurec.render.frame("):
                before = len(calls)
                self.assertFalse(timing.active)
                self.assertIsNone(frame_state.active)
                frame_state.active = label
                try:
                    yield
                finally:
                    self.assertFalse(timing.active)
                    self.assertTrue(all(timed for _, _, timed in calls[before:]))
                    frame_state.active = None
                    frame_ranges.append((label, before, calls[before:]))
                return
            if label.startswith("nurec.render.warmup("):
                before, measured = len(calls), len(timer.milliseconds)
                self.assertFalse(timing.active)
                self.assertIsNone(frame_state.active, "Warmup NVTX must be separate from the measured frame range")
                yield
                self.assertIsNone(frame_state.active)
                self.assertEqual(len(timer.milliseconds), measured, "Warmup must not record CUDA samples")
                self.assertEqual(len(calls), before + 1)
                self.assertFalse(calls[-1][2])
                warmup_ranges.append((label, before, len(calls)))
                return
            self.assertFalse(timing.active, "Image copy ranges belong outside CUDA timing")
            self.assertIsNone(copy_state.active)
            record = {"label": label, "after_render": len(calls), "operations": []}
            copy_ranges.append(record)
            copy_state.active = record
            try:
                yield
            finally:
                self.assertFalse(timing.active)
                copy_state.active = None

        def record_copy(operation):
            self.assertIsNotNone(copy_state.active, "Image transfers must have their own NVTX label")
            copy_state.active["operations"].append(operation)

        def cpu(tensor, *args, **kwargs):
            self.assertFalse(timing.active, "GPU output retrieval belongs outside CUDA timing")
            if tensor.ndim == 3 and tensor.shape[-1] == 3:
                self.assertEqual(tensor.dtype, torch.uint8, "Convert RGB before downloading through the native path")
            record_copy("cpu")
            return original_cpu(tensor, *args, **kwargs)

        def to(tensor, *args, **kwargs):
            if args and str(args[0]) == "cpu":
                self.assertFalse(timing.active, "Retained framebuffer copies belong outside CUDA timing")
                record_copy("to_cpu")
            elif args and args[0] == torch.uint8:
                record_copy("uint8")
            return original_to(tensor, *args, **kwargs)

        def empty_like(tensor, *args, **kwargs):
            if kwargs.get("pin_memory"):
                self.assertFalse(timing.active, "Pinned staging allocation belongs outside CUDA timing")
                self.assertEqual(str(kwargs.get("device")), "cpu")
                self.assertEqual(tensor.dtype, torch.uint8, "Stage converted RGB, not the floating-point render")
                self.assertIsNotNone(copy_state.active)
                # Emulate pinned storage with CPU tensors so these tests never require a GPU.
                buffer = original_empty_like(tensor, *args, **{**kwargs, "pin_memory": False})
                pinned_buffers.append(buffer)
                return buffer
            return original_empty_like(tensor, *args, **kwargs)

        def copy_(tensor, source, non_blocking=False):
            for index, buffer in enumerate(pinned_buffers):
                if tensor is buffer:
                    self.assertFalse(timing.active, "Pinned RGB downloads belong outside CUDA timing")
                    self.assertFalse(non_blocking, "The NumPy view must only read completed downloads")
                    self.assertEqual(source.dtype, torch.uint8)
                    self.assertEqual(tensor.dtype, torch.uint8)
                    record_copy("copy_cpu")
                    pinned_copies.append((index, len(calls)))
                    break
            return original_copy(tensor, source, non_blocking=non_blocking)

        def save(image, path):
            self.assertFalse(timing.active, "Image saving belongs outside CUDA timing")
            self.assertEqual(image.dtype, np.uint8, "Prepared images must not be rescaled during final saving")
            for buffer in pinned_buffers:
                self.assertFalse(np.shares_memory(image, buffer.numpy()), "Saved images must own their CPU storage")
            saved.append(Path(path))
            original_save(image, path)

        def synchronize(device):
            self.assertFalse(timing.active, "Warm-up completion belongs outside CUDA timing")
            self.assertIsNone(frame_state.active, "Drain warmup before entering the measured frame range")
            self.assertEqual(device, renderable.device)
            synchronizations.append((len(calls), wall_clock.call_count))

        with ExitStack() as stack:
            for target, attribute, value in (
                (module, "Artifact", MagicMock(return_value=artifact)),
                (module.RigTrajectories, "from_dict", rig_parser),
                (module, "SensorCalibProviderFromRigTrajectories", camera_provider_factory),
                (module, "SensorCalibProviderFromSensorTrajectories", sensor_provider_factory),
                (module, "build_training_frame_mapping", training_mapping),
                (module, "validate_camera_output", validate_output),
                (module, "get_trajectory_time_range", MagicMock(return_value=(0, 10_000))),
                (module, "load_manifest_from_usdz", MagicMock(return_value=None)),
                (module, "MetricsCollector", metrics_factory),
                (module, "CudaEventTimer", timer_factory),
                (module, "run_nsight_profile", nsight_helper),
                (module, "NsightCapture", nsight_factory),
                (module, "render_lidar_frames", lidar_helper),
                (module, "ThreadPoolExecutor", MagicMock(return_value=executor)),
                (module, "_save_rendered_frame_to_disk", save),
                (torch.Tensor, "cpu", cpu),
                (torch.Tensor, "to", to),
                (torch.Tensor, "copy_", copy_),
                (torch, "empty_like", empty_like),
                (torch.cuda.nvtx, "range", copy_range),
                (torch.cuda, "device", MagicMock(side_effect=lambda device: nullcontext())),
                (torch.cuda, "synchronize", synchronize),
            ):
                stack.enter_context(patch.object(target, attribute, value))
            if wall_durations is not None:
                wall_values = iter(
                    value for i, duration in enumerate(wall_durations) for value in (i, i + duration / 1000)
                )

                def read_wall_clock():
                    self.assertTrue(timing.active, "Wall-clock readings belong inside the CUDA timer context")
                    return next(wall_values)

                wall_clock.side_effect = read_wall_clock
                stack.enter_context(patch.object(module, "time", SimpleNamespace(perf_counter=wall_clock)))
            load = stack.enter_context(
                patch.object(module.RenderableModel, "load_from_artifact", side_effect=load_model)
            )
            result = CliRunner().invoke(
                module.render,
                [
                    "--artifact-path",
                    str(artifact_path),
                    "--output-dir",
                    str(output),
                    *(
                        ["--no-replicate-training-views"]
                        if "--lidar" not in options and not use_default_training_views
                        else []
                    ),
                    *options,
                ],
                env={module.NSIGHT_OUTPUT_DIR_ENV: None, **(environment or {})},
            )
        if metrics.write_metrics_to_yaml.called:
            run_output = Path(metrics.write_metrics_to_yaml.call_args.args[0])
        elif nsight_helper.called:
            run_output = nsight_helper.call_args.args[1]
        elif lidar_helper.called:
            run_output = lidar_helper.call_args.kwargs["output_dir"]
        else:
            run_output = output
        return SimpleNamespace(
            result=result,
            output=run_output,
            output_root=output,
            run_output=nsight_factory.call_args.args[0]
            if nsight_factory.called and nsight_factory.call_args.args[0] is not None
            else run_output,
            calls=calls,
            trajectory_calls=trajectory_calls,
            render_arguments=render_arguments,
            validations=validations,
            saved=saved,
            copy_ranges=copy_ranges,
            warmup_ranges=warmup_ranges,
            frame_ranges=frame_ranges,
            pinned_buffers=pinned_buffers,
            pinned_copies=pinned_copies,
            synchronizations=synchronizations,
            metrics=metrics,
            collectors=collectors,
            metrics_factory=metrics_factory,
            timer=timer,
            timer_calls=timer_calls,
            wall_clock=wall_clock,
            timer_factory=timer_factory,
            nsight_helper=nsight_helper,
            nsight_capture=nsight_capture,
            nsight_factory=nsight_factory,
            capture_windows=capture_windows,
            sensor_capture_windows=sensor_capture_windows,
            capture_lifecycle=capture_state.lifecycle,
            lidar_helper=lidar_helper,
            camera_provider_factory=camera_provider_factory,
            sensor_provider_factory=sensor_provider_factory,
            training_mapping=training_mapping,
            renderable=renderable,
            rigs=rigs,
            artifact=artifact,
            artifact_path=artifact_path,
            rig_parser=rig_parser,
            load=load,
        )

    def run_nsight_child(self, options=(), **kwargs):
        """Run the child path with an isolated capture directory and explicit Nsight marker."""
        module = importlib.import_module("nurec.utils.io.export.render")
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        parent = Path(directory.name)
        capture = parent / "capture"
        capture.mkdir()
        run = self.run_render(
            ["--nsight", *options],
            output_root=parent,
            environment={module.NSIGHT_OUTPUT_DIR_ENV: str(capture)},
            **kwargs,
        )
        return run, capture

    def assert_success(self, run):
        self.assertEqual(run.result.exit_code, 0, f"{run.result.output} {run.result.exception!r}")

    def assert_sensor_capture_lifecycle(self, run, cameras):
        expected = ["capture-start", "load-model"]
        for index, _ in enumerate(cameras):
            if index:
                expected.append("capture-start")
            expected.extend(["capture-stop", "write-sensor-manifest"])
        expected.append("write-manifest")
        self.assertEqual(run.capture_lifecycle, expected)
        self.assertEqual([entry[0] for entry in run.sensor_capture_windows], list(cameras))
        # Each sensor stops once; Click's final fallback stop is an idempotent no-op.
        self.assertEqual(run.nsight_capture.stop.call_args_list, [call()] * (len(cameras) + 1))
        self.assertEqual(
            run.nsight_capture.write_manifest.call_args_list,
            [call(final=False)] * len(cameras) + [call()],
        )

    def canonical_saved(self, run, camera=None):
        return [
            path
            for path in run.saved
            if path.parent.parent.name == CameraOutputSubdir.COLOR and (camera is None or path.parent.name == camera)
        ]

    def extra_saved(self, run, camera=None):
        return [
            path
            for path in run.saved
            if path.parent.parent.name == "color-extra" and (camera is None or path.parent.name == camera)
        ]

    def manifest(self, run, camera="cam0", *, extra=False):
        directory = "color-extra" if extra else CameraOutputSubdir.COLOR
        return json.loads((run.output / directory / camera / "timestamps.json").read_text())

    def assert_saved_outputs(self, run, expected, *, camera="cam0", extra=True):
        """Check explicit (filename, frame, iteration, render-call number, CUDA ms) rows."""
        saved = self.extra_saved(run, camera) if extra else self.canonical_saved(run, camera)
        entries = self.manifest(run, camera, extra=extra)
        names = [row[0] for row in expected]
        self.assertEqual([path.name for path in saved], names)
        self.assertEqual([entry["file_name"] for entry in entries], names)
        self.assertEqual([entry["render_frame_idx"] for entry in entries], list(range(len(expected))))
        for path, entry, (_, frame, iteration, call_number, cuda_ms) in zip(saved, entries, expected, strict=True):
            with Image.open(path) as image:
                self.assertEqual(image.getpixel((0, 0)), (int(255 * call_number / 32),) * 3)
            self.assertEqual(entry["source_frame_idx"], frame)
            self.assertEqual(entry["frame_start_timestamp_us"], frame * 100)
            self.assertEqual(entry["iteration"], iteration)
            self.assertEqual(entry["cuda_ms"], cuda_ms)

    def samples(self, run, name, camera="cam0"):
        return [
            call.kwargs[name] for call in run.metrics.record_frame.call_args_list if call.kwargs["sensor_id"] == camera
        ]

    def test_invalid_cuda_samples_preserve_paired_metrics_and_saved_outputs(self):
        for durations in ([None, 2.0, None, 4.0], [1.0, None, 3.0, None], [None] * 4):
            for mode in ("all-iterations", "last-iteration"):
                with self.subTest(durations=durations, mode=mode):
                    run = self.run_render(
                        ["--save", mode, "--iterations", "2"],
                        durations=durations,
                        wall_durations=[10, 20, 30, 40],
                    )
                    self.assert_success(run)
                    self.assertEqual(len(run.calls), 4)
                    self.assertEqual(len(run.validations), 4)
                    expected_ms = [0.0 if ms is None else ms for ms in durations]
                    expected_valid = [ms is not None for ms in durations]
                    self.assertEqual(self.samples(run, "cuda_ms"), expected_ms)
                    np.testing.assert_allclose(self.samples(run, "render_time_ms"), [10, 20, 30, 40])
                    self.assertEqual(
                        self.samples(run, "metric_validity"), [{"cuda_ms": valid} for valid in expected_valid]
                    )
                    self.assertEqual(
                        [
                            (call.kwargs["frame"], call.kwargs["iteration"])
                            for call in run.metrics.record_frame.call_args_list
                        ],
                        [(0, 1), (0, 2), (1, 1), (1, 2)],
                    )
                    saved_indices = list(range(4)) if mode == "all-iterations" else [1, 3]
                    self.assertEqual(
                        [path.name for path in self.extra_saved(run)],
                        [f"render-4x2-frame-{index // 2}-iteration-{index % 2 + 1}.png" for index in saved_indices],
                    )
                    entries = json.loads((run.output / "color-extra" / "cam0" / "timestamps.json").read_text())
                    self.assertEqual([entry["cuda_ms"] for entry in entries], [expected_ms[i] for i in saved_indices])
                    self.assertEqual([entry["valid"] for entry in entries], [expected_valid[i] for i in saved_indices])
                    canonical = self.canonical_saved(run)
                    self.assertEqual([path.name for path in canonical], ["000000.png", "000001.png"])
                    canonical_entries = self.manifest(run)
                    self.assertEqual(
                        [entry["cuda_ms"] for entry in canonical_entries], [expected_ms[i] for i in (1, 3)]
                    )
                    self.assertEqual(
                        [entry["valid"] for entry in canonical_entries], [expected_valid[i] for i in (1, 3)]
                    )
                    self.assertEqual([entry["iteration"] for entry in canonical_entries], [2, 2])
                    run.metrics.write_metrics_to_yaml.assert_called_once_with(str(run.output))

    def test_invalid_cuda_samples_are_excluded_from_extrema(self):
        for durations in ([None, 2.0, 4.0, None], [1.0, None, 3.0, None], [None] * 4):
            for mode in ("min,max", "all-iterations,min,max"):
                with self.subTest(durations=durations, mode=mode):
                    run = self.run_render(
                        ["--save", mode, "--iterations", "2"],
                        durations=durations,
                    )
                    self.assert_success(run)
                    self.assertEqual(len(self.samples(run, "cuda_ms")), 4)
                    self.assertEqual(len(self.samples(run, "render_time_ms")), 4)
                    valid_indices = [index for index, ms in enumerate(durations) if ms is not None]
                    expected_names = []
                    if "all-iterations" in mode:
                        expected_names.extend(
                            f"render-4x2-frame-{index // 2}-iteration-{index % 2 + 1}.png" for index in range(4)
                        )
                    if valid_indices:
                        for selection, select in (("min", min), ("max", max)):
                            index = select(valid_indices, key=lambda index: durations[index])
                            expected_names.append(
                                f"render-4x2-frame-{index // 2}-iteration-{index % 2 + 1}-{selection}.png"
                            )
                    self.assertEqual([path.name for path in self.extra_saved(run)], expected_names)
                    entries = json.loads((run.output / "color-extra" / "cam0" / "timestamps.json").read_text())
                    self.assertEqual([entry["file_name"] for entry in entries], expected_names)
                    extrema = [entry for entry in entries if entry["file_name"].endswith(("-min.png", "-max.png"))]
                    self.assertEqual(len(extrema), 2 if valid_indices else 0)
                    self.assertTrue(all(entry["valid"] and entry["cuda_ms"] > 0 for entry in extrema))
                    self.assertEqual([path.name for path in self.canonical_saved(run)], ["000000.png", "000001.png"])
                    self.assertEqual([entry["iteration"] for entry in self.manifest(run)], [2, 2])

    def test_default_saves_last_iteration_of_each_camera_frame(self):
        """Omitting --save saves each frame once, after its final measured iteration."""
        for iterations in (1, 3):
            with self.subTest(iterations=iterations):
                run = self.run_render(
                    ["--warmup-frames", "1", "--iterations", str(iterations)],
                    cameras=("cam0", "cam1"),
                )
                self.assert_success(run)
                self.assertEqual(len(run.saved), 4)
                self.assertEqual(
                    [path.name for path in run.saved],
                    [f"{frame:06d}.png" for _ in range(2) for frame in range(2)],
                )
                self.assertEqual(len(run.timer.milliseconds), 4 * iterations)
                run.lidar_helper.assert_not_called()

    def test_save_none_preserves_rendering_metrics_and_captures_without_output_copies(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        for nsight in (False, True):
            with self.subTest(nsight=nsight), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                with patch.object(module, "_save_semantics_legend") as legend:
                    run = self.run_render(
                        [
                            "--save",
                            "none",
                            "--warmup-frames",
                            "2",
                            "--iterations",
                            "2",
                            *(["--nsight"] if nsight else []),
                        ],
                        cameras=("cam0", "cam1"),
                        output_root=output,
                        environment={module.NSIGHT_OUTPUT_DIR_ENV: str(output)} if nsight else None,
                    )
                self.assert_success(run)
                self.assertEqual(len(run.calls), 12)  # Two warmups and four measured renders per camera.
                self.assertEqual(len(run.warmup_ranges), 4)
                self.assertEqual(len(run.validations), 12)
                self.assertEqual(run.metrics.record_frame.call_count, 8)
                for camera, first in (("cam0", 1), ("cam1", 5)):
                    self.assertEqual(self.samples(run, "cuda_ms", camera), list(range(first, first + 4)))
                    self.assertEqual(len(self.samples(run, "render_time_ms", camera)), 4)
                self.assertFalse(run.saved)
                self.assertFalse(run.copy_ranges)
                self.assertFalse(run.pinned_buffers)
                self.assertFalse(run.pinned_copies)
                legend.assert_not_called()
                run.lidar_helper.assert_not_called()
                run.metrics.write_metrics_to_yaml.assert_called_once_with(str(output))
                self.assertFalse((output / CameraOutputSubdir.COLOR).exists())
                self.assertFalse(list(output.rglob("timestamps.json")))
                self.assertFalse(list(output.rglob("*.png")))
                self.assertEqual(json.loads((output / "render_cli_args.json").read_text())["args"]["save"], "none")
                if nsight:
                    self.assertEqual(len(run.capture_windows), 4)
                    self.assertEqual(len(run.nsight_capture.records), 4)
                    self.assert_sensor_capture_lifecycle(run, ("cam0", "cam1"))
                    for record in run.nsight_capture.records:
                        self.assertEqual(record["trace"], f"nsight/{record['camera']}/performance.nsys-rep")

    def test_save_none_leaves_existing_images_and_manifests_untouched(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sensor_dir = output / CameraOutputSubdir.COLOR / "cam0"
            sensor_dir.mkdir(parents=True)
            image = sensor_dir / "000000.png"
            manifest = sensor_dir / "timestamps.json"
            image.write_bytes(b"previous image")
            manifest.write_text("previous manifest")
            with patch.object(module, "export_videos_and_json") as export:
                run = self.run_render(["--save", "none"], output_root=output)
            self.assert_success(run)
            self.assertEqual(image.read_bytes(), b"previous image")
            self.assertEqual(manifest.read_text(), "previous manifest")
            self.assertFalse(run.saved)
            self.assertFalse(run.copy_ranges)
            export.assert_not_called()

    def test_save_none_is_forwarded_to_lidar(self):
        run = self.run_render(["--lidar", "--save", "none", "--iterations", "2"])
        self.assert_success(run)
        run.load.assert_called_once()
        run.lidar_helper.assert_called_once()
        self.assertEqual(run.lidar_helper.call_args.kwargs["save_modes"], {"none"})
        self.assertEqual(run.lidar_helper.call_args.kwargs["iterations"], 2)
        self.assertFalse(run.saved)
        self.assertFalse(run.calls)

    def test_no_report_and_no_save_match_warmup_timing_without_reporting(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        for nsight in (False, True):
            with self.subTest(nsight=nsight), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                with patch.object(module.log, "info") as info:
                    run = self.run_render(
                        [
                            "--no-report",
                            "--save",
                            "none",
                            "--warmup-frames",
                            "2",
                            "--iterations",
                            "2",
                            *(["--nsight"] if nsight else []),
                        ],
                        cameras=("cam0", "cam1"),
                        output_root=output,
                        environment={module.NSIGHT_OUTPUT_DIR_ENV: str(output)} if nsight else None,
                        wall_durations=[],
                    )
                self.assert_success(run)
                self.assertEqual(len(run.calls), 12)
                self.assertEqual(len(run.validations), 12)
                self.assertEqual(sum(measured for _, _, measured in run.calls), 8)
                self.assertEqual(len(run.warmup_ranges), 4)
                self.assertEqual(len(run.timer_calls), 12)
                self.assertTrue(all(not record_sample for _, record_sample in run.timer_calls))
                self.assertEqual(sum(label is None for label, _ in run.timer_calls), 8)
                self.assertFalse(run.timer.milliseconds)
                self.assertFalse(run.timer.valid)
                run.wall_clock.assert_not_called()
                run.metrics_factory.assert_not_called()
                self.assertFalse(run.metrics.mock_calls)
                self.assertFalse(run.saved)
                self.assertFalse(run.copy_ranges)
                self.assertFalse((output / "metrics.yaml").exists())
                self.assertFalse(
                    any("rendered in" in str(call) or "cuda_ms=" in str(call) for call in info.call_args_list)
                )
                self.assertIs(json.loads((output / "render_cli_args.json").read_text())["args"]["report"], False)
                if nsight:
                    self.assertEqual(len(run.capture_windows), 4)
                    self.assertEqual(len(run.nsight_capture.records), 4)
                    self.assertEqual(
                        {record["trace"] for record in run.nsight_capture.records},
                        {f"nsight/{camera}/performance.nsys-rep" for camera in ("cam0", "cam1")},
                    )
                    self.assert_sensor_capture_lifecycle(run, ("cam0", "cam1"))

    def test_no_report_preserves_saving_and_cuda_extrema(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        for save in (None, "min,max"):
            with self.subTest(save=save), patch.object(module.log, "info") as info:
                run = self.run_render(
                    ["--no-report", "--iterations", "2", *(["--save", save] if save else [])],
                    durations=[9, 1, 5, 6],
                    wall_durations=[],
                )
                self.assert_success(run)
                run.wall_clock.assert_not_called()
                run.metrics_factory.assert_not_called()
                self.assertFalse(run.metrics.mock_calls)
                self.assertEqual(run.timer.milliseconds, [9, 1, 5, 6])
                self.assertEqual([entry["cuda_ms"] for entry in self.manifest(run)], [1, 6])
                self.assertEqual(len(self.canonical_saved(run)), 2)
                self.assertEqual(len(self.extra_saved(run)), 2 if save else 0)
                self.assertFalse(
                    any("rendered in" in str(call) or "cuda_ms=" in str(call) for call in info.call_args_list)
                )
                if save:
                    entries = json.loads((run.output / "color-extra" / "cam0" / "timestamps.json").read_text())
                    self.assertEqual([entry["cuda_ms"] for entry in entries], [1, 9])

    def test_no_report_preserves_existing_metrics_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metrics_path = output / "metrics.yaml"
            metrics_path.write_text("previous report")
            run = self.run_render(["--no-report", "--save", "none"], output_root=output, wall_durations=[])
            self.assert_success(run)
            self.assertEqual(metrics_path.read_text(), "previous report")
            run.metrics_factory.assert_not_called()

    def test_no_report_is_forwarded_to_lidar(self):
        run = self.run_render(["--lidar", "--no-report", "--save", "none"])
        self.assert_success(run)
        run.metrics_factory.assert_not_called()
        self.assertFalse(run.metrics.mock_calls)
        run.lidar_helper.assert_called_once()
        self.assertIsNone(run.lidar_helper.call_args.kwargs["metrics_collector"])
        self.assertEqual(run.lidar_helper.call_args.kwargs["save_modes"], {"none"})

    def test_no_report_nsight_parent_does_not_require_metrics_from_child(self):
        run = self.run_render(["--nsight", "--no-report", "--save", "none"])
        self.assert_success(run)
        run.nsight_helper.assert_called_once_with([sys.executable, *sys.argv], run.output, expect_metrics=False)
        run.load.assert_not_called()
        run.metrics_factory.assert_not_called()

    def test_video_export_uses_default_saved_frames(self):
        """Video export can consume the default saved frames without an explicit --save."""
        module = importlib.import_module("nurec.utils.io.export.render")
        with patch.object(module, "export_videos_and_json") as export:
            run = self.run_render(["--export-video"])
        self.assert_success(run)
        self.assertEqual(len(run.saved), 2)
        export.assert_called_once()
        self.assertEqual(export.call_args.args[0], str(run.output))
        self.assertEqual(export.call_args.kwargs["frames_subdir"], CameraOutputSubdir.COLOR)

    def test_default_measures_and_saves_every_frame_and_leaves_stride_unchanged(self):
        run = self.run_render()
        self.assert_success(run)
        self.assertEqual(run.calls, [("cam0", 0, True), ("cam0", 1, True)])
        self.assertFalse(run.synchronizations)
        self.assertFalse(run.warmup_ranges)
        self.assertEqual(len(run.pinned_buffers), 1)
        self.assertEqual(run.pinned_copies, [(0, 1), (0, 2)])
        self.assertEqual(
            [label for label, _, _ in run.frame_ranges],
            ["nurec.render.frame(camera=cam0, frame=0)", "nurec.render.frame(camera=cam0, frame=1)"],
        )
        self.assertEqual([path.name for path in run.saved], ["000000.png", "000001.png"])
        self.assertTrue((run.output / CameraOutputSubdir.COLOR).exists())
        self.assertEqual(len(list(run.output.rglob("timestamps.json"))), 1)
        self.assertEqual(self.samples(run, "cuda_ms"), [1.0, 2.0])
        self.assertEqual(len(self.samples(run, "render_time_ms")), 2)
        self.assertEqual(run.load.call_args.kwargs["tracks_timestamp_ray_stride"], 16)
        run.training_mapping.assert_not_called()
        run.metrics.write_metrics_to_yaml.assert_called_once_with(str(run.output))

    def test_ordinary_renders_omit_training_frame_indices(self):
        for calib_source in (
            "training-rig-poses",
            "training-rig-poses-per-frame",
            "training-sensor-poses-calib",
            "training-sensor-poses-nocalib",
        ):
            with self.subTest(calib_source=calib_source):
                run = self.run_render(
                    [
                        "--calib-source",
                        calib_source,
                        "--frame-index",
                        "1",
                        "--warmup-frames",
                        "1",
                        "--iterations",
                        "2",
                    ],
                    cameras=("cam0", "cam1"),
                    frame_count=3,
                    trajectory_count=2,
                )
                self.assert_success(run)
                self.assertEqual(len(run.render_arguments), 6)
                self.assertTrue(all(arguments["unique_frame_idx"] is None for arguments in run.render_arguments))
                self.assertTrue(all(arguments["training_frame_idx"] is None for arguments in run.render_arguments))
                run.training_mapping.assert_not_called()

    def test_nonreplicated_views_allow_missing_camera_offsets(self):
        for trajectory_count in (1, 2):
            with self.subTest(trajectory_count=trajectory_count):
                run = self.run_render(
                    ["--frame-index", "1"],
                    trajectory_count=trajectory_count,
                    invalid_offsets=True,
                )
                self.assert_success(run)
                run.load.assert_called_once()
                run.training_mapping.assert_not_called()
                self.assertEqual(len(run.render_arguments), 1)
                self.assertIsNone(run.render_arguments[0]["unique_frame_idx"])
                self.assertIsNone(run.render_arguments[0]["training_frame_idx"])
                self.assertTrue((run.output / "color" / "cam0" / "000000.png").is_file())
                run.lidar_helper.assert_not_called()

    def test_only_first_trajectory_renders_into_the_provided_output_directory(self):
        run = self.run_render(
            ["--frame-index", "1", "--warmup-frames", "2", "--iterations", "2", "--save", "min,max"],
            trajectory_count=2,
            durations=[8, 2],
        )
        self.assert_success(run)
        run.load.assert_called_once()
        self.assertEqual(len(run.collectors), 1)
        run.camera_provider_factory.assert_called_once()
        self.assertEqual(run.trajectory_calls, [(0, "cam0", 1, timed) for timed in (False, False, True, True)])
        self.assertFalse(run.synchronizations)
        self.assertEqual(len(run.validations), 4)
        self.assertEqual(len(run.warmup_ranges), 2)
        self.assertEqual(len(run.saved), 3)
        self.assertEqual(run.output, run.output_root.resolve())
        self.assertTrue((run.output / "render_cli_args.json").is_file())
        run.metrics.write_metrics_to_yaml.assert_called_once_with(str(run.output))
        records = [call.kwargs for call in run.metrics.record_frame.call_args_list]
        self.assertEqual([record["cuda_ms"] for record in records], [8, 2])
        self.assertEqual(
            [(record["sequence_id"], record["sensor_id"], record["frame"], record["iteration"]) for record in records],
            [("test-sequence-0", "cam0", 1, iteration) for iteration in (1, 2)],
        )
        for metadata in run.metrics.metadata_values.values():
            self.assertEqual(metadata["trajectory_id"], 0)
            self.assertEqual(metadata["sequence_id"], "test-sequence-0")
            self.assertEqual(metadata["artifact"], str(run.artifact_path.resolve()))
            self.assertEqual(metadata["frame_index"], "1")
            self.assertEqual(metadata["iterations"], 2)
            self.assertEqual(metadata["warmup_frames"], 2)
            self.assertEqual(metadata["model"], "MockRenderable")
            self.assertEqual(metadata["resolutions"], {"cam0": [4, 2]})
        camera = run.output / "color-extra" / "cam0"
        entries = json.loads((camera / "timestamps.json").read_text())
        self.assertEqual(len(entries), 2)
        for suffix, iteration, call_number in (("min", 2, 4), ("max", 1, 3)):
            path = camera / f"render-4x2-frame-1-iteration-{iteration}-{suffix}.png"
            with Image.open(path) as image:
                self.assertEqual(image.getpixel((0, 0)), (int(255 * call_number / 32),) * 3)
        selected = run.camera_provider_factory.call_args.args[0]
        self.assertIs(selected, run.rigs)
        self.assertEqual(len(selected.rig_trajectories), 2)
        self.assertFalse(list(run.output.glob("trajectory-*")))

    def test_unsafe_camera_ids_fail_before_any_selected_camera_renders(self):
        for explicit_selection in (False, True):
            with self.subTest(explicit_selection=explicit_selection), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "output"
                outside = root / "outside"
                outside.mkdir()
                sentinel = outside / "000000.png"
                sentinel.write_bytes(b"existing image must remain untouched")
                # Both escape targets stay inside this test's temporary directory, even if
                # validation regresses. The hostile camera follows a legitimate camera.
                bad_camera = "../../outside" if explicit_selection else str(outside)
                cameras = ("front", bad_camera)
                options = ["--frame-index", "0", "--save", "last-iteration"]
                if explicit_selection:
                    for camera in cameras:
                        options.extend(["--camera-id", camera])
                run = self.run_render(options, cameras=cameras, frame_count=1, output_root=output)

                self.assertNotEqual(run.result.exit_code, 0, run.result.output)
                self.assertIn("camera", run.result.output.lower())
                self.assertFalse(run.calls, "Validate the entire selected list before rendering the first camera")
                self.assertFalse(run.saved)
                run.metrics.record_frame.assert_not_called()
                self.assertFalse((output / CameraOutputSubdir.COLOR).exists())
                self.assertFalse((output / "color-extra").exists())
                self.assertEqual(sentinel.read_bytes(), b"existing image must remain untouched")
                self.assertEqual({path.name for path in outside.iterdir()}, {sentinel.name})

    def test_camera_output_names_preserve_identity_in_frames_metrics_and_video_index(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        # These remain distinct: replacing punctuation or spaces with underscores would
        # rename ordinary cameras and make "front left" collide with "front_left".
        cameras = ("front.left-120_fov", "front left", "front_left", "caméra")
        with patch.object(module, "export_video", return_value=True) as export:
            run = self.run_render(
                ["--frame-index", "0", "--save", "last-iteration", "--export-video"],
                cameras=cameras,
                frame_count=1,
            )
        self.assert_success(run)
        expected_saved = {
            path
            for camera in cameras
            for path in (
                run.output / CameraOutputSubdir.COLOR / camera / "000000.png",
                run.output / "color-extra" / camera / "render-4x2-frame-0-iteration-1.png",
            )
        }
        self.assertEqual(set(run.saved), expected_saved)
        self.assertTrue(all(path.is_file() for path in expected_saved))
        self.assertEqual({call.kwargs["sensor_id"] for call in run.metrics.record_frame.call_args_list}, set(cameras))
        for metadata in run.metrics.metadata_values.values():
            self.assertEqual(metadata["resolutions"], {camera: [4, 2] for camera in cameras})
        video_root = run.output / CameraOutputSubdir.COLOR
        self.assertEqual(
            [call.args[:2] for call in export.call_args_list],
            [(video_root / camera, video_root / camera / "video.mp4") for camera in cameras],
        )
        self.assertEqual(
            json.loads((video_root / "render_cli_videos.json").read_text()),
            {camera: f"{camera}/video.mp4" for camera in cameras},
        )

    def test_native_resolution_default_and_explicit_overrides(self):
        # Native calibration dimensions can be NumPy integers; YAML must receive plain integers.
        native = {"cam0": (np.int32(8), np.int32(4)), "cam1": (np.int32(12), np.int32(6))}
        for options, requested, expected in (
            ([], native, native),
            (["--height", "2"], {"cam0": (1, 2), "cam1": (1, 2)}, {"cam0": (4, 2), "cam1": (4, 2)}),
            (["--image-scale", "0.5"], {"cam0": (4, 2), "cam1": (6, 3)}, {"cam0": (4, 2), "cam1": (6, 3)}),
        ):
            with self.subTest(options=options):
                run = self.run_render(
                    [*options, "--frame-index", "0", "--save", "last-iteration"],
                    cameras=tuple(native),
                    native_resolutions=native,
                )
                self.assert_success(run)
                self.assertEqual([call["resolution"] for call in run.render_arguments], list(requested.values()))
                for metadata in run.metrics.metadata_values.values():
                    self.assertEqual(metadata["resolutions"], {key: list(size) for key, size in expected.items()})
                    for size in metadata["resolutions"].values():
                        self.assertTrue(all(type(dimension) is int for dimension in size))
                for camera, size in expected.items():
                    path = next((run.output / CameraOutputSubdir.COLOR / camera).glob("*.png"))
                    with Image.open(path) as image:
                        self.assertEqual(image.size, size)

    def test_native_resolution_normalization_preserves_requests_and_output_dimensions(self):
        # Native rendering historically normalizes these requests rather than rejecting them.
        # The two scale cases land on opposite sides of a floating-point integer boundary.
        for native, options, requested, actual in (
            ((1280, 720), ["--image-scale", "0.7"], (896, 503), (896, 504)),
            ((2880, 1860), ["--image-scale", "0.7"], (2015, 1302), (2016, 1302)),
            # Height requests historically keep width=1; for a narrow portrait camera that
            # minimum width controls the scale, so the native output height can be larger.
            ((2, 8), ["--height", "1"], (1, 1), (1, 4)),
        ):
            with self.subTest(native=native, options=options):
                run = self.run_render(
                    [
                        *options,
                        "--frame-index",
                        "1",
                        "--warmup-frames",
                        "1",
                        "--iterations",
                        "2",
                        "--save",
                        "all-iterations,min,max",
                    ],
                    native_resolutions={"cam0": native},
                    durations=[8, 2],
                )
                self.assert_success(run)
                # Precomputing output dimensions must not change what is passed to native
                # rendering, which still performs its own normalization of the original request.
                self.assertEqual([call["resolution"] for call in run.render_arguments], [requested] * 3)
                for call in run.render_arguments:
                    self.assertEqual(tuple(call["camera_intrinsics"].resolution), native)
                self.assertEqual([resolution for _, resolution in run.validations], [actual] * 3)
                for metadata in run.metrics.metadata_values.values():
                    self.assertEqual(metadata["resolutions"], {"cam0": list(actual)})

                ordinary = run.output / CameraOutputSubdir.COLOR / "cam0"
                extra = run.output / "color-extra" / "cam0"
                self.assertEqual({path.name for path in ordinary.glob("*.png")}, {"000000.png"})
                width, height = actual
                prefix = f"render-{width}x{height}-frame-1"
                extra_names = {
                    f"{prefix}-iteration-1.png",
                    f"{prefix}-iteration-2.png",
                    f"{prefix}-iteration-2-min.png",
                    f"{prefix}-iteration-1-max.png",
                }
                self.assertEqual({path.name for path in extra.glob("*.png")}, extra_names)
                ordinary_timestamps = json.loads((ordinary / "timestamps.json").read_text())
                self.assertEqual([entry["file_name"] for entry in ordinary_timestamps], ["000000.png"])
                extra_timestamps = json.loads((extra / "timestamps.json").read_text())
                self.assertEqual({entry["file_name"] for entry in extra_timestamps}, extra_names)
                for path in [ordinary / "000000.png", *(extra / name for name in extra_names)]:
                    with Image.open(path) as image:
                        self.assertEqual(image.size, actual)

    def test_height_and_scale_remain_mutually_exclusive(self):
        run = self.run_render(["--height", "2", "--image-scale", "1"])
        self.assertEqual(run.result.exit_code, 2)
        self.assertIn("--height and --image-scale cannot be used together", run.result.output)
        run.load.assert_not_called()

    def test_replicated_views_map_first_trajectory_and_preserve_training_indices(self):
        run = self.run_render(
            ["--replicate-training-views", "--frame-index", "1"], trajectory_count=2, invalid_offsets=True
        )
        self.assert_success(run)
        run.load.assert_called_once()
        run.training_mapping.assert_called_once()
        first_trajectory = run.rigs.rig_trajectories[0]
        self.assertIs(run.training_mapping.call_args.args[1], first_trajectory)
        self.assertEqual(first_trajectory.cameras_linear_start_frame_indices, {"cam0@test-sequence-0": 0})
        self.assertEqual([arguments["unique_frame_idx"] for arguments in run.render_arguments], [1])
        self.assertEqual([arguments["training_frame_idx"] for arguments in run.render_arguments], [1])
        run.sensor_provider_factory.assert_called_once()
        run.camera_provider_factory.assert_not_called()
        provider_arguments = run.sensor_provider_factory.call_args.kwargs
        self.assertIs(provider_arguments["camera_calibrations"], run.rigs.camera_calibrations)
        self.assertIs(provider_arguments["camera_trajectories"], run.renderable.get_camera_trajectories.return_value)
        self.assertIs(
            provider_arguments["cameras_start_frame_indices"], first_trajectory.cameras_linear_start_frame_indices
        )

    def test_custom_rig_file_also_uses_only_its_first_trajectory(self):
        for sensor_options in ([], ["--lidar"]):
            with self.subTest(options=sensor_options):
                run = self.run_render(
                    [*sensor_options, "--custom-rig-trajectory", "custom.json", "--frame-index", "0"],
                    trajectory_count=2,
                )
                self.assert_success(run)
                run.load.assert_called_once()
                self.assertEqual(run.rig_parser.call_count, 2)
                self.assertEqual(
                    run.trajectory_calls, [(0, "cam0", 0, True)] if "--lidar" not in sensor_options else []
                )
                if "--lidar" in sensor_options:
                    run.lidar_helper.assert_called_once()
                    self.assertIs(run.lidar_helper.call_args.args[1], run.rigs)
                else:
                    self.assertIs(run.camera_provider_factory.call_args.args[0], run.rigs)
                run.metrics_factory.assert_called_once()
                self.assertTrue(
                    all(metadata["trajectory_id"] == 0 for metadata in run.metrics.metadata_values.values())
                )

    def test_lidar_custom_trajectory_preserves_supplied_poses_and_metadata_without_camera_data(self):
        """LiDAR-only rendering selects the custom rig rather than the training rig."""
        module = importlib.import_module("nurec.utils.io.export.render")
        poses = torch.eye(4).repeat(2, 1, 1)
        poses[:, 0, 3] = torch.tensor([10.0, 20.0])
        trajectory = module.RigTrajectories.RigTrajectory(
            sequence_id="custom-lidar",
            rig_bbox=None,
            cameras_frame_timestamps_us={},
            lidars_frame_timestamps_us={"lidar@custom-lidar": torch.tensor([[100, 150], [150, 200]])},
            T_rig_worlds=poses,
            T_rig_world_timestamps_us=torch.tensor([100, 200]),
        )
        calibration = SimpleNamespace(logical_sensor_name="lidar_front", unique_sensor_idx=9)
        custom_rigs = module.RigTrajectories(
            T_world_base=torch.eye(4),
            world_to_nurec=object(),
            rig_trajectories=[trajectory],
            camera_calibrations=OrderedDict(),
            lidar_calibrations=OrderedDict({"lidar@custom-lidar": calibration}),
        )
        run = self.run_render(
            ["--lidar", "--custom-rig-trajectory", "custom.json", "--frame-index", "1"],
            custom_rigs=custom_rigs,
            invalid_offsets=True,
        )
        self.assert_success(run)
        run.artifact.load_custom_rig_trajectories.assert_called_once_with("custom.json")
        run.lidar_helper.assert_called_once()
        selected = run.lidar_helper.call_args.args[1]
        self.assertIs(selected.rig_trajectories[0], trajectory)
        self.assertIs(selected.rig_trajectories[0].T_rig_worlds, poses)
        self.assertIs(selected.lidar_calibrations["lidar@custom-lidar"], calibration)
        self.assertIs(selected.world_to_nurec, custom_rigs.world_to_nurec)
        self.assertFalse(selected.camera_calibrations)
        self.assertEqual(run.lidar_helper.call_args.kwargs["frame_index"], "1")
        self.assertEqual(run.metrics.metadata_values["cuda_ms"]["sequence_id"], "custom-lidar")
        run.camera_provider_factory.assert_not_called()
        run.sensor_provider_factory.assert_not_called()
        run.training_mapping.assert_not_called()
        self.assertFalse(run.calls)

    def test_custom_trajectory_rejects_training_sensor_poses_before_loading(self):
        for calib_source in ("training-sensor-poses-calib", "training-sensor-poses-nocalib"):
            with self.subTest(calib_source=calib_source):
                run = self.run_render(["--custom-rig-trajectory", "custom.json", "--calib-source", calib_source])
                self.assertEqual(run.result.exit_code, 2, run.result.output)
                self.assertIn("--custom-rig-trajectory cannot be used with --calib-source", run.result.output)
                self.assertIn(calib_source, run.result.output)
                self.assertIn("training-rig-poses", run.result.output)
                self.assertIn("training-rig-poses-per-frame", run.result.output)
                run.load.assert_not_called()
                run.metrics_factory.assert_not_called()
                run.camera_provider_factory.assert_not_called()
                run.sensor_provider_factory.assert_not_called()
                run.nsight_capture.start.assert_not_called()
                run.lidar_helper.assert_not_called()
                self.assertFalse(run.calls)

    def test_removed_trajectory_option_is_rejected(self):
        run = self.run_render(["--trajectory-id", "0"], trajectory_count=2)
        self.assertEqual(run.result.exit_code, 2)
        self.assertIn("No such option", run.result.output)
        self.assertIn("--trajectory-id", run.result.output)
        run.load.assert_not_called()
        run.metrics_factory.assert_not_called()

    def test_no_trajectories_reports_a_useful_error_before_loading(self):
        for sensor_options in ([], ["--lidar"]):
            for custom in (False, True):
                with self.subTest(options=sensor_options, custom=custom):
                    run = self.run_render(
                        [*sensor_options, *(["--custom-rig-trajectory", "custom.json"] if custom else [])],
                        trajectory_count=0,
                    )
                    self.assertEqual(run.result.exit_code, 1, run.result.output)
                    self.assertIn("No rig trajectories are available for rendering", run.result.output)
                    run.load.assert_not_called()
                    run.metrics_factory.assert_not_called()
                    run.lidar_helper.assert_not_called()
                    self.assertFalse(run.calls)

    def test_default_replicated_views_allow_legacy_missing_camera_offsets(self):
        run = self.run_render(
            ["--frame-index", "1", "--warmup-frames", "1", "--iterations", "2"],
            invalid_offsets=True,
            use_default_training_views=True,
        )
        self.assert_success(run)
        settings = json.loads((run.output / "render_cli_args.json").read_text())
        self.assertTrue(settings["args"]["replicate_training_views"])
        run.load.assert_called_once()
        run.training_mapping.assert_called_once_with(run.artifact.parsed_config, run.rigs.rig_trajectories[0])
        self.assertIsNone(run.rigs.rig_trajectories[0].cameras_linear_start_frame_indices)
        run.sensor_provider_factory.assert_called_once()
        self.assertIsNone(run.sensor_provider_factory.call_args.kwargs["cameras_start_frame_indices"])
        self.assertEqual(run.calls, [("cam0", 1, measured) for measured in (False, True, True)])
        self.assertTrue(all(arguments["unique_frame_idx"] is None for arguments in run.render_arguments))
        self.assertTrue(all(arguments["training_frame_idx"] is None for arguments in run.render_arguments))
        self.assertEqual(run.metrics.record_frame.call_count, 2)
        self.assertTrue((run.output / "color" / "cam0" / "000000.png").is_file())
        run.lidar_helper.assert_not_called()

    def test_native_trajectory_parser_errors_are_actionable_before_model_loading(self):
        fallback = "inconsistent sensor offsets, timestamps, or poses"
        for custom in (False, True):
            for error in (
                AssertionError(),
                ValueError("invalid poses"),
                KeyError("camera offsets"),
                TypeError("bad data"),
            ):
                with self.subTest(custom=custom, error=type(error).__name__):
                    options = ["--custom-rig-trajectory", "custom.json"] if custom else []
                    run = self.run_render(options, rig_parser_error=error)
                    self.assertEqual(run.result.exit_code, 1, run.result.output)
                    self.assertIn(
                        f"Error: Invalid rig trajectory metadata: {str(error) or fallback}", run.result.output
                    )
                    self.assertEqual(run.rig_parser.call_count, 2 if custom else 1)
                    run.load.assert_not_called()
                    run.metrics_factory.assert_not_called()
                    run.nsight_capture.start.assert_not_called()
                    self.assertFalse(run.calls)

    def test_malformed_raw_trajectory_metadata_uses_the_native_parser_and_reports_its_error(self):
        for custom in (False, True):
            with self.subTest(custom=custom):
                options = ["--custom-rig-trajectory", "custom.json"] if custom else []
                run = self.run_render(options, malformed_rigs=True)
                self.assertEqual(run.result.exit_code, 1, run.result.output)
                self.assertIn("Error: Invalid rig trajectory metadata:", run.result.output)
                self.assertIn("T_world_base", run.result.output)
                run.load.assert_not_called()
                run.metrics_factory.assert_not_called()
                self.assertFalse(run.calls)

    def test_custom_trajectory_loading_errors_are_actionable_before_parsing_or_model_loading(self):
        for sensor_options in ([], ["--lidar"]):
            for error in (AssertionError(), ValueError("custom JSON is invalid")):
                with self.subTest(options=sensor_options, error=type(error).__name__):
                    run = self.run_render(
                        [*sensor_options, "--custom-rig-trajectory", "custom.json"], custom_loader_error=error
                    )
                    self.assertEqual(run.result.exit_code, 1, run.result.output)
                    self.assertIn("Error: Invalid rig trajectory metadata:", run.result.output)
                    self.assertIn(str(error) or "inconsistent sensor offsets, timestamps, or poses", run.result.output)
                    run.artifact.load_custom_rig_trajectories.assert_called_once_with("custom.json")
                    self.assertEqual(run.rig_parser.call_count, 1)
                    run.load.assert_not_called()
                    run.metrics_factory.assert_not_called()
                    run.lidar_helper.assert_not_called()
                    self.assertFalse(run.calls)

    def test_invalid_requested_output_is_rejected_before_recording_or_saving(self):
        for warmup in (0, 1):
            for output_error, output_type in (
                ("missing", "color"),
                ("wrong-size", "color"),
                (None, "distance-colorized"),
            ):
                with self.subTest(warmup=warmup, output_error=output_error, output_type=output_type):
                    run = self.run_render(
                        ["--warmup-frames", str(warmup), "--output-type", output_type, "--save", "min,max"],
                        output_error=output_error,
                        wall_durations=[11],
                    )
                    self.assertEqual(run.result.exit_code, 1, run.result.output)
                    self.assertEqual(len(run.calls), 1)
                    self.assertEqual(len(run.validations), 1)
                    self.assertEqual(len(run.timer.milliseconds), 0 if warmup else 1)
                    run.metrics.record_frame.assert_not_called()
                    run.metrics.write_metrics_to_yaml.assert_not_called()
                    self.assertFalse(run.saved)
                    self.assertFalse(run.copy_ranges)

    def test_lidar_receives_full_rig_metadata_without_requiring_camera_offsets(self):
        run = self.run_render(["--lidar"], trajectory_count=2, invalid_offsets=True)
        self.assert_success(run)
        run.load.assert_called_once()
        run.lidar_helper.assert_called_once()
        selected = run.lidar_helper.call_args.args[1]
        self.assertIs(selected, run.rigs)
        self.assertEqual(len(selected.rig_trajectories), 2)
        self.assertEqual(run.lidar_helper.call_args.kwargs["output_dir"], run.output_root.resolve())
        run.training_mapping.assert_not_called()
        run.camera_provider_factory.assert_not_called()

    def test_nsight_profiles_only_first_trajectory_without_a_trajectory_directory(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "first-trajectory"
            capture.mkdir()
            run = self.run_render(
                ["--nsight", "--frame-index", "1", "--warmup-frames", "1", "--iterations", "2"],
                trajectory_count=2,
                output_root=Path(directory),
                environment={module.NSIGHT_OUTPUT_DIR_ENV: str(capture)},
            )
            self.assert_success(run)
            run.load.assert_called_once()
            run.nsight_factory.assert_called_once_with(capture)
            run.nsight_helper.assert_not_called()
            self.assert_sensor_capture_lifecycle(run, ("cam0",))
            self.assertEqual(
                run.capture_windows,
                [("cam0", 1, 0, [("cam0", 1, False), ("cam0", 1, True), ("cam0", 1, True)])],
            )
            self.assertEqual(
                [record["trace"] for record in run.nsight_capture.records],
                ["nsight/cam0/performance.nsys-rep"],
            )
            self.assertEqual(len(run.collectors), 1)
            self.assertEqual(run.metrics.record_frame.call_count, 2)
            run.metrics.write_metrics_to_yaml.assert_called_once_with(str(capture))
            self.assertEqual(list(Path(directory).iterdir()), [capture])

    def test_repeated_runs_use_the_exact_output_directory_without_generated_subdirectories(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "renders"
            first = self.run_render(["--save", "last-iteration"], output_root=output)
            second = self.run_render(["--save", "last-iteration"], output_root=output)
            for run in (first, second):
                self.assert_success(run)
                self.assertEqual(run.output, output.resolve())
                self.assertEqual(run.run_output, output.resolve())
                settings = json.loads((output / "render_cli_args.json").read_text())
                self.assertEqual(settings["args"]["output_dir"], str(output))
                self.assertTrue(
                    all(
                        path.parent in (output / CameraOutputSubdir.COLOR / "cam0", output / "color-extra" / "cam0")
                        for path in run.saved
                    )
                )
                self.assertTrue(all(path.is_file() for path in run.saved))
                run.metrics.write_metrics_to_yaml.assert_called_once_with(str(output))
            self.assertEqual(
                set(output.iterdir()),
                {output / "render_cli_args.json", output / CameraOutputSubdir.COLOR, output / "color-extra"},
            )

    def test_nsight_rejects_list_tracks_before_profiling_or_loading(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        for child in (False, True):
            with self.subTest(child=child), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "output"
                run = self.run_render(
                    ["--nsight", "--list-tracks", "actors"],
                    output_root=output,
                    environment={module.NSIGHT_OUTPUT_DIR_ENV: str(output)} if child else None,
                )
                self.assertEqual(run.result.exit_code, 2, run.result.output)
                self.assertIn("--nsight and --list-tracks are mutually exclusive", run.result.output)
                run.nsight_helper.assert_not_called()
                run.nsight_factory.assert_not_called()
                run.load.assert_not_called()
                run.metrics_factory.assert_not_called()
                self.assertFalse(run.calls)
                self.assertFalse(output.exists())

    def test_nsight_parent_uses_provided_output_directory_and_delegates_before_loading(self):
        run = self.run_render(["--nsight"])
        self.assert_success(run)
        run.nsight_helper.assert_called_once_with([sys.executable, *sys.argv], run.output, expect_metrics=True)
        self.assertEqual(run.run_output, run.output_root.resolve())
        self.assertEqual(list(run.output_root.iterdir()), [])
        self.assertTrue(run.output.is_dir())
        self.assertFalse(run.calls)
        run.load.assert_not_called()
        run.timer_factory.assert_not_called()
        run.metrics_factory.assert_not_called()
        run.nsight_factory.assert_not_called()

    def test_nsight_child_reuses_run_directory_and_profiles_normal_loop(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            capture = parent / "existing-run"
            capture.mkdir()
            run = self.run_render(
                ["--nsight", "--warmup-frames", "1", "--iterations", "2", "--save", "last-iteration"],
                output_root=parent,
                environment={module.NSIGHT_OUTPUT_DIR_ENV: str(capture)},
            )
            self.assert_success(run)
            self.assertEqual(run.output, capture)
            self.assertEqual(list(parent.iterdir()), [capture])
            run.nsight_helper.assert_not_called()
            run.metrics_factory.assert_called_once_with(
                mode="render", run_id=capture.name, frame_metrics=True, nsight=True
            )
            self.assertEqual(
                run.calls,
                [("cam0", 0, False), ("cam0", 0, True), ("cam0", 0, True), ("cam0", 1, True), ("cam0", 1, True)],
            )
            self.assertEqual(self.samples(run, "cuda_ms"), [1.0, 2.0, 3.0, 4.0])
            self.assertEqual(len(run.saved), 4)
            self.assertTrue(all(path.is_relative_to(capture) and path.is_file() for path in run.saved))
            for record in run.nsight_capture.records:
                self.assertEqual(
                    capture / record["trace"], capture / "nsight" / record["camera"] / "performance.nsys-rep"
                )
            self.assertEqual(run.nsight_capture.frame.call_count, 2)
            self.assertEqual([record["frame"] for record in run.nsight_capture.records], [0, 1])
            settings = json.loads((capture / "render_cli_args.json").read_text())
            self.assertEqual(settings["args"]["output_dir"], str(parent))
            self.assertTrue(settings["args"]["nsight"])

    def test_nsight_captures_selected_frames_and_iterations_continuously_per_camera(self):
        run, capture = self.run_nsight_child(
            [
                "--frame-index",
                "1-3",
                "--frame-step",
                "2",
                "--warmup-frames",
                "2",
                "--iterations",
                "3",
            ],
            cameras=("cam0", "cam1"),
            frame_count=4,
            wall_durations=[10.0] * 12,
        )
        self.assert_success(run)
        run.nsight_factory.assert_called_once_with(capture)
        run.nsight_helper.assert_not_called()
        self.assertEqual(
            [(camera, frame, start, len(renders)) for camera, frame, start, renders in run.capture_windows],
            [("cam0", 1, 0, 5), ("cam0", 3, 5, 3), ("cam1", 1, 8, 5), ("cam1", 3, 13, 3)],
        )
        for camera, frame, _, renders in run.capture_windows:
            warmup = [(camera, frame, False)] * 2 if frame == 1 else []
            self.assertEqual(renders, warmup + [(camera, frame, True)] * 3)
        self.assertEqual(len(run.calls), 16)
        self.assertEqual(run.sensor_capture_windows, [("cam0", 0, 8), ("cam1", 8, 16)])
        self.assertEqual(
            run.nsight_capture.sensor.call_args_list,
            [call(camera, sensor_dir=capture / "nsight" / camera) for camera in ("cam0", "cam1")],
        )
        self.assertEqual(run.nsight_capture.frame.call_count, 4)
        self.assertEqual(run.metrics.record_frame.call_count, 12)
        self.assert_sensor_capture_lifecycle(run, ("cam0", "cam1"))
        self.assertIs(run.metrics_factory.call_args.kwargs["nsight"], True)
        self.assertEqual(
            run.nsight_capture.records,
            [
                {"camera": "cam0", "frame": 1, "trace": "nsight/cam0/performance.nsys-rep"},
                {"camera": "cam0", "frame": 3, "trace": "nsight/cam0/performance.nsys-rep"},
                {"camera": "cam1", "frame": 1, "trace": "nsight/cam1/performance.nsys-rep"},
                {"camera": "cam1", "frame": 3, "trace": "nsight/cam1/performance.nsys-rep"},
            ],
        )
        self.assertEqual(len(run.saved), 4)
        for frame_call, record in zip(run.nsight_capture.frame.call_args_list, run.nsight_capture.records):
            self.assertEqual(frame_call.args[0], record["camera"])
            self.assertEqual(frame_call.args[1], record["frame"])
            self.assertEqual(capture / record["trace"], capture / "nsight" / record["camera"] / "performance.nsys-rep")
        self.assertEqual(
            [(label, start, len(renders)) for label, start, renders in run.frame_ranges],
            [
                ("nurec.render.frame(camera=cam0, frame=1)", 2, 3),
                ("nurec.render.frame(camera=cam0, frame=3)", 5, 3),
                ("nurec.render.frame(camera=cam1, frame=1)", 10, 3),
                ("nurec.render.frame(camera=cam1, frame=3)", 13, 3),
            ],
        )
        self.assertEqual(len(run.warmup_ranges), 4)
        self.assertEqual(list(run.output_root.iterdir()), [capture])

    def test_nsight_sensor_report_is_independent_of_output_products(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        for products in (
            ("color",),
            ("distance-colorized",),
            ("semantics-colorized",),
            ("semantics-colorized", "distance-colorized"),
            ("distance-colorized", "color"),
        ):
            with self.subTest(products=products):
                options = ["--save", "none", "--no-report"]
                for product in products:
                    options.extend(["--output-type", product])
                # Exercise capture placement independently of the fixture's RGB-only output.
                with patch.object(module, "validate_camera_output"):
                    run, capture = self.run_nsight_child(options, wall_durations=[])
                self.assert_success(run)
                self.assert_sensor_capture_lifecycle(run, ("cam0",))
                run.nsight_capture.sensor.assert_called_once_with("cam0", sensor_dir=capture / "nsight" / "cam0")
                self.assertEqual(
                    {record["trace"] for record in run.nsight_capture.records},
                    {"nsight/cam0/performance.nsys-rep"},
                )
                self.assertFalse(run.saved)
                self.assertFalse(run.copy_ranges)

    def test_nsight_zero_warmup_one_iteration_captures_only_selected_render(self):
        run, _ = self.run_nsight_child(
            ["--frame-index", "1", "--warmup-frames", "0", "--iterations", "1"],
        )
        self.assert_success(run)
        self.assertEqual(run.calls, [("cam0", 1, True)])
        self.assertEqual(run.capture_windows, [("cam0", 1, 0, [("cam0", 1, True)])])
        self.assertFalse(run.synchronizations)
        self.assertEqual(run.metrics.record_frame.call_count, 1)
        self.assertEqual(len(run.nsight_capture.records), 1)
        self.assertEqual(run.nsight_capture.records[0]["frame"], 1)
        self.assertEqual(run.frame_ranges, [("nurec.render.frame(camera=cam0, frame=1)", 0, [("cam0", 1, True)])])
        self.assertFalse(run.warmup_ranges)

    def test_nsight_environment_is_ignored_without_flag(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        run = self.run_render(environment={module.NSIGHT_OUTPUT_DIR_ENV: "/unused-capture-directory"})
        self.assert_success(run)
        self.assertEqual(run.run_output, run.output_root.resolve())
        run.nsight_helper.assert_not_called()
        self.assertFalse(run.capture_windows)
        self.assertFalse(run.nsight_capture.records)
        self.assertEqual(run.capture_lifecycle, ["load-model"])
        run.metrics_factory.assert_called_once_with(
            mode="render", run_id=run.run_output.name, frame_metrics=True, nsight=False
        )

    def test_nsight_initialization_failure_closes_early_capture(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "failed-run"
            capture.mkdir()
            run = self.run_render(
                ["--nsight"],
                output_root=Path(directory),
                environment={module.NSIGHT_OUTPUT_DIR_ENV: str(capture)},
                load_error=RuntimeError("model initialization failed"),
            )
            self.assertEqual(run.result.exit_code, 1)
            self.assertEqual(str(run.result.exception), "model initialization failed")
            self.assertEqual(run.capture_lifecycle, ["capture-start", "load-model", "capture-stop", "write-manifest"])
            run.nsight_capture.stop.assert_called_once_with()
            run.nsight_capture.write_manifest.assert_called_once_with()
            run.nsight_capture.frame.assert_not_called()
            run.metrics_factory.assert_not_called()

    def test_nsight_continuous_capture_closes_when_warmup_or_measured_output_is_invalid(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        for warmup in (0, 1):
            with self.subTest(warmup=warmup), tempfile.TemporaryDirectory() as directory:
                capture = Path(directory) / "failed-frame"
                capture.mkdir()
                run = self.run_render(
                    ["--nsight", "--warmup-frames", str(warmup), "--frame-index", "all", "--iterations", "2"],
                    output_root=Path(directory),
                    environment={module.NSIGHT_OUTPUT_DIR_ENV: str(capture)},
                    output_error="missing",
                    wall_durations=[10.0],
                )
                self.assertEqual(run.result.exit_code, 1, run.result.output)
                self.assertIn("invalid color_image", run.result.output)
                self.assertEqual(run.calls, [("cam0", 0, warmup == 0)])
                self.assertEqual(run.nsight_capture.frame.call_count, 1)
                self.assertEqual(
                    run.nsight_capture.records,
                    [
                        {
                            "camera": "cam0",
                            "frame": 0,
                            "trace": "nsight/cam0/performance.nsys-rep",
                            "complete": False,
                        }
                    ],
                )
                run.metrics.record_frame.assert_not_called()
                run.metrics.write_metrics_to_yaml.assert_not_called()
                self.assert_sensor_capture_lifecycle(run, ("cam0",))

    def test_nsight_render_interruption_finalizes_capture_after_stopping(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "interrupted-frame"
            capture.mkdir()
            run = self.run_render(
                ["--nsight"],
                output_root=Path(directory),
                environment={module.NSIGHT_OUTPUT_DIR_ENV: str(capture)},
                render_error=KeyboardInterrupt(),
            )
            self.assertEqual(run.result.exit_code, 1, run.result.output)
            self.assertIn("Aborted!", run.result.output)
            self.assertEqual(run.calls, [("cam0", 0, True)])
            self.assert_sensor_capture_lifecycle(run, ("cam0",))
            self.assertEqual(
                run.nsight_capture.records,
                [
                    {
                        "camera": "cam0",
                        "frame": 0,
                        "trace": "nsight/cam0/performance.nsys-rep",
                        "complete": False,
                    }
                ],
            )
            run.metrics.record_frame.assert_not_called()
            run.metrics.write_metrics_to_yaml.assert_not_called()

    def test_lidar_nsight_failure_finalizes_capture_after_stopping(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "failed-lidar"
            capture.mkdir()
            run = self.run_render(
                ["--lidar", "--nsight"],
                output_root=Path(directory),
                environment={module.NSIGHT_OUTPUT_DIR_ENV: str(capture)},
                lidar_error=RuntimeError("LiDAR rendering failed"),
            )
            self.assertEqual(run.result.exit_code, 1, run.result.output)
            self.assertIn("LiDAR rendering failed", run.result.output)
            self.assertEqual(run.capture_lifecycle, ["capture-start", "load-model", "capture-stop", "write-manifest"])
            run.nsight_capture.stop.assert_called_once_with()
            run.nsight_capture.write_manifest.assert_called_once_with()
            run.metrics.write_metrics_to_yaml.assert_not_called()

    def test_nsight_failure_reports_reason_without_loading(self):
        run = self.run_render(["--nsight"], nsight_error=RuntimeError("Nsight capture failed: test failure"))
        self.assertEqual(run.result.exit_code, 1, run.result.output)
        self.assertIn("Nsight capture failed: test failure", run.result.output)
        run.nsight_helper.assert_called_once()
        run.load.assert_not_called()
        run.timer_factory.assert_not_called()
        run.metrics_factory.assert_not_called()
        self.assertFalse(run.calls)

    def test_lidar_flag_selects_lidar_and_default_selects_camera_outputs(self):
        for options, lidar in (([], False), (["--lidar"], True)):
            with self.subTest(options=options):
                run = self.run_render(options)
                self.assert_success(run)
                run.load.assert_called_once()
                self.assertEqual(run.calls, [] if lidar else [("cam0", 0, True), ("cam0", 1, True)])
                self.assertEqual(run.lidar_helper.call_count, int(lidar))
                run.metrics_factory.assert_called_once()
                run.metrics.write_metrics_to_yaml.assert_called_once_with(str(run.output))
                settings = json.loads((run.output / "render_cli_args.json").read_text())
                self.assertEqual(settings["args"]["lidar"], lidar)
                self.assertNotIn("color", settings["args"])
                if lidar:
                    self.assertIs(run.lidar_helper.call_args.args[0], run.renderable)
                    self.assertIs(run.lidar_helper.call_args.kwargs["metrics_collector"], run.metrics)
                    run.camera_provider_factory.assert_not_called()
                    run.sensor_provider_factory.assert_not_called()
                    run.training_mapping.assert_not_called()
                    self.assertFalse(run.saved)
                else:
                    run.camera_provider_factory.assert_called_once()
                    run.lidar_helper.assert_not_called()

    def test_removed_color_option_is_rejected_before_loading(self):
        for options in (["--color"], ["--color", "--lidar"]):
            with self.subTest(options=options):
                run = self.run_render(options)
                self.assertEqual(run.result.exit_code, 2, run.result.output)
                self.assertRegex(run.result.output, r"No such option(?:: | ')--color\b")
                run.load.assert_not_called()
                run.metrics_factory.assert_not_called()
                run.lidar_helper.assert_not_called()
                self.assertFalse(run.calls)

    def test_help_exposes_only_lidar_as_the_modality_selector(self):
        run = self.run_render(["--help"])
        self.assert_success(run)
        self.assertRegex(run.result.output, r"(?m)^\s+--lidar(?:\s|$)")
        self.assertNotRegex(run.result.output, r"(?m)^\s+--color(?:[\s,=]|$)")
        run.load.assert_not_called()

    def test_camera_nsight_profiling_uses_requested_configuration_and_first_trajectory(self):
        run, capture = self.run_nsight_child(
            [
                "--camera-id",
                "cam1",
                "--height",
                "4",
                "--frame-index",
                "1",
                "--frame-step",
                "2",
                "--warmup-frames",
                "1",
                "--iterations",
                "2",
                "--save",
                "min,max",
                "--replicate-training-views",
            ],
            cameras=("cam0", "cam1"),
            trajectory_count=2,
        )
        self.assert_success(run)
        run.load.assert_called_once()
        run.training_mapping.assert_called_once()
        run.nsight_factory.assert_called_once_with(capture)
        run.nsight_helper.assert_not_called()
        self.assert_sensor_capture_lifecycle(run, ("cam1",))
        run.lidar_helper.assert_not_called()
        self.assertEqual(len(run.collectors), 1)
        self.assertEqual(
            run.trajectory_calls,
            [(0, "cam1", 1, timed) for timed in (False, True, True)],
        )
        self.assertTrue(all(arguments["resolution"] == (1, 4) for arguments in run.render_arguments))
        self.assertEqual(len(run.saved), 3)
        run.metrics.write_metrics_to_yaml.assert_called_once_with(str(capture))
        self.assertEqual(run.metrics.record_frame.call_count, 2)
        self.assertTrue(all(call.kwargs["sensor_id"] == "cam1" for call in run.metrics.record_frame.call_args_list))
        self.assertEqual(run.metrics.metadata_values["cuda_ms"]["resolutions"], {"cam1": [8, 4]})
        self.assertEqual(len(list((capture / CameraOutputSubdir.COLOR / "cam1").glob("*.png"))), 1)

    def test_lidar_defaults_dispatch_without_rgb_configuration(self):
        run = self.run_render(["--lidar"])
        self.assert_success(run)
        run.load.assert_called_once()
        run.lidar_helper.assert_called_once_with(
            run.renderable,
            ANY,  # Rig metadata is forwarded without comparing tensor values.
            output_dir=run.output,
            frame_index="all",
            frame_step=1,
            warmup_frames=0,
            iterations=1,
            save_modes=set(),
            frame_naming="contiguous-output-index",
            metrics_collector=run.metrics,
            nsight_capture=run.nsight_capture,
        )
        run.metrics_factory.assert_called_once_with(
            mode="render", run_id=run.run_output.name, frame_metrics=True, nsight=False
        )
        run.metrics.write_metrics_to_yaml.assert_called_once_with(str(run.output))
        selected_rigs = run.lidar_helper.call_args.args[1]
        self.assertEqual(selected_rigs.rig_trajectories, run.rigs.rig_trajectories)
        self.assertEqual(selected_rigs.camera_calibrations, run.rigs.camera_calibrations)
        run.camera_provider_factory.assert_not_called()
        run.training_mapping.assert_not_called()
        run.timer_factory.assert_not_called()
        self.assertFalse(run.calls)

    def test_lidar_reuses_model_options_and_forwards_profiling_options(self):
        run = self.run_render(
            [
                "--lidar",
                "--renderer",
                "gsplat",
                "--frame-index",
                "2-6",
                "--frame-step",
                "2",
                "--warmup-frames",
                "3",
                "--iterations",
                "4",
                "--save",
                "min,max",
            ]
        )
        self.assert_success(run)
        self.assertEqual(run.load.call_args.kwargs["tracks_timestamp_ray_stride"], 16)
        self.assertEqual(run.load.call_args.kwargs["config_overrides"], ("model.renderer.name=3dgut-gsplat",))
        run.lidar_helper.assert_called_once_with(
            run.renderable,
            ANY,  # Rig metadata is forwarded without comparing tensor values.
            output_dir=run.output,
            frame_index="2-6",
            frame_step=2,
            warmup_frames=3,
            iterations=4,
            save_modes={"min", "max"},
            frame_naming="contiguous-output-index",
            metrics_collector=run.metrics,
            nsight_capture=run.nsight_capture,
        )
        for metadata in run.metrics.metadata_values.values():
            self.assertEqual(metadata["warmup_frames"], 3)
            self.assertEqual(metadata["iterations"], 4)
            self.assertEqual(metadata["renderer"], "gsplat")
        run.camera_provider_factory.assert_not_called()
        run.training_mapping.assert_not_called()

    def test_lidar_nsight_child_reuses_capture_directory(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            capture = parent / "existing-lidar-run"
            capture.mkdir()
            run = self.run_render(
                ["--lidar", "--nsight", "--save", "min,max"],
                output_root=parent,
                environment={module.NSIGHT_OUTPUT_DIR_ENV: str(capture)},
            )
            self.assert_success(run)
            self.assertEqual(run.output, capture)
            self.assertEqual(list(parent.iterdir()), [capture])
            run.nsight_helper.assert_not_called()
            run.lidar_helper.assert_called_once()
            self.assertEqual(run.lidar_helper.call_args.kwargs["save_modes"], {"min", "max"})
            self.assertEqual(run.capture_lifecycle, ["capture-start", "load-model", "capture-stop", "write-manifest"])
            run.metrics_factory.assert_called_once_with(
                mode="render", run_id=capture.name, frame_metrics=True, nsight=True
            )
            settings = json.loads((capture / "render_cli_args.json").read_text())
            self.assertEqual(settings["args"]["output_dir"], str(parent))
            self.assertTrue(settings["args"]["lidar"])

    def test_lidar_missing_sensor_reports_helper_error(self):
        run = self.run_render(["--lidar"], lidar_error=ValueError("No LiDAR sensor in this artifact"))
        self.assertEqual(run.result.exit_code, 1, run.result.output)
        self.assertIn("No LiDAR sensor in this artifact", run.result.output)
        run.lidar_helper.assert_called_once()
        run.metrics.write_metrics_to_yaml.assert_not_called()
        run.camera_provider_factory.assert_not_called()
        self.assertFalse(run.calls)

    def test_lidar_rejects_explicit_rgb_options_before_loading(self):
        asset_file = tempfile.NamedTemporaryFile(suffix=".json")
        self.addCleanup(asset_file.close)
        for options in (
            ["--camera-id", "cam0"],
            ["--height", "2"],
            ["--image-scale", "0.5"],
            ["--image-format", "png"],
            ["--export-video"],
            ["--output-type", "color"],
            ["--asset-edits-schedule", asset_file.name],
            ["--asset-manifest", asset_file.name],
            ["--replicate-training-views"],
            ["--no-replicate-training-views"],
            ["--rig-translation-offset", "1", "0", "0"],
            ["--rig-rotation-offset", "0", "0", "0"],
            ["--calib-source", "training-rig-poses"],
        ):
            with self.subTest(options=options):
                run = self.run_render(["--lidar", *options])
                self.assertEqual(run.result.exit_code, 2, run.result.output)
                self.assertIn("is invalid with --lidar", run.result.output)
                self.assertIn("omit --lidar to render camera outputs", run.result.output)
                run.load.assert_not_called()
                run.lidar_helper.assert_not_called()
                run.camera_provider_factory.assert_not_called()
                run.training_mapping.assert_not_called()

    def test_warmup_repeats_first_frame_per_camera_without_skipping_frames(self):
        wall_durations = [11, 13, 17, 19, 23, 29, 31, 37]
        run = self.run_render(
            ["--warmup-frames", "2", "--iterations", "2", "--frame-step", "2", "--save", "last-iteration"],
            cameras=("cam0", "cam1"),
            frame_count=3,
            wall_durations=wall_durations,
        )
        self.assert_success(run)
        expected = []
        for camera in ("cam0", "cam1"):
            expected.extend([(camera, 0, False)] * 2 + [(camera, 0, True)] * 2 + [(camera, 2, True)] * 2)
            self.assertEqual(len(self.samples(run, "cuda_ms", camera)), 4)
            self.assertEqual(len(self.samples(run, "render_time_ms", camera)), 4)
        self.assertEqual(run.calls, expected)
        # Every warmup completes through the event timer; there is no extra device-wide drain.
        self.assertFalse(run.synchronizations)
        self.assertEqual(
            run.timer_calls,
            [(f"nurec.render.warmup(frame=0, iteration={iteration})", False) for iteration in (1, 2)]
            + [(None, True)] * 4
            + [(f"nurec.render.warmup(frame=0, iteration={iteration})", False) for iteration in (1, 2)]
            + [(None, True)] * 4,
        )
        self.assertEqual(
            run.warmup_ranges,
            [
                (f"nurec.render.warmup(frame=0, iteration={iteration})", before, before + 1)
                for before, iteration in ((0, 1), (1, 2), (6, 1), (7, 2))
            ],
        )
        self.assertEqual(len(run.saved), 8)
        self.assertTrue(all("-iteration-2." in path.name for path in self.extra_saved(run)))
        self.assertEqual(len(run.timer.milliseconds), 8)
        records = [call.kwargs for call in run.metrics.record_frame.call_args_list]
        self.assertEqual(
            [(record["sensor_id"], record["frame"], record["iteration"]) for record in records],
            [(camera, frame, iteration) for camera in ("cam0", "cam1") for frame in (0, 2) for iteration in (1, 2)],
        )
        self.assertTrue(all(record["sequence_id"] == "test-sequence" for record in records))
        np.testing.assert_allclose([record["render_time_ms"] for record in records], wall_durations)
        self.assertEqual([record["cuda_ms"] for record in records], list(range(1, 9)))
        metadata = run.metrics.metadata_values
        for name, method in (("render_time_ms", "wall_clock"), ("cuda_ms", "cuda_event")):
            self.assertEqual(metadata[name]["timing_method"], method)
            self.assertEqual(metadata[name]["warmup_frames"], 2)
            self.assertEqual(metadata[name]["iterations"], 2)

    def test_nonzero_single_frame_is_warmed_then_measured_and_saved(self):
        run = self.run_render(
            [
                "--frame-index",
                "3",
                "--frame-step",
                "2",
                "--warmup-frames",
                "2",
                "--iterations",
                "2",
                "--save",
                "last-iteration",
            ],
            frame_count=5,
        )
        self.assert_success(run)
        self.assertEqual(run.calls, [("cam0", 3, False)] * 2 + [("cam0", 3, True)] * 2)
        self.assertEqual(
            [entry[0] for entry in run.warmup_ranges],
            ["nurec.render.warmup(frame=3, iteration=1)", "nurec.render.warmup(frame=3, iteration=2)"],
        )
        self.assertEqual(
            [(call.kwargs["frame"], call.kwargs["iteration"]) for call in run.metrics.record_frame.call_args_list],
            [(3, 1), (3, 2)],
        )
        self.assertEqual([path.name for path in self.canonical_saved(run)], ["000000.png"])
        self.assertEqual([path.name for path in self.extra_saved(run)], ["render-4x2-frame-3-iteration-2.png"])
        path = self.canonical_saved(run)[0]
        with Image.open(path) as image:
            self.assertEqual(image.getpixel((0, 0)), (int(255 * 4 / 32),) * 3)
        entries = json.loads((path.parent / "timestamps.json").read_text())
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["source_frame_idx"], 3)
        self.assertEqual(entries[0]["frame_start_timestamp_us"], 300)
        self.assertEqual(entries[0]["iteration"], 2)
        self.assertEqual(entries[0]["cuda_ms"], 2.0)

    def test_nonzero_frame_range_anchors_stride_and_warmup_per_camera(self):
        run = self.run_render(
            [
                "--frame-index",
                "1-3",
                "--frame-step",
                "2",
                "--warmup-frames",
                "1",
                "--iterations",
                "2",
                "--save",
                "last-iteration",
            ],
            cameras=("cam0", "cam1"),
            frame_count=6,
        )
        self.assert_success(run)
        expected = []
        for camera in ("cam0", "cam1"):
            expected.extend([(camera, 1, False)] + [(camera, 1, True)] * 2 + [(camera, 3, True)] * 2)
            entries = json.loads((run.output / CameraOutputSubdir.COLOR / camera / "timestamps.json").read_text())
            self.assertEqual([entry["source_frame_idx"] for entry in entries], [1, 3])
            self.assertEqual(
                [entry["file_name"] for entry in entries],
                ["000000.png", "000001.png"],
            )
            self.assertEqual(len(self.samples(run, "cuda_ms", camera)), 4)
        self.assertEqual(run.calls, expected)
        self.assertEqual(len(run.saved), 8)

    def test_invalid_frame_selection_fails_before_loading_in_both_modes(self):
        for sensor_options in ([], ["--lidar"]):
            for selection in ("", "-1", "1-", "1-2-3", "all,1", "3-1", "1:3"):
                with self.subTest(sensor=sensor_options, selection=selection):
                    run = self.run_render([*sensor_options, "--frame-index", selection])
                    self.assertEqual(run.result.exit_code, 2, run.result.output)
                    self.assertIn("--frame-index", run.result.output)
                    run.load.assert_not_called()
                    run.nsight_helper.assert_not_called()
                    run.lidar_helper.assert_not_called()
                    self.assertFalse(run.calls)

    def test_frame_selection_endpoints_must_be_in_camera_range(self):
        for selection in ("2", "0-2", "1-3"):
            with self.subTest(selection=selection):
                run = self.run_render(["--frame-index", selection, "--frame-step", "3"], frame_count=2)
                self.assertEqual(run.result.exit_code, 1, run.result.output)
                self.assertIn("frame", run.result.output.lower())
                run.load.assert_called_once()
                run.camera_provider_factory.assert_called_once()
                self.assertFalse(run.calls)
                run.metrics.record_frame.assert_not_called()

    def test_save_modes_preserve_the_selected_iteration_alongside_canonical_frames(self):
        for mode, expected in (
            ("last-iteration", [(0, 2, 2), (1, 2, 4)]),
            ("all-iterations", [(0, 1, 1), (0, 2, 2), (1, 1, 3), (1, 2, 4)]),
            (None, []),
        ):
            with self.subTest(mode=mode):
                options = ["--iterations", "2"]
                if mode is not None:
                    options.extend(["--save", mode])
                run = self.run_render(options)
                self.assert_success(run)
                self.assertEqual(len(self.samples(run, "cuda_ms")), 4)
                self.assertEqual(len(self.samples(run, "render_time_ms")), 4)
                self.assert_saved_outputs(run, [("000000.png", 0, 2, 2, 2), ("000001.png", 1, 2, 4, 4)], extra=False)
                if mode is None:
                    self.assertFalse(self.extra_saved(run))
                    self.assertFalse((run.output / "color-extra").exists())
                else:
                    self.assert_saved_outputs(
                        run,
                        [
                            (f"render-4x2-frame-{frame}-iteration-{iteration}.png", frame, iteration, number, number)
                            for frame, iteration, number in expected
                        ],
                    )

    def test_extrema_span_all_iterations_and_are_retained_per_camera(self):
        for mode, expected in (
            ("min", [(0, 2, 2), (1, 1, 7)]),
            ("max", [(0, 1, 1), (0, 2, 6)]),
        ):
            with self.subTest(mode=mode):
                run = self.run_render(
                    ["--iterations", "2", "--save", mode],
                    cameras=("cam0", "cam1"),
                    durations=[9, 1, 5, 6, 2, 9, 1, 6],
                )
                self.assert_success(run)
                self.assertEqual(len(self.extra_saved(run)), 2)
                for camera, (frame, iteration, call_number) in zip(("cam0", "cam1"), expected):
                    name = f"render-4x2-frame-{frame}-iteration-{iteration}-{mode}.png"
                    self.assert_saved_outputs(
                        run, [(name, frame, iteration, call_number, 1 if mode == "min" else 9)], camera=camera
                    )
                    self.assertEqual(len(self.samples(run, "cuda_ms", camera)), 4)

    def test_extrema_prepare_uint8_images_once_without_final_downloads_or_rescaling(self):
        run = self.run_render(["--iterations", "2", "--save", "min,max"], durations=[9, 1, 5, 6])
        self.assert_success(run)
        self.assertEqual(len(run.pinned_buffers), 1)
        self.assertEqual(run.pinned_copies, [(0, 1), (0, 2), (0, 4)])
        self.assertEqual(
            run.copy_ranges,
            [
                {
                    "label": "nurec.render.image_copy(selection=min,max, frame=0, iteration=1)",
                    "after_render": 1,
                    "operations": ["uint8", "copy_cpu"],
                },
                {
                    "label": "nurec.render.image_copy(selection=index,min, frame=0, iteration=2)",
                    "after_render": 2,
                    "operations": ["uint8", "copy_cpu"],
                },
                {
                    "label": "nurec.render.image_copy(selection=index, frame=1, iteration=2)",
                    "after_render": 4,
                    "operations": ["uint8", "copy_cpu"],
                },
            ],
        )
        self.assertEqual(len(run.calls), 4)
        for selection, iteration in (("min", 2), ("max", 1)):
            path = run.output / "color-extra" / "cam0" / f"render-4x2-frame-0-iteration-{iteration}-{selection}.png"
            with Image.open(path) as image:
                self.assertEqual(image.getpixel((0, 0)), (int(255 * iteration / 32),) * 3)

    def test_ordinary_copy_ranges_include_frame_and_iteration_identity(self):
        run = self.run_render(["--iterations", "2", "--save", "all-iterations"])
        self.assert_success(run)
        self.assertEqual(
            [entry["label"] for entry in run.copy_ranges],
            [
                f"nurec.render.image_copy(selection=index, frame={frame}, iteration={iteration})"
                for frame in range(2)
                for iteration in (1, 2)
            ],
        )
        self.assertEqual([entry["after_render"] for entry in run.copy_ranges], [1, 2, 3, 4])
        self.assertTrue(all(entry["operations"] == ["uint8", "copy_cpu"] for entry in run.copy_ranges))
        self.assertEqual(len(run.pinned_buffers), 1)
        self.assertEqual(run.pinned_copies, [(0, 1), (0, 2), (0, 3), (0, 4)])

    def test_prepared_extrema_remain_correct_as_the_native_framebuffer_is_overwritten(self):
        run = self.run_render(["--iterations", "6", "--save", "min,max"], frame_count=1, durations=[5, 4, 3, 6, 7, 2])
        self.assert_success(run)
        self.assertEqual([entry["after_render"] for entry in run.copy_ranges], list(range(1, 7)))
        self.assertEqual(len(run.pinned_buffers), 1)
        self.assertEqual(run.pinned_copies, [(0, iteration) for iteration in range(1, 7)])
        self.assertTrue(all(entry["operations"] == ["uint8", "copy_cpu"] for entry in run.copy_ranges))
        self.assertEqual(len(run.timer.milliseconds), 6)
        for selection, iteration in (("min", 6), ("max", 5)):
            path = run.output / "color-extra" / "cam0" / f"render-4x2-frame-0-iteration-{iteration}-{selection}.png"
            with Image.open(path) as image:
                self.assertEqual(image.getpixel((0, 0)), (int(255 * iteration / 32),) * 3)

    def test_combined_extrema_preserve_both_winners_per_camera(self):
        for mode in ("min,max", "max,min"):
            with self.subTest(mode=mode):
                run = self.run_render(
                    ["--iterations", "2", "--save", mode],
                    cameras=("cam0", "cam1"),
                    durations=[9, 1, 5, 6, 2, 9, 1, 6],
                )
                self.assert_success(run)
                self.assertEqual(len(run.calls), 8)
                self.assertEqual(len(self.extra_saved(run)), 4)
                self.assertEqual(len(run.pinned_buffers), 2)
                self.assertEqual(run.pinned_copies, [(0, 1), (0, 2), (0, 4), (1, 5), (1, 6), (1, 7), (1, 8)])
                for camera, expected in (
                    ("cam0", [("min", 0, 2, 2, 1), ("max", 0, 1, 1, 9)]),
                    ("cam1", [("min", 1, 1, 7, 1), ("max", 0, 2, 6, 9)]),
                ):
                    self.assertEqual(len(self.samples(run, "cuda_ms", camera)), 4)
                    self.assert_saved_outputs(
                        run,
                        [
                            (
                                f"render-4x2-frame-{frame}-iteration-{iteration}-{suffix}.png",
                                frame,
                                iteration,
                                number,
                                ms,
                            )
                            for suffix, frame, iteration, number, ms in expected
                        ],
                        camera=camera,
                    )

    def test_mixed_save_modes_keep_ordinary_renders_and_labeled_extrema(self):
        for mode, ordinary in (
            ("last-iteration,min,max", [(0, 2, 2), (1, 2, 4)]),
            ("all-iterations,min,max,last-iteration", [(0, 1, 1), (0, 2, 2), (1, 1, 3), (1, 2, 4)]),
        ):
            with self.subTest(mode=mode):
                durations = [9, 1, 5, 6]
                run = self.run_render(["--iterations", "2", "--save", mode], durations=durations)
                self.assert_success(run)
                self.assertEqual(len(run.calls), 4)
                self.assertEqual(self.samples(run, "cuda_ms"), durations)
                selected = (
                    [("index,min,max", 0, 1, 1), ("index,min", 0, 2, 2), ("index", 1, 1, 3), ("index", 1, 2, 4)]
                    if "all-iterations" in mode
                    else [("min,max", 0, 1, 1), ("index,min", 0, 2, 2), ("index", 1, 2, 4)]
                )
                self.assertEqual(
                    run.copy_ranges,
                    [
                        {
                            "label": f"nurec.render.image_copy(selection={selection}, frame={frame}, iteration={iteration})",
                            "after_render": call_number,
                            "operations": ["uint8", "copy_cpu"],
                        }
                        for selection, frame, iteration, call_number in selected
                    ],
                )
                self.assertEqual(len(run.pinned_buffers), 1)
                self.assertEqual(run.pinned_copies, [(0, call_number) for _, _, _, call_number in selected])
                expected = [(frame, iteration, call_number, "") for frame, iteration, call_number in ordinary]
                expected.extend([(0, 2, 2, "-min"), (0, 1, 1, "-max")])
                self.assertEqual(len(self.extra_saved(run)), len(expected))
                self.assert_saved_outputs(
                    run,
                    [
                        (
                            f"render-4x2-frame-{frame}-iteration-{iteration}{suffix}.png",
                            frame,
                            iteration,
                            number,
                            durations[number - 1],
                        )
                        for frame, iteration, number, suffix in expected
                    ],
                )

    def test_combined_extrema_save_first_render_for_single_sample_and_ties(self):
        for mode, frame_count, suffixes in (
            ("min,max", 1, ("min", "max")),
            (" max, min , min ", 2, ("min", "max")),
            ("min,min", 2, ("min",)),
            ("max,max", 2, ("max",)),
            ("min", 2, ("min",)),
            ("max", 2, ("max",)),
        ):
            with self.subTest(mode=mode, frame_count=frame_count):
                run = self.run_render(["--save", mode], frame_count=frame_count, durations=[3] * frame_count)
                self.assert_success(run)
                self.assertEqual(len(run.calls), frame_count)
                self.assertEqual(len(run.copy_ranges), frame_count)
                self.assertEqual(run.copy_ranges[0]["after_render"], 1)
                self.assertEqual(len(run.pinned_buffers), 1)
                self.assertEqual(run.pinned_copies, [(0, iteration) for iteration in range(1, frame_count + 1)])
                self.assertEqual(run.copy_ranges[0]["operations"], ["uint8", "copy_cpu"])
                self.assertEqual(
                    {path.name for path in self.extra_saved(run)},
                    {f"render-4x2-frame-0-iteration-1-{suffix}.png" for suffix in suffixes},
                )
                self.assertEqual(len(self.extra_saved(run)), len(suffixes))
                for path in self.extra_saved(run):
                    with Image.open(path) as image:
                        self.assertEqual(image.getpixel((0, 0)), (int(255 / 32),) * 3)

    def test_explicit_frame_naming_only_changes_canonical_outputs(self):
        for naming, stems in (
            ("contiguous-output-index", ["000000", "000001"]),
            ("frame-end-timestamp", ["00000000000000000099", "00000000000000000199"]),
        ):
            with self.subTest(naming=naming):
                run = self.run_render(["--frame-naming", naming, "--save", "last-iteration"])
                self.assert_success(run)
                self.assertEqual([path.name for path in self.canonical_saved(run)], [f"{stem}.png" for stem in stems])
                self.assertEqual(
                    [path.name for path in self.extra_saved(run)],
                    [f"render-4x2-frame-{frame}-iteration-1.png" for frame in range(2)],
                )

    def test_descriptive_extras_keep_independent_manifest_indices(self):
        run = self.run_render(
            ["--frame-naming", "contiguous-output-index", "--iterations", "2", "--save", "all-iterations,min,max"],
            durations=[9, 1, 5, 6],
        )
        self.assert_success(run)
        self.assertEqual([path.name for path in self.canonical_saved(run)], ["000000.png", "000001.png"])
        self.assertEqual(
            [path.name for path in self.extra_saved(run)],
            [
                "render-4x2-frame-0-iteration-1.png",
                "render-4x2-frame-0-iteration-2.png",
                "render-4x2-frame-1-iteration-1.png",
                "render-4x2-frame-1-iteration-2.png",
                "render-4x2-frame-0-iteration-2-min.png",
                "render-4x2-frame-0-iteration-1-max.png",
            ],
        )
        self.assertEqual([row["render_frame_idx"] for row in self.manifest(run)], [0, 1])
        self.assertEqual([row["render_frame_idx"] for row in self.manifest(run, extra=True)], list(range(6)))
        self.assertEqual(run.pinned_copies, [(0, 1), (0, 2), (0, 3), (0, 4)])

    def test_video_and_ground_truth_sequence_use_only_canonical_files_with_extras(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        for naming, filenames in (
            ("contiguous-output-index", [f"{frame:06d}.png" for frame in range(12)]),
            ("frame-end-timestamp", [f"{frame * 100 + 99:020d}.png" for frame in range(12)]),
        ):
            with self.subTest(naming=naming), patch.object(module, "export_videos_and_json") as export:
                run = self.run_render(
                    [
                        "--export-video",
                        "--frame-naming",
                        naming,
                        "--iterations",
                        "2",
                        "--save",
                        "all-iterations,min,max",
                    ],
                    frame_count=12,
                )
                self.assert_success(run)
                self.assertEqual(export.call_args.kwargs["frames_subdir"], CameraOutputSubdir.COLOR)
                canonical_dir = run.output / CameraOutputSubdir.COLOR / "cam0"
                self.assertEqual(sorted(path.name for path in canonical_dir.glob("*.png")), filenames)
                self.assertEqual(len(self.extra_saved(run)), 26)
                with (
                    patch.object(module.iio, "imread", return_value=np.zeros((2, 4, 3), dtype=np.uint8)) as read,
                    patch.object(module.iio, "imwrite"),
                    patch.object(module, "get_codec_config", return_value=("libx264", [])),
                ):
                    self.assertTrue(module.export_video(canonical_dir, run.output / "video.mp4"))
                self.assertEqual([Path(call.args[0]).name for call in read.call_args_list], filenames)
                self.assertTrue(all(Path(call.args[0]).parent == canonical_dir for call in read.call_args_list))

    def test_invalid_benchmark_options_fail_before_loading(self):
        for options in (
            ["--frame-naming", "frame-index"],
            ["--warmup-frames", "-1"],
            ["--iterations", "0"],
            ["--iterations", "-1"],
            ["--save", "invalid"],
            ["--save", ""],
            ["--save", "none,min"],
            ["--save", "max,none"],
            ["--save", "none,last-iteration"],
            ["--save", "all-iterations,none"],
            ["--save", "min,"],
            ["--save", "none", "--export-video"],
        ):
            with self.subTest(options=options):
                run = self.run_render(options)
                self.assertEqual(run.result.exit_code, 2, run.result.output)
                self.assertFalse(run.calls)
                run.load.assert_not_called()


class TestVideoFrameSelection(unittest.TestCase):
    """Video export validates camera paths and uses filename order, matching main."""

    def test_camera_ids_are_validated_before_exporting_any_video(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            video_root = output / CameraOutputSubdir.COLOR
            (video_root / "front").mkdir(parents=True)
            outside = root / "outside"
            outside.mkdir()
            sentinel = outside / "video.mp4"
            sentinel.write_bytes(b"existing video must remain untouched")
            for camera in (
                "",
                ".",
                "..",
                "../../outside",
                str(outside),
                "nested/camera",
                "nested\\camera",
                "camera\0name",
                "C:camera",
            ):
                with self.subTest(camera=camera), patch.object(module, "export_video", return_value=True) as export:
                    with self.assertRaises(ValueError):
                        module.export_videos_and_json(str(output), ["front", camera])
                    export.assert_not_called()
                    self.assertFalse((video_root / "render_cli_videos.json").exists())
                    self.assertEqual(sentinel.read_bytes(), b"existing video must remain untouched")

    def test_video_export_uses_sorted_files_without_reading_manifest(self):
        module = importlib.import_module("nurec.utils.io.export.render")
        for manifest in (None, "[]", '[{"file_name": "missing.png"}]', "{"):
            with self.subTest(manifest=manifest), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ("000010.png", "000002.png", "stale.png"):
                    (root / name).touch()
                if manifest is not None:
                    (root / "timestamps.json").write_text(manifest)
                with (
                    patch.object(module.iio, "imread", return_value=np.zeros((2, 4, 3), dtype=np.uint8)) as read,
                    patch.object(module.iio, "imwrite") as write,
                    patch.object(module, "get_codec_config", return_value=("libx264", [])),
                ):
                    self.assertTrue(module.export_video(root, root / "video.mp4"))
                self.assertEqual(
                    [Path(call.args[0]).name for call in read.call_args_list],
                    ["000002.png", "000010.png", "stale.png"],
                )
                write.assert_called_once()


class TestExportVideoCodec(unittest.TestCase):
    def test_frames_nvenc_rejects_get_av1_with_matching_flags(self):
        from imageio_ffmpeg import _video_codec as video_codec

        module = importlib.import_module("nurec.utils.io.export.render")

        def run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0 if "nullsrc=s=256x256:d=0.1" in cmd else 1, "", "")

        video_codec._nvenc_accepts.cache_clear()
        self.addCleanup(video_codec._nvenc_accepts.cache_clear)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(video_codec, "_nvenc_state", None),
            patch.object(video_codec.subprocess, "run", side_effect=run),
            patch.object(module.iio, "imread", return_value=np.zeros((2, 4, 3), dtype=np.uint8)),
            patch.object(module.iio, "imwrite") as write,
        ):
            Path(tmp, "000000.png").touch()
            self.assertTrue(module.export_video(Path(tmp), Path(tmp) / "video.mp4"))

        self.assertEqual(write.call_args.kwargs["codec"], "libsvtav1")
        self.assertEqual(write.call_args.kwargs["macro_block_size"], 2)
        self.assertEqual(write.call_args.kwargs["output_params"][:2], ["-crf", "20"])


if __name__ == "__main__":
    unittest.main()


class ExportVideosLayoutTest(unittest.TestCase):
    """Where the exported video lands is a contract with eval-rendering-metrics.

    Its --render-dir help states the object-level input is <render-dir>/<image-seq>/video.mp4.
    Writing it a directory higher makes object-level metrics report no video and skip, which
    exits zero -- so nothing fails, and the metrics are simply absent.
    """

    def test_video_is_written_inside_its_camera_directory(self):
        with tempfile.TemporaryDirectory() as out_dir:
            subdir = CameraOutputSubdir.COLOR.value
            camera = "camera_front_wide_120fov"
            os.makedirs(os.path.join(out_dir, subdir, camera))

            with patch("nurec.utils.io.export.render.export_video", return_value=True) as encode:
                export_videos_and_json(out_dir, [camera], fps=30.0, crf=20, image_extension="png")

            written = encode.call_args.args[1]
            self.assertEqual(
                os.path.normpath(str(written)),
                os.path.normpath(os.path.join(out_dir, subdir, camera, "video.mp4")),
                "eval-rendering-metrics lists the camera directory; a sibling file is invisible to it",
            )

    def test_index_records_the_path_it_wrote(self):
        with tempfile.TemporaryDirectory() as out_dir:
            subdir = CameraOutputSubdir.COLOR.value
            camera = "camera_front_wide_120fov"
            os.makedirs(os.path.join(out_dir, subdir, camera))

            with patch("nurec.utils.io.export.render.export_video", return_value=True):
                export_videos_and_json(out_dir, [camera], fps=30.0, crf=20, image_extension="png")

            with open(os.path.join(out_dir, subdir, "render_cli_videos.json")) as f:
                index = json.load(f)
            # The index is relative to the output-type subdirectory, so a reader resolving it
            # against that directory lands on the same file export_video was handed.
            self.assertEqual(index[camera], f"{camera}/video.mp4")
