# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placeholder regression test for the template converter.

Single `py_test` exercising V4-reader, validator smoke, and eval smoke
against the placeholder converter, all in the template Bazel workspace.

Cross-workspace dependency contract
-----------------------------------
The template workspace owns the converter (`//ncore_example:lib`) and its
Bazel-managed pip deps, but the validator/eval binaries live in the
enclosing repository's root workspace
(`//skills/nurec-onboard/ncore_validation:validate`, `:eval`). Because
`.bazelignore` excludes the template from the root workspace's `//...`
glob, neither workspace can construct labels into the other. CI bridges this by building the
binaries on the platform side and forwarding their paths into the test
sandbox as environment variables (`NCORE_VALIDATE_BIN`, `NCORE_EVAL_BIN`)
via `--test_env`. Tests that depend on the bridged binaries are guarded
by `@unittest.skipUnless` so a developer running this file from inside
the template without the platform build still gets useful partial
output (the V4-shard assertions run; the binary-driven assertions skip
cleanly).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest

from pathlib import Path

import numpy as np
import yaml

from ncore.data.v4 import (
    CameraSensorComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
)

from ncore_example.converter import (
    ExampleConverter,
    ExampleConverterConfig,
)


SEQUENCE_ID = "example_sequence_001"

NCORE_VALIDATE_BIN = os.environ.get("NCORE_VALIDATE_BIN")
NCORE_EVAL_BIN = os.environ.get("NCORE_EVAL_BIN")

NCORE_EXAMPLE_CONVERT_BIN_RLOCATION = os.environ.get("NCORE_EXAMPLE_CONVERT_BIN_RLOCATION")
NCORE_EXAMPLE_PROBE_BIN_RLOCATION = os.environ.get("NCORE_EXAMPLE_PROBE_BIN_RLOCATION")


def _run_example_converter(base_dir: str | Path) -> tuple[Path, Path]:
    """Run ExampleConverter into ``base_dir`` and return ``(sequence_dir, sequence_meta_path)``.

    Creates ``root/`` and ``out/`` subdirectories under ``base_dir``, writes a
    single sequence with all sensor types enabled, and returns paths to the
    sequence output directory and its ``{sequence_id}.json`` meta file.

    Args:
        base_dir: Parent directory for the converter's ``root/`` and ``out/`` trees.

    Returns:
        Tuple of ``(sequence_dir, sequence_meta_path)`` where ``sequence_dir`` is
        the output sequence directory and ``sequence_meta_path`` is the
        ``{sequence_id}.json`` file inside it.
    """
    base = Path(base_dir)
    root_dir = base / "root"
    root_dir.mkdir(exist_ok=True)
    output_dir = base / "out"
    config = ExampleConverterConfig(
        root_dir=str(root_dir),
        output_dir=str(output_dir),
        store_type="directory",
        component_group_profile="separate-sensors",
        store_sequence_meta=True,
        no_cameras=False,
        camera_ids=None,
        no_lidars=False,
        lidar_ids=None,
        no_radars=False,
        radar_ids=None,
        verbose=False,
        debug=False,
        debug_port=5678,
    )
    ExampleConverter.from_config(config).convert(config)
    seq_dir = output_dir / SEQUENCE_ID
    sequence_meta_path = seq_dir / f"{SEQUENCE_ID}.json"
    return seq_dir, sequence_meta_path


class NcoreExampleShardTest(unittest.TestCase):
    """Run the skeleton converter and verify V4 output with all component types.

    These assertions depend only on the in-process converter and the
    NCore V4 reader; they always run, with no env-var prerequisites.
    """

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _run_converter(self) -> tuple[Path, Path]:
        """Run ExampleConverter and return ``(sequence_dir, sequence_meta_path)``.

        Returns:
            Tuple of ``(sequence_dir, sequence_meta_path)`` for the generated sequence.
        """
        return _run_example_converter(self._tmpdir.name)

    def test_produces_v4_output_and_sequence_meta(self) -> None:
        """Run ExampleConverter; verify sequence meta, component groups, and NuRec readiness."""
        seq_dir, sequence_meta_path = self._run_converter()

        self.assertTrue(seq_dir.is_dir(), f"Expected sequence dir {seq_dir}")
        self.assertTrue(sequence_meta_path.is_file(), f"Expected sequence meta {sequence_meta_path}")

        with sequence_meta_path.open() as f:
            meta = json.load(f)
        self.assertIn("sequence_id", meta)
        self.assertEqual(meta["sequence_id"], SEQUENCE_ID)
        self.assertIn("sequence_timestamp_interval_us", meta)
        interval = meta["sequence_timestamp_interval_us"]
        self.assertIsInstance(interval, dict, "sequence_timestamp_interval_us must be a dict")
        self.assertIn("start", interval)
        self.assertIn("stop", interval)
        self.assertLess(interval["start"], interval["stop"])

        all_components: dict[str, list[str]] = {}
        for store in meta["component_stores"]:
            for component_type, instances in store.get("components", {}).items():
                all_components.setdefault(component_type, []).extend(instances.keys())
        self.assertIn("poses", all_components)
        self.assertIn("intrinsics", all_components)
        self.assertIn("masks", all_components)
        self.assertIn("cameras", all_components)
        self.assertIn("pinhole_camera", all_components["cameras"])
        self.assertIn("lidars", all_components)
        self.assertIn("lidar_top", all_components["lidars"])
        self.assertIn("radars", all_components)
        self.assertIn("radar_front", all_components["radars"])
        self.assertIn("cuboids", all_components)

    def test_lidar_direction_unit_norm(self) -> None:
        """Verify that LiDAR direction vectors in the output are unit-norm."""
        _, sequence_meta_path = self._run_converter()
        reader = SequenceComponentGroupsReader([sequence_meta_path])

        lidar_readers = reader.open_component_readers(LidarSensorComponent.Reader)
        self.assertGreaterEqual(len(lidar_readers), 1)

        for lidar_id, lidar_reader in lidar_readers.items():
            frame_timestamps = lidar_reader.frames_timestamps_us
            self.assertGreater(frame_timestamps.size, 0, f"LiDAR {lidar_id} has no frames")
            self.assertEqual(frame_timestamps.ndim, 2, f"LiDAR {lidar_id} timestamps should be 2D")
            self.assertGreaterEqual(frame_timestamps.shape[1], 2, f"LiDAR {lidar_id} timestamps need >= 2 columns")
            frame_end_ts = int(frame_timestamps[0, 1])
            direction = lidar_reader.get_frame_ray_bundle_data(frame_end_ts, "direction")
            self.assertGreater(direction.shape[0], 0, f"LiDAR {lidar_id} frame has no points")
            norms = np.linalg.norm(direction, axis=1)
            np.testing.assert_allclose(
                norms,
                1.0,
                atol=1e-3,
                err_msg=f"LiDAR {lidar_id} direction vectors not unit-norm",
            )

    def test_lidar_model_element_matches_grid(self) -> None:
        """NuRec / NCore types: model_element [N,2] uint16 within spinning lidar grid."""
        _, sequence_meta_path = self._run_converter()
        reader = SequenceComponentGroupsReader([sequence_meta_path])
        lidar_readers = reader.open_component_readers(LidarSensorComponent.Reader)
        intrinsics_readers = reader.open_component_readers(IntrinsicsComponent.Reader)
        for lidar_id, lidar_reader in lidar_readers.items():
            params = None
            for intrinsics_reader in intrinsics_readers.values():
                params = intrinsics_reader.get_lidar_model_parameters(lidar_id)
                if params is not None:
                    break
            self.assertIsNotNone(params, f"{lidar_id}: no intrinsics")
            timestamps = lidar_reader.frames_timestamps_us
            self.assertGreater(timestamps.size, 0, f"LiDAR {lidar_id} has no frames")
            self.assertEqual(timestamps.ndim, 2, f"LiDAR {lidar_id} timestamps should be 2D")
            self.assertGreaterEqual(timestamps.shape[1], 2, f"LiDAR {lidar_id} timestamps need >= 2 columns")
            frame_end_ts = int(timestamps[0, 1])
            self.assertTrue(
                lidar_reader.has_frame_ray_bundle_data(frame_end_ts, "model_element"),
                f"{lidar_id}: expected model_element",
            )
            direction = lidar_reader.get_frame_ray_bundle_data(frame_end_ts, "direction")
            model_element = lidar_reader.get_frame_ray_bundle_data(frame_end_ts, "model_element")
            self.assertEqual(model_element.dtype, np.dtype(np.uint16))
            self.assertEqual(model_element.shape, (direction.shape[0], 2))
            self.assertTrue(
                (model_element[:, 0] < params.n_rows).all(), f"{lidar_id}: row index >= n_rows ({params.n_rows})"
            )
            self.assertTrue(
                (model_element[:, 1] < params.n_columns).all(),
                f"{lidar_id}: col index >= n_columns ({params.n_columns})",
            )

    def test_dynamic_pose_rig_moves(self) -> None:
        """Verify that the rig→world dynamic pose shows forward displacement."""
        _, sequence_meta_path = self._run_converter()
        reader = SequenceComponentGroupsReader([sequence_meta_path])

        poses_readers = reader.open_component_readers(PosesComponent.Reader)
        self.assertGreaterEqual(len(poses_readers), 1)

        for pose_reader in poses_readers.values():
            try:
                poses, _ts = pose_reader.get_dynamic_pose("rig", "world")
                break
            except KeyError:
                continue
        else:
            self.fail("No rig→world dynamic pose found")

        self.assertGreaterEqual(poses.shape[0], 2, "Need at least 2 poses")
        displacement = float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]))
        self.assertGreater(displacement, 1.0, "Rig should move forward")

    def test_static_pose_translation_sanity(self) -> None:
        """Verify that camera→rig and lidar→rig static poses exist with reasonable translations."""
        _, sequence_meta_path = self._run_converter()
        reader = SequenceComponentGroupsReader([sequence_meta_path])

        poses_readers = reader.open_component_readers(PosesComponent.Reader)
        self.assertGreaterEqual(len(poses_readers), 1)

        max_translation_m = 20.0
        expected_sensors = ["pinhole_camera", "lidar_top"]
        for sensor_id in expected_sensors:
            found = False
            for pose_reader in poses_readers.values():
                try:
                    pose = pose_reader.get_static_pose(sensor_id, "rig")
                    found = True
                    translation_norm = float(np.linalg.norm(pose[:3, 3]))
                    self.assertLess(
                        translation_norm,
                        max_translation_m,
                        f"{sensor_id}→rig translation {translation_norm:.1f} m exceeds {max_translation_m} m",
                    )
                    break
                except KeyError:
                    continue
            self.assertTrue(found, f"No static pose found for {sensor_id}→rig")

    def test_camera_frame_count(self) -> None:
        """Verify that each camera has at least 2 frames (needed for rate metrics)."""
        _, sequence_meta_path = self._run_converter()
        reader = SequenceComponentGroupsReader([sequence_meta_path])

        camera_readers = reader.open_component_readers(CameraSensorComponent.Reader)
        self.assertGreaterEqual(len(camera_readers), 1)
        for cam_id, cam_reader in camera_readers.items():
            n_frames = cam_reader.frames_timestamps_us.shape[0]
            self.assertGreaterEqual(n_frames, 2, f"{cam_id}: expected >= 2 frames, got {n_frames}")


@unittest.skipUnless(NCORE_VALIDATE_BIN, "NCORE_VALIDATE_BIN not set; skipping validator-binary tests")
class NcoreExampleValidateTest(unittest.TestCase):
    """Run the example converter and assert the platform validator binary exits 0.

    Driven by the `NCORE_VALIDATE_BIN` env var, populated by CI from the
    root-workspace `bazel build //skills/nurec-onboard/ncore_validation:validate` and
    forwarded into the test sandbox via `--test_env=NCORE_VALIDATE_BIN`.
    Local runs without the CI prelude skip via the class-level decorator.
    """

    def test_validate_exits_zero_on_example_output(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        _, sequence_meta_path = _run_example_converter(tmp.name)
        self.assertTrue(sequence_meta_path.is_file(), f"missing sequence meta {sequence_meta_path}")

        proc = subprocess.run(
            [NCORE_VALIDATE_BIN, "--component-group", str(sequence_meta_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        self.assertEqual(proc.returncode, 0, msg=combined)

        self.assertIn("Validation passed", combined)
        self.assertNotIn("\u2717", combined, "unexpected failure marker in output")

        for section in ["V4 Structure", "NuRec Readiness", "Data Quality"]:
            self.assertIn(section, proc.stderr, f"missing check section: {section}")

    def test_validate_writes_well_formed_result_sidecar(self) -> None:
        """``validate --output-result`` writes a sidecar matching the published schema."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        _, sequence_meta_path = _run_example_converter(tmp.name)
        result_path = Path(tmp.name) / "validate_result.yaml"

        proc = subprocess.run(
            [
                NCORE_VALIDATE_BIN,
                "--component-group",
                str(sequence_meta_path),
                "--output-result",
                str(result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=(proc.stdout or "") + (proc.stderr or ""),
        )

        self.assertTrue(result_path.is_file(), f"validate did not write sidecar at {result_path}")
        with result_path.open() as f:
            result = yaml.safe_load(f)
        self.assertIsInstance(result, dict, "sidecar root must be a YAML mapping")

        # Schema-identity constants live in the sibling platform workspace
        # (`ncore_validation/validate_helpers.py::VALIDATE_RESULT_SCHEMA_VERSION`
        # and `VALIDATE_RESULT_PROGRAM`); duplicated here as literals because
        # `.bazelignore` forbids importing across the workspace boundary. Bump
        # them together.
        self.assertEqual(result.get("schema_version"), 1)
        self.assertEqual(result.get("program"), "ncore_validation/validate.py")
        self.assertIsInstance(result.get("program_version"), str)
        self.assertTrue(result["program_version"], "program_version must be non-empty")

        self.assertIs(result.get("passed"), True, msg=result)
        self.assertEqual(result.get("exit_code"), 0, msg=result)
        self.assertEqual(result["exit_code"], proc.returncode)
        self.assertEqual(result.get("failed_checks"), [], msg=result)
        self.assertIsInstance(result.get("warnings"), list)

        self.assertEqual(result.get("component_groups"), [str(sequence_meta_path)])
        self.assertIsNone(result.get("sample_frames"), msg=result)
        self.assertIsInstance(result.get("started_at"), str)
        self.assertIsInstance(result.get("ended_at"), str)
        duration_sec = result.get("duration_sec")
        # Exclude bool explicitly: ``bool`` subclasses ``int`` in Python.
        self.assertNotIsInstance(duration_sec, bool, msg=result)
        self.assertIsInstance(duration_sec, (int, float))
        self.assertGreaterEqual(duration_sec, 0.0)


@unittest.skipUnless(NCORE_EVAL_BIN, "NCORE_EVAL_BIN not set; skipping eval-binary tests")
class NcoreExampleEvalTest(unittest.TestCase):
    """Run the example converter and assert the platform eval binary produces valid JSON.

    Driven by the `NCORE_EVAL_BIN` env var, populated by CI from the
    root-workspace `bazel build //skills/nurec-onboard/ncore_validation:eval` and
    forwarded into the test sandbox via `--test_env=NCORE_EVAL_BIN`.
    Local runs without the CI prelude skip via the class-level decorator.
    """

    def test_eval_produces_json(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        _, sequence_meta_path = _run_example_converter(tmp.name)
        self.assertTrue(sequence_meta_path.is_file(), f"missing sequence meta {sequence_meta_path}")

        json_output = Path(tmp.name) / "metrics.json"
        proc = subprocess.run(
            [
                NCORE_EVAL_BIN,
                "--component-group",
                str(sequence_meta_path),
                "--output-json",
                str(json_output),
                "--quiet",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=(proc.stdout or "") + (proc.stderr or ""),
        )

        self.assertTrue(json_output.is_file(), "eval did not produce JSON output")
        with json_output.open() as f:
            data = json.load(f)
        self.assertIn("schema_version", data)
        self.assertEqual(data["schema_version"], 1)
        self.assertIn("sequence_id", data)
        self.assertIn("metrics", data)

        metrics = data["metrics"]
        seq = "__sequence__"

        with self.subTest(group="poses"):
            self.assertIn("poses", metrics)
            poses = metrics["poses"][seq]
            for m in [
                "rate_hz",
                "jitter_us",
                "temporal_coverage_pct",
                "total_displacement_m",
                "max_angular_rate_deg_s",
                "rig_forward_alignment",
            ]:
                self.assertIn(m, poses, f"poses metric missing: {m}")
            self.assertAlmostEqual(poses["rate_hz"]["value"], 20.0, places=1)
            self.assertAlmostEqual(poses["total_displacement_m"]["value"], 10.0, places=1)
            self.assertAlmostEqual(poses["rig_forward_alignment"]["value"], 1.0, places=2)

        with self.subTest(group="camera"):
            self.assertIn("camera", metrics)
            cam_seq = metrics["camera"][seq]
            for m in ["count", "frame_balance", "mask_presence"]:
                self.assertIn(m, cam_seq, f"camera sequence metric missing: {m}")
            self.assertAlmostEqual(cam_seq["count"]["value"], 3.0)
            self.assertAlmostEqual(cam_seq["frame_balance"]["value"], 1.0)
            self.assertAlmostEqual(cam_seq["mask_presence"]["value"], 1.0)
            for cam_id in ["pinhole_camera", "fisheye_camera", "ftheta_camera"]:
                self.assertIn(cam_id, metrics["camera"], f"camera sensor missing: {cam_id}")
                camera = metrics["camera"][cam_id]
                for m in ["frame_rate_hz", "resolution_w", "resolution_h"]:
                    self.assertIn(m, camera, f"camera/{cam_id} metric missing: {m}")
                self.assertAlmostEqual(camera["resolution_w"]["value"], 1920.0)
                self.assertAlmostEqual(camera["resolution_h"]["value"], 1080.0)

        with self.subTest(group="lidar"):
            self.assertIn("lidar", metrics)
            for lidar_id in ["lidar_top"]:
                self.assertIn(lidar_id, metrics["lidar"], f"lidar sensor missing: {lidar_id}")
                lidar = metrics["lidar"][lidar_id]
                for m in [
                    "points_per_frame",
                    "distance_min_m",
                    "distance_max_m",
                    "distance_mean_m",
                    "intensity_completeness",
                    "model_angular_error_max_deg",
                    "model_angular_error_mean_deg",
                    "element_coverage_pct",
                ]:
                    self.assertIn(m, lidar, f"lidar/{lidar_id} metric missing: {m}")
                self.assertAlmostEqual(lidar["intensity_completeness"]["value"], 1.0)
                self.assertLess(lidar["model_angular_error_max_deg"]["value"], 0.1)

        with self.subTest(group="cuboids"):
            self.assertIn("cuboids", metrics)
            cuboids = metrics["cuboids"][seq]
            for m in [
                "track_count",
                "observation_density",
                "single_obs_ratio",
                "class_diversity",
                "lidar_points_per_cuboid",
                "lidar_density_pts_per_m3",
            ]:
                self.assertIn(m, cuboids, f"cuboids metric missing: {m}")
            self.assertAlmostEqual(cuboids["track_count"]["value"], 3.0)
            self.assertAlmostEqual(cuboids["class_diversity"]["value"], 1.0)
            self.assertGreater(cuboids["lidar_points_per_cuboid"]["avg"], 0.0)
            self.assertGreater(cuboids["lidar_density_pts_per_m3"]["avg"], 0.0)

        with self.subTest(group="cross_sensor"):
            self.assertIn("cross_sensor", metrics)
            cross_sensor = metrics["cross_sensor"][seq]
            for m in ["temporal_overlap_pct", "clock_consistency_ratio"]:
                self.assertIn(m, cross_sensor, f"cross_sensor metric missing: {m}")
            self.assertAlmostEqual(cross_sensor["clock_consistency_ratio"]["value"], 1.0, places=1)


@unittest.skipUnless(
    NCORE_EXAMPLE_CONVERT_BIN_RLOCATION and NCORE_EXAMPLE_PROBE_BIN_RLOCATION,
    "NCORE_EXAMPLE_{CONVERT,PROBE}_BIN_RLOCATION not set; "
    "skipping entry-point smoke tests (only set by Bazel via py_test `env`)",
)
class NcoreExampleEntryPointSmokeTest(unittest.TestCase):
    """Invoke the ``:convert`` / ``:probe`` py_binary wrappers as ``__main__``.

    The other tests in this file import ``ncore_example.converter`` and
    ``ncore_example.dataset_probe`` as library modules (``from
    ncore_example.converter import ExampleConverter``). That is a different
    execution mode than ``bazel run //ncore_example:convert``, where the same
    file runs as ``__main__`` with no parent package. Relative imports
    (``from .utils import foo``) resolve in library-import mode but raise
    ``ImportError: attempted relative import with no known parent package``
    when run as ``__main__``. Without this smoke test, that class of
    regression escapes CI: the library-import tests above pass even when
    the operator-facing ``bazel run`` entry point is broken.

    These tests invoke each py_binary wrapper via ``subprocess`` with
    ``--help`` and assert exit 0, which exercises every top-level
    ``from … import …`` in ``converter.py`` and ``dataset_probe.py`` in the
    same execution context the operator uses, without paying the cost of a
    full conversion.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Resolve the py_binary wrapper paths from runfiles via ``$(rlocationpath)``.

        Raises:
            RuntimeError: If the runfiles helper cannot initialize or either
                wrapper script is missing from the test's runfiles tree.
        """
        from python.runfiles import Runfiles

        runfiles = Runfiles.Create()
        if runfiles is None:
            raise RuntimeError(
                "python.runfiles.Runfiles.Create() returned None; "
                "cannot resolve py_binary paths from $(rlocationpath) keys"
            )
        cls.convert_bin = runfiles.Rlocation(NCORE_EXAMPLE_CONVERT_BIN_RLOCATION)
        cls.probe_bin = runfiles.Rlocation(NCORE_EXAMPLE_PROBE_BIN_RLOCATION)
        for label, path, key in (
            ("convert", cls.convert_bin, NCORE_EXAMPLE_CONVERT_BIN_RLOCATION),
            ("probe", cls.probe_bin, NCORE_EXAMPLE_PROBE_BIN_RLOCATION),
        ):
            if path is None or not Path(path).exists():
                raise RuntimeError(
                    f"{label} py_binary wrapper not found in runfiles "
                    f"(rlocation key {key!r}); ensure the BUILD `data` attr "
                    f"includes //ncore_example:{label}"
                )

    def _assert_help_exits_zero(self, binary: str, label: str) -> None:
        """Run ``<binary> --help`` and assert exit 0 with Click usage output.

        Args:
            binary: Absolute path to the py_binary wrapper script.
            label: Short label for diagnostic messages, e.g. ``"convert"``.
        """
        proc = subprocess.run(
            [binary, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        self.assertEqual(
            proc.returncode,
            0,
            msg=(
                f"`{label} --help` exited {proc.returncode}; expected 0. "
                f"If the failure is `ImportError: attempted relative import "
                f"with no known parent package`, an internal module under "
                f"`ncore_example/` is using `from .<module> import ...`; "
                f"`bazel run` executes the file as `__main__`, which "
                f"requires absolute imports "
                f"(`from ncore_example.<module> import ...`).\n"
                f"--- combined output ---\n{combined}"
            ),
        )
        self.assertRegex(
            combined,
            r"(?im)^usage:",
            msg=f"`{label} --help` produced no usage banner; combined output:\n{combined}",
        )

    def test_convert_help_resolves_main_imports(self) -> None:
        """``//ncore_example:convert --help`` exits 0 (entry-point importable as ``__main__``)."""
        self._assert_help_exits_zero(self.convert_bin, "convert")

    def test_probe_help_resolves_main_imports(self) -> None:
        """``//ncore_example:probe --help`` exits 0 (entry-point importable as ``__main__``)."""
        self._assert_help_exits_zero(self.probe_bin, "probe")


if __name__ == "__main__":
    unittest.main()
