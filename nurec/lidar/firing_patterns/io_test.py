# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import tempfile
import unittest
import zipfile

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import ncore.data

from nurec.artifact.artifact import Artifact
from nurec.config.checkpoint import ArtifactConfig
from nurec.lidar.firing_patterns.io import load_firing_patterns, serialize_firing_patterns
from nurec.lidar.firing_patterns.schema import (
    PACKET_RAY_OFFSETS_ARRAY_NAME,
    TEMPLATE_PACKET_OFFSETS_ARRAY_NAME,
    LidarFiringPatternTemplatesData,
)
from nurec.lidar.firing_patterns.templates import LidarFiringPatternTemplate
from nurec.ncore_backends.interface import UnsupportedBackendOperationError
from nurec.utils.io.artifact_cache import ArtifactCache
from nurec.utils.types import FrameConversion, RigTrajectories


class _FakeSensor:
    def __init__(
        self,
        n_rays: int = 8,
        *,
        has_templates: bool = True,
        retains_non_returns: bool = True,
    ) -> None:
        angles = np.linspace(-0.4, 0.4, n_rays)
        directions = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1).astype(np.float32)
        self.frames_count = 3
        self.frames_timestamps_us = np.asarray(
            [[1_000, 1_700], [2_000, 2_700], [3_000, 3_700]],
            dtype=np.int64,
        )
        self._directions = directions
        self._timestamp_offsets_us = np.arange(n_rays, dtype=np.int64) * 100
        self._return_distances_m = (
            np.zeros(n_rays, dtype=np.float32) if retains_non_returns else np.ones(n_rays, dtype=np.float32)
        )
        self.templates = (
            LidarFiringPatternTemplatesData(
                directions_sensor=directions,
                timestamp_offsets_us=np.arange(n_rays, dtype=np.int64) * 100,
                template_ray_offsets=np.array([0, n_rays], dtype=np.int64),
                packet_ray_offsets=np.array([0, n_rays // 2, n_rays], dtype=np.int64),
                template_packet_offsets=np.array([0, 2], dtype=np.int64),
                reference_frame_start_us=1_000,
                frame_period_us=1_000,
                frame_start_tolerance_us=10,
            )
            if has_templates
            else None
        )

    def get_frame_ray_bundle_direction(self, frame_index: int):
        del frame_index
        return self._directions

    def get_frame_ray_bundle_timestamp_us(self, frame_index: int):
        return self.frames_timestamps_us[frame_index, 0] + self._timestamp_offsets_us

    def get_frame_ray_bundle_return_distance_m(self, *, frame_index: int, return_index: int):
        del frame_index, return_index
        return self._return_distances_m


class _FakeBackend:
    def __init__(self, *, invalid_templates: bool = False, unsupported_templates: bool = False) -> None:
        self.invalid_templates = invalid_templates
        self.unsupported_templates = unsupported_templates
        self.sensors: dict[str, _FakeSensor] = {}
        self.calls: list[str] = []
        self.request = object()

    def get_lidar_firing_pattern_templates(self, request, sensor_id: str):
        assert request is self.request
        self.calls.append(sensor_id)
        if self.unsupported_templates:
            raise UnsupportedBackendOperationError("firing-pattern templates require NCore V4")
        if self.invalid_templates:
            raise ValueError("broken template descriptor")
        sensor = self.sensors.get(sensor_id)
        return None if sensor is None else sensor.templates


def _export(rig, sensors, backend, *, include_modeled_lidars=False, backend_sensors=None):
    """Export with independently configurable datasource and backend sensors."""
    backend.sensors = sensors if backend_sensors is None else backend_sensors
    dataset = SimpleNamespace(backend=backend, request=backend.request)
    return serialize_firing_patterns(rig, sensors, dataset, include_modeled_lidars=include_modeled_lidars)


def _lidar_model_parameters() -> ncore.data.ConcreteLidarModelParametersUnion:
    """Minimal spinning lidar intrinsics; only their presence matters here."""
    return ncore.data.RowOffsetStructuredSpinningLidarModelParameters(
        spinning_frequency_hz=10.0,
        spinning_direction="ccw",
        n_rows=2,
        n_columns=4,
        row_elevations_rad=np.linspace(0.1, -0.1, 2).astype(np.float32),
        column_azimuths_rad=np.linspace(-0.5, 0.5, 4).astype(np.float32),
        row_azimuth_offsets_rad=np.zeros(2, dtype=np.float32),
    )


def _rig_trajectories(*lidars: tuple[str, bool]) -> RigTrajectories:
    """Calibrations for ``(logical_id, has_model_parameters)``, keyed by unique sensor id."""
    calibrations: OrderedDict[str, RigTrajectories.LidarCalibration] = OrderedDict()
    for idx, (logical_id, has_model) in enumerate(lidars):
        calibrations[f"{logical_id}@sequence{idx}"] = RigTrajectories.LidarCalibration(
            sequence_id="sequence",
            logical_sensor_name=logical_id,
            unique_sensor_idx=idx,
            T_sensor_rig=torch.eye(4, dtype=torch.float32),
            lidar_model_parameters=_lidar_model_parameters() if has_model else None,
        )
    return RigTrajectories(
        T_world_base=torch.eye(4, dtype=torch.float64),
        world_to_nurec=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
        rig_trajectories=[],
        camera_calibrations=OrderedDict(),
        lidar_calibrations=calibrations,
    )


class TestSerializeFiringPatterns(unittest.TestCase):
    def test_writes_one_npz_per_model_free_lidar(self):
        members = _export(
            _rig_trajectories(("lidar_fu", False)),
            {"lidar_fu": _FakeSensor()},
            _FakeBackend(),
        )

        self.assertEqual([m.filename for m in members], ["firing_patterns/lidar_fu.npz"])
        templates = load_firing_patterns(io.BytesIO(members[0].serialized))
        self.assertEqual(len(templates.templates), 1)
        self.assertEqual(templates.templates[0].n_rays, 8)
        self.assertEqual(templates.templates[0].n_packets, 2)
        self.assertEqual(templates.frame_period_us, 1_000)

    def test_native_templates_do_not_require_a_datasource_sensor(self):
        members = _export(
            _rig_trajectories(("lidar_fu", False)),
            {},
            _FakeBackend(),
            backend_sensors={"lidar_fu": _FakeSensor()},
        )

        self.assertEqual([member.filename for member in members], ["firing_patterns/lidar_fu.npz"])
        templates = load_firing_patterns(io.BytesIO(members[0].serialized))
        self.assertEqual(templates.templates[0].n_rays, 8)
        self.assertEqual(templates.templates[0].n_packets, 2)

    def test_default_skips_modeled_lidar_and_keeps_model_free_lidar(self):
        # The model remains the fallback; do not recover a partial observed sweep.
        members = _export(
            _rig_trajectories(("lidar_modelled", True), ("lidar_free", False)),
            {
                "lidar_modelled": _FakeSensor(has_templates=False, retains_non_returns=False),
                "lidar_free": _FakeSensor(),
            },
            _FakeBackend(),
        )

        self.assertEqual([m.filename for m in members], ["firing_patterns/lidar_free.npz"])

    def test_modeled_lidar_does_not_require_a_recorded_sensor(self):
        rig = _rig_trajectories(("modeled", True))
        self.assertEqual(_export(rig, {}, _FakeBackend()), [])
        self.assertEqual(_export(rig, {"modeled": _FakeSensor()}, _FakeBackend()), [])

    def test_unsupported_backend_keeps_modeled_lidar_export_on_intrinsics(self):
        with patch("nurec.lidar.firing_patterns.io._recover_firing_pattern_templates") as recover:
            members = _export(
                _rig_trajectories(("modeled", True)),
                {"modeled": _FakeSensor()},
                _FakeBackend(unsupported_templates=True),
                include_modeled_lidars=True,
            )

        self.assertEqual(members, [])
        recover.assert_not_called()

    def test_unsupported_backend_rejects_model_free_export_without_recovery(self):
        with (
            patch("nurec.lidar.firing_patterns.io._recover_firing_pattern_templates") as recover,
            self.assertRaisesRegex(ValueError, "unsupported for lidar 'model_free'.*require NCore V4"),
        ):
            _export(
                _rig_trajectories(("model_free", False)),
                {"model_free": _FakeSensor(has_templates=False)},
                _FakeBackend(unsupported_templates=True),
                include_modeled_lidars=True,
            )

        recover.assert_not_called()

    def test_invalid_native_templates_do_not_fall_back_to_intrinsics(self):
        with self.assertRaisesRegex(ValueError, "invalid dataset firing-pattern templates.*broken"):
            _export(
                _rig_trajectories(("hybrid", True)),
                {"hybrid": _FakeSensor()},
                _FakeBackend(invalid_templates=True),
                include_modeled_lidars=True,
            )

    def test_duplicate_logical_ids_are_refused(self):
        # The member name is the logical id, so a second one would write a duplicate zip entry.
        rig = _rig_trajectories(("lidar_top", False), ("lidar_top", False))
        with self.assertRaisesRegex(ValueError, "share logical id"):
            _export(rig, {"lidar_top": _FakeSensor()}, _FakeBackend())

    def test_refuses_a_lidar_the_datasource_does_not_carry(self):
        with self.assertRaisesRegex(ValueError, "no datasource sensor.*lidar_fu"):
            _export(_rig_trajectories(("lidar_fu", False)), {}, _FakeBackend())

    def test_refuses_an_id_that_cannot_name_an_artifact_member(self):
        with self.assertRaisesRegex(ValueError, "contains '/'"):
            _export(
                _rig_trajectories(("bad/id", False)),
                {"bad/id": _FakeSensor()},
                _FakeBackend(),
            )

    def test_recovers_a_renderable_pattern_when_dataset_templates_are_missing(self):
        members = _export(
            _rig_trajectories(("lidar_fu", False)),
            {
                "lidar_fu": _FakeSensor(
                    has_templates=False,
                )
            },
            _FakeBackend(),
        )

        templates = load_firing_patterns(io.BytesIO(members[0].serialized))
        self.assertEqual(len(templates.templates), 1)
        self.assertEqual(templates.templates[0].n_rays, 8)
        self.assertIsNone(templates.templates[0].packet_ray_offsets)

    def test_modeled_lidar_defaults_to_intrinsics_without_reading_components(self):
        backend = _FakeBackend(invalid_templates=True)
        assert _export(_rig_trajectories(("hybrid", True)), {"hybrid": _FakeSensor()}, backend) == []
        assert backend.calls == []

    def test_modeled_lidar_native_export_requires_opt_in(self):
        backend = _FakeBackend()
        members = _export(
            _rig_trajectories(("hybrid", True)),
            {"hybrid": _FakeSensor()},
            backend,
            include_modeled_lidars=True,
        )
        assert backend.calls == ["hybrid"]
        assert len(members) == 1
        assert load_firing_patterns(io.BytesIO(members[0].serialized)).templates[0].n_rays == 8

    def test_recovery_preserves_observed_ray_order(self):
        sensor = _FakeSensor(n_rays=4, has_templates=False)
        sensor._directions = sensor._directions[[2, 0, 3, 1]]
        sensor._timestamp_offsets_us = np.asarray([300, 0, 200, 100], dtype=np.int64)

        members = _export(
            _rig_trajectories(("lidar_fu", False)),
            {"lidar_fu": sensor},
            _FakeBackend(),
        )
        templates = load_firing_patterns(io.BytesIO(members[0].serialized))

        np.testing.assert_array_equal(
            templates.templates[0].timestamp_offsets_us,
            [300, 0, 200, 100],
        )

    def test_recovery_uses_densest_eligible_full_sweep(self):
        sensor = _FakeSensor(n_rays=10, has_templates=False)
        directions = {
            0: np.tile(np.array([1.0, 0.0, 0.0], np.float32), (10, 1)),
            1: np.tile(np.array([0.0, 1.0, 0.0], np.float32), (8, 1)),
            2: np.tile(np.array([0.0, 0.0, 1.0], np.float32), (7, 1)),
        }
        offsets = {index: np.arange(len(value), dtype=np.int64) for index, value in directions.items()}
        sensor.get_frame_ray_bundle_direction = lambda frame_index: directions[frame_index]
        sensor.get_frame_ray_bundle_timestamp_us = lambda frame_index: (
            sensor.frames_timestamps_us[frame_index, 0] + offsets[frame_index]
        )
        sensor.get_frame_ray_bundle_return_distance_m = lambda *, frame_index, return_index: (
            np.ones(len(directions[frame_index]), dtype=np.float32)
            if frame_index == 0
            else np.zeros(len(directions[frame_index]), dtype=np.float32)
        )

        members = _export(
            _rig_trajectories(("lidar_fu", False)),
            {"lidar_fu": sensor},
            _FakeBackend(),
        )
        templates = load_firing_patterns(io.BytesIO(members[0].serialized))

        self.assertEqual(templates.reference_frame_start_us, 2_000)
        self.assertEqual(templates.templates[0].n_rays, 8)
        np.testing.assert_array_equal(templates.templates[0].directions_sensor, directions[1])

    def test_refuses_recovery_when_dense_non_returning_slots_cannot_be_proven(self):
        with self.assertRaisesRegex(ValueError, "observed-frame recovery failed.*dense rays"):
            _export(
                _rig_trajectories(("lidar_fu", False)),
                {
                    "lidar_fu": _FakeSensor(
                        has_templates=False,
                        retains_non_returns=False,
                    )
                },
                _FakeBackend(),
            )

    def test_does_not_hide_invalid_dataset_metadata_with_recovery(self):
        with self.assertRaisesRegex(ValueError, "invalid dataset firing-pattern templates.*broken"):
            _export(
                _rig_trajectories(("lidar_fu", False)),
                {"lidar_fu": _FakeSensor()},
                _FakeBackend(invalid_templates=True),
                include_modeled_lidars=True,
            )

    def test_reports_every_missing_required_pattern_together(self):
        with self.assertRaises(ValueError) as caught:
            _export(
                _rig_trajectories(("lidar_left", False), ("lidar_right", False)),
                {},
                _FakeBackend(),
            )

        self.assertIn("lidar_left", str(caught.exception))
        self.assertIn("lidar_right", str(caught.exception))


class TestFiringPatternsRoundTrip(unittest.TestCase):
    """What the export writes is what the artifact reader finds."""

    def test_patterns_reach_the_usdz_and_read_back(self):
        cache = ArtifactCache(artifact_config=ArtifactConfig())
        sensors = {"lidar_fu": _FakeSensor(n_rays=8), "lidar:top:center": _FakeSensor(n_rays=5)}
        cache.firing_patterns = _export(
            _rig_trajectories(("lidar_fu", False), ("lidar:top:center", True)),
            sensors,
            _FakeBackend(),
            include_modeled_lidars=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.usdz"
            cache.write_to_usdz(path)

            # ArtifactCache.archive_members picks the field up by type, with no wiring of its own.
            members = zipfile.ZipFile(path).namelist()
            self.assertIn("firing_patterns/lidar_fu.npz", members)

            patterns = Artifact(path).firing_patterns
            self.assertEqual(sorted(patterns), ["lidar:top:center", "lidar_fu"])
            for logical_id, sensor in sensors.items():
                with self.subTest(logical_id=logical_id):
                    templates = load_firing_patterns(io.BytesIO(patterns[logical_id]))
                    native = sensor.templates
                    assert native is not None
                    self.assertEqual(templates.frame_period_us, native.frame_period_us)
                    self.assertEqual(templates.reference_frame_start_us, native.reference_frame_start_us)
                    self.assertEqual(templates.frame_start_tolerance_us, native.frame_start_tolerance_us)
                    self.assertEqual(len(templates.templates), 1)
                    pattern = templates.templates[0]
                    np.testing.assert_array_equal(pattern.directions_sensor, native.directions_sensor)
                    np.testing.assert_array_equal(pattern.timestamp_offsets_us, native.timestamp_offsets_us)
                    np.testing.assert_array_equal(pattern.packet_ray_offsets, native.packet_ray_offsets)


if __name__ == "__main__":
    unittest.main()


def _template(
    n: int = 8,
    *,
    span_us: int = 70_000,
    azimuth_offset: float = 0.0,
) -> LidarFiringPatternTemplate:
    azimuths = np.linspace(-0.5, 0.5, n) + azimuth_offset
    directions = np.stack([np.cos(azimuths), np.sin(azimuths), np.zeros(n)], axis=-1).astype(np.float32)
    return LidarFiringPatternTemplate(
        directions_sensor=directions,
        timestamp_offsets_us=np.linspace(0, span_us, n).astype(np.int64),
        packet_ray_offsets=np.array([0, n // 2, n], dtype=np.int64),
    )


def _template_data(template_count: int = 4) -> LidarFiringPatternTemplatesData:
    templates = [_template(n=8, azimuth_offset=phase * 0.001) for phase in range(template_count)]
    return LidarFiringPatternTemplatesData(
        directions_sensor=np.concatenate([template.directions_sensor for template in templates]),
        timestamp_offsets_us=np.concatenate([template.timestamp_offsets_us for template in templates]),
        template_ray_offsets=np.arange(0, (template_count + 1) * 8, 8, dtype=np.int64),
        packet_ray_offsets=np.arange(0, (template_count * 2 + 1) * 4, 4, dtype=np.int64),
        template_packet_offsets=np.arange(0, (template_count + 1) * 2, 2, dtype=np.int64),
        reference_frame_start_us=1_000_000,
        frame_period_us=100_000,
        frame_start_tolerance_us=200,
    )


class TestLoadFiringPatterns:
    def test_loads_the_complete_flattened_payload(self, tmp_path):
        data = _template_data()
        path = tmp_path / "templates.npz"
        np.savez_compressed(path, **data.artifact_arrays())

        loaded = load_firing_patterns(path)

        assert len(loaded.templates) == 4
        assert loaded.frame_period_us == 100_000
        assert loaded.templates[0].n_rays == 8
        assert loaded.templates[0].n_packets == 2
        np.testing.assert_array_equal(
            loaded.templates[2].packet_ray_offsets,
            [0, 4, 8],
        )

    def test_loads_versioned_templates_without_optional_packet_layout(self, tmp_path):
        arrays = _template_data(template_count=1).artifact_arrays()
        arrays.pop(PACKET_RAY_OFFSETS_ARRAY_NAME)
        arrays.pop(TEMPLATE_PACKET_OFFSETS_ARRAY_NAME)
        path = tmp_path / "render_only_template.npz"
        np.savez_compressed(path, **arrays)

        loaded = load_firing_patterns(path)

        assert loaded.templates[0].packet_ray_offsets is None
        assert loaded.templates[0].n_packets is None
        assert loaded.select_index(123_456_789) == 0

    def test_rejects_partial_optional_packet_layout(self, tmp_path):
        arrays = _template_data(template_count=1).artifact_arrays()
        arrays.pop(TEMPLATE_PACKET_OFFSETS_ARRAY_NAME)
        path = tmp_path / "partial_packet_layout.npz"
        np.savez_compressed(path, **arrays)

        with pytest.raises(ValueError, match="packet layout is partial"):
            load_firing_patterns(path)

    def test_loads_legacy_recovered_pattern_without_inventing_packets(self, tmp_path):
        template = _template(n=4)
        path = tmp_path / "legacy_pattern.npz"
        np.savez_compressed(
            path,
            directions_sensor=template.directions_sensor,
            timestamp_offsets_us=template.timestamp_offsets_us,
            frame_duration_us=np.asarray(100_000, dtype=np.int64),
        )

        loaded = load_firing_patterns(path)

        assert len(loaded.templates) == 1
        assert loaded.templates[0].packet_ray_offsets is None
        assert loaded.frame_period_us == 100_000

    def test_loads_and_normalizes_legacy_array_dtypes(self, tmp_path):
        path = tmp_path / "legacy_pattern.npz"
        np.savez_compressed(
            path,
            directions_sensor=np.array([[2.0, 0.0, 0.0]], dtype=np.float64),
            timestamp_offsets_us=np.array([7], dtype=np.int32),
            frame_duration_us=np.asarray(100_000, dtype=np.int64),
        )

        loaded = load_firing_patterns(path)

        assert loaded.templates[0].directions_sensor.dtype == np.float32
        assert loaded.templates[0].timestamp_offsets_us.dtype == np.int64
        np.testing.assert_array_equal(loaded.templates[0].directions_sensor, [[1.0, 0.0, 0.0]])

    def test_loads_and_normalizes_slightly_non_unit_legacy_directions(self, tmp_path):
        path = tmp_path / "legacy_pattern.npz"
        np.savez_compressed(
            path,
            directions_sensor=np.array([[1.0005, 0.0, 0.0]], dtype=np.float32),
            timestamp_offsets_us=np.array([7], dtype=np.int64),
            frame_duration_us=np.asarray(100_000, dtype=np.int64),
        )

        loaded = load_firing_patterns(path)

        np.testing.assert_array_equal(loaded.templates[0].directions_sensor, [[1.0, 0.0, 0.0]])

    def test_unknown_schema_version_is_rejected(self, tmp_path):
        arrays = _template_data().artifact_arrays()
        arrays["schema_version"] = np.asarray(2, dtype=np.int64)
        path = tmp_path / "templates.npz"
        np.savez(path, **arrays)

        with pytest.raises(ValueError, match="schema_version 2"):
            load_firing_patterns(path)


def test_a_pattern_written_before_grids_existed_still_gains_one(tmp_path):
    # Deriving rather than storing is what buys this: no re-export needed.
    pattern = LidarFiringPatternTemplate(
        directions_sensor=np.tile(np.array([1.0, 0.0, 0.0], np.float32), (12, 1)),
        timestamp_offsets_us=np.repeat(np.arange(4, dtype=np.int64) * 49, 3),
    )
    path = tmp_path / "legacy-pattern.npz"
    np.savez(
        path,
        directions_sensor=pattern.directions_sensor,
        timestamp_offsets_us=pattern.timestamp_offsets_us,
        frame_duration_us=np.asarray(4 * 49, dtype=np.int64),
    )

    assert load_firing_patterns(path).templates[0].grid == (4, 3)
