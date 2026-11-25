# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import gc
import hashlib
import io
import math
import os
import socket
import tempfile
import threading
import time
import unittest
import zipfile

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from nurec.render.actors import ActorTracks

from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import grpc
import imageio.v2 as iio
import numpy as np
import pytest
import torch

from grpc_health.v1 import health, health_pb2, health_pb2_grpc  # type: ignore

import ncore.data

from nurec.artifact import Artifact
from nurec.grpc.cache_keys import SCENE_ID_HEADER, SCENE_URL_HEADER, BackendCacheKey, SceneCacheKey
from nurec.grpc.client_wait import wait_for_grpc_server
from nurec.grpc.fatal_worker import FatalWorkerExceptionInterceptor, FatalWorkerIncident, FatalWorkerState
from nurec.grpc.grpc_server_config import (
    HARMONIZER_TEMPORAL_COSMOS_BASE_MODEL_URL,
    HARMONIZER_TEMPORAL_COSMOS_TOKENIZER_URL,
    HARMONIZER_TEMPORAL_MODEL_FILENAME,
    HARMONIZER_TEMPORAL_MODEL_URL,
    GrpcServerConfig,
)
from nurec.grpc.nvjpeg_batch_encoder import NvjpegBatchEncoder
from nurec.grpc.protos import ipc_pb2, rdma_pb2
from nurec.grpc.protos.common_pb2 import Empty, Pose, PoseAtTime, VersionId
from nurec.grpc.protos.common_pb2 import Trajectory as ProtoTrajectory
from nurec.grpc.protos.sensorsim_pb2 import (
    LIDAR_FLOATING_FILTER_SOURCE_ANGLE,
    LIDAR_FLOATING_FILTER_SOURCE_RANGEVIEW,
    AspectState,
    AvailableCamerasRequest,
    AvailableCamerasReturn,
    AvailableDynamicObjectsRequest,
    AvailableTrajectoriesRequest,
    BatchCameraRenderRequest,
    BatchCameraRenderRequestItem,
    BatchRGBRenderRequest,
    BatchRGBRenderRequestItem,
    CameraRenderProduct,
    CameraRenderRequest,
    CameraRenderReturn,
    CameraSpec,
    DynamicObject,
    DynamicObjectTrack,
    EditAssetsRequest,
    ExternalAssetObjectsRequest,
    FthetaCameraParam,
    HarmonizerSessionId,
    ImageFormat,
    LidarFiringPatternRequest,
    LidarFiringPatternReturn,
    LidarOutputMode,
    LidarRenderFilter,
    LidarRenderRequest,
    LidarRenderReturn,
    LidarReturnType,
    LidarSpec,
    LinearCde,
    OpenCVFisheyeCameraParam,
    OpenCVPinholeCameraParam,
    OverlayBoundingBoxSource,
    PosePair,
    PrefetchSceneRequest,
    RestoreModelParametersRequest,
    RGBRenderRequest,
    RGBRenderReturn,
)
from nurec.grpc.protos.sensorsim_pb2 import BivariateWindshieldModelParameters as ProtoWParams
from nurec.grpc.protos.sensorsim_pb2_grpc import (
    SensorsimServiceServicer,
    SensorsimServiceStub,
    add_SensorsimServiceServicer_to_server,
)
from nurec.grpc.scene_cache import SceneCache
from nurec.grpc.serve import (
    BATCH_RENDER_REQUEST_CHUNK_SIZE,
    LEGACY_SENSORSIM_SERVICE_NAME,
    ArtifactRegistration,
    AssetBank,
    Backend,
    CacheFullError,
    CacheWaitAbortedError,
    CacheWaitAbortedReason,
    CameraBank,
    HarmonizerNotEnabledError,
    LidarBank,
    ModelIncompatibilityError,
    NoSpareBackendsError,
    PendingBatchRGBEncode,
    PendingCameraRenderOutput,
    PreparedBackend,
    PreparedScene,
    RdmaTransportUnavailableError,
    SceneDownloadInterceptor,
    SceneNotFoundError,
    SensorSimService,
    ServerMetricsManager,
    TimestampOutOfRangeError,
    TorchCamera,
    _AutomaticGcPause,
    _camera_batch_incompatibility,
    _CameraBatchDispatch,
    _classify_render_exception,
    _fill_firing_pattern,
    _grpc_backend_cache_wait_policy,
    _local_file_fingerprint,
    _register_local_artifact,
    _rig_trajectory_to_grpc_trajectory,
    _serve_once_harmonizer_loaded,
    _validate_local_artifact,
    actor_tracks_from_grpc,
    actor_tracks_to_grpc,
    actors_snapshot_from_render_request,
    add_sensorsim_servicer_to_server,
    collect_cuda_worker_diagnostics,
    download_scene_to_cache,
    is_local_scene_uri,
)
from nurec.lidar.firing_patterns.schema import LidarFiringPatternTemplatesData
from nurec.lidar.firing_patterns.templates import LidarFiringPatternTemplate
from nurec.render.render import CameraFrame, LidarRayBundle, LidarRayFrame, MissingRenderOutputError, RenderableModel
from nurec.utils.batch import RenderingData
from nurec.utils.metrics import MetricsCollector
from nurec.utils.types import FrameConversion, HalfClosedInterval, RigTrajectories
from nurec.utils.visualize import SEMANTICS_COLOR_LUT_DEFAULT_SIZE


def _artifact_bytes(scene_id: str, payload: bytes = b"test-payload") -> bytes:
    """Build the minimal valid USDZ needed by scene-cache tests."""
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("metadata.yaml", f"scene_id: {scene_id}\n")
        archive.writestr("payload.bin", payload)
    return data.getvalue()


def _registered_artifact(path: Path, scene_id: str) -> Artifact:
    """Create an Artifact with cached metadata for tests that mock loading."""
    artifact = Artifact(path)
    artifact._metadata = {"scene_id": scene_id}
    artifact._sha256_footprint = hashlib.sha256(str(path).encode()).hexdigest()
    return artifact


def _test_backend_cache_key(scene_id: str = "scene") -> BackendCacheKey:
    """Return a stable exact-artifact key for tests unrelated to cache identity."""
    return BackendCacheKey(SceneCacheKey(scene_id=scene_id), "0" * 64)


@pytest.mark.parametrize("change", ["chmod", "hardlink", "touch", "replace"])
def test_local_artifact_validation_accepts_unchanged_content(tmp_path: Path, change: str) -> None:
    source = tmp_path / "scene.usdz"
    source.write_bytes(_artifact_bytes("scene"))
    registration = _register_local_artifact(Artifact(source), SceneCacheKey(scene_id="scene"))
    if change == "chmod":
        source.chmod(0o600)
    elif change == "hardlink":
        (tmp_path / "link.usdz").hardlink_to(source)
    elif change == "touch":
        os.utime(source, ns=(0, 0))
    else:
        replacement = tmp_path / "replacement.usdz"
        replacement.write_bytes(source.read_bytes())
        replacement.replace(source)
    current = _local_file_fingerprint(source)
    if change in ("touch", "replace"):
        assert current != registration.source_fingerprint
    _validate_local_artifact(registration)
    assert registration.source_fingerprint == current
    with mock.patch.object(Artifact, "sha256_footprint", new_callable=mock.PropertyMock) as digest:
        _validate_local_artifact(registration)
        digest.assert_not_called()


def test_local_artifact_validation_detects_change_during_hash(tmp_path: Path) -> None:
    source = tmp_path / "scene.usdz"
    source.write_bytes(_artifact_bytes("scene"))
    registration = _register_local_artifact(Artifact(source), SceneCacheKey(scene_id="scene"))
    fingerprint = registration.source_fingerprint
    with (
        mock.patch("nurec.grpc.serve._local_file_fingerprint", side_effect=[(1, 2, 3, 4, 5), fingerprint]),
        pytest.raises(ValueError, match="changed while it was validated"),
    ):
        _validate_local_artifact(registration)


@pytest.mark.parametrize("camera_only", [False, True])
def test_metadata_change_allows_prefetch_and_sync_load(tmp_path: Path, camera_only: bool) -> None:
    source = tmp_path / "scene.usdz"
    source.write_bytes(_artifact_bytes("scene"))
    service = _make_service_no_server(camera_only=camera_only)
    key = SceneCacheKey(scene_id="scene")
    registration = _register_local_artifact(Artifact(source), key)
    service._artifact_registrations[key.key_hash] = registration
    os.utime(source, ns=(0, 0))
    prepared = mock.MagicMock(spec=PreparedBackend)
    prepared.backend_cache_key = registration.backend_cache_key
    backend = TestRenderGrpcStatusCodes._make_backend(registration.backend_cache_key)
    prepared.activate.return_value = backend
    with mock.patch.object(PreparedBackend, "from_artifact", return_value=prepared) as prepare:
        prepared_scene = service._prepare_scene(key)
    prepare.assert_called_once()
    assert prepare.call_args.kwargs["camera_only"] is camera_only
    assert prepared_scene is not None
    source.touch()
    with (
        mock.patch.object(service._scene_prefetch, "take", return_value=prepared_scene),
        mock.patch.object(Backend, "from_artifact") as sync_load,
    ):
        with service.get_backend(key) as loaded:
            assert loaded is backend
        prepared.activate.assert_called_once_with()
        sync_load.assert_not_called()
    service.backend_cache.evict_one_spare()
    backend = TestRenderGrpcStatusCodes._make_backend(registration.backend_cache_key)
    with mock.patch.object(Backend, "from_artifact", return_value=backend) as sync_load:
        with service.get_backend(key) as loaded:
            assert loaded is backend
        sync_load.assert_called_once()
        assert sync_load.call_args.kwargs["camera_only"] is camera_only
    service.close()


def test_prefetch_wait_timeout_falls_back_to_sync_load(tmp_path: Path) -> None:
    source = tmp_path / "scene.usdz"
    source.write_bytes(_artifact_bytes("scene"))
    service = _make_service_no_server()
    key = SceneCacheKey(scene_id="scene")
    registration = _register_local_artifact(Artifact(source), key)
    service._artifact_registrations[key.key_hash] = registration
    service._backend_cache_wait_timeout_s = 0.0
    backend = TestRenderGrpcStatusCodes._make_backend(registration.backend_cache_key)

    def take(_key, *, check_wait):
        check_wait()
        pytest.fail("prefetch wait should expire")

    with (
        mock.patch.object(service._scene_prefetch, "take", side_effect=take),
        mock.patch.object(service._scene_prefetch, "discard") as discard,
        mock.patch.object(Backend, "from_artifact", return_value=backend) as sync_load,
    ):
        with service.get_backend(key) as loaded:
            assert loaded is backend
        discard.assert_called_once_with(key)
        sync_load.assert_called_once()
        assert service.backend_cache.pending_global == 0


def _lidar_model_parameters(n_rows: int = 4, n_columns: int = 8) -> ncore.data.ConcreteLidarModelParametersUnion:
    """Minimal spinning lidar intrinsics for LidarBank tests.

    ncore validates the angle layout: row elevations must descend (cw) and column
    azimuths must follow spinning_direction, neither wrapping past the start element.
    Both spans are kept narrow so they cannot wrap.
    """
    return ncore.data.RowOffsetStructuredSpinningLidarModelParameters(
        spinning_frequency_hz=10.0,
        spinning_direction="ccw",
        n_rows=n_rows,
        n_columns=n_columns,
        row_elevations_rad=np.linspace(0.2, -0.2, n_rows).astype(np.float32),
        column_azimuths_rad=np.linspace(-1.0, 1.0, n_columns).astype(np.float32),
        row_azimuth_offsets_rad=np.zeros(n_rows, dtype=np.float32),
    )


def _firing_pattern_npz(n_rays: int, template_count: int = 1, *, frame_period_us: int = 100) -> bytes:
    """Templates serialized as an artifact's ``firing_patterns/<id>.npz`` member."""
    template_directions = []
    for phase in range(template_count):
        angle = phase * 0.01
        directions = np.zeros((n_rays, 3), dtype=np.float32)
        directions[:, 0] = np.cos(angle)
        directions[:, 1] = np.sin(angle)
        template_directions.append(directions)
    data = LidarFiringPatternTemplatesData(
        directions_sensor=np.concatenate(template_directions),
        timestamp_offsets_us=np.tile(np.arange(n_rays, dtype=np.int64), template_count),
        template_ray_offsets=np.arange(0, (template_count + 1) * n_rays, n_rays, dtype=np.int64),
        packet_ray_offsets=np.arange(0, (template_count + 1) * n_rays, n_rays, dtype=np.int64),
        template_packet_offsets=np.arange(template_count + 1, dtype=np.int64),
        reference_frame_start_us=1_000,
        frame_period_us=frame_period_us,
        frame_start_tolerance_us=1,
    )
    buffer = io.BytesIO()
    np.savez(buffer, **data.artifact_arrays())
    return buffer.getvalue()


def _lidar_artifact(*lidars: tuple, firing_patterns: dict[str, bytes] | None = None) -> mock.Mock:
    """An artifact stub whose rig_trajectories carry the given lidars.

    Each entry is ``(logical_id, unique_sensor_idx, has_model_parameters)``, optionally
    followed by a ``sequence_id`` (calibrations are keyed by ``<logical_id>@<sequence_id>``,
    so a distinct sequence is what lets two entries share one logical name). The
    rig-trajectory list is left empty so no sensor cross-references need to be satisfied;
    LidarBank only reads the calibrations. ``firing_patterns`` is what the artifact carries
    itself, keyed by logical id.
    """
    lidar_calibrations: OrderedDict[str, RigTrajectories.LidarCalibration] = OrderedDict()
    for entry in lidars:
        logical_id, unique_sensor_idx, has_model = entry[:3]
        sequence_id = entry[3] if len(entry) > 3 else "sequence"
        lidar_calibrations[f"{logical_id}@{sequence_id}"] = RigTrajectories.LidarCalibration(
            sequence_id=sequence_id,
            logical_sensor_name=logical_id,
            unique_sensor_idx=unique_sensor_idx,
            T_sensor_rig=torch.eye(4, dtype=torch.float32),
            lidar_model_parameters=_lidar_model_parameters() if has_model else None,
        )

    rig_trajectories = RigTrajectories(
        T_world_base=torch.eye(4, dtype=torch.float64),
        world_to_nurec=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
        rig_trajectories=[],
        camera_calibrations=OrderedDict(),
        lidar_calibrations=lidar_calibrations,
    )

    artifact = mock.Mock()
    artifact.rig_trajectories = rig_trajectories.to_dict()
    artifact.firing_patterns = dict(firing_patterns or {})
    return artifact


class TestAutomaticGcPause(unittest.TestCase):
    def test_nested_pauses_resume_collection_once(self):
        pause = _AutomaticGcPause()
        was_enabled = gc.isenabled()
        with pause.paused():
            self.assertFalse(gc.isenabled())
            with pause.paused():
                self.assertFalse(gc.isenabled())
            self.assertFalse(gc.isenabled())
        self.assertEqual(gc.isenabled(), was_enabled)

    def test_leaves_collection_disabled_when_it_was_disabled(self):
        was_enabled = gc.isenabled()
        self.addCleanup(gc.enable if was_enabled else gc.disable)
        gc.disable()
        with _AutomaticGcPause().paused():
            self.assertFalse(gc.isenabled())
        self.assertFalse(gc.isenabled())

    def test_resumes_collection_when_the_load_fails(self):
        pause = _AutomaticGcPause()
        was_enabled = gc.isenabled()
        with self.assertRaises(RuntimeError), pause.paused():
            raise RuntimeError("load failed")
        self.assertEqual(gc.isenabled(), was_enabled)


class TestAssetBankDimsOffset(unittest.TestCase):
    """The track padding is read from the artifact's parsed config as a plain mapping."""

    @staticmethod
    def _artifact(parsed_config: dict) -> mock.Mock:
        artifact = mock.Mock(spec=Artifact)
        artifact.external_assets = {}
        artifact.parsed_config = parsed_config
        return artifact

    def test_reads_the_configured_track_padding(self):
        bank = AssetBank(
            self._artifact({"dataset": {"valid_pixels_cuboid_track_params": {"track_padding_m": [1.0, 1.0, 0.25]}}})
        )
        self.assertEqual(list(bank._dims_offset), [1.0, 1.0, 0.25])

    def test_defaults_to_zero_padding_when_the_key_is_absent(self):
        for parsed_config in (
            {},
            {"dataset": {}},
            {"dataset": None},
            {"dataset": {"valid_pixels_cuboid_track_params": None}},
        ):
            with self.subTest(parsed_config=parsed_config):
                bank = AssetBank(self._artifact(parsed_config))
                self.assertEqual(list(bank._dims_offset), [0.0, 0.0, 0.0])


class TestLidarBank(unittest.TestCase):
    """LidarBank keeps every lidar of the artifact, keyed by logical sensor id."""

    def test_retains_all_lidars_keyed_by_logical_id(self):
        bank = LidarBank(_lidar_artifact(("lidar_top", 0, True), ("lidar_left", 1, True)))

        self.assertEqual(list(bank.lidars), ["lidar_top", "lidar_left"])
        self.assertEqual(bank.lidars["lidar_left"].unique_sensor_idx, 1)

    def test_empty_artifact_has_no_lidar(self):
        bank = LidarBank(_lidar_artifact())

        self.assertEqual(bank.lidars, OrderedDict())
        self.assertIsNone(bank.get_entry())

    def test_duplicate_logical_id_is_rejected(self):
        # Two sequences contributing the same logical name would make both discovery and
        # per-sensor config resolution ambiguous.
        with self.assertRaisesRegex(ValueError, "duplicate logical lidar id"):
            LidarBank(_lidar_artifact(("lidar_top", 0, True, "seq_a"), ("lidar_top", 1, True, "seq_b")))

    def test_get_entry_by_id(self):
        bank = LidarBank(_lidar_artifact(("lidar_top", 0, True), ("lidar_left", 1, True)))

        self.assertEqual(bank.get_entry("lidar_left").logical_id, "lidar_left")

    def test_get_entry_rejects_unknown_id(self):
        bank = LidarBank(_lidar_artifact(("lidar_top", 0, True)))

        with self.assertRaisesRegex(KeyError, "Unknown lidar 'nope'"):
            bank.get_entry("nope")

    def test_get_entry_without_id_requires_a_single_lidar(self):
        single = LidarBank(_lidar_artifact(("lidar_top", 0, True)))
        self.assertEqual(single.get_entry().logical_id, "lidar_top")

        multi = LidarBank(_lidar_artifact(("lidar_top", 0, True), ("lidar_left", 1, True)))
        with self.assertRaisesRegex(ValueError, "multiple lidars"):
            multi.get_entry()

    def test_firing_pattern_templates_are_read_from_the_artifact(self):
        bank = LidarBank(
            _lidar_artifact(("lidar_top", 0, False), firing_patterns={"lidar_top": _firing_pattern_npz(7)})
        )

        templates = bank.get_entry("lidar_top").firing_pattern_templates
        self.assertIsNotNone(templates)
        self.assertEqual(templates.templates[0].n_rays, 7)

    def test_select_firing_pattern_delegates_to_template_phase_selection(self):
        entry = LidarBank(
            _lidar_artifact(
                ("lidar_top", 0, False),
                firing_patterns={"lidar_top": _firing_pattern_npz(7, template_count=2)},
            )
        ).get_entry("lidar_top")

        self.assertIs(entry.select_firing_pattern(1_000), entry.firing_pattern_templates.templates[0])
        self.assertIs(entry.select_firing_pattern(1_100), entry.firing_pattern_templates.templates[1])

    def test_explicit_firing_pattern_id_takes_precedence_over_frame_time(self):
        entry = LidarBank(
            _lidar_artifact(
                ("lidar_top", 0, False),
                firing_patterns={"lidar_top": _firing_pattern_npz(7, template_count=2)},
            )
        ).get_entry("lidar_top")
        requested = entry.firing_pattern_templates.templates[1]

        self.assertIs(entry.select_firing_pattern(1_000, requested.pattern_id(100)), requested)

    def test_explicit_unknown_firing_pattern_id_is_not_found(self):
        entry = LidarBank(
            _lidar_artifact(("lidar_top", 0, False), firing_patterns={"lidar_top": _firing_pattern_npz(7)})
        ).get_entry("lidar_top")

        with self.assertRaisesRegex(KeyError, "Unknown firing pattern id 'not-a-pattern'") as caught:
            entry.select_firing_pattern(1_000, "not-a-pattern")

        self.assertEqual(_classify_render_exception(caught.exception)[0], grpc.StatusCode.NOT_FOUND)

    def test_bundle_is_built_once_and_skipped_without_intrinsics(self):
        bank = LidarBank(_lidar_artifact(("lidar_top", 0, True), ("lidar_blind", 1, False)))

        self.assertIsNone(bank.get_entry("lidar_blind").bundle())

        # Building the bundle allocates on the GPU, so it must be lazy and cached rather
        # than rebuilt per request.
        entry = bank.get_entry("lidar_top")
        self.assertIsNone(entry._bundle)
        with mock.patch("nurec.grpc.serve.LidarModelBundle.load_from_config") as load:
            load.return_value = mock.sentinel.bundle
            self.assertIs(entry.bundle(), mock.sentinel.bundle)
            self.assertIs(entry.bundle(), mock.sentinel.bundle)
        load.assert_called_once()


class TestBackendLidarDispatch(unittest.TestCase):
    """Exercise the production request dispatcher, not a test-side copy of its branches."""

    def _backend(self, artifact: mock.Mock) -> Backend:
        model = mock.MagicMock()
        model.default_lidar_neighbor_source.return_value = "rangeview"
        model.render_lidar_frame_from_ray_bundle.return_value = mock.Mock(
            point_positions=torch.empty((0, 3), dtype=torch.float32),
            point_intensities=None,
        )
        model.render_lidar_ray_frame_from_ray_bundle.return_value = LidarRayFrame(
            ray_distances=torch.ones(7, dtype=torch.float32),
            ray_valid=torch.ones(7, dtype=torch.bool),
            ray_intensities=None,
        )
        backend = Backend(
            renderable_model=model,
            camera_bank=mock.Mock(),
            world_to_nurec=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
            scene_time_range_us=HalfClosedInterval(0, 2**64),
            lidar_bank=LidarBank(artifact),
            asset_bank=mock.Mock(),
            source_artifact=artifact,
            backend_cache_key=_test_backend_cache_key(),
        )
        backend._get_lidar_stream = mock.MagicMock(return_value=mock.sentinel.stream)  # type: ignore[method-assign]
        return backend

    @staticmethod
    def _request(lidar_id: str = "") -> LidarRenderRequest:
        request = LidarRenderRequest(
            lidar_id=lidar_id,
            frame_start_us=1_000,
            frame_end_us=1_100,
            output_mode=LidarOutputMode.LIDAR_OUTPUT_PER_RAY,
        )
        request.sensor_pose.start_pose.quat.w = 1.0
        request.sensor_pose.end_pose.quat.w = 1.0
        return request

    @staticmethod
    def _mixed_artifact() -> mock.Mock:
        return _lidar_artifact(
            ("modeled", 3, True),
            ("model_free", 4, False),
            firing_patterns={"model_free": _firing_pattern_npz(7)},
        )

    @staticmethod
    def _hybrid_artifact() -> mock.Mock:
        return _lidar_artifact(
            ("hybrid", 5, True),
            firing_patterns={"hybrid": _firing_pattern_npz(7, template_count=2)},
        )

    def test_defaults_a_modeled_lidar_to_point_cloud_without_exposed_geometry(self):
        backend = self._backend(self._mixed_artifact())
        request = self._request("modeled")
        request.ClearField("output_mode")
        with (
            mock.patch("nurec.grpc.serve.LidarModelBundle.load_from_config", return_value=mock.sentinel.model),
            mock.patch.object(LidarRayBundle, "build", return_value=mock.sentinel.bundle) as build,
            mock.patch.object(LidarRayBundle, "build_model_free") as build_model_free,
        ):
            response = backend.render_lidar_request(request)

        build.assert_called_once()
        self.assertIs(build.call_args.kwargs["lidar_model"], mock.sentinel.model)
        self.assertEqual(build.call_args.kwargs["unique_sensor_idx"], 3)
        build_model_free.assert_not_called()
        self.assertEqual(response.num_return_layers, 0)
        backend.renderable_model.render_lidar_frame_from_ray_bundle.assert_called_once()
        backend.renderable_model.render_lidar_ray_frame_from_ray_bundle.assert_not_called()

    def test_rejects_explicit_per_ray_output_for_a_modeled_lidar(self):
        request = self._request("modeled")
        request.output_mode = LidarOutputMode.LIDAR_OUTPUT_PER_RAY
        self._assert_rejected_before_ray_preparation(
            request, "artifact-resident firing pattern", ModelIncompatibilityError
        )

    def _assert_rejected_before_ray_preparation(self, request, message, exception_type=ValueError):
        backend = self._backend(self._mixed_artifact())
        with (
            mock.patch("nurec.grpc.serve.LidarEntry.bundle") as bundle,
            mock.patch("nurec.grpc.serve.torch.tensor") as tensor,
            mock.patch.object(LidarRayBundle, "build") as build,
            mock.patch.object(LidarRayBundle, "build_model_free") as build_model_free,
            mock.patch.object(backend, "_update_aspect_states") as update_aspects,
            self.assertRaisesRegex(exception_type, message),
        ):
            backend.render_lidar_request(request)

        bundle.assert_not_called()
        tensor.assert_not_called()
        build.assert_not_called()
        build_model_free.assert_not_called()
        update_aspects.assert_not_called()
        backend._get_lidar_stream.assert_not_called()
        backend.renderable_model.render_lidar_ray_frame_from_ray_bundle.assert_not_called()
        backend.renderable_model.render_lidar_frame_from_ray_bundle.assert_not_called()

    def test_dispatches_a_model_free_lidar_to_the_observed_pattern_builder(self):
        backend = self._backend(self._mixed_artifact())
        request = self._request("model_free")
        request.return_types.append(LidarReturnType.LIDAR_RETURN_STRONGEST)
        with (
            mock.patch.object(LidarRayBundle, "build") as build,
            mock.patch.object(
                LidarRayBundle, "build_model_free", return_value=mock.sentinel.bundle
            ) as build_model_free,
        ):
            response = backend.render_lidar_request(request)

        build.assert_not_called()
        build_model_free.assert_called_once()
        self.assertEqual(
            build_model_free.call_args.kwargs["firing_pattern_template"].n_rays,
            7,
        )
        self.assertEqual(build_model_free.call_args.kwargs["frame_period_us"], 100)
        self.assertEqual(build_model_free.call_args.kwargs["unique_sensor_idx"], 4)
        self.assertEqual(list(response.return_types), [LidarReturnType.LIDAR_RETURN_STRONGEST])
        backend.renderable_model.render_lidar_ray_frame_from_ray_bundle.assert_called_once()
        backend.renderable_model.render_lidar_frame_from_ray_bundle.assert_not_called()

    def test_stored_pattern_precedes_intrinsics_without_building_model(self):
        backend = self._backend(self._hybrid_artifact())
        request = self._request("hybrid")
        with (
            mock.patch("nurec.grpc.serve.LidarModelBundle.load_from_config") as load_model,
            mock.patch.object(
                LidarRayBundle, "build_model_free", return_value=mock.sentinel.bundle
            ) as build_model_free,
        ):
            response = backend.render_lidar_request(request)

        load_model.assert_not_called()
        build_model_free.assert_called_once()
        self.assertTrue(response.firing_pattern_id)
        backend.renderable_model.render_lidar_ray_frame_from_ray_bundle.assert_called_once()

    def test_hybrid_lidar_explicit_point_cloud_still_uses_stored_pattern(self):
        backend = self._backend(self._hybrid_artifact())
        request = self._request("hybrid")
        request.output_mode = LidarOutputMode.LIDAR_OUTPUT_POINT_CLOUD
        with (
            mock.patch("nurec.grpc.serve.LidarModelBundle.load_from_config") as load_model,
            mock.patch.object(LidarRayBundle, "build_model_free", return_value=mock.sentinel.bundle),
        ):
            backend.render_lidar_request(request)

        load_model.assert_not_called()
        backend.renderable_model.render_lidar_frame_from_ray_bundle.assert_called_once()

    def test_hybrid_lidar_unknown_pattern_does_not_fall_back_to_model(self):
        backend = self._backend(self._hybrid_artifact())
        request = self._request("hybrid")
        request.firing_pattern_id = "missing"
        with (
            mock.patch("nurec.grpc.serve.LidarModelBundle.load_from_config") as load_model,
            self.assertRaisesRegex(KeyError, "Unknown firing pattern id"),
        ):
            backend.render_lidar_request(request)

        load_model.assert_not_called()

    def test_accepts_first_and_multiple_requested_return_layers(self):
        supported = (
            [LidarReturnType.LIDAR_RETURN_FIRST],
            [LidarReturnType.LIDAR_RETURN_FIRST, LidarReturnType.LIDAR_RETURN_STRONGEST],
            [LidarReturnType.LIDAR_RETURN_STRONGEST, LidarReturnType.LIDAR_RETURN_FIRST],
        )
        for requested in supported:
            with self.subTest(requested=requested):
                backend = self._backend(self._mixed_artifact())
                request = self._request("model_free")
                request.return_types.extend(requested)
                with mock.patch.object(LidarRayBundle, "build_model_free", return_value=mock.sentinel.bundle):
                    response = backend.render_lidar_request(request)

                self.assertEqual(response.num_return_layers, len(requested))
                self.assertEqual(list(response.return_types), requested)
                backend.renderable_model.render_lidar_ray_frame_from_ray_bundle.assert_called_once()

    def test_return_layer_wire_values_reserve_zero_for_unspecified(self):
        """Keep real layers distinct from an explicitly transmitted zero sentinel."""
        self.assertEqual(LidarReturnType.LIDAR_RETURN_UNSPECIFIED, 0)
        self.assertEqual(LidarReturnType.LIDAR_RETURN_FIRST, 1)
        self.assertEqual(LidarReturnType.LIDAR_RETURN_STRONGEST, 2)
        for values in ([], [0], [1], [2], [1, 2], [99]):
            with self.subTest(values=values):
                request = LidarRenderRequest(return_types=values)
                decoded = LidarRenderRequest.FromString(request.SerializeToString())
                self.assertEqual(list(decoded.return_types), values)

    def test_rejects_unspecified_return_layers_before_preparing_rays(self):
        """Reject the zero sentinel alone or mixed with a supported layer."""
        for values in (
            [LidarReturnType.LIDAR_RETURN_UNSPECIFIED],
            [LidarReturnType.LIDAR_RETURN_STRONGEST, LidarReturnType.LIDAR_RETURN_UNSPECIFIED],
        ):
            with self.subTest(values=values):
                request = self._request("model_free")
                request.return_types.extend(values)
                self._assert_rejected_before_ray_preparation(request, "unsupported LiDAR return layer")

    def test_rejects_duplicate_requested_return_layers(self):
        request = self._request("model_free")
        request.return_types.extend([LidarReturnType.LIDAR_RETURN_STRONGEST, LidarReturnType.LIDAR_RETURN_STRONGEST])
        self._assert_rejected_before_ray_preparation(request, "return layers must be unique")

    def test_rejects_unknown_return_layers_before_preparing_rays(self):
        request = self._request("model_free")
        request.return_types.append(99)
        self._assert_rejected_before_ray_preparation(request, "unsupported LiDAR return layer")

    def test_explicit_point_cloud_mode_uses_stored_geometry(self):
        backend = self._backend(self._mixed_artifact())
        request = self._request("model_free")
        request.output_mode = LidarOutputMode.LIDAR_OUTPUT_POINT_CLOUD

        with mock.patch.object(LidarRayBundle, "build_model_free", return_value=mock.sentinel.bundle):
            backend.render_lidar_request(request)

        self.assertTrue(request.HasField("output_mode"))
        backend.renderable_model.render_lidar_frame_from_ray_bundle.assert_called_once()
        backend.renderable_model.render_lidar_ray_frame_from_ray_bundle.assert_not_called()

    def test_rejects_an_unknown_output_mode(self):
        for lidar_id in ("model_free", "modeled"):
            with self.subTest(lidar_id=lidar_id):
                request = self._request(lidar_id)
                request.output_mode = 99
                self._assert_rejected_before_ray_preparation(request, "unsupported lidar output mode 99")

    def test_render_request_explicit_pattern_id_overrides_timestamp_phase(self):
        artifact = _lidar_artifact(
            ("model_free", 4, False),
            firing_patterns={"model_free": _firing_pattern_npz(7, template_count=2)},
        )
        backend = self._backend(artifact)
        requested = backend.lidar_bank.get_entry("model_free").firing_pattern_templates.templates[1]
        request = self._request("model_free")
        request.firing_pattern_id = requested.pattern_id(100)

        with mock.patch.object(
            LidarRayBundle, "build_model_free", return_value=mock.sentinel.bundle
        ) as build_model_free:
            backend.render_lidar_request(request)

        self.assertIs(build_model_free.call_args.kwargs["firing_pattern_template"], requested)

    def test_variable_phase_ray_counts_reach_lookup_and_render_buffers(self):
        data = LidarFiringPatternTemplatesData(
            directions_sensor=np.tile(np.array([[1, 0, 0]], dtype=np.float32), (10, 1)),
            timestamp_offsets_us=np.arange(10, dtype=np.int64),
            template_ray_offsets=np.array([0, 4, 10], dtype=np.int64),
            packet_ray_offsets=np.array([0, 2, 4, 6, 10], dtype=np.int64),
            template_packet_offsets=np.array([0, 2, 4], dtype=np.int64),
            reference_frame_start_us=1_000,
            frame_period_us=100,
            frame_start_tolerance_us=1,
        )
        buffer = io.BytesIO()
        np.savez(buffer, **data.artifact_arrays())
        backend = self._backend(
            _lidar_artifact(
                ("model_free", 4, False),
                firing_patterns={"model_free": buffer.getvalue()},
            )
        )
        for phase, count in enumerate((4, 6)):
            with self.subTest(phase=phase):
                request = self._request("model_free")
                request.frame_start_us += 100 * phase
                request.frame_end_us += 100 * phase
                pattern = backend.get_lidar_firing_pattern(
                    LidarFiringPatternRequest(
                        lidar_id="model_free",
                        frame_start_us=request.frame_start_us,
                    )
                )
                self.assertEqual(pattern.num_rays, count)
                self.assertEqual(len(pattern.directions_sensor), count * 3 * 4)
                np.testing.assert_array_equal(
                    np.frombuffer(pattern.packet_layout.ray_offsets_u32_le, dtype="<u4"),
                    [0, 2, count],
                )
                backend.renderable_model.render_lidar_ray_frame_from_ray_bundle.return_value = LidarRayFrame(
                    ray_distances=torch.ones(count),
                    ray_valid=torch.ones(count, dtype=torch.bool),
                    ray_intensities=torch.ones(count),
                )
                with mock.patch.object(LidarRayBundle, "build_model_free", return_value=mock.sentinel.bundle):
                    rendered = backend.render_lidar_request(request)
                self.assertEqual(rendered.num_rays, count)
                self.assertEqual(len(rendered.ray_distances_m_buffer), count * 4)
                self.assertEqual(rendered.firing_pattern_id, pattern.firing_pattern_id)

    def test_pattern_lookup_uses_time_by_default_and_explicit_id_with_precedence(self):
        artifact = _lidar_artifact(
            ("model_free", 4, False),
            firing_patterns={"model_free": _firing_pattern_npz(7, template_count=2)},
        )
        backend = self._backend(artifact)
        templates = backend.lidar_bank.get_entry("model_free").firing_pattern_templates.templates

        selected_by_time = backend.get_lidar_firing_pattern(
            LidarFiringPatternRequest(lidar_id="model_free", frame_start_us=1_100)
        )
        selected_by_id = backend.get_lidar_firing_pattern(
            LidarFiringPatternRequest(
                lidar_id="model_free",
                frame_start_us=1_000,
                firing_pattern_id=templates[1].pattern_id(100),
            )
        )

        self.assertEqual(selected_by_time.firing_pattern_id, templates[1].pattern_id(100))
        self.assertEqual(selected_by_id.firing_pattern_id, templates[1].pattern_id(100))
        self.assertEqual(selected_by_id.frame_period_us, 100)
        self.assertEqual(selected_by_id.packet_layout.num_packets, 1)

    def test_pattern_lookup_reports_whether_the_full_request_window_is_in_scene(self):
        artifact = _lidar_artifact(
            ("model_free", 4, False),
            firing_patterns={"model_free": _firing_pattern_npz(7, template_count=2)},
        )
        backend = self._backend(artifact)
        backend.scene_time_range_us = HalfClosedInterval(1_000, 1_200)
        backend.camera_bank = object()  # type: ignore[assignment]

        contained = backend.get_lidar_firing_pattern(
            LidarFiringPatternRequest(lidar_id="model_free", frame_start_us=1_100)
        )
        crosses_end = backend.get_lidar_firing_pattern(
            LidarFiringPatternRequest(lidar_id="model_free", frame_start_us=1_101)
        )
        before_start = backend.get_lidar_firing_pattern(
            LidarFiringPatternRequest(lidar_id="model_free", frame_start_us=999)
        )
        unspecified_start = backend.get_lidar_firing_pattern(
            LidarFiringPatternRequest(lidar_id="model_free", firing_pattern_id=contained.firing_pattern_id)
        )

        assert contained.HasField("requested_frame_within_scene_time_range")
        assert contained.requested_frame_within_scene_time_range
        assert crosses_end.HasField("requested_frame_within_scene_time_range")
        assert not crosses_end.requested_frame_within_scene_time_range
        assert before_start.HasField("requested_frame_within_scene_time_range")
        assert not before_start.requested_frame_within_scene_time_range
        assert not unspecified_start.HasField("requested_frame_within_scene_time_range")

    def test_pattern_cache_ids_cover_frame_period_for_lookup_and_rendering(self):
        artifact = _lidar_artifact(
            ("fast", 4, False),
            ("slow", 5, False),
            ("same_as_fast", 6, False),
            firing_patterns={
                "fast": _firing_pattern_npz(7, frame_period_us=100),
                "slow": _firing_pattern_npz(7, frame_period_us=200),
                "same_as_fast": _firing_pattern_npz(7, frame_period_us=100),
            },
        )
        backend = self._backend(artifact)
        patterns = {
            lidar_id: backend.get_lidar_firing_pattern(LidarFiringPatternRequest(lidar_id=lidar_id))
            for lidar_id in ("fast", "slow", "same_as_fast")
        }
        self.assertEqual(patterns["fast"].directions_sensor, patterns["slow"].directions_sensor)
        self.assertEqual(patterns["fast"].timestamp_offsets_us, patterns["slow"].timestamp_offsets_us)
        self.assertEqual(patterns["fast"].packet_layout, patterns["slow"].packet_layout)
        self.assertNotEqual(patterns["fast"].firing_pattern_id, patterns["slow"].firing_pattern_id)
        self.assertEqual(patterns["fast"], patterns["same_as_fast"])
        cache = {pattern.firing_pattern_id: pattern for pattern in patterns.values()}
        self.assertEqual(len(cache), 2)

        for lidar_id, period in (("fast", 100), ("slow", 200)):
            expected = patterns[lidar_id]
            self.assertEqual(expected.frame_period_us, period)
            pinned = backend.get_lidar_firing_pattern(
                LidarFiringPatternRequest(lidar_id=lidar_id, firing_pattern_id=expected.firing_pattern_id)
            )
            self.assertEqual(pinned, expected)
            for include_pattern in (False, True):
                with self.subTest(lidar_id=lidar_id, include_pattern=include_pattern):
                    request = self._request(lidar_id)
                    request.frame_end_us = request.frame_start_us + period
                    request.firing_pattern_id = expected.firing_pattern_id
                    request.include_firing_pattern = include_pattern
                    with mock.patch.object(LidarRayBundle, "build_model_free", return_value=mock.sentinel.bundle):
                        response = backend.render_lidar_request(request)
                    self.assertEqual(response.firing_pattern_id, expected.firing_pattern_id)
                    self.assertEqual(cache[response.firing_pattern_id].frame_period_us, period)
                    self.assertEqual(response.HasField("firing_pattern"), include_pattern)
                    if include_pattern:
                        self.assertEqual(response.firing_pattern, expected)

            other_id = patterns["slow" if lidar_id == "fast" else "fast"].firing_pattern_id
            with self.assertRaisesRegex(KeyError, "Unknown firing pattern id"):
                backend.get_lidar_firing_pattern(
                    LidarFiringPatternRequest(lidar_id=lidar_id, firing_pattern_id=other_id)
                )
            request = self._request(lidar_id)
            request.firing_pattern_id = other_id
            with (
                mock.patch.object(LidarRayBundle, "build_model_free") as build_model_free,
                self.assertRaisesRegex(KeyError, "Unknown firing pattern id"),
            ):
                backend.render_lidar_request(request)
            build_model_free.assert_not_called()

    def test_omitted_id_on_multi_lidar_artifact_is_invalid_argument(self):
        backend = self._backend(self._mixed_artifact())

        with self.assertRaisesRegex(ValueError, "multiple lidars") as caught:
            backend.render_lidar_request(self._request())

        status, message = _classify_render_exception(caught.exception)
        self.assertEqual(status, grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("multiple lidars", message)

    def test_pattern_backed_lidars_default_to_nonempty_point_clouds(self):
        legacy = io.BytesIO()
        np.savez(
            legacy,
            directions_sensor=np.tile([1.0, 0.0, 0.0], (7, 1)).astype(np.float32),
            timestamp_offsets_us=np.arange(7, dtype=np.int64),
            frame_duration_us=np.array(100, dtype=np.int64),
        )
        for artifact, lidar_id in (
            (self._mixed_artifact(), "model_free"),
            (self._hybrid_artifact(), "hybrid"),
            (_lidar_artifact(("legacy", 0, False), firing_patterns={"legacy": legacy.getvalue()}), "legacy"),
        ):
            with self.subTest(lidar_id=lidar_id):
                backend = self._backend(artifact)
                backend.renderable_model.render_lidar_frame_from_ray_bundle.return_value.point_positions = torch.ones(
                    1, 3
                )
                request = self._request(lidar_id)
                request.ClearField("output_mode")
                with mock.patch.object(LidarRayBundle, "build_model_free", return_value=mock.sentinel.bundle):
                    response = backend.render_lidar_request(request)
                self.assertEqual(response.num_points, 1)
                self.assertEqual(len(response.point_xyzs_buffer), 12)
                self.assertEqual(response.num_return_layers, 0)
                backend.renderable_model.render_lidar_ray_frame_from_ray_bundle.assert_not_called()

    def test_point_cloud_rejects_per_ray_options_before_preparing_rays(self):
        for options in (
            {"include_firing_pattern": True},
            {"return_types": [LidarReturnType.LIDAR_RETURN_STRONGEST]},
            {"return_types": [99]},
            {"return_types": [0, 0]},
        ):
            for explicit in (False, True):
                with self.subTest(options=options, explicit=explicit):
                    request = self._request("model_free")
                    request.ClearField("output_mode")
                    if explicit:
                        request.output_mode = LidarOutputMode.LIDAR_OUTPUT_POINT_CLOUD
                    request.MergeFrom(LidarRenderRequest(**options))
                    self._assert_rejected_before_ray_preparation(request, "require LIDAR_OUTPUT_PER_RAY")

    def test_invalid_artifact_scale_is_rejected_before_gpu_work(self):
        for scale in (0.0, -1.0, float("inf"), float("nan")):
            backend = self._backend(self._mixed_artifact())
            backend.world_to_nurec = mock.Mock(target_scale=scale)
            with (
                mock.patch.object(LidarRayBundle, "build_model_free") as build,
                self.assertRaisesRegex(ModelIncompatibilityError, "finite and positive") as caught,
            ):
                backend.render_lidar_request(self._request("model_free"))
            build.assert_not_called()
            backend._get_lidar_stream.assert_not_called()
            self.assertEqual(_classify_render_exception(caught.exception)[0], grpc.StatusCode.FAILED_PRECONDITION)

    def test_rendered_ray_count_must_match_pattern_even_without_embedding(self):
        backend = self._backend(self._mixed_artifact())
        backend.renderable_model.render_lidar_ray_frame_from_ray_bundle.return_value = LidarRayFrame(
            ray_distances=torch.ones(6),
            ray_valid=torch.ones(6, dtype=torch.bool),
            ray_intensities=None,
        )
        with (
            mock.patch.object(LidarRayBundle, "build_model_free", return_value=mock.sentinel.bundle),
            self.assertRaisesRegex(RuntimeError, "Rendered 6 rays.*7 rays"),
        ):
            backend.render_lidar_request(self._request("model_free"))

    def test_multiphase_lookup_requires_a_selector_but_singleton_does_not(self):
        backend = self._backend(self._hybrid_artifact())
        with self.assertRaisesRegex(ValueError, "2 phases; set frame_start_us or firing_pattern_id"):
            backend.get_lidar_firing_pattern(LidarFiringPatternRequest(lidar_id="hybrid"))
        singleton = self._backend(self._mixed_artifact())
        self.assertTrue(
            singleton.get_lidar_firing_pattern(LidarFiringPatternRequest(lidar_id="model_free")).firing_pattern_id
        )

    def test_point_cloud_and_per_ray_agree_in_metres_with_scaled_scene(self):
        for scale in (0.5, 1.0, 2.0):
            with self.subTest(scale=scale):
                backend = self._backend(self._mixed_artifact())
                backend.world_to_nurec = FrameConversion.from_origin_scale_axis(np.zeros(3), scale, [0, 1, 2])
                frame = LidarRayFrame(
                    ray_distances=torch.arange(1, 8, dtype=torch.float32),
                    ray_valid=torch.ones(7, dtype=torch.bool),
                    ray_intensities=None,
                )
                rays = torch.zeros(7, 6)
                rays[:, 3] = 1
                bundle = mock.Mock(T_nurec_sensor_end=torch.eye(4))
                backend.renderable_model._render_lidar_rays.return_value = (frame, rays)
                backend.renderable_model.render_lidar_frame_from_ray_bundle.side_effect = lambda *args, **kwargs: (
                    RenderableModel.render_lidar_frame_from_ray_bundle(backend.renderable_model, *args, **kwargs)
                )
                backend.renderable_model.render_lidar_ray_frame_from_ray_bundle.return_value = frame
                request = self._request("model_free")
                with mock.patch.object(LidarRayBundle, "build_model_free", return_value=bundle):
                    per_ray = backend.render_lidar_request(request)
                    request.output_mode = LidarOutputMode.LIDAR_OUTPUT_POINT_CLOUD
                    point_cloud = backend.render_lidar_request(request)
                distances = np.frombuffer(per_ray.ray_distances_m_buffer, dtype="<f4")
                xyz = np.frombuffer(point_cloud.point_xyzs_buffer, dtype="<f4").reshape(-1, 3)
                np.testing.assert_allclose(distances, np.arange(1, 8) / scale)
                np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), distances)

    def test_point_cloud_preserves_unclipped_intensities(self):
        backend = self._backend(self._mixed_artifact())
        backend.renderable_model.render_lidar_frame_from_ray_bundle.return_value = mock.Mock(
            point_positions=torch.ones(2, 3),
            point_intensities=torch.tensor([-0.2, 1.2]),
        )
        request = self._request("model_free")
        request.output_mode = LidarOutputMode.LIDAR_OUTPUT_POINT_CLOUD
        with mock.patch.object(LidarRayBundle, "build_model_free", return_value=mock.sentinel.bundle):
            response = backend.render_lidar_request(request)
        np.testing.assert_allclose(np.frombuffer(response.point_intensities_buffer, dtype="<f4"), [-0.2, 1.2])


class TestCamera(unittest.TestCase):
    def test_camera_spec(self):
        camera_spec = CameraSpec()
        camera_spec.logical_id = "camera1"
        camera_spec.resolution_h = 567
        camera_spec.resolution_w = 320
        camera_spec.shutter_type = 1

        fthetaCameraParam = FthetaCameraParam()
        fthetaCameraParam.principal_point_x = 0
        fthetaCameraParam.principal_point_y = 1
        fthetaCameraParam.reference_poly = 1
        fthetaCameraParam.pixeldist_to_angle_poly.extend([1, 2, 3, 4, 5])
        fthetaCameraParam.angle_to_pixeldist_poly.extend([0, 0, 0, 0, 0])
        fthetaCameraParam.max_angle = 120
        fthetaCameraParam.linear_cde.linear_c = 1
        fthetaCameraParam.linear_cde.linear_d = 2
        fthetaCameraParam.linear_cde.linear_e = 3

        camera_spec.ftheta_param.CopyFrom(fthetaCameraParam)

        np.testing.assert_equal(camera_spec.logical_id, "camera1")
        np.testing.assert_equal(camera_spec.resolution_h, 567)
        np.testing.assert_equal(camera_spec.resolution_w, 320)
        np.testing.assert_equal(camera_spec.shutter_type, 1)
        np.testing.assert_equal(camera_spec.WhichOneof("camera_param"), "ftheta_param")
        np.testing.assert_equal(camera_spec.ftheta_param.principal_point_x, 0)
        np.testing.assert_equal(camera_spec.ftheta_param.principal_point_y, 1)
        np.testing.assert_equal(camera_spec.ftheta_param.reference_poly, 1)
        np.testing.assert_array_equal(camera_spec.ftheta_param.pixeldist_to_angle_poly, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(camera_spec.ftheta_param.angle_to_pixeldist_poly, [0, 0, 0, 0, 0])
        np.testing.assert_equal(camera_spec.ftheta_param.max_angle, 120)
        np.testing.assert_equal(camera_spec.ftheta_param.linear_cde.linear_c, 1)
        np.testing.assert_equal(camera_spec.ftheta_param.linear_cde.linear_d, 2)
        np.testing.assert_equal(camera_spec.ftheta_param.linear_cde.linear_e, 3)

        # Check if the field 'temporary_camera_spec' is marked as deprecated
        descriptor = CameraSpec.DESCRIPTOR
        field_descriptor = descriptor.fields_by_name.get("temporary_camera_spec")
        self.assertIsNotNone(field_descriptor, "Field 'temporary_camera_spec' does not exist in CameraSpec.")
        self.assertTrue(
            field_descriptor.GetOptions().deprecated, "Field 'temporary_camera_spec' is not marked as deprecated."
        )

        # Check if the field 'trajectory_idx' is marked as deprecated
        field_descriptor = descriptor.fields_by_name.get("trajectory_idx")
        self.assertIsNotNone(field_descriptor, "Field 'trajectory_idx' does not exist in CameraSpec.")
        self.assertTrue(field_descriptor.GetOptions().deprecated, "Field 'trajectory_idx' is not marked as deprecated.")


class TestFThetaCameraModelParameters(unittest.TestCase):
    def test_get_camera_model_parameters_populates_external_distortion(self):
        linear_cde = LinearCde(linear_c=1.0, linear_d=2.0, linear_e=3.0)
        intrinsics = CameraSpec(
            logical_id="camera1",
            resolution_h=567,
            resolution_w=320,
            shutter_type=1,
            ftheta_param=FthetaCameraParam(
                principal_point_x=1.0,
                principal_point_y=2.0,
                reference_poly=ProtoWParams.ReferencePolynomial.BACKWARD,
                pixeldist_to_angle_poly=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                angle_to_pixeldist_poly=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                max_angle=1.0,
                linear_cde=linear_cde,
            ),
            bivariate_windshield_model_param=ProtoWParams(
                reference_poly=1,
                horizontal_poly=[1.0, 2.0],
                vertical_poly=[3.0],
                horizontal_poly_inverse=[4.0],
                vertical_poly_inverse=[5.0, 6.0],
            ),
        )

        camera_parameters = CameraBank._get_camera_model_parameters(intrinsics)
        assert isinstance(camera_parameters, ncore.data.FThetaCameraModelParameters)
        np.testing.assert_array_equal(camera_parameters.resolution, np.array([320, 567], dtype=np.uint64))
        np.testing.assert_equal(camera_parameters.shutter_type, 1)
        np.testing.assert_array_equal(camera_parameters.principal_point, np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_equal(camera_parameters.reference_poly, 1)
        np.testing.assert_array_equal(
            camera_parameters.pixeldist_to_angle_poly, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            camera_parameters.angle_to_pixeldist_poly, np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32)
        )
        np.testing.assert_equal(camera_parameters.max_angle, 1.0)
        np.testing.assert_array_equal(camera_parameters.linear_cde, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        ext = camera_parameters.external_distortion_parameters
        np.testing.assert_equal(ext.reference_poly, ncore.data.ReferencePolynomial.BACKWARD)
        np.testing.assert_array_equal(ext.horizontal_poly, np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(ext.vertical_poly, np.array([3.0], dtype=np.float32))
        np.testing.assert_array_equal(ext.horizontal_poly_inverse, np.array([4.0], dtype=np.float32))
        np.testing.assert_array_equal(ext.vertical_poly_inverse, np.array([5.0, 6.0], dtype=np.float32))


class TestOpencvFisheyeCameraModelParameters(unittest.TestCase):
    def test_get_camera_model_parameters_populates_external_distortion(self):
        intrinsics = CameraSpec(
            logical_id="camera1",
            resolution_h=567,
            resolution_w=320,
            shutter_type=1,
            opencv_fisheye_param=OpenCVFisheyeCameraParam(
                principal_point_x=1.0,
                principal_point_y=2.0,
                focal_length_x=3.0,
                focal_length_y=4.0,
                radial_coeffs=[5.0, 6.0, 7.0, 8.0],
                max_angle=1.0,
            ),
        )

        camera_parameters = CameraBank._get_camera_model_parameters(intrinsics)
        assert isinstance(camera_parameters, ncore.data.OpenCVFisheyeCameraModelParameters)
        np.testing.assert_array_equal(camera_parameters.resolution, np.array([320, 567], dtype=np.uint64))
        np.testing.assert_equal(camera_parameters.shutter_type, 1)
        np.testing.assert_array_equal(camera_parameters.principal_point, np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(camera_parameters.focal_length, np.array([3.0, 4.0], dtype=np.float32))
        np.testing.assert_array_equal(camera_parameters.radial_coeffs, np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
        np.testing.assert_equal(camera_parameters.max_angle, 1.0)

        assert not intrinsics.HasField("external_distortion")


class TestOpencvPinholeCameraModelParameters(unittest.TestCase):
    def test_get_camera_model_parameters_populates_external_distortion(self):
        intrinsics = CameraSpec(
            logical_id="camera1",
            resolution_h=567,
            resolution_w=320,
            shutter_type=1,
            opencv_pinhole_param=OpenCVPinholeCameraParam(
                principal_point_x=1.0,
                principal_point_y=2.0,
                focal_length_x=3.0,
                focal_length_y=4.0,
                radial_coeffs=[5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                tangential_coeffs=[11.0, 12.0],
                thin_prism_coeffs=[13.0, 14.0, 15.0, 16.0],
            ),
        )

        camera_parameters = CameraBank._get_camera_model_parameters(intrinsics)
        assert isinstance(camera_parameters, ncore.data.OpenCVPinholeCameraModelParameters)
        np.testing.assert_array_equal(camera_parameters.resolution, np.array([320, 567], dtype=np.uint64))
        np.testing.assert_equal(camera_parameters.shutter_type, 1)
        np.testing.assert_array_equal(camera_parameters.principal_point, np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(camera_parameters.focal_length, np.array([3.0, 4.0], dtype=np.float32))
        np.testing.assert_array_equal(
            camera_parameters.radial_coeffs, np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(camera_parameters.tangential_coeffs, np.array([11.0, 12.0], dtype=np.float32))
        np.testing.assert_array_equal(
            camera_parameters.thin_prism_coeffs, np.array([13.0, 14.0, 15.0, 16.0], dtype=np.float32)
        )

        assert not intrinsics.HasField("external_distortion")


class TestActorTracksConversion(unittest.TestCase):
    @staticmethod
    def _signed_tracks(timestamps, quaternions=None):
        from nurec.render.actors import ActorTracks

        if quaternions is None:
            quaternions = [[0, 0, 0, 1]] * len(timestamps)
        return ActorTracks.from_dict_cpu(
            {
                "tracks_data": {
                    "tracks_id": ["44"],
                    "tracks_poses": [[[3 + t / 1000, 0, 0, *q] for t, q in zip(timestamps, quaternions)]],
                    "tracks_timestamps_us": [timestamps],
                    "tracks_label_class": ["automobile"],
                    "tracks_flags": ["DYNAMIC|CONTROLLABLE"],
                },
                "cuboidtracks_data": {"cuboids_dims": [[4.5, 2, 1.8]]},
            }
        )

    def test_negative_keyframes_are_clipped_with_interpolated_zero_pose(self):
        objects = actor_tracks_to_grpc(self._signed_tracks([-2622, 1000, 2000]), {"44": "car"})
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].asset_id, "car")
        poses = objects[0].trajectory.poses
        self.assertEqual([p.timestamp_us for p in poses], [0, 1000, 2000])
        np.testing.assert_allclose([p.pose.vec.x for p in poses], [3, 4, 5], atol=1e-5)

    def test_existing_zero_keyframe_is_not_duplicated(self):
        objects = actor_tracks_to_grpc(self._signed_tracks([-2622, 0, 1000]))
        self.assertEqual([p.timestamp_us for p in objects[0].trajectory.poses], [0, 1000])

    def test_zero_boundary_interpolates_rotation_as_well_as_translation(self):
        tracks = self._signed_tracks([-1000, 1000], [[0, 0, 0, 1], [0, 0, 2**-0.5, 2**-0.5]])
        pose = actor_tracks_to_grpc(tracks)[0].trajectory.poses[0].pose
        self.assertAlmostEqual(pose.vec.x, 3, places=5)
        self.assertAlmostEqual(pose.quat.z, np.sin(np.pi / 8), places=5)
        self.assertAlmostEqual(pose.quat.w, np.cos(np.pi / 8), places=5)

    def test_tracks_ending_before_zero_are_omitted(self):
        self.assertEqual(actor_tracks_to_grpc(self._signed_tracks([-2622, -1000])), [])

    def test_clipped_zero_singleton_round_trips_without_extending_its_lifetime(self):
        objects = actor_tracks_to_grpc(self._signed_tracks([-1000, 0]))
        self.assertEqual([p.timestamp_us for p in objects[0].trajectory.poses], [0])
        imported = actor_tracks_from_grpc(objects)
        snapshot = imported.get_snapshot_at_frame(0, 1)
        self.assertEqual(snapshot.actor_ids, ["44"])
        np.testing.assert_allclose(snapshot.actor_poses.cpu().numpy()[0, :, 0], [3, 3], atol=1e-5)
        self.assertEqual(imported.get_snapshot_at_frame(1, 2).num_actors(), 0)
        round_tripped = actor_tracks_to_grpc(imported)
        self.assertEqual([p.timestamp_us for p in round_tripped[0].trajectory.poses], [0])
        self.assertAlmostEqual(round_tripped[0].trajectory.poses[0].pose.vec.x, 3, places=5)

    def test_unsigned_trajectory_rejects_decreasing_timestamps(self):
        from nurec.grpc.trajectory import QVec, Trajectory

        with self.assertRaises(AssertionError):
            Trajectory(
                timestamps_us=np.array([18446744073709548994, 1000], dtype=np.uint64),
                poses=QVec.from_se3(np.stack([np.eye(4), np.eye(4)])),
            )

    def test_actor_tracks_round_trip(self):
        track = DynamicObjectTrack()
        track.id = "1"
        track.semantic_class = "automobile"

        track.object_size.size_x = 4.5
        track.object_size.size_y = 2.0
        track.object_size.size_z = 1.8

        trajectory = ProtoTrajectory()

        timestamps = [0, 500000, 1000000, 1500000, 2000000]

        for i, timestamp_us in enumerate(timestamps):
            pose = Pose()
            pose.vec.x = i * 5.0
            pose.vec.y = 0.0
            pose.vec.z = 0.0
            pose.quat.w = 1.0
            pose.quat.x = 0.0
            pose.quat.y = 0.0
            pose.quat.z = 0.0

            pose_at_time = PoseAtTime()
            pose_at_time.pose.CopyFrom(pose)
            pose_at_time.timestamp_us = timestamp_us

            trajectory.poses.append(pose_at_time)

        track.trajectory.CopyFrom(trajectory)

        original_tracks = [track]

        actor_tracks = actor_tracks_from_grpc(original_tracks)
        converted_tracks = actor_tracks_to_grpc(actor_tracks)

        self.assertEqual(len(converted_tracks), 1)
        converted_track = converted_tracks[0]

        self.assertEqual(converted_track.id, "1")
        self.assertEqual(converted_track.semantic_class, "automobile")

        np.testing.assert_almost_equal(converted_track.object_size.size_x, 4.5, decimal=6)
        np.testing.assert_almost_equal(converted_track.object_size.size_y, 2.0, decimal=6)
        np.testing.assert_almost_equal(converted_track.object_size.size_z, 1.8, decimal=6)

        self.assertEqual(len(converted_track.trajectory.poses), 5)

        for i in range(5):
            original_pose_at_time = track.trajectory.poses[i]
            converted_pose_at_time = converted_track.trajectory.poses[i]

            self.assertEqual(converted_pose_at_time.timestamp_us, timestamps[i])

            np.testing.assert_equal(converted_pose_at_time.pose.vec.x, original_pose_at_time.pose.vec.x)
            np.testing.assert_equal(converted_pose_at_time.pose.vec.y, original_pose_at_time.pose.vec.y)
            np.testing.assert_equal(converted_pose_at_time.pose.vec.z, original_pose_at_time.pose.vec.z)

            np.testing.assert_equal(converted_pose_at_time.pose.quat.w, original_pose_at_time.pose.quat.w)
            np.testing.assert_equal(converted_pose_at_time.pose.quat.x, original_pose_at_time.pose.quat.x)
            np.testing.assert_equal(converted_pose_at_time.pose.quat.y, original_pose_at_time.pose.quat.y)
            np.testing.assert_equal(converted_pose_at_time.pose.quat.z, original_pose_at_time.pose.quat.z)


class TestLidarPerRayEncoding(unittest.TestCase):
    """Backend._lidar_ray_frame_to_grpc: the wire form of a per-ray render."""

    def _backend(self, target_scale: float = 1.0):
        backend = object.__new__(Backend)
        backend.world_to_nurec = FrameConversion.from_origin_scale_axis(
            target_origin=np.zeros(3, dtype=np.float32), target_scale=target_scale, target_axis=[0, 1, 2]
        )
        return backend

    @staticmethod
    def _frame(distances, valid, intensities=None):
        return LidarRayFrame(
            ray_distances=torch.tensor(distances, dtype=torch.float32),
            ray_valid=torch.tensor(valid, dtype=torch.bool),
            ray_intensities=None if intensities is None else torch.tensor(intensities, dtype=torch.float32),
        )

    def test_reports_every_emitted_ray_including_the_ones_that_did_not_return(self):
        # The point of the per-ray form: index i is stable for one firing-pattern ID, so a
        # dropped ray occupies its slot rather than shifting the rest.
        result = self._backend()._lidar_ray_frame_to_grpc(
            self._frame([1.5, 9.0, 3.25], [True, False, True], [0.5, 0.0, 0.25])
        )

        self.assertEqual(result.num_rays, 3)
        np.testing.assert_allclose(np.frombuffer(result.ray_distances_m_buffer, dtype=np.float32), [1.5, 9.0, 3.25])
        np.testing.assert_allclose(np.frombuffer(result.ray_intensities_buffer, dtype=np.float32), [0.5, 0.0, 0.25])

    def test_valid_mask_is_packed_lsb_first_as_the_proto_documents(self):
        # Rays 0 and 2 valid -> 0b101 in the low bits of the first byte. Big-endian packing
        # would decode as rays 5 and 7 on the client, silently mislabelling every return.
        result = self._backend()._lidar_ray_frame_to_grpc(self._frame([1.0, 2.0, 3.0], [True, False, True]))

        self.assertEqual(result.ray_valid_masks_packed[0] & 0b111, 0b101)
        unpacked = np.unpackbits(
            np.frombuffer(result.ray_valid_masks_packed, dtype=np.uint8), count=3, bitorder="little"
        )
        np.testing.assert_array_equal(unpacked, [1, 0, 1])

    def test_valid_mask_covers_a_ray_count_that_is_not_a_whole_byte(self):
        n = 437040  # the ET30x frame: 1214 columns x 360 lines
        valid = np.zeros(n, dtype=bool)
        valid[[0, 7, 8, n - 1]] = True

        result = self._backend()._lidar_ray_frame_to_grpc(self._frame(np.zeros(n), valid))

        self.assertEqual(len(result.ray_valid_masks_packed), (n + 7) // 8)
        unpacked = np.unpackbits(
            np.frombuffer(result.ray_valid_masks_packed, dtype=np.uint8), count=n, bitorder="little"
        )
        np.testing.assert_array_equal(np.flatnonzero(unpacked), [0, 7, 8, n - 1])

    def test_point_cloud_fields_stay_empty(self):
        # A client reading the wrong pair of fields should get nothing, not stale points.
        result = self._backend()._lidar_ray_frame_to_grpc(self._frame([1.0], [True]))

        self.assertEqual(result.num_points, 0)
        self.assertEqual(result.point_xyzs_buffer, b"")
        self.assertEqual(len(result.point_xyzs), 0)

    def test_intensity_is_omitted_when_the_model_predicts_none(self):
        result = self._backend()._lidar_ray_frame_to_grpc(self._frame([1.0, 2.0], [True, True]))

        self.assertEqual(result.ray_intensities_buffer, b"")
        self.assertEqual(result.num_rays, 2)

    def test_firing_pattern_is_absent_unless_the_request_asked_for_it(self):
        # The default keeps a response inside gRPC's stock 4 MiB receive limit; embedding the
        # selected pattern would repeat ~8 MiB in every frame instead of caching it by ID.
        result = self._backend()._lidar_ray_frame_to_grpc(self._frame([1.0], [True]))

        self.assertFalse(result.HasField("firing_pattern"))

    def test_selected_pattern_id_is_returned_without_embedding_the_pattern(self):
        pattern = LidarFiringPatternTemplate(
            directions_sensor=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            timestamp_offsets_us=np.array([0], dtype=np.int64),
        )

        result = self._backend()._lidar_ray_frame_to_grpc(
            self._frame([1.0], [True]),
            firing_pattern_id=pattern.pattern_id(15),
        )

        self.assertEqual(result.firing_pattern_id, pattern.pattern_id(15))
        self.assertFalse(result.HasField("firing_pattern"))

    def test_embedded_firing_pattern_matches_the_standalone_rpc(self):
        # A client mixing the two sources must not see them disagree about the sensor.
        pattern = LidarFiringPatternTemplate(
            directions_sensor=np.eye(3, dtype=np.float32)[:, :3],
            timestamp_offsets_us=np.array([0, 5, 10], dtype=np.int64),
        )

        result = self._backend()._lidar_ray_frame_to_grpc(
            self._frame([1.0, 2.0, 3.0], [True, True, False]),
            firing_pattern=pattern,
            firing_pattern_id=pattern.pattern_id(15),
            frame_period_us=15,
        )

        self.assertTrue(result.HasField("firing_pattern"))
        expected = LidarFiringPatternReturn()
        _fill_firing_pattern(pattern, 15, expected)
        self.assertEqual(result.firing_pattern, expected)
        self.assertEqual(result.firing_pattern.num_rays, 3)
        self.assertEqual(result.firing_pattern_id, pattern.pattern_id(15))

    def test_embedded_pattern_carries_the_grid_when_the_sensor_has_one(self):
        # A unit that fires a whole column at once: the grid follows from these times.
        pattern = LidarFiringPatternTemplate(
            directions_sensor=np.tile(np.array([1.0, 0.0, 0.0], np.float32), (6, 1)),
            timestamp_offsets_us=np.repeat(np.arange(3, dtype=np.int64) * 49, 2),
            packet_ray_offsets=np.array([0, 2, 4, 6], dtype=np.int64),
        )

        result = self._backend()._lidar_ray_frame_to_grpc(
            self._frame([1.0] * 6, [True] * 6),
            firing_pattern=pattern,
            firing_pattern_id=pattern.pattern_id(3 * 49),
            frame_period_us=3 * 49,
        )

        self.assertEqual((result.firing_pattern.num_columns, result.firing_pattern.num_rows), (3, 2))
        self.assertEqual(result.firing_pattern.packet_layout.num_packets, 3)
        np.testing.assert_array_equal(
            np.frombuffer(result.firing_pattern.packet_layout.ray_offsets_u32_le, dtype="<u4"),
            [0, 2, 4, 6],
        )
        self.assertEqual(result.firing_pattern.timestamp_offsets_us, b"")
        np.testing.assert_array_equal(
            np.frombuffer(result.firing_pattern.packet_layout.timestamp_offsets_us, dtype=np.int64),
            [0, 49, 98],
        )

    def test_embedded_pattern_keeps_per_ray_timestamps_when_they_vary_inside_a_packet(self):
        pattern = LidarFiringPatternTemplate(
            directions_sensor=np.tile(np.array([1.0, 0.0, 0.0], np.float32), (3, 1)),
            timestamp_offsets_us=np.array([0, 5, 10], dtype=np.int64),
            packet_ray_offsets=np.array([0, 3], dtype=np.int64),
        )

        result = self._backend()._lidar_ray_frame_to_grpc(
            self._frame([1.0] * 3, [True] * 3),
            firing_pattern=pattern,
            firing_pattern_id=pattern.pattern_id(15),
            frame_period_us=15,
        )

        np.testing.assert_array_equal(
            np.frombuffer(result.firing_pattern.timestamp_offsets_us, dtype=np.int64),
            [0, 5, 10],
        )
        self.assertEqual(result.firing_pattern.packet_layout.timestamp_offsets_us, b"")

    def test_encodes_one_strongest_return_layer(self):
        result = self._backend()._lidar_ray_frame_to_grpc(self._frame([1.0], [True]))

        self.assertEqual(result.num_return_layers, 1)
        self.assertEqual(list(result.return_types), [LidarReturnType.LIDAR_RETURN_STRONGEST])

    def test_first_layer_repeats_the_strongest_layer(self):
        result = self._backend()._lidar_ray_frame_to_grpc(
            self._frame([1.0, 2.0, 3.0], [True, False, True], [0.1, 0.2, 0.3]),
            return_types=(LidarReturnType.LIDAR_RETURN_FIRST, LidarReturnType.LIDAR_RETURN_STRONGEST),
        )

        self.assertEqual(result.num_return_layers, 2)
        self.assertEqual(
            list(result.return_types),
            [LidarReturnType.LIDAR_RETURN_FIRST, LidarReturnType.LIDAR_RETURN_STRONGEST],
        )
        distances = np.frombuffer(result.ray_distances_m_buffer, dtype=np.float32).reshape(2, 3)
        intensities = np.frombuffer(result.ray_intensities_buffer, dtype=np.float32).reshape(2, 3)
        np.testing.assert_allclose(distances[0], distances[1])
        np.testing.assert_allclose(intensities[0], intensities[1])
        self.assertEqual(result.ray_valid_masks_packed, bytes([0b101, 0b101]))

    def test_scaled_world_to_nurec_distances_are_reported_in_metres(self):
        result = self._backend(target_scale=0.5)._lidar_ray_frame_to_grpc(self._frame([0.5, 1.0], [True, True]))

        np.testing.assert_allclose(
            np.frombuffer(result.ray_distances_m_buffer, dtype=np.float32),
            [1.0, 2.0],
        )


class TestLidarRenderFilter(unittest.TestCase):
    """Test LidarRenderFilter protobuf message functionality."""

    def test_lidar_render_filter_defaults_to_unset(self):
        """Test that filter fields are unset by default."""
        filter_msg = LidarRenderFilter()

        self.assertFalse(filter_msg.HasField("raydrop_threshold"))
        self.assertFalse(filter_msg.HasField("opacity_threshold"))
        self.assertFalse(filter_msg.HasField("enable_distance_filter"))
        self.assertFalse(filter_msg.HasField("distance_filter_threshold"))

    def test_lidar_render_filter_set_all_fields(self):
        """Test setting all filter fields."""
        filter_msg = LidarRenderFilter(
            raydrop_threshold=0.5,
            opacity_threshold=0.0,
            enable_distance_filter=True,
            distance_filter_threshold=0.02,
        )

        self.assertTrue(filter_msg.HasField("raydrop_threshold"))
        self.assertTrue(filter_msg.HasField("opacity_threshold"))
        self.assertTrue(filter_msg.HasField("enable_distance_filter"))
        self.assertTrue(filter_msg.HasField("distance_filter_threshold"))

        np.testing.assert_almost_equal(filter_msg.raydrop_threshold, 0.5)
        np.testing.assert_almost_equal(filter_msg.opacity_threshold, 0.0)
        self.assertTrue(filter_msg.enable_distance_filter)
        np.testing.assert_almost_equal(filter_msg.distance_filter_threshold, 0.02)

    def test_lidar_render_filter_partial_fields(self):
        """Test setting only some filter fields."""
        filter_msg = LidarRenderFilter(
            raydrop_threshold=0.3,
            enable_distance_filter=False,
        )

        self.assertTrue(filter_msg.HasField("raydrop_threshold"))
        self.assertFalse(filter_msg.HasField("opacity_threshold"))
        self.assertTrue(filter_msg.HasField("enable_distance_filter"))
        self.assertFalse(filter_msg.HasField("distance_filter_threshold"))

        np.testing.assert_almost_equal(filter_msg.raydrop_threshold, 0.3)
        self.assertFalse(filter_msg.enable_distance_filter)

    def test_lidar_render_filter_zero_values_are_set(self):
        """Test that zero values are correctly detected as set (not default)."""
        filter_msg = LidarRenderFilter(
            raydrop_threshold=0.0,
            opacity_threshold=0.0,
            distance_filter_threshold=0.0,
        )

        # Zero values should still be detected as "set"
        self.assertTrue(filter_msg.HasField("raydrop_threshold"))
        self.assertTrue(filter_msg.HasField("opacity_threshold"))
        self.assertTrue(filter_msg.HasField("distance_filter_threshold"))

        np.testing.assert_almost_equal(filter_msg.raydrop_threshold, 0.0)
        np.testing.assert_almost_equal(filter_msg.opacity_threshold, 0.0)
        np.testing.assert_almost_equal(filter_msg.distance_filter_threshold, 0.0)


class TestLidarRenderRequest(unittest.TestCase):
    """Test LidarRenderRequest with render_filter field."""

    def test_lidar_render_request_without_filter(self):
        """Test that LidarRenderRequest works without a filter."""
        request = LidarRenderRequest(
            scene_id="test_scene",
            frame_start_us=1000000,
            frame_end_us=1100000,
        )

        self.assertEqual(request.scene_id, "test_scene")
        self.assertEqual(request.frame_start_us, 1000000)
        self.assertEqual(request.frame_end_us, 1100000)
        self.assertFalse(request.HasField("render_filter"))

    def test_lidar_render_request_with_filter(self):
        """Test that LidarRenderRequest correctly includes render_filter."""
        render_filter = LidarRenderFilter(
            raydrop_threshold=0.5,
            opacity_threshold=0.0,
            enable_distance_filter=True,
            distance_filter_threshold=0.02,
        )

        request = LidarRenderRequest(
            scene_id="test_scene",
            frame_start_us=1000000,
            frame_end_us=1100000,
            render_filter=render_filter,
        )

        self.assertTrue(request.HasField("render_filter"))
        np.testing.assert_almost_equal(request.render_filter.raydrop_threshold, 0.5)
        np.testing.assert_almost_equal(request.render_filter.opacity_threshold, 0.0)
        self.assertTrue(request.render_filter.enable_distance_filter)
        np.testing.assert_almost_equal(request.render_filter.distance_filter_threshold, 0.02)

    def test_lidar_render_request_with_partial_filter(self):
        """Test LidarRenderRequest with partially set filter fields."""
        render_filter = LidarRenderFilter(
            raydrop_threshold=0.7,
        )

        request = LidarRenderRequest(
            scene_id="test_scene",
            frame_start_us=1000000,
            frame_end_us=1100000,
            render_filter=render_filter,
        )

        self.assertTrue(request.HasField("render_filter"))
        self.assertTrue(request.render_filter.HasField("raydrop_threshold"))
        self.assertFalse(request.render_filter.HasField("opacity_threshold"))
        self.assertFalse(request.render_filter.HasField("enable_distance_filter"))
        self.assertFalse(request.render_filter.HasField("distance_filter_threshold"))

        np.testing.assert_almost_equal(request.render_filter.raydrop_threshold, 0.7)

    def test_lidar_render_request_with_lidar_spec(self):
        """Test LidarRenderRequest with both lidar_config and render_filter."""
        lidar_spec = LidarSpec(
            lidar_type=LidarSpec.DESCRIPTOR.fields_by_name["lidar_type"].enum_type.values_by_name["AT128"].number,
        )

        render_filter = LidarRenderFilter(
            raydrop_threshold=0.5,
            enable_distance_filter=False,
        )

        request = LidarRenderRequest(
            scene_id="test_scene",
            lidar_config=lidar_spec,
            frame_start_us=1000000,
            frame_end_us=1100000,
            render_filter=render_filter,
        )

        self.assertEqual(request.scene_id, "test_scene")
        self.assertEqual(request.lidar_config.lidar_type, 1)
        self.assertTrue(request.HasField("render_filter"))
        np.testing.assert_almost_equal(request.render_filter.raydrop_threshold, 0.5)
        self.assertFalse(request.render_filter.enable_distance_filter)


class TestLidarFilterExtraction(unittest.TestCase):
    """Test the filter extraction logic used in serve.py."""

    def _extract_filter_params(self, request: LidarRenderRequest):
        """
        Mimics the filter extraction logic from serve.py render_lidar_request().
        Returns a dict of filter parameters, with None for unset numeric values. neighbor_source is
        not Optional: serve.py seeds it from the model's server-config default
        (default_lidar_neighbor_source(), "rangeview" unless the artifact configures otherwise) and
        the request overrides it; here we use "rangeview" to stand in for that default.
        """
        raydrop_threshold = None
        opacity_threshold = None
        enable_distance_filter = None
        distance_filter_threshold = None
        neighbor_source = "rangeview"
        elevation_resolution_deg = None
        azimuth_max_gap_deg = None

        if request.HasField("render_filter"):
            render_filter = request.render_filter
            if render_filter.HasField("raydrop_threshold"):
                raydrop_threshold = render_filter.raydrop_threshold
            if render_filter.HasField("opacity_threshold"):
                opacity_threshold = render_filter.opacity_threshold
            if render_filter.HasField("enable_distance_filter"):
                enable_distance_filter = render_filter.enable_distance_filter
            if render_filter.HasField("distance_filter_threshold"):
                distance_filter_threshold = render_filter.distance_filter_threshold
            if render_filter.HasField("neighbor_source"):
                neighbor_source = (
                    "angle" if render_filter.neighbor_source == LIDAR_FLOATING_FILTER_SOURCE_ANGLE else "rangeview"
                )
            if render_filter.HasField("elevation_resolution_deg"):
                elevation_resolution_deg = render_filter.elevation_resolution_deg
            if render_filter.HasField("azimuth_max_gap_deg"):
                azimuth_max_gap_deg = render_filter.azimuth_max_gap_deg

        return {
            "raydrop_threshold": raydrop_threshold,
            "opacity_threshold": opacity_threshold,
            "enable_distance_filter": enable_distance_filter,
            "distance_filter_threshold": distance_filter_threshold,
            "neighbor_source": neighbor_source,
            "elevation_resolution_deg": elevation_resolution_deg,
            "azimuth_max_gap_deg": azimuth_max_gap_deg,
        }

    def test_extract_filter_params_no_filter(self):
        """Test extraction when no filter is provided."""
        request = LidarRenderRequest(scene_id="test")
        params = self._extract_filter_params(request)

        self.assertIsNone(params["raydrop_threshold"])
        self.assertIsNone(params["opacity_threshold"])
        self.assertIsNone(params["enable_distance_filter"])
        self.assertIsNone(params["distance_filter_threshold"])
        # neighbor_source is not Optional: an absent field defaults concretely to "rangeview".
        self.assertEqual(params["neighbor_source"], "rangeview")
        self.assertIsNone(params["elevation_resolution_deg"])
        self.assertIsNone(params["azimuth_max_gap_deg"])

    def test_extract_filter_params_all_set(self):
        """Test extraction when all filter params are set."""
        request = LidarRenderRequest(
            scene_id="test",
            render_filter=LidarRenderFilter(
                raydrop_threshold=0.5,
                opacity_threshold=0.0,
                enable_distance_filter=True,
                distance_filter_threshold=0.02,
            ),
        )
        params = self._extract_filter_params(request)

        np.testing.assert_almost_equal(params["raydrop_threshold"], 0.5)
        np.testing.assert_almost_equal(params["opacity_threshold"], 0.0)
        self.assertTrue(params["enable_distance_filter"])
        np.testing.assert_almost_equal(params["distance_filter_threshold"], 0.02)

    def test_extract_filter_params_partial_set(self):
        """Test extraction when only some filter params are set."""
        request = LidarRenderRequest(
            scene_id="test",
            render_filter=LidarRenderFilter(
                raydrop_threshold=0.3,
            ),
        )
        params = self._extract_filter_params(request)

        np.testing.assert_almost_equal(params["raydrop_threshold"], 0.3)
        self.assertIsNone(params["opacity_threshold"])
        self.assertIsNone(params["enable_distance_filter"])
        self.assertIsNone(params["distance_filter_threshold"])

    def test_extract_filter_params_zero_values(self):
        """Test that zero values are extracted correctly (not as None)."""
        request = LidarRenderRequest(
            scene_id="test",
            render_filter=LidarRenderFilter(
                raydrop_threshold=0.0,
                opacity_threshold=0.0,
            ),
        )
        params = self._extract_filter_params(request)

        # Zero should be extracted as 0.0, not None
        np.testing.assert_almost_equal(params["raydrop_threshold"], 0.0)
        np.testing.assert_almost_equal(params["opacity_threshold"], 0.0)
        self.assertIsNone(params["enable_distance_filter"])
        self.assertIsNone(params["distance_filter_threshold"])

    def test_extract_filter_params_boolean_false(self):
        """Test that boolean False is extracted correctly."""
        request = LidarRenderRequest(
            scene_id="test",
            render_filter=LidarRenderFilter(
                enable_distance_filter=False,
            ),
        )
        params = self._extract_filter_params(request)

        self.assertIsNone(params["raydrop_threshold"])
        self.assertIsNone(params["opacity_threshold"])
        self.assertFalse(params["enable_distance_filter"])
        self.assertIsNone(params["distance_filter_threshold"])

    def test_extract_filter_params_angle_source(self):
        """Angle neighbour source + its params map through (enum -> str)."""
        request = LidarRenderRequest(
            scene_id="test",
            render_filter=LidarRenderFilter(
                enable_distance_filter=True,
                distance_filter_threshold=0.02,
                neighbor_source=LIDAR_FLOATING_FILTER_SOURCE_ANGLE,
                elevation_resolution_deg=0.055,
                azimuth_max_gap_deg=1.0,
            ),
        )
        params = self._extract_filter_params(request)

        self.assertEqual(params["neighbor_source"], "angle")
        np.testing.assert_almost_equal(params["elevation_resolution_deg"], 0.055)
        np.testing.assert_almost_equal(params["azimuth_max_gap_deg"], 1.0)

    def test_extract_filter_params_rangeview_source_default(self):
        """RANGEVIEW enum maps to 'rangeview'; unset angle params stay None."""
        request = LidarRenderRequest(
            scene_id="test",
            render_filter=LidarRenderFilter(neighbor_source=LIDAR_FLOATING_FILTER_SOURCE_RANGEVIEW),
        )
        params = self._extract_filter_params(request)

        self.assertEqual(params["neighbor_source"], "rangeview")
        self.assertIsNone(params["elevation_resolution_deg"])
        self.assertIsNone(params["azimuth_max_gap_deg"])


class TestRenderErrorClassification(unittest.TestCase):
    def test_timestamp_out_of_range_maps_to_out_of_range(self):
        status, message = _classify_render_exception(TimestampOutOfRangeError("timestamp outside range"))
        self.assertEqual(status, grpc.StatusCode.OUT_OF_RANGE)
        self.assertEqual(message, "timestamp outside range")

    def test_model_incompatibility_maps_to_failed_precondition(self):
        status, message = _classify_render_exception(ModelIncompatibilityError("model mismatch"))
        self.assertEqual(status, grpc.StatusCode.FAILED_PRECONDITION)
        self.assertEqual(message, "model mismatch")

    def test_value_error_maps_to_invalid_argument(self):
        status, message = _classify_render_exception(ValueError("bad argument"))
        self.assertEqual(status, grpc.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(message, "bad argument")

    def test_key_error_maps_to_not_found(self):
        status, message = _classify_render_exception(KeyError("missing-scene"))
        self.assertEqual(status, grpc.StatusCode.NOT_FOUND)
        self.assertEqual(message, "missing-scene")

    def test_out_of_memory_maps_to_resource_exhausted(self):
        status, message = _classify_render_exception(torch.OutOfMemoryError("oom"))
        self.assertEqual(status, grpc.StatusCode.RESOURCE_EXHAUSTED)
        self.assertIn("oom", message)
        self.assertIn("unconverged checkpoint", message)

    def test_cache_full_maps_to_resource_exhausted(self):
        status, message = _classify_render_exception(CacheFullError("cache is full"))
        self.assertEqual(status, grpc.StatusCode.RESOURCE_EXHAUSTED)
        self.assertEqual(message, "cache is full")

    def test_cache_wait_aborted_cancelled_maps_to_cancelled(self):
        status, message = _classify_render_exception(
            CacheWaitAbortedError("cancelled", CacheWaitAbortedReason.CANCELLED)
        )
        self.assertEqual(status, grpc.StatusCode.CANCELLED)
        self.assertEqual(message, "cancelled")

    def test_cache_wait_aborted_timeout_maps_to_deadline_exceeded(self):
        status, message = _classify_render_exception(
            CacheWaitAbortedError("timeout", CacheWaitAbortedReason.SERVER_TIMEOUT)
        )
        self.assertEqual(status, grpc.StatusCode.DEADLINE_EXCEEDED)
        self.assertEqual(message, "timeout")

    def test_cache_wait_aborted_client_deadline_maps_to_deadline_exceeded(self):
        status, message = _classify_render_exception(
            CacheWaitAbortedError("deadline", CacheWaitAbortedReason.DEADLINE_EXCEEDED)
        )
        self.assertEqual(status, grpc.StatusCode.DEADLINE_EXCEEDED)
        self.assertEqual(message, "deadline")

    def test_backend_cache_wait_policy_bounds_deadline(self):
        client_deadline = mock.MagicMock()
        client_deadline.is_active.return_value = True
        client_deadline.time_remaining.return_value = 5.0
        no_client_deadline = mock.MagicMock()
        no_client_deadline.is_active.return_value = True
        no_client_deadline.time_remaining.return_value = None

        for context, server_timeout_s in ((client_deadline, 10.0), (no_client_deadline, 5.0), (None, 5.0)):
            with self.subTest(context=context):
                before = time.monotonic()
                policy = _grpc_backend_cache_wait_policy(context, server_timeout_s=server_timeout_s)
                assert policy is not None
                assert policy.wait_deadline is not None
                self.assertLessEqual(policy.wait_deadline, before + 5.1)
                self.assertGreater(policy.wait_deadline, before + 4.5)

    def test_grpc_backend_cache_wait_policy_respects_inactive_context(self):
        context = mock.MagicMock()
        context.is_active.return_value = False
        context.time_remaining.return_value = None
        policy = _grpc_backend_cache_wait_policy(context)
        assert policy is not None
        assert policy.should_continue is not None
        self.assertFalse(policy.should_continue())
        assert policy.abort_reason is not None
        self.assertEqual(policy.abort_reason(), CacheWaitAbortedReason.CANCELLED)

    def test_grpc_backend_cache_wait_policy_maps_expired_deadline(self):
        context = mock.MagicMock()
        context.is_active.return_value = True
        context.time_remaining.return_value = 0.0
        policy = _grpc_backend_cache_wait_policy(context)
        assert policy is not None
        assert policy.should_continue is not None
        self.assertFalse(policy.should_continue())
        assert policy.abort_reason is not None
        self.assertEqual(policy.abort_reason(), CacheWaitAbortedReason.DEADLINE_EXCEEDED)

    def test_grpc_backend_cache_wait_policy_prefers_expired_deadline_when_inactive(self):
        context = mock.MagicMock()
        context.is_active.return_value = False
        context.time_remaining.return_value = 0.0
        policy = _grpc_backend_cache_wait_policy(context)
        assert policy is not None
        assert policy.abort_reason is not None
        self.assertEqual(policy.abort_reason(), CacheWaitAbortedReason.DEADLINE_EXCEEDED)

    def test_grpc_backend_cache_wait_policy_stops_for_fatal_worker(self):
        context = mock.MagicMock()
        context.is_active.return_value = True
        context.time_remaining.return_value = None
        fatal_worker_state = FatalWorkerState()
        policy = _grpc_backend_cache_wait_policy(context, fatal_worker_state=fatal_worker_state)
        assert policy is not None
        assert policy.should_continue is not None
        self.assertTrue(policy.should_continue())

        fatal_worker_state.mark_fatal(
            FatalWorkerIncident(
                rpc_method="render_rgb",
                scene_ids=("scene",),
                request_id=None,
                process_id=1,
                diagnostics={},
            )
        )

        self.assertFalse(policy.should_continue())
        assert policy.abort_reason is not None
        self.assertEqual(policy.abort_reason(), CacheWaitAbortedReason.WORKER_UNAVAILABLE)

    def test_cache_wait_aborted_worker_unavailable_maps_to_unavailable(self):
        status, message = _classify_render_exception(
            CacheWaitAbortedError("stopped", CacheWaitAbortedReason.WORKER_UNAVAILABLE)
        )
        self.assertEqual(status, grpc.StatusCode.UNAVAILABLE)
        self.assertIn("replacement worker", message)

    def test_no_spare_backends_maps_to_resource_exhausted(self):
        status, message = _classify_render_exception(NoSpareBackendsError("no spare backends"))
        self.assertEqual(status, grpc.StatusCode.RESOURCE_EXHAUSTED)
        self.assertEqual(message, "no spare backends")

    def test_unknown_exception_maps_to_unknown(self):
        status, message = _classify_render_exception(RuntimeError("unexpected"))
        self.assertEqual(status, grpc.StatusCode.UNKNOWN)
        self.assertEqual(message, "unexpected")

    def test_cuda_worker_diagnostics_use_cuda_identity_independently_of_visibility(self):
        class _CudaUuid:
            def __str__(self):
                return "01234567-89ab-cdef-0123-456789abcdef"

        for visibility in (None, "0,1", "GPU-a,GPU-b", "MIG-test", ",", ""):
            with (
                self.subTest(visibility=visibility),
                mock.patch.dict("os.environ", {}, clear=True),
                mock.patch("nurec.grpc.serve.torch.cuda.current_device", return_value=1),
                mock.patch(
                    "nurec.grpc.serve.torch.cuda.get_device_properties", return_value=SimpleNamespace(uuid=_CudaUuid())
                ) as properties,
                mock.patch(
                    "nurec.grpc.serve.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout="580.65\n580.65\n"),
                ) as run,
            ):
                if visibility is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = visibility
                diagnostics = collect_cuda_worker_diagnostics()
                self.assertEqual(diagnostics["gpu_uuid"], f"GPU-{_CudaUuid()}")
                self.assertEqual(diagnostics["nvidia_driver_version"], "580.65")
                self.assertEqual(diagnostics["cuda_device_index"], "1")
                properties.assert_called_once_with(1)
                self.assertFalse(any(arg.startswith("--id=") for arg in run.call_args.args[0]))

    def test_cuda_identity_survives_driver_query_failure(self):
        with (
            mock.patch("nurec.grpc.serve.torch.cuda.current_device", return_value=0),
            mock.patch(
                "nurec.grpc.serve.torch.cuda.get_device_properties",
                return_value=SimpleNamespace(uuid="01234567-89ab-cdef-0123-456789abcdef"),
            ),
            mock.patch("nurec.grpc.serve.subprocess.run", side_effect=FileNotFoundError),
        ):
            diagnostics = collect_cuda_worker_diagnostics()
        self.assertEqual(diagnostics["gpu_uuid"], "GPU-01234567-89ab-cdef-0123-456789abcdef")
        self.assertEqual(diagnostics["nvidia_driver_version"], "unavailable")

    def test_cuda_device_failure_does_not_guess_a_gpu(self):
        with (
            mock.patch("nurec.grpc.serve.torch.cuda.current_device", side_effect=RuntimeError("no device")),
            mock.patch("nurec.grpc.serve.torch.cuda.get_device_properties") as properties,
            mock.patch("nurec.grpc.serve.subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="")),
        ):
            diagnostics = collect_cuda_worker_diagnostics()
        self.assertEqual(diagnostics["gpu_uuid"], "unavailable")
        self.assertEqual(diagnostics["cuda_device_index"], "unavailable")
        properties.assert_not_called()


class TestBackendDynamicObjectValidation(unittest.TestCase):
    def test_dynamic_object_updates_require_server_flag(self):
        backend = Backend(
            renderable_model=mock.MagicMock(),
            camera_bank=mock.MagicMock(),
            world_to_nurec=mock.MagicMock(),
            scene_time_range_us=HalfClosedInterval(0, 1),
            lidar_bank=mock.MagicMock(),
            asset_bank=mock.MagicMock(),
            source_artifact=mock.MagicMock(),
            backend_cache_key=_test_backend_cache_key(),
        )

        with self.assertRaises(ValueError) as exc:
            backend._validate_dynamic_object_updates_request(num_dynamic_objects=1, enable_editing_actors=False)
        self.assertIn("--enable-editing-actors", str(exc.exception))

    def test_dynamic_object_updates_require_model_support(self):
        renderable_model = mock.MagicMock()
        renderable_model.supports_edit_actors.return_value = False
        backend = Backend(
            renderable_model=renderable_model,
            camera_bank=mock.MagicMock(),
            world_to_nurec=mock.MagicMock(),
            scene_time_range_us=HalfClosedInterval(0, 1),
            lidar_bank=mock.MagicMock(),
            asset_bank=mock.MagicMock(),
            source_artifact=mock.MagicMock(),
            backend_cache_key=_test_backend_cache_key(),
        )

        with self.assertRaises(ValueError) as exc:
            backend._validate_dynamic_object_updates_request(num_dynamic_objects=1, enable_editing_actors=True)
        self.assertIn("does not support actor updates", str(exc.exception))

    def test_insert_assets_uses_zero_dims_offset_not_track_padding(self):
        # asset_bank.dims_offset (track_padding_m) applies to replacement; inserted assets must
        # be scaled to their full cuboid dimensions, so insert_asset must receive a zero offset.
        renderable_model = mock.MagicMock()
        renderable_model.device = "cpu"
        renderable_model.insert_asset.return_value = True
        asset_bank = mock.MagicMock()
        asset_bank.has_track.return_value = True
        asset_bank.dims_offset = torch.tensor([0.1, 0.2, 0.3])  # non-zero padding, must NOT reach insert
        backend = Backend(
            renderable_model=renderable_model,
            camera_bank=mock.MagicMock(),
            world_to_nurec=mock.MagicMock(),
            scene_time_range_us=HalfClosedInterval(0, 1),
            lidar_bank=mock.MagicMock(),
            asset_bank=asset_bank,
            source_artifact=mock.MagicMock(),
            backend_cache_key=_test_backend_cache_key(),
        )

        cuboid_track = mock.MagicMock()
        cuboid_track.tracks_id = ["track-1"]
        cuboid_track.cuboids_dims = [[4.0, 2.0, 1.5]]
        actor_tracks = mock.MagicMock()
        actor_tracks._cuboid_tracks_list = [cuboid_track]

        dynamic_object = mock.MagicMock()
        dynamic_object.id = "track-1"
        dynamic_object.asset_id = "asset-1"

        with (
            mock.patch("nurec.grpc.serve.actor_tracks_from_grpc", return_value=actor_tracks),
            mock.patch("nurec.grpc.serve.CuboidTracks.Ops.transform_with_frame_conversion", return_value=cuboid_track),
        ):
            backend.insert_assets([dynamic_object])

        renderable_model.insert_asset.assert_called_once()
        passed_offset = renderable_model.insert_asset.call_args.args[2]
        torch.testing.assert_close(passed_offset, torch.zeros(3))


class TestBackendEditRollback(unittest.TestCase):
    @staticmethod
    def _make_backend() -> tuple[Backend, mock.MagicMock]:
        model = mock.MagicMock()
        model.supports_edit_actors.return_value = True
        backend = Backend(
            renderable_model=model,
            camera_bank=mock.MagicMock(),
            world_to_nurec=mock.MagicMock(),
            scene_time_range_us=HalfClosedInterval(0, 1),
            lidar_bank=mock.MagicMock(),
            asset_bank=mock.MagicMock(),
            source_artifact=mock.MagicMock(),
            backend_cache_key=_test_backend_cache_key(),
        )
        return backend, model

    def test_fatal_edit_error_bypasses_training_parameter_rollback(self):
        for operation in ("replace_assets", "insert_assets"):
            with self.subTest(operation=operation):
                backend, model = self._make_backend()
                failure = RuntimeError("CUDA error: an illegal memory access was encountered")
                actions = {"replace_assets": [], "insert_assets": []}
                actions[operation] = [mock.MagicMock()]
                with mock.patch.object(backend, operation, side_effect=failure):
                    with self.assertRaises(RuntimeError) as raised:
                        backend.edit_assets(**actions)

                self.assertIs(raised.exception, failure)
                model.save_training_parameters.assert_called_once()
                model.restore_training_parameters.assert_not_called()

    def test_recoverable_edit_error_restores_training_parameters(self):
        for operation in ("replace_assets", "insert_assets"):
            with self.subTest(operation=operation):
                backend, model = self._make_backend()
                failure = RuntimeError("asset edit failed")
                actions = {"replace_assets": [], "insert_assets": []}
                actions[operation] = [mock.MagicMock()]
                with mock.patch.object(backend, operation, side_effect=failure):
                    with self.assertRaises(RuntimeError) as raised:
                        backend.edit_assets(**actions)

                self.assertIs(raised.exception, failure)
                model.save_training_parameters.assert_called_once()
                model.restore_training_parameters.assert_called_once()


class TestRenderGrpcStatusCodes(unittest.TestCase):
    def _start_service(
        self,
        cache_size=1,
        max_backends_per_scene=None,
        max_workers=1,
        server_config: GrpcServerConfig | None = None,
        fatal_worker_interceptor: FatalWorkerExceptionInterceptor | None = None,
    ):
        interceptors = [] if fatal_worker_interceptor is None else [fatal_worker_interceptor]
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), interceptors=interceptors)
        server_config = server_config or GrpcServerConfig()
        service = SensorSimService(
            server,
            artifacts_glob=None,
            ray_chunk_size=1,
            egocar_hoods_dir=None,
            downloader=None,
            scene_cache=None,
            cache_size=cache_size,
            max_backends_per_scene=max_backends_per_scene,
            metrics_output_dir=None,
            enable_editing_actors=False,
            server_config=server_config,
            fatal_worker_interceptor=fatal_worker_interceptor,
        )
        if fatal_worker_interceptor is not None:
            fatal_worker_interceptor.configure_shutdown(
                mark_unhealthy=service.backend_cache.wake_waiters,
                stop_server=server.stop,
            )
        add_SensorsimServiceServicer_to_server(service, server)
        port = server.add_insecure_port("localhost:0")
        server.start()

        channel = grpc.insecure_channel(f"localhost:{port}")
        grpc.channel_ready_future(channel).result(timeout=5)
        stub = SensorsimServiceStub(channel)
        return server, channel, stub, service

    @staticmethod
    def _make_backend(backend_cache_key: BackendCacheKey | None = None):
        return Backend(
            renderable_model=mock.MagicMock(),
            camera_bank=mock.MagicMock(),
            world_to_nurec=mock.MagicMock(),
            scene_time_range_us=HalfClosedInterval(0, 1),
            lidar_bank=mock.MagicMock(),
            asset_bank=mock.MagicMock(),
            source_artifact=mock.MagicMock(),
            backend_cache_key=backend_cache_key or _test_backend_cache_key(),
        )

    @contextlib.contextmanager
    def _blocking_rgb_service(self):
        server, channel, stub, service = self._start_service(
            cache_size=1,
            max_backends_per_scene=1,
            max_workers=3,
        )
        scene_key = SceneCacheKey(scene_id="scene")
        registration = ArtifactRegistration(
            _registered_artifact(Path("/scenes/scene.usdz"), "scene"),
            False,
            scene_key,
        )
        backend = self._make_backend(registration.backend_cache_key)
        rgb_acquired = threading.Event()
        release_rgb = threading.Event()

        def render_camera(_request, _enable_editing_actors):
            rgb_acquired.set()
            if not release_rgb.wait(timeout=5):
                raise TimeoutError("Timed out waiting to release RGB")
            return torch.zeros((1, 1, 3)), (1, 1)

        backend.render_rgb_request = mock.MagicMock(side_effect=render_camera)
        backend.render_lidar_request = mock.MagicMock(return_value=LidarRenderReturn())
        service.encode_image = mock.MagicMock(return_value=RGBRenderReturn(image_bytes=b"rgb"))
        with service._artifact_registry_lock:
            service._artifact_registrations[scene_key.key_hash] = registration
        service.backend_cache.put(registration.backend_cache_key.key_hash, backend)
        service.backend_cache.checkin(backend)
        rgb_call = stub.render_rgb.future(RGBRenderRequest(scene_id="scene"))
        self.assertTrue(rgb_acquired.wait(timeout=5))
        try:
            yield stub, service, backend, rgb_call, release_rgb
        finally:
            release_rgb.set()
            try:
                rgb_call.result(timeout=5)
            except grpc.RpcError:
                pass
            channel.close()
            server.stop(0)

    def test_get_loaded_scenes_returns_cache_snapshot(self):
        """get_loaded_scenes reports the cache capacity and each scene's loaded and reusable instance counts.

        get_version reports the current API version.
        """
        server, channel, stub, service = self._start_service(cache_size=3)
        try:
            in_use_a = self._make_backend(_test_backend_cache_key("scene-a"))
            spare_a = self._make_backend(_test_backend_cache_key("scene-a"))
            spare_b = self._make_backend(_test_backend_cache_key("scene-b"))

            service.backend_cache.put("scene-a", in_use_a)
            service.backend_cache.put("scene-a", spare_a)
            service.backend_cache.checkin(spare_a)
            service.backend_cache.put("scene-b", spare_b)
            service.backend_cache.checkin(spare_b)

            response = stub.get_loaded_scenes(Empty())

            self.assertEqual(response.loaded_instance_capacity, 3)
            self.assertEqual(
                [
                    (entry.scene_id, entry.loaded_instance_count, entry.reusable_instance_count)
                    for entry in response.scenes
                ],
                [("scene-a", 2, 1), ("scene-b", 1, 1)],
            )
            api_version = stub.get_version(Empty()).grpc_api_version
            self.assertEqual((api_version.major, api_version.minor, api_version.patch), (0, 1, 0))
        finally:
            channel.close()
            server.stop(0)

    def test_disallowed_image_formats_are_rejected_before_rendering(self):
        server, channel, stub, _ = self._start_service(server_config=GrpcServerConfig(allowed_image_formats="JPEG"))
        disallowed_format = ImageFormat.RGB_UINT8_PLANAR
        calls = {
            "render_rgb": lambda: stub.render_rgb(RGBRenderRequest(scene_id="missing", image_format=disallowed_format)),
            "render_camera": lambda: stub.render_camera(
                CameraRenderRequest(scene_id="missing", image_format=disallowed_format)
            ),
            "batch_render_rgb": lambda: stub.batch_render_rgb(
                BatchRGBRenderRequest(
                    items=[
                        BatchRGBRenderRequestItem(
                            camera_name="front",
                            request=RGBRenderRequest(scene_id="missing", image_format=disallowed_format),
                        )
                    ]
                )
            ),
            "batch_render_camera": lambda: stub.batch_render_camera(
                BatchCameraRenderRequest(
                    items=[
                        BatchCameraRenderRequestItem(
                            camera_name="front",
                            request=CameraRenderRequest(scene_id="missing", image_format=disallowed_format),
                        )
                    ]
                )
            ),
        }
        try:
            for rpc_name, call in calls.items():
                with self.subTest(rpc=rpc_name), self.assertRaises(grpc.RpcError) as ctx:
                    call()
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
                details = ctx.exception.details() or ""
                self.assertIn("RGB_UINT8_PLANAR", details)
                self.assertIn("Allowed image formats: JPEG", details)
        finally:
            channel.close()
            server.stop(0)

    def test_allowed_image_format_reaches_normal_request_handling(self):
        server, channel, stub, _ = self._start_service(server_config=GrpcServerConfig(allowed_image_formats="JPEG"))
        try:
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.render_camera(CameraRenderRequest(scene_id="missing", image_format=ImageFormat.JPEG))
            self.assertEqual(ctx.exception.code(), grpc.StatusCode.NOT_FOUND)
        finally:
            channel.close()
            server.stop(0)

    def test_unknown_firing_pattern_id_returns_not_found(self):
        server, channel, stub, service = self._start_service()
        backend = mock.MagicMock()
        backend.get_lidar_firing_pattern.side_effect = KeyError("Unknown firing pattern id 'missing'")

        @contextlib.contextmanager
        def backend_ctx(*args, **kwargs):
            yield backend

        try:
            with (
                mock.patch.object(
                    service,
                    "_request_scene_cache_key",
                    return_value=SceneCacheKey(scene_id="scene"),
                ),
                mock.patch.object(service, "get_backend", backend_ctx),
                self.assertRaises(grpc.RpcError) as ctx,
            ):
                stub.get_lidar_firing_pattern(
                    LidarFiringPatternRequest(
                        scene_id="scene",
                        firing_pattern_id="missing",
                    )
                )

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.NOT_FOUND)
            self.assertIn("Unknown firing pattern id 'missing'", ctx.exception.details())
        finally:
            channel.close()
            server.stop(0)

    def test_lidar_wire_round_trip_and_capability_statuses(self):
        server, channel, stub, service = self._start_service()
        fixture = TestBackendLidarDispatch()
        backend = fixture._backend(fixture._mixed_artifact())

        @contextlib.contextmanager
        def backend_ctx(*args, **kwargs):
            yield backend

        try:
            with (
                mock.patch.object(service, "_request_scene_cache_key", return_value=SceneCacheKey(scene_id="scene")),
                mock.patch.object(service, "get_backend", backend_ctx),
                mock.patch.object(LidarRayBundle, "build_model_free", return_value=mock.sentinel.bundle),
            ):
                request = fixture._request("model_free")
                request.scene_id = "scene"
                request.include_firing_pattern = True
                response = stub.render_lidar(request)
                self.assertEqual(response.num_rays, 7)
                self.assertEqual(list(response.return_types), [LidarReturnType.LIDAR_RETURN_STRONGEST])
                self.assertEqual(response.firing_pattern_id, response.firing_pattern.firing_pattern_id)
                np.testing.assert_array_equal(np.frombuffer(response.ray_distances_m_buffer, dtype="<f4"), np.ones(7))
                np.testing.assert_array_equal(
                    np.frombuffer(response.firing_pattern.timestamp_offsets_us, dtype="<i8"), np.arange(7)
                )
                for values in ([0], [2, 0], [99]):
                    with self.subTest(return_types=values):
                        invalid = fixture._request("model_free")
                        invalid.scene_id = "scene"
                        invalid.return_types.extend(values)
                        backend.renderable_model.render_lidar_ray_frame_from_ray_bundle.reset_mock()
                        with self.assertRaises(grpc.RpcError) as caught:
                            stub.render_lidar(invalid)
                        self.assertEqual(caught.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
                        backend.renderable_model.render_lidar_ray_frame_from_ray_bundle.assert_not_called()
                for rpc, req, expected in (
                    (stub.render_lidar, fixture._request("modeled"), grpc.StatusCode.FAILED_PRECONDITION),
                    (
                        stub.get_lidar_firing_pattern,
                        LidarFiringPatternRequest(lidar_id="modeled"),
                        grpc.StatusCode.FAILED_PRECONDITION,
                    ),
                    (stub.render_lidar, fixture._request("missing"), grpc.StatusCode.NOT_FOUND),
                    (
                        stub.get_lidar_firing_pattern,
                        LidarFiringPatternRequest(lidar_id="missing"),
                        grpc.StatusCode.NOT_FOUND,
                    ),
                ):
                    req.scene_id = "scene"
                    with self.subTest(lidar=req.lidar_id, rpc=rpc), self.assertRaises(grpc.RpcError) as caught:
                        rpc(req)
                    self.assertEqual(caught.exception.code(), expected)
        finally:
            channel.close()
            server.stop(0)

    def test_spare_hit_uses_queued_acquisition_without_loading_artifact(self):
        server, channel, _, service = self._start_service(cache_size=1, max_backends_per_scene=1)
        scene_key = SceneCacheKey(scene_id="scene")
        registration = ArtifactRegistration(
            _registered_artifact(Path("/scenes/scene.usdz"), "scene"),
            False,
            scene_key,
        )
        backend = self._make_backend(registration.backend_cache_key)
        with service._artifact_registry_lock:
            service._artifact_registrations[scene_key.key_hash] = registration
        service.backend_cache.put(registration.backend_cache_key.key_hash, backend)
        service.backend_cache.checkin(backend)
        service._scene_prefetch = mock.MagicMock()
        original_acquire = service.backend_cache.checkout_or_reserve

        try:
            with (
                mock.patch.object(
                    service.backend_cache, "checkout", side_effect=AssertionError("direct checkout used")
                ),
                mock.patch.object(
                    service.backend_cache,
                    "checkout_or_reserve",
                    wraps=original_acquire,
                ) as queued_acquire,
                mock.patch.object(Backend, "from_artifact") as from_artifact,
            ):
                with service.get_backend(SceneCacheKey(scene_id="scene")) as acquired:
                    self.assertIs(acquired, backend)

            queued_acquire.assert_called_once()
            from_artifact.assert_not_called()
            service._scene_prefetch.discard.assert_called_once_with(scene_key)
            service._scene_prefetch.take.assert_not_called()
        finally:
            channel.close()
            server.stop(0)

    def test_prefetched_backend_activates_without_synchronous_reload(self):
        server, channel, _, service = self._start_service(cache_size=1, max_backends_per_scene=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "scene.usdz"
            source.write_bytes(b"scene")
            scene_key = SceneCacheKey.from_url("scene", source.as_uri())
            fingerprint = _local_file_fingerprint(source)
            registration = ArtifactRegistration(
                _registered_artifact(source, "scene"),
                False,
                scene_key,
                source_fingerprint=fingerprint,
            )
            prepared = mock.MagicMock(spec=PreparedBackend)
            prepared.backend_cache_key = registration.backend_cache_key
            backend = self._make_backend(registration.backend_cache_key)
            prepared.activate.return_value = backend
            service._scene_prefetch = mock.MagicMock()
            service._scene_prefetch.take.return_value = PreparedScene(
                registration,
                prepared,
            )
            with service._artifact_registry_lock:
                service._artifact_registrations[scene_key.key_hash] = registration

            try:
                with mock.patch.object(Backend, "from_artifact") as from_artifact:
                    with service.get_backend(SceneCacheKey(scene_id="scene")) as acquired:
                        self.assertIs(acquired, backend)

                prepared.activate.assert_called_once_with()
                from_artifact.assert_not_called()
                service._scene_prefetch.take.assert_called_once_with(scene_key, check_wait=mock.ANY)
            finally:
                channel.close()
                server.stop(0)

    def test_prefetch_activation_failure_loads_synchronously(self):
        server, channel, _, service = self._start_service(cache_size=1, max_backends_per_scene=1)
        scene_key = SceneCacheKey(scene_id="scene")
        source = Path("/scenes/scene.usdz")
        registration = ArtifactRegistration(
            _registered_artifact(source, "scene"),
            False,
            scene_key,
            source_fingerprint=(1, 2, 3, 4, 5),
        )
        prepared = mock.MagicMock(spec=PreparedBackend)
        prepared.backend_cache_key = registration.backend_cache_key
        prepared.activate.side_effect = RuntimeError("activation failed")
        service._scene_prefetch = mock.MagicMock()
        service._scene_prefetch.take.return_value = PreparedScene(
            registration,
            prepared,
        )
        with service._artifact_registry_lock:
            service._artifact_registrations[scene_key.key_hash] = registration
        loaded_backend = self._make_backend(registration.backend_cache_key)
        artifact = mock.MagicMock(sha256_footprint=registration.backend_cache_key.content_sha256)
        artifact_context = mock.MagicMock()
        artifact_context.temporary_cache.return_value.__enter__.return_value = artifact

        try:
            with (
                mock.patch("nurec.grpc.serve._local_file_fingerprint", return_value=(1, 2, 3, 4, 5)),
                mock.patch("nurec.grpc.serve.Artifact", return_value=artifact_context),
                mock.patch.object(Backend, "from_artifact", return_value=loaded_backend) as load,
                mock.patch("nurec.grpc.serve.torch.cuda.empty_cache"),
            ):
                with service.get_backend(scene_key) as acquired:
                    self.assertIs(acquired, loaded_backend)

            prepared.activate.assert_called_once_with()
            load.assert_called_once()
        finally:
            channel.close()
            server.stop(0)

    def test_terminal_prefetch_activation_errors_do_not_reload(self):
        def check(activation_error: Exception) -> None:
            server, channel, _, service = self._start_service(cache_size=1, max_backends_per_scene=1)
            with tempfile.TemporaryDirectory() as temp_dir:
                source = Path(temp_dir) / "scene.usdz"
                source.write_bytes(b"scene")
                scene_key = SceneCacheKey.from_url("scene", source.as_uri())
                fingerprint = _local_file_fingerprint(source)
                registration = ArtifactRegistration(
                    _registered_artifact(source, "scene"),
                    False,
                    scene_key,
                    source_fingerprint=fingerprint,
                )
                prepared = mock.MagicMock(spec=PreparedBackend)
                prepared.backend_cache_key = registration.backend_cache_key
                prepared.activate.side_effect = activation_error
                service._scene_prefetch = mock.MagicMock()
                service._scene_prefetch.take.return_value = PreparedScene(
                    registration,
                    prepared,
                )
                with service._artifact_registry_lock:
                    service._artifact_registrations[scene_key.key_hash] = registration

                def run_factory(*_args, **kwargs):
                    return kwargs["backend_factory"]()

                try:
                    with (
                        mock.patch.object(service.backend_cache, "put_with_retries", side_effect=run_factory),
                        mock.patch.object(Backend, "from_artifact") as load,
                        mock.patch("nurec.grpc.serve.torch.cuda.empty_cache"),
                        self.assertRaises(type(activation_error)),
                    ):
                        with service.get_backend(scene_key):
                            pass

                    prepared.activate.assert_called_once_with()
                    load.assert_not_called()
                finally:
                    channel.close()
                    server.stop(0)

        for activation_error in (
            torch.cuda.OutOfMemoryError("activation OOM"),
            ModelIncompatibilityError("model mismatch"),
        ):
            with self.subTest(error=type(activation_error).__name__):
                check(activation_error)

    def test_fatal_prefetch_activation_error_propagates_without_cleanup(self):
        server, channel, _, service = self._start_service(cache_size=1, max_backends_per_scene=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "scene.usdz"
            source.write_bytes(b"scene")
            scene_key = SceneCacheKey.from_url("scene", source.as_uri())
            fingerprint = _local_file_fingerprint(source)
            registration = ArtifactRegistration(
                _registered_artifact(source, "scene"),
                False,
                scene_key,
                source_fingerprint=fingerprint,
            )
            prepared = mock.MagicMock(spec=PreparedBackend)
            prepared.backend_cache_key = registration.backend_cache_key
            prepared.activate.side_effect = RuntimeError("CUDA error: an illegal memory access was encountered")
            service._scene_prefetch = mock.MagicMock()
            service._scene_prefetch.take.return_value = PreparedScene(registration, prepared)
            with service._artifact_registry_lock:
                service._artifact_registrations[scene_key.key_hash] = registration

            try:
                with (
                    mock.patch.object(Backend, "from_artifact") as load,
                    mock.patch("nurec.grpc.serve._cleanup_failed_backend") as cleanup,
                    self.assertRaisesRegex(RuntimeError, "illegal memory access"),
                ):
                    with service.get_backend(scene_key):
                        pass

                prepared.activate.assert_called_once_with()
                load.assert_not_called()
                cleanup.assert_not_called()
                self.assertEqual(service.backend_cache.pending_global, 0)
            finally:
                channel.close()
                server.stop(0)

    def test_missing_scene_wins_over_saturated_cache_with_unset_limit(self):
        server, channel, _, service = self._start_service(cache_size=1)
        service.backend_cache.put("loaded", self._make_backend(_test_backend_cache_key("loaded")))
        try:
            with self.assertRaises(SceneNotFoundError):
                with service.get_backend(SceneCacheKey(scene_id="missing")):
                    pass
            self.assertEqual(service.backend_cache.pending_global, 0)
        finally:
            channel.close()
            server.stop(0)

    def test_missing_scene_wins_over_saturated_cache_with_configured_limit(self):
        server, channel, _, service = self._start_service(cache_size=1, max_backends_per_scene=1)
        service.backend_cache.put("loaded", self._make_backend(_test_backend_cache_key("loaded")))
        try:
            with self.assertRaises(SceneNotFoundError):
                with service.get_backend(SceneCacheKey(scene_id="missing")):
                    pass
            self.assertEqual(service.backend_cache.pending_global, 0)
        finally:
            channel.close()
            server.stop(0)

    def test_cancelled_waiter_does_not_clear_active_artifact_state(self):
        server, channel, _, service = self._start_service(cache_size=1, max_backends_per_scene=1)
        source = Path("/scenes/scene.usdz")
        scene_key = SceneCacheKey(scene_id="scene")
        with service._artifact_registry_lock:
            service._artifact_registrations[scene_key.key_hash] = ArtifactRegistration(
                _registered_artifact(source, "scene"),
                False,
                scene_key,
            )

        constructor_started = threading.Event()
        finish_constructor = threading.Event()
        wait_started = threading.Event()
        cancelled = threading.Event()
        constructed_artifacts = []
        backend = self._make_backend(
            BackendCacheKey(
                scene_key,
                hashlib.sha256(str(source).encode()).hexdigest(),
            )
        )
        errors = []
        original_wait = service.backend_cache._wait_for_capacity

        def construct(artifact, _renderer, _stride, backend_cache_key, **kwargs):
            self.assertFalse(kwargs["camera_only"])
            self.assertEqual(backend_cache_key, backend.backend_cache_key)
            artifact._checkpoint = io.BytesIO(b"checkpoint")
            constructed_artifacts.append(artifact)
            constructor_started.set()
            self.assertTrue(finish_constructor.wait(timeout=5))
            artifact._checkpoint.seek(0)
            return backend

        def record_wait(*args, **kwargs):
            wait_started.set()
            return original_wait(*args, **kwargs)

        def first_request():
            with service.get_backend(SceneCacheKey(scene_id="scene")):
                pass

        context = mock.MagicMock()
        context.is_active.side_effect = lambda: not cancelled.is_set()
        context.time_remaining.return_value = None

        def waiting_request():
            try:
                with service.get_backend(SceneCacheKey(scene_id="scene"), context):
                    pass
            except Exception as exc:
                errors.append(exc)

        try:
            with (
                mock.patch.object(
                    Artifact,
                    "sha256_footprint",
                    new_callable=mock.PropertyMock,
                    return_value=hashlib.sha256(str(source).encode()).hexdigest(),
                ),
                mock.patch.object(Backend, "from_artifact", side_effect=construct) as from_artifact,
                mock.patch("nurec.grpc.serve._local_file_fingerprint", return_value=(1, 2, 3, 4, 5)),
                mock.patch.object(service.backend_cache, "_wait_for_capacity", side_effect=record_wait),
                mock.patch("nurec.grpc.serve.BACKEND_CACHE_WAIT_POLL_S", 0.01),
                futures.ThreadPoolExecutor(max_workers=2) as executor,
            ):
                first = executor.submit(first_request)
                self.assertTrue(constructor_started.wait(timeout=5))
                waiter = executor.submit(waiting_request)
                self.assertTrue(wait_started.wait(timeout=5))
                cancelled.set()
                waiter.result(timeout=5)

                self.assertEqual(len(constructed_artifacts), 1)
                self.assertIsNotNone(constructed_artifacts[0]._checkpoint)
                finish_constructor.set()
                first.result(timeout=5)

            from_artifact.assert_called_once()
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], CacheWaitAbortedError)
            self.assertEqual(errors[0].reason, CacheWaitAbortedReason.CANCELLED)
            self.assertEqual(service.backend_cache.pending_global, 0)
            self.assertIsNone(constructed_artifacts[0]._checkpoint)
        finally:
            finish_constructor.set()
            channel.close()
            server.stop(0)

    def test_concurrent_same_scene_constructions_use_independent_artifacts(self):
        server, channel, _, service = self._start_service(cache_size=2, max_backends_per_scene=2)
        source = Path("/scenes/scene.usdz")
        scene_key = SceneCacheKey(scene_id="scene")
        with service._artifact_registry_lock:
            service._artifact_registrations[scene_key.key_hash] = ArtifactRegistration(
                _registered_artifact(source, "scene"),
                False,
                scene_key,
            )

        constructors_ready = threading.Barrier(2)
        constructed_artifacts = []
        constructed_artifacts_lock = threading.Lock()

        def construct(artifact, _renderer, _stride, backend_cache_key, **kwargs):
            self.assertFalse(kwargs["camera_only"])
            with constructed_artifacts_lock:
                constructed_artifacts.append(artifact)
            constructors_ready.wait(timeout=5)
            return self._make_backend(backend_cache_key)

        def acquire_backend():
            with service.get_backend(SceneCacheKey(scene_id="scene")):
                pass

        try:
            with (
                mock.patch.object(
                    Artifact,
                    "sha256_footprint",
                    new_callable=mock.PropertyMock,
                    return_value=hashlib.sha256(str(source).encode()).hexdigest(),
                ),
                mock.patch.object(Backend, "from_artifact", side_effect=construct),
                mock.patch("nurec.grpc.serve._local_file_fingerprint", return_value=(1, 2, 3, 4, 5)),
                futures.ThreadPoolExecutor(max_workers=2) as executor,
            ):
                calls = [executor.submit(acquire_backend) for _ in range(2)]
                for call in calls:
                    call.result(timeout=5)

            self.assertEqual(len(constructed_artifacts), 2)
            self.assertIsNot(constructed_artifacts[0], constructed_artifacts[1])
            self.assertEqual([artifact.source for artifact in constructed_artifacts], [source, source])
            self.assertEqual(service.backend_cache.pending_global, 0)
        finally:
            channel.close()
            server.stop(0)

    def test_render_oom_evicts_same_scene_duplicate_and_other_spares(self):
        server, channel, _, service = self._start_service(cache_size=3, max_backends_per_scene=2)
        scene_a_source_key = SceneCacheKey(scene_id="scene-a")
        scene_b_source_key = SceneCacheKey(scene_id="scene-b")
        registration_a = ArtifactRegistration(
            _registered_artifact(Path("/scenes/scene-a.usdz"), "scene-a"),
            False,
            scene_a_source_key,
        )
        registration_b = ArtifactRegistration(
            _registered_artifact(Path("/scenes/scene-b.usdz"), "scene-b"),
            False,
            scene_b_source_key,
        )
        with service._artifact_registry_lock:
            service._artifact_registrations[scene_a_source_key.key_hash] = registration_a
            service._artifact_registrations[scene_b_source_key.key_hash] = registration_b
        scene_a_key = registration_a.backend_cache_key.key_hash
        scene_b_key = registration_b.backend_cache_key.key_hash
        active = self._make_backend(registration_a.backend_cache_key)
        duplicate = self._make_backend(registration_a.backend_cache_key)
        other = self._make_backend(registration_b.backend_cache_key)
        service.backend_cache.put(scene_a_key, active)
        service.backend_cache.checkin(active)
        service.backend_cache.put(scene_a_key, duplicate)
        service.backend_cache.checkin(duplicate)
        service.backend_cache.put(scene_b_key, other)
        service.backend_cache.checkin(other)

        try:
            with mock.patch("nurec.grpc.serve.torch.cuda.is_available", return_value=False):
                with self.assertRaises(torch.cuda.OutOfMemoryError):
                    with service.get_backend(SceneCacheKey(scene_id="scene-a")) as acquired:
                        self.assertIs(acquired, active)
                        raise torch.cuda.OutOfMemoryError("render failed")

            self.assertEqual(service.backend_cache.in_use, set())
            self.assertEqual(service.backend_cache.spares, [active])
        finally:
            channel.close()
            server.stop(0)

    def test_in_process_lidar_waits_for_rgb_and_reuses_backend(self):
        with self._blocking_rgb_service() as (stub, service, backend, rgb_call, release_rgb):
            wait_started = threading.Event()
            original_wait = service.backend_cache._wait_for_capacity

            def record_wait(*args, **kwargs):
                wait_started.set()
                return original_wait(*args, **kwargs)

            with (
                mock.patch.object(service.backend_cache, "_wait_for_capacity", side_effect=record_wait),
                mock.patch.object(Backend, "from_artifact", side_effect=AssertionError("factory called")),
            ):
                lidar_call = stub.render_lidar.future(LidarRenderRequest(scene_id="scene"))
                self.assertTrue(wait_started.wait(timeout=5))
                release_rgb.set()
                rgb_call.result(timeout=5)
                lidar_call.result(timeout=5)

            backend.render_lidar_request.assert_called_once()
            self.assertEqual(service.backend_cache.pending_global, 0)
            self.assertEqual(service.backend_cache.spares, [backend])

    def test_in_process_lidar_deadline_unwinds_and_subsequent_request_succeeds(self):
        with self._blocking_rgb_service() as (stub, service, backend, rgb_call, release_rgb):
            wait_started = threading.Event()
            handler_unwound = threading.Event()
            original_wait = service.backend_cache._wait_for_capacity
            original_put = service.backend_cache.put_with_retries

            def record_wait(*args, **kwargs):
                wait_started.set()
                return original_wait(*args, **kwargs)

            def record_unwind(*args, **kwargs):
                try:
                    return original_put(*args, **kwargs)
                finally:
                    handler_unwound.set()

            with (
                mock.patch.object(service.backend_cache, "_wait_for_capacity", side_effect=record_wait),
                mock.patch.object(service.backend_cache, "put_with_retries", side_effect=record_unwind),
                mock.patch("nurec.grpc.serve.BACKEND_CACHE_WAIT_POLL_S", 0.01),
            ):
                with self.assertRaises(grpc.RpcError) as error:
                    stub.render_lidar(LidarRenderRequest(scene_id="scene"), timeout=0.1)
                self.assertEqual(error.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)
                self.assertTrue(wait_started.is_set())
                self.assertTrue(handler_unwound.wait(timeout=5))
                self.assertEqual(service.backend_cache.pending_global, 0)

                release_rgb.set()
                rgb_call.result(timeout=5)
                stub.render_lidar(LidarRenderRequest(scene_id="scene"), timeout=5)

            self.assertEqual(service.backend_cache.spares, [backend])

    def test_in_process_lidar_cancellation_unwinds_and_subsequent_request_succeeds(self):
        with self._blocking_rgb_service() as (stub, service, backend, rgb_call, release_rgb):
            wait_started = threading.Event()
            handler_unwound = threading.Event()
            original_wait = service.backend_cache._wait_for_capacity
            original_put = service.backend_cache.put_with_retries

            def record_wait(*args, **kwargs):
                wait_started.set()
                return original_wait(*args, **kwargs)

            def record_unwind(*args, **kwargs):
                try:
                    return original_put(*args, **kwargs)
                finally:
                    handler_unwound.set()

            with (
                mock.patch.object(service.backend_cache, "_wait_for_capacity", side_effect=record_wait),
                mock.patch.object(service.backend_cache, "put_with_retries", side_effect=record_unwind),
                mock.patch("nurec.grpc.serve.BACKEND_CACHE_WAIT_POLL_S", 0.01),
            ):
                lidar_call = stub.render_lidar.future(LidarRenderRequest(scene_id="scene"))
                self.assertTrue(wait_started.wait(timeout=5))
                self.assertTrue(lidar_call.cancel())
                with self.assertRaises(grpc.FutureCancelledError):
                    lidar_call.result(timeout=5)
                self.assertTrue(handler_unwound.wait(timeout=5))
                self.assertEqual(service.backend_cache.pending_global, 0)

                release_rgb.set()
                rgb_call.result(timeout=5)
                stub.render_lidar(LidarRenderRequest(scene_id="scene"), timeout=5)

            self.assertEqual(service.backend_cache.spares, [backend])

    def _backend_acquiring_rpc_cases(self):
        return [
            ("render_rgb", lambda stub: stub.render_rgb(RGBRenderRequest(scene_id="scene"))),
            (
                "batch_render_rgb",
                lambda stub: stub.batch_render_rgb(
                    BatchRGBRenderRequest(
                        items=[
                            BatchRGBRenderRequestItem(
                                camera_name="cam",
                                request=RGBRenderRequest(scene_id="scene"),
                            )
                        ]
                    )
                ),
            ),
            ("render_lidar", lambda stub: stub.render_lidar(LidarRenderRequest(scene_id="scene"))),
            (
                "get_lidar_firing_pattern",
                lambda stub: stub.get_lidar_firing_pattern(LidarFiringPatternRequest(scene_id="scene")),
            ),
            (
                "get_available_cameras",
                lambda stub: stub.get_available_cameras(AvailableCamerasRequest(scene_id="scene")),
            ),
            (
                "get_available_trajectories",
                lambda stub: stub.get_available_trajectories(AvailableTrajectoriesRequest(scene_id="scene")),
            ),
            (
                "get_external_asset_objects",
                lambda stub: stub.get_external_asset_objects(ExternalAssetObjectsRequest(scene_id="scene")),
            ),
            (
                "get_dynamic_objects",
                lambda stub: stub.get_dynamic_objects(AvailableDynamicObjectsRequest(scene_id="scene")),
            ),
            ("edit_assets", lambda stub: stub.edit_assets(EditAssetsRequest(scene_id="scene"))),
            (
                "restore_model_parameters",
                lambda stub: stub.restore_model_parameters(RestoreModelParametersRequest(scene_id="scene")),
            ),
        ]

    def _assert_get_backend_error(self, call_rpc, exc, expected_status, expected_details):
        @contextlib.contextmanager
        def _raise_backend(*args, **kwargs):
            raise exc
            yield  # pragma: no cover - unreachable

        server, channel, stub, service = self._start_service()
        try:
            with mock.patch.object(service, "get_backend", _raise_backend):
                with self.assertRaises(grpc.RpcError) as ctx:
                    call_rpc(stub)
                self.assertEqual(ctx.exception.code(), expected_status)
                details = ctx.exception.details() or ""
                self.assertIn(expected_details, details)
        finally:
            channel.close()
            server.stop(0)

    def test_backend_capacity_errors_match_endpoint_status_contracts(self):
        expected_statuses = {
            "render_rgb": grpc.StatusCode.RESOURCE_EXHAUSTED,
            "batch_render_rgb": grpc.StatusCode.RESOURCE_EXHAUSTED,
            "render_lidar": grpc.StatusCode.RESOURCE_EXHAUSTED,
            "get_lidar_firing_pattern": grpc.StatusCode.RESOURCE_EXHAUSTED,
            "get_available_cameras": grpc.StatusCode.FAILED_PRECONDITION,
            "get_available_trajectories": grpc.StatusCode.FAILED_PRECONDITION,
            "get_external_asset_objects": grpc.StatusCode.UNKNOWN,
            "get_dynamic_objects": grpc.StatusCode.FAILED_PRECONDITION,
            "restore_model_parameters": grpc.StatusCode.UNKNOWN,
        }
        error_cases = [
            ("cache_full", lambda: CacheFullError("cache is full")),
            ("no_spare_backends", lambda: NoSpareBackendsError("no spare backends")),
        ]
        for rpc_name, call_rpc in self._backend_acquiring_rpc_cases():
            for error_name, error_factory in error_cases:
                with self.subTest(rpc=rpc_name, error=error_name):
                    if rpc_name == "edit_assets":

                        @contextlib.contextmanager
                        def _raise_backend(*args, **kwargs):
                            raise error_factory()
                            yield  # pragma: no cover - unreachable

                        server, channel, stub, service = self._start_service()
                        try:
                            with mock.patch.object(service, "get_backend", _raise_backend):
                                response = call_rpc(stub)
                            self.assertFalse(response.success)
                            self.assertIn(error_factory().args[0], response.message)
                        finally:
                            channel.close()
                            server.stop(0)
                    else:
                        self._assert_get_backend_error(
                            call_rpc,
                            error_factory(),
                            expected_statuses[rpc_name],
                            error_factory().args[0],
                        )

    def test_backend_wait_aborts_map_consistently_for_all_endpoints(self):
        abort_cases = [
            (CacheWaitAbortedReason.CANCELLED, grpc.StatusCode.CANCELLED),
            (CacheWaitAbortedReason.DEADLINE_EXCEEDED, grpc.StatusCode.DEADLINE_EXCEEDED),
        ]
        for rpc_name, call_rpc in self._backend_acquiring_rpc_cases():
            for reason, expected_status in abort_cases:
                with self.subTest(rpc=rpc_name, reason=reason):
                    self._assert_get_backend_error(
                        call_rpc,
                        CacheWaitAbortedError("wait aborted", reason),
                        expected_status,
                        "wait aborted",
                    )

    def test_unknown_scene_returns_not_found_for_backend_acquiring_rpcs(self):
        for rpc_name, call_rpc in self._backend_acquiring_rpc_cases():
            with self.subTest(rpc=rpc_name):
                self._assert_get_backend_error(
                    call_rpc,
                    SceneNotFoundError("Scene artifact_id='missing' not available."),
                    grpc.StatusCode.NOT_FOUND,
                    "Scene artifact_id='missing' not available.",
                )

    def test_edit_assets_preserves_error_payload_for_backend_edit_failure(self):
        backend = mock.MagicMock()
        backend.edit_assets.side_effect = RuntimeError("edit failed")

        @contextlib.contextmanager
        def _return_backend(*args, **kwargs):
            yield backend

        server, channel, stub, service = self._start_service()
        try:
            with mock.patch.object(service, "get_backend", _return_backend):
                response = stub.edit_assets(EditAssetsRequest(scene_id="scene"))
            self.assertFalse(response.success)
            self.assertIn("Failed to edit assets: edit failed", response.message)
        finally:
            channel.close()
            server.stop(0)

    def test_edit_assets_does_not_swallow_fatal_worker_error(self):
        backend = mock.MagicMock()
        backend.edit_assets.side_effect = RuntimeError("CUDA error: an illegal memory access was encountered")

        @contextlib.contextmanager
        def _return_backend(*args, **kwargs):
            yield backend

        server, channel, _, service = self._start_service()
        try:
            with (
                mock.patch.object(service, "get_backend", _return_backend),
                self.assertRaisesRegex(RuntimeError, "illegal memory access"),
            ):
                service.edit_assets(EditAssetsRequest(scene_id="scene"), mock.MagicMock())
        finally:
            channel.close()
            server.stop(0)

    def test_get_available_cameras_fatal_worker_error_exits_through_server_interceptor(self):
        terminate_worker = mock.Mock()
        interceptor = FatalWorkerExceptionInterceptor(terminate_worker=terminate_worker)
        backend = mock.MagicMock()
        backend.get_available_cameras.side_effect = RuntimeError("CUDA error: an illegal memory access was encountered")

        @contextlib.contextmanager
        def _return_backend(*args, **kwargs):
            yield backend

        server, channel, stub, service = self._start_service(fatal_worker_interceptor=interceptor)
        try:
            with mock.patch.object(service, "get_backend", _return_backend):
                with self.assertRaises(grpc.RpcError):
                    stub.get_available_cameras(AvailableCamerasRequest(scene_id="scene"))
            self.assertTrue(interceptor.wait_for_shutdown(timeout=2))
            terminate_worker.assert_called_once_with(1)
        finally:
            channel.close()
            server.stop(0)

    def test_batch_render_rgb_item_does_not_swallow_fatal_worker_error(self):
        server, channel, _, service = self._start_service()
        error = RuntimeError("CUDA error: an illegal memory access was encountered")
        try:
            with self.assertRaisesRegex(RuntimeError, "illegal memory access"):
                service._batch_rgb_error_item("cam", error)
        finally:
            channel.close()
            server.stop(0)

    def test_edit_assets_preserves_error_payload_for_unknown_track_key_error(self):
        # A KeyError raised inside the backend edit (e.g. unknown track id) must not be
        # misclassified as an unknown-scene NOT_FOUND abort; it should stay in the payload.
        backend = mock.MagicMock()
        backend.edit_assets.side_effect = KeyError("Track ID 'missing' not found.")

        @contextlib.contextmanager
        def _return_backend(*args, **kwargs):
            yield backend

        server, channel, stub, service = self._start_service()
        try:
            with mock.patch.object(service, "get_backend", _return_backend):
                response = stub.edit_assets(EditAssetsRequest(scene_id="scene"))
            self.assertFalse(response.success)
            self.assertIn("Failed to edit assets", response.message)
        finally:
            channel.close()
            server.stop(0)

    def test_render_rgb_returns_out_of_range(self):
        @contextlib.contextmanager
        def _raise_backend(*args, **kwargs):
            raise TimestampOutOfRangeError("timestamp outside range")
            yield  # pragma: no cover - unreachable

        server, channel, stub, service = self._start_service()
        try:
            with mock.patch.object(service, "get_backend", _raise_backend):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.render_rgb(RGBRenderRequest(scene_id="scene"))
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.OUT_OF_RANGE)
                details = ctx.exception.details() or ""
                self.assertIn("timestamp outside range", details)
        finally:
            channel.close()
            server.stop(0)

    def test_render_rgb_returns_resource_exhausted(self):
        @contextlib.contextmanager
        def _raise_backend(*args, **kwargs):
            raise torch.OutOfMemoryError("oom")
            yield  # pragma: no cover - unreachable

        server, channel, stub, service = self._start_service()
        try:
            with mock.patch.object(service, "get_backend", _raise_backend):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.render_rgb(RGBRenderRequest(scene_id="scene"))
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)
                details = ctx.exception.details() or ""
                self.assertIn("oom", details)
        finally:
            channel.close()
            server.stop(0)

    def test_batch_render_rgb_returns_failed_precondition(self):
        @contextlib.contextmanager
        def _raise_backend(*args, **kwargs):
            raise ModelIncompatibilityError("model mismatch")
            yield  # pragma: no cover - unreachable

        server, channel, stub, service = self._start_service()
        try:
            with mock.patch.object(service, "get_backend", _raise_backend):
                request = BatchRGBRenderRequest(
                    items=[
                        BatchRGBRenderRequestItem(
                            camera_name="cam",
                            request=RGBRenderRequest(scene_id="scene"),
                        )
                    ]
                )
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.batch_render_rgb(request)
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
                details = ctx.exception.details() or ""
                self.assertIn("model mismatch", details)
        finally:
            channel.close()
            server.stop(0)

    def test_render_lidar_returns_invalid_argument(self):
        @contextlib.contextmanager
        def _raise_backend(*args, **kwargs):
            raise ValueError("actor editing is disabled")
            yield  # pragma: no cover - unreachable

        server, channel, stub, service = self._start_service()
        try:
            with mock.patch.object(service, "get_backend", _raise_backend):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.render_lidar(LidarRenderRequest(scene_id="scene"))
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
                details = ctx.exception.details() or ""
                self.assertIn("actor editing is disabled", details)
        finally:
            channel.close()
            server.stop(0)


class TestHarmonizerDeferredLoad(unittest.TestCase):
    """set_harmonizer must not block startup: the port has to bind before the model loads."""

    def _service(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        return SensorSimService(
            server,
            artifacts_glob=None,
            ray_chunk_size=1,
            egocar_hoods_dir=None,
            downloader=None,
            scene_cache=None,
            cache_size=1,
            metrics_output_dir=None,
            enable_editing_actors=False,
            server_config=GrpcServerConfig(),
        )

    def test_set_harmonizer_returns_before_the_model_is_loaded(self) -> None:
        service = self._service()
        release = threading.Event()

        def slow_get(*args, **kwargs):
            assert release.wait(timeout=30), "loader thread was never released"
            return "model"

        with mock.patch("nurec.grpc.serve.HarmonizerModelFactory.get", side_effect=slow_get):
            service.set_harmonizer(True, "url", None, "file.pt", (1, 1))
            # Startup continues while the load is still in flight; the flag is already visible.
            self.assertTrue(service.harmonizer_enabled)
            self.assertFalse(service._harmonizer_ready.is_set())
            release.set()
            self.assertEqual(service._await_harmonizer_model(), "model")

    def test_a_render_waits_for_the_load_rather_than_skipping_it(self) -> None:
        service = self._service()
        release = threading.Event()
        with mock.patch(
            "nurec.grpc.serve.HarmonizerModelFactory.get", side_effect=lambda *a, **k: release.wait(30) and "model"
        ):
            service.set_harmonizer(True, "url", None, "file.pt", (1, 1))
            seen = []
            reader = threading.Thread(target=lambda: seen.append(service._await_harmonizer_model()))
            reader.start()
            reader.join(timeout=0.5)
            self.assertTrue(reader.is_alive(), "harmonizer_model must block while the load is pending")
            release.set()
            reader.join(timeout=30)
            self.assertEqual(seen, ["model"])

    def test_a_failed_load_is_raised_to_the_caller(self) -> None:
        service = self._service()
        with mock.patch("nurec.grpc.serve.HarmonizerModelFactory.get", side_effect=RuntimeError("boom")):
            service.set_harmonizer(True, "url", None, "file.pt", (1, 1))
            with self.assertRaises(RuntimeError):
                service._await_harmonizer_model()

    def test_a_cancelled_rpc_releases_the_worker(self) -> None:
        """max_workers defaults to 1, so a handler stuck past cancellation is the whole server."""
        service = self._service()
        with mock.patch("nurec.grpc.serve.HarmonizerModelFactory.get", side_effect=lambda *a, **k: time.sleep(60)):
            service.set_harmonizer(True, "url", None, "file.pt", (1, 1))
            ctx = mock.Mock()
            ctx.is_active.return_value = False
            with self.assertRaises(CacheWaitAbortedError) as caught:
                service._await_harmonizer_model(ctx)
            self.assertEqual(caught.exception.reason, CacheWaitAbortedReason.CANCELLED)

    def test_an_expired_deadline_releases_the_worker(self) -> None:
        service = self._service()
        with mock.patch("nurec.grpc.serve.HarmonizerModelFactory.get", side_effect=lambda *a, **k: time.sleep(60)):
            service.set_harmonizer(True, "url", None, "file.pt", (1, 1))
            ctx = mock.Mock()
            ctx.is_active.return_value = True
            ctx.time_remaining.return_value = 0.0
            with self.assertRaises(CacheWaitAbortedError) as caught:
                service._await_harmonizer_model(ctx)
            self.assertEqual(caught.exception.reason, CacheWaitAbortedReason.DEADLINE_EXCEEDED)

    def test_disabled_harmonizer_starts_no_loader(self) -> None:
        service = self._service()
        with mock.patch("nurec.grpc.serve.HarmonizerModelFactory.get") as get:
            service.set_harmonizer(False, "url", None, "file.pt", (1, 1))
            get.assert_not_called()
        self.assertFalse(service.harmonizer_enabled)

    def test_render_camera_forwards_the_rpc_context_to_the_harmonizer_wait(self) -> None:
        """_camera_render_return_from_request must pass its context through, not drop it, so a
        cancelled/expired RPC waiting on the model load is still detected (see MR review). The
        Harmonizer is opt-in per request, so the request asks for it."""
        with _service() as (stub, service):
            backend_ctx = _mock_backend_ctx_with_frame(CameraFrame(color_image=torch.zeros(4, 4, 3)))
            service.harmonizer_enabled = True
            with mock.patch.object(service, "get_backend", backend_ctx):
                with mock.patch.object(
                    service, "_await_harmonizer_model", return_value=lambda flat, size, state: flat
                ) as mock_await:
                    stub.render_camera(
                        CameraRenderRequest(scene_id="scene", image_format=ImageFormat.PNG, apply_harmonizer=True)
                    )
            mock_await.assert_called_once()
            (context_arg,) = mock_await.call_args.args
            self.assertIsNotNone(context_arg)


class TestHarmonizerSession(unittest.TestCase):
    """create/end_harmonizer_session RPCs + render_rgb temporal session routing.

    Exercises session management + the request contract only; temporal is enabled directly
    (bypassing the checkpoint load), so these never touch the model / GPU.
    """

    def _start_service(self, *, temporal_enabled: bool, max_sessions: int = 32):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        service = SensorSimService(
            server,
            artifacts_glob=None,
            ray_chunk_size=1,
            egocar_hoods_dir=None,
            downloader=None,
            scene_cache=None,
            cache_size=1,
            metrics_output_dir=None,
            enable_editing_actors=False,
            server_config=GrpcServerConfig(),
        )
        # Enable temporal without loading a checkpoint — these tests never reach the model.
        service.harmonizer_temporal_enabled = temporal_enabled
        service._max_harmonizer_sessions = max_sessions
        add_SensorsimServiceServicer_to_server(service, server)
        port = server.add_insecure_port("localhost:0")
        server.start()
        channel = grpc.insecure_channel(f"localhost:{port}")
        grpc.channel_ready_future(channel).result(timeout=5)
        return server, channel, SensorsimServiceStub(channel), service

    def test_create_requires_temporal_enabled(self):
        server, channel, stub, service = self._start_service(temporal_enabled=False)
        try:
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.create_harmonizer_session(Empty())
            self.assertEqual(ctx.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
        finally:
            channel.close()
            server.stop(0)

    def test_explicit_harmonizer_request_requires_enabled_model(self):
        server, channel, stub, service = self._start_service(temporal_enabled=False)
        try:
            backend = mock.MagicMock()
            backend.render_rgb_request.return_value = (torch.zeros((2, 2, 3)), (2, 2))

            @contextlib.contextmanager
            def _backend_ctx(*args, **kwargs):
                yield backend

            with mock.patch.object(service, "get_backend", _backend_ctx):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.render_rgb(
                        RGBRenderRequest(scene_id="scene", image_format=ImageFormat.PNG, apply_harmonizer=True)
                    )

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
            backend.render_rgb_request.assert_called_once()
        finally:
            channel.close()
            server.stop(0)

    def test_create_returns_session_id_and_registers_state(self):
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            resp = stub.create_harmonizer_session(Empty())
            self.assertTrue(resp.session_id)
            self.assertIn(resp.session_id, service._harmonizer_sessions)
        finally:
            channel.close()
            server.stop(0)

    def test_create_respects_max_sessions(self):
        server, channel, stub, service = self._start_service(temporal_enabled=True, max_sessions=1)
        try:
            stub.create_harmonizer_session(Empty())
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.create_harmonizer_session(Empty())
            self.assertEqual(ctx.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)
        finally:
            channel.close()
            server.stop(0)

    def test_end_closes_and_removes_session(self):
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            session_id = stub.create_harmonizer_session(Empty()).session_id
            state = service._harmonizer_sessions[session_id]
            stub.end_harmonizer_session(HarmonizerSessionId(session_id=session_id))
            self.assertNotIn(session_id, service._harmonizer_sessions)
            self.assertTrue(state.closed)
        finally:
            channel.close()
            server.stop(0)

    def test_end_unknown_session_returns_not_found(self):
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.end_harmonizer_session(HarmonizerSessionId(session_id="does-not-exist"))
            self.assertEqual(ctx.exception.code(), grpc.StatusCode.NOT_FOUND)
        finally:
            channel.close()
            server.stop(0)

    def test_temporal_rgb_opt_in_requires_session_id(self):
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.render_rgb(RGBRenderRequest(scene_id="scene", apply_harmonizer=True))
            self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        finally:
            channel.close()
            server.stop(0)

    def test_temporal_color_camera_opt_in_requires_session_id(self):
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.render_camera(
                    CameraRenderRequest(
                        scene_id="scene",
                        render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR],
                        apply_harmonizer=True,
                    )
                )
            self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        finally:
            channel.close()
            server.stop(0)

    def test_render_temporal_unknown_session_returns_not_found(self):
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            for apply_harmonizer in (False, True):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.render_rgb(
                        RGBRenderRequest(
                            scene_id="scene", harmonizer_session_id="nope", apply_harmonizer=apply_harmonizer
                        )
                    )
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.NOT_FOUND)
        finally:
            channel.close()
            server.stop(0)

    def test_temporal_raw_rgb_requires_no_session(self):
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            backend = mock.MagicMock()
            backend.render_rgb_request.return_value = (torch.zeros((2, 2, 3)), (2, 2))

            @contextlib.contextmanager
            def _backend_ctx(*args, **kwargs):
                yield backend

            with mock.patch.object(service, "get_backend", _backend_ctx):
                response = stub.render_rgb(RGBRenderRequest(scene_id="scene", image_format=ImageFormat.PNG))

            self.assertTrue(response.image_bytes)
            backend.render_rgb_request.assert_called_once()
        finally:
            channel.close()
            server.stop(0)

    def test_temporal_raw_color_camera_requires_no_session(self):
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:

            @contextlib.contextmanager
            def _backend_ctx(*args, **kwargs):
                yield mock.MagicMock()

            with (
                mock.patch.object(service, "get_backend", _backend_ctx),
                mock.patch.object(service, "_camera_render_return_from_request", return_value=CameraRenderReturn()),
            ):
                response = stub.render_camera(
                    CameraRenderRequest(
                        scene_id="scene",
                        render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR],
                    )
                )

            self.assertEqual(response, CameraRenderReturn())
        finally:
            channel.close()
            server.stop(0)

    def test_temporal_opt_out_resets_session_history(self):
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            session = stub.create_harmonizer_session(Empty())
            state = service._harmonizer_sessions[session.session_id]
            state.push(torch.zeros((3, 2, 2)))

            backend = mock.MagicMock()
            backend.render_rgb_request.return_value = (torch.zeros((2, 2, 3)), (2, 2))

            @contextlib.contextmanager
            def _backend_ctx(*args, **kwargs):
                yield backend

            with mock.patch.object(service, "get_backend", _backend_ctx):
                stub.render_rgb(
                    RGBRenderRequest(
                        scene_id="scene",
                        image_format=ImageFormat.PNG,
                        harmonizer_session_id=session.session_id,
                        apply_harmonizer=False,
                    )
                )

            self.assertFalse(state.closed)
            self.assertEqual(len(state.history), 0)
        finally:
            channel.close()
            server.stop(0)

    def test_camera_temporal_opt_out_resets_session_history(self):
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            session = stub.create_harmonizer_session(Empty())
            state = service._harmonizer_sessions[session.session_id]
            state.push(torch.zeros((3, 2, 2)))

            @contextlib.contextmanager
            def _backend_ctx(*args, **kwargs):
                yield mock.MagicMock()

            with (
                mock.patch.object(service, "get_backend", _backend_ctx),
                mock.patch.object(service, "_camera_render_return_from_request", return_value=CameraRenderReturn()),
            ):
                stub.render_camera(
                    CameraRenderRequest(
                        scene_id="scene",
                        render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR],
                        harmonizer_session_id=session.session_id,
                        apply_harmonizer=False,
                    )
                )

            self.assertFalse(state.closed)
            self.assertEqual(len(state.history), 0)
        finally:
            channel.close()
            server.stop(0)

    def test_temporal_opt_out_resets_session_history_when_render_fails(self):
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            session = stub.create_harmonizer_session(Empty())
            state = service._harmonizer_sessions[session.session_id]
            state.push(torch.zeros((3, 2, 2)))

            backend = mock.MagicMock()
            backend.render_rgb_request.side_effect = RuntimeError("render failed")

            @contextlib.contextmanager
            def _backend_ctx(*args, **kwargs):
                yield backend

            with mock.patch.object(service, "get_backend", _backend_ctx):
                with self.assertRaises(grpc.RpcError):
                    stub.render_rgb(
                        RGBRenderRequest(
                            scene_id="scene",
                            image_format=ImageFormat.PNG,
                            harmonizer_session_id=session.session_id,
                            apply_harmonizer=False,
                        )
                    )

            self.assertFalse(state.closed)
            self.assertEqual(len(state.history), 0)
        finally:
            channel.close()
            server.stop(0)

    def test_non_color_camera_render_does_not_require_temporal_session(self):
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:

            @contextlib.contextmanager
            def _backend_ctx(*args, **kwargs):
                yield mock.MagicMock()

            with (
                mock.patch.object(service, "get_backend", _backend_ctx),
                mock.patch.object(service, "_camera_render_return_from_request", return_value=CameraRenderReturn()),
            ):
                response = stub.render_camera(
                    CameraRenderRequest(
                        scene_id="scene",
                        render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED],
                    )
                )

            self.assertEqual(response, CameraRenderReturn())
        finally:
            channel.close()
            server.stop(0)

    @staticmethod
    def _rgb_batch_item(camera_name: str, **request_fields) -> BatchRGBRenderRequestItem:
        """Build a PNG batch_render_rgb item for scene "scene" with the given request fields."""
        return BatchRGBRenderRequestItem(
            camera_name=camera_name,
            request=RGBRenderRequest(scene_id="scene", image_format=ImageFormat.PNG, **request_fields),
        )

    @staticmethod
    def _camera_batch_item(camera_name: str, **request_fields) -> BatchCameraRenderRequestItem:
        """Build a colour batch_render_camera item for scene "scene" with the given request fields."""
        return BatchCameraRenderRequestItem(
            camera_name=camera_name,
            request=CameraRenderRequest(
                scene_id="scene",
                render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR],
                **request_fields,
            ),
        )

    def _assert_harmonized_with(self, model: mock.Mock, expected_states: list) -> None:
        """Assert each Harmonizer call got exactly the expected session object, in order.

        Compared by identity: HarmonizerState is a dataclass, so two fresh sessions compare equal.
        """
        self.assertEqual(len(model.call_args_list), len(expected_states))
        for call, expected in zip(model.call_args_list, expected_states):
            self.assertIs(call.args[2], expected)

    def test_batch_render_rgb_routes_each_item_to_its_own_temporal_session(self):
        """One session per camera is the intended batch pattern: distinct sessions are independent
        frame sequences, so a batch of them renders, each conditioned on its own history.
        """
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            session_a = stub.create_harmonizer_session(Empty()).session_id
            session_b = stub.create_harmonizer_session(Empty()).session_id
            state_a = service._harmonizer_sessions[session_a]
            state_b = service._harmonizer_sessions[session_b]
            service._harmonizer_model = mock.Mock(side_effect=lambda pixels, _img_size, _state: pixels)
            service._harmonizer_ready.set()

            backend = mock.MagicMock()
            backend.render_rgb_request.return_value = (torch.zeros((2, 2, 3)), (2, 2))

            @contextlib.contextmanager
            def _backend_ctx(*args, **kwargs):
                yield backend

            with mock.patch.object(service, "get_backend", _backend_ctx):
                response = stub.batch_render_rgb(
                    BatchRGBRenderRequest(
                        items=[
                            self._rgb_batch_item("cam_a", apply_harmonizer=True, harmonizer_session_id=session_a),
                            self._rgb_batch_item("cam_b", apply_harmonizer=True, harmonizer_session_id=session_b),
                            self._rgb_batch_item("cam_raw"),
                        ]
                    )
                )

            self.assertEqual([item.success for item in response.items], [True, True, True])
            self._assert_harmonized_with(service._harmonizer_model, [state_a, state_b])
        finally:
            channel.close()
            server.stop(0)

    def test_batch_render_rgb_rejects_a_repeated_temporal_session_before_rendering(self):
        """A session is one coherent frame sequence; two items in the same batch would touch it twice
        with no defined order, so the whole batch is refused and nothing renders.
        """
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            session = stub.create_harmonizer_session(Empty()).session_id
            get_backend = mock.MagicMock()

            with mock.patch.object(service, "get_backend", get_backend):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.batch_render_rgb(
                        BatchRGBRenderRequest(
                            items=[
                                self._rgb_batch_item("cam_a", apply_harmonizer=True, harmonizer_session_id=session),
                                self._rgb_batch_item("cam_b", apply_harmonizer=True, harmonizer_session_id=session),
                            ]
                        )
                    )

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
            self.assertIn("at most once per batch", ctx.exception.details())
            self.assertIn("'cam_a'", ctx.exception.details())
            self.assertIn("'cam_b'", ctx.exception.details())
            get_backend.assert_not_called()
        finally:
            channel.close()
            server.stop(0)

    def test_batch_render_rgb_unknown_temporal_session_fails_the_whole_batch(self):
        """A bad session on any item fails the batch before anything renders or resets, so the
        opt-out item's session keeps its history.
        """
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            session = stub.create_harmonizer_session(Empty()).session_id
            state = service._harmonizer_sessions[session]
            state.push(torch.zeros((3, 2, 2)))
            get_backend = mock.MagicMock()

            with mock.patch.object(service, "get_backend", get_backend):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.batch_render_rgb(
                        BatchRGBRenderRequest(
                            items=[
                                self._rgb_batch_item("cam_ok", apply_harmonizer=False, harmonizer_session_id=session),
                                self._rgb_batch_item("cam_bad", apply_harmonizer=True, harmonizer_session_id="nope"),
                            ]
                        )
                    )

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.NOT_FOUND)
            self.assertIn("item[1] 'cam_bad'", ctx.exception.details())
            get_backend.assert_not_called()
            self.assertEqual(len(state.history), 1)
        finally:
            channel.close()
            server.stop(0)

    def test_batch_render_rgb_temporal_opt_in_without_session_fails_the_whole_batch(self):
        """An opted-in item without a session fails the batch with INVALID_ARGUMENT before anything renders."""
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            get_backend = mock.MagicMock()

            with mock.patch.object(service, "get_backend", get_backend):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.batch_render_rgb(
                        BatchRGBRenderRequest(items=[self._rgb_batch_item("cam", apply_harmonizer=True)])
                    )

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
            self.assertIn("item[0] 'cam'", ctx.exception.details())
            get_backend.assert_not_called()
        finally:
            channel.close()
            server.stop(0)

    def test_batch_render_camera_rejects_a_repeated_temporal_session_before_rendering(self):
        """batch_render_camera refuses two items that name one session, before anything renders."""
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            session = stub.create_harmonizer_session(Empty()).session_id
            get_backend = mock.MagicMock()

            with mock.patch.object(service, "get_backend", get_backend):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.batch_render_camera(
                        BatchCameraRenderRequest(
                            items=[
                                self._camera_batch_item("cam_a", apply_harmonizer=True, harmonizer_session_id=session),
                                self._camera_batch_item("cam_b", apply_harmonizer=True, harmonizer_session_id=session),
                            ]
                        )
                    )

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
            self.assertIn("at most once per batch", ctx.exception.details())
            get_backend.assert_not_called()
        finally:
            channel.close()
            server.stop(0)

    def test_batch_render_camera_routes_each_item_to_its_own_temporal_session(self):
        """batch_render_camera renders JPEG items through the batched nvJPEG encoder and every other
        format inline, so both branches must carry the item's own session.
        """
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            session_jpeg = stub.create_harmonizer_session(Empty()).session_id
            session_png = stub.create_harmonizer_session(Empty()).session_id
            state_jpeg = service._harmonizer_sessions[session_jpeg]
            state_png = service._harmonizer_sessions[session_png]
            service._harmonizer_model = mock.Mock(side_effect=lambda pixels, _img_size, _state: pixels)
            service._harmonizer_ready.set()

            backend = mock.MagicMock()
            backend.render_camera_request.return_value = (CameraFrame(color_image=torch.zeros(2, 2, 3)), None, None)
            backend.source_artifact.semantic_class_colors = None
            backend.source_artifact.semantic_class_names = None

            @contextlib.contextmanager
            def _backend_ctx(*args, **kwargs):
                yield backend

            encoder = mock.MagicMock()
            encoder.max_batch_size = BATCH_RENDER_REQUEST_CHUNK_SIZE
            encoder.encode.side_effect = lambda images, _quality: [b"jpeg"] * len(images)

            # A chunk size of 1 puts the two items in different chunks.
            with (
                mock.patch.object(service, "get_backend", _backend_ctx),
                mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder),
                mock.patch("nurec.grpc.serve.BATCH_RENDER_REQUEST_CHUNK_SIZE", 1),
            ):
                response = stub.batch_render_camera(
                    BatchCameraRenderRequest(
                        items=[
                            self._camera_batch_item(
                                "cam_jpeg",
                                image_format=ImageFormat.JPEG,
                                image_quality=90,
                                apply_harmonizer=True,
                                harmonizer_session_id=session_jpeg,
                            ),
                            self._camera_batch_item(
                                "cam_png",
                                image_format=ImageFormat.PNG,
                                apply_harmonizer=True,
                                harmonizer_session_id=session_png,
                            ),
                        ]
                    )
                )

            self.assertEqual([item.success for item in response.items], [True, True])
            self._assert_harmonized_with(service._harmonizer_model, [state_jpeg, state_png])
        finally:
            channel.close()
            server.stop(0)

    def test_batch_render_rgb_routes_ipc_and_encoded_items_to_their_own_temporal_sessions(self):
        """batch_render_rgb writes use_cuda_ipc items into reserved buffers and encodes the rest,
        so both branches must carry the item's own session.
        """
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            session_ipc = stub.create_harmonizer_session(Empty()).session_id
            session_png = stub.create_harmonizer_session(Empty()).session_id
            state_ipc = service._harmonizer_sessions[session_ipc]
            state_png = service._harmonizer_sessions[session_png]
            service._harmonizer_model = mock.Mock(side_effect=lambda pixels, _img_size, _state: pixels)
            service._harmonizer_ready.set()

            backend = mock.MagicMock()
            backend.render_rgb_request.return_value = (torch.zeros((2, 2, 3)), (2, 2))

            @contextlib.contextmanager
            def _backend_ctx(*args, **kwargs):
                yield backend

            mgr = mock.MagicMock()
            mgr.reserve_buffers.return_value = [3]

            # A chunk size of 1 puts the two items in different chunks.
            with (
                mock.patch.object(service, "get_backend", _backend_ctx),
                mock.patch("nurec.grpc.serve.BATCH_RENDER_REQUEST_CHUNK_SIZE", 1),
                mock.patch.object(service, "_get_ipc_mgr", return_value=mgr),
                mock.patch.object(
                    service,
                    "_ipc_write_reserved_rgb",
                    return_value=RGBRenderReturn(format="rgb_uint8_planar", buffer_id=3),
                ),
            ):
                response = stub.batch_render_rgb(
                    BatchRGBRenderRequest(
                        items=[
                            BatchRGBRenderRequestItem(
                                camera_name="cam_ipc",
                                request=RGBRenderRequest(
                                    scene_id="scene",
                                    use_cuda_ipc=True,
                                    ipc_session_id="ipc",
                                    image_format=ImageFormat.RGB_UINT8_PLANAR,
                                    apply_harmonizer=True,
                                    harmonizer_session_id=session_ipc,
                                ),
                            ),
                            self._rgb_batch_item("cam_png", apply_harmonizer=True, harmonizer_session_id=session_png),
                        ]
                    )
                )

            self.assertEqual([item.success for item in response.items], [True, True])
            self.assertEqual(response.items[0].result.buffer_id, 3)
            self._assert_harmonized_with(service._harmonizer_model, [state_ipc, state_png])
        finally:
            channel.close()
            server.stop(0)

    def test_batch_render_rgb_bad_temporal_session_fails_before_reserving_ipc_buffers(self):
        """Sessions are resolved before CUDA IPC buffers are reserved, so a failed batch holds none."""
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            mgr = mock.MagicMock()

            with mock.patch.object(service, "_get_ipc_mgr", return_value=mgr):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.batch_render_rgb(
                        BatchRGBRenderRequest(
                            items=[
                                BatchRGBRenderRequestItem(
                                    camera_name="cam_ipc",
                                    request=RGBRenderRequest(
                                        scene_id="scene",
                                        use_cuda_ipc=True,
                                        ipc_session_id="ipc",
                                        image_format=ImageFormat.RGB_UINT8_PLANAR,
                                        apply_harmonizer=True,
                                        harmonizer_session_id="nope",
                                    ),
                                )
                            ]
                        )
                    )

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.NOT_FOUND)
            mgr.reserve_buffers.assert_not_called()
        finally:
            channel.close()
            server.stop(0)

    def test_batch_render_rgb_opt_out_item_resets_its_session(self):
        """An item that names its session without opting in resets that session, as render_rgb does,
        while an opted-in item in the same batch harmonizes on its own session.
        """
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            session_out = stub.create_harmonizer_session(Empty()).session_id
            session_in = stub.create_harmonizer_session(Empty()).session_id
            state_out = service._harmonizer_sessions[session_out]
            state_in = service._harmonizer_sessions[session_in]
            state_out.push(torch.zeros((3, 2, 2)))
            service._harmonizer_model = mock.Mock(side_effect=lambda pixels, _img_size, _state: pixels)
            service._harmonizer_ready.set()

            backend = mock.MagicMock()
            backend.render_rgb_request.return_value = (torch.zeros((2, 2, 3)), (2, 2))

            with mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)):
                response = stub.batch_render_rgb(
                    BatchRGBRenderRequest(
                        items=[
                            self._rgb_batch_item("cam_out", apply_harmonizer=False, harmonizer_session_id=session_out),
                            self._rgb_batch_item("cam_in", apply_harmonizer=True, harmonizer_session_id=session_in),
                        ]
                    )
                )

            self.assertEqual([item.success for item in response.items], [True, True])
            self.assertEqual(len(state_out.history), 0)
            self.assertFalse(state_out.closed)
            self._assert_harmonized_with(service._harmonizer_model, [state_in])
        finally:
            channel.close()
            server.stop(0)

    def test_batch_render_camera_opt_out_resets_its_session_even_when_the_batch_fails(self):
        """Resets happen once the batch is accepted, before rendering, so a later failure does not
        leave stale history behind (render_camera behaves the same).
        """
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            session = stub.create_harmonizer_session(Empty()).session_id
            state = service._harmonizer_sessions[session]
            state.push(torch.zeros((3, 2, 2)))

            with mock.patch.object(service, "get_backend", side_effect=CacheFullError("cache is full")):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.batch_render_camera(
                        BatchCameraRenderRequest(
                            items=[
                                self._camera_batch_item(
                                    "cam",
                                    image_format=ImageFormat.JPEG,
                                    image_quality=90,
                                    harmonizer_session_id=session,
                                )
                            ]
                        )
                    )

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)
            self.assertEqual(len(state.history), 0)
        finally:
            channel.close()
            server.stop(0)

    def test_batch_render_rgb_on_a_non_temporal_server_ignores_session_ids(self):
        """Without the temporal model, session IDs are ignored as in render_rgb: a repeated one is not
        an error, and every opted-in item gets the non-temporal Harmonizer.
        """
        server, channel, stub, service = self._start_service(temporal_enabled=False)
        try:
            service.harmonizer_enabled = True
            service._harmonizer_model = mock.Mock(side_effect=lambda pixels, _img_size, _flag: pixels)
            service._harmonizer_ready.set()

            backend = mock.MagicMock()
            backend.render_rgb_request.return_value = (torch.zeros((2, 2, 3)), (2, 2))

            with mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)):
                response = stub.batch_render_rgb(
                    BatchRGBRenderRequest(
                        items=[
                            self._rgb_batch_item("cam_a", apply_harmonizer=True, harmonizer_session_id="ignored"),
                            self._rgb_batch_item("cam_b", apply_harmonizer=True, harmonizer_session_id="ignored"),
                        ]
                    )
                )

            self.assertEqual([item.success for item in response.items], [True, True])
            self.assertEqual(
                service._harmonizer_model.call_args_list,
                [mock.call(mock.ANY, (2, 2), False), mock.call(mock.ANY, (2, 2), False)],
            )
        finally:
            channel.close()
            server.stop(0)

    def test_resolve_batch_harmonizer_states_ignores_non_colour_camera_items(self):
        """Harmonizer only applies to the colour product, so a distance-only item needs no session
        and does not count towards the once-per-batch rule even if it names one.
        """
        server, channel, stub, service = self._start_service(temporal_enabled=True)
        try:
            session = stub.create_harmonizer_session(Empty()).session_id
            state = service._harmonizer_sessions[session]
            context = mock.Mock()

            states = service._resolve_batch_harmonizer_states(
                [
                    self._camera_batch_item("colour", apply_harmonizer=True, harmonizer_session_id=session),
                    BatchCameraRenderRequestItem(
                        camera_name="distance",
                        request=CameraRenderRequest(
                            scene_id="scene",
                            render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED],
                            apply_harmonizer=True,
                            harmonizer_session_id=session,
                        ),
                    ),
                ],
                context,
                "batch_render_camera",
            )

            self.assertEqual(states, [state, None])
            context.abort.assert_not_called()
        finally:
            channel.close()
            server.stop(0)

    def test_set_harmonizer_temporal_loads_temporal_model(self):
        service = SensorSimService(
            None,
            artifacts_glob=None,
            ray_chunk_size=1,
            egocar_hoods_dir=None,
            downloader=None,
            scene_cache=None,
            cache_size=1,
            metrics_output_dir=None,
            enable_editing_actors=False,
            server_config=GrpcServerConfig(),
        )
        with mock.patch("nurec.grpc.serve.HarmonizerModelFactory.get") as mock_get:
            service.set_harmonizer(
                harmonizer_enabled=False,
                harmonizer_url="unused",
                harmonizer_cache_dir="~/.cache/nurec/harmonizer",
                model_filename="unused.pt",
                harmonizer_resolution=(576, 1024),
                harmonizer_temporal_enabled=True,
                harmonizer_temporal_cosmos_cache_dir="/tmp/cosmos",
                max_harmonizer_sessions=8,
            )
            # set_harmonizer only starts the load; wait for the background thread to reach
            # mock_get before asserting on it, or this is racy (see MR review).
            self.assertTrue(service._harmonizer_ready.wait(timeout=5))
        self.assertTrue(service.harmonizer_temporal_enabled)
        self.assertEqual(service._max_harmonizer_sessions, 8)
        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        self.assertTrue(kwargs.get("temporal"))
        self.assertEqual(kwargs.get("cosmos_base_model_url"), HARMONIZER_TEMPORAL_COSMOS_BASE_MODEL_URL)
        self.assertEqual(kwargs.get("cosmos_tokenizer_url"), HARMONIZER_TEMPORAL_COSMOS_TOKENIZER_URL)
        self.assertEqual(kwargs.get("cosmos_cache_dir"), "/tmp/cosmos")


class TestHarmonizerPerRequestControl(unittest.TestCase):
    def setUp(self):
        self.service = SensorSimService.__new__(SensorSimService)
        self.service.harmonizer_enabled = True
        self.service.harmonizer_temporal_enabled = False
        self.service._harmonizer_model = mock.Mock(side_effect=lambda pixels, _img_size, _color_transfer: pixels + 1)
        self.service._harmonizer_ready = threading.Event()
        self.service._harmonizer_ready.set()
        self.service._harmonizer_error = None
        self.raster = torch.zeros((2, 3, 3))

    def test_harmonizer_is_opt_in_for_rgb_and_camera_requests(self):
        for request in (RGBRenderRequest(), CameraRenderRequest()):
            result = self.service._maybe_apply_harmonizer(self.raster, (2, 3), request)
            self.assertIs(result, self.raster)
        self.service._harmonizer_model.assert_not_called()

        for request in (RGBRenderRequest(apply_harmonizer=True), CameraRenderRequest(apply_harmonizer=True)):
            result = self.service._maybe_apply_harmonizer(self.raster, (2, 3), request)
            torch.testing.assert_close(result, torch.ones_like(self.raster))
        self.assertEqual(self.service._harmonizer_model.call_count, 2)

    def test_harmonizer_request_requires_a_loaded_model(self):
        self.service.harmonizer_enabled = False

        with self.assertRaises(HarmonizerNotEnabledError):
            self.service._maybe_apply_harmonizer(self.raster, (2, 3), RGBRenderRequest(apply_harmonizer=True))

        self.service._harmonizer_model.assert_not_called()

    def test_harmonizer_uses_the_background_loaded_model(self):
        service = SensorSimService.__new__(SensorSimService)
        service.harmonizer_enabled = True
        service.harmonizer_temporal_enabled = False
        service._harmonizer_model = mock.Mock(side_effect=lambda pixels, _img_size, _color_transfer: pixels + 1)
        service._harmonizer_ready = threading.Event()
        service._harmonizer_ready.set()
        service._harmonizer_error = None

        result = service._maybe_apply_harmonizer(self.raster, (2, 3), RGBRenderRequest(apply_harmonizer=True))

        torch.testing.assert_close(result, torch.ones_like(self.raster))
        service._harmonizer_model.assert_called_once_with(mock.ANY, (2, 3), False)

    def test_temporal_harmonizer_uses_the_resolved_session_when_requested(self):
        self.service.harmonizer_enabled = False
        self.service.harmonizer_temporal_enabled = True
        state = mock.sentinel.harmonizer_state

        result = self.service._maybe_apply_harmonizer(
            self.raster, (2, 3), RGBRenderRequest(apply_harmonizer=True), state
        )

        torch.testing.assert_close(result, torch.ones_like(self.raster))
        self.service._harmonizer_model.assert_called_once_with(mock.ANY, (2, 3), state)

    def test_temporal_harmonizer_requires_resolved_session(self):
        self.service.harmonizer_enabled = False
        self.service.harmonizer_temporal_enabled = True

        with self.assertRaisesRegex(RuntimeError, "resolved session"):
            self.service._maybe_apply_harmonizer(self.raster, (2, 3), RGBRenderRequest(apply_harmonizer=True))


class TestServerConfig(unittest.TestCase):
    def setUp(self):
        """Create a default GrpcServerConfig and the map build_server_config_map should produce from it."""
        self.server_config = GrpcServerConfig()
        self.expected_server_config_map = {
            "artifact_glob": "null",
            "host": "localhost",
            "port": "8080",
            "health_port": "null",
            "test_scenes_are_valid": "False",
            "renderer": "default",
            "allowed_image_formats": "null",
            "enable_harmonizer": "False",
            "harmonizer_url": "https://huggingface.co/nvidia/Harmonizer/blob/main/harmonizer_nontemporal.pt",
            "harmonizer_cache": "~/.cache/nurec/harmonizer",
            "harmonizer_model_filename": "harmonizer_nontemporal.pt",
            "harmonizer_resolution": "(576, 1024)",
            "enable_harmonizer_temporal": "False",
            "harmonizer_temporal_url": HARMONIZER_TEMPORAL_MODEL_URL,
            "harmonizer_temporal_model_filename": HARMONIZER_TEMPORAL_MODEL_FILENAME,
            "harmonizer_temporal_resolution": "(576, 1024)",
            "harmonizer_temporal_cosmos_base_model_url": HARMONIZER_TEMPORAL_COSMOS_BASE_MODEL_URL,
            "harmonizer_temporal_cosmos_tokenizer_url": HARMONIZER_TEMPORAL_COSMOS_TOKENIZER_URL,
            "harmonizer_temporal_cosmos_cache_dir": "null",
            "max_harmonizer_sessions": "32",
            "ray_chunk_size": "4611686018427387904",
            "egocar_hood_dir": "null",
            "egocar_hood_cache_size": "32",
            "download_cache_dir": "~/.cache/nurec/downloaded_scenes",
            "download_cache_size": "5",
            "max_workers": "1",
            "enable_editing_actors": "False",
            "camera_only": "False",
            "cache_size": "10",
            "max_backends_per_scene": "null",
            "cuda_ipc_max_buffer_count": "64",
            "backend_cache_wait_timeout_s": "120.0",
            "metrics_output_dir": "null",
        }

    def test_server_config_map(self):
        server_config_map = SensorSimService.build_server_config_map(self.server_config)
        self.assertEqual(
            server_config_map,
            self.expected_server_config_map,
        )

    def test_server_config_map_reports_image_format_policy(self):
        server_config = GrpcServerConfig(allowed_image_formats="jpeg, rgb")
        server_config_map = SensorSimService.build_server_config_map(server_config)

        self.assertEqual(server_config_map["allowed_image_formats"], "JPEG,RGB_UINT8_PLANAR")

    @unittest.skip("Skipping server client round trip test by default to avoid flakiness in CI.")
    def test_server_client_round_trip(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        server_config = GrpcServerConfig()
        service = SensorSimService(
            server,
            artifacts_glob=None,
            ray_chunk_size=1,
            egocar_hoods_dir=None,
            downloader=None,
            scene_cache=None,
            cache_size=1,
            metrics_output_dir=None,
            enable_editing_actors=False,
            server_config=server_config,
        )
        add_SensorsimServiceServicer_to_server(service, server)
        port = server.add_insecure_port("localhost:0")
        server.start()

        channel = grpc.insecure_channel(f"localhost:{port}")
        try:
            grpc.channel_ready_future(channel).result(timeout=5)
            stub = SensorsimServiceStub(channel)
            response = stub.get_server_config(Empty())

            self.assertEqual(response.server_config, self.expected_server_config_map)
        finally:
            channel.close()
            server.stop(0)


def _smooth_rgb_image(height: int, width: int, offset: int) -> np.ndarray:
    """Build a low-frequency RGB image, which JPEG at quality 100 and 4:4:4 sampling reproduces near-exactly."""
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    columns = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ramp = (rows + columns) / 2.0
    channels = [np.clip(ramp * 180.0 + offset + channel_offset, 0.0, 255.0) for channel_offset in (0, 20, 40)]
    return np.stack(channels, axis=-1).astype(np.uint8)


class TestNvjpegBatchEncoderSemantics(unittest.TestCase):
    @staticmethod
    def _pending_jpeg(camera_name: str, pixel_value: int, quality: float) -> PendingBatchRGBEncode:
        return PendingBatchRGBEncode(
            camera_name=camera_name,
            image_u8=torch.full((2, 2, 3), pixel_value, dtype=torch.uint8),
            request=RGBRenderRequest(image_format=ImageFormat.JPEG, image_quality=quality),
        )

    @staticmethod
    def _recording_encoder() -> mock.MagicMock:
        encoder = mock.MagicMock()
        encoder.max_batch_size = BATCH_RENDER_REQUEST_CHUNK_SIZE
        encoder.encode.side_effect = lambda images, quality: [
            f"{quality}:{int(image[0, 0, 0].item())}".encode() for image in images
        ]
        return encoder

    @staticmethod
    def _mock_cuda_image(device: int = 0) -> mock.MagicMock:
        image = mock.MagicMock(spec=torch.Tensor)
        image.is_cuda = True
        image.get_device.return_value = device
        return image

    @staticmethod
    @contextlib.contextmanager
    def _backend_context(backend: mock.MagicMock):
        yield backend

    def test_poisoned_device_encoder_is_replaced_for_next_request(self):
        class NativePoisonedError(RuntimeError):
            pass

        first_encoder = mock.MagicMock()
        first_encoder.encode_batch.side_effect = NativePoisonedError("poisoned")
        replacement_encoder = mock.MagicMock()
        replacement_encoder.encode_batch.return_value = [b"recovered"]
        native_module = mock.MagicMock()
        native_module.NvjpegEncoderPoisonedError = NativePoisonedError
        native_module.NvjpegOutOfMemoryError = type("NativeOutOfMemoryError", (RuntimeError,), {})
        native_module.NvjpegBatchEncoder.side_effect = [first_encoder, replacement_encoder]

        encoder = NvjpegBatchEncoder.__new__(NvjpegBatchEncoder)
        encoder._module = native_module
        encoder._encoders = {}
        encoder._encoders_lock = threading.Lock()
        image = self._mock_cuda_image()

        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            encoder.encode([image], 90)

        self.assertEqual(encoder.encode([image], 90), [b"recovered"])
        self.assertEqual(native_module.NvjpegBatchEncoder.call_count, 2)

    def test_poison_cleanup_does_not_remove_concurrent_replacement(self):
        class NativePoisonedError(RuntimeError):
            pass

        encoder = NvjpegBatchEncoder.__new__(NvjpegBatchEncoder)
        native_module = mock.MagicMock()
        native_module.NvjpegEncoderPoisonedError = NativePoisonedError
        native_module.NvjpegOutOfMemoryError = type("NativeOutOfMemoryError", (RuntimeError,), {})
        encoder._module = native_module
        encoder._encoders_lock = threading.Lock()
        replacement_encoder = mock.MagicMock()
        replacement_encoder.encode_batch.return_value = [b"replacement"]
        failed_encoder = mock.MagicMock()

        def replace_then_fail(*args, **kwargs):
            with encoder._encoders_lock:
                encoder._encoders[0] = replacement_encoder
            raise NativePoisonedError("poisoned")

        failed_encoder.encode_batch.side_effect = replace_then_fail
        encoder._encoders = {0: failed_encoder}
        image = self._mock_cuda_image()

        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            encoder.encode([image], 90)

        self.assertEqual(encoder.encode([image], 90), [b"replacement"])
        native_module.NvjpegBatchEncoder.assert_not_called()

    def test_single_jpeg_encode_uses_native_batch_encoder(self):
        service = SensorSimService(
            None,
            artifacts_glob=None,
            ray_chunk_size=1,
        )
        encoder = mock.MagicMock()
        encoder.encode.return_value = [b"jpeg-bytes"]

        with mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder):
            result = service.encode_image(
                torch.zeros((1, 1, 3), dtype=torch.uint8),
                RGBRenderRequest(image_format=ImageFormat.JPEG, image_quality=95),
            )

        self.assertEqual(result.image_bytes, b"jpeg-bytes")
        encoder.encode.assert_called_once()
        images_arg, quality_arg = encoder.encode.call_args.args
        self.assertEqual(len(images_arg), 1)
        self.assertEqual(quality_arg, 95)

    def test_same_quality_pending_jpegs_use_one_encoder_batch(self):
        service = SensorSimService(None, artifacts_glob=None, ray_chunk_size=1)
        encoder = self._recording_encoder()
        pending = [
            self._pending_jpeg("cam_0", pixel_value=10, quality=90),
            self._pending_jpeg("cam_1", pixel_value=20, quality=90),
            self._pending_jpeg("cam_2", pixel_value=30, quality=90),
        ]

        with mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder):
            results = service._finish_pending_batch_rgb_results(pending)

        self.assertEqual([result.camera_name for result in results], ["cam_0", "cam_1", "cam_2"])
        self.assertTrue(all(result.success for result in results))
        self.assertEqual([result.result.image_bytes for result in results], [b"90:10", b"90:20", b"90:30"])
        encoder.encode.assert_called_once()
        images_arg, quality_arg = encoder.encode.call_args.args
        self.assertEqual(len(images_arg), 3)
        self.assertEqual(quality_arg, 90)

    def test_mixed_quality_pending_jpegs_batch_by_quality_and_preserve_order(self):
        service = SensorSimService(None, artifacts_glob=None, ray_chunk_size=1)
        encoder = self._recording_encoder()
        pending = [
            self._pending_jpeg("cam_0", pixel_value=10, quality=80),
            self._pending_jpeg("cam_1", pixel_value=20, quality=95),
            self._pending_jpeg("cam_2", pixel_value=30, quality=80),
        ]

        with mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder):
            results = service._finish_pending_batch_rgb_results(pending)

        self.assertEqual([result.camera_name for result in results], ["cam_0", "cam_1", "cam_2"])
        self.assertTrue(all(result.success for result in results))
        self.assertEqual([result.result.image_bytes for result in results], [b"80:10", b"95:20", b"80:30"])
        self.assertCountEqual(
            [(len(call.args[0]), call.args[1]) for call in encoder.encode.call_args_list],
            [(2, 80), (1, 95)],
        )

    def test_invalid_quality_fails_only_its_item(self):
        service = SensorSimService(None, artifacts_glob=None, ray_chunk_size=1)
        encoder = self._recording_encoder()
        pending = [
            self._pending_jpeg("cam_0", pixel_value=10, quality=80),
            self._pending_jpeg("cam_bad", pixel_value=20, quality=0),
            self._pending_jpeg("cam_2", pixel_value=30, quality=95),
        ]

        with mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder):
            results = service._finish_pending_batch_rgb_results(pending)

        self.assertEqual([result.camera_name for result in results], ["cam_0", "cam_bad", "cam_2"])
        self.assertEqual([result.success for result in results], [True, False, True])
        self.assertEqual(results[0].result.image_bytes, b"80:10")
        self.assertIn("quality", results[1].error_message.lower())
        self.assertEqual(results[2].result.image_bytes, b"95:30")
        self.assertCountEqual(
            [(len(call.args[0]), call.args[1]) for call in encoder.encode.call_args_list],
            [(1, 80), (1, 95)],
        )

    def test_non_finite_quality_fails_only_its_rgb_item(self):
        for invalid_quality in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid_quality=invalid_quality):
                service = SensorSimService(None, artifacts_glob=None, ray_chunk_size=1)
                encoder = self._recording_encoder()
                pending = [
                    self._pending_jpeg("cam_0", pixel_value=10, quality=80),
                    self._pending_jpeg("cam_bad", pixel_value=20, quality=invalid_quality),
                    self._pending_jpeg("cam_2", pixel_value=30, quality=95),
                ]

                with mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder):
                    results = service._finish_pending_batch_rgb_results(pending)

                self.assertEqual([result.success for result in results], [True, False, True])
                self.assertIn("finite", results[1].error_message.lower())
                self.assertCountEqual(
                    [(len(call.args[0]), call.args[1]) for call in encoder.encode.call_args_list],
                    [(1, 80), (1, 95)],
                )

    def test_non_finite_quality_fails_only_its_camera_output(self):
        for invalid_quality in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid_quality=invalid_quality):
                service = SensorSimService(None, artifacts_glob=None, ray_chunk_size=1)
                encoder = self._recording_encoder()
                pending_outputs = [
                    PendingCameraRenderOutput(
                        image_u8=torch.full((2, 2, 3), pixel_value, dtype=torch.uint8),
                        request=CameraRenderRequest(image_format=ImageFormat.JPEG, image_quality=quality),
                        product=CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR,
                        semantics_class_mapping=None,
                        distance_colormap_vmax_m=None,
                    )
                    for pixel_value, quality in ((10, 80), (20, invalid_quality), (30, 95))
                ]

                with mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder):
                    outcomes = service._encode_pending_camera_render_output_bytes(pending_outputs)

                self.assertEqual(outcomes[0], b"80:10")
                self.assertIsInstance(outcomes[1], ValueError)
                self.assertIn("finite", str(outcomes[1]).lower())
                self.assertEqual(outcomes[2], b"95:30")
                self.assertCountEqual(
                    [(len(call.args[0]), call.args[1]) for call in encoder.encode.call_args_list],
                    [(1, 80), (1, 95)],
                )

    def test_one_bad_jpeg_isolated_from_other_same_quality_items(self):
        class ImageSpecificEncodingError(RuntimeError):
            pass

        service = SensorSimService(None, artifacts_glob=None, ray_chunk_size=1)
        encoder = self._recording_encoder()
        original_encode = encoder.encode.side_effect

        def reject_pixel_value_20(images, quality):
            if any(int(image[0, 0, 0].item()) == 20 for image in images):
                raise ImageSpecificEncodingError("bad image")
            return original_encode(images, quality)

        encoder.encode.side_effect = reject_pixel_value_20
        pending = [
            self._pending_jpeg("cam_0", pixel_value=10, quality=90),
            self._pending_jpeg("cam_bad", pixel_value=20, quality=90),
            self._pending_jpeg("cam_2", pixel_value=30, quality=90),
        ]

        with (
            mock.patch("nurec.grpc.serve.NvjpegImageEncodingError", ImageSpecificEncodingError, create=True),
            mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder),
        ):
            results = service._finish_pending_batch_rgb_results(pending)

        self.assertEqual([result.success for result in results], [True, False, True])
        self.assertEqual(results[0].result.image_bytes, b"90:10")
        self.assertIn("bad image", results[1].error_message)
        self.assertEqual(results[2].result.image_bytes, b"90:30")

    def test_encoder_global_failure_is_not_bisected(self):
        service = SensorSimService(None, artifacts_glob=None, ray_chunk_size=1)
        encoder = self._recording_encoder()
        encoder.encode.side_effect = RuntimeError("global encoder failure")
        pending = [
            self._pending_jpeg("cam_0", pixel_value=10, quality=90),
            self._pending_jpeg("cam_1", pixel_value=20, quality=90),
            self._pending_jpeg("cam_2", pixel_value=30, quality=90),
        ]

        with (
            mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder),
            self.assertRaisesRegex(RuntimeError, "global encoder failure"),
        ):
            service._finish_pending_batch_rgb_results(pending)

        encoder.encode.assert_called_once()

    def test_native_oom_propagates_for_server_recovery(self):
        service = SensorSimService(None, artifacts_glob=None, ray_chunk_size=1)
        encoder = mock.MagicMock()
        encoder.encode.side_effect = torch.cuda.OutOfMemoryError("nvjpeg allocation failed")

        with (
            mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder),
            self.assertRaises(torch.cuda.OutOfMemoryError),
        ):
            service._finish_pending_batch_rgb_results([self._pending_jpeg("cam_0", pixel_value=10, quality=90)])

    def test_render_camera_batches_same_quality_products_in_one_encode(self):
        service = SensorSimService(None, artifacts_glob=None, ray_chunk_size=1)
        backend = mock.MagicMock()
        backend.render_camera_request.return_value = (
            CameraFrame(color_image=torch.zeros(4, 4, 3), distance_image=torch.zeros(4, 4)),
            None,
            None,
        )
        backend.source_artifact.semantic_class_names = None
        backend.source_artifact.semantic_class_colors = None
        encoder = mock.MagicMock()
        encoder.max_batch_size = BATCH_RENDER_REQUEST_CHUNK_SIZE
        encoder.encode.side_effect = lambda images, quality: [
            f"jpeg-{index}".encode() for index, _ in enumerate(images)
        ]
        request = CameraRenderRequest(
            scene_id="scene",
            image_format=ImageFormat.JPEG,
            image_quality=90,
            distance_colormap_vmax_m=1.0,
            render_products=[
                CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR,
                CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED,
            ],
        )

        with mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder):
            result = service._camera_render_return_from_request(backend, request, enable_editing_actors=False)

        self.assertEqual(
            [output.image_bytes for output in result.camera_render_outputs],
            [b"jpeg-0", b"jpeg-1"],
        )
        encoder.encode.assert_called_once()
        images_arg, quality_arg = encoder.encode.call_args.args
        self.assertEqual(len(images_arg), 2)
        self.assertEqual(quality_arg, 90)

    def test_batch_render_camera_batches_same_quality_cameras_in_one_encode(self):
        service = SensorSimService(None, artifacts_glob=None, ray_chunk_size=1)
        backend = mock.MagicMock()
        backend.render_camera_request.side_effect = [
            (CameraFrame(color_image=torch.zeros(4, 4, 3)), None, None),
            (CameraFrame(color_image=torch.ones(4, 4, 3)), None, None),
        ]
        backend.source_artifact.semantic_class_names = None
        backend.source_artifact.semantic_class_colors = None
        encoder = self._recording_encoder()
        context = mock.MagicMock(spec=grpc.ServicerContext)
        context.invocation_metadata.return_value = []
        request = BatchCameraRenderRequest(
            items=[
                BatchCameraRenderRequestItem(
                    camera_name="cam_0",
                    request=CameraRenderRequest(
                        scene_id="scene",
                        image_format=ImageFormat.JPEG,
                        image_quality=90,
                        render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR],
                    ),
                ),
                BatchCameraRenderRequestItem(
                    camera_name="cam_1",
                    request=CameraRenderRequest(
                        scene_id="scene",
                        image_format=ImageFormat.JPEG,
                        image_quality=90,
                        render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR],
                    ),
                ),
            ]
        )

        backend_checked_out = False

        @contextlib.contextmanager
        def backend_context(*args, **kwargs):
            nonlocal backend_checked_out
            backend_checked_out = True
            try:
                yield backend
            finally:
                backend_checked_out = False

        original_encode = encoder.encode.side_effect

        def assert_backend_released(images, quality):
            self.assertFalse(backend_checked_out)
            return original_encode(images, quality)

        encoder.encode.side_effect = assert_backend_released

        with (
            mock.patch.object(service, "get_backend", backend_context),
            mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder),
        ):
            result = service.batch_render_camera(request, context)

        self.assertEqual([item.success for item in result.items], [True, True])
        self.assertEqual(
            [item.result.camera_render_outputs[0].image_bytes for item in result.items],
            [b"90:0", b"90:255"],
        )
        encoder.encode.assert_called_once()
        images_arg, quality_arg = encoder.encode.call_args.args
        self.assertEqual(len(images_arg), 2)
        self.assertEqual(quality_arg, 90)

    def test_batch_render_rgb_chunks_before_encoding(self):
        service = SensorSimService(None, artifacts_glob=None, ray_chunk_size=1)
        context = mock.MagicMock(spec=grpc.ServicerContext)
        context.invocation_metadata.return_value = []
        backend = mock.MagicMock()
        encoder = self._recording_encoder()
        backend_checked_out = False
        backend_acquisitions = 0

        @contextlib.contextmanager
        def backend_context(*args, **kwargs):
            nonlocal backend_checked_out, backend_acquisitions
            backend_checked_out = True
            backend_acquisitions += 1
            try:
                yield backend
            finally:
                backend_checked_out = False

        def render_image(*args, **kwargs):
            return torch.zeros((2, 2, 3), dtype=torch.uint8)

        original_encode = encoder.encode.side_effect

        def assert_bounded_encode(images, quality):
            self.assertFalse(backend_checked_out)
            self.assertLessEqual(len(images), BATCH_RENDER_REQUEST_CHUNK_SIZE)
            return original_encode(images, quality)

        encoder.encode.side_effect = assert_bounded_encode
        item_count = BATCH_RENDER_REQUEST_CHUNK_SIZE + 1
        request = BatchRGBRenderRequest(
            items=[
                BatchRGBRenderRequestItem(
                    camera_name=f"cam_{index}",
                    request=RGBRenderRequest(
                        scene_id="scene",
                        image_format=ImageFormat.JPEG,
                        image_quality=90,
                    ),
                )
                for index in range(item_count)
            ]
        )

        with (
            mock.patch.object(service, "get_backend", backend_context),
            mock.patch.object(service, "_render_single_camera_image_for_batch", side_effect=render_image),
            mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder),
        ):
            result = service.batch_render_rgb(request, context)

        self.assertEqual(len(result.items), item_count)
        self.assertTrue(all(item.success for item in result.items))
        self.assertEqual(backend_acquisitions, 2)
        self.assertEqual(
            [len(call.args[0]) for call in encoder.encode.call_args_list],
            [BATCH_RENDER_REQUEST_CHUNK_SIZE, 1],
        )

    def test_batch_render_camera_chunks_encoded_outputs_to_native_limit(self):
        service = SensorSimService(None, artifacts_glob=None, ray_chunk_size=1)
        context = mock.MagicMock(spec=grpc.ServicerContext)
        context.invocation_metadata.return_value = []
        backend = mock.MagicMock()
        encoder = self._recording_encoder()
        encode_sizes = []

        def bounded_encode(images, quality):
            encode_sizes.append(len(images))
            if len(images) > encoder.max_batch_size:
                raise RuntimeError("native batch is oversized")
            return [b"jpeg"] * len(images)

        encoder.encode.side_effect = bounded_encode

        def prepare_outputs(_backend, request, **kwargs):
            return [
                PendingCameraRenderOutput(
                    image_u8=torch.zeros((2, 2, 3), dtype=torch.uint8),
                    request=request,
                    product=product,
                    semantics_class_mapping=None,
                    distance_colormap_vmax_m=None,
                )
                for product in request.render_products
            ]

        products = [
            CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR,
            CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED,
            CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED,
        ]
        request = BatchCameraRenderRequest(
            items=[
                BatchCameraRenderRequestItem(
                    camera_name=f"cam_{index}",
                    request=CameraRenderRequest(
                        scene_id="scene",
                        image_format=ImageFormat.JPEG,
                        image_quality=90,
                        render_products=products,
                    ),
                )
                for index in range(BATCH_RENDER_REQUEST_CHUNK_SIZE)
            ]
        )

        with (
            mock.patch.object(
                service, "get_backend", side_effect=lambda *args, **kwargs: self._backend_context(backend)
            ),
            mock.patch.object(service, "_prepare_camera_render_outputs_from_request", side_effect=prepare_outputs),
            mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder),
        ):
            result = service.batch_render_camera(request, context)

        self.assertTrue(all(item.success for item in result.items))
        self.assertEqual(encode_sizes, [BATCH_RENDER_REQUEST_CHUNK_SIZE] * 3)

    def _assert_post_backend_encode_oom_runs_cleanup(self, invoke):
        service = SensorSimService(None, artifacts_glob=None, ray_chunk_size=1)
        context = mock.MagicMock(spec=grpc.ServicerContext)
        context.invocation_metadata.return_value = []
        backend = mock.MagicMock()
        encoder = self._recording_encoder()
        encoder.encode.side_effect = torch.cuda.OutOfMemoryError("nvjpeg allocation failed")

        with (
            mock.patch.object(
                service, "get_backend", side_effect=lambda *args, **kwargs: self._backend_context(backend)
            ),
            mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder),
            mock.patch("nurec.grpc.serve.torch.cuda.is_available", return_value=True),
            mock.patch("nurec.grpc.serve.torch.cuda.empty_cache") as empty_cache,
            mock.patch("nurec.grpc.serve.gc.collect") as collect,
            mock.patch.object(service.backend_cache, "evict_one_spare", side_effect=[True, False]) as evict,
            self.assertRaises(torch.cuda.OutOfMemoryError),
        ):
            invoke(service, context)

        empty_cache.assert_called_once_with()
        collect.assert_called_once_with()
        self.assertEqual(evict.call_count, 2)
        context.abort.assert_called_once()
        self.assertEqual(context.abort.call_args.args[0], grpc.StatusCode.RESOURCE_EXHAUSTED)

    def test_batch_render_rgb_post_backend_encode_oom_runs_cleanup(self):
        request = BatchRGBRenderRequest(
            items=[
                BatchRGBRenderRequestItem(
                    camera_name="cam",
                    request=RGBRenderRequest(scene_id="scene", image_format=ImageFormat.JPEG, image_quality=90),
                )
            ]
        )

        def invoke(service, context):
            with mock.patch.object(
                service,
                "_render_single_camera_image_for_batch",
                return_value=torch.zeros((2, 2, 3), dtype=torch.uint8),
            ):
                service.batch_render_rgb(request, context)

        self._assert_post_backend_encode_oom_runs_cleanup(invoke)

    def test_batch_render_camera_post_backend_encode_oom_runs_cleanup(self):
        camera_request = CameraRenderRequest(
            scene_id="scene",
            image_format=ImageFormat.JPEG,
            image_quality=90,
            render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR],
        )
        request = BatchCameraRenderRequest(
            items=[BatchCameraRenderRequestItem(camera_name="cam", request=camera_request)]
        )

        def invoke(service, context):
            pending_output = PendingCameraRenderOutput(
                image_u8=torch.zeros((2, 2, 3), dtype=torch.uint8),
                request=camera_request,
                product=CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR,
                semantics_class_mapping=None,
                distance_colormap_vmax_m=None,
            )
            with mock.patch.object(
                service,
                "_prepare_camera_render_outputs_from_request",
                return_value=[pending_output],
            ):
                service.batch_render_camera(request, context)

        self._assert_post_backend_encode_oom_runs_cleanup(invoke)

    def test_batch_jpeg_encoder_initialization_failure_propagates_once(self):
        service = SensorSimService(
            None,
            artifacts_glob=None,
            ray_chunk_size=1,
        )
        pending = [
            PendingBatchRGBEncode(
                camera_name="cam_0",
                image_u8=torch.zeros((1, 1, 3), dtype=torch.uint8),
                request=RGBRenderRequest(image_format=ImageFormat.JPEG, image_quality=95),
            ),
            PendingBatchRGBEncode(
                camera_name="cam_1",
                image_u8=torch.zeros((1, 1, 3), dtype=torch.uint8),
                request=RGBRenderRequest(image_format=ImageFormat.JPEG, image_quality=95),
            ),
        ]

        with (
            mock.patch.object(
                service,
                "_get_nvjpeg_batch_encoder",
                side_effect=RuntimeError("native failed"),
            ) as get_encoder,
            self.assertRaisesRegex(RuntimeError, "native failed"),
        ):
            service._finish_pending_batch_rgb_results(pending)

        get_encoder.assert_called_once_with()

    def test_native_encoder_uses_css444_sampling(self):
        source = Path(__file__).with_name("nvjpeg_batch_encoder.cpp").read_text()

        self.assertIn("NVJPEG_CSS_444", source)
        self.assertNotIn("NVJPEG_CSS_420", source)

    def _assert_jpeg_decodes_to(self, jpeg_bytes: bytes, expected: np.ndarray) -> None:
        # Check the framing markers first: an empty or truncated payload fails these exactly, whereas the
        # pixel comparison below is necessarily tolerance-based.
        self.assertTrue(jpeg_bytes.startswith(b"\xff\xd8"), "encoded payload has no JPEG start-of-image marker")
        self.assertTrue(jpeg_bytes.endswith(b"\xff\xd9"), "encoded payload has no JPEG end-of-image marker")

        decoded = iio.imread(io.BytesIO(jpeg_bytes))

        self.assertEqual(decoded.shape, expected.shape)
        max_error = int(np.abs(decoded.astype(np.int16) - expected.astype(np.int16)).max())
        self.assertLessEqual(max_error, 8, f"JPEG round trip differs from the source by {max_error}")

    @unittest.skipUnless(torch.cuda.is_available(), "native nvJPEG encoding requires a CUDA GPU")
    def test_native_encoder_round_trips_real_cuda_images(self):
        # Exercises the real CUDA encode path end to end; the coverage above only ever sees a mocked encoder.
        # Bitstream sizing bugs here surface as empty or truncated payloads rather than a hard failure, so the
        # returned bytes are checked rather than just their presence.
        encoder = NvjpegBatchEncoder()

        # A fresh encoder has empty size and buffer caches, so this first call covers the cold path.
        small = _smooth_rgb_image(48, 64, offset=0)
        encoded = encoder.encode([torch.from_numpy(small).cuda()], quality=100)

        self.assertEqual(len(encoded), 1)
        self._assert_jpeg_decodes_to(encoded[0], small)

        # Larger images in a larger batch reuse the cached buffers, which were sized for the smaller image.
        larger = [_smooth_rgb_image(96, 128, offset=10), _smooth_rgb_image(72, 96, offset=20)]
        encoded = encoder.encode([torch.from_numpy(image).cuda() for image in larger], quality=100)

        self.assertEqual(len(encoded), 2)
        for jpeg_bytes, expected in zip(encoded, larger):
            self._assert_jpeg_decodes_to(jpeg_bytes, expected)


def _make_handler_call_details(invocation_metadata=None, method="/nurec.Sensorsim/RenderRgb"):
    """Fake HandlerCallDetails for interceptor tests"""
    details = unittest.mock.MagicMock()
    details.method = method
    details.invocation_metadata = list(invocation_metadata or [])
    return details


class TestIsLocalSceneUri(unittest.TestCase):
    """Unit tests for is_local_scene_uri (dispatch: file:// or absolute path = local; else remote)."""

    def test_file_uri_returns_true(self):
        self.assertTrue(is_local_scene_uri("file:///data/scene.usdz"))
        self.assertTrue(is_local_scene_uri("FILE:///other/path.usdz"))

    def test_absolute_path_returns_true(self):
        self.assertTrue(is_local_scene_uri("/data/scenes/foo.usdz"))

    def test_https_uri_returns_false(self):
        self.assertFalse(is_local_scene_uri("https://example.com/scene.usdz"))

    def test_http_uri_returns_false(self):
        self.assertFalse(is_local_scene_uri("http://example.com/scene.usdz"))

    def test_bare_hostname_or_relative_returns_false(self):
        """Scheme-less hostnames and relative paths are not treated as local (routed to download)."""
        self.assertFalse(is_local_scene_uri("urm.nvidia.com/path/to/scene.usdz"))
        self.assertFalse(is_local_scene_uri("scenes/foo.usdz"))


class TestSceneCacheKeysGrpc(unittest.TestCase):
    """End-to-end coverage for artifact identity through the gRPC interceptor."""

    @contextlib.contextmanager
    def _start_scene_server(self, url_to_bytes: dict[str, bytes]):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            downloader = mock.Mock()

            def download(uri: str, destination: Path) -> int:
                blob = url_to_bytes[uri]
                destination.write_bytes(blob)
                return len(blob)

            downloader.download.side_effect = download
            interceptor = SceneDownloadInterceptor()
            server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=1),
                interceptors=(interceptor,),
            )
            service = SensorSimService(
                server=server,
                artifacts_glob=None,
                ray_chunk_size=1,
                downloader=downloader,
                scene_cache=SceneCache(cache_dir, max_size=2),
                cache_size=2,
                max_backends_per_scene=1,
                server_config=GrpcServerConfig(cache_size=2, max_backends_per_scene=1),
            )
            interceptor.set_service(service)
            add_SensorsimServiceServicer_to_server(service, server)
            port = server.add_insecure_port("localhost:0")
            server.start()
            channel = grpc.insecure_channel(f"localhost:{port}")
            try:
                grpc.channel_ready_future(channel).result(timeout=5)
                yield SensorsimServiceStub(channel), downloader, cache_dir
            finally:
                channel.close()
                server.stop(0)

    @staticmethod
    def _metadata(scene_url: str):
        return [
            (SCENE_URL_HEADER, scene_url),
            (SCENE_ID_HEADER, "scene"),
        ]

    def test_scene_url_identity_is_preserved_through_grpc(self):
        first_url = "https://example.com/scene.usdz?X-Amz-Date=20260722T000000Z&X-Amz-Signature=one"
        rotated_url = "https://example.com/scene.usdz?X-Amz-Date=20260723T000000Z&X-Amz-Signature=two"
        other_url = "https://example.com/other.usdz"
        scene_bytes = _artifact_bytes("scene")
        other_bytes = _artifact_bytes("scene", b"other")

        url_to_bytes = {first_url: scene_bytes, rotated_url: scene_bytes, other_url: other_bytes}
        with self._start_scene_server(url_to_bytes) as (stub, downloader, cache_dir):
            for scene_url in url_to_bytes:
                stub.get_server_config(Empty(), metadata=self._metadata(scene_url))

            self.assertEqual(downloader.download.call_count, 2)
            first_key = SceneCacheKey.from_url("scene", first_url).key_hash
            other_key = SceneCacheKey.from_url("scene", other_url).key_hash
            self.assertEqual((cache_dir / f"{first_key}.usdz").read_bytes(), scene_bytes)
            self.assertEqual((cache_dir / f"{other_key}.usdz").read_bytes(), other_bytes)


class TestSceneDownloadInterceptor(unittest.TestCase):
    """Unit tests for SceneDownloadInterceptor: x-nre-scene-url / x-nre-scene-id dispatch and errors."""

    def setUp(self):
        self.interceptor = SceneDownloadInterceptor()
        self.service = unittest.mock.MagicMock(spec=SensorSimService)
        self.interceptor.set_service(self.service)

    @staticmethod
    def _handler():
        callback = unittest.mock.Mock(return_value=unittest.mock.sentinel.response)
        request_deserializer = unittest.mock.Mock()
        response_serializer = unittest.mock.Mock()
        return (
            grpc.unary_unary_rpc_method_handler(
                callback,
                request_deserializer=request_deserializer,
                response_serializer=response_serializer,
            ),
            callback,
        )

    @staticmethod
    def _context(handler_call_details):
        context = unittest.mock.Mock()
        context.invocation_metadata.return_value = handler_call_details.invocation_metadata
        context.is_active.return_value = True
        context.time_remaining.return_value = None
        return context

    @staticmethod
    def _download_service(cache_dir: Path, max_size: int = 1) -> SensorSimService:
        downloader = unittest.mock.Mock()

        def download(_uri: str, destination: Path) -> int:
            stem = Path(_uri.partition("?")[0]).stem
            scene_id = stem if stem in {"scene", "other"} or stem.startswith("scene-") else f"scene-{stem}"
            blob = _artifact_bytes(scene_id)
            destination.write_bytes(blob)
            return len(blob)

        downloader.download.side_effect = download
        return SensorSimService(
            server=None,
            artifacts_glob=None,
            ray_chunk_size=1,
            downloader=downloader,
            scene_cache=SceneCache(cache_dir, max_size=max_size),
            cache_size=2,
            max_backends_per_scene=1,
            server_config=GrpcServerConfig(cache_size=2, max_backends_per_scene=1),
        )

    def test_no_x_nurec_scene_url_calls_continuation(self):
        details = _make_handler_call_details(invocation_metadata=[])
        continuation = unittest.mock.Mock(return_value=unittest.mock.sentinel.handler)
        result = self.interceptor.intercept_service(continuation, details)
        continuation.assert_called_once_with(details)
        self.assertIs(result, unittest.mock.sentinel.handler)

    def test_prefetch_defers_local_registration_to_worker(self):
        for service in ("nurec.grpc.protos.sensorsim.SensorsimService", LEGACY_SENSORSIM_SERVICE_NAME):
            with self.subTest(service=service):
                details = _make_handler_call_details(
                    method=f"/{service}/prefetch_scene",
                    invocation_metadata=[
                        (SCENE_URL_HEADER, "file:///data/scene.usdz"),
                        (SCENE_ID_HEADER, "scene"),
                    ],
                )
                handler = unittest.mock.sentinel.handler
                continuation = unittest.mock.Mock(return_value=handler)

                result = self.interceptor.intercept_service(continuation, details)

                self.assertIs(result, handler)
                self.service._register_local_scene.assert_not_called()

    def test_file_uri_calls_register_local_scene(self):
        details = _make_handler_call_details(
            invocation_metadata=[
                (SCENE_URL_HEADER, "file:///data/scene.usdz"),
                (SCENE_ID_HEADER, "scene_1"),
            ]
        )
        underlying, callback = self._handler()
        continuation = unittest.mock.Mock(return_value=underlying)
        result = self.interceptor.intercept_service(continuation, details)

        self.service._register_local_scene.assert_not_called()
        response = result.unary_unary(unittest.mock.sentinel.request, unittest.mock.Mock())

        self.service._register_local_scene.assert_called_once_with(
            SceneCacheKey.from_url("scene_1", "file:///data/scene.usdz"),
            "file:///data/scene.usdz",
        )
        continuation.assert_called_once_with(details)
        callback.assert_called_once()
        self.assertIs(response, unittest.mock.sentinel.response)
        self.assertIs(result.request_deserializer, underlying.request_deserializer)
        self.assertIs(result.response_serializer, underlying.response_serializer)

    def test_bare_path_calls_register_local_scene(self):
        details = _make_handler_call_details(
            invocation_metadata=[
                (SCENE_URL_HEADER, "/data/scenes/foo.usdz"),
                (SCENE_ID_HEADER, "scene_2"),
            ]
        )
        underlying, callback = self._handler()
        continuation = unittest.mock.Mock(return_value=underlying)
        result = self.interceptor.intercept_service(continuation, details)

        self.service._register_local_scene.assert_not_called()
        result.unary_unary(unittest.mock.sentinel.request, unittest.mock.Mock())

        self.service._register_local_scene.assert_called_once_with(
            SceneCacheKey.from_url("scene_2", "/data/scenes/foo.usdz"),
            "/data/scenes/foo.usdz",
        )
        continuation.assert_called_once_with(details)
        callback.assert_called_once()

    def test_https_uri_calls_download_scene(self):
        details = _make_handler_call_details(
            invocation_metadata=[
                (SCENE_URL_HEADER, "https://example.com/scene.usdz"),
                (SCENE_ID_HEADER, "scene_2"),
            ]
        )
        lease = unittest.mock.MagicMock()
        self.service._download_scene.return_value = lease
        underlying, callback = self._handler()
        continuation = unittest.mock.Mock(return_value=underlying)
        result = self.interceptor.intercept_service(continuation, details)

        self.service._download_scene.assert_not_called()
        result.unary_unary(unittest.mock.sentinel.request, unittest.mock.Mock())

        self.service._download_scene.assert_called_once_with(
            SceneCacheKey.from_url("scene_2", "https://example.com/scene.usdz"),
            "https://example.com/scene.usdz",
        )
        self.service._release_scene_lease.assert_called_once_with(lease)
        continuation.assert_called_once_with(details)
        callback.assert_called_once()

    def test_request_scene_id_must_match_metadata_scene_id(self):
        for url in (None, "https://example.com/scene.usdz"):
            with self.subTest(url=url):
                metadata = [(SCENE_ID_HEADER, "metadata-scene")]
                if url is not None:
                    metadata.append((SCENE_URL_HEADER, url))
                details = _make_handler_call_details(invocation_metadata=metadata)
                with self.assertRaisesRegex(SceneNotFoundError, "does not match metadata scene"):
                    SensorSimService._request_scene_cache_key("request-scene", self._context(details))

    def test_matching_scene_metadata_preserves_the_key(self):
        for url in (None, "https://example.com/scene.usdz"):
            with self.subTest(url=url):
                metadata = [(SCENE_ID_HEADER, "scene")]
                expected = SceneCacheKey(scene_id="scene")
                if url is not None:
                    metadata.append((SCENE_URL_HEADER, url))
                    expected = SceneCacheKey.from_url("scene", url)
                details = _make_handler_call_details(invocation_metadata=metadata)
                self.assertEqual(SensorSimService._request_scene_cache_key("scene", self._context(details)), expected)

    def test_scene_url_requires_nonempty_id_metadata(self):
        for scene_id in (None, ""):
            with self.subTest(scene_id=scene_id):
                metadata = [(SCENE_URL_HEADER, "https://example.com/scene.usdz")]
                if scene_id is not None:
                    metadata.append((SCENE_ID_HEADER, scene_id))
                details = _make_handler_call_details(invocation_metadata=metadata)
                with self.assertRaisesRegex(ValueError, "Missing required metadata"):
                    SensorSimService._request_scene_cache_key("scene", self._context(details))

    def test_url_less_lookup_rejects_multiple_registrations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._download_service(Path(temp_dir))
            static_key = SceneCacheKey(scene_id="scene")
            url_key = SceneCacheKey.from_url("scene", "https://example.com/scene.usdz")
            with service._artifact_registry_lock:
                service._artifact_registrations[static_key.key_hash] = ArtifactRegistration(
                    _registered_artifact(Path(temp_dir) / "static.usdz", "scene"),
                    False,
                    static_key,
                )
                service._artifact_registrations[url_key.key_hash] = ArtifactRegistration(
                    _registered_artifact(Path(temp_dir) / "remote.usdz", "scene"),
                    True,
                    url_key,
                )

            with self.assertRaisesRegex(SceneNotFoundError, "multiple registrations"):
                with service.get_backend(SceneCacheKey(scene_id="scene")):
                    pass

    def test_startup_artifact_mutation_is_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene.usdz"
            scene_path.write_bytes(_artifact_bytes("scene", b"scene-v1"))
            service = SensorSimService(
                server=None,
                artifacts_glob=str(scene_path),
                ray_chunk_size=1,
                cache_size=1,
                server_config=GrpcServerConfig(cache_size=1),
            )
            scene_path.write_bytes(_artifact_bytes("scene", b"expanded-scene-v2"))

            with mock.patch.object(Backend, "from_artifact") as from_artifact:
                with self.assertRaisesRegex(ValueError, "changed after registration"):
                    with service.get_backend(SceneCacheKey(scene_id="scene")):
                        pass

            from_artifact.assert_not_called()

    def test_untouched_startup_artifact_is_not_rehashed_on_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene.usdz"
            scene_path.write_bytes(_artifact_bytes("scene", b"scene-v1"))
            service = SensorSimService(
                server=None,
                artifacts_glob=str(scene_path),
                ray_chunk_size=1,
                cache_size=1,
                server_config=GrpcServerConfig(cache_size=1),
            )
            registration = service._artifact_registrations[SceneCacheKey(scene_id="scene").key_hash]
            # Startup hashed the file right after it was written, inside the racy-clean window of
            # Artifact.footprint_is_current; hash it again as if a minute had passed so the digest is
            # trusted on the file's stat (ctime cannot be back-dated the way mtime can).
            registration.artifact.clear_cache()
            with mock.patch("time.time", return_value=time.time() + 60.0):
                registration.artifact.sha256_footprint
            backend = TestRenderGrpcStatusCodes._make_backend(registration.backend_cache_key)

            real_footprint = Artifact.sha256_footprint

            def cached_footprint_only(artifact: Artifact) -> str:
                # The registration's artifact answers from its cache; a fresh Artifact would read the file.
                if artifact._sha256_footprint is None:
                    raise AssertionError("re-hashed")
                return real_footprint.fget(artifact)

            with (
                mock.patch.object(Artifact, "sha256_footprint", property(cached_footprint_only)),
                mock.patch.object(Backend, "from_artifact", return_value=backend) as from_artifact,
            ):
                with service.get_backend(SceneCacheKey(scene_id="scene")) as served:
                    self.assertIs(served, backend)

            from_artifact.assert_called_once()

    def test_startup_artifact_mutation_during_load_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene.usdz"
            scene_path.write_bytes(_artifact_bytes("scene", b"scene-v1"))
            service = SensorSimService(
                server=None,
                artifacts_glob=str(scene_path),
                ray_chunk_size=1,
                cache_size=1,
                server_config=GrpcServerConfig(cache_size=1),
            )

            def mutate_artifact(*_args, **_kwargs):
                scene_path.write_bytes(_artifact_bytes("scene", b"expanded-scene-v2"))
                return mock.MagicMock()

            with mock.patch.object(Backend, "from_artifact", side_effect=mutate_artifact):
                with self.assertRaisesRegex(ValueError, "changed after registration"):
                    with service.get_backend(SceneCacheKey(scene_id="scene")):
                        pass

    def test_redownloaded_content_does_not_reuse_backend_from_evicted_disk_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._download_service(Path(temp_dir), max_size=1)
            scene_url = "https://example.com/scene.usdz"
            eviction_url = "https://example.com/other.usdz"
            scene_blob = [b"scene-v1"]

            def download(uri: str, destination: Path) -> int:
                blob = b"other-scene" if uri == eviction_url else scene_blob[0]
                destination.write_bytes(blob)
                return len(blob)

            service.downloader.download.side_effect = download
            scene_key = SceneCacheKey.from_url("scene", scene_url)
            eviction_key = SceneCacheKey.from_url("other", eviction_url)
            backend_v1 = TestRenderGrpcStatusCodes._make_backend(
                BackendCacheKey(scene_key, hashlib.sha256(b"scene-v1").hexdigest())
            )
            backend_v2 = TestRenderGrpcStatusCodes._make_backend(
                BackendCacheKey(scene_key, hashlib.sha256(b"scene-v2").hexdigest())
            )

            with mock.patch.object(Backend, "from_artifact", side_effect=[backend_v1, backend_v2]) as factory:
                scene_lease = service._download_scene(scene_key, scene_url)
                assert scene_lease is not None
                service._release_scene_lease(scene_lease)
                with service.get_backend(scene_key) as backend:
                    self.assertIs(backend, backend_v1)

                eviction_lease = service._download_scene(eviction_key, eviction_url)
                assert eviction_lease is not None
                service._release_scene_lease(eviction_lease)
                scene_blob[0] = b"scene-v2"
                new_scene_lease = service._download_scene(scene_key, scene_url)
                assert new_scene_lease is not None
                service._release_scene_lease(new_scene_lease)
                with service.get_backend(scene_key) as backend:
                    self.assertIs(backend, backend_v2)

            self.assertEqual(factory.call_count, 2)
            self.assertNotEqual(backend_v1.cache_key, backend_v2.cache_key)

    def test_invalid_scene_id_aborts_with_invalid_argument(self):
        details = _make_handler_call_details(
            invocation_metadata=[
                (SCENE_URL_HEADER, "file:///data/scene.usdz"),
                (SCENE_ID_HEADER, "../invalid"),
            ]
        )
        underlying, _ = self._handler()
        continuation = unittest.mock.Mock(return_value=underlying)
        result = self.interceptor.intercept_service(continuation, details)
        self.service._register_local_scene.assert_not_called()
        self.service._download_scene.assert_not_called()
        continuation.assert_called_once_with(details)
        self.assertIsNot(result, underlying)
        # Returned handler should be a terminator that aborts with INVALID_ARGUMENT
        handler = result
        self.assertIsNotNone(handler.unary_unary)
        context = unittest.mock.Mock()
        handler.unary_unary(None, context)
        context.abort.assert_called_once()
        call_args = context.abort.call_args[0]
        self.assertEqual(call_args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("Invalid scene ID", call_args[1])

    def test_register_local_scene_raises_aborts_with_invalid_argument(self):
        self.service._register_local_scene.side_effect = ValueError("path does not exist")
        details = _make_handler_call_details(
            invocation_metadata=[
                (SCENE_URL_HEADER, "file:///nonexistent.usdz"),
                (SCENE_ID_HEADER, "scene_1"),
            ]
        )
        underlying, callback = self._handler()
        continuation = unittest.mock.Mock(return_value=underlying)
        result = self.interceptor.intercept_service(continuation, details)
        context = unittest.mock.Mock()
        context.abort.side_effect = RuntimeError("aborted")
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            result.unary_unary(None, context)
        context.abort.assert_called_once()
        callback.assert_not_called()
        call_args = context.abort.call_args[0]
        self.assertEqual(call_args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("Failed to register local scene", call_args[1])
        self.assertIn("path does not exist", call_args[1])

    def test_download_scene_raises_aborts_with_invalid_argument(self):
        self.service._download_scene.side_effect = RuntimeError("network error")
        details = _make_handler_call_details(
            invocation_metadata=[
                (SCENE_URL_HEADER, "https://example.com/scene.usdz"),
                (SCENE_ID_HEADER, "scene_1"),
            ]
        )
        underlying, callback = self._handler()
        continuation = unittest.mock.Mock(return_value=underlying)
        result = self.interceptor.intercept_service(continuation, details)
        context = unittest.mock.Mock()
        context.abort.side_effect = RuntimeError("aborted")
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            result.unary_unary(None, context)
        context.abort.assert_called_once()
        callback.assert_not_called()
        call_args = context.abort.call_args[0]
        self.assertEqual(call_args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("Failed to download scene", call_args[1])
        self.assertIn("network error", call_args[1])

    def test_request_lease_spans_handler_backend_acquisition(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._download_service(Path(temp_dir))
            interceptor = SceneDownloadInterceptor()
            interceptor.set_service(service)
            handler_entered = threading.Event()
            scene_b_complete = threading.Event()
            backend_acquired = threading.Event()
            backend = TestRenderGrpcStatusCodes._make_backend(_test_backend_cache_key("scene-a"))
            scene_a_url = "https://example.com/a.usdz"
            scene_b_url = "https://example.com/b.usdz"
            scene_a_path = Path(temp_dir) / f"{SceneCacheKey.from_url('scene-a', scene_a_url).key_hash}.usdz"
            scene_b_path = Path(temp_dir) / f"{SceneCacheKey.from_url('scene-b', scene_b_url).key_hash}.usdz"

            def handle_scene_a(_request, context):
                handler_entered.set()
                self.assertTrue(scene_b_complete.wait(timeout=5))
                self.assertTrue(scene_a_path.exists())
                with service.get_backend(service._request_scene_cache_key("scene-a", context), context) as acquired:
                    self.assertIs(acquired, backend)
                    backend_acquired.set()
                return unittest.mock.sentinel.scene_a_response

            scene_a_handler = grpc.unary_unary_rpc_method_handler(handle_scene_a)
            scene_b_handler = grpc.unary_unary_rpc_method_handler(
                lambda _request, _context: unittest.mock.sentinel.scene_b_response
            )
            scene_a_details = _make_handler_call_details(
                invocation_metadata=[
                    (SCENE_URL_HEADER, scene_a_url),
                    (SCENE_ID_HEADER, "scene-a"),
                ]
            )
            scene_b_details = _make_handler_call_details(
                invocation_metadata=[
                    (SCENE_URL_HEADER, scene_b_url),
                    (SCENE_ID_HEADER, "scene-b"),
                ]
            )
            wrapped_a = interceptor.intercept_service(lambda _details: scene_a_handler, scene_a_details)
            wrapped_b = interceptor.intercept_service(lambda _details: scene_b_handler, scene_b_details)
            scene_a_context = self._context(scene_a_details)
            scene_b_context = self._context(scene_b_details)

            with (
                mock.patch.object(Backend, "from_artifact", return_value=backend),
                futures.ThreadPoolExecutor(max_workers=2) as executor,
            ):
                scene_a_call = executor.submit(wrapped_a.unary_unary, None, scene_a_context)
                self.assertTrue(handler_entered.wait(timeout=5))
                wrapped_b.unary_unary(None, scene_b_context)
                self.assertTrue(scene_a_path.exists())
                self.assertFalse(scene_b_path.exists())
                scene_b_complete.set()
                self.assertIs(scene_a_call.result(timeout=5), unittest.mock.sentinel.scene_a_response)

            self.assertTrue(backend_acquired.is_set())
            self.assertTrue(scene_a_path.exists())
            self.assertFalse(scene_b_path.exists())
            self.assertEqual(set(service._artifact_snapshot()), {"scene-a"})

            scene_c_url = "https://example.com/c.usdz"
            scene_c_lease = service._download_scene(
                SceneCacheKey.from_url("scene-c", scene_c_url),
                scene_c_url,
            )
            assert scene_c_lease is not None
            self.assertFalse(scene_a_path.exists())
            self.assertEqual(set(service._artifact_snapshot()), {"scene-c"})
            service._release_scene_lease(scene_c_lease)

    def test_downloaded_eviction_preserves_static_and_local_registrations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            static_path = root / "static.usdz"
            local_path = root / "local.usdz"
            static_path.write_bytes(_artifact_bytes("static"))
            local_path.write_bytes(_artifact_bytes("local"))
            service = self._download_service(root)
            static_scene_key = SceneCacheKey(scene_id="static")
            with service._artifact_registry_lock:
                service._artifact_registrations[static_scene_key.key_hash] = ArtifactRegistration(
                    Artifact(static_path),
                    False,
                    static_scene_key,
                )
            service._register_local_scene(
                SceneCacheKey.from_url("local", str(local_path)),
                str(local_path),
            )

            scene_a_url = "https://example.com/a.usdz"
            scene_a_lease = service._download_scene(
                SceneCacheKey.from_url("scene-a", scene_a_url),
                scene_a_url,
            )
            assert scene_a_lease is not None
            service._release_scene_lease(scene_a_lease)
            scene_b_url = "https://example.com/b.usdz"
            scene_b_lease = service._download_scene(
                SceneCacheKey.from_url("scene-b", scene_b_url),
                scene_b_url,
            )
            assert scene_b_lease is not None

            self.assertEqual(set(service._artifact_snapshot()), {"static", "local", "scene-b"})
            self.assertTrue(static_path.exists())
            self.assertTrue(local_path.exists())
            service._release_scene_lease(scene_b_lease)

    def test_download_publication_failure_discards_unregistered_file(self):
        class RejectingRegistry(dict):
            def __setitem__(self, key, value):
                raise RuntimeError("publication failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._download_service(Path(temp_dir))
            service._artifact_registrations = RejectingRegistry(service._artifact_registrations)
            scene_key = SceneCacheKey.from_url("scene", "https://example.com/scene.usdz")

            with self.assertRaisesRegex(RuntimeError, "publication failed"):
                service._download_scene(scene_key, "https://example.com/scene.usdz")

            self.assertFalse(service.scene_cache.has_scene(scene_key.key_hash))
            self.assertFalse((Path(temp_dir) / f"{scene_key.key_hash}.usdz").exists())
            self.assertNotIn("scene", service._artifact_snapshot())

    def test_registry_read_and_competing_insert_wait_for_atomic_publication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = self._download_service(Path(temp_dir))
            scene_a_inserted = threading.Event()
            allow_scene_a_publication = threading.Event()
            scene_b_attempted = threading.Event()
            reader_attempted = threading.Event()
            original_add = service.scene_cache.add_and_acquire_scene
            scene_a_key = SceneCacheKey.from_url("scene-a", "https://example.com/a.usdz").key_hash
            scene_a_path = Path(temp_dir) / f"{scene_a_key}.usdz"

            def add_and_pause(registration_key, temp_path):
                result = original_add(registration_key, temp_path)
                if registration_key == scene_a_key:
                    scene_a_inserted.set()
                    self.assertTrue(allow_scene_a_publication.wait(timeout=5))
                return result

            def download_scene_b():
                scene_b_attempted.set()
                scene_b_url = "https://example.com/b.usdz"
                return service._download_scene(
                    SceneCacheKey.from_url("scene-b", scene_b_url),
                    scene_b_url,
                )

            def read_registry():
                reader_attempted.set()
                with service._artifact_registry_lock:
                    return dict(service._artifact_registrations)

            try:
                with (
                    mock.patch.object(service.scene_cache, "add_and_acquire_scene", side_effect=add_and_pause),
                    futures.ThreadPoolExecutor(max_workers=3) as executor,
                ):
                    scene_a_url = "https://example.com/a.usdz"
                    scene_a_call = executor.submit(
                        service._download_scene,
                        SceneCacheKey.from_url("scene-a", scene_a_url),
                        scene_a_url,
                    )
                    self.assertTrue(scene_a_inserted.wait(timeout=5))
                    self.assertTrue(scene_a_path.exists())

                    scene_b_call = executor.submit(download_scene_b)
                    reader_call = executor.submit(read_registry)
                    self.assertTrue(scene_b_attempted.wait(timeout=5))
                    self.assertTrue(reader_attempted.wait(timeout=5))
                    self.assertFalse(scene_b_call.done())
                    self.assertFalse(reader_call.done())

                    allow_scene_a_publication.set()
                    scene_a_lease = scene_a_call.result(timeout=5)
                    registrations = reader_call.result(timeout=5)
                    scene_b_lease = scene_b_call.result(timeout=5)

                assert scene_a_lease is not None
                assert scene_b_lease is not None
                self.assertIn(scene_a_key, registrations)
                self.assertTrue(registrations[scene_a_key].downloaded)
                self.assertTrue(scene_a_path.exists())
                service._release_scene_lease(scene_b_lease)
                service._release_scene_lease(scene_a_lease)
            finally:
                allow_scene_a_publication.set()


def _make_ftheta_params(resolution_w=320, resolution_h=567):
    """Create minimal FThetaCameraModelParameters for testing."""
    return ncore.data.FThetaCameraModelParameters(
        resolution=np.array([resolution_w, resolution_h], dtype=np.uint64),
        principal_point=np.array([160.0, 283.5], dtype=np.float32),
        reference_poly=1,
        pixeldist_to_angle_poly=np.array([0.0, 0.005, 0.0, 0.0, 0.0], dtype=np.float32),
        angle_to_pixeldist_poly=np.array([0.0, 200.0, 0.0, 0.0, 0.0], dtype=np.float32),
        max_angle=1.2,
        linear_cde=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        shutter_type=5,  # GLOBAL
    )


def _make_torch_camera(logical_id="camera_front_wide_120fov", unique_sensor_idx=0):
    """Create a TorchCamera for testing CameraBank._get_camera fallback."""
    return TorchCamera(
        time_range_us=HalfClosedInterval(1000000, 2000000),
        camera_model_parameters=_make_ftheta_params(),
        logical_camera_id=logical_id,
        unique_sensor_idx=unique_sensor_idx,
        world_to_nurec=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
        T_camera_rig=torch.eye(4),
    )


def test_torch_camera_preserves_uniform_scale_for_rounded_resolution() -> None:
    camera_parameters = ncore.data.OpenCVPinholeCameraModelParameters(
        resolution=np.array([3840, 2160], dtype=np.uint64),
        focal_length=np.array([3840.0, 2160.0], dtype=np.float32),
        principal_point=np.array([1920.0, 1080.0], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        shutter_type=ncore.data.ShutterType.GLOBAL,
    )
    camera = TorchCamera(
        time_range_us=HalfClosedInterval(1_000_000, 2_000_000),
        camera_model_parameters=camera_parameters,
        logical_camera_id="camera_front_wide_120fov",
        unique_sensor_idx=0,
        world_to_nurec=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
        T_camera_rig=torch.eye(4),
    )
    request = RGBRenderRequest(
        resolution_w=1000,
        resolution_h=1,
        frame_start_us=1_000_000,
        frame_end_us=1_000_001,
    )
    request.sensor_pose.start_pose.quat.w = 1.0
    request.sensor_pose.end_pose.quat.w = 1.0

    with mock.patch("nurec.grpc.serve.RayBundle.build") as build:
        camera.build_ray_bundle(request)

    transformed = build.call_args.kwargs["camera_model_parameters"]
    np.testing.assert_array_equal(transformed.resolution, [1000, 562])
    np.testing.assert_allclose(transformed.focal_length, [1000.0, 562.5])
    np.testing.assert_allclose(transformed.principal_point, [500.0, 281.25])


def _make_custom_ftheta_spec(logical_id):
    return CameraSpec(
        logical_id=logical_id,
        resolution_h=480,
        resolution_w=640,
        shutter_type="ROLLING_RIGHT_TO_LEFT",
        ftheta_param=FthetaCameraParam(
            principal_point_x=321.5,
            principal_point_y=240.5,
            reference_poly="ANGLE_TO_PIXELDIST",
            pixeldist_to_angle_poly=[0.1, 0.006, 0.002, 0.003, 0.004, 0.005],
            angle_to_pixeldist_poly=[0.2, 210.0, 0.5, 0.6, 0.7, 0.8],
            max_angle=1.5,
            linear_cde=LinearCde(linear_c=1.1, linear_d=0.2, linear_e=0.9),
        ),
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
def test_global_camera_ray_batch_matches_individual_bundles() -> None:
    camera = _make_torch_camera(unique_sensor_idx=7)
    camera.world_to_nurec = FrameConversion(
        matrix=np.array([[0, -1, 0, 3], [1, 0, 0, -2], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.float32)
    )
    pose_a = Pose(quat={"w": 1.0})
    pose_b = Pose(vec={"x": 2.0}, quat={"z": math.sqrt(0.5), "w": math.sqrt(0.5)})

    def request(start_pose: Pose, end_pose: Pose) -> CameraRenderRequest:
        return CameraRenderRequest(
            resolution_h=567,
            resolution_w=320,
            frame_start_us=1_000_010,
            frame_end_us=1_000_011,
            sensor_pose=PosePair(start_pose=start_pose, end_pose=end_pose),
        )

    requests = [request(pose_a, pose_b), request(pose_b, pose_a)]
    individual = [camera.build_ray_bundle(item) for item in requests]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        batched, frame_meta = camera.build_ray_batch(requests, torch.device("cuda"))
    torch.cuda.current_stream().wait_stream(stream)

    for actual, bundle in zip(frame_meta, individual, strict=True):
        assert actual.unique_sensor_idx == bundle.frame_meta.unique_sensor_idx
        assert actual.unique_frame_idx == bundle.frame_meta.unique_frame_idx
        assert actual.training_frame_idx == bundle.frame_meta.training_frame_idx
    expected = RenderingData.collate_fn([bundle.rendering_data for bundle in individual], device=torch.device("cuda"))
    torch.testing.assert_close(batched.rays, expected.rays)
    torch.testing.assert_close(batched.poses_tquat_startend, expected.poses_tquat_startend)
    torch.testing.assert_close(batched.timestamps_startend_us, expected.timestamps_startend_us)
    torch.testing.assert_close(batched.timestamps_startend_us_cpu, expected.timestamps_startend_us_cpu)
    torch.testing.assert_close(batched.rays_timestamps_us, expected.rays_timestamps_us)
    assert batched.rays_timestamps_us.stride()[1:3] == (0, 0)
    assert batched.rays_timestamp_layout == "global"
    assert all(
        actual is reference
        for actual, reference in zip(batched.sensor_model_parameters, expected.sensor_model_parameters)
    )
    for name in ("_rays_footprints", "_distance_to_depth_scale", "_rays_is_sky", "ray_flags", "active_tile_mask"):
        assert getattr(batched, name) is None
        assert getattr(expected, name) is None


def _enum_value(value):
    return value.value if hasattr(value, "value") else value


class TestGetCameraFallback(unittest.TestCase):
    """Tests for CameraBank._get_camera allowing unknown logical_ids when camera_param is provided."""

    def _make_camera_bank_with_cameras(self, cameras_dict):
        """Create a CameraBank without calling __init__ (which requires Artifact), then set cameras."""
        bank = object.__new__(CameraBank)
        bank.cameras = cameras_dict
        bank._custom_camera = None
        bank._logged_missing_reconstruction_cameras = set()
        bank._logged_missing_reconstruction_cameras_lock = threading.Lock()
        return bank

    def _assert_ftheta_camera_matches_spec(self, camera, spec):
        params = camera.camera_model_parameters
        self.assertIsInstance(params, ncore.data.FThetaCameraModelParameters)
        np.testing.assert_array_equal(params.resolution, np.array([spec.resolution_w, spec.resolution_h]))
        self.assertEqual(_enum_value(params.shutter_type), spec.shutter_type)

        ftheta_param = spec.ftheta_param
        np.testing.assert_allclose(
            params.principal_point,
            np.array([ftheta_param.principal_point_x, ftheta_param.principal_point_y]),
        )
        self.assertEqual(_enum_value(params.reference_poly), ftheta_param.reference_poly)
        np.testing.assert_allclose(params.pixeldist_to_angle_poly, ftheta_param.pixeldist_to_angle_poly)
        np.testing.assert_allclose(params.angle_to_pixeldist_poly, ftheta_param.angle_to_pixeldist_poly)
        self.assertEqual(params.max_angle, ftheta_param.max_angle)
        np.testing.assert_allclose(
            params.linear_cde,
            np.array(
                [
                    ftheta_param.linear_cde.linear_c,
                    ftheta_param.linear_cde.linear_d,
                    ftheta_param.linear_cde.linear_e,
                ]
            ),
        )

    def _assert_ftheta_camera_differs_from_ref(self, camera, ref):
        params = camera.camera_model_parameters
        ref_params = ref.camera_model_parameters
        self.assertIsInstance(params, ncore.data.FThetaCameraModelParameters)
        self.assertIsInstance(ref_params, ncore.data.FThetaCameraModelParameters)
        self.assertFalse(np.array_equal(params.resolution, ref_params.resolution))
        self.assertNotEqual(_enum_value(params.shutter_type), _enum_value(ref_params.shutter_type))
        self.assertFalse(np.allclose(params.principal_point, ref_params.principal_point))
        self.assertNotEqual(_enum_value(params.reference_poly), _enum_value(ref_params.reference_poly))
        self.assertFalse(np.allclose(params.pixeldist_to_angle_poly, ref_params.pixeldist_to_angle_poly))
        self.assertFalse(np.allclose(params.angle_to_pixeldist_poly, ref_params.angle_to_pixeldist_poly))
        self.assertNotEqual(params.max_angle, ref_params.max_angle)
        self.assertFalse(np.allclose(params.linear_cde, ref_params.linear_cde))

    def test_known_camera_with_params_returns_customized_camera(self):
        """When logical_id IS in the bank and camera_param is provided, it should return a TorchCamera
        with the provided camera params but the bank camera's metadata."""
        ref = _make_torch_camera("cam_front", unique_sensor_idx=42)
        bank = self._make_camera_bank_with_cameras({"cam_front": ref})

        spec = _make_custom_ftheta_spec("cam_front")
        result = bank._get_camera(spec)
        self.assertEqual(result.logical_camera_id, "cam_front")
        self.assertEqual(result.unique_sensor_idx, 42)
        self._assert_ftheta_camera_matches_spec(result, spec)
        self._assert_ftheta_camera_differs_from_ref(result, ref)

    def test_custom_camera_cache_reuses_only_matching_specs(self):
        ref = _make_torch_camera("cam_front")
        bank = self._make_camera_bank_with_cameras({"cam_front": ref})
        spec = _make_custom_ftheta_spec("cam_front")

        first = bank._get_camera(spec)
        self.assertIs(bank._get_camera(spec), first)

        changed = CameraSpec()
        changed.CopyFrom(spec)
        changed.ftheta_param.principal_point_x += 1.0
        changed_camera = bank._get_camera(changed)
        self.assertIsNot(changed_camera, first)
        self._assert_ftheta_camera_matches_spec(changed_camera, changed)

    def test_unknown_camera_with_params_falls_back_to_scene_defaults(self):
        """When logical_id is NOT in the bank but camera_param IS provided,
        it should succeed using scene-level defaults from an existing camera."""
        ref = _make_torch_camera("cam_front", unique_sensor_idx=7)
        bank = self._make_camera_bank_with_cameras({"cam_front": ref})

        spec = _make_custom_ftheta_spec("camera_rear_fisheye")  # NOT in bank
        result = bank._get_camera(spec)
        # Should get a camera with the requested logical_id
        self.assertEqual(result.logical_camera_id, "camera_rear_fisheye")
        # Scene-level defaults should come from the reference camera
        self.assertEqual(result.time_range_us, ref.time_range_us)
        self._assert_ftheta_camera_matches_spec(result, spec)
        self._assert_ftheta_camera_differs_from_ref(result, ref)

    def test_unknown_camera_fallback_logs_once_per_camera(self):
        """Repeated fallback for the same logical_id should only log once."""
        ref = _make_torch_camera("cam_front", unique_sensor_idx=7)
        bank = self._make_camera_bank_with_cameras({"cam_front": ref})

        spec = CameraSpec(
            logical_id="camera_rear_fisheye",
            resolution_h=567,
            resolution_w=320,
            shutter_type=5,
            ftheta_param=FthetaCameraParam(
                principal_point_x=160.0,
                principal_point_y=283.5,
                reference_poly=1,
                pixeldist_to_angle_poly=[0.0, 0.005, 0.0, 0.0, 0.0],
                angle_to_pixeldist_poly=[0.0, 200.0, 0.0, 0.0, 0.0],
                max_angle=1.2,
                linear_cde=LinearCde(linear_c=1.0, linear_d=0.0, linear_e=1.0),
            ),
        )

        with self.assertLogs("nurec.grpc.serve", level="INFO") as captured:
            bank._get_camera(spec)
            bank._get_camera(spec)

        matching_logs = [
            message
            for message in captured.output
            if "Camera 'camera_rear_fisheye' not in reconstruction cameras" in message
        ]
        self.assertEqual(len(matching_logs), 1)

    def test_unknown_camera_without_params_raises_key_error(self):
        """When logical_id is NOT in the bank and NO camera_param is provided,
        it should raise KeyError (existing behavior — can't render without params)."""
        ref = _make_torch_camera("cam_front")
        bank = self._make_camera_bank_with_cameras({"cam_front": ref})

        spec = CameraSpec(logical_id="camera_nonexistent")  # No camera_param
        with self.assertRaises(KeyError):
            bank._get_camera(spec)

    def test_known_camera_without_params_returns_bank_camera(self):
        """When logical_id IS in the bank and NO camera_param is provided,
        it should return the bank camera directly (existing behavior)."""
        ref = _make_torch_camera("cam_front", unique_sensor_idx=99)
        bank = self._make_camera_bank_with_cameras({"cam_front": ref})

        spec = CameraSpec(logical_id="cam_front")  # No camera_param
        result = bank._get_camera(spec)
        self.assertIs(result, ref)


class TestWaitForGrpcServer(unittest.TestCase):
    @staticmethod
    def _free_port():
        """Reserve and release a local port so nothing is listening on it."""
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("localhost", 0))
            return s.getsockname()[1]

    def test_returns_once_server_becomes_available(self):
        # Exercise the unavailable -> ready transition that --wait-server exists for: start
        # waiting while nothing is listening, confirm the wait stays pending, then bring the
        # server up on that port and confirm the wait completes.
        port = self._free_port()

        async def run():
            async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
                waiter = asyncio.ensure_future(wait_for_grpc_server(channel, timeout_s=10))

                # Nothing is listening yet, so the wait must not resolve.
                await asyncio.sleep(1)
                self.assertFalse(waiter.done())

                # Bring the server up on the same port; the wait should now complete.
                server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
                self.assertEqual(server.add_insecure_port(f"localhost:{port}"), port)
                server.start()
                try:
                    await asyncio.wait_for(waiter, timeout=10)
                finally:
                    server.stop(grace=None)

        asyncio.run(run())

    def test_waits_until_health_reports_serving(self):
        # The port binds before the Harmonizer model is loaded, so transport readiness alone is
        # not readiness: while health says NOT_SERVING the wait must stay pending.
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
        port = server.add_insecure_port("localhost:0")
        server.start()

        async def run():
            async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
                waiter = asyncio.ensure_future(wait_for_grpc_server(channel, timeout_s=30))

                # Connected, but the server is not serving yet.
                await asyncio.sleep(3)
                self.assertFalse(waiter.done())

                health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
                await asyncio.wait_for(waiter, timeout=30)

        try:
            asyncio.run(run())
        finally:
            server.stop(grace=None)

    def test_raises_when_server_never_comes_up(self):
        port = self._free_port()

        async def run():
            async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
                await wait_for_grpc_server(channel, timeout_s=1)

        with self.assertRaises(TimeoutError):
            asyncio.run(run())


class TestHarmonizerLoadPending(unittest.TestCase):
    def test_failed_load_reports_serving_and_logs_the_error(self):
        failure = RuntimeError("checkpoint is missing")

        def _failing_get(*args, **kwargs):
            raise failure

        with _service() as (_, service):
            with mock.patch("nurec.grpc.serve.HarmonizerModelFactory.get", _failing_get):
                service.set_harmonizer(True, "url", "cache", "harmonizer.pt", (1, 1))
                service.wait_for_harmonizer_load()

            self.assertIs(service.harmonizer_load_error, failure)
            self.assertFalse(service.harmonizer_load_pending)

            health_servicer = health.HealthServicer()
            health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
            with self.assertLogs("nurec.grpc.serve", level="ERROR") as logs:
                _serve_once_harmonizer_loaded(service, health_servicer)

            self.assertIn("failed to load", "".join(logs.output))
            response = health_servicer.Check(health_pb2.HealthCheckRequest(service=""), None)
            self.assertEqual(response.status, health_pb2.HealthCheckResponse.SERVING)

    def test_pending_until_the_background_load_finishes(self):
        release = threading.Event()

        def _blocking_get(*args, **kwargs):
            release.wait(timeout=30)
            return mock.MagicMock()

        with _service() as (_, service):
            try:
                self.assertFalse(service.harmonizer_load_pending)

                with mock.patch("nurec.grpc.serve.HarmonizerModelFactory.get", _blocking_get):
                    service.set_harmonizer(True, "url", "cache", "harmonizer.pt", (1, 1))
                    self.assertTrue(service.harmonizer_load_pending)
                    release.set()
                    service.wait_for_harmonizer_load()

                self.assertFalse(service.harmonizer_load_pending)
            finally:
                release.set()


@contextlib.contextmanager
def _service(
    *,
    allowed_image_formats: str | None = None,
    camera_only: bool = False,
    max_workers: int = 1,
) -> Iterator[tuple[SensorsimServiceStub, SensorSimService]]:
    """Run a throwaway SensorSimService."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    service = SensorSimService(
        server,
        artifacts_glob=None,
        ray_chunk_size=1,
        egocar_hoods_dir=None,
        downloader=None,
        scene_cache=None,
        cache_size=1,
        metrics_output_dir=None,
        enable_editing_actors=False,
        server_config=GrpcServerConfig(
            allowed_image_formats=allowed_image_formats,
            camera_only=camera_only,
            max_workers=max_workers,
        ),
    )
    add_SensorsimServiceServicer_to_server(service, server)
    port = server.add_insecure_port("localhost:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        stub = SensorsimServiceStub(channel)
        yield stub, service
    finally:
        service.close()
        channel.close()
        server.stop(0)


def _register_prefetch_scene(service: SensorSimService, scene_key: SceneCacheKey) -> None:
    artifact = _registered_artifact(Path(f"/scenes/{scene_key.scene_id}.usdz"), scene_key.scene_id)
    registration = ArtifactRegistration(artifact, False, scene_key)
    with service._artifact_registry_lock:
        service._artifact_registrations[scene_key.key_hash] = registration


class TestCameraOnly(unittest.TestCase):
    def test_pattern_lookup_fails_before_scene_loading(self):
        with _service(camera_only=True) as (stub, service), mock.patch.object(service, "get_backend") as get_backend:
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.get_lidar_firing_pattern(LidarFiringPatternRequest(scene_id="scene"))

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
            self.assertIn("camera-only", ctx.exception.details())
            get_backend.assert_not_called()

    def test_lidar_rpc_fails_before_scene_loading(self):
        with _service(camera_only=True) as (stub, _):
            self.assertEqual(stub.get_server_config(Empty()).server_config["camera_only"], "True")
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.render_lidar(LidarRenderRequest(scene_id="scene"))

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
            self.assertIn("camera-only", ctx.exception.details())


class TestScenePrefetch(unittest.TestCase):
    def test_headerless_hint_uses_unique_registered_url_key(self):
        with _service() as (stub, service):
            scene_key = SceneCacheKey.from_url("scene", "file:///scenes/scene.usdz")
            _register_prefetch_scene(service, scene_key)
            service._scene_prefetch = mock.MagicMock()
            service._scene_prefetch.submit.return_value = True
            stub.prefetch_scene(PrefetchSceneRequest(scene_id="scene"))

            service._scene_prefetch.submit.assert_called_once_with(scene_key)

    def test_mismatched_id_without_url_rejects_hint_before_submission(self):
        with _service() as (stub, service):
            _register_prefetch_scene(service, SceneCacheKey(scene_id="scene"))
            service._scene_prefetch = mock.MagicMock()
            for metadata_id in ("other", ""):
                with self.subTest(metadata_id=metadata_id), self.assertRaises(grpc.RpcError) as ctx:
                    stub.prefetch_scene(
                        PrefetchSceneRequest(scene_id="scene"), metadata=((SCENE_ID_HEADER, metadata_id),)
                    )
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.NOT_FOUND)
                self.assertIn("does not match metadata scene", ctx.exception.details())
            service._scene_prefetch.submit.assert_not_called()

    def test_scene_query_validates_metadata_before_loading(self):
        with _service() as (stub, service), mock.patch.object(service, "get_backend") as get_backend:
            backend = mock.Mock()
            backend.get_available_cameras.return_value = AvailableCamerasReturn()
            get_backend.return_value = contextlib.nullcontext(backend)
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.get_available_cameras(
                    AvailableCamerasRequest(scene_id="scene"), metadata=((SCENE_ID_HEADER, "other"),)
                )
            self.assertEqual(ctx.exception.code(), grpc.StatusCode.NOT_FOUND)
            self.assertIn("does not match metadata scene", ctx.exception.details())
            get_backend.assert_not_called()
            for metadata in ((), ((SCENE_ID_HEADER, "scene"),)):
                with self.subTest(metadata=metadata):
                    response = stub.get_available_cameras(AvailableCamerasRequest(scene_id="scene"), metadata=metadata)
                    self.assertEqual(response, AvailableCamerasReturn())

    def test_unknown_scene_returns_not_found(self):
        with _service() as (stub, _):
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.prefetch_scene(PrefetchSceneRequest(scene_id="missing"))

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.NOT_FOUND)

    def test_blocking_prefetch_requires_two_workers(self):
        with _service() as (stub, service):
            service._scene_prefetch = mock.MagicMock()
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.prefetch_scene(PrefetchSceneRequest(scene_id="scene", wait_for_capacity=True))

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
            service._scene_prefetch.submit.assert_not_called()

    def test_blocking_prefetch_respects_server_timeout(self):
        with _service(max_workers=2) as (stub, service):
            _register_prefetch_scene(service, SceneCacheKey(scene_id="scene"))
            service._backend_cache_wait_timeout_s = 0.0

            def submit(_key, *, wait_for_capacity, check_wait) -> None:
                self.assertTrue(wait_for_capacity)
                check_wait()

            service._scene_prefetch = mock.MagicMock()
            service._scene_prefetch.submit.side_effect = submit
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.prefetch_scene(PrefetchSceneRequest(scene_id="scene", wait_for_capacity=True))

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)

    def test_prefetch_reports_a_full_pending_queue(self):
        with _service() as (stub, service):
            _register_prefetch_scene(service, SceneCacheKey(scene_id="scene"))
            service._scene_prefetch = mock.MagicMock()
            service._scene_prefetch.submit.return_value = False
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.prefetch_scene(PrefetchSceneRequest(scene_id="scene"))

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)

    def test_remote_prefetch_is_rejected(self):
        with _service() as (stub, service):
            service._scene_prefetch = mock.MagicMock()
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.prefetch_scene(
                    PrefetchSceneRequest(scene_id="scene"),
                    metadata=(
                        (SCENE_ID_HEADER, "scene"),
                        (SCENE_URL_HEADER, "https://example.test/scene.usdz"),
                    ),
                )

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
            service._scene_prefetch.submit.assert_not_called()

    def test_headerless_remote_prefetch_is_rejected(self):
        with _service() as (stub, service):
            scene_key = SceneCacheKey.from_url("scene", "https://example.test/scene.usdz")
            _register_prefetch_scene(service, scene_key)
            service._scene_prefetch = mock.MagicMock()

            with self.assertRaises(grpc.RpcError) as ctx:
                stub.prefetch_scene(PrefetchSceneRequest(scene_id="scene"))

            self.assertEqual(ctx.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
            service._scene_prefetch.submit.assert_not_called()

    def test_registration_rejects_file_changed_while_hashing(self):
        artifact = _registered_artifact(Path("/scenes/scene.usdz"), "scene")
        with (
            mock.patch(
                "nurec.grpc.serve._local_file_fingerprint",
                side_effect=[(1, 2, 3, 4, 5), (1, 2, 3, 4, 6)],
            ),
            self.assertRaisesRegex(ValueError, "changed while it was registered"),
        ):
            _register_local_artifact(artifact, SceneCacheKey(scene_id="scene"))

    def test_backend_wait_for_prefetch_respects_cancelled_context(self):
        with _service() as (_, service):
            scene_key = SceneCacheKey(scene_id="scene")
            registration = ArtifactRegistration(
                _registered_artifact(Path("/scenes/scene.usdz"), "scene"),
                True,
                scene_key,
            )
            with service._artifact_registry_lock:
                service._artifact_registrations[scene_key.key_hash] = registration

            def take(_key, *, check_wait):
                check_wait()

            service._scene_prefetch = mock.MagicMock()
            service._scene_prefetch.take.side_effect = take
            selection_lease = mock.MagicMock()
            selection_lease.release.return_value = []
            service.scene_cache = mock.MagicMock()
            service.scene_cache.acquire_scene.return_value = selection_lease
            context = mock.MagicMock()
            context.is_active.return_value = False
            context.time_remaining.return_value = 30.0

            with self.assertRaises(CacheWaitAbortedError) as error:
                with service.get_backend(scene_key, context):
                    pass

            self.assertEqual(error.exception.reason, CacheWaitAbortedReason.CANCELLED)
            selection_lease.release.assert_called_once_with()


@contextlib.contextmanager
def _mock_backend_ctx(*args, **kwargs):
    """Context manager that yields a MagicMock backend (satisfies `with service.get_backend(...) as b:`)."""
    yield mock.MagicMock()


def _mock_backend_ctx_with_frame(frame: CameraFrame):
    """Build a get_backend replacement whose backend.render_camera_request returns (frame, None).

    Sets the artifact's semantic metadata to None explicitly. Without this, MagicMock makes
    ``backend.source_artifact.semantic_class_names`` truthy but iterable-as-empty, and the
    SEMANTICS-output branch's ``max(names) + 1`` raises ``ValueError("max() arg is an empty sequence")``.
    """

    @contextlib.contextmanager
    def _ctx(*args, **kwargs):
        backend = mock.MagicMock()
        # Backend.render_camera_request returns (frame, (h, w), model_overlay_snapshot); the
        # snapshot is None unless overlay_bounding_boxes=MODEL was set on the request.
        backend.render_camera_request.return_value = (frame, None, None)
        backend.source_artifact.semantic_class_names = None
        backend.source_artifact.semantic_class_colors = None
        yield backend

    return _ctx


def _mock_backend_ctx_render_raises(exc: Exception):
    """get_backend replacement whose backend.render_camera_request raises ``exc``.

    Mirrors the real render path: render_camera_frame_from_ray_bundle raises
    MissingRenderOutputError when the model lacks a requested head, rather than returning a frame
    with the field left None.
    """

    @contextlib.contextmanager
    def _ctx(*args, **kwargs):
        backend = mock.MagicMock()
        backend.render_camera_request.side_effect = exc
        backend.source_artifact.semantic_class_names = None
        backend.source_artifact.semantic_class_colors = None
        yield backend

    return _ctx


class TestRigTrajectoryConversion(unittest.TestCase):
    @staticmethod
    def _rig(timestamps, quaternions=None):
        if quaternions is None:
            quaternions = [[0, 0, 0, 1]] * len(timestamps)
        poses = np.broadcast_to(np.eye(4), (len(timestamps), 4, 4)).copy()
        poses[:, 0, 3] = 3 + np.asarray(timestamps) / 1000
        from scipy.spatial.transform import Rotation

        poses[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
        return RigTrajectories.RigTrajectory(
            sequence_id="rig",
            rig_bbox=None,
            cameras_frame_timestamps_us={},
            lidars_frame_timestamps_us={},
            T_rig_worlds=torch.tensor(poses, dtype=torch.float64),
            T_rig_world_timestamps_us=torch.tensor(timestamps, dtype=torch.int64),
        )

    def test_camera_bank_accepts_rig_padding_before_zero(self):
        rig = self._rig([-1, 0, 1000])
        rig_trajectories = mock.Mock(rig_trajectories=[rig], camera_calibrations={})
        with mock.patch("nurec.grpc.serve.RigTrajectories.from_dict", return_value=rig_trajectories):
            bank = CameraBank(
                mock.Mock(),
                FrameConversion(matrix=np.eye(4, dtype=np.float32)),
                scene_time_range_us=HalfClosedInterval(0, 1001),
            )
        self.assertEqual([p.timestamp_us for p in bank.trajectory.poses], [0, 1000])
        np.testing.assert_allclose([p.pose.vec.x for p in bank.trajectory.poses], [3, 4], atol=1e-5)

    def test_zero_boundary_interpolates_rig_translation_and_rotation(self):
        rig = self._rig([-1000, 1000], [[0, 0, 0, 1], [0, 0, 2**-0.5, 2**-0.5]])
        trajectory = _rig_trajectory_to_grpc_trajectory(rig)
        self.assertEqual([p.timestamp_us for p in trajectory.poses], [0, 1000])
        pose = trajectory.poses[0].pose
        self.assertAlmostEqual(pose.vec.x, 3, places=5)
        self.assertAlmostEqual(pose.quat.z, np.sin(np.pi / 8), places=5)
        self.assertAlmostEqual(pose.quat.w, np.cos(np.pi / 8), places=5)

    def test_positive_only_rig_is_unchanged(self):
        trajectory = _rig_trajectory_to_grpc_trajectory(self._rig([100, 1000, 2000]))
        self.assertEqual([p.timestamp_us for p in trajectory.poses], [100, 1000, 2000])
        np.testing.assert_allclose([p.pose.vec.x for p in trajectory.poses], [3.1, 4, 5], atol=1e-5)


class TestRenderCameraStatusCodes(unittest.TestCase):
    def test_nonfinite_pose_is_invalid_argument_and_next_valid_request_works(self):
        camera = TorchCamera(
            camera_model_parameters=CameraBank._get_camera_model_parameters(
                CameraSpec(
                    logical_id="front",
                    resolution_h=4,
                    resolution_w=4,
                    shutter_type=1,
                    opencv_pinhole_param=OpenCVPinholeCameraParam(
                        principal_point_x=2,
                        principal_point_y=2,
                        focal_length_x=2,
                        focal_length_y=2,
                        radial_coeffs=[0] * 6,
                        tangential_coeffs=[0] * 2,
                        thin_prism_coeffs=[0] * 4,
                    ),
                )
            ),
            logical_camera_id="front",
            unique_sensor_idx=0,
            time_range_us=HalfClosedInterval(0, 2000),
            world_to_nurec=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
            T_camera_rig=torch.eye(4),
        )

        @contextlib.contextmanager
        def backend_context(*args, **kwargs):
            backend = mock.Mock()
            backend.source_artifact.semantic_class_names = None
            backend.source_artifact.semantic_class_colors = None

            def render(request, **kwargs):
                camera.build_ray_bundle(request)
                return CameraFrame(color_image=torch.zeros(4, 4, 3)), (4, 4), None

            backend.render_camera_request.side_effect = render
            yield backend

        request = CameraRenderRequest(
            scene_id="scene",
            frame_start_us=1000,
            frame_end_us=1001,
            resolution_h=4,
            resolution_w=4,
            image_format=ImageFormat.PNG,
        )
        request.sensor_pose.start_pose.quat.w = request.sensor_pose.end_pose.quat.w = 1
        with _service() as (stub, service):
            with mock.patch.object(service, "get_backend", backend_context):
                for endpoint in ("start_pose", "end_pose"):
                    for value in (float("nan"), float("inf"), float("-inf")):
                        pose = getattr(request.sensor_pose, endpoint)
                        pose.vec.x = value
                        with self.subTest(endpoint=endpoint, value=value):
                            with self.assertRaises(grpc.RpcError) as caught:
                                stub.render_camera(request)
                            self.assertEqual(caught.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
                            self.assertIn(
                                f"Pose component vec.x must be finite; got {value}", caught.exception.details()
                            )
                        pose.vec.x = 0

                with (
                    mock.patch("nurec.grpc.serve.RayBundle.build", return_value=mock.sentinel.bundle),
                    mock.patch.object(service, "_encode_image_bytes", return_value=b"valid"),
                ):
                    response = stub.render_camera(request)
                self.assertEqual(response.camera_render_outputs[0].image_bytes, b"valid")

    def test_temporal_harmonizer_is_ignored_for_non_color_camera_products(self):
        with _service() as (stub, service):
            service.harmonizer_temporal_enabled = True
            backend_ctx = _mock_backend_ctx_with_frame(CameraFrame(distance_image=torch.zeros(4, 4)))
            with mock.patch.object(service, "get_backend", backend_ctx):
                response = stub.render_camera(
                    CameraRenderRequest(
                        scene_id="scene",
                        image_format=ImageFormat.PNG,
                        apply_harmonizer=True,
                        render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED],
                    )
                )

            self.assertEqual(len(response.camera_render_outputs), 1)
            self.assertEqual(
                response.camera_render_outputs[0].render_product,
                CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED,
            )

    def test_renderer_fails_to_produce_semantics_returns_failed_precondition(self):
        with _service() as (stub, service):
            # The real render path raises MissingRenderOutputError when the model lacks a semantic
            # head; the server must surface that as FAILED_PRECONDITION, not INVALID_ARGUMENT.
            backend_ctx = _mock_backend_ctx_render_raises(
                MissingRenderOutputError(
                    "Semantic image requested, but the model or render path did not produce semantic_logits."
                )
            )
            with mock.patch.object(service, "get_backend", backend_ctx):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.render_camera(
                        CameraRenderRequest(
                            scene_id="scene",
                            render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED],
                        )
                    )
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
                self.assertIn("semantic", (ctx.exception.details() or "").lower())

    def test_renderer_fails_to_produce_distance_returns_failed_precondition(self):
        with _service() as (stub, service):
            backend_ctx = _mock_backend_ctx_render_raises(
                MissingRenderOutputError(
                    "Distance image requested, but the model or render path did not produce distance output."
                )
            )
            with mock.patch.object(service, "get_backend", backend_ctx):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.render_camera(
                        CameraRenderRequest(
                            scene_id="scene",
                            render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED],
                        )
                    )
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
                self.assertIn("distance", (ctx.exception.details() or "").lower())

    def test_render_camera_duplicate_products_returns_invalid_argument(self):
        with _service() as (stub, service):
            backend_ctx = _mock_backend_ctx_with_frame(CameraFrame(color_image=torch.zeros(4, 4, 3)))
            with mock.patch.object(service, "get_backend", backend_ctx):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.render_camera(
                        CameraRenderRequest(
                            scene_id="scene",
                            render_products=[
                                CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR,
                                CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR,
                            ],
                        )
                    )
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
                self.assertIn("unique", (ctx.exception.details() or "").lower())

    def test_render_camera_ego_mask_without_color_returns_invalid_argument(self):
        with _service() as (stub, service):
            backend_ctx = _mock_backend_ctx_with_frame(CameraFrame(distance_image=torch.zeros(4, 4)))
            with mock.patch.object(service, "get_backend", backend_ctx):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.render_camera(
                        CameraRenderRequest(
                            scene_id="scene",
                            render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED],
                            insert_ego_mask=True,
                        )
                    )
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
                self.assertIn("color", (ctx.exception.details() or "").lower())

    def test_render_camera_ego_mask_with_color_reaches_egocar_selection(self):
        with _service() as (stub, service):
            backend_ctx = _mock_backend_ctx_with_frame(
                CameraFrame(color_image=torch.zeros(4, 4, 3), distance_image=torch.zeros(4, 4))
            )
            with mock.patch.object(service, "get_backend", backend_ctx):
                with mock.patch.object(
                    service.egocar_rig_bank, "select_from_request", return_value=None
                ) as select_mock:
                    response = stub.render_camera(
                        CameraRenderRequest(
                            scene_id="scene",
                            image_format=ImageFormat.PNG,
                            render_products=[
                                CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR,
                                CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED,
                            ],
                            insert_ego_mask=True,
                        )
                    )
            # RPC reached the server, returned a real response — no grpc.RpcError raised.
            self.assertEqual(len(response.camera_render_outputs), 2)
            # Ego selection invoked exactly once (only during the COLOR product encoding).
            select_mock.assert_called_once()

    def test_render_camera_multi_product_returns_one_output_per_product(self):
        with _service() as (stub, service):
            backend_ctx = _mock_backend_ctx_with_frame(
                CameraFrame(color_image=torch.zeros(4, 4, 3), distance_image=torch.zeros(4, 4))
            )
            with mock.patch.object(service, "get_backend", backend_ctx):
                response = stub.render_camera(
                    CameraRenderRequest(
                        scene_id="scene",
                        image_format=ImageFormat.PNG,
                        render_products=[
                            CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR,
                            CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED,
                        ],
                    )
                )
            self.assertEqual(len(response.camera_render_outputs), 2)
            self.assertEqual(
                response.camera_render_outputs[0].render_product,
                CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR,
            )
            self.assertEqual(
                response.camera_render_outputs[1].render_product,
                CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED,
            )

    def test_render_camera_overlay_applied_to_every_output_product(self):
        with _service() as (stub, service):
            backend_ctx = _mock_backend_ctx_with_frame(
                CameraFrame(
                    color_image=torch.zeros(4, 4, 3),
                    distance_image=torch.zeros(4, 4),
                    semantic_image=torch.zeros(4, 4, dtype=torch.int64),
                )
            )
            with mock.patch.object(service, "get_backend", backend_ctx):
                with mock.patch.object(
                    service,
                    "_apply_bounding_box_overlay_if_requested",
                    side_effect=lambda image_u8, *args, **kwargs: image_u8,
                ) as overlay_mock:
                    stub.render_camera(
                        CameraRenderRequest(
                            scene_id="scene",
                            image_format=ImageFormat.PNG,
                            render_products=[
                                CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR,
                                CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED,
                                CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED,
                            ],
                            overlay_bounding_boxes=OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_MODEL,
                        )
                    )
                self.assertEqual(overlay_mock.call_count, 3)

    def test_render_camera_timestamp_out_of_range(self):
        with _service() as (stub, service):

            @contextlib.contextmanager
            def _raise(*args, **kwargs):
                raise TimestampOutOfRangeError("ts out of range")
                yield  # pragma: no cover

            with mock.patch.object(service, "get_backend", _raise):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.render_camera(CameraRenderRequest(scene_id="scene"))
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.OUT_OF_RANGE)


class TestBatchRenderCameraValidation(unittest.TestCase):
    def test_item_render_does_not_swallow_fatal_worker_error(self):
        service = _make_service_no_server()
        item = BatchCameraRenderRequestItem(camera_name="cam", request=CameraRenderRequest(scene_id="scene"))
        with (
            mock.patch.object(
                service,
                "_camera_render_return_from_request",
                side_effect=RuntimeError("CUDA error: an illegal memory access was encountered"),
            ),
            self.assertRaisesRegex(RuntimeError, "illegal memory access"),
        ):
            service._render_single_camera_for_batch_camera(mock.MagicMock(), item)

    def test_mismatched_scene_ids_returns_invalid_argument(self):
        with _service() as (stub, service):
            request = BatchCameraRenderRequest(
                items=[
                    BatchCameraRenderRequestItem(
                        camera_name="cam_a",
                        request=CameraRenderRequest(scene_id="scene_1"),
                    ),
                    BatchCameraRenderRequestItem(
                        camera_name="cam_b",
                        request=CameraRenderRequest(scene_id="scene_2"),
                    ),
                ]
            )
            with self.assertRaises(grpc.RpcError) as ctx:
                stub.batch_render_camera(request)
            self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
            details = ctx.exception.details() or ""
            self.assertIn("scene_id", details)
            self.assertIn("scene_1", details)
            self.assertIn("scene_2", details)

    def test_empty_batch_returns_empty_response(self):
        with _service() as (stub, service):
            response = stub.batch_render_camera(BatchCameraRenderRequest(items=[]))
            self.assertEqual(len(response.items), 0)

    def test_oom_during_item_render_aborts_batch_and_reaches_get_backend_cleanup(self):
        with _service() as (stub, service):
            cleanup_calls = []

            @contextlib.contextmanager
            def _backend_ctx_with_oom_cleanup(*args, **kwargs):
                backend = mock.MagicMock()
                try:
                    yield backend
                except torch.cuda.OutOfMemoryError:
                    cleanup_calls.append("evicted")
                    raise

            with mock.patch.object(service, "get_backend", _backend_ctx_with_oom_cleanup):
                with mock.patch.object(
                    service,
                    "_camera_render_return_from_request",
                    side_effect=torch.cuda.OutOfMemoryError("oom"),
                ):
                    request = BatchCameraRenderRequest(
                        items=[
                            BatchCameraRenderRequestItem(
                                camera_name="cam_a", request=CameraRenderRequest(scene_id="scene")
                            ),
                            BatchCameraRenderRequestItem(
                                camera_name="cam_b", request=CameraRenderRequest(scene_id="scene")
                            ),
                        ]
                    )
                    with self.assertRaises(grpc.RpcError) as ctx:
                        stub.batch_render_camera(request)
                    self.assertEqual(ctx.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)

            # get_backend's OOM cleanup path ran — proof the exception propagated out of the
            # per-item loop instead of being swallowed into a per-item success=False result.
            self.assertEqual(cleanup_calls, ["evicted"])


class TestCameraBatchDispatch(unittest.TestCase):
    @staticmethod
    def _camera_batch_request(
        *, track_id: str | None = "car", moving: bool = False, aspect: str | None = None
    ) -> CameraRenderRequest:
        pose = Pose(vec={"x": 1.0}, quat={"w": 1.0})
        end_pose = Pose(vec={"x": 2.0}, quat={"w": 1.0}) if moving else pose
        dynamic_objects = []
        if track_id is not None:
            dynamic_object = DynamicObject(
                track_id=track_id,
                pose_pair=PosePair(start_pose=pose, end_pose=end_pose),
            )
            if aspect is not None:
                dynamic_object.aspect_state.CopyFrom(AspectState(aspects=[aspect]))
            dynamic_objects.append(dynamic_object)
        return CameraRenderRequest(
            scene_id="scene",
            frame_start_us=10,
            frame_end_us=11,
            camera_intrinsics=CameraSpec(logical_id="front"),
            dynamic_objects=dynamic_objects,
        )

    def test_camera_batch_reports_shared_scene_incompatibility(self):
        first = self._camera_batch_request()
        second = self._camera_batch_request()
        second.sensor_pose.start_pose.vec.x = 4.0
        self.assertIsNone(_camera_batch_incompatibility([first, second], {"car"}))

        self.assertEqual(
            _camera_batch_incompatibility([first], {"car"}),
            _CameraBatchDispatch.SEQUENTIAL_SINGLE_ITEM,
        )

        other = self._camera_batch_request()
        other.scene_id = "other"
        self.assertEqual(
            _camera_batch_incompatibility([first, other], {"car"}),
            _CameraBatchDispatch.SEQUENTIAL_SCENE_MISMATCH,
        )

        other = self._camera_batch_request()
        other.frame_end_us += 1
        self.assertEqual(
            _camera_batch_incompatibility([first, other], {"car"}),
            _CameraBatchDispatch.SEQUENTIAL_FRAME_INTERVAL_MISMATCH,
        )

        other = self._camera_batch_request()
        other.camera_intrinsics.logical_id = "other"
        self.assertEqual(
            _camera_batch_incompatibility([first, other], {"car"}),
            _CameraBatchDispatch.SEQUENTIAL_INTRINSICS_MISMATCH,
        )

        moving = self._camera_batch_request(moving=True)
        self.assertEqual(
            _camera_batch_incompatibility([first, moving], {"car"}),
            _CameraBatchDispatch.SEQUENTIAL_ACTOR_OVERRIDES_MISMATCH,
        )
        different_aspect = self._camera_batch_request(aspect="brake:on")
        self.assertEqual(
            _camera_batch_incompatibility([first, different_aspect], {"car"}),
            _CameraBatchDispatch.SEQUENTIAL_ACTOR_OVERRIDES_MISMATCH,
        )
        matching_aspects = [self._camera_batch_request(aspect="brake:on") for _ in range(2)]
        self.assertIsNone(_camera_batch_incompatibility(matching_aspects, {"car"}))
        self.assertEqual(
            _camera_batch_incompatibility([first, second], {"pedestrian"}),
            _CameraBatchDispatch.SEQUENTIAL_ACTOR_OVERRIDES_MISMATCH,
        )

        without_overrides = [self._camera_batch_request(track_id=None) for _ in range(2)]
        self.assertIsNone(_camera_batch_incompatibility(without_overrides, {"car"}))

    def _camera_batch_backend(self, shutter_type, dynamic_actor_ids):
        renderable_model = mock.create_autospec(RenderableModel, instance=True)
        renderable_model.device = torch.device("cuda")
        renderable_model.supports_shared_camera_batch.return_value = True
        renderable_model.supports_edit_actors.return_value = True
        renderable_model.dynamic_actor_ids.return_value = dynamic_actor_ids
        frames = [
            mock.MagicMock(),
            mock.MagicMock(),
        ]
        renderable_model.render_camera_color_batch.return_value = frames
        renderable_model.render_camera_color_batch_from_ray_bundles.return_value = frames
        ray_bundle = mock.MagicMock()
        camera = mock.MagicMock()
        camera.camera_model_parameters.shutter_type = shutter_type
        camera.build_ray_bundle.return_value = ray_bundle
        camera.build_ray_batch.return_value = (mock.MagicMock(), [mock.MagicMock(), mock.MagicMock()])
        camera_bank = mock.MagicMock()
        camera_bank._get_camera.return_value = camera
        backend = Backend(
            renderable_model=renderable_model,
            camera_bank=camera_bank,
            world_to_nurec=mock.MagicMock(),
            scene_time_range_us=HalfClosedInterval(0, 20),
            lidar_bank=mock.MagicMock(),
            asset_bank=mock.MagicMock(),
            source_artifact=mock.MagicMock(),
            backend_cache_key=_test_backend_cache_key(),
        )
        return backend, camera, renderable_model

    def test_camera_batch_dispatches_by_shutter_and_actor_state(self):
        cases = (
            (
                "global fixed actors",
                ncore.data.ShutterType.GLOBAL,
                {"car"},
                "car",
                True,
                _CameraBatchDispatch.NATIVE_BATCHED_RAYS,
            ),
            (
                "rolling static scene",
                ncore.data.ShutterType.ROLLING_TOP_TO_BOTTOM,
                set(),
                None,
                False,
                _CameraBatchDispatch.NATIVE_PER_CAMERA_RAYS,
            ),
            (
                "rolling scene actors",
                ncore.data.ShutterType.ROLLING_TOP_TO_BOTTOM,
                {"car"},
                None,
                False,
                _CameraBatchDispatch.SEQUENTIAL_ROLLING_ACTORS,
            ),
            (
                "rolling fixed actors",
                ncore.data.ShutterType.ROLLING_TOP_TO_BOTTOM,
                {"car"},
                "car",
                True,
                _CameraBatchDispatch.SEQUENTIAL_ROLLING_ACTORS,
            ),
        )
        for name, shutter_type, dynamic_actor_ids, track_id, enable_editing_actors, expected in cases:
            with self.subTest(name=name):
                backend, camera, renderable_model = self._camera_batch_backend(shutter_type, dynamic_actor_ids)
                requests = [self._camera_batch_request(track_id=track_id) for _ in range(2)]

                frames, dispatch = backend.render_camera_batch_shared_scene(
                    requests,
                    enable_editing_actors=enable_editing_actors,
                )

                self.assertEqual(dispatch, expected)
                if expected == _CameraBatchDispatch.SEQUENTIAL_ROLLING_ACTORS:
                    self.assertIsNone(frames)
                else:
                    self.assertEqual(len(frames or []), 2)
                if expected == _CameraBatchDispatch.NATIVE_BATCHED_RAYS:
                    rendering_data, frame_meta = camera.build_ray_batch.return_value
                    renderable_model.render_camera_color_batch.assert_called_once_with(
                        rendering_data,
                        frame_meta,
                        actors_snapshot=mock.ANY if track_id is not None else None,
                    )
                elif expected == _CameraBatchDispatch.NATIVE_PER_CAMERA_RAYS:
                    renderable_model.render_camera_color_batch_from_ray_bundles.assert_called_once()


def _make_service_no_server(*, camera_only: bool = False) -> SensorSimService:
    """Construct a SensorSimService without binding a port — for unit-testing internal helpers."""
    return SensorSimService(
        server=None,
        artifacts_glob=None,
        ray_chunk_size=1,
        egocar_hoods_dir=None,
        downloader=None,
        scene_cache=None,
        cache_size=1,
        metrics_output_dir=None,
        enable_editing_actors=False,
        server_config=GrpcServerConfig(camera_only=camera_only),
    )


class TestCudaIpcTransport(unittest.TestCase):
    """Test IPC RPC validation and cleanup with a mocked native manager."""

    def _service_with_mock_mgr(self) -> tuple[SensorSimService, mock.MagicMock]:
        service = _make_service_no_server()
        mgr = mock.MagicMock()
        service._get_ipc_mgr = mock.MagicMock(return_value=mgr)
        mgr.ready_event_handles.return_value = []
        mgr.runtime_version = 12080
        mgr.layout.return_value = (3, 2, False)
        # Real _get_ipc_mgr() sets this as a side effect of constructing IpcManager;
        # since that's mocked away here, set it explicitly to match the device the
        # render-path tests' CUDA tensors land on (device "cuda" resolves to the
        # current device, index 0, under this test's single-GPU-visible runner).
        service._ipc_device_index = 0
        return service, mgr

    def test_open_ipc_session_returns_handles_from_the_manager(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.open_session.return_value = ("sess-1", [b"h0" * 32, b"h1" * 32], 1234, b"u" * 16, 8, 6)
        request = ipc_pb2.OpenIpcSessionRequest(requested_width=8, requested_height=6, requested_buffer_count=2)

        response = service.open_ipc_session(request, mock.MagicMock())

        mgr.open_session.assert_called_once_with(8, 6, 2, event_ordered=False)
        self.assertEqual(response.session_id, "sess-1")
        self.assertEqual(list(response.cuda_ipc_mem_handles), [b"h0" * 32, b"h1" * 32])
        self.assertEqual(response.buffer_size, 1234)
        self.assertEqual(response.actual_width, 8)
        self.assertEqual(response.actual_height, 6)
        self.assertEqual(service._ipc_open_sessions, 1)

    def test_close_ipc_session_decrements_only_when_a_session_was_removed(self):
        service, mgr = self._service_with_mock_mgr()
        service._ipc_open_sessions = 1
        mgr.close_session.return_value = False

        response = service.close_ipc_session(ipc_pb2.CloseIpcSessionRequest(session_id="missing"), mock.MagicMock())

        self.assertTrue(response.success)
        self.assertEqual(service._ipc_open_sessions, 1)  # unchanged: no session was actually closed

        mgr.close_session.return_value = True
        service.close_ipc_session(ipc_pb2.CloseIpcSessionRequest(session_id="sess-1"), mock.MagicMock())
        self.assertEqual(service._ipc_open_sessions, 0)

    def test_open_event_ordered_session_exports_completion_metadata(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.open_session.return_value = ("session", [b"m" * 64], 18, b"u" * 16, 3, 2)
        mgr.ready_event_handles.return_value = [b"e" * 64]
        response = service.open_ipc_session(
            ipc_pb2.OpenIpcSessionRequest(
                requested_width=3,
                requested_height=2,
                requested_buffer_count=1,
                completion_mode=ipc_pb2.EVENT_ORDERED,
            ),
            mock.MagicMock(),
        )
        mgr.open_session.assert_called_once_with(3, 2, 1, event_ordered=True)
        self.assertEqual(response.completion_mode, ipc_pb2.EVENT_ORDERED)
        self.assertEqual(list(response.cuda_ipc_ready_event_handles), [b"e" * 64])
        self.assertEqual(response.cuda_runtime_version, 12080)
        self.assertTrue(response.linux_boot_id)

    def test_camera_batch_rejects_invalid_layout_before_reserving(self):
        for field, value in (
            ("use_cuda_ipc", False),
            ("ipc_session_id", "other"),
            ("scene_id", "other"),
            ("resolution_w", 4),
        ):
            with self.subTest(field=field):
                service, mgr = self._service_with_mock_mgr()
                request = BatchCameraRenderRequest(
                    items=[
                        BatchCameraRenderRequestItem(
                            camera_name=str(i),
                            request=CameraRenderRequest(
                                scene_id="scene",
                                use_cuda_ipc=True,
                                ipc_session_id="session",
                                resolution_h=2,
                                resolution_w=3,
                                image_format=ImageFormat.RGB_UINT8_PLANAR,
                            ),
                        )
                        for i in range(2)
                    ]
                )
                setattr(request.items[0].request, field, value)
                context = mock.MagicMock()
                context.abort.side_effect = RuntimeError("aborted")
                with self.assertRaisesRegex(RuntimeError, "aborted"):
                    service.batch_render_camera(request, context)
                self.assertEqual(context.abort.call_args.args[0], grpc.StatusCode.INVALID_ARGUMENT)
                mgr.reserve_buffers.assert_not_called()
                if field == "resolution_w":
                    self.assertIn("dimensions", context.abort.call_args.args[1])

    def test_camera_batch_rejects_temporal_history_before_side_effects(self):
        for fields in (
            {"apply_harmonizer": True},
            {"harmonizer_session_id": "history"},
            {"apply_harmonizer": False, "harmonizer_session_id": "history"},
            {"harmonizer_session_id": ""},
        ):
            with self.subTest(fields=fields):
                service, mgr = self._service_with_mock_mgr()
                service.harmonizer_temporal_enabled = True
                request = BatchCameraRenderRequest(
                    items=[
                        BatchCameraRenderRequestItem(
                            camera_name=str(i),
                            request=CameraRenderRequest(
                                scene_id="scene",
                                use_cuda_ipc=True,
                                ipc_session_id="session",
                                resolution_h=2,
                                resolution_w=3,
                                image_format=ImageFormat.RGB_UINT8_PLANAR,
                                **({"released_buffer_id": 3} if i == 0 else fields),
                            ),
                        )
                        for i in range(2)
                    ]
                )
                context = mock.MagicMock()
                context.abort.side_effect = RuntimeError("aborted")
                with (
                    mock.patch.object(service, "get_backend") as get_backend,
                    mock.patch.object(service, "_resolve_harmonizer_session") as resolve_session,
                    self.assertRaisesRegex(RuntimeError, "aborted"),
                ):
                    service.batch_render_camera(request, context)
                self.assertEqual(context.abort.call_args.args[0], grpc.StatusCode.FAILED_PRECONDITION)
                self.assertEqual(mgr.mock_calls, [])
                get_backend.assert_not_called()
                resolve_session.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA IPC packing needs a CUDA device")
    def test_camera_batch_allows_raw_requests_on_temporal_server(self):
        service, mgr = self._service_with_mock_mgr()
        service.harmonizer_temporal_enabled = True
        mgr.reserve_buffers.return_value = [0, 1]
        image = torch.rand(2, 3, 3, device="cuda")
        backend = mock.MagicMock()
        backend.render_camera_request.return_value = (mock.Mock(color_image=image), None, None)
        request = BatchCameraRenderRequest(
            items=[
                BatchCameraRenderRequestItem(
                    camera_name=str(i),
                    request=CameraRenderRequest(
                        scene_id="scene",
                        use_cuda_ipc=True,
                        ipc_session_id="session",
                        resolution_h=2,
                        resolution_w=3,
                        image_format=ImageFormat.RGB_UINT8_PLANAR,
                        **({"apply_harmonizer": False} if i else {}),
                    ),
                )
                for i in range(2)
            ]
        )
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")
        with (
            mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)),
            mock.patch.object(service, "_await_harmonizer_model") as load_harmonizer,
        ):
            response = service.batch_render_camera(request, context)
        context.abort.assert_not_called()
        load_harmonizer.assert_not_called()
        self.assertEqual([item.success for item in response.items], [True, True])
        self.assertEqual([item.camera_name for item in response.items], ["0", "1"])
        self.assertEqual([item.result.camera_render_outputs[0].buffer_id for item in response.items], [0, 1])
        mgr.write_reserved_buffers.assert_called_once_with(
            "session", [0, 1], [image.data_ptr()] * 2, [9, 9], 2, 3, torch.cuda.current_stream().cuda_stream
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA IPC packing needs a CUDA device")
    def test_camera_batch_hint_preserves_fallback_and_oom_behavior(self):
        image = torch.rand(2, 3, 3, device="cuda")
        frame = mock.Mock(color_image=image)
        for error in (None, ValueError("batch unsupported"), torch.cuda.OutOfMemoryError("out of memory")):
            with self.subTest(error=error):
                service, mgr = self._service_with_mock_mgr()
                mgr.reserve_buffers.return_value = [0, 1]
                backend = mock.MagicMock()
                backend.render_camera_batch_shared_scene.return_value = (
                    [frame, frame],
                    _CameraBatchDispatch.NATIVE_PER_CAMERA_RAYS,
                )
                backend.render_camera_batch_shared_scene.side_effect = error
                backend.render_camera_request.return_value = (frame, None, None)
                request = BatchCameraRenderRequest(
                    prefer_camera_batching=True,
                    items=[
                        BatchCameraRenderRequestItem(
                            camera_name=str(i),
                            request=CameraRenderRequest(
                                scene_id="scene",
                                use_cuda_ipc=True,
                                ipc_session_id="session",
                                resolution_h=2,
                                resolution_w=3,
                                image_format=ImageFormat.RGB_UINT8_PLANAR,
                            ),
                        )
                        for i in range(2)
                    ],
                )
                context = mock.MagicMock()
                context.abort.side_effect = RuntimeError("aborted")
                with (
                    mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)),
                    mock.patch.object(service, "_apply_color_postprocessing_f32", return_value=image),
                ):
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        with self.assertRaisesRegex(RuntimeError, "aborted"):
                            service.batch_render_camera(request, context)
                        self.assertEqual(context.abort.call_args.args[0], grpc.StatusCode.RESOURCE_EXHAUSTED)
                        mgr.write_reserved_buffers.assert_not_called()
                        self.assertEqual(mgr.release_buffer.call_count, 2)
                    else:
                        response = service.batch_render_camera(request, context)
                        self.assertTrue(all(item.success for item in response.items))
                        mgr.write_reserved_buffers.assert_called_once()
                        mgr.release_buffer.assert_not_called()
                backend.render_camera_batch_shared_scene.assert_called_once()
                self.assertEqual(backend.render_camera_request.call_count, 2 if isinstance(error, ValueError) else 0)

    def test_camera_batch_fatal_shared_scene_error_propagates_without_retry(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.layout.return_value = (3, 2, False)
        mgr.reserve_buffers.return_value = [0, 1]
        backend = mock.MagicMock()
        backend.render_camera_batch_shared_scene.side_effect = RuntimeError(
            "CUDA error: an illegal memory access was encountered"
        )
        request = BatchCameraRenderRequest(
            prefer_camera_batching=True,
            items=[
                BatchCameraRenderRequestItem(
                    camera_name=str(index),
                    request=CameraRenderRequest(
                        scene_id="scene",
                        use_cuda_ipc=True,
                        ipc_session_id="session",
                        resolution_h=2,
                        resolution_w=3,
                        image_format=ImageFormat.RGB_UINT8_PLANAR,
                    ),
                )
                for index in range(2)
            ],
        )
        context = mock.MagicMock()
        stream = mock.MagicMock()

        with (
            mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)),
            mock.patch("nurec.grpc.serve.torch.cuda.device", return_value=contextlib.nullcontext()),
            mock.patch("nurec.grpc.serve.torch.cuda.current_stream", return_value=stream),
            mock.patch.object(service, "_recover_from_cuda_oom") as recover,
            self.assertRaisesRegex(RuntimeError, "illegal memory access"),
        ):
            service._batch_render_camera_ipc(request, context)

        backend.render_camera_batch_shared_scene.assert_called_once()
        backend.render_camera_request.assert_not_called()
        context.abort.assert_not_called()
        mgr.write_reserved_buffers.assert_not_called()
        self.assertEqual({call.args[1] for call in mgr.release_buffer.call_args_list}, {0, 1})
        recover.assert_not_called()

    def test_open_session_reclaims_allocation_if_event_export_fails(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.open_session.return_value = ("session", [b"m" * 64], 18, b"u" * 16, 3, 2)
        mgr.ready_event_handles.side_effect = RuntimeError("event export failed")
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            service.open_ipc_session(
                ipc_pb2.OpenIpcSessionRequest(requested_width=3, requested_height=2, requested_buffer_count=1),
                context,
            )
        mgr.close_session.assert_called_once_with("session")
        self.assertEqual(service._ipc_open_sessions, 0)
        self.assertEqual(service._ipc_reserved_bytes, 0)
        self.assertEqual(service._ipc_session_bytes, {})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA IPC packing needs a CUDA device")
    def test_camera_batch_packs_crops_on_current_stream_and_reclaims_failed_items(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.reserve_buffers.return_value = [0, 1, 2]
        images = [torch.rand(2, 4, 3, device="cuda"), torch.rand(3, 3, 3, device="cuda")]
        backend = mock.MagicMock()
        backend.render_camera_request.side_effect = [
            (mock.Mock(color_image=images[0]), None, None),
            ValueError("camera missing"),
            (mock.Mock(color_image=images[1]), None, None),
        ]
        request = BatchCameraRenderRequest(
            items=[
                BatchCameraRenderRequestItem(
                    camera_name=str(i),
                    request=CameraRenderRequest(
                        scene_id="scene",
                        use_cuda_ipc=True,
                        ipc_session_id="session",
                        resolution_h=2,
                        resolution_w=3,
                        image_format=ImageFormat.RGB_UINT8_PLANAR,
                        render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR],
                        **({"released_buffer_id": 3} if i == 0 else {}),
                    ),
                )
                for i in range(3)
            ]
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with (
            torch.cuda.stream(stream),
            mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)),
            mock.patch.object(service, "_apply_color_postprocessing_f32", side_effect=lambda image, *_: image),
        ):
            response = service.batch_render_camera(request, mock.MagicMock())
        self.assertEqual([item.success for item in response.items], [True, False, True])
        self.assertIn("camera missing", response.items[1].error_message)
        mgr.reserve_buffers.assert_called_once_with("session", [3], 3)
        mgr.write_reserved_buffers.assert_called_once_with(
            "session", [0, 2], [image.data_ptr() for image in images], [12, 9], 2, 3, stream.cuda_stream
        )
        mgr.release_buffer.assert_called_once_with("session", 1)
        stream.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA IPC packing needs a CUDA device")
    def test_camera_batch_failure_reclaims_unpublished_buffers(self):
        image = torch.rand(2, 3, 3, device="cuda")
        request = BatchCameraRenderRequest(
            items=[
                BatchCameraRenderRequestItem(
                    camera_name="camera",
                    request=CameraRenderRequest(
                        scene_id="scene",
                        use_cuda_ipc=True,
                        ipc_session_id="session",
                        resolution_h=2,
                        resolution_w=3,
                        image_format=ImageFormat.RGB_UINT8_PLANAR,
                        render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR],
                    ),
                )
            ]
        )
        for stage, status in (("render", grpc.StatusCode.RESOURCE_EXHAUSTED), ("pack", grpc.StatusCode.INTERNAL)):
            with self.subTest(stage=stage):
                service, mgr = self._service_with_mock_mgr()
                mgr.reserve_buffers.return_value = [0]
                backend = mock.MagicMock()
                backend.render_camera_request.return_value = (mock.Mock(color_image=image), None, None)
                if stage == "render":
                    backend.render_camera_request.side_effect = torch.OutOfMemoryError("render failed")
                else:
                    mgr.write_reserved_buffers.side_effect = RuntimeError("pack failed")
                context = mock.MagicMock()
                context.abort.side_effect = RuntimeError("aborted")
                with (
                    mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)),
                    mock.patch.object(service, "_apply_color_postprocessing_f32", return_value=image),
                    self.assertRaisesRegex(RuntimeError, "aborted"),
                ):
                    service.batch_render_camera(request, context)
                mgr.release_buffer.assert_called_once_with("session", 0)
                context.abort.assert_called_once()
                self.assertEqual(context.abort.call_args.args[0], status)
                self.assertIn(f"{stage} failed", context.abort.call_args.args[1])
                if stage == "render":
                    mgr.write_reserved_buffers.assert_not_called()

    def test_open_ipc_session_aborts_when_transport_unavailable(self):
        # _get_ipc_mgr() isn't stubbed here (unlike _service_with_mock_mgr), so this
        # exercises the real "native extension failed to import" path.
        service = _make_service_no_server()
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")

        with mock.patch("nurec.grpc.serve._CUDA_IPC_AVAILABLE", False):
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                service.open_ipc_session(ipc_pb2.OpenIpcSessionRequest(), context)

        context.abort.assert_called_once()
        args, _kwargs = context.abort.call_args
        self.assertEqual(args[0], grpc.StatusCode.INTERNAL)
        self.assertIn("CUDA IPC transport not available", args[1])
        # The session-count reservation taken before the transport lookup must be
        # rolled back on failure, or a run of failed opens would eventually trip
        # the max-sessions guard even though no session is actually open.
        self.assertEqual(service._ipc_open_sessions, 0)

    def test_get_ipc_mgr_constructs_native_manager_with_configured_max_buffer_count(self):
        # --cuda-ipc-max-buffer-count (GrpcServerConfig.cuda_ipc_max_buffer_count) must reach the
        # native IpcManager constructor, not just live in the config -- this is the only thing
        # that actually enforces the cap (see cuda_ipc_module.cpp's open_session).
        service = SensorSimService(
            server=None,
            artifacts_glob=None,
            ray_chunk_size=1,
            egocar_hoods_dir=None,
            downloader=None,
            scene_cache=None,
            cache_size=1,
            metrics_output_dir=None,
            enable_editing_actors=False,
            server_config=GrpcServerConfig(cuda_ipc_max_buffer_count=128),
        )
        with (
            mock.patch("nurec.grpc.serve._CUDA_IPC_AVAILABLE", True),
            mock.patch("nurec.grpc.serve.IpcManager") as mock_ipc_manager_cls,
        ):
            service._get_ipc_mgr()
        mock_ipc_manager_cls.assert_called_once_with(0, 128)

    def test_get_ipc_mgr_falls_back_to_config_default_without_a_server_config(self):
        service = SensorSimService(
            server=None,
            artifacts_glob=None,
            ray_chunk_size=1,
            egocar_hoods_dir=None,
            downloader=None,
            scene_cache=None,
            cache_size=1,
            metrics_output_dir=None,
            enable_editing_actors=False,
            server_config=None,
        )
        with (
            mock.patch("nurec.grpc.serve._CUDA_IPC_AVAILABLE", True),
            mock.patch("nurec.grpc.serve.IpcManager") as mock_ipc_manager_cls,
        ):
            service._get_ipc_mgr()
        mock_ipc_manager_cls.assert_called_once_with(
            0, GrpcServerConfig.model_fields["cuda_ipc_max_buffer_count"].default
        )

    def test_open_ipc_session_rolls_back_reservation_when_native_open_fails(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.open_session.side_effect = RuntimeError("cudaMalloc failed: out of memory")
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")

        with self.assertRaisesRegex(RuntimeError, "aborted"):
            service.open_ipc_session(
                ipc_pb2.OpenIpcSessionRequest(requested_width=8, requested_height=6, requested_buffer_count=2),
                context,
            )

        self.assertEqual(service._ipc_open_sessions, 0)
        # Failed opens must not consume the byte budget.
        self.assertEqual(service._ipc_reserved_bytes, 0)

    def test_open_ipc_session_aborts_resource_exhausted_at_max_total_bytes(self):
        # NRE_CUDA_IPC_MAX_SESSIONS alone doesn't bound GPU memory -- a single session can
        # request many large buffers -- so NRE_CUDA_IPC_MAX_TOTAL_BYTES caps the aggregate
        # footprint across every currently-open session too.
        service, mgr = self._service_with_mock_mgr()
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")

        with mock.patch.dict(os.environ, {"NRE_CUDA_IPC_MAX_TOTAL_BYTES": "100"}):
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                service.open_ipc_session(
                    # Two buffers require 4 MiB after allocation rounding.
                    ipc_pb2.OpenIpcSessionRequest(requested_width=8, requested_height=6, requested_buffer_count=2),
                    context,
                )

        mgr.open_session.assert_not_called()
        context.abort.assert_called_once()
        args, _kwargs = context.abort.call_args
        self.assertEqual(args[0], grpc.StatusCode.RESOURCE_EXHAUSTED)
        self.assertEqual(service._ipc_open_sessions, 0)
        self.assertEqual(service._ipc_reserved_bytes, 0)

    def test_open_ipc_session_and_close_ipc_session_track_reserved_bytes(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.open_session.return_value = ("sess-1", [b"h" * 64] * 2, 1234, b"u" * 16, 8, 6)

        service.open_ipc_session(
            ipc_pb2.OpenIpcSessionRequest(requested_width=8, requested_height=6, requested_buffer_count=2),
            mock.MagicMock(),
        )
        self.assertEqual(service._ipc_reserved_bytes, 4 * 1024**2)
        self.assertEqual(service._ipc_session_bytes, {"sess-1": 4 * 1024**2})

        mgr.close_session.return_value = True
        service.close_ipc_session(ipc_pb2.CloseIpcSessionRequest(session_id="sess-1"), mock.MagicMock())
        self.assertEqual(service._ipc_reserved_bytes, 0)
        self.assertNotIn("sess-1", service._ipc_session_bytes)

    def test_open_ipc_session_aborts_resource_exhausted_at_max_sessions(self):
        service, mgr = self._service_with_mock_mgr()
        service._ipc_open_sessions = 1
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")

        with mock.patch.dict(os.environ, {"NRE_CUDA_IPC_MAX_SESSIONS": "1"}):
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                service.open_ipc_session(
                    ipc_pb2.OpenIpcSessionRequest(requested_width=8, requested_height=6, requested_buffer_count=2),
                    context,
                )

        mgr.open_session.assert_not_called()
        context.abort.assert_called_once()
        args, _kwargs = context.abort.call_args
        self.assertEqual(args[0], grpc.StatusCode.RESOURCE_EXHAUSTED)
        # The count that made this call over-budget was reserved before the check
        # aborted, not left at the pre-call value.
        self.assertEqual(service._ipc_open_sessions, 1)

    def test_open_ipc_session_reserves_the_slot_before_calling_the_native_manager(self):
        # Regression test for a TOCTOU race: the count must be incremented in the same
        # locked section as the max-sessions check, not only after mgr.open_session
        # returns, or concurrent gRPC-threadpool calls could all read the same
        # pre-increment count and all pass the check before any of them finishes
        # allocating -- bypassing NRE_CUDA_IPC_MAX_SESSIONS entirely.
        service, mgr = self._service_with_mock_mgr()
        mgr.open_session.return_value = ("sess-1", [b"h" * 64], 1234, b"u" * 16, 8, 6)

        seen_count_during_native_call = []

        def _record_count_and_return(*_args, **_kwargs):
            seen_count_during_native_call.append(service._ipc_open_sessions)
            return mgr.open_session.return_value

        mgr.open_session.side_effect = _record_count_and_return

        service.open_ipc_session(
            ipc_pb2.OpenIpcSessionRequest(requested_width=8, requested_height=6, requested_buffer_count=1),
            mock.MagicMock(),
        )

        self.assertEqual(seen_count_during_native_call, [1])
        self.assertEqual(service._ipc_open_sessions, 1)

    def _render_one_ipc_frame(self, request_kwargs: dict) -> mock.MagicMock:
        # Allocates a real CUDA tensor to stand in for the raster _ipc_write_rgb receives.
        # Callers are individually GPU-skipped; this isn't a TestCase method itself.
        service, mgr = self._service_with_mock_mgr()
        mgr.write_render_hwc_f32.return_value = 1
        raster = torch.zeros(2, 3, 3, dtype=torch.float32, device="cuda")
        request = RGBRenderRequest(
            use_cuda_ipc=True,
            ipc_session_id="sess-1",
            image_format=ImageFormat.RGB_UINT8_PLANAR,
            **request_kwargs,
        )
        backend = mock.MagicMock()
        backend.render_rgb_request.return_value = (raster, (2, 3))
        with mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)):
            response = service.render_rgb(request, mock.MagicMock())
        self.assertEqual(response.buffer_id, 1)
        self.assertEqual(response.format, "rgb_uint8_planar")
        return mgr

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA IPC render path needs a CUDA device")
    def test_render_rgb_forwards_released_buffer_id_when_set(self):
        mgr = self._render_one_ipc_frame({"released_buffer_id": 0})
        mgr.write_render_hwc_f32.assert_called_once()
        (_session_id, released_buffer_id, *_rest), _kwargs = mgr.write_render_hwc_f32.call_args
        self.assertEqual(released_buffer_id, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA IPC render path needs a CUDA device")
    def test_render_rgb_forwards_none_when_released_buffer_id_absent(self):
        mgr = self._render_one_ipc_frame({})
        (_session_id, released_buffer_id, *_rest), _kwargs = mgr.write_render_hwc_f32.call_args
        self.assertIsNone(released_buffer_id)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA IPC render path needs a CUDA device")
    def test_render_rgb_ipc_requires_rgb_uint8_planar_format(self):
        # The CUDA IPC path always fills the session's buffers with rgb_uint8_planar
        # data regardless of image_format; a declared JPEG would otherwise pass a
        # JPEG-only --allowed-image-formats admission check while silently returning
        # raw planar bytes instead. Aborts before touching the backend, so no GPU or
        # mocked render is needed for this one.
        service, _mgr = self._service_with_mock_mgr()
        request = RGBRenderRequest(
            use_cuda_ipc=True,
            ipc_session_id="sess-1",
            image_format=ImageFormat.JPEG,
        )
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            service.render_rgb(request, context)
        context.abort.assert_called_once()
        args, _kwargs = context.abort.call_args
        self.assertEqual(args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("RGB_UINT8_PLANAR", args[1])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA IPC render path needs a CUDA device")
    def test_render_rgb_ipc_resolution_mismatch_aborts_invalid_argument(self):
        service, _mgr = self._service_with_mock_mgr()
        raster = torch.zeros(2, 3, 3, dtype=torch.float32, device="cuda")  # renders 3x2, request claims 8x8
        request = RGBRenderRequest(
            use_cuda_ipc=True,
            ipc_session_id="sess-1",
            image_format=ImageFormat.RGB_UINT8_PLANAR,
            resolution_h=8,
            resolution_w=8,
        )
        backend = mock.MagicMock()
        backend.render_rgb_request.return_value = (raster, (2, 3))
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")
        with mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)):
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                service.render_rgb(request, context)
        context.abort.assert_called_once()
        args, _kwargs = context.abort.call_args
        self.assertEqual(args[0], grpc.StatusCode.INVALID_ARGUMENT)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA IPC render path needs a CUDA device")
    def test_render_rgb_ipc_aborts_failed_precondition_when_device_mismatched(self):
        # A raster on a different physical GPU than NRE_CUDA_IPC_DEVICE can't be read by
        # the pack kernel there without peer access, which nothing requests or enables.
        service, mgr = self._service_with_mock_mgr()
        service._ipc_device_index = 99  # deliberately not the raster's real device (0)
        raster = torch.zeros(2, 3, 3, dtype=torch.float32, device="cuda")
        request = RGBRenderRequest(
            use_cuda_ipc=True,
            ipc_session_id="sess-1",
            image_format=ImageFormat.RGB_UINT8_PLANAR,
        )
        backend = mock.MagicMock()
        backend.render_rgb_request.return_value = (raster, (2, 3))
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")
        with mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)):
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                service.render_rgb(request, context)
        mgr.write_render_hwc_f32.assert_not_called()
        context.abort.assert_called_once()
        args, _kwargs = context.abort.call_args
        self.assertEqual(args[0], grpc.StatusCode.FAILED_PRECONDITION)
        self.assertIn("CUDA device", args[1])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA IPC render path needs a CUDA device")
    def test_render_camera_ipc_writes_into_the_selected_buffer(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.write_render_hwc_f32.return_value = 4
        raster = torch.zeros(2, 3, 3, dtype=torch.float32, device="cuda")
        request = CameraRenderRequest(
            use_cuda_ipc=True,
            ipc_session_id="sess-1",
            image_format=ImageFormat.RGB_UINT8_PLANAR,
        )
        backend = mock.MagicMock()
        backend.render_camera_request.return_value = (CameraFrame(color_image=raster), None, None)
        with mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)):
            response = service.render_camera(request, mock.MagicMock())

        self.assertEqual(len(response.camera_render_outputs), 1)
        output = response.camera_render_outputs[0]
        self.assertEqual(output.buffer_id, 4)
        self.assertEqual(output.format, "rgb_uint8_planar")
        self.assertEqual(output.render_product, CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR)
        self.assertFalse(output.image_bytes)
        mgr.write_render_hwc_f32.assert_called_once()

    def test_render_camera_ipc_requires_single_color_product(self):
        # The IPC path always delivers exactly one RGB_UINT8_PLANAR COLOR output written by the
        # fused pack kernel; it has no way to also produce a DISTANCE_COLORIZED output (a
        # different dtype/layout with no pack kernel of its own).
        service, _mgr = self._service_with_mock_mgr()
        request = CameraRenderRequest(
            use_cuda_ipc=True,
            ipc_session_id="sess-1",
            image_format=ImageFormat.RGB_UINT8_PLANAR,
            render_products=[
                CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR,
                CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED,
            ],
        )
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            service.render_camera(request, context)
        context.abort.assert_called_once()
        args, _kwargs = context.abort.call_args
        self.assertEqual(args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("CAMERA_RENDER_PRODUCT_COLOR", args[1])

    def test_render_camera_ipc_rejects_bounding_box_overlay(self):
        # The IPC path never materializes a host-visible uint8 image for the numpy-based
        # bounding-box overlay to draw onto.
        service, _mgr = self._service_with_mock_mgr()
        request = CameraRenderRequest(
            use_cuda_ipc=True,
            ipc_session_id="sess-1",
            image_format=ImageFormat.RGB_UINT8_PLANAR,
            overlay_bounding_boxes=OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_MODEL,
        )
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            service.render_camera(request, context)
        context.abort.assert_called_once()
        args, _kwargs = context.abort.call_args
        self.assertEqual(args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("overlay_bounding_boxes", args[1])

    def test_render_camera_ipc_requires_rgb_uint8_planar_format(self):
        service, _mgr = self._service_with_mock_mgr()
        request = CameraRenderRequest(
            use_cuda_ipc=True,
            ipc_session_id="sess-1",
            image_format=ImageFormat.JPEG,
        )
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            service.render_camera(request, context)
        context.abort.assert_called_once()
        args, _kwargs = context.abort.call_args
        self.assertEqual(args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("RGB_UINT8_PLANAR", args[1])

    def _ipc_batch_item(self, camera_name: str, *, released_buffer_id: int | None = None) -> BatchRGBRenderRequestItem:
        request = RGBRenderRequest(
            scene_id="scene",
            use_cuda_ipc=True,
            ipc_session_id="sess-1",
            image_format=ImageFormat.RGB_UINT8_PLANAR,
        )
        if released_buffer_id is not None:
            request.released_buffer_id = released_buffer_id
        return BatchRGBRenderRequestItem(camera_name=camera_name, request=request)

    def test_batch_render_rgb_ipc_requires_a_single_shared_session(self):
        # Every use_cuda_ipc item's buffer comes from one session's pool, reserved in one
        # call; there is no single pool to check "does this batch fit" against otherwise.
        service, _mgr = self._service_with_mock_mgr()
        request = BatchRGBRenderRequest(
            items=[
                self._ipc_batch_item("cam_0"),
                BatchRGBRenderRequestItem(
                    camera_name="cam_1",
                    request=RGBRenderRequest(
                        scene_id="scene",
                        use_cuda_ipc=True,
                        ipc_session_id="sess-2",
                        image_format=ImageFormat.RGB_UINT8_PLANAR,
                    ),
                ),
            ]
        )
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            service.batch_render_rgb(request, context)
        context.abort.assert_called_once()
        args, _kwargs = context.abort.call_args
        self.assertEqual(args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("ipc_session_id", args[1])

    def test_batch_render_rgb_ipc_insufficient_free_buffers_aborts_resource_exhausted(self):
        # The "reject the whole batch" design: if the batch's IPC items don't all fit in
        # the session's currently-free buffers, none of them are written -- no partial
        # fulfillment, and no item is left holding a buffer the others couldn't get.
        service, mgr = self._service_with_mock_mgr()
        mgr.reserve_buffers.side_effect = RuntimeError(
            "insufficient free CUDA IPC buffers: requested 2, 1 free of 2 total in session sess-1"
        )
        request = BatchRGBRenderRequest(items=[self._ipc_batch_item("cam_0"), self._ipc_batch_item("cam_1")])
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            service.batch_render_rgb(request, context)
        mgr.reserve_buffers.assert_called_once_with("sess-1", [], 2)
        context.abort.assert_called_once()
        args, _kwargs = context.abort.call_args
        self.assertEqual(args[0], grpc.StatusCode.RESOURCE_EXHAUSTED)

    def test_batch_render_rgb_ipc_reservation_applies_released_buffer_ids(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.reserve_buffers.return_value = [0]
        request = BatchRGBRenderRequest(items=[self._ipc_batch_item("cam_0", released_buffer_id=4)])
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")
        # Rendering itself isn't under test here; bail out right after reservation so this
        # test needs no GPU or backend mock.
        with mock.patch.object(service, "get_backend", side_effect=RuntimeError("stop before render")):
            with self.assertRaises(RuntimeError):
                service.batch_render_rgb(request, context)
        mgr.reserve_buffers.assert_called_once_with("sess-1", [4], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA IPC render path needs a CUDA device")
    def test_batch_render_rgb_ipc_writes_into_reserved_buffers(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.reserve_buffers.return_value = [3, 5]
        request = BatchRGBRenderRequest(items=[self._ipc_batch_item("cam_0"), self._ipc_batch_item("cam_1")])
        raster = torch.zeros(2, 3, 3, dtype=torch.float32, device="cuda")
        backend = mock.MagicMock()
        with (
            mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)),
            mock.patch.object(service, "_render_single_camera_raster_for_batch", return_value=raster),
        ):
            result = service.batch_render_rgb(request, mock.MagicMock())

        mgr.reserve_buffers.assert_called_once_with("sess-1", [], 2)
        self.assertEqual([call.args[1] for call in mgr.write_reserved_buffer.call_args_list], [3, 5])
        self.assertTrue(all(item.success for item in result.items))
        self.assertEqual([item.result.buffer_id for item in result.items], [3, 5])
        self.assertEqual([item.result.format for item in result.items], ["rgb_uint8_planar", "rgb_uint8_planar"])

    def test_batch_render_rgb_ipc_item_failure_releases_its_reserved_buffer(self):
        # A reserved buffer whose render then fails has no successful result to carry its
        # id back to the client, so it must be freed here or it would sit in_use forever.
        service, mgr = self._service_with_mock_mgr()
        mgr.reserve_buffers.return_value = [3]
        request = BatchRGBRenderRequest(items=[self._ipc_batch_item("cam_0")])
        backend = mock.MagicMock()
        with (
            mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)),
            mock.patch.object(
                service, "_render_single_camera_raster_for_batch", side_effect=RuntimeError("render failed")
            ),
        ):
            result = service.batch_render_rgb(request, mock.MagicMock())

        mgr.release_buffer.assert_called_once_with("sess-1", 3)
        self.assertFalse(result.items[0].success)

    def test_batch_render_rgb_ipc_cleanup_failure_preserves_item_error(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.reserve_buffers.return_value = [3]
        mgr.release_buffer.side_effect = RuntimeError("release failed")
        request = BatchRGBRenderRequest(items=[self._ipc_batch_item("cam_0")])
        with (
            mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(mock.MagicMock())),
            mock.patch.object(
                service, "_render_single_camera_raster_for_batch", side_effect=ValueError("invalid camera")
            ),
        ):
            result = service.batch_render_rgb(request, mock.MagicMock())

        mgr.release_buffer.assert_called_once_with("sess-1", 3)
        self.assertFalse(result.items[0].success)
        self.assertEqual(result.items[0].error_message, "INVALID_ARGUMENT: invalid camera")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA IPC render path needs a CUDA device")
    def test_batch_render_rgb_allows_mixing_ipc_and_encoded_items(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.reserve_buffers.return_value = [2]
        request = BatchRGBRenderRequest(
            items=[
                self._ipc_batch_item("cam_ipc"),
                BatchRGBRenderRequestItem(
                    camera_name="cam_jpeg",
                    request=RGBRenderRequest(scene_id="scene", image_format=ImageFormat.JPEG, image_quality=90),
                ),
            ]
        )
        raster = torch.zeros(2, 3, 3, dtype=torch.float32, device="cuda")
        backend = mock.MagicMock()
        encoder = mock.MagicMock()
        encoder.max_batch_size = BATCH_RENDER_REQUEST_CHUNK_SIZE
        encoder.encode.side_effect = lambda images, quality: [b"jpeg" for _ in images]
        with (
            mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)),
            mock.patch.object(service, "_render_single_camera_raster_for_batch", return_value=raster),
            mock.patch.object(
                service, "_render_single_camera_image_for_batch", return_value=torch.zeros((2, 2, 3), dtype=torch.uint8)
            ),
            mock.patch.object(service, "_get_nvjpeg_batch_encoder", return_value=encoder),
        ):
            result = service.batch_render_rgb(request, mock.MagicMock())

        mgr.reserve_buffers.assert_called_once_with("sess-1", [], 1)
        self.assertTrue(all(item.success for item in result.items))
        self.assertEqual(result.items[0].result.buffer_id, 2)
        self.assertEqual(result.items[1].result.image_bytes, b"jpeg")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA IPC render path needs a CUDA device")
    def test_batch_render_rgb_ipc_oom_releases_every_outstanding_reserved_buffer(self):
        # A batch-level abort (OOM re-raised past the per-item handler on purpose, to trigger
        # _recover_from_cuda_oom) means no BatchRGBRenderReturn is ever sent -- so even an
        # earlier item's already-*successful* buffer, which the per-item handler never touches,
        # must be released here too, or it leaks as in_use until the session is closed.
        service, mgr = self._service_with_mock_mgr()
        mgr.reserve_buffers.return_value = [3, 5, 7]
        request = BatchRGBRenderRequest(
            items=[self._ipc_batch_item("cam_ok"), self._ipc_batch_item("cam_oom"), self._ipc_batch_item("cam_never")]
        )
        raster = torch.zeros(2, 3, 3, dtype=torch.float32, device="cuda")

        def _raster_side_effect(_backend, item, *_args, **_kwargs):
            if item.camera_name == "cam_oom":
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return raster

        backend = mock.MagicMock()
        context = mock.MagicMock()
        with (
            mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)),
            mock.patch.object(service, "_render_single_camera_raster_for_batch", side_effect=_raster_side_effect),
            mock.patch.object(service, "_recover_from_cuda_oom"),
        ):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                service.batch_render_rgb(request, context)

        # cam_ok's buffer (3) succeeded but its result never reached the client; cam_oom's
        # buffer (5) was reserved but never written; cam_never's buffer (7) was never even
        # attempted (the OOM aborts the whole batch before its turn). All three must come back.
        self.assertEqual(
            {call.args[1] for call in mgr.release_buffer.call_args_list},
            {3, 5, 7},
        )

    def test_batch_render_rgb_ipc_releases_reserved_buffers_when_get_backend_fails_before_any_item_renders(self):
        # An exception outside the per-item loop entirely (e.g. the scene cache being full)
        # strands every reserved buffer just as surely as a mid-item OOM would, since the
        # per-item handler never runs at all.
        service, mgr = self._service_with_mock_mgr()
        mgr.reserve_buffers.return_value = [1, 2]
        request = BatchRGBRenderRequest(items=[self._ipc_batch_item("cam_0"), self._ipc_batch_item("cam_1")])
        context = mock.MagicMock()
        with mock.patch.object(service, "get_backend", side_effect=RuntimeError("scene cache full")):
            with self.assertRaises(RuntimeError):
                service.batch_render_rgb(request, context)

        self.assertEqual(
            {call.args[1] for call in mgr.release_buffer.call_args_list},
            {1, 2},
        )

    def test_batch_render_rgb_ipc_cleanup_failure_preserves_batch_error(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.reserve_buffers.return_value = [1, 2]
        mgr.release_buffer.side_effect = RuntimeError("release failed")
        request = BatchRGBRenderRequest(items=[self._ipc_batch_item("cam_0"), self._ipc_batch_item("cam_1")])
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")
        with mock.patch.object(service, "get_backend", side_effect=ValueError("invalid scene")):
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                service.batch_render_rgb(request, context)

        context.abort.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT, "invalid scene")
        self.assertEqual(mgr.release_buffer.call_count, 2)
        self.assertEqual({call.args[1] for call in mgr.release_buffer.call_args_list}, {1, 2})


class TestRdmaTransport(unittest.TestCase):
    """Session lifecycle and the use_rdma render branches, with a mocked RdmaManager standing in
    for the native extension (only built with --config=rdma, and needing RoCE hardware)."""

    def _service_with_mock_mgr(self) -> tuple[SensorSimService, mock.MagicMock]:
        service = _make_service_no_server()
        mgr = mock.MagicMock()
        mgr.cuda_device = 0
        service._get_rdma_mgr = mock.MagicMock(return_value=mgr)
        return service, mgr

    def _abort_args(self, call: Callable[[mock.MagicMock], object]) -> tuple:
        context = mock.MagicMock()
        context.abort.side_effect = RuntimeError("aborted")
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            call(context)
        context.abort.assert_called_once()
        args, _kwargs = context.abort.call_args
        return args

    def test_open_rdma_session_returns_server_qp_info_and_counts_the_session(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.open_session.return_value = ("sess-1", b"g" * 16, 3, 42, 7)
        request = rdma_pb2.OpenRdmaSessionRequest(gid=b"c" * 16, gid_index=1, qp_num=9, psn=5, max_buf_size=1024)

        response = service.open_rdma_session(request, mock.MagicMock())

        mgr.open_session.assert_called_once_with(b"c" * 16, 9, 5, 1024)
        self.assertEqual(response.session_id, "sess-1")
        self.assertEqual(response.gid, b"g" * 16)
        self.assertEqual((response.gid_index, response.qp_num, response.psn), (3, 42, 7))
        self.assertEqual(service._rdma_open_sessions, 1)

    def test_open_rdma_session_is_unimplemented_without_the_transport(self):
        # _get_rdma_mgr() isn't stubbed here, so this exercises the real "not built with
        # --config=rdma" path: not a server error, so not INTERNAL.
        service = _make_service_no_server()
        with mock.patch("nurec.grpc.serve._rdma", None):
            args = self._abort_args(
                lambda context: service.open_rdma_session(rdma_pb2.OpenRdmaSessionRequest(), context)
            )
        self.assertEqual(args[0], grpc.StatusCode.UNIMPLEMENTED)
        self.assertIn("--config=rdma", args[1])
        self.assertEqual(service._rdma_open_sessions, 0)

    def test_open_rdma_session_maps_native_argument_errors_to_invalid_argument(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.open_session.side_effect = ValueError("max_buf_size must be > 0")
        args = self._abort_args(lambda context: service.open_rdma_session(rdma_pb2.OpenRdmaSessionRequest(), context))
        self.assertEqual(args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(service._rdma_open_sessions, 0)

    def test_open_rdma_session_reports_native_setup_failures_as_internal(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.open_session.side_effect = RuntimeError("ibv_reg_mr(GPU staging) failed — check nvidia_peermem")
        args = self._abort_args(lambda context: service.open_rdma_session(rdma_pb2.OpenRdmaSessionRequest(), context))
        self.assertEqual(args[0], grpc.StatusCode.INTERNAL)
        self.assertIn("nvidia_peermem", args[1])
        self.assertEqual(service._rdma_open_sessions, 0)

    def test_register_receive_buffer_reports_native_errors_in_the_response(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.register_buffer.side_effect = KeyError("session not found: missing")
        request = rdma_pb2.RegisterReceiveBufferRequest(
            session_id="missing", remote_addr=0x1000, rkey=11, buffer_size=64, buffer_id=2
        )

        response = service.register_receive_buffer(request, mock.MagicMock())

        mgr.register_buffer.assert_called_once_with("missing", 0x1000, 11, 64, 2)
        self.assertFalse(response.success)
        self.assertEqual(response.error_message, "session not found: missing")

    def test_close_rdma_session_decrements_only_when_a_session_was_removed(self):
        service, mgr = self._service_with_mock_mgr()
        service._rdma_open_sessions = 1
        mgr.close_session.return_value = False

        response = service.close_rdma_session(rdma_pb2.CloseRdmaSessionRequest(session_id="missing"), mock.MagicMock())

        self.assertTrue(response.success)
        self.assertEqual(service._rdma_open_sessions, 1)  # unchanged: no session was actually closed

        mgr.close_session.return_value = True
        service.close_rdma_session(rdma_pb2.CloseRdmaSessionRequest(session_id="sess-1"), mock.MagicMock())
        self.assertEqual(service._rdma_open_sessions, 0)

    def test_close_rdma_session_soft_fails_without_the_transport(self):
        service = _make_service_no_server()
        with mock.patch("nurec.grpc.serve._rdma", None):
            response = service.close_rdma_session(rdma_pb2.CloseRdmaSessionRequest(session_id="s"), mock.MagicMock())
        self.assertFalse(response.success)
        self.assertIn("RDMA transport not available", response.error_message)

    @unittest.skipUnless(torch.cuda.is_available(), "RDMA render path needs a CUDA device")
    def test_render_rgb_rdma_writes_the_aspect_fit_render_into_the_registered_buffer(self):
        # Unlike CUDA IPC, RDMA has no session-fixed buffer shape to match: a render whose
        # aspect-fit extent differs from the requested resolution is still delivered, at its own
        # size, and bytes_written tells the client how much landed.
        service, mgr = self._service_with_mock_mgr()
        mgr.rdma_write_render_hwc_f32.return_value = 18
        raster = torch.zeros(2, 3, 3, dtype=torch.float32, device="cuda")  # renders 3x2, request claims 8x8
        request = RGBRenderRequest(
            use_rdma=True,
            rdma_session_id="sess-1",
            rdma_buffer_id=5,
            image_format=ImageFormat.RGB_UINT8_PLANAR,
            resolution_h=8,
            resolution_w=8,
        )
        backend = mock.MagicMock()
        backend.render_rgb_request.return_value = (raster, (2, 3))
        with mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)):
            response = service.render_rgb(request, mock.MagicMock())

        self.assertEqual(response.bytes_written, 18)
        self.assertEqual(response.buffer_id, 5)
        self.assertEqual(response.format, "rgb_uint8_planar")
        self.assertFalse(response.image_bytes)
        (session_id, buffer_id, _ptr, height, width), _kwargs = mgr.rdma_write_render_hwc_f32.call_args
        self.assertEqual((session_id, buffer_id, height, width), ("sess-1", 5, 2, 3))

    @unittest.skipUnless(torch.cuda.is_available(), "RDMA render path needs a CUDA device")
    def test_render_camera_rdma_writes_into_the_registered_buffer(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.rdma_write_render_hwc_f32.return_value = 18
        raster = torch.zeros(2, 3, 3, dtype=torch.float32, device="cuda")
        request = CameraRenderRequest(
            use_rdma=True,
            rdma_session_id="sess-1",
            rdma_buffer_id=5,
            image_format=ImageFormat.RGB_UINT8_PLANAR,
        )
        backend = mock.MagicMock()
        backend.render_camera_request.return_value = (CameraFrame(color_image=raster), None, None)
        with mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)):
            response = service.render_camera(request, mock.MagicMock())

        self.assertEqual(len(response.camera_render_outputs), 1)
        output = response.camera_render_outputs[0]
        self.assertEqual(output.bytes_written, 18)
        self.assertEqual(output.buffer_id, 5)
        self.assertEqual(output.format, "rgb_uint8_planar")
        self.assertEqual((output.image_height, output.image_width), (2, 3))
        self.assertEqual(output.render_product, CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR)
        self.assertFalse(output.image_bytes)

    @unittest.skipUnless(torch.cuda.is_available(), "RDMA render path needs a CUDA device")
    def test_render_rgb_rdma_refuses_a_render_off_the_staging_device(self):
        service, mgr = self._service_with_mock_mgr()
        mgr.cuda_device = 1
        backend = mock.MagicMock()
        backend.render_rgb_request.return_value = (torch.zeros(2, 3, 3, dtype=torch.float32, device="cuda:0"), (2, 3))
        request = RGBRenderRequest(use_rdma=True, rdma_session_id="sess-1", image_format=ImageFormat.RGB_UINT8_PLANAR)
        with mock.patch.object(service, "get_backend", return_value=contextlib.nullcontext(backend)):
            args = self._abort_args(lambda context: service.render_rgb(request, context))

        self.assertEqual(args[0], grpc.StatusCode.FAILED_PRECONDITION)
        self.assertIn("render output is on CUDA device 0", args[1])
        mgr.rdma_write_render_hwc_f32.assert_not_called()

    def test_render_rgb_rejects_use_rdma_with_use_cuda_ipc(self):
        service, _mgr = self._service_with_mock_mgr()
        request = RGBRenderRequest(use_rdma=True, use_cuda_ipc=True, image_format=ImageFormat.RGB_UINT8_PLANAR)
        args = self._abort_args(lambda context: service.render_rgb(request, context))
        self.assertEqual(args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("mutually exclusive", args[1])

    def test_render_camera_rejects_use_rdma_with_use_cuda_ipc(self):
        service, _mgr = self._service_with_mock_mgr()
        request = CameraRenderRequest(use_rdma=True, use_cuda_ipc=True, image_format=ImageFormat.RGB_UINT8_PLANAR)
        args = self._abort_args(lambda context: service.render_camera(request, context))
        self.assertEqual(args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("mutually exclusive", args[1])

    def test_render_rgb_rdma_requires_rgb_uint8_planar_format(self):
        service, _mgr = self._service_with_mock_mgr()
        request = RGBRenderRequest(use_rdma=True, rdma_session_id="sess-1", image_format=ImageFormat.JPEG)
        args = self._abort_args(lambda context: service.render_rgb(request, context))
        self.assertEqual(args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("use_rdma=true requires image_format=RGB_UINT8_PLANAR", args[1])

    def test_render_camera_rdma_rejects_bounding_box_overlay(self):
        service, _mgr = self._service_with_mock_mgr()
        request = CameraRenderRequest(
            use_rdma=True,
            rdma_session_id="sess-1",
            image_format=ImageFormat.RGB_UINT8_PLANAR,
            overlay_bounding_boxes=OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_MODEL,
        )
        args = self._abort_args(lambda context: service.render_camera(request, context))
        self.assertEqual(args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("the RDMA path writes directly into the client's registered buffer", args[1])

    def test_batch_render_rgb_rejects_use_rdma(self):
        service, _mgr = self._service_with_mock_mgr()
        request = BatchRGBRenderRequest(
            items=[
                BatchRGBRenderRequestItem(
                    camera_name="cam",
                    request=RGBRenderRequest(
                        scene_id="scene", use_rdma=True, image_format=ImageFormat.RGB_UINT8_PLANAR
                    ),
                )
            ]
        )
        args = self._abort_args(lambda context: service.batch_render_rgb(request, context))
        self.assertEqual(args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("batch_render_rgb does not support use_rdma", args[1])

    def test_batch_render_camera_rejects_use_rdma(self):
        service, _mgr = self._service_with_mock_mgr()
        request = BatchCameraRenderRequest(
            items=[
                BatchCameraRenderRequestItem(
                    camera_name="cam",
                    request=CameraRenderRequest(
                        scene_id="scene", use_rdma=True, image_format=ImageFormat.RGB_UINT8_PLANAR
                    ),
                )
            ]
        )
        args = self._abort_args(lambda context: service.batch_render_camera(request, context))
        self.assertEqual(args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("batch_render_camera does not support use_rdma", args[1])

    def test_proto_skew_warning_fires_once_while_a_session_is_open(self):
        service, _mgr = self._service_with_mock_mgr()
        with self.assertNoLogs("nurec.grpc.serve", level="WARNING"):
            service._warn_once_on_rdma_proto_skew(RGBRenderRequest(), "render_rgb")  # no session open
        service._rdma_open_sessions = 1
        with self.assertNoLogs("nurec.grpc.serve", level="WARNING"):
            service._warn_once_on_rdma_proto_skew(RGBRenderRequest(use_rdma=True), "render_rgb")
        with self.assertLogs("nurec.grpc.serve", level="WARNING") as logs:
            service._warn_once_on_rdma_proto_skew(RGBRenderRequest(), "render_rgb")
            service._warn_once_on_rdma_proto_skew(CameraRenderRequest(), "render_camera")
        self.assertEqual(len(logs.records), 1)
        self.assertIn("proto version skew", logs.records[0].getMessage())

    def test_rdma_errors_are_classified_by_type(self):
        class PreconditionError(RuntimeError):
            pass

        with mock.patch("nurec.grpc.serve._rdma", mock.MagicMock(PreconditionError=PreconditionError)):
            for exc, status in (
                (ValueError("max_buf_size must be > 0"), grpc.StatusCode.INVALID_ARGUMENT),
                (KeyError("session not found: abc"), grpc.StatusCode.NOT_FOUND),
                (PreconditionError("buffer_id not registered: 3"), grpc.StatusCode.FAILED_PRECONDITION),
                (RdmaTransportUnavailableError("RDMA transport not available"), grpc.StatusCode.UNIMPLEMENTED),
                (
                    RuntimeError("render output is on CUDA device 1, but the RDMA staging buffers are on device 0"),
                    grpc.StatusCode.FAILED_PRECONDITION,
                ),
                # Link/hardware failures stay loud rather than being blamed on the client.
                (RuntimeError("ibv_poll_cq timed out after 5s waiting for completion"), grpc.StatusCode.UNKNOWN),
            ):
                with self.subTest(exc=exc):
                    self.assertEqual(_classify_render_exception(exc)[0], status)

    def test_frame_dump_writes_the_wire_bytes(self):
        service = _make_service_no_server()
        raster = torch.tensor([[[0.5, 0.0, 1.0], [0.25, 0.75, 2.0]]], dtype=torch.float32)
        request = RGBRenderRequest(
            use_rdma=True,
            image_format=ImageFormat.RGB_UINT8_PLANAR,
            camera_intrinsics=CameraSpec(logical_id="camera_front_tele_30fov"),
            frame_start_us=1707499198100038,
        )
        with tempfile.TemporaryDirectory() as dump_dir:
            with mock.patch.dict(os.environ, {"NRE_RDMA_FRAME_DUMP_DIR": dump_dir}):
                service._maybe_dump_rdma_frame(raster, request)
            (path,) = Path(dump_dir).iterdir()
            self.assertEqual(path.name, "camera_front_tele_30fov_1707499198100038.bin")
            # floor(255*x) saturated, then CHW planar -- the RDMA payload's own arithmetic.
            expected = np.array([[[127, 0, 255], [63, 191, 255]]], dtype=np.uint8).transpose(2, 0, 1)
            np.testing.assert_array_equal(np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(3, 1, 2), expected)

    def test_frame_dump_writes_nothing_without_the_env_var(self):
        service = _make_service_no_server()
        request = RGBRenderRequest(use_rdma=True, image_format=ImageFormat.RGB_UINT8_PLANAR)
        with (
            mock.patch.dict(os.environ, {"NRE_RDMA_FRAME_DUMP_DIR": ""}),
            mock.patch.object(Path, "write_bytes") as write_bytes,
        ):
            service._maybe_dump_rdma_frame(torch.zeros(1, 2, 3), request)
        write_bytes.assert_not_called()


class TestRenderResultToImageAndMappingSemantics(unittest.TestCase):
    def test_sparse_class_names_uses_max_id_plus_one_for_lut_size(self):
        service = _make_service_no_server()
        raster = torch.tensor([[0, 5], [3, 5]], dtype=torch.int64)
        image, mapping = service._render_result_to_image_and_mapping(
            raster_tensor=raster,
            product=CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED,
            distance_colormap_vmax_m=None,
            request=CameraRenderRequest(),
            semantic_class_colors=None,
            semantic_class_names={0: "bg", 5: "ped"},
        )
        self.assertIsNotNone(mapping)
        self.assertEqual(len(mapping), 6)
        self.assertEqual(mapping[0][:2], (0, "bg"))
        for i in range(1, 5):
            self.assertEqual(mapping[i][:2], (i, f"class_{i}"))
        self.assertEqual(mapping[5][:2], (5, "ped"))
        self.assertEqual(image.dtype, torch.uint8)
        self.assertEqual(image.shape, (2, 2, 3))

    def test_dense_class_names_uses_max_id_plus_one(self):
        service = _make_service_no_server()
        raster = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        _, mapping = service._render_result_to_image_and_mapping(
            raster_tensor=raster,
            product=CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED,
            distance_colormap_vmax_m=None,
            request=CameraRenderRequest(),
            semantic_class_colors=None,
            semantic_class_names={0: "bg", 1: "car", 2: "ped"},
        )
        self.assertEqual(len(mapping), 3)
        self.assertEqual([entry[1] for entry in mapping], ["bg", "car", "ped"])

    def test_no_class_names_uses_default_lut_size(self):
        service = _make_service_no_server()
        raster = torch.tensor([[0, 1]], dtype=torch.int64)
        _, mapping = service._render_result_to_image_and_mapping(
            raster_tensor=raster,
            product=CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED,
            distance_colormap_vmax_m=None,
            request=CameraRenderRequest(),
            semantic_class_colors=None,
            semantic_class_names=None,
        )
        self.assertEqual(len(mapping), SEMANTICS_COLOR_LUT_DEFAULT_SIZE)
        for i, entry in enumerate(mapping):
            self.assertEqual(entry[:2], (i, f"class_{i}"))

    def test_negative_semantic_id_renders_as_miss_black(self):
        service = _make_service_no_server()
        raster = torch.tensor([[-1, 0], [1, 2]], dtype=torch.int64)
        image, mapping = service._render_result_to_image_and_mapping(
            raster_tensor=raster,
            product=CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED,
            distance_colormap_vmax_m=None,
            request=CameraRenderRequest(),
            semantic_class_colors=None,
            semantic_class_names={0: "bg", 1: "car", 2: "ped"},
        )
        self.assertEqual(image.dtype, torch.uint8)
        self.assertEqual(image.shape, (2, 2, 3))
        self.assertEqual(tuple(image[0, 0].tolist()), (0, 0, 0))
        self.assertNotEqual(tuple(image[0, 1].tolist()), (0, 0, 0))
        self.assertEqual(len(mapping), 3)
        self.assertTrue(all(entry[0] >= 0 for entry in mapping))
        self.assertEqual([entry[1] for entry in mapping], ["bg", "car", "ped"])

    def test_negative_semantic_id_renders_as_miss_black_without_names(self):
        service = _make_service_no_server()
        raster = torch.tensor([[-1, 0]], dtype=torch.int64)
        image, mapping = service._render_result_to_image_and_mapping(
            raster_tensor=raster,
            product=CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED,
            distance_colormap_vmax_m=None,
            request=CameraRenderRequest(),
            semantic_class_colors=None,
            semantic_class_names=None,
        )
        self.assertEqual(tuple(image[0, 0].tolist()), (0, 0, 0))
        self.assertNotEqual(tuple(image[0, 1].tolist()), (0, 0, 0))
        self.assertEqual(len(mapping), SEMANTICS_COLOR_LUT_DEFAULT_SIZE)
        self.assertTrue(all(entry[0] >= 0 for entry in mapping))

    def test_pathological_class_id_caps_lut_size(self):
        from nurec.grpc.serve import _SEMANTICS_COLOR_LUT_MAX_SIZE

        service = _make_service_no_server()
        raster = torch.tensor([[0, 1]], dtype=torch.int64)
        _, mapping = service._render_result_to_image_and_mapping(
            raster_tensor=raster,
            product=CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED,
            distance_colormap_vmax_m=None,
            request=CameraRenderRequest(),
            semantic_class_colors=None,
            semantic_class_names={0: "bg", 10_000_000: "huge"},
        )
        self.assertEqual(len(mapping), _SEMANTICS_COLOR_LUT_MAX_SIZE)


class TestApplyBoundingBoxOverlayGate(unittest.TestCase):
    def test_unspecified_source_returns_input_unchanged(self):
        service = _make_service_no_server()
        image = torch.zeros((4, 4, 3), dtype=torch.uint8)
        out = service._apply_bounding_box_overlay_if_requested(image, CameraRenderRequest(), mock.MagicMock())
        self.assertIs(out, image)

    def test_artifact_source_no_tracks_raises(self):
        service = _make_service_no_server()
        image = torch.zeros((4, 4, 3), dtype=torch.uint8)
        request = CameraRenderRequest(
            render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR],
            overlay_bounding_boxes=OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_ARTIFACT,
        )
        backend = mock.MagicMock()
        backend._get_artifact_actor_tracks.return_value.num_tracks.return_value = 0
        with self.assertRaises(ValueError):
            service._apply_bounding_box_overlay_if_requested(image, request, backend)

    def test_model_source_without_snapshot_is_internal_error(self):
        service = _make_service_no_server()
        image = torch.zeros((4, 4, 3), dtype=torch.uint8)
        request = CameraRenderRequest(
            render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR],
            overlay_bounding_boxes=OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_MODEL,
        )
        with self.assertRaises(RuntimeError):
            service._apply_bounding_box_overlay_if_requested(image, request, mock.MagicMock())

    def test_model_source_with_empty_snapshot_returns_input_unchanged(self):
        from nurec.render.actors import ActorsSnapshot

        service = _make_service_no_server()
        image = torch.zeros((4, 4, 3), dtype=torch.uint8)
        request = CameraRenderRequest(
            render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR],
            overlay_bounding_boxes=OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_MODEL,
        )
        empty_snapshot = ActorsSnapshot.empty()
        out = service._apply_bounding_box_overlay_if_requested(
            image, request, mock.MagicMock(), model_actors_snapshot=empty_snapshot
        )
        self.assertIs(out, image)

    def test_unknown_overlay_source_raises(self):
        service = _make_service_no_server()
        image = torch.zeros((4, 4, 3), dtype=torch.uint8)
        request = CameraRenderRequest(render_products=[CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR])
        # Patch the source resolver to return a bogus enum value; proto3 enum fields accept any
        # int at the wire layer, so this is the cleanest way to exercise the default branch.
        with mock.patch("nurec.grpc.serve._overlay_bounding_box_source", return_value=999):
            with self.assertRaisesRegex(ValueError, "Unknown overlay_bounding_boxes value"):
                service._apply_bounding_box_overlay_if_requested(image, request, mock.MagicMock())


def _make_actor_tracks_with_one_track(
    track_id: str,
    cuboid_dims: tuple,
) -> "ActorTracks":
    """Build a real ActorTracks with a single track using actor_tracks_from_grpc.

    The track has two keyframes at t=0 and t=500_000 us with the actor at the origin.
    """

    grpc_track = DynamicObjectTrack()
    grpc_track.id = track_id
    grpc_track.object_size.size_x = cuboid_dims[0]
    grpc_track.object_size.size_y = cuboid_dims[1]
    grpc_track.object_size.size_z = cuboid_dims[2]

    for ts_us in [0, 500_000]:
        pat = PoseAtTime()
        pat.pose.quat.w = 1.0
        pat.timestamp_us = ts_us
        grpc_track.trajectory.poses.append(pat)

    return actor_tracks_from_grpc([grpc_track])


class TestActorsSnapshot(unittest.TestCase):
    def test_duplicate_cleaned_track_ids_are_rejected(self):
        for first_id, second_id in [("actor@a", "actor@b"), ("actor@source", "actor"), ("actor", "actor")]:
            with self.subTest(first_id=first_id, second_id=second_id):
                first = DynamicObject(track_id=first_id)
                first.pose_pair.start_pose.quat.w = 1.0
                first.pose_pair.end_pose.quat.w = 1.0
                second = DynamicObject(track_id=second_id)
                second.pose_pair.start_pose.quat.w = 1.0
                second.pose_pair.end_pose.quat.w = 1.0

                with self.assertRaises(ValueError) as error:
                    actors_snapshot_from_render_request(CameraRenderRequest(dynamic_objects=[first, second]))
                self.assertEqual(
                    str(error.exception),
                    f"DynamicObject IDs {first_id!r} and {second_id!r} both normalize to 'actor'",
                )


class TestBackendRenderCameraRequestModelOverlay(unittest.TestCase):
    """Backend.render_camera_request MODEL overlay: bounding-box snapshot must follow request-derived
    actor poses when dynamic_objects are present, not the stale model track poses.

    Regression for: MODEL overlay snapshot was rebuilt from persistent model tracks, ignoring any
    dynamic_objects pose overrides passed to the renderer.  Boxes therefore stayed at their original
    model positions even after --demo-actor-transform moved the rendered actors.
    """

    def _make_backend_with_actor_tracks(self, actor_tracks):
        """Return a Backend whose renderable_model is wired up for MODEL-overlay testing."""

        renderable_model = mock.MagicMock()
        renderable_model.supports_edit_actors.return_value = True
        renderable_model.get_actor_tracks.return_value = actor_tracks
        renderable_model.render_camera_frame_from_ray_bundle.return_value = CameraFrame(
            color_image=torch.zeros(4, 4, 3, dtype=torch.uint8)
        )

        camera_bank = mock.MagicMock()
        ray_bundle = mock.MagicMock()
        ray_bundle.rendering_data.h = 4
        ray_bundle.rendering_data.w = 4
        camera_bank._get_camera.return_value.build_ray_bundle.return_value = ray_bundle

        return Backend(
            renderable_model=renderable_model,
            camera_bank=camera_bank,
            world_to_nurec=mock.MagicMock(),
            scene_time_range_us=HalfClosedInterval(0, 1),
            lidar_bank=mock.MagicMock(),
            asset_bank=mock.MagicMock(),
            source_artifact=mock.MagicMock(),
            backend_cache_key=_test_backend_cache_key(),
        )

    def test_dynamic_objects_pose_used_for_overlay_not_model_track_pose(self):
        """When dynamic_objects are present, the MODEL overlay snapshot must use the request
        poses (not the model-track interpolated poses) so rendered actors and their boxes align."""
        track_id = "actor-42"
        dims = (4.0, 2.0, 1.8)
        actor_tracks = _make_actor_tracks_with_one_track(track_id, dims)
        backend = self._make_backend_with_actor_tracks(actor_tracks)

        # Send a pose that is deliberately different from the model track (which sits at origin).
        override_tx, override_ty, override_tz = 10.0, 20.0, 30.0
        dynamic_obj = DynamicObject()
        dynamic_obj.track_id = track_id
        dynamic_obj.pose_pair.start_pose.vec.x = override_tx
        dynamic_obj.pose_pair.start_pose.vec.y = override_ty
        dynamic_obj.pose_pair.start_pose.vec.z = override_tz
        dynamic_obj.pose_pair.start_pose.quat.w = 1.0
        dynamic_obj.pose_pair.end_pose.vec.x = override_tx
        dynamic_obj.pose_pair.end_pose.vec.y = override_ty
        dynamic_obj.pose_pair.end_pose.vec.z = override_tz
        dynamic_obj.pose_pair.end_pose.quat.w = 1.0

        request = CameraRenderRequest(
            overlay_bounding_boxes=OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_MODEL,
            dynamic_objects=[dynamic_obj],
            frame_start_us=0,
            frame_end_us=100_000,
        )

        _, _, snapshot = backend.render_camera_request(request, enable_editing_actors=True)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.actor_ids, [track_id])

        # Poses must come from the request, not the model track (which is at origin).
        start_pose = snapshot.actor_poses[0, 0]  # [tx, ty, tz, qx, qy, qz, qw]
        self.assertAlmostEqual(float(start_pose[0]), override_tx, places=5)
        self.assertAlmostEqual(float(start_pose[1]), override_ty, places=5)
        self.assertAlmostEqual(float(start_pose[2]), override_tz, places=5)

        # Cuboid dims must come from the model tracks (compare on CPU; CuboidTracks may be on GPU).
        self.assertIsNotNone(snapshot.cuboid_dims)
        torch.testing.assert_close(
            snapshot.cuboid_dims.cpu(),
            torch.tensor([list(dims)], dtype=torch.float32),
        )

    def test_unknown_dynamic_object_track_id_raises_descriptive_error(self):
        """A MODEL overlay with a dynamic_object whose track_id is absent from the model tracks must
        abort with a descriptive error naming the unknown id and listing the known ids, not the
        cryptic 'No cuboid dimensions for actor track id ...' from the bare get_cuboid_dims lookup."""
        actor_tracks = _make_actor_tracks_with_one_track("actor-42", (4.0, 2.0, 1.8))
        backend = self._make_backend_with_actor_tracks(actor_tracks)

        dynamic_obj = DynamicObject()
        dynamic_obj.track_id = "actor-99999-not-in-checkpoint"
        dynamic_obj.pose_pair.start_pose.quat.w = 1.0
        dynamic_obj.pose_pair.end_pose.quat.w = 1.0

        request = CameraRenderRequest(
            overlay_bounding_boxes=OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_MODEL,
            dynamic_objects=[dynamic_obj],
            frame_start_us=0,
            frame_end_us=100_000,
        )

        with self.assertRaises(ValueError) as ctx:
            backend.render_camera_request(request, enable_editing_actors=True)
        message = str(ctx.exception)
        self.assertIn("actor-99999-not-in-checkpoint", message)  # names the offending id
        self.assertIn("actor-42", message)  # lists the known id(s)
        self.assertIn("MODEL", message)  # attributes the failure to the MODEL overlay path

    def test_no_dynamic_objects_falls_back_to_model_track_snapshot(self):
        """Without dynamic_objects, the MODEL overlay snapshot is still derived from model tracks
        (existing behaviour; this test guards against regression in the fallback path)."""
        track_id = "actor-42"
        dims = (4.0, 2.0, 1.8)
        actor_tracks = _make_actor_tracks_with_one_track(track_id, dims)
        backend = self._make_backend_with_actor_tracks(actor_tracks)

        request = CameraRenderRequest(
            overlay_bounding_boxes=OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_MODEL,
            frame_start_us=0,
            frame_end_us=100_000,
        )

        _, _, snapshot = backend.render_camera_request(request, enable_editing_actors=False)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.actor_ids, [track_id])
        # Cuboid dims still come from the model tracks (compare on CPU; CuboidTracks may be on GPU).
        self.assertIsNotNone(snapshot.cuboid_dims)
        torch.testing.assert_close(
            snapshot.cuboid_dims.cpu(),
            torch.tensor([list(dims)], dtype=torch.float32),
        )
        # Pose should be near origin (the model track pose at t=0).
        start_pose = snapshot.actor_poses[0, 0]
        self.assertAlmostEqual(float(start_pose[0]), 0.0, places=4)
        self.assertAlmostEqual(float(start_pose[1]), 0.0, places=4)
        self.assertAlmostEqual(float(start_pose[2]), 0.0, places=4)


def _make_backend() -> Backend:
    return Backend(
        renderable_model=mock.MagicMock(),
        camera_bank=mock.MagicMock(),
        world_to_nurec=mock.MagicMock(),
        scene_time_range_us=HalfClosedInterval(0, 1),
        lidar_bank=mock.MagicMock(),
        asset_bank=mock.MagicMock(),
        source_artifact=mock.MagicMock(),
        backend_cache_key=_test_backend_cache_key(),
    )


class TestGetArtifactActorTracks(unittest.TestCase):
    def test_empty_sequence_tracks_returns_empty(self):
        backend = _make_backend()
        backend.source_artifact.sequence_tracks = {}
        result = backend._get_artifact_actor_tracks()
        self.assertEqual(result.num_tracks(), 0)

    def test_passes_controllable_only_false_to_from_cuboid_tracks(self):
        backend = _make_backend()
        backend.source_artifact.sequence_tracks = {"seq": {"placeholder": True}}
        with mock.patch("nurec.grpc.serve.CuboidTracks.from_dict", return_value=mock.MagicMock()):
            with mock.patch("nurec.grpc.serve.ActorTracks._from_cuboid_tracks") as mock_from_cuboid:
                mock_from_cuboid.return_value = mock.MagicMock()
                backend._get_artifact_actor_tracks()
                _, kwargs = mock_from_cuboid.call_args
                self.assertIs(kwargs.get("controllable_only"), False)

    def test_propagates_load_failure(self):
        backend = _make_backend()
        # Scoped patch on the MagicMock class avoids leaking sequence_tracks=PropertyMock to other tests.
        # create=True is needed because the MagicMock class doesn't have a real sequence_tracks attribute.
        with mock.patch.object(
            type(backend.source_artifact),
            "sequence_tracks",
            new_callable=mock.PropertyMock,
            side_effect=KeyError("missing"),
            create=True,
        ):
            with self.assertRaises(KeyError):
                backend._get_artifact_actor_tracks()

    def test_concurrent_initialization_is_thread_safe(self):
        backend = _make_backend()
        backend.source_artifact.sequence_tracks = {"seq": {"placeholder": True}}
        num_threads = 8
        barrier = threading.Barrier(num_threads)

        # Sleep inside the patched parse so threads that lose the lock race wait long enough for
        # the winner to populate the cache; a delay ensures the test fails if the lock is absent.
        def _slow_from_dict(_value):
            time.sleep(0.05)
            return mock.MagicMock()

        sentinel_actor_tracks = mock.MagicMock()
        with mock.patch("nurec.grpc.serve.CuboidTracks.from_dict", side_effect=_slow_from_dict) as mock_from_dict:
            with mock.patch(
                "nurec.grpc.serve.ActorTracks._from_cuboid_tracks",
                return_value=sentinel_actor_tracks,
            ) as mock_from_cuboid:

                def _call():
                    barrier.wait()  # release all threads together to maximize lock contention
                    return backend._get_artifact_actor_tracks()

                with futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
                    results = list(pool.map(lambda _: _call(), range(num_threads)))

                # Lock-safe one-time init: parsing and the ActorTracks build each run exactly once.
                self.assertEqual(mock_from_dict.call_count, 1)
                self.assertEqual(mock_from_cuboid.call_count, 1)
        # Every thread sees the same cached ActorTracks instance.
        self.assertEqual(len(results), num_threads)
        for r in results:
            self.assertIs(r, sentinel_actor_tracks)

    def test_concurrent_load_failure_propagates_to_every_caller(self):
        backend = _make_backend()
        num_threads = 8
        barrier = threading.Barrier(num_threads)

        # Scoped patch on the MagicMock class avoids leaking sequence_tracks=PropertyMock to other tests.
        # create=True is needed because the MagicMock class doesn't have a real sequence_tracks attribute.
        with mock.patch.object(
            type(backend.source_artifact),
            "sequence_tracks",
            new_callable=mock.PropertyMock,
            side_effect=KeyError("missing"),
            create=True,
        ):

            def _call():
                barrier.wait()
                try:
                    backend._get_artifact_actor_tracks()
                except KeyError:
                    return "raised"
                return "no-raise"

            with futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
                outcomes = list(pool.map(lambda _: _call(), range(num_threads)))
        self.assertTrue(all(o == "raised" for o in outcomes), outcomes)


class TestBackendFromArtifact(unittest.TestCase):
    def _call_from_artifact(self, artifact, renderer=Backend.RendererType.DEFAULT):
        with mock.patch("nurec.grpc.serve.RenderableModel.load_from_artifact") as load_mock:
            with mock.patch("nurec.grpc.serve.RigTrajectories") as rig_mock:
                with mock.patch("nurec.grpc.serve.CameraBank"):
                    with mock.patch("nurec.grpc.serve.LidarBank"):
                        with mock.patch("nurec.grpc.serve.AssetBank"):
                            load_mock.return_value = mock.MagicMock()
                            parsed_rig = rig_mock.from_dict.return_value
                            parsed_rig.world_to_nurec = mock.MagicMock()
                            rig_trajectory = mock.MagicMock()
                            rig_trajectory.T_rig_world_timestamps_us = torch.tensor([1_000, 1_199])
                            parsed_rig.rig_trajectories = [rig_trajectory]
                            Backend.from_artifact(
                                artifact,
                                renderer,
                                backend_cache_key=_test_backend_cache_key(),
                            )
                            return load_mock.call_args

    def test_semantic_artifact_enables_extended_features_and_distance(self):
        artifact = mock.MagicMock()
        artifact.semantic_class_names = {0: "bg", 1: "car"}
        call_args = self._call_from_artifact(artifact)
        overrides = call_args.kwargs.get("config_overrides", ())
        self.assertIn("model.renderer.outputs.camera.enable_extended_features=True", overrides)
        self.assertIn("model.renderer.outputs.camera.enable_sensor_features=True", overrides)
        self.assertIn("model.renderer.outputs.camera.enable_distance=True", overrides)

    def test_non_semantic_artifact_enables_distance_but_not_extended_features(self):
        artifact = mock.MagicMock()
        artifact.semantic_class_names = None
        call_args = self._call_from_artifact(artifact)
        overrides = call_args.kwargs.get("config_overrides", ())
        # Distance is a first-class CameraRender product; it must be enabled regardless of semantic
        # status so DISTANCE_COLORIZED works consistently across renderers (gsplat honors the gate).
        self.assertIn("model.renderer.outputs.camera.enable_distance=True", overrides)
        # Extended/sensor features are only required for semantic logits.
        self.assertFalse(any("enable_extended_features" in o for o in overrides))
        self.assertFalse(any("enable_sensor_features" in o for o in overrides))

    def test_non_semantic_artifact_with_gsplat_renderer_enables_distance(self):
        # Regression guard: a non-semantic artifact served under gsplat must still enable the
        # distance gate, otherwise gsplat renders in "RGB" mode and DISTANCE_COLORIZED fails.
        artifact = mock.MagicMock()
        artifact.semantic_class_names = None
        call_args = self._call_from_artifact(artifact, renderer=Backend.RendererType.GSPLAT)
        overrides = call_args.kwargs.get("config_overrides", ())
        self.assertIn("model.renderer.name=3dgut-gsplat", overrides)
        self.assertIn("model.renderer.outputs.camera.enable_distance=True", overrides)
        self.assertFalse(any("enable_extended_features" in o for o in overrides))

    def test_semantic_artifact_with_gsplat_renderer_includes_both_gsplat_and_semantic_overrides(self):
        artifact = mock.MagicMock()
        artifact.semantic_class_names = {0: "bg"}
        call_args = self._call_from_artifact(artifact, renderer=Backend.RendererType.GSPLAT)
        overrides = call_args.kwargs.get("config_overrides", ())
        self.assertIn("model.renderer.name=3dgut-gsplat", overrides)
        self.assertIn("model.renderer.outputs.camera.enable_extended_features=True", overrides)
        self.assertIn("model.renderer.outputs.camera.enable_distance=True", overrides)

    def test_prepared_backend_stays_on_cpu_until_activation(self):
        artifact = mock.MagicMock()
        artifact.semantic_class_names = None
        renderable = mock.MagicMock()
        world_to_nurec = mock.MagicMock()
        with (
            mock.patch("nurec.grpc.serve.RenderableModel.load_from_artifact", return_value=renderable) as load,
            mock.patch("nurec.grpc.serve.RigTrajectories") as trajectories,
            mock.patch("nurec.grpc.serve.CameraBank") as camera_bank,
            mock.patch("nurec.grpc.serve.LidarBank") as lidar_bank,
            mock.patch("nurec.grpc.serve.AssetBank"),
        ):
            trajectories.from_dict.return_value.world_to_nurec = world_to_nurec
            trajectories.from_dict.return_value.rig_trajectories = [
                mock.Mock(T_rig_world_timestamps_us=torch.tensor([100, 200]))
            ]

            prepared = PreparedBackend.from_artifact(
                artifact,
                Backend.RendererType.DEFAULT,
                1,
                backend_cache_key=_test_backend_cache_key(),
            )

        self.assertEqual(load.call_args.kwargs["device"], "cpu")
        self.assertEqual(prepared.scene_time_range_us, HalfClosedInterval(100, 201))
        camera_bank.assert_called_once_with(artifact, world_to_nurec, prepared.scene_time_range_us)
        lidar_bank.assert_called_once_with(artifact, enabled=True)
        renderable.to.assert_not_called()

    def test_prepared_backend_activates_renderable_model(self):
        renderable = mock.MagicMock()
        lidar_bank = mock.MagicMock()
        prepared = PreparedBackend(
            renderable_model=renderable,
            camera_bank=mock.MagicMock(),
            world_to_nurec=mock.MagicMock(),
            scene_time_range_us=HalfClosedInterval(100, 201),
            lidar_bank=lidar_bank,
            asset_bank=mock.MagicMock(),
            source_artifact=mock.MagicMock(),
            backend_cache_key=_test_backend_cache_key(),
        )
        with (
            mock.patch("nurec.grpc.serve.torch.cuda.empty_cache"),
            mock.patch("nurec.grpc.serve.gc.collect"),
            mock.patch("nurec.grpc.serve._install_edit_overrides") as install_edit_overrides,
        ):
            backend = prepared.activate()

        renderable.to.assert_called_once_with("cuda")
        install_edit_overrides.assert_called_once_with(backend, prepared.source_artifact)
        self.assertIs(backend.renderable_model, renderable)
        self.assertEqual(backend.scene_time_range_us, prepared.scene_time_range_us)

    def test_disabled_lidar_bank_does_not_parse_artifact(self):
        artifact = mock.MagicMock()
        with mock.patch("nurec.grpc.serve.RigTrajectories.from_dict") as trajectories:
            lidar_bank = LidarBank(artifact, enabled=False)

        trajectories.assert_not_called()
        self.assertIsNone(lidar_bank.get_entry())


class TestDownloadSceneToCacheTOCTOU(unittest.TestCase):
    """`download_scene_to_cache` must tolerate a scene evicted between `has_scene` and `get_scene_path`."""

    def test_eviction_between_has_scene_and_get_path_falls_through_to_download(self):
        # `has_scene` reports a hit on both the lock-free fast path and the re-check under the lock, but the
        # entry is evicted before each `get_scene_path`, so both fast paths raise KeyError. The function must
        # treat that as a miss and (re-)download rather than surfacing the KeyError to the caller.
        scene_cache = mock.MagicMock(spec=SceneCache)
        scene_cache.has_scene.return_value = True
        scene_cache.get_scene_path.side_effect = KeyError("evicted between check and lookup")
        scene_cache.lock_scene_path.return_value = contextlib.nullcontext()
        redownloaded_path = Path("/tmp/nurec-toctou-redownloaded.usdz")
        scene_cache.add_scene.return_value = redownloaded_path

        downloader = mock.MagicMock()
        downloader.download.return_value = 4096

        result = download_scene_to_cache("scene-x", "https://example.test/scene.usdz", scene_cache, downloader)

        self.assertEqual(result, redownloaded_path)
        downloader.download.assert_called_once()
        scene_cache.add_scene.assert_called_once()

    def test_add_scene_valueerror_from_concurrent_register_returns_existing_path(self):
        # A concurrent registrant (e.g. a separate process sharing the cache dir, outside this process's
        # in-memory lock) caches scene_id between our under-lock miss check and add_scene, so add_scene
        # raises ValueError("... already in cache"). The function must fall through to get_scene_path and
        # return the winner's cached path rather than surfacing the benign race — and clean up its own temp.
        scene_cache = mock.MagicMock(spec=SceneCache)
        scene_cache.has_scene.return_value = False  # miss on both fast paths -> we download
        scene_cache.lock_scene_path.return_value = contextlib.nullcontext()
        scene_cache.add_scene.side_effect = ValueError("Scene scene-x already in cache")
        winner_path = Path("/tmp/nurec-race-winner.usdz")
        scene_cache.get_scene_path.return_value = winner_path

        downloader = mock.MagicMock()
        downloader.download.return_value = 4096

        with (
            mock.patch("nurec.grpc.serve.os.path.exists", return_value=True),
            mock.patch("nurec.grpc.serve.os.unlink") as unlink_mock,
        ):
            result = download_scene_to_cache("scene-x", "https://example.test/scene.usdz", scene_cache, downloader)

        self.assertEqual(result, winner_path)  # the winner's path, not a raised ValueError
        scene_cache.add_scene.assert_called_once()
        scene_cache.get_scene_path.assert_called_once_with("scene-x")
        unlink_mock.assert_called_once()  # our now-unused temp download was cleaned up

    def test_add_scene_valueerror_then_get_scene_path_keyerror_retries_download(self):
        # F022: add_scene loses the concurrent-register race (ValueError), but the winner's entry is itself
        # evicted before we can read it, so get_scene_path raises KeyError. The function must treat that as a
        # retryable miss and re-run the locked download, rather than surfacing the KeyError to the caller as a
        # confusing INVALID_ARGUMENT.
        scene_cache = mock.MagicMock(spec=SceneCache)
        scene_cache.has_scene.return_value = False  # miss on the fast paths -> we download each pass
        scene_cache.lock_scene_path.return_value = contextlib.nullcontext()
        redownloaded_path = Path("/tmp/nurec-race-retry.usdz")
        # First add_scene loses the race; the winner's entry then vanishes (get_scene_path KeyError); the
        # retry's add_scene wins.
        scene_cache.add_scene.side_effect = [ValueError("Scene scene-x already in cache"), redownloaded_path]
        scene_cache.get_scene_path.side_effect = KeyError("winner evicted before read")

        downloader = mock.MagicMock()
        downloader.download.return_value = 4096

        with (
            mock.patch("nurec.grpc.serve.os.path.exists", return_value=True),
            mock.patch("nurec.grpc.serve.os.unlink"),
        ):
            result = download_scene_to_cache("scene-x", "https://example.test/scene.usdz", scene_cache, downloader)

        self.assertEqual(result, redownloaded_path)  # retried and won, not a raised KeyError
        self.assertEqual(scene_cache.add_scene.call_count, 2)  # first lost the race, the retry succeeded
        self.assertEqual(downloader.download.call_count, 2)  # re-downloaded on the retry


class TestResidentBackendRegistration(unittest.TestCase):
    """`register_resident_backend` — the seam the multi-process workers use.

    A worker's backend aliases the owner's IPC weight storage and is installed straight into the
    `BackendCache`. Since `get_backend` resolves the artifact registry BEFORE the cache, such a scene
    needs a registration or it reads NOT_FOUND while resident; and that registration must never be used
    to REBUILD the scene (a rebuild would load a private, non-aliased copy).
    """

    @staticmethod
    def _resident_backend(scene_id: str = "resident-scene", sha: str = "a" * 64) -> Backend:
        artifact = mock.MagicMock()
        artifact.sha256_footprint = sha
        return Backend(
            renderable_model=None,  # type: ignore[arg-type]
            camera_bank=None,  # type: ignore[arg-type]
            world_to_nurec=None,  # type: ignore[arg-type]
            scene_time_range_us=HalfClosedInterval(0, 1),
            lidar_bank=None,  # type: ignore[arg-type]
            asset_bank=None,  # type: ignore[arg-type]
            source_artifact=artifact,
            backend_cache_key=BackendCacheKey(SceneCacheKey(scene_id=scene_id), sha),
        )

    def test_registers_under_the_key_a_plain_request_resolves(self):
        # `_request_scene_cache_key` builds SceneCacheKey(scene_id=...) with no canonical_url for a request
        # carrying no x-nre-scene-url header (which is every worker-bound call: the proxy consumes that
        # header itself). Registering under any other key would leave the resident scene NOT_FOUND.
        service = _make_service_no_server()
        backend = self._resident_backend()
        service.register_resident_backend(backend)

        expected_key = SensorSimService._request_scene_cache_key("resident-scene").key_hash
        self.assertIn(expected_key, service._artifact_registrations)
        registration = service._artifact_registrations[expected_key]
        self.assertTrue(registration.resident_only)
        # The registry-derived backend key must match the key the worker publishes the backend under,
        # or get_backend would look up a cache entry that does not exist.
        self.assertEqual(registration.backend_cache_key.key_hash, backend.backend_cache_key.key_hash)

    def test_unregister_removes_it(self):
        service = _make_service_no_server()
        backend = self._resident_backend()
        service.register_resident_backend(backend)
        service.unregister_resident_backend(backend.backend_cache_key)
        self.assertEqual(service._artifact_registrations, {})
        # Idempotent: a detach retry after the record is already gone must not raise.
        service.unregister_resident_backend(backend.backend_cache_key)

    def test_cache_miss_on_a_resident_scene_fails_loud_instead_of_rebuilding(self):
        # The resident backend is gone (detached/evicted) but the registration is still present. get_backend
        # must NOT call Backend.from_artifact to rebuild it -- that would produce a non-aliased copy that
        # silently diverges from the owner's IPC-shared weights.
        service = _make_service_no_server()
        service.register_resident_backend(self._resident_backend())

        with mock.patch("nurec.grpc.serve.Backend.from_artifact") as from_artifact:
            with self.assertRaises(SceneNotFoundError) as ctx:
                with service.get_backend(SceneCacheKey(scene_id="resident-scene")):
                    pass
        self.assertIn("no longer resident", str(ctx.exception))
        from_artifact.assert_not_called()  # never rebuilt

    def test_a_normal_registration_still_rebuilds_on_a_cache_miss(self):
        # Guard the guard: the resident_only branch must not suppress the ordinary rebuild path.
        service = _make_service_no_server()
        artifact = mock.MagicMock()
        artifact.sha256_footprint = "b" * 64
        artifact.source = Path("/tmp/whatever.usdz")
        scene_key = SceneCacheKey(scene_id="normal-scene")
        service._artifact_registrations[scene_key.key_hash] = ArtifactRegistration(
            artifact=artifact, downloaded=False, scene_key=scene_key
        )

        rebuilt = self._resident_backend(scene_id="normal-scene", sha="b" * 64)
        with (
            mock.patch("nurec.grpc.serve.Backend.from_artifact", return_value=rebuilt) as from_artifact,
            mock.patch("nurec.grpc.serve.Artifact", return_value=artifact),
            mock.patch("nurec.grpc.serve._local_file_fingerprint", return_value=(1, 2, 3, 4, 5)),
        ):
            artifact.temporary_cache.return_value.__enter__.return_value = artifact
            with service.get_backend(scene_key) as backend:
                self.assertIs(backend, rebuilt)
        from_artifact.assert_called_once()


def test_record_duration_declares_its_timing_method():
    # Every timing metric in the file needs a method: the published render rate differs by ~7%
    # between the CUDA-event and wall-clock readings, against a 5% regression tolerance, so a
    # consumer must never have to infer which one it is holding.
    collector = MetricsCollector(mode="grpc_server", run_id="timing-method")
    manager = ServerMetricsManager(None)
    manager._metrics_collector = collector
    manager._metrics_output_dir = None

    manager.record_duration("grpc.decode.render_rgb_ms", 1.5)

    assert collector._metric_metadata["grpc.decode.render_rgb_ms"]["timing_method"] == "wall_clock"


class TestLegacyServiceName(unittest.TestCase):
    """Clients built from the pre-26.10 protos call the service under its old package name."""

    class _Servicer(SensorsimServiceServicer):
        def get_version(self, request, context):
            return VersionId(version_id="x")

        def render_rgb(self, request, context):
            return RGBRenderReturn(image_bytes=b"ok")

    def test_old_and_new_service_names_are_served(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        add_sensorsim_servicer_to_server(self._Servicer(), server)
        port = server.add_insecure_port("localhost:0")
        server.start()
        try:
            with grpc.insecure_channel(f"localhost:{port}") as channel:
                legacy = f"/{LEGACY_SENSORSIM_SERVICE_NAME}"
                get_version = channel.unary_unary(
                    f"{legacy}/get_version",
                    request_serializer=Empty.SerializeToString,
                    response_deserializer=VersionId.FromString,
                )
                render_rgb = channel.unary_unary(
                    f"{legacy}/render_rgb",
                    request_serializer=RGBRenderRequest.SerializeToString,
                    response_deserializer=RGBRenderReturn.FromString,
                )
                self.assertEqual(get_version(Empty(), timeout=10).version_id, "x")
                self.assertEqual(render_rgb(RGBRenderRequest(), timeout=10).image_bytes, b"ok")
                self.assertEqual(SensorsimServiceStub(channel).get_version(Empty(), timeout=10).version_id, "x")
        finally:
            server.stop(None)
