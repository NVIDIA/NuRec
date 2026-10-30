# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the live aspect-state gRPC surface: DynamicObject.aspect_state round-trips
on the wire, and Backend._update_aspect_states applies EXACTLY the named aspects
(no evaluation server-side — the sender owns all animation/semantics)."""

import hashlib
import json
import tempfile
import unittest
import zipfile

from concurrent import futures
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest import mock

import grpc
import torch

from nurec.artifact import Artifact
from nurec.asset_edits.runtime import (
    EDIT_MANIFEST_FILENAME,
    AspectOverrideRuntime,
    AspectScheduleRegistry,
    AspectSwitch,
    BakedCluster,
    EditManifest,
    EditSchedule,
    TrackSchedule,
)
from nurec.grpc.cache_keys import SceneCacheKey
from nurec.grpc.protos.sensorsim_pb2 import (
    CameraRenderRequest,
    DynamicObject,
    LidarRenderRequest,
    RGBRenderRequest,
    RGBRenderReturn,
)
from nurec.grpc.protos.sensorsim_pb2_grpc import (
    SensorsimServiceStub,
    add_SensorsimServiceServicer_to_server,
)
from nurec.grpc.serve import (
    ArtifactRegistration,
    Backend,
    GrpcServerConfig,
    SensorSimService,
    _install_edit_overrides,
    actors_snapshot_from_render_request,
)


# The aspects the fake scene's bake produced. Deliberately meaningless strings — the
# server matches them byte-for-byte and assigns them no meaning — but they must be
# DECLARED, because the server rejects an aspect no baked cluster carries.
_BAKED_ASPECTS = {
    "170": {"opaque:key_a", "another/opaque.key-b", "file:aspect", "x"},
    "171": {"detailstate:off", "detailstate:pulse"},
}


def _make_backend(
    track_index: dict | None = None, track_aspects: dict | None = None
) -> tuple[Backend, AspectScheduleRegistry]:
    """Builds a Backend with mocked banks, a fresh registry, and track '170' indexed to ('dyn', 2)."""
    registry = AspectScheduleRegistry()
    backend = Backend(
        renderable_model=mock.MagicMock(),
        camera_bank=mock.MagicMock(),
        world_to_nurec=mock.MagicMock(),
        lidar_bank=mock.MagicMock(),
        asset_bank=mock.MagicMock(),
        source_artifact=mock.MagicMock(),
        backend_cache_key=mock.MagicMock(),
        aspect_registry=registry,
        edit_track_index={"170": ("dyn", 2)} if track_index is None else track_index,
        edit_track_aspects=dict(_BAKED_ASPECTS) if track_aspects is None else track_aspects,
    )
    return backend, registry


def _request(track_id: str, aspects=None, gains=None) -> RGBRenderRequest:
    """Builds a one-object render request; aspects=None leaves the aspect_state field unset."""
    request = RGBRenderRequest(frame_start_us=1_000_000, frame_end_us=1_000_002)
    dyn = request.dynamic_objects.add()
    dyn.track_id = track_id
    if aspects is not None:
        dyn.aspect_state.SetInParent()
        dyn.aspect_state.aspects.extend(aspects)
        for aspect, gain, cold in gains or []:
            g = dyn.aspect_state.gains.add()
            g.aspect = aspect
            g.gain = gain
            if cold:
                g.cold_rgb.extend(cold)
    return request


class AspectStateProtoTest(unittest.TestCase):
    """Wire-level behavior of the DynamicObject.aspect_state proto field."""

    def test_wire_roundtrip_and_presence(self):
        """Aspects and gains survive serialization, and an empty aspect_state still reports presence."""
        # Realistic aspect names here only DOCUMENT the bake's naming convention
        # ("<control>:<state>", "<control>:shared:<key>"); the wire and server treat
        # them as opaque strings.
        obj = DynamicObject(track_id="170")
        self.assertFalse(obj.HasField("aspect_state"))
        obj.aspect_state.aspects.extend(["appearancemode:alternate", "panelstate:on"])
        g = obj.aspect_state.gains.add()
        g.aspect = "detailstate:shared:Detail/surface"
        g.gain = 0.42
        g.cold_rgb.extend([0.1, 0.2, 0.3])

        parsed = DynamicObject.FromString(obj.SerializeToString())
        self.assertTrue(parsed.HasField("aspect_state"))
        self.assertEqual(list(parsed.aspect_state.aspects), ["appearancemode:alternate", "panelstate:on"])
        self.assertAlmostEqual(parsed.aspect_state.gains[0].gain, 0.42, places=6)

        # Explicit empty aspects (all-hidden) is distinguishable from absent.
        empty = DynamicObject(track_id="170")
        empty.aspect_state.SetInParent()
        self.assertTrue(DynamicObject.FromString(empty.SerializeToString()).HasField("aspect_state"))

    def test_field_numbers_frozen(self):
        """DynamicObject field numbers stay pinned so serialized messages remain wire-compatible."""
        fields = {f.name: f.number for f in DynamicObject.DESCRIPTOR.fields}
        self.assertEqual(fields, {"track_id": 1, "pose_pair": 2, "aspect_state": 3})


class UpdateAspectStatesTest(unittest.TestCase):
    """Aspect keys below are deliberately MEANINGLESS strings: the server treats
    aspects as opaque cluster names (byte-for-byte match against the baked manifest).
    Anything that looks like animation ("alternate", cycles, phases) happens in the SENDER."""

    def _aspects(self, registry) -> set:
        """Returns the aspect set currently active in the registry for track ('dyn', 2)."""
        return set(registry.tracks[("dyn", 2)].switches[0].aspects)

    def test_applies_named_aspects_verbatim(self):
        """Requested aspect strings are applied byte-for-byte, with no server-side interpretation."""
        backend, registry = _make_backend()
        backend._update_aspect_states(_request("170", ["opaque:key_a", "another/opaque.key-b"]))
        self.assertEqual(self._aspects(registry), {"opaque:key_a", "another/opaque.key-b"})

    def test_gains_carried(self):
        """Per-aspect gain and cold RGB values reach the registry within float32 wire precision."""
        backend, registry = _make_backend()
        backend._update_aspect_states(_request("170", ["opaque:key_a"], gains=[("opaque:key_a", 0.5, [0.1, 0.2, 0.3])]))
        gain, cold = registry.tracks[("dyn", 2)].aspect_gains["opaque:key_a"]
        self.assertAlmostEqual(gain, 0.5, places=6)
        for got, want in zip(cold, [0.1, 0.2, 0.3]):
            self.assertAlmostEqual(got, want, places=6)  # float32 wire precision

    def test_track_id_source_suffix_cleaned(self):
        """An '@sensor' suffix on the track id is stripped before the edit-track index lookup."""
        backend, registry = _make_backend()
        backend._update_aspect_states(_request("170@camera_front", ["opaque:key_a"]))
        self.assertIn("opaque:key_a", self._aspects(registry))

    def test_stateless_absent_aspect_state_reverts_to_file_schedule(self):
        """The renderer keeps NO aspect state between requests: a request without
        aspect_state renders the file-scheduled/baked state, regardless of what any
        earlier request sent. An explicit empty list hides every cluster for that frame."""
        backend, registry = _make_backend()
        filed = TrackSchedule("dyn", 2, "asset_1", [AspectSwitch(["file:aspect"], 0)])
        backend.file_schedules = {("dyn", 2): filed}
        backend._update_aspect_states(_request("170", ["opaque:key_a"]))
        self.assertEqual(self._aspects(registry), {"opaque:key_a"})
        backend._update_aspect_states(_request("170"))  # no aspect_state -> file schedule
        self.assertIs(registry.tracks[("dyn", 2)], filed)
        backend._update_aspect_states(_request("170", []))  # explicit empty -> all hidden
        self.assertEqual(self._aspects(registry), set())

    def test_stateless_without_file_schedule_reverts_to_baked(self):
        """With no file schedule, an undriven request leaves the track unregistered
        (baked default) even right after a driven request."""
        backend, registry = _make_backend()
        backend._update_aspect_states(_request("170", ["opaque:key_a"]))
        self.assertEqual(self._aspects(registry), {"opaque:key_a"})
        backend._update_aspect_states(_request("170"))
        self.assertEqual(registry.tracks, {})

    def test_file_scheduled_tracks_on_other_keys_survive(self):
        """A live update supersedes only its own track for this request; file
        schedules on other keys keep applying."""
        backend, registry = _make_backend()
        other_layer = TrackSchedule("custom_rigids", 3, "prop_1", [AspectSwitch(["variant_a"], 0)])
        filed = TrackSchedule("dyn", 2, "asset_1", [AspectSwitch(["x"], 0)])
        backend.file_schedules = {("custom_rigids", 3): other_layer, ("dyn", 2): filed}
        backend._update_aspect_states(_request("170", ["opaque:key_a"]))
        self.assertIs(registry.tracks[("custom_rigids", 3)], other_layer)  # preserved
        self.assertEqual(self._aspects(registry), {"opaque:key_a"})  # superseded this frame

    def test_addressable_but_undriven_track_keeps_file_schedule(self):
        """An addressable track the request does not drive plays its file schedule;
        driving it in a later request is per-request only (it reverts after)."""
        backend, registry = _make_backend(track_index={"170": ("dyn", 2), "171": ("dyn", 5)})
        undriven = TrackSchedule("dyn", 5, "detail", [AspectSwitch(["detailstate:pulse"], 0)])
        backend.file_schedules = {("dyn", 5): undriven}
        backend._update_aspect_states(_request("170", ["opaque:key_a"]))
        self.assertIs(registry.tracks[("dyn", 5)], undriven)  # file schedule applies
        self.assertEqual(self._aspects(registry), {"opaque:key_a"})  # driven track live
        backend._update_aspect_states(_request("171", ["detailstate:off"]))
        self.assertEqual(set(registry.tracks[("dyn", 5)].switches[0].aspects), {"detailstate:off"})
        # Stateless: the next request not driving 171 reverts it to the file schedule.
        backend._update_aspect_states(_request("170", ["opaque:key_a"]))
        self.assertIs(registry.tracks[("dyn", 5)], undriven)

    def test_noop_when_request_states_nothing(self):
        """A request carrying no aspect_state applies nothing."""
        backend, registry = _make_backend()
        backend._update_aspect_states(_request("170"))
        self.assertEqual(registry.tracks, {})

    def test_aspect_state_on_a_scene_without_edits_is_rejected(self):
        """Driving aspects on a scene with no manifest fails the request rather than
        rendering unchanged frames the sender cannot distinguish from success — and the
        error names the RIGHT reason (no manifest vs. a manifest with no drivable tracks)."""
        backend, registry = _make_backend(track_index={}, track_aspects={})
        backend._update_aspect_states(_request("170"))  # no aspect_state: still fine
        self.assertEqual(registry.tracks, {})
        with self.assertRaisesRegex(ValueError, "no drivable tracks"):
            backend._update_aspect_states(_request("170", ["opaque:key_a"]))
        backend.aspect_registry = None
        with self.assertRaisesRegex(ValueError, "no asset-edit manifest"):
            backend._update_aspect_states(_request("170", ["opaque:key_a"]))

    def test_duplicate_track_and_gain_entries_are_rejected(self):
        """Duplicate aspect_state entries for one track, or duplicate per-aspect gains,
        fail the request rather than silently last-winning."""
        backend, _ = _make_backend()
        request = _request("170", ["opaque:key_a"])
        dup = request.dynamic_objects.add()
        dup.track_id = "170"
        dup.aspect_state.SetInParent()
        with self.assertRaisesRegex(ValueError, "more than once"):
            backend._update_aspect_states(request)
        backend2, _ = _make_backend()
        gains = [("opaque:key_a", 0.5, None), ("opaque:key_a", 0.7, None)]
        with self.assertRaisesRegex(ValueError, "more than once"):
            backend2._update_aspect_states(_request("170", ["opaque:key_a"], gains=gains))

    def test_unknown_track_is_a_request_error(self):
        """A track id absent from the edit-track index fails the request.

        Ignoring it would leave a sender that mistyped one character with hidden geometry, a
        success status and no diagnostic — gRPC clients never see server logs. The
        message names the tracks this scene CAN drive.
        """
        backend, registry = _make_backend()
        with self.assertRaisesRegex(ValueError, r"track '999'.*Edit-drivable tracks: \['170'\]"):
            backend._update_aspect_states(_request("999", ["opaque:key_a"]))
        self.assertEqual(registry.tracks, {})

    def test_unknown_aspect_is_a_request_error(self):
        """An aspect string no baked cluster carries fails the request too.

        Without this check the failure is completely silent: the runtime matches aspects
        byte-for-byte, so a case slip hides the actor's clusters with nothing logged
        anywhere.
        """
        backend, _ = _make_backend()
        with self.assertRaisesRegex(ValueError, r"\['Opaque:Key_A'\].*no baked cluster"):
            backend._update_aspect_states(_request("170", ["Opaque:Key_A"]))

    def test_unknown_aspect_in_a_gain_is_also_rejected(self):
        """Gains address aspects too, so an unknown aspect there is caught the same way."""
        backend, _ = _make_backend()
        with self.assertRaisesRegex(ValueError, "no baked cluster"):
            backend._update_aspect_states(_request("170", ["opaque:key_a"], gains=[("opaque:typo", 0.5, None)]))


class _InstallComposite:
    """Typed fake composite: hasattr(set_edit_overrides) must be real, not MagicMock."""

    def __init__(self, tracks_id: list[str], cuboid_ids: Optional[list[int]] = None) -> None:
        """Builds a 'dyn' gaussians node holding the given cuboid track ids and an empty install log."""
        self.gaussians_nodes = {
            "dyn": SimpleNamespace(
                cuboid_tracks=SimpleNamespace(tracks_id=tracks_id),
                gaussian_cuboid_ids=torch.tensor(cuboid_ids if cuboid_ids is not None else [0], dtype=torch.int32),
            )
        }
        self.installed: list[AspectOverrideRuntime] = []

    def set_edit_overrides(self, runtime: AspectOverrideRuntime) -> None:
        """Records each installed runtime for later assertions."""
        self.installed.append(runtime)


class _NoOverridesComposite:
    """A model type without the edit-overrides capability hook."""


def _cluster(layer_id: str = "dyn", nurec_track_id: int = 0, on_indices: Optional[list[int]] = None) -> BakedCluster:
    """Builds a minimal one-row 'on' BakedCluster for the given layer and track."""
    return BakedCluster(
        cluster_id=f"b_{layer_id}_{nurec_track_id}",
        edit_id="fx",
        layer_id=layer_id,
        nurec_track_id=nurec_track_id,
        aspect="on",
        on_indices=on_indices if on_indices is not None else [0],
        centroid_world=[0.0, 0.0, 0.0],
    )


def _write_usdz(path: Path, manifest: Optional[EditManifest]) -> None:
    """Writes a fake USDZ zip, embedding the edit manifest member only when one is given."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("checkpoint.ckpt", b"x")
        if manifest is not None:
            zf.writestr(EDIT_MANIFEST_FILENAME, json.dumps(manifest.to_dict()))


def _install_backend(composite: Any) -> Backend:
    """Builds a Backend whose renderable model wraps the given composite, with no registry installed."""
    renderable = mock.MagicMock()
    renderable._model = composite
    return Backend(
        renderable_model=renderable,
        camera_bank=mock.MagicMock(),
        world_to_nurec=mock.MagicMock(),
        lidar_bank=mock.MagicMock(),
        asset_bank=mock.MagicMock(),
        source_artifact=mock.MagicMock(),
        backend_cache_key=mock.MagicMock(),
    )


class InstallEditOverridesTest(unittest.TestCase):
    """Auto-detection at scene load: manifest in the USDZ, schedule data file by name."""

    def test_plain_artifact_without_manifest_is_untouched(self) -> None:
        """A USDZ with no embedded manifest installs nothing and leaves the registry unset."""
        composite = _InstallComposite(tracks_id=["100"])
        backend = _install_backend(composite)
        with tempfile.TemporaryDirectory() as tmp:
            usdz = Path(tmp) / "plain.usdz"
            _write_usdz(usdz, manifest=None)
            _install_edit_overrides(backend, mock.MagicMock(source=str(usdz)))
        self.assertIsNone(backend.aspect_registry)
        self.assertEqual(composite.installed, [])

    def test_missing_source_is_a_noop(self) -> None:
        """An empty artifact source path skips override installation entirely."""
        backend = _install_backend(_InstallComposite(tracks_id=["100"]))
        _install_edit_overrides(backend, mock.MagicMock(source=""))
        self.assertIsNone(backend.aspect_registry)

    def test_model_without_capability_fails_loud(self) -> None:
        """A baked artifact whose model lacks set_edit_overrides refuses to install: with
        replace_original the captured actor is hidden in the bake, so serving without the
        edits would silently render a hole where it was."""
        backend = _install_backend(_NoOverridesComposite())
        with tempfile.TemporaryDirectory() as tmp:
            usdz = Path(tmp) / "baked.usdz"
            _write_usdz(usdz, EditManifest(clusters=[_cluster()]))
            with self.assertRaisesRegex(ValueError, "does not support edit overrides"):
                _install_edit_overrides(backend, mock.MagicMock(source=str(usdz)))
        self.assertIsNone(backend.aspect_registry)

    def test_runtime_installed_with_shared_registry(self) -> None:
        """Install wires one runtime sharing the backend's registry and builds the edit-track index."""
        composite = _InstallComposite(tracks_id=["100"])
        backend = _install_backend(composite)
        with tempfile.TemporaryDirectory() as tmp:
            usdz = Path(tmp) / "baked.usdz"
            _write_usdz(usdz, EditManifest(clusters=[_cluster()]))
            _install_edit_overrides(backend, mock.MagicMock(source=str(usdz)))
        (runtime,) = composite.installed
        self.assertEqual(len(runtime.manifest.clusters), 1)
        self.assertIsNotNone(backend.aspect_registry)
        self.assertIs(runtime.registry, backend.aspect_registry)  # live set_many reaches the runtime
        self.assertEqual(backend.edit_track_index, {"100": ("dyn", 0)})

    def test_schedule_file_picked_up_by_naming_convention(self) -> None:
        """A sibling "<stem>.schedule.json" preloads the registry — the FULL stem, so a
        dotted artifact name like scene.artifact.usdz pairs with scene.artifact.schedule.json."""
        composite = _InstallComposite(tracks_id=["100"])
        backend = _install_backend(composite)
        with tempfile.TemporaryDirectory() as tmp:
            usdz = Path(tmp) / "scene.artifact.usdz"
            _write_usdz(usdz, EditManifest(clusters=[_cluster()]))
            schedule = EditSchedule(
                time_base="absolute_us",
                tracks=[TrackSchedule("dyn", 0, "fx", [AspectSwitch(["on"], 0)])],
            )
            (Path(tmp) / "scene.artifact.schedule.json").write_text(json.dumps(schedule.to_dict()))
            _install_edit_overrides(backend, mock.MagicMock(source=str(usdz)))
        assert backend.aspect_registry is not None
        self.assertEqual(set(backend.aspect_registry.tracks), {("dyn", 0)})

    def test_without_schedule_file_registry_awaits_live_schedule(self) -> None:
        """Without a schedule file the runtime still installs, leaving the registry empty for live updates."""
        composite = _InstallComposite(tracks_id=["100"])
        backend = _install_backend(composite)
        with tempfile.TemporaryDirectory() as tmp:
            usdz = Path(tmp) / "baked.usdz"
            _write_usdz(usdz, EditManifest(clusters=[_cluster()]))
            _install_edit_overrides(backend, mock.MagicMock(source=str(usdz)))
        assert backend.aspect_registry is not None
        self.assertEqual(backend.aspect_registry.tracks, {})
        self.assertEqual(len(composite.installed), 1)  # the runtime is still installed

    def test_unresolvable_manifest_key_warned_and_skipped(self) -> None:
        """Manifest tracks with no matching cuboid track id warn and are omitted from the edit-track index."""
        # Only track 0 exists in the track LIST, but the checkpoint carries rows tagged
        # to track 7 (validate_against_layers passes; the index resolution cannot).
        composite = _InstallComposite(tracks_id=["100"], cuboid_ids=[0, 7])
        backend = _install_backend(composite)
        manifest = EditManifest(clusters=[_cluster(nurec_track_id=0), _cluster(nurec_track_id=7, on_indices=[1])])
        with tempfile.TemporaryDirectory() as tmp:
            usdz = Path(tmp) / "baked.usdz"
            _write_usdz(usdz, manifest)
            with self.assertLogs("nurec.grpc.serve", level="WARNING") as cm:
                _install_edit_overrides(backend, mock.MagicMock(source=str(usdz)))
        self.assertEqual(backend.edit_track_index, {"100": ("dyn", 0)})
        self.assertTrue(any("not resolvable" in line for line in cm.output))


class EditAssetsGuardTest(unittest.TestCase):
    """edit_assets must refuse to mutate an edit-baked scene: asset replacement reorders
    layer rows, silently invalidating the manifest's baked row indices."""

    def test_rejected_on_edit_baked_scene(self) -> None:
        backend, _ = _make_backend()
        with self.assertRaisesRegex(ValueError, "edit-baked"):
            backend.edit_assets(replace_assets=[mock.MagicMock()], insert_assets=[])

    def test_empty_request_is_not_rejected(self) -> None:
        backend, _ = _make_backend()
        backend.edit_assets(replace_assets=[], insert_assets=[])  # no raise


class LiveGainValidationTest(unittest.TestCase):
    """Wire-supplied AspectGain values are validated at ingestion (stateless: per request)."""

    def test_negative_gain_rejected(self) -> None:
        backend, _ = _make_backend()
        request = _request("170", ["opaque:key_a"], gains=[("opaque:key_a", -3.0, None)])
        with self.assertRaisesRegex(ValueError, "gain"):
            backend._update_aspect_states(request)

    def test_nan_gain_rejected(self) -> None:
        backend, _ = _make_backend()
        request = _request("170", ["opaque:key_a"], gains=[("opaque:key_a", float("nan"), None)])
        with self.assertRaisesRegex(ValueError, "gain"):
            backend._update_aspect_states(request)


class RenderRpcAspectStateTest(unittest.TestCase):
    """The REAL RPC boundary: a render_rgb call through SensorSimService.

    Everything else in this file stops one or more levels below the servicer, so
    deleting the ``_update_aspect_states`` call sites in serve.py would keep it green.
    These two tests fail if the call site is dropped or reordered, and they are the only
    place the aspect contract is checked as a CLIENT sees it — a status code, not a log.
    """

    def setUp(self) -> None:
        """Start a real gRPC service with one backend whose scene has baked edits."""
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        self._service = SensorSimService(
            self._server,
            artifacts_glob=None,
            ray_chunk_size=1,
            egocar_hoods_dir=None,
            downloader=None,
            scene_cache=None,
            cache_size=1,
            max_backends_per_scene=1,
            metrics_output_dir=None,
            enable_editing_actors=True,
            server_config=GrpcServerConfig(),
        )
        add_SensorsimServiceServicer_to_server(self._service, self._server)
        port = self._server.add_insecure_port("localhost:0")
        self._server.start()
        self._channel = grpc.insecure_channel(f"localhost:{port}")
        grpc.channel_ready_future(self._channel).result(timeout=5)
        self.stub = SensorsimServiceStub(self._channel)

        artifact = Artifact(Path("/scenes/scene.usdz"))
        artifact._metadata = {"scene_id": "scene"}
        artifact._sha256_footprint = hashlib.sha256(b"scene").hexdigest()
        scene_key = SceneCacheKey(scene_id="scene")
        registration = ArtifactRegistration(artifact, False, scene_key)

        self.backend, self.registry = _make_backend()
        self.backend.backend_cache_key = registration.backend_cache_key
        self.rendered_aspects: list = []

        def _render_camera_frame(*_args: Any, **_kwargs: Any) -> Any:
            """Stands in for the GPU render, recording what the registry held AT render time.

            The real Backend.render_rgb_request runs — this only replaces the render
            itself — so the assertion covers the actual _update_aspect_states call site.
            """
            entry = self.registry.tracks.get(("dyn", 2))
            self.rendered_aspects.append(None if entry is None else set(entry.switches[0].aspects))
            return SimpleNamespace(color_image=torch.zeros(1, 1, 3))

        model = cast(Any, self.backend.renderable_model)
        model.render_camera_frame_from_ray_bundle.side_effect = _render_camera_frame
        # aspect_state rides the scene-editing channel, so the model must support actor
        # edits — the servicer rejects DynamicObject updates outright otherwise.
        model.supports_edit_actors.return_value = True
        ray_bundle = mock.MagicMock()
        ray_bundle.rendering_data.w = 1
        ray_bundle.rendering_data.h = 1
        cast(Any, self.backend.camera_bank)._get_camera.return_value.build_ray_bundle.return_value = ray_bundle
        cast(Any, self._service).encode_image = mock.MagicMock(return_value=RGBRenderReturn(image_bytes=b"rgb"))

        with self._service._artifact_registry_lock:
            self._service._artifact_registrations[scene_key.key_hash] = registration
        self._service.backend_cache.put(registration.backend_cache_key.key_hash, self.backend)
        self._service.backend_cache.checkin(self.backend)

    def tearDown(self) -> None:
        """Close the channel and stop the server."""
        self._channel.close()
        self._server.stop(0)

    @staticmethod
    def _rgb_request(track_id: str, aspects: list[str]) -> RGBRenderRequest:
        """A one-actor render_rgb request driving ``track_id``'s aspects."""
        request = RGBRenderRequest(scene_id="scene", frame_start_us=1_000_000, frame_end_us=1_000_002)
        dyn = request.dynamic_objects.add()
        dyn.track_id = track_id
        dyn.aspect_state.aspects.extend(aspects)
        dyn.pose_pair.start_pose.quat.w = 1.0
        dyn.pose_pair.end_pose.quat.w = 1.0
        return request

    def test_render_rgb_applies_aspect_state_before_rendering(self) -> None:
        """A successful render_rgb has the request's aspects live in the registry AT render time."""
        self.stub.render_rgb(self._rgb_request("170", ["opaque:key_a"]))
        self.assertEqual(self.rendered_aspects, [{"opaque:key_a"}])

    def test_unknown_aspect_fails_the_rpc_with_invalid_argument(self) -> None:
        """An aspect this scene has no baked cluster for is reported to the CALLER.

        This is the contract that matters to a sender: a typo is a status code naming
        the baked aspects, not a silently hidden actor.
        """
        with self.assertRaises(grpc.RpcError) as ctx:
            self.stub.render_rgb(self._rgb_request("170", ["opaque:TYPO"]))
        self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("no baked cluster", ctx.exception.details())
        self.assertEqual(self.rendered_aspects, [])  # nothing rendered


class RenderRequestAspectPathTest(unittest.TestCase):
    """The wire-visible contract one level under the servicer: render_rgb_request applies
    the request's AspectState BEFORE rendering and passes the actors snapshot through."""

    def test_aspect_state_applied_before_render(self) -> None:
        backend, registry = _make_backend()
        seen: dict[str, Any] = {}

        def _render(ray_bundle: Any, actors_snapshot: Any = None, frame_start_us: int = 0, frame_end_us: int = 0):
            seen["registry_tracks"] = dict(registry.tracks)
            seen["actor_ids"] = None if actors_snapshot is None else list(actors_snapshot.actor_ids)
            return SimpleNamespace(color_image=torch.zeros(2, 2, 3))

        # Backend declares these fields with real types, so configure the mocks through
        # Any — otherwise mypy resolves `.side_effect` against RenderableModel.
        model = cast(Any, backend.renderable_model)
        model.render_camera_frame_from_ray_bundle.side_effect = _render
        model.supports_edit_actors.return_value = True
        ray_bundle = mock.MagicMock()
        ray_bundle.rendering_data.w = 2
        ray_bundle.rendering_data.h = 2
        cast(Any, backend.camera_bank)._get_camera.return_value.build_ray_bundle.return_value = ray_bundle

        request = _request("170", ["opaque:key_a"])
        dyn = request.dynamic_objects[0]
        dyn.pose_pair.start_pose.quat.w = 1.0
        dyn.pose_pair.end_pose.quat.w = 1.0
        _, (h, w) = backend.render_rgb_request(request, enable_editing_actors=True)
        self.assertEqual((h, w), (2, 2))
        # The live aspect state was active in the registry AT render time...
        self.assertIn(("dyn", 2), seen["registry_tracks"])
        self.assertEqual(seen["registry_tracks"][("dyn", 2)].active_state_at(0), {"opaque:key_a"})
        # ...and the same DynamicObject also flowed into the actors snapshot.
        self.assertEqual(seen["actor_ids"], ["170"])


class AspectCallSiteTest(unittest.TestCase):
    """render_camera_request and render_lidar_request apply the request's AspectState.

    A pure call-site mutation guard: the rgb path's full behavior is pinned by
    RenderRequestAspectPathTest; here we only require that the other two entry points
    consult _update_aspect_states, even when the downstream render mocks cannot finish.
    Deleting either call site fails these.
    """

    def _drive(self, method_name: str, request: Any) -> list:
        backend, _ = _make_backend()
        seen: list = []
        original = backend._update_aspect_states

        def spy(req: Any) -> None:
            seen.append(req)
            original(req)

        backend._update_aspect_states = spy  # type: ignore[assignment, method-assign]
        try:
            getattr(backend, method_name)(request, enable_editing_actors=True)
        except Exception:
            pass  # downstream MagicMocks cannot complete a render; the call site is the test
        return seen

    def _driven(self, request: Any) -> Any:
        dyn = request.dynamic_objects.add()
        dyn.track_id = "170"
        dyn.aspect_state.aspects.append("opaque:key_a")
        return request

    def test_camera_path_applies_aspect_state(self) -> None:
        request = self._driven(CameraRenderRequest(frame_start_us=1_000_000, frame_end_us=1_000_002))
        self.assertEqual(len(self._drive("render_camera_request", request)), 1)

    def test_lidar_path_applies_aspect_state(self) -> None:
        request = self._driven(LidarRenderRequest(frame_start_us=1_000_000, frame_end_us=1_000_002))
        self.assertEqual(len(self._drive("render_lidar_request", request)), 1)


class ActorsSnapshotPoseValidationTest(unittest.TestCase):
    """Pose-less DynamicObjects are rejected: converting an unset pose_pair would re-pose
    the actor to the origin with a degenerate (all-zero) quaternion."""

    def test_poseless_dynamic_object_rejected(self) -> None:
        """A DynamicObject carrying only aspect_state (no pose_pair) fails loud."""
        request = RGBRenderRequest()
        dyn = request.dynamic_objects.add()
        dyn.track_id = "170"
        dyn.aspect_state.aspects.append("panel:on")
        with self.assertRaisesRegex(ValueError, "pose_pair"):
            actors_snapshot_from_render_request(request)

    def test_posed_dynamic_object_accepted(self) -> None:
        """A DynamicObject with pose_pair set converts as before."""
        request = RGBRenderRequest()
        dyn = request.dynamic_objects.add()
        dyn.track_id = "170"
        dyn.pose_pair.start_pose.quat.w = 1.0
        dyn.pose_pair.end_pose.quat.w = 1.0
        snapshot = actors_snapshot_from_render_request(request)
        self.assertEqual(snapshot.actor_ids, ["170"])


class AspectGainPresenceTest(unittest.TestCase):
    """An AspectGain sent only to carry cold_rgb must not hide the cluster.

    `gain` has explicit presence, so an unset gain is distinguishable from an explicit
    0.0 and means "no attenuation".
    """

    def test_unset_gain_is_treated_as_full_intensity(self) -> None:
        backend, registry = _make_backend()
        request = _request("170", ["opaque:key_a"])
        g = request.dynamic_objects[0].aspect_state.gains.add()
        g.aspect = "opaque:key_a"
        g.cold_rgb.extend([0.02, 0.02, 0.02])  # gain deliberately not set
        self.assertFalse(g.HasField("gain"))

        backend._update_aspect_states(request)
        gain, cold = registry.tracks[("dyn", 2)].aspect_gains["opaque:key_a"]
        self.assertEqual(gain, 1.0)
        self.assertIsNotNone(cold)

    def test_explicit_zero_gain_is_still_honoured(self) -> None:
        """An explicit 0.0 is a legitimate "fully hidden this frame"."""
        backend, registry = _make_backend()
        request = _request("170", ["opaque:key_a"], gains=[("opaque:key_a", 0.0, None)])
        self.assertTrue(request.dynamic_objects[0].aspect_state.gains[0].HasField("gain"))
        backend._update_aspect_states(request)
        gain, _ = registry.tracks[("dyn", 2)].aspect_gains["opaque:key_a"]
        self.assertEqual(gain, 0.0)


class CapabilityKeyTest(unittest.TestCase):
    """`asset_edits.aspect_state` advertises the feature only where it can be used.

    aspect_state rides DynamicObject, which a server without --enable-editing-actors
    rejects outright; and the key must live in the shared builder so the multiprocess
    proxy, whose workers do support the field, reports it too.
    """

    _KEY = "asset_edits.aspect_state"

    def test_absent_when_actor_editing_is_disabled(self) -> None:
        config = GrpcServerConfig(enable_editing_actors=False)
        self.assertNotIn(self._KEY, SensorSimService.build_server_config_map(config))

    def test_present_when_actor_editing_is_enabled(self) -> None:
        config = GrpcServerConfig(enable_editing_actors=True)
        self.assertEqual(SensorSimService.build_server_config_map(config)[self._KEY], "stateless-v1")
