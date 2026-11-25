# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OOM recovery preserves coordinator-owned residency and single-process reload.

Model loading, local-file validation, and CUDA allocation are stand-ins. Cache,
registration, attachment, coordinator, and acquisition use production implementations.
"""

import contextlib
import json
import queue
import threading

from unittest import mock

import pytest
import torch

from nurec.grpc.cache_keys import BackendCacheKey, SceneCacheKey
from nurec.grpc.grpc_server_config import GrpcServerConfig
from nurec.grpc.multiproc_serve import (
    OwnerSceneRegistry,
    SceneAttachChannel,
    SceneCoordinator,
    _scene_attach_listener,
    _WorkerSceneTable,
)
from nurec.grpc.serve import ArtifactRegistration, Backend, SensorSimService


SCENES = ("active-scene", "idle-scene")


def _backend(scene_id: str) -> Backend:
    artifact = mock.MagicMock()
    artifact.sha256_footprint = "a" * 64
    artifact.source = f"/unused/{scene_id}.usdz"
    return Backend(
        renderable_model=None,  # type: ignore[arg-type]
        camera_bank=None,  # type: ignore[arg-type]
        world_to_nurec=None,  # type: ignore[arg-type]
        scene_time_range_us=None,  # type: ignore[arg-type]
        lidar_bank=None,  # type: ignore[arg-type]
        asset_bank=None,  # type: ignore[arg-type]
        source_artifact=artifact,
        backend_cache_key=BackendCacheKey(SceneCacheKey(scene_id), "a" * 64),
    )


def _service(*, allow_automatic_eviction: bool = True) -> SensorSimService:
    return SensorSimService(
        server=None,
        artifacts_glob=None,
        ray_chunk_size=1,
        cache_size=3,
        server_config=GrpcServerConfig(),
        allow_automatic_backend_eviction=allow_automatic_eviction,
    )


def _cached_scene_ids(service: SensorSimService) -> set[str]:
    cache = service.backend_cache
    with cache.lock:
        return {backend.scene_id for backend in [*cache.spares, *cache.in_use]}


@pytest.fixture(autouse=True)
def _no_cuda_allocations(monkeypatch):
    # Exercise production OOM handling without allocating or exhausting physical GPU memory.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@contextlib.contextmanager
def _pool(worker_count: int):
    owner = OwnerSceneRegistry()
    services = [_service(allow_automatic_eviction=False) for _ in range(worker_count)]
    tables = [_WorkerSceneTable(service) for service in services]
    channels = []
    threads = []
    builds = []

    def build(scene_id, artifact_path, ipc_tensors, edit_journal=()):
        builds.append(scene_id)
        return _backend(scene_id)

    try:
        for table in tables:
            commands: queue.Queue[tuple[object, ...]] = queue.Queue()
            acknowledgments: queue.Queue[tuple[int, str, object]] = queue.Queue()
            thread = threading.Thread(
                target=_scene_attach_listener,
                args=(table, build, commands, acknowledgments),
                daemon=True,
            )
            thread.start()
            threads.append(thread)
            channels.append(SceneAttachChannel(commands, acknowledgments, owner.registry, is_worker_alive=lambda: True))

        coordinator = SceneCoordinator(
            owner,
            channels,
            {scene_id: f"/unused/{scene_id}.usdz" for scene_id in SCENES},
            loader=lambda scene_id, artifact_path: torch.nn.Linear(2, 2, device="cpu"),
            max_scenes=2,
        )
        for scene_id in SCENES:
            coordinator.ensure_attached(scene_id)
        yield owner, coordinator, channels, tables, services, builds
    finally:
        for channel in channels:
            channel.stop()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive(), "scene-attach listener did not stop"


def _inject_oom(service: SensorSimService, phase: str) -> None:
    if phase == "during-render":
        with pytest.raises(torch.cuda.OutOfMemoryError, match="injected render OOM"):
            with service.get_backend(SceneCacheKey(SCENES[0])):
                raise torch.cuda.OutOfMemoryError("injected render OOM")
    else:
        # Batched JPEG encoding invokes this helper after get_backend has checked in.
        with service.get_backend(SceneCacheKey(SCENES[0])):
            pass
        service._recover_from_cuda_oom("injected post-checkin encoding OOM")


@pytest.mark.parametrize("worker_count", [1, 2])
def test_oom_residency_without_oom_keeps_attached_scenes_usable(worker_count):
    with _pool(worker_count) as (owner, coordinator, channels, tables, services, builds):
        for service in services:
            for scene_id in SCENES:
                coordinator.acquire(scene_id)
                try:
                    with service.get_backend(SceneCacheKey(scene_id)) as backend:
                        assert backend.scene_id == scene_id
                finally:
                    coordinator.release_scene(scene_id)
            assert _cached_scene_ids(service) == set(SCENES)
        assert len(builds) == len(SCENES) * worker_count
        assert all(owner.registry.attached_count(scene_id) == worker_count for scene_id in SCENES)


@pytest.mark.parametrize("worker_count", [1, 2])
@pytest.mark.parametrize("phase", ["during-render", "after-checkin"])
def test_oom_residency_preserves_pool_attachment(worker_count, phase):
    with _pool(worker_count) as (owner, coordinator, channels, tables, services, builds):
        worker = services[0]
        victim = SCENES[1]
        original_backend = tables[0][victim]
        coordinator.acquire(SCENES[0])
        try:
            _inject_oom(worker, phase)
        finally:
            coordinator.release_scene(SCENES[0])

        assert _cached_scene_ids(worker) == set(SCENES)
        assert coordinator.attached_scenes() == set(SCENES)
        assert not coordinator.is_degraded()
        assert all(channel.is_attached(victim) for channel in channels)
        assert owner.registry.attached_count(victim) == worker_count
        # All views retain the original owner-aliased backend after OOM.
        assert tables[0][victim] is original_backend

        # Follow-up requests reuse the original backend; no private reconstruction.
        with mock.patch("nurec.grpc.serve.Backend.from_artifact") as rebuild:
            for _ in range(3):
                coordinator.acquire(victim)
                try:
                    with worker.get_backend(SceneCacheKey(victim)) as backend:
                        assert backend is original_backend
                finally:
                    coordinator.release_scene(victim)
            rebuild.assert_not_called()
        assert len(builds) == len(SCENES) * worker_count
        for healthy_worker in services[1:]:
            with healthy_worker.get_backend(SceneCacheKey(victim)) as backend:
                assert backend.scene_id == victim

        print(
            "OOM_RESIDENCY_EVIDENCE "
            + json.dumps(
                {
                    "mode": "multiprocess",
                    "worker_count": worker_count,
                    "phase": phase,
                    "cached_after_oom": sorted(_cached_scene_ids(worker)),
                    "coordinator_attached": sorted(coordinator.attached_scenes()),
                    "owner_attachment_count": owner.registry.attached_count(victim),
                    "table_still_retains_backend": tables[0][victim] is original_backend,
                    "followup_acquisition_count": 3,
                    "reattach_build_count": 0,
                    "healthy_worker_count": worker_count - 1,
                },
                sort_keys=True,
            )
        )


@pytest.mark.parametrize("worker_count", [1, 2])
def test_oom_residency_still_allows_coordinated_eviction(worker_count):
    with _pool(worker_count) as (owner, coordinator, channels, tables, services, builds):
        coordinator.acquire(SCENES[0])
        try:
            _inject_oom(services[0], "during-render")
            coordinator.register_scene("new-scene", "/unused/new-scene.usdz")
            coordinator.ensure_attached("new-scene")

            # A remains leased; the coordinator detaches idle B from every worker.
            assert coordinator.attached_scenes() == {SCENES[0], "new-scene"}
            assert not owner.is_loaded(SCENES[1])
            for channel, table, service in zip(channels, tables, services):
                assert not channel.is_attached(SCENES[1])
                assert SCENES[1] not in table
                assert _cached_scene_ids(service) == {SCENES[0], "new-scene"}
                assert SceneCacheKey(SCENES[1]).key_hash not in service._artifact_registrations
        finally:
            coordinator.release_scene(SCENES[0])

        # The owner can later load/attach B again through the normal barrier.
        coordinator.acquire(SCENES[1])
        try:
            assert owner.registry.attached_count(SCENES[1]) == worker_count
            for service in services:
                with service.get_backend(SceneCacheKey(SCENES[1])) as backend:
                    assert backend.scene_id == SCENES[1]
        finally:
            coordinator.release_scene(SCENES[1])


@pytest.mark.parametrize("phase", ["during-render", "after-checkin"])
def test_oom_residency_single_process_reloadable_scene_recovers(phase):
    service = _service()
    for scene_id in SCENES:
        backend = _backend(scene_id)
        key = backend.backend_cache_key
        service._artifact_registrations[key.scene_key.key_hash] = ArtifactRegistration(
            artifact=backend.source_artifact,
            downloaded=False,
            scene_key=key.scene_key,
        )
        service.backend_cache.put(key.key_hash, backend)
        service.backend_cache.checkin(backend)

    _inject_oom(service, phase)
    assert SCENES[1] not in _cached_scene_ids(service)
    rebuilt = _backend(SCENES[1])
    with (
        mock.patch("nurec.grpc.serve.Artifact") as artifact_cls,
        mock.patch("nurec.grpc.serve._validate_local_artifact"),
        mock.patch("nurec.grpc.serve.Backend.from_artifact", return_value=rebuilt) as rebuild,
    ):
        artifact_cls.return_value.temporary_cache.return_value.__enter__.return_value = rebuilt.source_artifact
        with service.get_backend(SceneCacheKey(SCENES[1])) as backend:
            assert backend is rebuilt
        rebuild.assert_called_once()
    print("OOM_RESIDENCY_EVIDENCE " + json.dumps({"mode": "single_process", "phase": phase, "reload_succeeded": True}))
