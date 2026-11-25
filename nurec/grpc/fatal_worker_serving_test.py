# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise new fault paths against the shared serving interceptor."""

from unittest import mock

import grpc
import pytest

from grpc_health.v1 import health, health_pb2  # type: ignore[import-not-found,import-untyped]

from nurec.grpc.cache_keys import BackendCacheKey, SceneCacheKey
from nurec.grpc.fatal_worker import FatalWorkerExceptionInterceptor
from nurec.grpc.multiproc_serve import OwnerSceneRegistry
from nurec.grpc.protos.sensorsim_pb2 import AvailableCamerasRequest, EditAssetsRequest
from nurec.grpc.serve import (
    SENSORSIM_HEALTH_SERVICE_NAME,
    SensorSimService,
    _serve_once_harmonizer_loaded,
)


class _Aborted(Exception):
    pass


@pytest.fixture
def guarded_service():
    servicer = health.HealthServicer()
    interceptor = FatalWorkerExceptionInterceptor(terminate_worker=mock.Mock())
    service = SensorSimService(None, artifacts_glob=None, ray_chunk_size=1, fatal_worker_interceptor=interceptor)

    def unhealthy():
        for name in ("", SENSORSIM_HEALTH_SERVICE_NAME):
            servicer.set(name, health_pb2.HealthCheckResponse.NOT_SERVING)
        service.backend_cache.wake_waiters()

    interceptor.configure_shutdown(mark_unhealthy=unhealthy, stop_server=lambda grace: None)
    with mock.patch("nurec.grpc.fatal_worker.threading.Timer"):
        yield service, interceptor, servicer
        assert interceptor.wait_for_shutdown(2)
    service._metrics_manager.shutdown()


def _invoke(interceptor, operation, request):
    context = mock.Mock()
    context.abort.side_effect = _Aborted
    details = mock.Mock()
    details.method = f"/{SENSORSIM_HEALTH_SERVICE_NAME}/test"
    details.invocation_metadata = (("x-request-id", "regression-request"),)
    context.invocation_metadata.return_value = details.invocation_metadata
    handler = grpc.unary_unary_rpc_method_handler(operation)
    guarded = interceptor.intercept_service(lambda _: handler, details)
    with pytest.raises(_Aborted):
        guarded.unary_unary(request, context)
    context.abort.assert_called_once()
    assert context.abort.call_args.args[0] == grpc.StatusCode.UNAVAILABLE
    assert interceptor.state.is_fatal
    incident = interceptor.state.incident
    assert incident is not None
    assert incident.request_id == "regression-request"


@pytest.mark.parametrize(
    "rpc,rpc_request",
    [
        ("get_available_cameras", AvailableCamerasRequest(scene_id="s")),
        ("edit_assets", EditAssetsRequest(scene_id="s")),
    ],
)
def test_discovery_and_edit_faults_reach_interceptor(guarded_service, rpc, rpc_request):
    service, interceptor, _ = guarded_service
    with mock.patch.object(
        service, "get_backend", side_effect=RuntimeError("CUDA error: uncorrectable ECC error encountered")
    ):
        _invoke(interceptor, getattr(service, rpc), rpc_request)


@pytest.mark.parametrize("helper", ["_batch_camera_error_item", "_batch_rgb_error_item"])
def test_per_item_errors_preserve_fatal_cause_and_rpc_diagnostics(guarded_service, helper):
    service, interceptor, _ = guarded_service
    wrapped = ValueError("image encoding failed")
    wrapped.__cause__ = RuntimeError("CUDA error: unknown error")
    _invoke(interceptor, lambda request, context: getattr(service, helper)("camera", wrapped), None)


@pytest.mark.parametrize("fatal", [False, True])
def test_backend_checkin_only_for_recoverable_faults(guarded_service, fatal):
    service, interceptor, _ = guarded_service
    key = SceneCacheKey(scene_id="s")
    backend = mock.Mock()
    backend.backend_cache_key = BackendCacheKey(key, "0" * 64)
    backend.source_artifact.sha256_footprint = "0" * 64
    service.register_resident_backend(backend)
    error = RuntimeError("CUDA error: unknown error") if fatal else ValueError("bad camera")
    with (
        mock.patch.object(service.backend_cache, "put_with_retries", return_value=backend),
        mock.patch.object(service.backend_cache, "checkin") as checkin,
    ):
        with pytest.raises(type(error)):
            with service.get_backend(key):
                raise error
        assert checkin.call_count == (0 if fatal else 1)


def test_harmonizer_fault_cannot_publish_serving(guarded_service):
    service, interceptor, servicer = guarded_service
    with mock.patch(
        "nurec.grpc.serve.HarmonizerModelFactory.get", side_effect=RuntimeError("CUDA error: unknown error")
    ):
        service.set_harmonizer(True, "url", "cache", "harmonizer.pt", (1, 1))
        service.wait_for_harmonizer_load()
    _serve_once_harmonizer_loaded(service, servicer)
    assert interceptor.state.is_fatal
    for name in ("", SENSORSIM_HEALTH_SERVICE_NAME):
        assert (
            servicer.Check(health_pb2.HealthCheckRequest(service=name), None).status
            == health_pb2.HealthCheckResponse.NOT_SERVING
        )


def test_successful_harmonizer_load_cannot_clear_prior_fault(guarded_service):
    service, interceptor, servicer = guarded_service
    interceptor.handle_fatal_exception(RuntimeError("CUDA error: unknown error"), operation="render")
    service._harmonizer_ready.set()
    _serve_once_harmonizer_loaded(service, servicer)
    assert servicer.Check(health_pb2.HealthCheckRequest(), None).status == health_pb2.HealthCheckResponse.NOT_SERVING


def test_owner_load_uses_shared_interceptor(guarded_service):
    _, interceptor, servicer = guarded_service
    registry = OwnerSceneRegistry(fatal_worker_interceptor=interceptor)

    def fail():
        raise RuntimeError("CUDA error: unknown error")

    with pytest.raises(RuntimeError, match="unknown error"):
        registry.ensure_loaded("s", fail)
    assert interceptor.state.is_fatal
    incident = interceptor.state.incident
    assert incident is not None
    assert incident.scene_ids == ("s",)
    assert not registry.is_loaded("s")
    assert servicer.Check(health_pb2.HealthCheckRequest(), None).status == health_pb2.HealthCheckResponse.NOT_SERVING


def test_owner_load_preserves_rpc_request_diagnostics(guarded_service):
    _, interceptor, _ = guarded_service
    registry = OwnerSceneRegistry(fatal_worker_interceptor=interceptor)

    def fail():
        raise RuntimeError("CUDA error: unknown error")

    _invoke(interceptor, lambda request, context: registry.ensure_loaded("s", fail), None)
    incident = interceptor.state.incident
    assert incident is not None
    assert incident.rpc_method is not None and incident.rpc_method.endswith("/test")
    assert incident.scene_ids == ("s",)
