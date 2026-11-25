# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests for fatal CUDA detection and shutdown deadlines."""

import importlib.util
import os
import subprocess
import sys
import threading

from concurrent import futures
from pathlib import Path
from unittest import mock

import grpc
import pytest

from grpc_health.v1 import health, health_pb2, health_pb2_grpc  # type: ignore[import-not-found,import-untyped]

from nurec.grpc.fatal_worker import (
    FatalWorkerExceptionInterceptor,
    FatalWorkerIncident,
    FatalWorkerState,
    is_fatal_worker_exception,
)


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: uncorrectable ECC error encountered",
        "CUDA error: unknown error",
        "CUDA error: device-side assert triggered",
        "CUDA error: unspecified launch failure",
        "CUDA runtime error (700): an illegal memory access was encountered",
        "kernel failed: cudaErrorIllegalAddress",
        "driver returned CUDA_ERROR_ECC_UNCORRECTABLE",
        "cudaStreamSynchronize(sizes) failed: an illegal memory access was encountered",
        "cudaMalloc failed: uncorrectable ECC error encountered",
        "render_pack kernel launch failed: an illegal memory access was encountered",
        "NvjpegBatchEncoder is unusable after an asynchronous CUDA failure.",
    ],
)
def test_additional_native_fatal_diagnostics(message):
    assert is_fatal_worker_exception(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: out of memory",
        "cudaMalloc failed: out of memory",
        "CUDA error: invalid argument",
        "CUDA error: invalid device ordinal",
        "unknown error",
        "checkpoint is missing",
    ],
)
def test_recoverable_runtime_diagnostics(message):
    assert not is_fatal_worker_exception(RuntimeError(message))


@pytest.mark.parametrize(
    "diagnostic",
    [
        "CUDA error: an illegal memory access was encountered",
        "CUDA error: unknown error",
        "CUDA_ERROR_ILLEGAL_ADDRESS",
        "cudaErrorIllegalAddress",
        "NvjpegBatchEncoder is unusable after an asynchronous CUDA failure.",
    ],
)
def test_client_text_cannot_trigger_exit(diagnostic):
    error = RuntimeError(f"Failed to replace original_id='{diagnostic}' with replacement_id='valid-asset'")
    terminate = mock.Mock()
    interceptor = FatalWorkerExceptionInterceptor(terminate_worker=terminate)
    with mock.patch("nurec.grpc.fatal_worker.threading.Timer") as timer:
        assert not interceptor.handle_fatal_exception(error, operation="edit_assets")
    assert not interceptor.state.is_fatal
    timer.assert_not_called()
    terminate.assert_not_called()


def test_both_exception_chains_and_suppressed_context_are_inspected_without_cycles():
    fatal = RuntimeError("CUDA error: unknown error")
    wrapper = ValueError("failed to load scene")
    wrapper.__cause__ = ValueError("cleanup failed")
    wrapper.__context__ = fatal
    wrapper.__suppress_context__ = True
    fatal.__cause__ = wrapper
    assert is_fatal_worker_exception(wrapper)
    wrapper.__context__ = wrapper
    assert not is_fatal_worker_exception(wrapper)


def test_remote_cuda_fault_does_not_poison_proxy():
    error = grpc.RpcError("CUDA error: an illegal memory access was encountered")
    error.__context__ = RuntimeError("CUDA error: unknown error")
    assert not is_fatal_worker_exception(error)
    wrapper = RuntimeError("forwarding failed")
    wrapper.__cause__ = error
    assert not is_fatal_worker_exception(wrapper)


@pytest.mark.parametrize("blocked", ["logging", "health", "stop", "readiness"])
def test_exit_deadline_survives_blocked_shutdown_callbacks(blocked):
    spec = importlib.util.find_spec("nurec.grpc.fatal_worker")
    assert spec is not None and spec.origin is not None
    script = r"""
import atexit
import importlib.util
import sys
import threading
spec = importlib.util.spec_from_file_location("fatal_worker", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
state = module.FatalWorkerExceptionInterceptor(exit_delay_s=0.1)
block = lambda *args, **kwargs: threading.Event().wait()
atexit.register(block)
state.configure_shutdown(mark_unhealthy=block if sys.argv[2] == "health" else lambda: None,
                         stop_server=block if sys.argv[2] == "stop" else lambda grace: None)
if sys.argv[2] == "logging":
    module.log.critical = block
if sys.argv[2] == "readiness":
    entered = threading.Event()
    def publish():
        entered.set()
        block()
    threading.Thread(target=lambda: state.state.publish_readiness(publish), daemon=True).start()
    assert entered.wait(5)
print("fatal-path-entered", flush=True)
state.handle_fatal_exception(RuntimeError("CUDA error: unknown error"), operation="test")
threading.Event().wait()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(spec.origin).resolve()), blocked],
        timeout=10,
        capture_output=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )
    assert b"fatal-path-entered" in result.stdout, result.stderr.decode()
    assert result.returncode == 1, result.stderr.decode()


def test_failure_to_arm_watchdog_exits_immediately():
    terminate = mock.Mock()
    interceptor = FatalWorkerExceptionInterceptor(terminate_worker=terminate)
    with mock.patch("nurec.grpc.fatal_worker.threading.Timer") as timer:
        timer.return_value.start.side_effect = RuntimeError("cannot start thread")
        with pytest.raises(RuntimeError, match="cannot start thread"):
            interceptor.handle_fatal_exception(RuntimeError("CUDA error: unknown error"), operation="test")
    terminate.assert_called_once_with(1)


def test_watchdog_arms_once_before_logging_or_health_publication():
    order = []
    terminate = mock.Mock()
    interceptor = FatalWorkerExceptionInterceptor(terminate_worker=terminate)
    interceptor.configure_shutdown(mark_unhealthy=lambda: order.append("health"), stop_server=lambda grace: None)
    with (
        mock.patch("nurec.grpc.fatal_worker.threading.Timer") as timer,
        mock.patch("nurec.grpc.fatal_worker.log.critical", side_effect=lambda *a, **k: order.append("log")),
    ):
        timer.return_value.start.side_effect = lambda: order.append("armed")
        with futures.ThreadPoolExecutor(max_workers=8) as executor:
            assert all(
                executor.map(
                    lambda _: interceptor.handle_fatal_exception(
                        RuntimeError("CUDA error: unknown error"), operation="test"
                    ),
                    range(20),
                )
            )
        assert interceptor.wait_for_shutdown(2)
        timer.assert_called_once()
    assert order[0] == "armed"
    assert "log" in order and "health" in order
    terminate.assert_called_once_with(1)


def test_readiness_cannot_clear_fatal_state():
    state = FatalWorkerState()
    publish = mock.Mock()
    assert state.publish_readiness(publish)
    state.mark_fatal(FatalWorkerIncident("test", (), None, 1, {}))
    assert not state.publish_readiness(publish)
    publish.assert_called_once_with()


@pytest.mark.parametrize("delay", [-1, float("inf"), float("nan")])
def test_invalid_exit_deadline_is_rejected(delay):
    with pytest.raises(ValueError, match="finite and non-negative"):
        FatalWorkerExceptionInterceptor(exit_delay_s=delay)


SENSORSIM_SERVICE = "nurec.grpc.protos.sensorsim.SensorsimService"


@pytest.mark.parametrize("dedicated_health", [False, True])
def test_real_grpc_rejects_queued_and_later_requests_and_keeps_health_accessible(dedicated_health):
    servicer = health.HealthServicer()
    state = FatalWorkerExceptionInterceptor(terminate_worker=mock.Mock())

    def mark_unhealthy() -> None:
        for name in ("", SENSORSIM_SERVICE):
            servicer.set(name, health_pb2.HealthCheckResponse.NOT_SERVING)

    state.configure_shutdown(mark_unhealthy=mark_unhealthy, stop_server=lambda grace: None)
    for name in ("", SENSORSIM_SERVICE):
        servicer.set(name, health_pb2.HealthCheckResponse.SERVING)
    entered, release, queued = (threading.Event(), threading.Event(), threading.Event())
    calls = []

    def invoke(request, context):
        calls.append(request)
        if request == b"recoverable":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid scene")
        if request == b"fatal":
            entered.set()
            assert release.wait(5)
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return b"ok"

    class QueueObserver(grpc.ServerInterceptor):
        def intercept_service(self, continuation, details):
            result = continuation(details)
            if entered.is_set():
                queued.set()
            return result

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), interceptors=[state, QueueObserver()])
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                SENSORSIM_SERVICE, {"get_available_cameras": grpc.unary_unary_rpc_method_handler(invoke)}
            ),
        )
    )
    port = server.add_insecure_port("127.0.0.1:0")
    health_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1)) if dedicated_health else server
    health_port = health_server.add_insecure_port("127.0.0.1:0") if dedicated_health else port
    health_pb2_grpc.add_HealthServicer_to_server(servicer, health_server)
    server.start()
    if dedicated_health:
        health_server.start()
    try:
        with (
            grpc.insecure_channel(f"127.0.0.1:{port}") as channel,
            grpc.insecure_channel(f"127.0.0.1:{health_port}") as health_channel,
            mock.patch("nurec.grpc.fatal_worker.threading.Timer"),
        ):
            rpc = channel.unary_unary(f"/{SENSORSIM_SERVICE}/get_available_cameras")
            probe = health_pb2_grpc.HealthStub(health_channel)
            assert rpc(b"healthy", timeout=5) == b"ok"
            with pytest.raises(grpc.RpcError) as recoverable:
                rpc(b"recoverable", timeout=5)
            assert recoverable.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert not state.state.is_fatal
            watch = probe.Watch(health_pb2.HealthCheckRequest(service=SENSORSIM_SERVICE), timeout=5)
            assert next(watch).status == health_pb2.HealthCheckResponse.SERVING
            failed = rpc.future(b"fatal", timeout=5)
            assert entered.wait(5)
            waiting = rpc.future(b"queued", timeout=5)
            assert queued.wait(5)
            release.set()
            for response in (failed, waiting):
                with pytest.raises(grpc.RpcError) as error:
                    response.result()
                assert error.value.code() == grpc.StatusCode.UNAVAILABLE
            with pytest.raises(grpc.RpcError) as later:
                rpc(b"later", timeout=5)
            assert later.value.code() == grpc.StatusCode.UNAVAILABLE
            assert calls == [b"healthy", b"recoverable", b"fatal"]
            assert next(watch).status == health_pb2.HealthCheckResponse.NOT_SERVING
            watch.cancel()
            for name in ("", SENSORSIM_SERVICE):
                assert (
                    probe.Check(health_pb2.HealthCheckRequest(service=name), timeout=5).status
                    == health_pb2.HealthCheckResponse.NOT_SERVING
                )
    finally:
        assert state.wait_for_shutdown(2)
        release.set()
        server.stop(0).wait(5)
        if dedicated_health:
            health_server.stop(0).wait(5)


@pytest.mark.parametrize("grace", [-1, float("inf"), float("nan")])
def test_invalid_drain_grace_is_rejected(grace):
    with pytest.raises(ValueError, match="finite and non-negative"):
        FatalWorkerExceptionInterceptor(drain_grace_s=grace)


def test_shutdown_wait_includes_time_before_thread_start():
    entered, release = threading.Event(), threading.Event()
    terminate = mock.Mock()
    interceptor = FatalWorkerExceptionInterceptor(terminate_worker=terminate)

    def health():
        entered.set()
        assert release.wait(5)

    interceptor.configure_shutdown(mark_unhealthy=health, stop_server=lambda grace: None)
    with mock.patch("nurec.grpc.fatal_worker.threading.Timer"):
        observer = threading.Thread(
            target=lambda: interceptor.handle_fatal_exception(
                RuntimeError("CUDA error: unknown error"), operation="test"
            )
        )
        observer.start()
        try:
            assert entered.wait(2)
            assert not interceptor.wait_for_shutdown(0.01)
            terminate.assert_not_called()
        finally:
            release.set()
            observer.join(2)
        assert interceptor.wait_for_shutdown(2)
    terminate.assert_called_once_with(1)


def test_raising_stop_callback_still_terminates_once():
    terminate = mock.Mock()
    interceptor = FatalWorkerExceptionInterceptor(terminate_worker=terminate)
    interceptor.configure_shutdown(
        mark_unhealthy=lambda: None, stop_server=mock.Mock(side_effect=RuntimeError("reap failed"))
    )
    with mock.patch("nurec.grpc.fatal_worker.threading.Timer"):
        assert interceptor.handle_fatal_exception(RuntimeError("CUDA error: unknown error"), operation="owner")
        assert interceptor.wait_for_shutdown(2)
    terminate.assert_called_once_with(1)


def test_control_fault_preserves_operation_and_explicit_request_id_inside_rpc():
    interceptor = FatalWorkerExceptionInterceptor(terminate_worker=mock.Mock())
    interceptor.configure_shutdown(mark_unhealthy=lambda: None, stop_server=lambda grace: None)
    details = mock.Mock(method="/s/render", invocation_metadata=())

    def invoke(request, context):
        assert interceptor.handle_fatal_exception(
            RuntimeError("CUDA error: unknown error"),
            operation="owner_scene_load",
            scene_ids=("s", "s"),
            request_id="explicit-id",
        )

    handler = interceptor.intercept_service(lambda _: grpc.unary_unary_rpc_method_handler(invoke), details)
    assert handler.unary_unary is not None
    with mock.patch("nurec.grpc.fatal_worker.threading.Timer"):
        handler.unary_unary(mock.Mock(scene_id="s", items=(), request_id=None), None)
        assert interceptor.wait_for_shutdown(2)
    incident = interceptor.state.incident
    assert incident is not None
    assert incident.operation == "owner_scene_load"
    assert incident.rpc_method == "/s/render"
    assert incident.scene_ids == ("s",)
    assert incident.request_id == "explicit-id"


@pytest.mark.parametrize("stop_blocks", [False, True])
def test_fatal_lifecycle_cannot_return_into_interpreter_finalization(stop_blocks):
    spec = importlib.util.find_spec("nurec.grpc.fatal_worker")
    assert spec is not None and spec.origin is not None
    script = r"""
import atexit, importlib.util, sys, threading
spec = importlib.util.spec_from_file_location("fatal_worker", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
block = lambda *args: threading.Event().wait()
atexit.register(block)
interceptor = module.FatalWorkerExceptionInterceptor(exit_delay_s=60)
interceptor.configure_shutdown(mark_unhealthy=lambda: None,
    stop_server=block if sys.argv[2] == "True" else lambda grace: None)
print("fatal-lifecycle-entered", flush=True)
interceptor.handle_fatal_exception(RuntimeError("CUDA error: unknown error"), operation="startup")
interceptor.exit_if_fatal(timeout=0.05)
raise AssertionError("fatal lifecycle returned")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, spec.origin, str(stop_blocks)],
        timeout=10,
        capture_output=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )
    assert b"fatal-lifecycle-entered" in result.stdout, result.stderr.decode()
    assert result.returncode == 1, result.stderr.decode()


def test_nonfatal_lifecycle_does_not_exit():
    terminate = mock.Mock()
    interceptor = FatalWorkerExceptionInterceptor(terminate_worker=terminate)
    interceptor.exit_if_fatal(timeout=0)
    assert interceptor.wait_for_shutdown(0)
    terminate.assert_not_called()


@pytest.mark.parametrize("message", ["cuda error illegal address-book entry is missing", "cuda-error-illegal-address"])
def test_unsupported_colonless_diagnostic_does_not_trigger_exit(message):
    assert not is_fatal_worker_exception(RuntimeError(message))
