# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import unittest

from types import SimpleNamespace
from unittest import mock

import grpc

from nurec.grpc.fatal_worker import (
    FATAL_WORKER_UNAVAILABLE_DETAILS,
    FatalWorkerExceptionInterceptor,
    FatalWorkerIncident,
    FatalWorkerState,
    is_fatal_worker_exception,
)


class _RpcAborted(Exception):
    pass


class _Context:
    def __init__(self) -> None:
        self.code = None
        self.details = None

    def abort(self, code, details):
        self.code = code
        self.details = details
        raise _RpcAborted(details)


class TestFatalWorkerExceptionClassification(unittest.TestCase):
    def test_cuda_illegal_address_errors_are_fatal(self):
        messages = [
            "CUDA error: an illegal memory access was encountered",
            "cudaErrorIllegalAddress: an illegal memory access was encountered",
            "CUDA_ERROR_ILLEGAL_ADDRESS",
        ]
        for message in messages:
            with self.subTest(message=message):
                self.assertTrue(is_fatal_worker_exception(RuntimeError(message)))

    def test_wrapped_cuda_illegal_address_error_is_fatal(self):
        try:
            try:
                raise RuntimeError("CUDA error: an illegal memory access was encountered")
            except RuntimeError as cause:
                raise RuntimeError("render failed") from cause
        except RuntimeError as exc:
            self.assertTrue(is_fatal_worker_exception(exc))

    def test_other_cuda_and_request_errors_are_not_fatal(self):
        self.assertFalse(is_fatal_worker_exception(RuntimeError("CUDA out of memory")))
        self.assertFalse(is_fatal_worker_exception(ValueError("illegal memory access")))
        self.assertFalse(
            is_fatal_worker_exception(
                KeyError("Scene artifact_id=CUDA error: an illegal memory access was encountered not available")
            )
        )
        self.assertFalse(
            is_fatal_worker_exception(
                RuntimeError("Failed to replace original_id='CUDA error: an illegal memory access was encountered'")
            )
        )


class TestFatalWorkerState(unittest.TestCase):
    def test_only_one_concurrent_caller_wins_fatal_transition(self):
        state = FatalWorkerState()
        barrier = threading.Barrier(16)
        results = []
        results_lock = threading.Lock()

        def mark(index):
            incident = FatalWorkerIncident(
                rpc_method=f"rpc-{index}",
                scene_ids=(),
                request_id=None,
                process_id=index,
                diagnostics={},
            )
            barrier.wait()
            won = state.mark_fatal(incident)
            with results_lock:
                results.append(won)

        threads = [threading.Thread(target=mark, args=(index,)) for index in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(results.count(True), 1)
        self.assertTrue(state.is_fatal)
        self.assertIsNotNone(state.incident)


class TestFatalWorkerExceptionInterceptor(unittest.TestCase):
    @staticmethod
    def _intercept(error: Exception, terminate_worker: mock.Mock):
        def fail(_request, _context):
            raise error

        handler = grpc.unary_unary_rpc_method_handler(fail)
        call_details = mock.MagicMock()
        call_details.method = "/nurec.grpc.SensorsimService/render_camera"
        interceptor = FatalWorkerExceptionInterceptor(terminate_worker=terminate_worker)
        intercepted = interceptor.intercept_service(lambda _details: handler, call_details)
        assert intercepted.unary_unary is not None
        return intercepted.unary_unary

    def test_fatal_error_exits_worker_with_failure(self):
        error = RuntimeError("CUDA error: an illegal memory access was encountered")
        terminate_worker = mock.Mock()
        invoke = self._intercept(error, terminate_worker)

        with self.assertRaisesRegex(RuntimeError, "illegal memory access"):
            invoke(None, None)

        terminate_worker.assert_called_once_with(1)

    def test_nonfatal_error_does_not_exit_worker(self):
        error = RuntimeError("ordinary render failure")
        terminate_worker = mock.Mock()
        invoke = self._intercept(error, terminate_worker)

        with self.assertRaisesRegex(RuntimeError, "ordinary render failure"):
            invoke(None, None)

        terminate_worker.assert_not_called()

    def test_fatal_error_coordinates_unhealthy_stop_and_exit_with_first_fault_context(self):
        events = []
        stopped = threading.Event()
        terminate_worker = mock.Mock(side_effect=lambda code: events.append(("exit", code)))
        interceptor = FatalWorkerExceptionInterceptor(
            terminate_worker=terminate_worker,
            diagnostics={"cuda_driver_version": "580.65", "gpu_uuid": "GPU-test"},
            drain_grace_s=0.25,
        )

        def stop_server(grace):
            events.append(("stop", grace))
            stopped.set()
            return stopped

        interceptor.configure_shutdown(
            mark_unhealthy=lambda: events.append(("health", "not-serving")),
            stop_server=stop_server,
        )

        def fail(_request, _context):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        handler = grpc.unary_unary_rpc_method_handler(fail)
        call_details = mock.MagicMock()
        call_details.method = "/nurec.grpc.protos.sensorsim.SensorsimService/get_available_cameras"
        call_details.invocation_metadata = (
            ("x-request-id", "request-7"),
            ("x-nre-scene-id", "metadata-scene"),
        )
        intercepted = interceptor.intercept_service(lambda _details: handler, call_details)
        assert intercepted.unary_unary is not None
        context = _Context()

        with self.assertRaisesRegex(_RpcAborted, "replacement worker"):
            intercepted.unary_unary(SimpleNamespace(scene_id="request-scene"), context)

        self.assertTrue(interceptor.wait_for_shutdown(timeout=1))
        self.assertEqual(context.code, grpc.StatusCode.UNAVAILABLE)
        self.assertEqual(context.details, FATAL_WORKER_UNAVAILABLE_DETAILS)
        self.assertEqual(events, [("health", "not-serving"), ("stop", 0.25), ("exit", 1)])
        incident = interceptor.state.incident
        assert incident is not None
        self.assertEqual(incident.rpc_method, call_details.method)
        self.assertEqual(incident.scene_ids, ("metadata-scene", "request-scene"))
        self.assertEqual(incident.request_id, "request-7")
        self.assertEqual(incident.diagnostics["gpu_uuid"], "GPU-test")

    def test_request_after_fatal_transition_is_rejected_before_handler(self):
        state = FatalWorkerState()
        state.mark_fatal(
            FatalWorkerIncident(
                rpc_method="first-rpc",
                scene_ids=(),
                request_id=None,
                process_id=1,
                diagnostics={},
            )
        )
        handler_callback = mock.Mock()
        handler = grpc.unary_unary_rpc_method_handler(handler_callback)
        call_details = mock.MagicMock()
        call_details.method = "/nurec.grpc.protos.sensorsim.SensorsimService/render_rgb"
        call_details.invocation_metadata = ()
        interceptor = FatalWorkerExceptionInterceptor(terminate_worker=mock.Mock(), state=state)
        intercepted = interceptor.intercept_service(lambda _details: handler, call_details)
        assert intercepted.unary_unary is not None
        context = _Context()

        with self.assertRaisesRegex(_RpcAborted, "replacement worker"):
            intercepted.unary_unary(None, context)

        self.assertEqual(context.code, grpc.StatusCode.UNAVAILABLE)
        handler_callback.assert_not_called()

    def test_fatal_error_outside_rpc_uses_same_coordinated_shutdown(self):
        events = []
        stopped = threading.Event()
        interceptor = FatalWorkerExceptionInterceptor(
            terminate_worker=lambda code: events.append(("exit", code)),
            diagnostics={"gpu_uuid": "GPU-worker"},
            drain_grace_s=0.1,
        )

        def stop_server(grace):
            events.append(("stop", grace))
            stopped.set()
            return stopped

        interceptor.configure_shutdown(
            mark_unhealthy=lambda: events.append(("health", "not-serving")),
            stop_server=stop_server,
        )

        handled = interceptor.handle_fatal_exception(
            RuntimeError("CUDA error: an illegal memory access was encountered"),
            operation="worker_scene_attach_build",
            scene_ids=("scene-b", "scene-a", "scene-b"),
        )

        self.assertTrue(handled)
        self.assertTrue(interceptor.wait_for_shutdown(timeout=1))
        self.assertEqual(events, [("health", "not-serving"), ("stop", 0.1), ("exit", 1)])
        incident = interceptor.state.incident
        assert incident is not None
        self.assertEqual(incident.operation, "worker_scene_attach_build")
        self.assertEqual(incident.scene_ids, ("scene-a", "scene-b"))
        self.assertEqual(incident.diagnostics["gpu_uuid"], "GPU-worker")

    def test_nonfatal_error_outside_rpc_is_ignored(self):
        terminate_worker = mock.Mock()
        interceptor = FatalWorkerExceptionInterceptor(terminate_worker=terminate_worker)

        self.assertFalse(
            interceptor.handle_fatal_exception(RuntimeError("ordinary load failure"), operation="scene_load")
        )
        self.assertFalse(interceptor.state.is_fatal)
        terminate_worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
