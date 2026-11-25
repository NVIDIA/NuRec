# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import os
import re
import threading

from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import grpc


log = logging.getLogger(__name__)

FATAL_WORKER_UNAVAILABLE_DETAILS = (
    "Worker unavailable after a fatal CUDA error; retry the request on a replacement worker"
)
DEFAULT_FATAL_WORKER_DRAIN_GRACE_S = 1.0
DEFAULT_FATAL_WORKER_EXIT_DELAY_S = 5.0
_HEALTH_RPC_PREFIX = "/grpc.health.v1.Health/"
REQUEST_ID_HEADERS = ("x-request-id", "request-id", "x-correlation-id")
_SCENE_ID_HEADER = "x-nre-scene-id"


_FATAL_CUDA_MESSAGE = re.compile(
    r"\b(?:CUDA(?: runtime)? error(?:\s*\(\d+\))?|"
    r"cuda[A-Za-z]+(?:\([^\n)]*\))? failed|render_pack kernel launch failed):\s*(?:"
    r"an illegal memory access was encountered|"
    r"uncorrectable ECC error encountered|"
    r"device-side assert triggered|"
    r"unspecified launch failure|"
    r"the launch timed out and was terminated|"
    r"an illegal instruction was encountered|"
    r"misaligned address|"
    r"hardware stack error|"
    r"invalid address space|"
    r"invalid program counter|"
    r"unknown error"
    r")\b|"
    r"(?:kernel failed:\s*|driver returned\s+)?\b(?:cudaError(?:IllegalAddress|ECCUncorrectable|Assert|LaunchFailure|LaunchTimeout|"
    r"IllegalInstruction|MisalignedAddress|HardwareStackError|InvalidAddressSpace|InvalidPc|Unknown)|"
    r"CUDA_ERROR_(?:ILLEGAL_ADDRESS|ECC_UNCORRECTABLE|ASSERT|LAUNCH_FAILED|LAUNCH_TIMEOUT|"
    r"ILLEGAL_INSTRUCTION|MISALIGNED_ADDRESS|HARDWARE_STACK_ERROR|INVALID_ADDRESS_SPACE|INVALID_PC|UNKNOWN))\b|"
    r"NvjpegBatchEncoder is unusable after an asynchronous CUDA failure\.",
    re.IGNORECASE,
)


def is_fatal_worker_exception(exc: BaseException) -> bool:
    """Detect local CUDA faults without interpreting request or downstream error text.

    Native diagnostics must start the RuntimeError message. Wrapped exceptions
    are inspected through both chains; a gRPC error belongs to another process.
    Client identifiers containing diagnostics do not establish a local fault;
    ordinary OOM and invalid-request errors retain their existing handling.
    """
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen or isinstance(current, grpc.RpcError):
            continue
        seen.add(id(current))
        if isinstance(current, RuntimeError):
            message = str(current).strip()
            if _FATAL_CUDA_MESSAGE.match(message):
                return True
        for related in (current.__cause__, current.__context__):
            if related is not None:
                pending.append(related)
    return False


@dataclass(frozen=True)
class FatalWorkerIncident:
    """First-fault context retained without keeping the exception or CUDA objects alive."""

    rpc_method: str | None
    scene_ids: tuple[str, ...]
    request_id: str | None
    process_id: int
    diagnostics: Mapping[str, str]
    operation: str = "rpc"


class FatalWorkerState:
    """Thread-safe one-way transition from serving to fatally unhealthy."""

    def __init__(self) -> None:
        self._fatal = threading.Event()
        self._lock = threading.Lock()
        self._publication_lock = threading.Lock()
        self._incident: FatalWorkerIncident | None = None

    @property
    def is_fatal(self) -> bool:
        return self._fatal.is_set()

    @property
    def incident(self) -> FatalWorkerIncident | None:
        with self._lock:
            return self._incident

    def mark_fatal(self, incident: FatalWorkerIncident) -> bool:
        """Publish ``incident`` once and return whether this caller won the transition."""

        with self._lock:
            if self._incident is not None:
                return False
            self._incident = incident
            self._fatal.set()
            return True

    def publish_readiness(self, publish: Callable[[], None]) -> bool:
        """Serialize health publication without blocking the fatal-state latch."""
        with self._publication_lock:
            if self._fatal.is_set():
                return False
            publish()
            return True

    def publish_unhealthy(self, publish: Callable[[], None]) -> None:
        # The exit watchdog is already armed before waiting for publication. A
        # blocked startup publisher must not hold the incident/transition lock.
        with self._publication_lock:
            publish()


def _invocation_metadata(handler_call_details: grpc.HandlerCallDetails) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key, value in handler_call_details.invocation_metadata or ():
        normalized_key = str(key).casefold()
        if isinstance(value, bytes):
            metadata[normalized_key] = value.decode(errors="replace")
        else:
            metadata[normalized_key] = str(value)
    return metadata


def _request_scene_ids(request: Any, metadata: Mapping[str, str]) -> tuple[str, ...]:
    scene_ids: set[str] = set()

    def add_scene_id(candidate: Any) -> None:
        if isinstance(candidate, str) and candidate:
            scene_ids.add(candidate)

    add_scene_id(metadata.get(_SCENE_ID_HEADER))
    add_scene_id(getattr(request, "scene_id", None))
    for item in getattr(request, "items", ()):
        add_scene_id(getattr(item, "scene_id", None))
        add_scene_id(getattr(getattr(item, "request", None), "scene_id", None))
    return tuple(sorted(scene_ids))


def _request_id(request: Any, metadata: Mapping[str, str]) -> str | None:
    metadata_id = next((metadata[header] for header in REQUEST_ID_HEADERS if metadata.get(header)), None)
    if metadata_id is not None:
        return metadata_id
    request_id = getattr(request, "request_id", None)
    return request_id if isinstance(request_id, str) and request_id else None


class FatalWorkerExceptionInterceptor(grpc.ServerInterceptor):
    """Coordinate shutdown after a fatal local CUDA error.

    ``is_fatal_worker_exception`` identifies diagnostics that leave the CUDA context
    unusable. The first observer atomically marks the worker fatal, publishes
    diagnostics, marks health down, stops the gRPC server with a bounded grace period,
    and exits nonzero. Later requests fail with ``UNAVAILABLE`` without entering the
    application handler. ``os._exit`` deliberately skips Python and CUDA cleanup,
    which may itself hang or fail in this state.
    """

    def __init__(
        self,
        terminate_worker: Callable[[int], Any] = os._exit,
        *,
        state: FatalWorkerState | None = None,
        diagnostics: Mapping[str, str] | None = None,
        drain_grace_s: float = DEFAULT_FATAL_WORKER_DRAIN_GRACE_S,
        exit_delay_s: float = DEFAULT_FATAL_WORKER_EXIT_DELAY_S,
    ) -> None:
        if not math.isfinite(drain_grace_s) or drain_grace_s < 0:
            raise ValueError("drain_grace_s must be finite and non-negative")
        if not math.isfinite(exit_delay_s) or exit_delay_s < 0:
            raise ValueError("exit_delay_s must be finite and non-negative")
        self._exit_delay_s = exit_delay_s
        self._exit_timer: threading.Timer | None = None
        self._terminate_worker = terminate_worker
        self.state = state if state is not None else FatalWorkerState()
        self._diagnostics = dict(diagnostics or {})
        self._drain_grace_s = drain_grace_s
        self._configuration_lock = threading.Lock()
        self._mark_unhealthy: Callable[[], None] | None = None
        self._stop_server: Callable[[float], Any] | None = None
        self._shutdown_complete = threading.Event()
        self._rpc_context: ContextVar[tuple[Any, grpc.HandlerCallDetails] | None] = ContextVar(
            "fatal_worker_rpc_context", default=None
        )

    def configure_shutdown(
        self,
        *,
        mark_unhealthy: Callable[[], None],
        stop_server: Callable[[float], Any],
    ) -> None:
        """Connect lifecycle callbacks after the interceptor's gRPC server exists."""

        with self._configuration_lock:
            self._mark_unhealthy = mark_unhealthy
            self._stop_server = stop_server

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        """Wait for termination, including the interval before shutdown callbacks start."""
        if not self.state.is_fatal:
            return True
        return self._shutdown_complete.wait(timeout)

    def exit_if_fatal(self, timeout: float = DEFAULT_FATAL_WORKER_EXIT_DELAY_S) -> None:
        """Finish a fatal lifecycle before returning into interpreter finalization.

        Give coordinated shutdown a bounded chance to drain/reap, then hard-exit
        even if it has stalled. Production termination never returns.
        """
        if self.state.is_fatal:
            self.wait_for_shutdown(timeout)
            self._exit_now()

    def _exit_now(self) -> None:
        if self._exit_timer is not None:
            self._exit_timer.cancel()
        try:
            self._terminate_worker(1)
        finally:
            # Observable only when an injected test terminator returns or raises.
            self._shutdown_complete.set()

    def _stop_and_terminate(self, stop_server: Callable[[float], Any]) -> None:
        try:
            stopped = stop_server(self._drain_grace_s)
            wait = getattr(stopped, "wait", None)
            if callable(wait) and wait(timeout=self._drain_grace_s + 1.0) is False:
                log.error("Timed out waiting for the fatal CUDA worker's gRPC server to stop")
        except Exception:
            log.exception("Failed while stopping the fatal CUDA worker's gRPC server")
        finally:
            self._exit_now()

    def handle_fatal_exception(
        self,
        exc: BaseException,
        *,
        operation: str,
        scene_ids: tuple[str, ...] = (),
        request_id: str | None = None,
    ) -> bool:
        """Enter the fatal transition for a CUDA error outside an RPC handler.

        Some CUDA work happens in worker control threads (for example, runtime scene
        attachment), so the gRPC wrapper cannot observe every poisoned-context error.
        Return whether ``exc`` is fatal; callers use that result to stop their control
        loop instead of swallowing the exception and attempting more CUDA work.
        """

        if not is_fatal_worker_exception(exc):
            return False
        rpc_method = None
        active_rpc = self._rpc_context.get()
        if active_rpc is not None:
            request, details = active_rpc
            metadata = _invocation_metadata(details)
            rpc_method = details.method
            scene_ids += _request_scene_ids(request, metadata)
            request_id = _request_id(request, metadata) or request_id
        incident = FatalWorkerIncident(
            rpc_method=rpc_method,
            operation=operation,
            scene_ids=tuple(sorted(set(scene_ids))),
            request_id=request_id,
            process_id=os.getpid(),
            diagnostics=self._diagnostics,
        )
        self._begin_shutdown(exc, incident)
        return True

    def _begin_shutdown(self, exc: BaseException, incident: FatalWorkerIncident) -> None:
        if not self.state.mark_fatal(incident):
            return

        # Arm the independent hard-exit deadline before callbacks or logging can
        # block. Normal coordinated shutdown still drains/reaps before exiting.
        timer = threading.Timer(self._exit_delay_s, self._exit_now)
        timer.name = "fatal-cuda-exit"
        timer.daemon = True
        self._exit_timer = timer
        try:
            timer.start()
        except Exception:
            self._exit_now()
            raise

        log.critical(
            "Fatal CUDA worker error; operation=%s rpc_method=%s scene_ids=%s request_id=%s pid=%d diagnostics=%s; "
            "marking the worker unavailable before exit",
            incident.operation,
            incident.rpc_method,
            incident.scene_ids,
            incident.request_id,
            incident.process_id,
            dict(sorted(incident.diagnostics.items())),
            exc_info=(type(exc), exc, exc.__traceback__),
        )

        with self._configuration_lock:
            mark_unhealthy = self._mark_unhealthy
            stop_server = self._stop_server

        if mark_unhealthy is not None:
            try:
                self.state.publish_unhealthy(mark_unhealthy)
            except Exception:
                log.exception("Failed to mark the fatal CUDA worker unhealthy")

        if stop_server is None:
            # Preserve fail-closed behavior if lifecycle wiring was unavailable.
            self._exit_now()
            return

        thread = threading.Thread(
            target=self._stop_and_terminate,
            args=(stop_server,),
            name="fatal-cuda-worker-shutdown",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            self._exit_now()
            raise

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        handler = continuation(handler_call_details)
        if handler is None or handler.unary_unary is None:
            return handler

        unary_unary = handler.unary_unary

        def invoke(request, context):
            if (
                context is not None
                and self.state.is_fatal
                and not handler_call_details.method.startswith(_HEALTH_RPC_PREFIX)
            ):
                context.abort(grpc.StatusCode.UNAVAILABLE, FATAL_WORKER_UNAVAILABLE_DETAILS)
            try:
                response = unary_unary(request, context)
            except Exception as exc:
                if is_fatal_worker_exception(exc):
                    metadata = _invocation_metadata(handler_call_details)
                    incident = FatalWorkerIncident(
                        rpc_method=handler_call_details.method,
                        scene_ids=_request_scene_ids(request, metadata),
                        request_id=_request_id(request, metadata),
                        process_id=os.getpid(),
                        diagnostics=self._diagnostics,
                    )
                    self._begin_shutdown(exc, incident)
                if (
                    context is not None
                    and self.state.is_fatal
                    and not handler_call_details.method.startswith(_HEALTH_RPC_PREFIX)
                ):
                    context.abort(grpc.StatusCode.UNAVAILABLE, FATAL_WORKER_UNAVAILABLE_DETAILS)
                raise
            if (
                context is not None
                and self.state.is_fatal
                and not handler_call_details.method.startswith(_HEALTH_RPC_PREFIX)
            ):
                context.abort(grpc.StatusCode.UNAVAILABLE, FATAL_WORKER_UNAVAILABLE_DETAILS)
            return response

        def terminate_on_fatal_worker_error(request, context):
            token = self._rpc_context.set((request, handler_call_details))
            try:
                return invoke(request, context)
            finally:
                self._rpc_context.reset(token)

        return grpc.unary_unary_rpc_method_handler(
            terminate_on_fatal_worker_error,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )
