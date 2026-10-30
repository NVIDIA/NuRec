# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the `RenderProxy` request dispatch (no GPU / no real gRPC servers).

These exercise the dispatch contract directly with recording fake stubs: render RPCs round-robin
across workers, discovery RPCs target one worker without disturbing the cursor, and — the key
concurrency property — the round-robin lock is held only to pick a worker, so N concurrent renders
fan out to N distinct workers and overlap in wall-clock rather than serializing.
"""

from __future__ import annotations

import contextlib
import gc
import inspect
import io
import logging
import multiprocessing
import os
import queue
import signal
import sys
import threading
import time
import weakref

from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import grpc
import pytest
import torch
import yaml

from nurec.grpc import multiproc_serve as _mp  # for monkeypatching module-level mps_active
from nurec.grpc.cache_keys import BackendCacheKey, SceneCacheKey
from nurec.grpc.model_owner import (
    AttachSceneRequest,
    ModelOwnerRegistry,
    WorkerLifecycleCommand,
    send_worker_stop,
)
from nurec.grpc.multiproc_serve import (
    _SCENE_ATTACH,
    _SCENE_DETACH,
    _SCENE_DETACH_MAX_ATTEMPTS,
    _SCENE_STOP,
    OnDemandSceneRegistrar,
    OwnerSceneRegistry,
    PoolShutdown,
    PoolTerminalReason,
    PostProcessConfig,
    RenderProxy,
    SceneAttachChannel,
    SceneAttachError,
    SceneBudgetError,
    SceneCoordinator,
    SceneDetachBusyError,
    SceneDownloadInterceptor,
    SceneLoadError,
    SceneNotKnownError,
    ScenePoolDegradedError,
    WorkerPool,
    WorkerSpawnConfig,
    WorkerStopReason,
    _await_pool_shutdown,
    _await_worker_ready,
    _bring_up_pool,
    _BringupCancelled,
    _drain_and_close_proxy,
    _plan_glob_scenes,
    _renderer_name_from_backend,
    _require_supported_renderer,
    _resolve_download_cache_size,
    _scene_attach_listener,
    _serve_until_stop,
    _signal_shutdown,
    _wait_for_worker_stop_reason,
    _WorkerHandle,
    _WorkerSceneTable,
    merge_metrics_yaml,
    mps_preflight,
    serve_proxy,
)
from nurec.grpc.protos import ipc_pb2, sensorsim_pb2
from nurec.grpc.protos.common_pb2 import Empty
from nurec.grpc.protos.sensorsim_pb2_grpc import (
    SensorsimServiceServicer,
    SensorsimServiceStub,
    add_SensorsimServiceServicer_to_server,
)
from nurec.grpc.scene_cache import SceneCache, SceneCacheEviction, SceneCacheEvictionError


@dataclass
class _CallRecord:
    worker_id: int
    method: str
    request: object
    start_s: float
    end_s: float
    timeout: float | None = None  # the deadline the proxy forwarded on this hop (None = no client deadline)


class _RecordingStub:
    """Stand-in for `SensorsimServiceStub` that records every call (and optionally blocks)."""

    def __init__(
        self,
        worker_id: int,
        calls: list[_CallRecord],
        calls_lock: threading.Lock,
        sleep_s: float = 0.0,
        barrier: threading.Barrier | None = None,
    ) -> None:
        self.worker_id = worker_id
        self._calls = calls
        self._calls_lock = calls_lock
        self._sleep_s = sleep_s
        self._barrier = barrier

    def _record(self, method: str, request: object, timeout: float | None = None) -> tuple:
        if self._barrier is not None:
            self._barrier.wait()  # gather all concurrent callers so their work windows line up
        start = time.perf_counter()
        if self._sleep_s:
            time.sleep(self._sleep_s)  # models a blocking worker render (releases the GIL, like gRPC I/O)
        end = time.perf_counter()
        with self._calls_lock:
            self._calls.append(_CallRecord(self.worker_id, method, request, start, end, timeout))
        return ("worker_response", self.worker_id, method, request)

    # Every RPC accepts the optional `timeout=` the proxy now forwards on each worker hop (the client's
    # remaining deadline budget) and records it, so a hop-less/deadline-less forward is still distinguishable.
    def render_rgb(self, request: object, timeout: float | None = None) -> tuple:
        return self._record("render_rgb", request, timeout)

    def batch_render_rgb(self, request: object, timeout: float | None = None) -> tuple:
        return self._record("batch_render_rgb", request, timeout)

    def render_lidar(self, request: object, timeout: float | None = None) -> tuple:
        return self._record("render_lidar", request, timeout)

    def get_version(self, request: object, timeout: float | None = None) -> tuple:
        return self._record("get_version", request, timeout)

    def get_available_scenes(self, request: object, timeout: float | None = None) -> tuple:
        return self._record("get_available_scenes", request, timeout)

    def get_loaded_scenes(self, request: object, timeout: float | None = None) -> tuple:
        return self._record("get_loaded_scenes", request, timeout)

    def get_available_cameras(self, request: object, timeout: float | None = None) -> tuple:
        return self._record("get_available_cameras", request, timeout)

    def get_available_trajectories(self, request: object, timeout: float | None = None) -> tuple:
        return self._record("get_available_trajectories", request, timeout)

    def get_dynamic_objects(self, request: object, timeout: float | None = None) -> tuple:
        return self._record("get_dynamic_objects", request, timeout)

    def get_external_asset_objects(self, request: object, timeout: float | None = None) -> tuple:
        return self._record("get_external_asset_objects", request, timeout)

    def get_server_config(self, request: object, timeout: float | None = None) -> object:
        # A real worker's SensorSimService(server_config=None) reports serving.mode=single_process; mirror
        # that (returning a ServerConfig proto) so the proxy's fallback-overlay path is exercised realistically.
        self._record("get_server_config", request, timeout)
        return sensorsim_pb2.ServerConfig(server_config={"serving.mode": "single_process"})

    def get_available_ego_masks(self, request: object, timeout: float | None = None) -> tuple:
        return self._record("get_available_ego_masks", request, timeout)


def _make_proxy(
    num_workers: int,
    sleep_s: float = 0.0,
    barrier: threading.Barrier | None = None,
) -> tuple[RenderProxy, list[_CallRecord]]:
    calls: list[_CallRecord] = []
    calls_lock = threading.Lock()
    stubs = [_RecordingStub(worker_id, calls, calls_lock, sleep_s, barrier) for worker_id in range(num_workers)]
    return RenderProxy(stubs), calls


def test_constructor_rejects_empty_stub_list() -> None:
    with pytest.raises(ValueError, match="at least one worker stub"):
        RenderProxy([])


@pytest.mark.parametrize(
    ("rpc_name", "render_request"),
    [
        (
            "render_rgb",
            sensorsim_pb2.RGBRenderRequest(scene_id="scene", image_format=sensorsim_pb2.ImageFormat.RGB_UINT8_PLANAR),
        ),
        (
            "render_camera",
            sensorsim_pb2.CameraRenderRequest(
                scene_id="scene", image_format=sensorsim_pb2.ImageFormat.RGB_UINT8_PLANAR
            ),
        ),
        (
            "batch_render_rgb",
            sensorsim_pb2.BatchRGBRenderRequest(
                items=[
                    sensorsim_pb2.BatchRGBRenderRequestItem(
                        camera_name="front",
                        request=sensorsim_pb2.RGBRenderRequest(
                            scene_id="scene", image_format=sensorsim_pb2.ImageFormat.RGB_UINT8_PLANAR
                        ),
                    )
                ]
            ),
        ),
        (
            "batch_render_camera",
            sensorsim_pb2.BatchCameraRenderRequest(
                items=[
                    sensorsim_pb2.BatchCameraRenderRequestItem(
                        camera_name="front",
                        request=sensorsim_pb2.CameraRenderRequest(
                            scene_id="scene", image_format=sensorsim_pb2.ImageFormat.RGB_UINT8_PLANAR
                        ),
                    )
                ]
            ),
        ),
    ],
)
def test_image_format_policy_rejects_at_proxy_before_worker_dispatch(rpc_name: str, render_request: object) -> None:
    calls: list[_CallRecord] = []
    proxy = RenderProxy(
        [_RecordingStub(0, calls, threading.Lock())],
        server_config_map={"allowed_image_formats": "JPEG"},
    )

    with pytest.raises(_AbortError) as exc:
        getattr(proxy, rpc_name)(render_request, _FakeContext())

    assert exc.value.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "RGB_UINT8_PLANAR" in exc.value.details
    assert "Allowed image formats: JPEG" in exc.value.details
    assert calls == []


@pytest.mark.parametrize(
    "server_config_map", [None, {"allowed_image_formats": "null"}, {"allowed_image_formats": "JPEG"}]
)
def test_image_format_policy_allows_unrestricted_or_listed_format(server_config_map: dict[str, str] | None) -> None:
    calls: list[_CallRecord] = []
    proxy = RenderProxy(
        [_RecordingStub(0, calls, threading.Lock())],
        server_config_map=server_config_map,
    )
    image_format = (
        sensorsim_pb2.ImageFormat.JPEG
        if server_config_map and server_config_map.get("allowed_image_formats") == "JPEG"
        else sensorsim_pb2.ImageFormat.RGB_UINT8_PLANAR
    )

    proxy.render_rgb(sensorsim_pb2.RGBRenderRequest(image_format=image_format), _FakeContext())

    assert [call.method for call in calls] == ["render_rgb"]


def test_render_rgb_round_robins_across_workers_in_order() -> None:
    proxy, calls = _make_proxy(num_workers=3)

    for index in range(7):
        proxy.render_rgb(f"req-{index}", context=None)

    assert [call.worker_id for call in calls] == [0, 1, 2, 0, 1, 2, 0]
    assert [call.method for call in calls] == ["render_rgb"] * 7


def test_render_rgb_and_lidar_share_one_round_robin_cursor() -> None:
    # render_rgb and render_lidar (single-dispatch RPCs) share the same cursor. (batch_render_rgb now
    # fans out per item — covered in the batch tests below.)
    proxy, calls = _make_proxy(num_workers=2)

    proxy.render_rgb("a", context=None)  # worker 0
    proxy.render_lidar("b", context=None)  # worker 1
    proxy.render_rgb("c", context=None)  # worker 0
    proxy.render_lidar("d", context=None)  # worker 1

    assert [(call.method, call.worker_id) for call in calls] == [
        ("render_rgb", 0),
        ("render_lidar", 1),
        ("render_rgb", 0),
        ("render_lidar", 1),
    ]


def test_discovery_rpcs_target_worker_zero_without_advancing_cursor() -> None:
    proxy, calls = _make_proxy(num_workers=3)

    proxy.get_version("v", context=None)
    proxy.get_available_scenes("s", context=None)
    proxy.get_available_cameras("c", context=None)
    proxy.get_available_trajectories("t", context=None)
    # The very next render still starts at worker 0: discovery must not consume the round-robin cursor.
    proxy.render_rgb("r0", context=None)
    proxy.render_rgb("r1", context=None)

    discovery = [call for call in calls if call.method != "render_rgb"]
    renders = [call for call in calls if call.method == "render_rgb"]
    assert {call.worker_id for call in discovery} == {0}
    assert [call.worker_id for call in renders] == [0, 1]


def test_render_rgb_forwards_request_and_returns_worker_response() -> None:
    proxy, _ = _make_proxy(num_workers=2)
    request = object()

    response = proxy.render_rgb(request, context=None)

    assert response[0] == "worker_response"
    assert response[1] == 0  # first worker
    assert response[2] == "render_rgb"
    assert response[3] is request  # request passed through untouched


def test_concurrent_renders_hit_distinct_workers_and_overlap() -> None:
    num_workers = 4
    # Barrier releases all callers together; the sleep then runs concurrently iff the proxy does not
    # serialize forwards. A timeout turns a (buggy) serialized implementation into a failure, not a hang.
    barrier = threading.Barrier(num_workers, timeout=5.0)
    proxy, calls = _make_proxy(num_workers=num_workers, sleep_s=0.2, barrier=barrier)

    with futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(lambda i: proxy.render_rgb(f"req-{i}", context=None), range(num_workers)))

    # Each of the N concurrent requests landed on a distinct worker.
    assert sorted(call.worker_id for call in calls) == list(range(num_workers))
    assert {result[1] for result in results} == set(range(num_workers))

    # All N work windows overlap: the latest start precedes the earliest end (true parallelism, not
    # serialization). With a 0.2 s sleep gated by the barrier, serialized execution could not overlap.
    latest_start = max(call.start_s for call in calls)
    earliest_end = min(call.end_s for call in calls)
    assert latest_start < earliest_end


# ---------------------------------------------------------------------------
# Lifecycle gate: owner death -> worker stops serving fast (no dangling mapping)
# ---------------------------------------------------------------------------


def _start_bare_server() -> tuple[grpc.Server, str]:
    """A real gRPC server on a random loopback port (no servicer needed to test reachability)."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}"


def _assert_reachable(address: str, timeout_s: float = 5.0) -> None:
    channel = grpc.insecure_channel(address)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
    finally:
        channel.close()


def _assert_unreachable(address: str, timeout_s: float = 3.0) -> None:
    channel = grpc.insecure_channel(address)
    try:
        with pytest.raises(grpc.FutureTimeoutError):
            grpc.channel_ready_future(channel).result(timeout=timeout_s)
    finally:
        channel.close()


def test_serve_until_stop_owner_sentinel_ready_stops_server() -> None:
    # Abnormal owner death: the owner's parent-process sentinel becomes ready, so the worker stops serving
    # at once (authoritative, no traffic / heartbeat / timeout involved).
    server, address = _start_bare_server()
    _assert_reachable(address)
    lifecycle_rx, lifecycle_tx = multiprocessing.Pipe(duplex=False)  # silent; never sends a stop
    sentinel_r, sentinel_w = os.pipe()  # models the owner's parent-process death sentinel fd
    os.close(sentinel_w)  # owner already gone -> the sentinel is readable from the first wait()
    try:
        reason = _serve_until_stop(
            server,
            lifecycle_rx,
            shutdown_grace_s=0.2,
            parent_sentinel=sentinel_r,
            is_parent_alive=lambda: False,
        )
    finally:
        with contextlib.suppress(OSError):
            os.close(sentinel_r)
        lifecycle_rx.close()
        lifecycle_tx.close()

    assert reason is WorkerStopReason.OWNER_DEATH
    _assert_unreachable(address)  # server is down: it no longer accepts connections


def test_serve_until_stop_graceful_stop_request_stops_server() -> None:
    # Normal shutdown: the owner sends a lifecycle STOP while it stays alive; the worker stops gracefully and
    # reports an owner-directed STOP (never owner death).
    server, address = _start_bare_server()
    _assert_reachable(address)
    lifecycle_rx, lifecycle_tx = multiprocessing.Pipe(duplex=False)
    sentinel_r, sentinel_w = os.pipe()  # owner stays alive: its sentinel never becomes ready
    send_worker_stop(lifecycle_tx)
    try:
        reason = _serve_until_stop(
            server,
            lifecycle_rx,
            shutdown_grace_s=0.2,
            parent_sentinel=sentinel_r,
            is_parent_alive=lambda: True,
        )
    finally:
        with contextlib.suppress(OSError):
            os.close(sentinel_w)
        with contextlib.suppress(OSError):
            os.close(sentinel_r)
        lifecycle_rx.close()
        lifecycle_tx.close()

    assert reason is WorkerStopReason.STOP
    _assert_unreachable(address)


def test_serve_until_stop_idle_worker_survives_without_traffic_or_messages() -> None:
    # Liveness is OS-derived, not traffic/heartbeat-derived. An idle worker with a live owner and no lifecycle
    # message keeps serving indefinitely (there is no heartbeat lease that could time it out), then stops
    # cleanly the moment the owner sends a lifecycle STOP (an owner-directed graceful stop, never inferred death).
    server, address = _start_bare_server()
    lifecycle_rx, lifecycle_tx = multiprocessing.Pipe(duplex=False)  # silent until the explicit stop below
    sentinel_r, sentinel_w = os.pipe()  # owner stays alive: its sentinel never becomes ready
    result: dict[str, WorkerStopReason] = {}

    def _serve() -> None:
        result["reason"] = _serve_until_stop(
            server,
            lifecycle_rx,
            shutdown_grace_s=0.2,
            parent_sentinel=sentinel_r,
            is_parent_alive=lambda: True,
        )

    serve_thread = threading.Thread(target=_serve)
    serve_thread.start()
    try:
        time.sleep(0.5)  # a brief idle window: liveness is OS-derived, so no traffic/message must not stop it
        assert serve_thread.is_alive()  # still serving: no traffic and no message is NOT death
        _assert_reachable(address)

        send_worker_stop(lifecycle_tx)  # owner requests a graceful stop
        serve_thread.join(timeout=5.0)
        assert not serve_thread.is_alive()
        assert result["reason"] is WorkerStopReason.STOP
        _assert_unreachable(address)
    finally:
        with contextlib.suppress(OSError):
            os.close(sentinel_w)
        with contextlib.suppress(OSError):
            os.close(sentinel_r)
        serve_thread.join(timeout=2.0)
        lifecycle_rx.close()
        lifecycle_tx.close()


# ---------------------------------------------------------------------------
# Worker stop-reason classification: parent-sentinel death vs. explicit lifecycle control
# ---------------------------------------------------------------------------


def _live_sentinel() -> "tuple[int, int]":
    """An os.pipe whose read fd models a still-open (never-ready) parent-death sentinel."""
    return os.pipe()  # both ends open -> the read fd is not readable, i.e. the parent is still alive


def test_wait_for_worker_stop_reason_stop_command_is_not_owner_death() -> None:
    # A lifecycle STOP while the parent is alive is an owner-directed graceful stop, not owner death.
    lifecycle_rx, lifecycle_tx = multiprocessing.Pipe(duplex=False)
    sentinel_r, sentinel_w = _live_sentinel()
    send_worker_stop(lifecycle_tx)
    try:
        reason = _wait_for_worker_stop_reason(lifecycle_rx, sentinel_r, lambda: True)
    finally:
        for fd in (sentinel_w, sentinel_r):
            with contextlib.suppress(OSError):
                os.close(fd)
        lifecycle_rx.close()
        lifecycle_tx.close()

    assert reason is WorkerStopReason.STOP


def test_wait_for_worker_stop_reason_owner_sentinel_ready_is_owner_death() -> None:
    # A ready parent-process sentinel is authoritative, immediate owner death.
    lifecycle_rx, lifecycle_tx = multiprocessing.Pipe(duplex=False)  # silent
    sentinel_r, sentinel_w = os.pipe()
    os.close(sentinel_w)  # owner gone -> sentinel readable
    try:
        reason = _wait_for_worker_stop_reason(lifecycle_rx, sentinel_r, lambda: False)
    finally:
        with contextlib.suppress(OSError):
            os.close(sentinel_r)
        lifecycle_rx.close()
        lifecycle_tx.close()

    assert reason is WorkerStopReason.OWNER_DEATH


def test_wait_for_worker_stop_reason_prefers_sentinel_over_lifecycle() -> None:
    # If both fire at once (owner SIGKILL closes the channel AND readies the sentinel), owner death wins over
    # any channel signal — the worker never misreports abrupt death as a graceful/requested stop.
    lifecycle_rx, lifecycle_tx = multiprocessing.Pipe(duplex=False)
    sentinel_r, sentinel_w = os.pipe()
    send_worker_stop(lifecycle_tx)  # a stop is also pending on the channel
    os.close(sentinel_w)  # ...but the sentinel is ready too
    try:
        reason = _wait_for_worker_stop_reason(lifecycle_rx, sentinel_r, lambda: False)
    finally:
        with contextlib.suppress(OSError):
            os.close(sentinel_r)
        lifecycle_rx.close()
        lifecycle_tx.close()

    assert reason is WorkerStopReason.OWNER_DEATH


def test_wait_for_worker_stop_reason_lifecycle_eof_with_live_parent_is_control_plane_failure() -> None:
    # The lifecycle channel closed (EOF) while the owner is still alive: the control plane broke under a live
    # owner, so the worker fails closed as a control-plane failure (never a clean/requested stop).
    lifecycle_rx, lifecycle_tx = multiprocessing.Pipe(duplex=False)
    sentinel_r, sentinel_w = _live_sentinel()
    lifecycle_tx.close()  # EOF on the read end
    try:
        reason = _wait_for_worker_stop_reason(lifecycle_rx, sentinel_r, lambda: True)
    finally:
        for fd in (sentinel_w, sentinel_r):
            with contextlib.suppress(OSError):
                os.close(fd)
        lifecycle_rx.close()

    assert reason is WorkerStopReason.CONTROL_PLANE_FAILURE


def test_wait_for_worker_stop_reason_lifecycle_bad_payload_with_live_parent_is_control_plane_failure() -> None:
    # A non-command payload on the lifecycle channel is a control-plane contract violation.
    lifecycle_rx, lifecycle_tx = multiprocessing.Pipe(duplex=False)
    sentinel_r, sentinel_w = _live_sentinel()
    lifecycle_tx.send("not-a-lifecycle-command")
    try:
        reason = _wait_for_worker_stop_reason(lifecycle_rx, sentinel_r, lambda: True)
    finally:
        for fd in (sentinel_w, sentinel_r):
            with contextlib.suppress(OSError):
                os.close(fd)
        lifecycle_rx.close()
        lifecycle_tx.close()

    assert reason is WorkerStopReason.CONTROL_PLANE_FAILURE


def test_wait_for_worker_stop_reason_lifecycle_eof_with_dead_parent_is_owner_death() -> None:
    # The same EOF, but the owner is gone: this is just owner death observed via the channel, not a control-
    # plane failure. (The sentinel is still open here so the channel EOF is what resolves the wait.)
    lifecycle_rx, lifecycle_tx = multiprocessing.Pipe(duplex=False)
    sentinel_r, sentinel_w = _live_sentinel()
    lifecycle_tx.close()
    try:
        reason = _wait_for_worker_stop_reason(lifecycle_rx, sentinel_r, lambda: False)
    finally:
        for fd in (sentinel_w, sentinel_r):
            with contextlib.suppress(OSError):
                os.close(fd)
        lifecycle_rx.close()

    assert reason is WorkerStopReason.OWNER_DEATH


def test_wait_for_worker_stop_reason_without_parent_signal_still_honors_stop_command() -> None:
    # A direct call with no multiprocessing parent (parent_sentinel/is_parent_alive both None): a lifecycle
    # STOP still resolves to a graceful stop, so unit calls do not depend on a real parent.
    lifecycle_rx, lifecycle_tx = multiprocessing.Pipe(duplex=False)
    send_worker_stop(lifecycle_tx)
    try:
        reason = _wait_for_worker_stop_reason(lifecycle_rx, None, None)
    finally:
        lifecycle_rx.close()
        lifecycle_tx.close()

    assert reason is WorkerStopReason.STOP


# ---------------------------------------------------------------------------
# Admission control: cap concurrent renders, shed RESOURCE_EXHAUSTED
# ---------------------------------------------------------------------------


class _AbortError(Exception):
    """What our fake gRPC context raises from abort() (real grpc terminates the RPC the same way)."""

    def __init__(self, code: object, details: str) -> None:
        self.code = code
        self.details = details
        super().__init__(f"{code}: {details}")


class _FakeContext:
    """Minimal ServicerContext double. ``time_remaining`` defaults to None (no client deadline) so the
    proxy forwards no ``timeout=`` and preserves historical single-arg calls; pass a value to exercise
    deadline forwarding. ``add_callback`` records the proxy's cancellation hook so a test can fire it to
    simulate the client cancelling/abandoning the RPC."""

    def __init__(self, time_remaining: float | None = None) -> None:
        self._time_remaining = time_remaining
        self._callbacks: list = []

    def abort(self, code: object, details: str) -> None:
        raise _AbortError(code, details)

    def time_remaining(self) -> float | None:
        return self._time_remaining

    def add_callback(self, callback) -> bool:  # type: ignore[no-untyped-def]
        self._callbacks.append(callback)
        return True

    def _fire_callbacks(self) -> None:
        """Test hook: simulate the client cancelling/expiring the RPC (gRPC fires these on termination)."""
        for callback in list(self._callbacks):
            callback()


def test_admission_disabled_by_default_does_not_shed() -> None:
    proxy, calls = _make_proxy(num_workers=2)  # max_in_flight defaults to 0 (unlimited)
    for index in range(6):
        proxy.render_rgb(f"r{index}", _FakeContext())
    assert len(calls) == 6  # all admitted, none shed


def test_admission_sheds_render_when_over_max_in_flight() -> None:
    # max_in_flight=1: while one render holds the slot, a concurrent render is shed RESOURCE_EXHAUSTED.
    holding = threading.Event()
    release = threading.Event()
    calls: list[_CallRecord] = []
    calls_lock = threading.Lock()

    class _BlockingStub(_RecordingStub):
        def render_rgb(self, request: object, timeout: float | None = None) -> tuple:
            holding.set()
            release.wait(timeout=5.0)
            return super().render_rgb(request)

    proxy = RenderProxy([_BlockingStub(0, calls, calls_lock)], max_in_flight=1)
    first_result: dict[str, object] = {}

    def _first() -> None:
        first_result["value"] = proxy.render_rgb("r0", _FakeContext())

    holder = threading.Thread(target=_first)
    holder.start()
    assert holding.wait(timeout=2.0)  # first render acquired the only slot and is blocked

    with pytest.raises(_AbortError) as exc:  # second render: no slot free → shed
        proxy.render_rgb("r1", _FakeContext())
    assert exc.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED

    release.set()
    holder.join(timeout=2.0)
    assert first_result["value"][0] == "worker_response"  # type: ignore[index] # the admitted render completed


def test_admission_releases_slot_after_render() -> None:
    # The slot is freed when a render returns, so sequential renders under a cap of 1 all succeed.
    proxy = RenderProxy([_RecordingStub(0, [], threading.Lock())], max_in_flight=1)
    for index in range(3):
        assert proxy.render_rgb(f"r{index}", _FakeContext())[0] == "worker_response"


def test_admission_queues_until_slot_frees_within_timeout() -> None:
    # With a non-zero admission timeout, an over-limit render *waits* for the slot rather than shedding.
    release = threading.Event()
    calls: list[_CallRecord] = []
    calls_lock = threading.Lock()

    class _BlockingStub(_RecordingStub):
        def __init__(self, *args: object) -> None:
            super().__init__(*args)  # type: ignore[arg-type]
            self._blocked_once = False

        def render_rgb(self, request: object, timeout: float | None = None) -> tuple:
            if not self._blocked_once:
                self._blocked_once = True
                release.wait(timeout=5.0)
            return super().render_rgb(request)

    proxy = RenderProxy([_BlockingStub(0, calls, calls_lock)], max_in_flight=1, admission_timeout_s=3.0)
    results: dict[str, object] = {}

    def _first() -> None:
        results["first"] = proxy.render_rgb("r0", _FakeContext())

    holder = threading.Thread(target=_first)
    holder.start()
    time.sleep(0.2)  # let the first render take the slot and block

    def _second() -> None:
        results["second"] = proxy.render_rgb("r1", _FakeContext())  # queues for the slot (timeout 3s)

    waiter = threading.Thread(target=_second)
    waiter.start()
    release.set()  # free the first render → the queued second acquires the slot and proceeds
    holder.join(timeout=3.0)
    waiter.join(timeout=3.0)
    assert results["first"][0] == "worker_response"  # type: ignore[index]
    assert results["second"][0] == "worker_response"  # type: ignore[index] # admitted after queueing, not shed


class _CountingBoundedSemaphore:
    """Wraps a BoundedSemaphore and counts successful acquires + releases, so a test can assert the shared
    admission primitive releases a permit exactly once per admitted render and never releases a shed one.
    (BoundedSemaphore itself also raises on any over-release, a second independent double-release guard.)"""

    def __init__(self, value: int) -> None:
        self._sem = threading.BoundedSemaphore(value)
        self.acquired = 0
        self.released = 0

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        got = self._sem.acquire(blocking=blocking, timeout=timeout)
        if got:
            self.acquired += 1
        return got

    def release(self) -> None:
        self.released += 1
        self._sem.release()


def test_admission_permit_accounting_across_success_error_and_exception() -> None:
    # every ADMITTED singular render releases its permit exactly once — on success, on a downstream
    # grpc.RpcError (relayed), and on an unexpected non-RpcError exception (propagated). The shared `_admit()`
    # primitive owns the release in a `finally`, so no outcome can leak a permit.
    class _MultiOutcomeStub(_RecordingStub):
        def __init__(self, *args: object) -> None:
            super().__init__(*args)  # type: ignore[arg-type]
            self.mode = "ok"

        def render_rgb(self, request: object, timeout: float | None = None) -> tuple:
            if self.mode == "rpc_error":
                raise _FakeRpcError(grpc.StatusCode.INTERNAL, "worker boom")
            if self.mode == "unexpected":
                raise RuntimeError("unexpected worker crash")
            return super().render_rgb(request)

    stub = _MultiOutcomeStub(0, [], threading.Lock())
    proxy = RenderProxy([stub], max_in_flight=1)
    sem = _CountingBoundedSemaphore(1)
    proxy._inflight = sem  # type: ignore[assignment] # noqa: SLF001 - instrument the real permit

    stub.mode = "ok"
    assert proxy.render_rgb("r", _FakeContext())[0] == "worker_response"
    assert (sem.acquired, sem.released) == (1, 1)  # success: acquired once, released once

    stub.mode = "rpc_error"
    with pytest.raises(_AbortError) as exc:
        proxy.render_rgb("r", _FakeContext())
    assert exc.value.code == grpc.StatusCode.INTERNAL  # relayed worker code, not a leaked-permit shed
    assert (sem.acquired, sem.released) == (2, 2)  # error path still released exactly once

    stub.mode = "unexpected"
    with pytest.raises(RuntimeError):
        proxy.render_rgb("r", _FakeContext())
    assert (sem.acquired, sem.released) == (3, 3)  # unexpected exception still released exactly once


def test_admission_shed_releases_no_permit() -> None:
    # a rejected acquisition must NOT release a permit it never took. Hold the only slot, then shed a
    # second render and assert the shed added zero acquires and zero releases.
    holding = threading.Event()
    release = threading.Event()

    class _BlockingStub(_RecordingStub):
        def render_rgb(self, request: object, timeout: float | None = None) -> tuple:
            holding.set()
            release.wait(timeout=5.0)
            return super().render_rgb(request)

    proxy = RenderProxy([_BlockingStub(0, [], threading.Lock())], max_in_flight=1)
    sem = _CountingBoundedSemaphore(1)
    proxy._inflight = sem  # type: ignore[assignment] # noqa: SLF001

    first: dict[str, object] = {}

    def _hold_slot() -> None:
        first["value"] = proxy.render_rgb("r0", _FakeContext())

    holder = threading.Thread(target=_hold_slot)
    holder.start()
    assert holding.wait(timeout=2.0)  # first render holds the only permit

    acquired_before, released_before = sem.acquired, sem.released
    with pytest.raises(_AbortError) as exc:
        proxy.render_rgb("r1", _FakeContext())  # no slot free → shed
    assert exc.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert sem.acquired == acquired_before  # shed acquired nothing
    assert sem.released == released_before  # and released nothing

    release.set()
    holder.join(timeout=2.0)
    assert first["value"][0] == "worker_response"  # type: ignore[index]
    assert sem.released == released_before + 1  # the admitted render released its one permit


# ---------------------------------------------------------------------------
# Deadline + cancellation propagation to the worker hop (single-process parity)
# ---------------------------------------------------------------------------


def test_render_forwards_client_deadline_to_worker() -> None:
    # The proxy must forward the client's remaining deadline budget as `timeout=` on the worker hop, so the
    # worker's queued-shedding policy sees the true budget instead of a deadline-less second hop.
    proxy, calls = _make_proxy(num_workers=1)
    proxy.render_rgb("r", _FakeContext(time_remaining=12.5))
    assert len(calls) == 1
    assert calls[0].timeout == pytest.approx(12.5)  # forwarded verbatim to the worker


def test_render_without_client_deadline_forwards_no_timeout() -> None:
    # With no client deadline (time_remaining None) the forward carries no timeout — byte-for-byte the
    # historical single-argument worker call.
    proxy, calls = _make_proxy(num_workers=1)
    proxy.render_rgb("r", _FakeContext())  # default: no deadline
    assert len(calls) == 1
    assert calls[0].timeout is None


def test_render_forwards_deadline_on_lidar_and_discovery_hops_too() -> None:
    # Deadline forwarding is uniform across the render_lidar and discovery hops, not just render_rgb.
    proxy, calls = _make_proxy(num_workers=1)
    proxy.render_lidar("r", _FakeContext(time_remaining=7.0))
    proxy.get_version("v", _FakeContext(time_remaining=3.0))
    by_method = {c.method: c.timeout for c in calls}
    assert by_method["render_lidar"] == pytest.approx(7.0)
    assert by_method["get_version"] == pytest.approx(3.0)


def test_client_without_deadline_forwards_no_timeout_even_when_time_remaining_is_inf_future() -> None:
    # Regression: grpc's ServicerContext.time_remaining() is max(deadline - now, 0) and NEVER returns None;
    # a deadline-less client surfaces as grpc's inf-future sentinel (~9.2e18 s). Forwarding that huge value as
    # a downstream timeout= overflows grpc's deadline math into an immediate DEADLINE_EXCEEDED, so a
    # deadline-less client would spuriously fail. `_client_timeout` must treat it as "no deadline" -> no
    # downstream timeout.
    proxy, calls = _make_proxy(num_workers=1)
    inf_future_remaining = 9.2e18  # what grpc reports for a client that set no deadline
    proxy.render_rgb("r", _FakeContext(time_remaining=inf_future_remaining))
    assert len(calls) == 1
    assert calls[0].timeout is None  # NOT forwarded verbatim (which would overflow into DEADLINE_EXCEEDED)


class _DeadlineExceededCall:
    """gRPC future-style call double whose `.result()` raises DEADLINE_EXCEEDED, as a worker hop does when the
    forwarded client budget expires."""

    def __init__(self, timeout: float | None) -> None:
        self.timeout = timeout

    def cancel(self) -> bool:
        return False

    def result(self) -> object:
        raise _FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "Deadline Exceeded")


class _DeadlineExceededMethod:
    def __init__(self, calls: list) -> None:
        self._calls = calls

    def future(self, request: object, timeout: float | None = None) -> _DeadlineExceededCall:
        call = _DeadlineExceededCall(timeout)
        self._calls.append(call)
        return call


class _DeadlineExceededStub:
    """A stub whose render_rgb AND get_server_config expose `.future` (so the proxy takes the future path)
    and always fail DEADLINE_EXCEEDED, to pin the typed relay on both a render and a discovery forward."""

    def __init__(self, calls: list) -> None:
        self.render_rgb = _DeadlineExceededMethod(calls)
        self.get_server_config = _DeadlineExceededMethod(calls)


def test_render_relays_downstream_deadline_exceeded_as_typed_status() -> None:
    # A downstream DEADLINE_EXCEEDED must relay to the client as DEADLINE_EXCEEDED, never collapse to UNKNOWN
    # ("Exception calling application") from an uncaught rendezvous escaping the handler.
    proxy = RenderProxy([_DeadlineExceededStub([])])
    with pytest.raises(_AbortError) as exc:
        proxy.render_rgb("r", _FakeContext())
    assert exc.value.code == grpc.StatusCode.DEADLINE_EXCEEDED
    assert "Deadline Exceeded" in exc.value.details


def test_discovery_forward_relays_downstream_deadline_exceeded_as_typed_status() -> None:
    # The same guarantee for a discovery forward (get_server_config): it must NOT let the downstream
    # rendezvous escape as UNKNOWN — the exact GPU-smoke regression this pins.
    proxy = RenderProxy([_DeadlineExceededStub([])])  # no server_config_map -> fallback forward path
    with pytest.raises(_AbortError) as exc:
        proxy.get_server_config(Empty(), _FakeContext())
    assert exc.value.code == grpc.StatusCode.DEADLINE_EXCEEDED
    assert "Deadline Exceeded" in exc.value.details


class _WedgedCall:
    """gRPC future-style call double modelling a wedged-but-alive worker: `.result()` blocks until `.cancel()`
    (the proxy's client-cancellation hook) fires, then raises CANCELLED like a real cancelled RPC."""

    def __init__(self, timeout: float | None) -> None:
        self.timeout = timeout
        self._cancelled = threading.Event()

    def cancel(self) -> bool:
        self._cancelled.set()
        return True

    def result(self) -> object:
        self._cancelled.wait(timeout=5.0)  # a wedged worker never returns on its own
        raise _FakeRpcError(grpc.StatusCode.CANCELLED, "downstream cancelled")


class _WedgedMethod:
    """A stub multicallable exposing `.future` (like a real gRPC stub method) so the proxy takes the
    cancellation-aware future path."""

    def __init__(self, calls: list) -> None:
        self._calls = calls

    def future(self, request: object, timeout: float | None = None) -> _WedgedCall:
        call = _WedgedCall(timeout)
        self._calls.append(call)
        return call


class _WedgedStub:
    def __init__(self, calls: list) -> None:
        self.render_rgb = _WedgedMethod(calls)


def test_render_client_cancellation_releases_admission_permit() -> None:
    # A client that cancels/abandons its render must free the proxy's admission permit: the proxy wires
    # context.add_callback to cancel the downstream future, unblocking the forward so `_admit`'s finally
    # releases the slot. Otherwise an abandoned request behind a wedged-but-alive worker pins a permit (and,
    # at scale, the whole executor) forever.
    made_calls: list = []
    proxy = RenderProxy([_WedgedStub(made_calls)], max_in_flight=1)

    first_ctx = _FakeContext()
    started = threading.Event()

    def _first() -> None:
        started.set()
        with contextlib.suppress(_AbortError):  # cancellation is relayed to the client as an abort
            proxy.render_rgb("r0", first_ctx)

    holder = threading.Thread(target=_first)
    holder.start()
    assert started.wait(timeout=2.0)

    # Wait until the forward is in flight AND its cancellation hook is registered (so the permit is held).
    for _ in range(500):
        if first_ctx._callbacks and made_calls:  # noqa: SLF001
            break
        time.sleep(0.01)
    assert made_calls and first_ctx._callbacks  # noqa: SLF001 - downstream future in flight, holding the permit

    # A second render is shed while the first still holds the only permit.
    with pytest.raises(_AbortError) as shed:
        proxy.render_rgb("r1", _FakeContext())
    assert shed.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED

    # The client cancels/abandons the first render -> the registered hook cancels the downstream future.
    first_ctx._fire_callbacks()  # noqa: SLF001
    holder.join(timeout=3.0)
    assert not holder.is_alive()  # the forward unwound once cancellation propagated

    # The permit is now free: a fresh render is admitted (its downstream future goes in flight).
    third_ctx = _FakeContext()

    def _third() -> None:
        with contextlib.suppress(_AbortError):
            proxy.render_rgb("r2", third_ctx)

    third = threading.Thread(target=_third)
    third.start()
    try:
        for _ in range(500):
            if len(made_calls) >= 2:
                break
            time.sleep(0.01)
        assert len(made_calls) >= 2  # admitted after the permit freed (would stay 1 if the permit leaked)
    finally:
        third_ctx._fire_callbacks()  # noqa: SLF001 - unblock the wedged third render so its thread exits
        third.join(timeout=3.0)


class _FutureExcCall:
    """A gRPC future-style call whose `.result()` raises a non-`grpc.RpcError` future exception — exactly what
    real grpc raises from `call.result()` on a cancelled/timed-out downstream future."""

    def __init__(self, exc: BaseException, timeout: float | None = None) -> None:
        self._exc = exc
        self.timeout = timeout

    def cancel(self) -> bool:
        return True

    def result(self) -> object:
        raise self._exc


class _FutureExcStub:
    """A stub exposing `.future` (so the proxy takes the cancellation-aware future path) whose call raises the
    given non-RpcError future exception."""

    def __init__(self, exc: BaseException) -> None:
        self.render_rgb = SimpleNamespace(future=lambda request, timeout=None: _FutureExcCall(exc, timeout))


@pytest.mark.parametrize(
    ("future_exc", "expected_code"),
    [
        (grpc.FutureCancelledError(), grpc.StatusCode.CANCELLED),
        (grpc.FutureTimeoutError(), grpc.StatusCode.DEADLINE_EXCEEDED),
    ],
)
def test_forward_translates_future_cancel_and_timeout_to_clean_abort(future_exc, expected_code) -> None:
    # F034: `stub.method.future(...).result()` raises grpc.FutureCancelledError (client cancel/deadline mid-
    # forward) or grpc.FutureTimeoutError. Neither is a grpc.RpcError subclass, so without translation they
    # escape render_rgb as an uncaught "Exception calling application" ERROR traceback. The proxy must convert
    # them to a clean CANCELLED / DEADLINE_EXCEEDED abort via context.abort, not raise them through uncaught.
    proxy = RenderProxy([_FutureExcStub(future_exc)])
    ctx = _FakeContext()
    with pytest.raises(_AbortError) as exc:
        proxy.render_rgb("r", ctx)
    assert exc.value.code == expected_code


def test_admission_wait_bounded_by_client_deadline() -> None:
    # F019: a render queued for an admission slot must be bounded by the client's remaining deadline, not the
    # full admission timeout. With admission_timeout_s=30 but a 0.3s client deadline, the second render sheds
    # in well under the 30s timeout instead of pinning its handler thread for the whole wait.
    holding = threading.Event()
    release = threading.Event()

    class _BlockingStub(_RecordingStub):
        def render_rgb(self, request: object, timeout: float | None = None) -> tuple:
            holding.set()
            release.wait(timeout=5.0)
            return super().render_rgb(request)

    proxy = RenderProxy([_BlockingStub(0, [], threading.Lock())], max_in_flight=1, admission_timeout_s=30.0)
    holder = threading.Thread(target=lambda: proxy.render_rgb("r0", _FakeContext()))
    holder.start()
    try:
        assert holding.wait(timeout=2.0)  # first render holds the only slot

        started = time.monotonic()
        with pytest.raises(_AbortError) as shed:
            proxy.render_rgb("r1", _FakeContext(time_remaining=0.3))  # queues, bounded by the 0.3s deadline
        elapsed = time.monotonic() - started
        assert shed.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert elapsed < 5.0  # shed on the short client deadline, NOT the 30s admission timeout
    finally:
        release.set()
        holder.join(timeout=2.0)


def test_admission_wait_bails_on_client_cancellation() -> None:
    # F019: a client that cancels while queued for a slot frees its handler thread promptly (the wait polls
    # cancellation), instead of staying pinned for the full admission timeout.
    class _CancelableContext(_FakeContext):
        def __init__(self) -> None:
            super().__init__(time_remaining=None)
            self._active = True

        def is_active(self) -> bool:
            return self._active

        def cancel(self) -> None:
            self._active = False

    holding = threading.Event()
    release = threading.Event()

    class _BlockingStub(_RecordingStub):
        def render_rgb(self, request: object, timeout: float | None = None) -> tuple:
            holding.set()
            release.wait(timeout=5.0)
            return super().render_rgb(request)

    proxy = RenderProxy([_BlockingStub(0, [], threading.Lock())], max_in_flight=1, admission_timeout_s=30.0)
    holder = threading.Thread(target=lambda: proxy.render_rgb("r0", _FakeContext()))
    holder.start()
    ctx = _CancelableContext()
    outcome: dict[str, Any] = {}

    def _second() -> None:
        try:
            proxy.render_rgb("r1", ctx)
        except _AbortError as exc:
            outcome["abort"] = exc

    waiter = threading.Thread(target=_second)
    try:
        assert holding.wait(timeout=2.0)  # first render holds the only slot
        waiter.start()
        time.sleep(0.3)  # let the second render queue on admission
        ctx.cancel()  # the client goes away
        waiter.join(timeout=3.0)
        assert not waiter.is_alive()  # freed promptly on cancel, not pinned for 30s
        assert outcome["abort"].code == grpc.StatusCode.RESOURCE_EXHAUSTED
    finally:
        release.set()
        holder.join(timeout=2.0)
        waiter.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Drop-in parity: full RPC surface forwarded (no UNIMPLEMENTED) + graceful shut_down
# ---------------------------------------------------------------------------

# Every RPC of the SensorsimService proto the proxy must handle to be a drop-in for serve. Derived by
# introspecting the generated `SensorsimServiceServicer` (its public methods ARE the proto RPCs) rather than
# hand-listing, so a future proto RPC cannot be silently omitted from the parity guard. The exclusion set is
# empty: every RPC must be overridden — including create_harmonizer_session/end_harmonizer_session, which are
# overridden as FAILED_PRECONDITION refusals (temporal-harmonizer sessions are per-process state the proxy
# cannot route, and --enable-harmonizer-temporal is refused at multiproc launch) rather than left to fall
# through to the base servicer's generic UNIMPLEMENTED.
_SENSORSIM_RPC_EXCLUSIONS: set[str] = set()
_ALL_SENSORSIM_RPCS = sorted(
    name
    for name, attr in vars(SensorsimServiceServicer).items()
    if callable(attr) and not name.startswith("_") and name not in _SENSORSIM_RPC_EXCLUSIONS
)


def test_all_sensorsim_rpcs_introspection_covers_the_proto_surface() -> None:
    # Guard the guard: introspection must actually enumerate the full RPC set (a non-trivial list that
    # includes the two harmonizer-session RPCs the old hand-list omitted), or the parity check below is vacuous.
    assert len(_ALL_SENSORSIM_RPCS) >= 15
    assert {"create_harmonizer_session", "end_harmonizer_session"} <= set(_ALL_SENSORSIM_RPCS)


def test_proxy_overrides_every_sensorsim_rpc() -> None:
    # If any RPC fell through to the base servicer it would return UNIMPLEMENTED (not drop-in). edit_assets and
    # restore_model_parameters stay overridden too — as refusal stubs (editing is unsupported under multiproc,
    # see the two tests below) — so the base servicer's typed UNIMPLEMENTED never leaks for them either.
    for name in _ALL_SENSORSIM_RPCS:
        assert getattr(RenderProxy, name) is not getattr(SensorsimServiceServicer, name), (
            f"RenderProxy does not override {name} -> would return UNIMPLEMENTED"
        )


def test_multiproc_edit_assets_refuses_in_band_with_success_false() -> None:
    # Editing is not supported with multi-process serving: edit_assets returns transport-OK with
    # success=False and an explanatory message (never a typed abort), preserving the `response.success`
    # client contract. Use single-process serve-grpc for in-place edits.
    proxy = RenderProxy([_RecordingStub(0, [], threading.Lock())])
    response = proxy.edit_assets(sensorsim_pb2.EditAssetsRequest(scene_id="s"), _FakeContext())
    assert response.success is False
    assert "not supported with multi-process serving" in response.message


def test_multiproc_restore_model_parameters_aborts_unimplemented() -> None:
    # There are no in-place edits to restore under multi-process serving, so restore_model_parameters aborts
    # UNIMPLEMENTED (a typed status) rather than silently succeeding.
    proxy = RenderProxy([_RecordingStub(0, [], threading.Lock())])
    with pytest.raises(_AbortError) as exc:
        proxy.restore_model_parameters(sensorsim_pb2.RestoreModelParametersRequest(scene_id="s"), _FakeContext())
    assert exc.value.code == grpc.StatusCode.UNIMPLEMENTED
    assert "not supported with multi-process serving" in exc.value.details


def test_multiproc_create_harmonizer_session_aborts_failed_precondition() -> None:
    # Harmonizer sessions are per-process state the proxy cannot route across the worker pool; the RPC must
    # abort FAILED_PRECONDITION with an actionable message (mirroring single-process serve), never fall
    # through to the base servicer's generic UNIMPLEMENTED "Method not implemented!".
    proxy = RenderProxy([_RecordingStub(0, [], threading.Lock())])
    with pytest.raises(_AbortError) as exc:
        proxy.create_harmonizer_session(Empty(), _FakeContext())
    assert exc.value.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "harmonizer sessions are not supported with multi-process serving" in exc.value.details


def test_multiproc_end_harmonizer_session_aborts_failed_precondition() -> None:
    proxy = RenderProxy([_RecordingStub(0, [], threading.Lock())])
    with pytest.raises(_AbortError) as exc:
        proxy.end_harmonizer_session(sensorsim_pb2.HarmonizerSessionId(session_id="x"), _FakeContext())
    assert exc.value.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "harmonizer sessions are not supported with multi-process serving" in exc.value.details


def test_multiproc_open_ipc_session_aborts_unimplemented() -> None:
    # A CUDA IPC session's buffer pool lives on one worker process; the proxy cannot
    # guarantee a use_cuda_ipc render reaches that same worker, so both IPC RPCs are
    # refused with a typed UNIMPLEMENTED rather than silently succeeding.
    proxy = RenderProxy([_RecordingStub(0, [], threading.Lock())])
    with pytest.raises(_AbortError) as exc:
        proxy.open_ipc_session(ipc_pb2.OpenIpcSessionRequest(), _FakeContext())
    assert exc.value.code == grpc.StatusCode.UNIMPLEMENTED
    assert "not supported with multi-process serving" in exc.value.details


def test_multiproc_close_ipc_session_aborts_unimplemented() -> None:
    proxy = RenderProxy([_RecordingStub(0, [], threading.Lock())])
    with pytest.raises(_AbortError) as exc:
        proxy.close_ipc_session(ipc_pb2.CloseIpcSessionRequest(session_id="x"), _FakeContext())
    assert exc.value.code == grpc.StatusCode.UNIMPLEMENTED
    assert "not supported with multi-process serving" in exc.value.details


def test_multiproc_render_rgb_use_cuda_ipc_aborts_unimplemented() -> None:
    # open_ipc_session is always refused under multi-process serving, so a
    # use_cuda_ipc=true render can never have a valid ipc_session_id here. It should
    # get the same typed UNIMPLEMENTED the dedicated IPC RPCs give, not be forwarded
    # to an arbitrary worker to fail there with an opaque error.
    proxy = RenderProxy([_RecordingStub(0, [], threading.Lock())])
    request = sensorsim_pb2.RGBRenderRequest(use_cuda_ipc=True, ipc_session_id="sess-1")
    with pytest.raises(_AbortError) as exc:
        proxy.render_rgb(request, _FakeContext())
    assert exc.value.code == grpc.StatusCode.UNIMPLEMENTED
    assert "not supported with multi-process serving" in exc.value.details


def test_multiproc_render_camera_use_cuda_ipc_aborts_unimplemented() -> None:
    # Same reasoning as render_rgb, for the non-deprecated single-camera render.
    proxy = RenderProxy([_RecordingStub(0, [], threading.Lock())])
    request = sensorsim_pb2.CameraRenderRequest(use_cuda_ipc=True, ipc_session_id="sess-1")
    with pytest.raises(_AbortError) as exc:
        proxy.render_camera(request, _FakeContext())
    assert exc.value.code == grpc.StatusCode.UNIMPLEMENTED
    assert "not supported with multi-process serving" in exc.value.details


def test_get_server_config_and_ego_masks_forward_to_worker_zero() -> None:
    proxy, calls = _make_proxy(num_workers=3)
    proxy.get_server_config("cfg", context=None)
    proxy.get_available_ego_masks("masks", context=None)
    # The very next render still starts at worker 0: these discovery RPCs must not consume the cursor.
    proxy.render_rgb("r0", context=None)
    proxy.render_rgb("r1", context=None)

    discovery = [c for c in calls if c.method in ("get_server_config", "get_available_ego_masks")]
    renders = [c for c in calls if c.method == "render_rgb"]
    assert {c.worker_id for c in discovery} == {0}
    assert [c.worker_id for c in renders] == [0, 1]


def test_shut_down_sets_event_and_acks() -> None:
    event = threading.Event()
    proxy = RenderProxy([_RecordingStub(0, [], threading.Lock())], shutdown_event=event)
    result = proxy.shut_down("x", context=None)
    assert event.is_set()  # signals the owner main loop to tear down the fleet
    assert result is not None  # returns common_pb2.Empty()


def test_shut_down_without_event_is_noop_ack() -> None:
    proxy = RenderProxy([_RecordingStub(0, [], threading.Lock())])
    assert proxy.shut_down("x", context=None) is not None  # no event wired (unit) -> no-op ack, no crash


# ---------------------------------------------------------------------------
# Config surface + launch + MPS preflight
# ---------------------------------------------------------------------------


def test_renderer_name_from_backend_maps_all() -> None:
    from nurec.config.model import RendererBackend

    assert _renderer_name_from_backend(RendererBackend.GSPLAT) == "gsplat"
    assert _renderer_name_from_backend(RendererBackend.DEFAULT) == "default"


def test_require_supported_renderer_gsplat_only() -> None:
    # The multi-process pool is validated only for gsplat. gsplat passes; the unvalidated
    # `default` fallback is rejected fast, before any worker spawns.
    _require_supported_renderer("gsplat")  # no raise

    with pytest.raises(ValueError, match="only the 'gsplat' renderer"):
        _require_supported_renderer("default")


def test_mps_preflight_noop_for_single_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mp, "mps_active", lambda: False)
    msgs: list[str] = []
    mps_preflight(1, require_mps=True, progress=msgs.append)  # single worker doesn't need MPS
    assert msgs == []


def test_mps_preflight_warns_when_mps_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mp, "mps_active", lambda: False)
    msgs: list[str] = []
    mps_preflight(2, require_mps=False, progress=msgs.append)
    assert any("MPS not detected" in m for m in msgs)


def test_mps_preflight_raises_with_require_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mp, "mps_active", lambda: False)
    with pytest.raises(RuntimeError, match="MPS not detected"):
        mps_preflight(2, require_mps=True)


def test_mps_preflight_silent_when_mps_active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mp, "mps_active", lambda: True)
    msgs: list[str] = []
    mps_preflight(8, require_mps=True, progress=msgs.append)
    assert msgs == []  # MPS active -> no warning, no raise


# NOTE: MPS is started externally (operator / container entrypoint / initContainer), never by this server;
# nurec/utils/mps_prelaunch_test.py covers the stdlib-only detection helper. This module keeps only the
# in-process mps_preflight verification tests.


# ---: scene-cache budget guard --------------------------------------------------------------


def test_resolve_download_cache_size_noop_when_large_enough() -> None:
    msgs: list[str] = []
    # download cache already >= resident budget: nothing to do, no warning.
    assert _resolve_download_cache_size(10, 10, progress=msgs.append) == 10
    assert _resolve_download_cache_size(20, 10, progress=msgs.append) == 20
    assert msgs == []


def test_resolve_download_cache_size_noop_when_eviction_disabled() -> None:
    msgs: list[str] = []
    # max_scenes <= 0 means no resident cap / no eviction, so the on-disk cache can't be undersized.
    assert _resolve_download_cache_size(1, 0, progress=msgs.append) == 1
    assert msgs == []


def test_resolve_download_cache_size_auto_bumps_with_info() -> None:
    msgs: list[str] = []
    # Default: align the on-disk cache with the resident budget and report the adjustment at INFO.
    assert _resolve_download_cache_size(5, 10, progress=msgs.append) == 10
    assert len(msgs) == 1
    assert msgs[0].startswith("INFO: ")
    assert "WARNING" not in msgs[0]
    assert "download_cache_size=5 < max_scenes=10" in msgs[0]
    assert "Auto-bumping download_cache_size -> 10" in msgs[0]


def test_resolve_download_cache_size_raises_with_require_scene_cache() -> None:
    # opt-in fail-fast: an undersized download cache is a hard error instead of a silent bump.
    with pytest.raises(ValueError, match=r"download_cache_size=5 < max_scenes=10.*--multiproc-require-scene-cache"):
        _resolve_download_cache_size(5, 10, require_scene_cache=True)


def test_resolve_download_cache_size_require_scene_cache_ok_when_large_enough() -> None:
    # --multiproc-require-scene-cache only fails on a real mismatch; a correctly-sized cache is unchanged.
    assert _resolve_download_cache_size(10, 10, require_scene_cache=True) == 10


def test_get_server_config_returns_effective_map_when_set() -> None:
    # Includes the zero-scene startup values (0 startup + 0 validated scenes) to prove they round-trip
    # unchanged through the owner proxy alongside the rest of the effective map.
    config_map = {
        "port": "8080",
        "renderer": "gsplat",
        "multiproc.workers": "2",
        "multiproc.startup_scenes": "0",
        "multiproc.validated_startup_scenes": "0",
    }
    proxy = RenderProxy([_RecordingStub(0, [], threading.Lock())], server_config_map=config_map)
    response = proxy.get_server_config("x", context=None)
    assert dict(response.server_config) == config_map  # answered by the owner proxy, not forwarded


def test_get_server_config_forwards_when_no_map() -> None:
    proxy, calls = _make_proxy(num_workers=2)  # no server_config_map
    proxy.get_server_config("cfg", context=None)
    assert [c.method for c in calls] == ["get_server_config"]  # falls back to forwarding to worker 0


def test_get_server_config_fallback_overlays_true_multiprocess_topology() -> None:
    # The standalone `multiproc-serve` CLI builds no server_config_map, so get_server_config falls back to
    # forwarding worker 0's config — which reports serving.mode=single_process (SensorSimService(None)). The
    # proxy must overlay the pool's true topology, else a drop-in client reads single_process for a running
    # pool (a present-but-wrong discovery key).
    proxy = RenderProxy([_RecordingStub(0, [], threading.Lock()) for _ in range(3)])  # no server_config_map
    response = proxy.get_server_config("cfg", _FakeContext())
    config = dict(response.server_config)
    assert config["serving.mode"] == "multiprocess"  # overlaid over the worker's single_process
    assert config["multiproc.workers"] == "3"  # reflects the real worker count


# ---------------------------------------------------------------------------
# Batch_render_rgb fan-out across workers (per-item dispatch + per-item error semantics)
# ---------------------------------------------------------------------------


class _FakeRpcError(grpc.RpcError):
    """A grpc.RpcError stand-in with code()/details(), as a worker render_rgb failure would raise."""

    def __init__(self, code: object, details: str) -> None:
        super().__init__()
        self._code = code
        self._details = details

    def code(self) -> object:
        return self._code

    def details(self) -> str:
        return self._details


class _BatchRenderStub:
    """Worker stub for batch tests: `render_rgb` records the worker, returns a real `RGBRenderReturn`, and
    fails items whose `frame_start_us` is in ``fail_indices`` (set per item so failure is worker-agnostic)."""

    def __init__(
        self, worker_id: int, used: list[int], lock: threading.Lock, fail_indices: set[int] | None = None
    ) -> None:
        self.worker_id = worker_id
        self._used = used
        self._lock = lock
        self._fail_indices = fail_indices or set()

    def render_rgb(self, request: object, timeout: float | None = None) -> object:
        with self._lock:
            self._used.append(self.worker_id)
        if request.frame_start_us in self._fail_indices:  # type: ignore[attr-defined]
            raise _FakeRpcError(grpc.StatusCode.INTERNAL, f"boom item {request.frame_start_us}")  # type: ignore[attr-defined]
        return sensorsim_pb2.RGBRenderReturn(image_bytes=f"w{self.worker_id}".encode())


def _batch_request(num_items: int, scene_id: str = "s", scene_ids: list[str] | None = None) -> object:
    items = []
    for i in range(num_items):
        sid = scene_ids[i] if scene_ids is not None else scene_id
        items.append(
            sensorsim_pb2.BatchRGBRenderRequestItem(
                camera_name=f"cam{i}",
                request=sensorsim_pb2.RGBRenderRequest(scene_id=sid, frame_start_us=i),
            )
        )
    return sensorsim_pb2.BatchRGBRenderRequest(items=items)


def test_batch_render_rgb_rejects_use_cuda_ipc() -> None:
    # Batch CUDA IPC delivery is refused under multi-process serving even though
    # single-process batch_render_rgb now supports it: the proxy's round-robin worker
    # fan-out has no way to reserve buffers across workers, and open_ipc_session is
    # refused here entirely, so no ipc_session_id could ever be valid regardless.
    proxy = RenderProxy([_BatchRenderStub(0, [], threading.Lock())])
    request = sensorsim_pb2.BatchRGBRenderRequest(
        items=[
            sensorsim_pb2.BatchRGBRenderRequestItem(
                camera_name="cam0",
                request=sensorsim_pb2.RGBRenderRequest(scene_id="s", use_cuda_ipc=True, ipc_session_id="sess-1"),
            )
        ]
    )
    with pytest.raises(_AbortError) as exc:
        proxy.batch_render_rgb(request, _FakeContext())
    assert exc.value.code == grpc.StatusCode.UNIMPLEMENTED
    assert "not supported with multi-process serving" in exc.value.details
    assert "cam0" in exc.value.details


def test_batch_render_camera_rejects_use_cuda_ipc() -> None:
    # Same reasoning as test_batch_render_rgb_rejects_use_cuda_ipc, for the non-deprecated
    # batch camera render.
    proxy = RenderProxy([_BatchRenderStub(0, [], threading.Lock())])
    request = sensorsim_pb2.BatchCameraRenderRequest(
        items=[
            sensorsim_pb2.BatchCameraRenderRequestItem(
                camera_name="cam0",
                request=sensorsim_pb2.CameraRenderRequest(scene_id="s", use_cuda_ipc=True, ipc_session_id="sess-1"),
            )
        ]
    )
    with pytest.raises(_AbortError) as exc:
        proxy.batch_render_camera(request, _FakeContext())
    assert exc.value.code == grpc.StatusCode.UNIMPLEMENTED
    assert "not supported with multi-process serving" in exc.value.details
    assert "cam0" in exc.value.details


def test_batch_render_rgb_empty_returns_empty() -> None:
    proxy = RenderProxy([_BatchRenderStub(0, [], threading.Lock())])
    response = proxy.batch_render_rgb(_batch_request(0), _FakeContext())
    assert list(response.items) == []


def test_batch_render_rgb_fans_out_and_preserves_order() -> None:
    used: list[int] = []
    stubs = [_BatchRenderStub(w, used, threading.Lock()) for w in range(2)]
    proxy = RenderProxy(stubs)

    response = proxy.batch_render_rgb(_batch_request(4), _FakeContext())

    assert [it.camera_name for it in response.items] == ["cam0", "cam1", "cam2", "cam3"]  # order preserved
    assert all(it.success for it in response.items)
    assert all(it.result.image_bytes.startswith(b"w") for it in response.items)
    assert set(used) == {0, 1}  # 4 items round-robined → both workers used (fan-out)


def test_batch_render_rgb_per_item_failure_does_not_fail_batch() -> None:
    used: list[int] = []
    lock = threading.Lock()
    stubs = [_BatchRenderStub(w, used, lock, fail_indices={1}) for w in range(2)]  # item index 1 fails
    proxy = RenderProxy(stubs)

    response = proxy.batch_render_rgb(_batch_request(4), _FakeContext())

    assert [it.camera_name for it in response.items] == ["cam0", "cam1", "cam2", "cam3"]
    assert [it.success for it in response.items] == [True, False, True, True]  # only item 1 failed
    assert "boom item 1" in response.items[1].error_message
    assert "INTERNAL" in response.items[1].error_message


def test_batch_render_rgb_mismatched_scene_aborts() -> None:
    proxy = RenderProxy([_BatchRenderStub(0, [], threading.Lock())])
    request = _batch_request(2, scene_ids=["s", "other"])
    with pytest.raises(_AbortError) as exc:
        proxy.batch_render_rgb(request, _FakeContext())
    assert exc.value.code == grpc.StatusCode.INVALID_ARGUMENT


def _write_metrics_yaml(directory: str, samples_by_name: dict[str, list[float]]) -> None:
    os.makedirs(directory, exist_ok=True)
    content = {
        "metrics": {"general": {name: [{"value": v} for v in vals] for name, vals in samples_by_name.items()}},
        "run_info": {"mode": "grpc_server"},
    }
    with open(os.path.join(directory, "metrics.yaml"), "w") as handle:
        yaml.dump(content, handle)


def test_merge_metrics_pools_samples_across_workers_and_proxy(tmp_path) -> None:
    base = str(tmp_path)
    _write_metrics_yaml(os.path.join(base, "worker_0"), {"render_time_ms": [10.0, 20.0]})
    _write_metrics_yaml(os.path.join(base, "worker_1"), {"render_time_ms": [30.0, 40.0]})
    _write_metrics_yaml(os.path.join(base, "proxy"), {"grpc.e2e_sent.render_rgb_ms": [50.0]})
    sources = [os.path.join(base, d) for d in ("worker_0", "worker_1", "proxy")]

    assert merge_metrics_yaml(base, sources)

    with open(os.path.join(base, "metrics.yaml")) as handle:
        merged = yaml.safe_load(handle)
    general = merged["metrics"]["general"]
    # render_time_ms pooled across both workers (fleet percentiles computed from these raw samples).
    assert sorted(s["value"] for s in general["render_time_ms"]) == [10.0, 20.0, 30.0, 40.0]
    # proxy e2e is a distinct key, preserved separately (not conflated with render_time_ms).
    assert [s["value"] for s in general["grpc.e2e_sent.render_rgb_ms"]] == [50.0]
    # The merged file carries aggregates too (full single-process shape), not just raw samples.
    assert merged.get("aggregated_metrics")
    assert "render_time_ms" in merged["aggregated_metrics"]


def test_merge_metrics_skips_missing_sources(tmp_path) -> None:
    base = str(tmp_path)
    _write_metrics_yaml(os.path.join(base, "worker_0"), {"render_time_ms": [1.0]})
    # worker_1 dir intentionally absent (a worker that was terminated before flushing)
    assert merge_metrics_yaml(base, [os.path.join(base, "worker_0"), os.path.join(base, "worker_1")])
    with open(os.path.join(base, "metrics.yaml")) as handle:
        merged = yaml.safe_load(handle)
    assert [s["value"] for s in merged["metrics"]["general"]["render_time_ms"]] == [1.0]


def _write_per_sequence_metrics_yaml(directory: str, scene_id: str, samples_by_name: dict[str, list[float]]) -> None:
    """Write the shape `serve` produces for scene-attributed timings: per_sequence, empty sensor id."""
    os.makedirs(directory, exist_ok=True)
    content = {
        "metrics": {
            "general": {},
            "per_sequence": {
                scene_id: {
                    "per_camera": {"": {name: [{"value": v} for v in vals] for name, vals in samples_by_name.items()}},
                    "per_lidar": {},
                }
            },
        },
        "metric_metadata": {name: {"timing_method": "wall_clock"} for name in samples_by_name},
        "run_info": {"mode": "grpc_server"},
    }
    with open(os.path.join(directory, "metrics.yaml"), "w") as handle:
        yaml.dump(content, handle)


def test_merge_metrics_pools_scene_attributed_samples(tmp_path) -> None:
    # serve records render_time_ms against its scene, so the samples land in per_sequence, not
    # general. Pooling only general would merge to an empty file and return False.
    base = str(tmp_path)
    _write_per_sequence_metrics_yaml(os.path.join(base, "worker_0"), "scene-a", {"render_time_ms": [10.0, 20.0]})
    _write_per_sequence_metrics_yaml(os.path.join(base, "worker_1"), "scene-a", {"render_time_ms": [30.0]})

    assert merge_metrics_yaml(base, [os.path.join(base, d) for d in ("worker_0", "worker_1")])

    with open(os.path.join(base, "metrics.yaml")) as handle:
        merged = yaml.safe_load(handle)
    pooled = merged["metrics"]["per_sequence"]["scene-a"]["per_camera"][""]["render_time_ms"]
    assert sorted(s["value"] for s in pooled) == [10.0, 20.0, 30.0]  # pooled across workers
    assert "scene-a" in merged["metrics"]["per_sequence"]  # and still attributed to the scene
    assert merged["aggregated_metrics"]["render_time_ms"]  # aggregates computed over the pooled samples
    # timing_method survives the merge, else a wall-clock span is indistinguishable from a CUDA-event one.
    assert merged["metric_metadata"]["render_time_ms"]["timing_method"] == "wall_clock"


def test_merge_metrics_returns_false_when_no_sources(tmp_path) -> None:
    assert not merge_metrics_yaml(str(tmp_path), [os.path.join(str(tmp_path), "nonexistent")])


class _RecordingMetricsManager:
    """Captures `record_batch_camera_render_time` calls (the proxy's batch metric).

    Both proxy batch paths record to this one counter, mirroring single-process `serve` — which
    routes its own `batch_render_rgb` to `record_batch_camera_render_time` as well.
    """

    def __init__(self) -> None:
        self.batch_calls: list[tuple[float, int]] = []
        self.batch_scene_ids: list[str | None] = []

    def record_batch_camera_render_time(
        self, render_time_ms: float, num_cameras: int, scene_id: str | None = None
    ) -> None:
        self.batch_calls.append((render_time_ms, num_cameras))
        self.batch_scene_ids.append(scene_id)


def test_batch_render_camera_attributes_the_batch_metric_to_its_scene() -> None:
    # The proxy already leases the batch's single scene; the batch timing must carry it, or the
    # merged file has no way to attribute the fleet's batch percentiles to a scene.
    metrics = _RecordingMetricsManager()
    proxy = RenderProxy([_CameraBatchRenderStub(0, [], threading.Lock())], metrics_manager=metrics)

    proxy.batch_render_camera(_camera_batch_request(2, scene_id="scene-7"), _FakeContext())

    assert metrics.batch_scene_ids == ["scene-7"]


def test_batch_render_rgb_records_batch_metric() -> None:
    metrics = _RecordingMetricsManager()
    stubs = [_BatchRenderStub(w, [], threading.Lock()) for w in range(2)]
    proxy = RenderProxy(stubs, metrics_manager=metrics)

    proxy.batch_render_rgb(_batch_request(3), _FakeContext())

    assert len(metrics.batch_calls) == 1  # one batch_render_time_ms recorded per batch RPC (parity w/ serve)
    render_time_ms, num_cameras = metrics.batch_calls[0]
    assert num_cameras == 3
    assert render_time_ms >= 0.0


class _CameraBatchRenderStub:
    """Worker stub for the camera batch path: `render_camera` records the worker and returns a real
    `CameraRenderReturn`, failing items whose `frame_start_us` is in ``fail_indices``."""

    def __init__(
        self, worker_id: int, used: list[int], lock: threading.Lock, fail_indices: set[int] | None = None
    ) -> None:
        self.worker_id = worker_id
        self._used = used
        self._lock = lock
        self._fail_indices = fail_indices or set()

    def render_camera(self, request: object, timeout: float | None = None) -> object:
        with self._lock:
            self._used.append(self.worker_id)
        if request.frame_start_us in self._fail_indices:  # type: ignore[attr-defined]
            raise _FakeRpcError(grpc.StatusCode.INTERNAL, f"boom item {request.frame_start_us}")  # type: ignore[attr-defined]
        return sensorsim_pb2.CameraRenderReturn()


def _camera_batch_request(num_items: int, scene_id: str = "s") -> object:
    return sensorsim_pb2.BatchCameraRenderRequest(
        items=[
            sensorsim_pb2.BatchCameraRenderRequestItem(
                camera_name=f"cam{i}",
                request=sensorsim_pb2.CameraRenderRequest(scene_id=scene_id, frame_start_us=i),
            )
            for i in range(num_items)
        ]
    )


def test_batch_render_camera_fans_out_and_preserves_per_item_contract() -> None:
    # `batch_render_camera` is the CURRENT batch RPC (main replaced the deprecated batch_render_rgb). It must
    # route through the pool rather than fall through to the base servicer's UNIMPLEMENTED, spread items across
    # workers, and keep serve's per-item contract: order + camera_name preserved, and a failing item surfaces
    # as that item's success=false (never a whole-batch abort).
    used: list[int] = []
    stubs = [_CameraBatchRenderStub(w, used, threading.Lock(), fail_indices={1}) for w in range(2)]
    metrics = _RecordingMetricsManager()
    proxy = RenderProxy(stubs, metrics_manager=metrics)

    response = proxy.batch_render_camera(_camera_batch_request(4), _FakeContext())

    assert [item.camera_name for item in response.items] == ["cam0", "cam1", "cam2", "cam3"]  # order preserved
    assert [item.success for item in response.items] == [True, False, True, True]  # only the failing item
    assert "INTERNAL: boom item 1" in response.items[1].error_message  # worker status decoded, not UNKNOWN
    assert set(used) == {0, 1}  # fanned across both workers
    assert metrics.batch_calls == [(metrics.batch_calls[0][0], 4)]  # one batch metric, 4 cameras


def test_batch_render_camera_empty_request_returns_empty_without_touching_workers() -> None:
    used: list[int] = []
    proxy = RenderProxy([_CameraBatchRenderStub(0, used, threading.Lock())])

    response = proxy.batch_render_camera(_camera_batch_request(0), _FakeContext())

    assert list(response.items) == []
    assert used == []  # short-circuits before any worker RPC


def test_render_camera_routes_to_a_worker() -> None:
    # The singular current-generation camera render must also be proxied (serve's render_grpc CLI calls it).
    used: list[int] = []
    proxy = RenderProxy([_CameraBatchRenderStub(7, used, threading.Lock())])

    proxy.render_camera(sensorsim_pb2.CameraRenderRequest(scene_id="s"), _FakeContext())

    assert used == [7]


def test_batch_render_rgb_admits_per_item() -> None:
    # max_in_flight=1: while one item holds the slot (blocked), the other items are shed per-item — the
    # batch can't spawn more than max_in_flight concurrent renders.
    release = threading.Event()
    used: list[int] = []

    class _BlockingBatchStub(_BatchRenderStub):
        def render_rgb(self, request: object, timeout: float | None = None) -> object:
            release.wait(timeout=5.0)
            return super().render_rgb(request)

    proxy = RenderProxy([_BlockingBatchStub(0, used, threading.Lock())], max_in_flight=1, admission_timeout_s=0.0)
    result: dict[str, object] = {}

    def _run() -> None:
        result["response"] = proxy.batch_render_rgb(_batch_request(4), _FakeContext())

    runner = threading.Thread(target=_run)
    runner.start()
    time.sleep(0.3)  # the one admitted item blocks; the other three shed
    release.set()
    runner.join(timeout=5.0)

    items = list(result["response"].items)  # type: ignore[attr-defined]
    successes = [it for it in items if it.success]
    sheds = [it for it in items if not it.success]
    assert len(successes) == 1  # only max_in_flight=1 admitted
    assert len(sheds) == 3
    assert all("RESOURCE_EXHAUSTED" in it.error_message for it in sheds)


class _CodelessRpcError(grpc.RpcError):
    """A worker RpcError missing code()/details() (both falsy), used to exercise the shared decoder's
    UNKNOWN/str(exc) fallbacks through both the singular and batch paths."""

    def __str__(self) -> str:
        return "codeless boom"

    def code(self) -> object:
        return None

    def details(self) -> object:
        return None


def test_extract_worker_rpc_error_normal_and_fallbacks() -> None:
    # the one shared decoder returns (code, details) verbatim for a well-formed RpcError, and applies
    # the canonical fallbacks — falsy code -> UNKNOWN, falsy details -> str(exc) — so the singular relay and
    # batch item formatting cannot diverge on the same downstream error.
    normal = _FakeRpcError(grpc.StatusCode.INVALID_ARGUMENT, "bad request")
    assert RenderProxy._extract_worker_rpc_error(normal) == (  # noqa: SLF001
        grpc.StatusCode.INVALID_ARGUMENT,
        "bad request",
    )

    code, details = RenderProxy._extract_worker_rpc_error(_CodelessRpcError())  # noqa: SLF001
    assert code == grpc.StatusCode.UNKNOWN
    assert details == "codeless boom"


def test_worker_error_strings_stable_across_singular_and_batch() -> None:
    # a normal downstream RpcError surfaces byte-for-byte identically via the shared decoder — the
    # singular relay aborts with (code, details) and the matching batch item is exactly "<CODE.name>: <details>".
    code = grpc.StatusCode.INVALID_ARGUMENT
    details = "bad request"

    class _FailingStub(_RecordingStub):
        def render_rgb(self, request: object, timeout: float | None = None) -> tuple:
            raise _FakeRpcError(code, details)

    singular = RenderProxy([_FailingStub(0, [], threading.Lock())])
    with pytest.raises(_AbortError) as exc:
        singular.render_rgb("r", _FakeContext())
    assert exc.value.code == code
    assert exc.value.details == details  # singular relay carries (code, details) verbatim

    class _FailingBatchStub(_BatchRenderStub):
        def render_rgb(self, request: object, timeout: float | None = None) -> object:
            with self._lock:
                self._used.append(self.worker_id)
            raise _FakeRpcError(code, details)

    batch = RenderProxy([_FailingBatchStub(0, [], threading.Lock())])
    response = batch.batch_render_rgb(_batch_request(1), _FakeContext())
    assert response.items[0].error_message == "INVALID_ARGUMENT: bad request"  # same code+details, batch shape


def test_singular_and_batch_use_shared_extractor_fallbacks() -> None:
    # a downstream RpcError WITHOUT code/details uses the same UNKNOWN/str(exc) fallback through both
    # public paths. Singular relays that status directly; batch formats it "<CODE.name>: <details>".
    class _CodelessSingularStub(_RecordingStub):
        def render_rgb(self, request: object, timeout: float | None = None) -> tuple:
            raise _CodelessRpcError()

    singular = RenderProxy([_CodelessSingularStub(0, [], threading.Lock())])
    with pytest.raises(_AbortError) as exc:
        singular.render_rgb("r", _FakeContext())
    assert exc.value.code == grpc.StatusCode.UNKNOWN
    assert exc.value.details == "codeless boom"

    class _CodelessBatchStub(_BatchRenderStub):
        def render_rgb(self, request: object, timeout: float | None = None) -> object:
            with self._lock:
                self._used.append(self.worker_id)
            raise _CodelessRpcError()

    proxy = RenderProxy([_CodelessBatchStub(0, [], threading.Lock())])
    response = proxy.batch_render_rgb(_batch_request(1), _FakeContext())
    item = response.items[0]
    assert not item.success
    assert item.error_message == "UNKNOWN: codeless boom"


def test_batch_item_failures_do_not_leak_admission_permit() -> None:
    # after a batch where one item hit a worker RpcError and another hit a generic (non-RpcError) error,
    # every admitted item must have released its permit. A positive wait makes all three concurrent items
    # deterministically acquire in sequence; a follow-up singular render succeeds only if nothing leaked.
    class _MixedFailBatchStub(_BatchRenderStub):
        def render_rgb(self, request: object, timeout: float | None = None) -> object:
            with self._lock:
                self._used.append(self.worker_id)
            idx = request.frame_start_us  # type: ignore[attr-defined]
            if idx == 0:
                raise _FakeRpcError(grpc.StatusCode.INTERNAL, "worker boom 0")
            if idx == 1:
                raise RuntimeError("generic boom 1")
            return sensorsim_pb2.RGBRenderReturn(image_bytes=b"ok")

    proxy = RenderProxy([_MixedFailBatchStub(0, [], threading.Lock())], max_in_flight=1, admission_timeout_s=1.0)
    response = proxy.batch_render_rgb(_batch_request(3), _FakeContext())
    items = list(response.items)
    assert items[0].error_message == "INTERNAL: worker boom 0"  # worker RpcError via shared decoder
    assert items[1].error_message == "INTERNAL: generic boom 1"  # generic per-item INTERNAL conversion
    assert items[2].success

    # No permit leaked: a follow-up capped render is admitted and returns the worker response.
    followup = proxy.render_rgb(sensorsim_pb2.RGBRenderRequest(scene_id="s", frame_start_us=99), _FakeContext())
    assert followup.image_bytes == b"ok"


# ---------------------------------------------------------------------------------------------------
# Owner scene registry (first-load serialization) + runtime scene attach/detach channel.
# CPU-only, no GPU: a bare nn.Module stands in for a loaded scene (its params/buffers are what the
# registry exports), and a fake `build_backend` stands in for the worker's load+alias.
# ---------------------------------------------------------------------------------------------------


def _tiny_model() -> torch.nn.Module:
    """A minimal module with params/buffers so `ModelOwnerRegistry.put_loaded` can enumerate them."""
    return torch.nn.Linear(2, 2)


def test_owner_scene_registry_loads_each_scene_once_under_concurrency() -> None:
    # First-load serialization: 8 concurrent callers for the same not-yet-loaded scene must trigger the
    # (expensive) loader exactly once and all receive the same owner object.
    osr = OwnerSceneRegistry()
    calls: list[int] = []
    calls_lock = threading.Lock()

    def loader() -> torch.nn.Module:
        with calls_lock:
            calls.append(1)
        time.sleep(0.05)  # widen the load window so the concurrent callers overlap on the per-scene lock
        return _tiny_model()

    barrier = threading.Barrier(8)
    results: list[object] = [None] * 8

    def run(i: int) -> None:
        barrier.wait()
        results[i] = osr.ensure_loaded("scene-a", loader)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert len(calls) == 1  # loaded exactly once despite 8 concurrent callers
    assert all(result is results[0] for result in results)  # everyone got the same owner object
    assert osr.is_loaded("scene-a") and osr.loaded_scenes() == ("scene-a",)


def test_owner_scene_registry_loads_distinct_scenes_independently() -> None:
    osr = OwnerSceneRegistry()
    seen: list[str] = []

    def loader_for(scene_id: str):
        def load() -> torch.nn.Module:
            seen.append(scene_id)
            return _tiny_model()

        return load

    owner_a = osr.ensure_loaded("a", loader_for("a"))
    owner_b = osr.ensure_loaded("b", loader_for("b"))
    again = osr.ensure_loaded("a", loader_for("a"))  # cached -> loader NOT called again

    assert seen == ["a", "b"]
    assert again is owner_a and owner_a is not owner_b
    assert set(osr.loaded_scenes()) == {"a", "b"}


def test_owner_scene_registry_loader_failure_allows_retry() -> None:
    osr = OwnerSceneRegistry()

    def bad() -> torch.nn.Module:
        raise RuntimeError("load boom")

    with pytest.raises(RuntimeError, match="load boom"):
        osr.ensure_loaded("a", bad)
    assert not osr.is_loaded("a")  # a failed load leaves the scene unloaded

    owner = osr.ensure_loaded("a", _tiny_model)  # a later call retries and succeeds
    assert osr.is_loaded("a") and owner is not None


def test_owner_scene_registry_release_only_when_idle() -> None:
    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)
    attachment = osr.registry.attach("a")  # a worker attachment is live

    assert osr.release("a") is False  # still attached -> not freed
    assert osr.is_loaded("a")

    osr.registry.detach(attachment.attachment_id)
    assert osr.release("a") is True  # idle now -> freed
    assert not osr.is_loaded("a")
    assert osr.release("a") is False  # already gone


# ---------------------------------------------------------------------------------------------------
# Worker-side alias and post-alias release (gc.collect + empty_cache)
# ---------------------------------------------------------------------------------------------------


class _AliasModule(torch.nn.Module):
    """A tiny module with BOTH a parameter and a buffer, so alias_external_weights exercises both paths."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 2))
        self.register_buffer("running_stat", torch.randn(2))


class _AliasBackend:
    """A minimal `Backend` stand-in: exposes ``renderable_model`` + a settable ``cache_key``, and is hashable by
    identity (unlike ``SimpleNamespace``, which defines ``__eq__`` and is thus unhashable) so it can live in the
    worker BackendCache's in-use/spares sets exactly like the real backend."""

    def __init__(self, module: "torch.nn.Module | None" = None, scene_id: str = "a") -> None:
        self.renderable_model = module if module is not None else torch.nn.Linear(2, 2)
        self.cache_key: "str | None" = None
        # `_WorkerSceneTable` publishes/removes by `backend_cache_key.key_hash` and registers the scene
        # off `backend_cache_key.scene_key`, so a Backend stand-in must carry one.
        self.backend_cache_key = BackendCacheKey(SceneCacheKey(scene_id=scene_id), "0" * 64)
        self.source_artifact = None


def _aliasable_backend(module: "torch.nn.Module | None" = None, scene_id: str = "a") -> _AliasBackend:
    """A minimal worker backend whose canonical model has real params (default: Linear) so
    alias_external_weights runs for real."""
    return _AliasBackend(module, scene_id)


class _StubArtifact:
    """Stand-in for `Artifact` at the pool's load seams.

    Carries the `scene_id` / `sha256_footprint` that `_pool_backend_cache_key` reads to build the
    backend's `BackendCacheKey`, so tests that stub out the artifact still exercise the real keying.
    """

    def __init__(self, _path=None, scene_id: str = "scene-a") -> None:  # noqa: ANN001
        self.scene_id = scene_id
        self.sha256_footprint = "0" * 64


class _ResidentSceneService:
    """Minimal `SensorSimService` stand-in for `_WorkerSceneTable`.

    Carries the backend cache plus the resident-backend registration seam the table drives: since
    `get_backend` resolves the artifact registry BEFORE the cache, the table must register every scene
    it publishes or a resident scene reads as NOT_FOUND. `registered` records the lifecycle so tests can
    assert attach/detach keeps the registry and the cache in step.
    """

    def __init__(self, cache) -> None:  # noqa: ANN001
        self.backend_cache = cache
        self.registered: dict = {}

    def register_resident_backend(self, backend) -> None:  # noqa: ANN001
        self.registered[backend.backend_cache_key.scene_key.key_hash] = backend

    def unregister_resident_backend(self, backend_cache_key) -> None:  # noqa: ANN001
        self.registered.pop(backend_cache_key.scene_key.key_hash, None)


def _matching_ipc_tensors(module: torch.nn.Module) -> dict:
    """External (owner IPC stand-in) tensors matching each param/buffer name/shape/dtype of ``module``."""
    tensors: dict = {}
    for name, param in module.named_parameters(recurse=True):
        tensors[name] = torch.zeros_like(param.detach())
    for name, buffer in module.named_buffers(recurse=True):
        tensors[name] = torch.zeros_like(buffer.detach())
    return tensors


def _named_tensor_ptr(module: torch.nn.Module, name: str) -> int:
    """The current storage pointer for ``name`` looked up via named_parameters/named_buffers (no dynamic attr
    access, so the value stays a typed tensor for mypy on a plain ``torch.nn.Module``)."""
    named = {**dict(module.named_parameters()), **dict(module.named_buffers())}
    return named[name].data_ptr()


def test_alias_and_release_frees_replaced_tensors_before_empty_cache(monkeypatch) -> None:
    # Central invariant: the discarded local parameter AND buffer objects must be UNREFERENCED (released),
    # not merely rebound to other storage, by the time empty_cache() runs — otherwise empty_cache frees nothing
    # the still-live locals occupy. Hold ONLY weakrefs to the originals (no strong test-local ref), run the real
    # gc.collect(), and assert both are dead inside the mocked empty_cache(); also confirm the aliasing landed.
    backend = _aliasable_backend(_AliasModule())
    module = _mp.canonical_model(backend.renderable_model)
    ipc = _matching_ipc_tensors(module)
    original_param_ref = weakref.ref(dict(module.named_parameters())["weight"])
    original_buffer_ref = weakref.ref(dict(module.named_buffers())["running_stat"])

    dead_at_empty_cache: dict = {}
    monkeypatch.setattr(_mp.torch.cuda, "is_available", lambda: True)

    def _empty_cache() -> None:
        dead_at_empty_cache["param"] = original_param_ref() is None
        dead_at_empty_cache["buffer"] = original_buffer_ref() is None

    monkeypatch.setattr(_mp.torch.cuda, "empty_cache", _empty_cache)

    _mp._alias_and_release_worker_weights(cast(_mp.Backend, backend), ipc)

    # The discarded local param + buffer were collected BEFORE empty_cache returned the blocks to the driver.
    assert dead_at_empty_cache == {"param": True, "buffer": True}
    # And the module now aliases the owner tensors (zero-copy rebinding of both a parameter and a buffer).
    assert _named_tensor_ptr(module, "weight") == ipc["weight"].data_ptr()
    assert _named_tensor_ptr(module, "running_stat") == ipc["running_stat"].data_ptr()


def test_alias_and_release_skips_empty_cache_without_cuda(monkeypatch) -> None:
    # On a CPU-only host the helper still aliases + collects, but must not touch empty_cache (no CUDA context).
    backend = _aliasable_backend()
    ipc = _matching_ipc_tensors(_mp.canonical_model(backend.renderable_model))
    collected: list[int] = []
    monkeypatch.setattr(_mp.gc, "collect", lambda *a, **k: collected.append(1))
    monkeypatch.setattr(_mp.torch.cuda, "is_available", lambda: False)

    def _boom() -> None:
        raise AssertionError("empty_cache must not run without CUDA")

    monkeypatch.setattr(_mp.torch.cuda, "empty_cache", _boom)

    _mp._alias_and_release_worker_weights(cast(_mp.Backend, backend), ipc)  # must not raise
    assert collected == [1]


def test_scene_backend_builder_aliases_and_releases(monkeypatch) -> None:
    # The runtime builder must alias the freshly built backend onto the owner IPC tensors AND drop the CUDA
    # cache for the replaced worker-local storage before returning the routable backend.
    ipc = _matching_ipc_tensors(torch.nn.Linear(2, 2))

    monkeypatch.setattr(_mp.Backend, "from_artifact", staticmethod(lambda *_a, **_k: _aliasable_backend()))
    monkeypatch.setattr(_mp, "Artifact", _StubArtifact)
    monkeypatch.setattr(_mp.torch.cuda, "is_available", lambda: True)
    empties: list[int] = []
    monkeypatch.setattr(_mp.torch.cuda, "empty_cache", lambda: empties.append(1))

    built = _mp._scene_backend_builder("gsplat")("scene-a", "/art", ipc)

    module = _mp.canonical_model(built.renderable_model)
    assert _named_tensor_ptr(module, "weight") == ipc["weight"].data_ptr()  # aliased
    assert empties == [1]  # cache returned to the driver as part of the build


def _fake_backend_cache():
    """A minimal worker BackendCache honoring the put/checkin/remove_by_key protocol `_WorkerSceneTable` drives."""

    class _Cache:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.in_use: set = set()
            self.spares: list = []
            self.maxsize = 8  # read (grow-only) by _WorkerSceneTable.__setitem__'s capacity bump

        def put(self, key: str, backend) -> None:  # noqa: ANN001
            backend.cache_key = key
            self.in_use.add(backend)

        def checkin(self, backend) -> None:  # noqa: ANN001
            self.in_use.discard(backend)
            self.spares.append(backend)

        def remove_by_key(self, key: str):  # noqa: ANN201
            with self.lock:
                backend = next((b for b in self.in_use if b.cache_key == key), None)
                if backend is not None:
                    self.in_use.discard(backend)
                else:
                    backend = next((b for b in self.spares if b.cache_key == key), None)
                    self.spares = [b for b in self.spares if b.cache_key != key]
                if backend is not None:
                    backend.cache_key = None
                return backend

    return _Cache()


def test_scene_attach_listener_runtime_cycle_releases_on_attach_and_reattach(monkeypatch) -> None:
    # Faithful runtime cycle: drive the REAL _scene_attach_listener + SceneAttachChannel + _WorkerSceneTable
    # (with the real _scene_backend_builder) through attach -> detach(evict) -> reattach. Each build must run
    # the alias+release, so the CUDA cache is returned to the driver once per attach (initial + reattach), and
    # the evicted scene leaves no resident/cache state between them.
    monkeypatch.setattr(_mp.Backend, "from_artifact", staticmethod(lambda *_a, **_k: _aliasable_backend()))
    monkeypatch.setattr(_mp, "Artifact", _StubArtifact)
    monkeypatch.setattr(_mp.torch.cuda, "is_available", lambda: True)
    empties: list[int] = []
    monkeypatch.setattr(_mp.torch.cuda, "empty_cache", lambda: empties.append(1))

    registry = OwnerSceneRegistry()
    registry.ensure_loaded("a", _tiny_model)  # owner Linear(2,2)-shaped weights the worker aliases onto
    cache = _fake_backend_cache()
    table = _WorkerSceneTable(_ResidentSceneService(cache))
    push_q: queue.Queue = queue.Queue()
    ack_q: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=_scene_attach_listener,
        args=(table, _mp._scene_backend_builder("gsplat"), push_q, ack_q),
        daemon=True,
    )
    thread.start()
    channel = SceneAttachChannel(push_q, ack_q, registry.registry, is_worker_alive=lambda: True)
    try:
        channel.attach("a", "/art", ack_timeout_s=2.0)
        assert "a" in table and len(cache.spares) == 1  # resident + registered as a spare
        assert empties == [1]  # released on the initial attach

        channel.detach("a", ack_timeout_s=2.0)  # evict
        assert "a" not in table and not cache.in_use and not cache.spares  # no residual worker cache state

        channel.attach("a", "/art", ack_timeout_s=2.0)  # reattach rebuilds + re-aliases + re-releases
        assert "a" in table and len(cache.spares) == 1
        assert empties == [1, 1]  # released again on the reattach build
    finally:
        channel.stop()
        thread.join(timeout=2.0)


def test_worker_bootstrap_aliases_and_releases_before_service_and_ready(monkeypatch) -> None:
    # Drive the REAL `_worker` bootstrap far enough to prove the build-and-release seam (line ~2903) runs and
    # completes BEFORE SensorSimService construction, cache publication, server start, or the "ready" message.
    # A monkeypatched SensorSimService records its own turn and raises, stopping the worker right after the
    # seam so no port is ever published; the attach handshake is stubbed so no real owner/GPU is needed.
    order: list[str] = []
    ipc = _matching_ipc_tensors(torch.nn.Linear(2, 2))

    monkeypatch.setattr(_mp, "_owner_parent_signals", lambda: (object(), lambda: True))  # OS supervision present
    monkeypatch.setattr(_mp, "Artifact", _StubArtifact)
    monkeypatch.setattr(_mp.Backend, "from_artifact", staticmethod(lambda *_a, **_k: _aliasable_backend()))

    def _record_release(_backend, _ipc) -> None:  # noqa: ANN001
        order.append("release")

    monkeypatch.setattr(_mp, "_alias_and_release_worker_weights", _record_release)

    class _StopAfterSeam(Exception):
        pass

    def _service(*_a, **_k):  # noqa: ANN002, ANN003
        order.append("service")  # reached only after the seam; stop before cache/server/ready
        raise _StopAfterSeam

    monkeypatch.setattr(_mp, "SensorSimService", _service)

    attach_resp_q: queue.Queue = queue.Queue()
    attach_resp_q.put(_mp.OwnerResponse(ok=True, attachment=cast(Any, SimpleNamespace(tensors=ipc))))
    port_q: queue.Queue = queue.Queue()

    with pytest.raises(SystemExit) as caught:
        _mp._worker(
            worker_id=2,
            scene_id="s",
            artifact_path="/art",
            config=WorkerSpawnConfig(renderer_name="gsplat", enable_editing_actors=False),
            attach_req_q=queue.Queue(),
            attach_resp_q=attach_resp_q,
            port_q=port_q,
            lifecycle_rx=None,
        )

    assert caught.value.code == 1  # our sentinel routed through the fault path (nonzero)
    # The bootstrap alias/release ran BEFORE the service was constructed...
    assert order == ["release", "service"]
    # ...and because we stopped at service construction, the worker never published a "ready" port.
    kind, wid, _payload = port_q.get_nowait()
    assert (kind, wid) == ("error", 2)
    assert port_q.empty()  # no "ready" was ever emitted


def test_worker_failed_port_bind_fails_startup_loudly(monkeypatch) -> None:
    # A worker whose `add_insecure_port` returns 0 (bind failure) must fail startup NONZERO and never
    # publish a "ready" port — mirroring the proxy/health bind guards. Drive the REAL `_worker` on a
    # zero-scene startup (scene_id=None → no attach/backend build) with a fake gRPC server whose
    # `add_insecure_port` returns 0; the guard must raise before `server.start()`/ready.
    started: list[int] = []

    class _UnbindableServer:
        def add_insecure_port(self, _target: str) -> int:
            return 0  # simulate an unavailable/in-use port

        def start(self) -> None:
            started.append(1)  # must NEVER be reached: the guard raises before start()

    monkeypatch.setattr(_mp, "_owner_parent_signals", lambda: (object(), lambda: True))  # OS supervision present
    monkeypatch.setattr(_mp.grpc, "server", lambda *_a, **_k: _UnbindableServer())
    monkeypatch.setattr(_mp, "SensorSimService", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(_mp, "add_SensorsimServiceServicer_to_server", lambda *_a, **_k: None)

    port_q: queue.Queue = queue.Queue()

    with pytest.raises(SystemExit) as caught:
        _mp._worker(
            worker_id=7,
            scene_id=None,  # zero-scene startup: no attach handshake / GPU needed
            artifact_path=None,
            config=WorkerSpawnConfig(renderer_name="gsplat", enable_editing_actors=False),
            attach_req_q=queue.Queue(),
            attach_resp_q=queue.Queue(),
            port_q=port_q,
            lifecycle_rx=None,
        )

    assert caught.value.code == 1  # bind failure routed through the fault path (nonzero)
    assert started == []  # the server was never started
    kind, wid, payload = port_q.get_nowait()
    assert (kind, wid) == ("error", 7)  # an error, not a phantom "ready" port
    assert "failed to bind worker gRPC server" in payload
    assert port_q.empty()  # no "ready" was ever emitted


def _start_scene_listener(builder):
    """Start a real `_scene_attach_listener` over fresh queues; returns (push_q, ack_q, attached, thread)."""
    push_q: queue.Queue = queue.Queue()
    ack_q: queue.Queue = queue.Queue()
    attached: dict = {}
    thread = threading.Thread(target=_scene_attach_listener, args=(attached, builder, push_q, ack_q), daemon=True)
    thread.start()
    return push_q, ack_q, attached, thread


def test_scene_attach_channel_attaches_and_refcounts() -> None:
    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)
    built: list[str] = []
    push_q, ack_q, attached, thread = _start_scene_listener(
        lambda scene_id, artifact_path, ipc_tensors: built.append(scene_id) or f"backend::{scene_id}"  # type: ignore[func-returns-value]
    )
    channel = SceneAttachChannel(push_q, ack_q, osr.registry)

    channel.attach("a", "/art")

    assert attached == {"a": "backend::a"}  # worker built + cached the scene
    assert built == ["a"]
    assert osr.registry.attached_count("a") == 1  # owner refcount bumped
    assert channel.attached_scenes() == {"a"}

    channel.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_scene_attach_channel_idempotent_attach() -> None:
    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)
    built: list[str] = []
    push_q, ack_q, _attached, thread = _start_scene_listener(
        lambda scene_id, artifact_path, ipc_tensors: built.append(scene_id) or f"backend::{scene_id}"  # type: ignore[func-returns-value]
    )
    channel = SceneAttachChannel(push_q, ack_q, osr.registry)

    channel.attach("a", "/art")
    channel.attach("a", "/art")  # idempotent: no second registry.attach and no rebuild

    assert built == ["a"]
    assert osr.registry.attached_count("a") == 1
    channel.stop()
    thread.join(timeout=2.0)


def test_scene_attach_channel_detach_drops_refcount_and_mapping() -> None:
    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)
    push_q, ack_q, attached, thread = _start_scene_listener(
        lambda scene_id, artifact_path, ipc_tensors: f"backend::{scene_id}"
    )
    channel = SceneAttachChannel(push_q, ack_q, osr.registry)

    channel.attach("a", "/art")
    channel.detach("a")

    assert attached == {}  # worker dropped its mapping
    assert osr.registry.attached_count("a") == 0
    assert channel.attached_scenes() == set()
    assert osr.release("a") is True  # idle after detach -> freeable
    channel.stop()
    thread.join(timeout=2.0)


def test_scene_attach_channel_rolls_back_refcount_on_worker_failure() -> None:
    # If the worker fails to build/alias, the channel must roll the owner refcount back so a failed attach
    # leaves no dangling attachment (otherwise release_if_idle would never free the scene).
    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)

    def failing_builder(scene_id, artifact_path, ipc_tensors):
        raise RuntimeError(f"build boom {scene_id}")

    push_q, ack_q, attached, thread = _start_scene_listener(failing_builder)
    channel = SceneAttachChannel(push_q, ack_q, osr.registry)

    with pytest.raises(RuntimeError, match="failed on worker"):
        channel.attach("a", "/art", ack_timeout_s=2.0)

    assert attached == {}
    assert osr.registry.attached_count("a") == 0  # refcount rolled back
    assert channel.attached_scenes() == set()
    channel.stop()
    thread.join(timeout=2.0)


def test_scene_attach_channel_two_scenes_detach_one_keeps_other() -> None:
    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)
    osr.ensure_loaded("b", _tiny_model)
    push_q, ack_q, attached, thread = _start_scene_listener(
        lambda scene_id, artifact_path, ipc_tensors: f"backend::{scene_id}"
    )
    channel = SceneAttachChannel(push_q, ack_q, osr.registry)

    channel.attach("a", "/art")
    channel.attach("b", "/art")
    assert set(attached) == {"a", "b"}

    channel.detach("a")

    assert set(attached) == {"b"}  # detaching a left b intact
    assert osr.registry.attached_count("a") == 0
    assert osr.registry.attached_count("b") == 1
    assert channel.attached_scenes() == {"b"}
    channel.stop()
    thread.join(timeout=2.0)


def test_scene_attach_channel_detach_keeps_refcount_when_worker_alive_but_unacked() -> None:
    # A detach whose ack times out while the worker is still ALIVE must NOT drop the owner refcount —
    # otherwise release_if_idle could cudaFree the IPC tensors the live worker still maps (use-after-free).
    # The refcount + attachment are retained (so a later detach can retry) and the fault is raised.
    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)
    attachment = osr.registry.attach("a")  # a live worker attachment (refcount 1)
    push_q: queue.Queue = queue.Queue()
    ack_q: queue.Queue = queue.Queue()  # no listener -> no ack -> detach times out
    channel = SceneAttachChannel(push_q, ack_q, osr.registry, is_worker_alive=lambda: True)
    channel.seed_attachment("a", attachment.attachment_id)

    with pytest.raises(RuntimeError, match="still-live worker"):
        channel.detach("a", ack_timeout_s=0.2)

    assert osr.registry.attached_count("a") == 1  # refcount RETAINED — not freed under a possibly-live mapping
    assert channel.is_attached("a")  # attachment kept so a later detach can retry
    assert osr.release("a") is False  # still attached -> release_if_idle cannot free it


def test_scene_attach_channel_detach_drops_refcount_when_worker_dead() -> None:
    # Once the worker is provably dead its IPC mapping is gone, so force-dropping the refcount is safe
    # (avoids a leak) — and the fault is still surfaced.
    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)
    attachment = osr.registry.attach("a")
    push_q: queue.Queue = queue.Queue()
    ack_q: queue.Queue = queue.Queue()  # no ack
    channel = SceneAttachChannel(push_q, ack_q, osr.registry, is_worker_alive=lambda: False)
    channel.seed_attachment("a", attachment.attachment_id)

    with pytest.raises(RuntimeError, match="dead worker"):
        channel.detach("a", ack_timeout_s=0.2)

    assert osr.registry.attached_count("a") == 0  # provably dead -> safe to drop
    assert not channel.is_attached("a")
    assert osr.release("a") is True  # idle now -> freeable


def test_scene_attach_listener_dispatch_attach_detach_unknown_stop() -> None:
    push_q: queue.Queue = queue.Queue()
    ack_q: queue.Queue = queue.Queue()
    attached: dict = {}
    built: list[tuple] = []

    def builder(scene_id, artifact_path, ipc_tensors):
        built.append((scene_id, artifact_path))
        return f"backend::{scene_id}"

    thread = threading.Thread(target=_scene_attach_listener, args=(attached, builder, push_q, ack_q), daemon=True)
    thread.start()

    push_q.put((_SCENE_ATTACH, 1, "a", "/art-a", {"w": 0}))
    assert ack_q.get(timeout=2.0) == (1, "ok", _SCENE_ATTACH)
    assert attached == {"a": "backend::a"} and built == [("a", "/art-a")]

    push_q.put((_SCENE_DETACH, 2, "a"))
    assert ack_q.get(timeout=2.0) == (2, "ok", _SCENE_DETACH)
    assert attached == {}

    push_q.put(("bogus", 3))
    seq, status, detail = ack_q.get(timeout=2.0)
    assert (seq, status) == (3, "error") and "unknown scene command" in detail

    push_q.put((_SCENE_STOP,))
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_worker_scene_table_partial_attach_reports_uncertain_and_detach_recovers() -> None:
    # A cache mutation can fail after retaining the backend. The production listener must not claim the attach
    # was rejected: it reports uncertainty, the owner keeps its refcount, and the compensating detach removes
    # both the partial worker cache state and the owner attachment.
    class _Backend:
        cache_key = None
        source_artifact = None
        backend_cache_key = BackendCacheKey(SceneCacheKey(scene_id="a"), "0" * 64)

    class _Cache:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.in_use: set = set()
            self.spares: list = []
            self.fail_checkin = True
            self.maxsize = 8  # read (grow-only) by _WorkerSceneTable.__setitem__'s capacity bump

        def put(self, key: str, backend) -> None:  # noqa: ANN001
            backend.cache_key = key
            self.in_use.add(backend)

        def checkin(self, backend) -> None:  # noqa: ANN001
            if self.fail_checkin:
                self.fail_checkin = False
                raise RuntimeError("checkin boom after cache put")
            self.in_use.remove(backend)
            self.spares.append(backend)

        def remove_by_key(self, key: str):  # noqa: ANN201
            with self.lock:
                backend = next((b for b in self.in_use if b.cache_key == key), None)
                if backend is not None:
                    self.in_use.discard(backend)
                else:
                    backend = next((b for b in self.spares if b.cache_key == key), None)
                    self.spares = [b for b in self.spares if b.cache_key != key]
                if backend is not None:
                    backend.cache_key = None
                return backend

    registry = OwnerSceneRegistry()
    registry.ensure_loaded("a", _tiny_model)
    cache = _Cache()
    table = _WorkerSceneTable(_ResidentSceneService(cache))
    push_q: queue.Queue = queue.Queue()
    ack_q: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=_scene_attach_listener,
        args=(table, lambda _sid, _path, _ipc: _Backend(), push_q, ack_q),
        daemon=True,
    )
    thread.start()
    channel = SceneAttachChannel(push_q, ack_q, registry.registry, is_worker_alive=lambda: True)
    try:
        with pytest.raises(RuntimeError, match=r"uncertain.*still-live worker"):
            channel.attach("a", "/art", ack_timeout_s=2.0)

        assert "a" in table and channel.is_attached("a")
        assert registry.registry.attached_count("a") == 1
        assert len(cache.in_use) == 1  # worker retained the partial installation

        channel.detach("a", ack_timeout_s=2.0)

        assert "a" not in table and not cache.in_use and not cache.spares
        assert registry.registry.attached_count("a") == 0
        assert registry.release("a") is True
    finally:
        channel.stop()
        thread.join(timeout=2.0)


def test_worker_scene_table_partial_detach_remains_retryable() -> None:
    # Do not remove `_scenes` before cache cleanup succeeds. A first detach that fails during cache removal stays
    # uncertain/owned; the idempotent retry can still find the backend and drive it to a proven-clean state.
    class _Backend:
        cache_key = None
        source_artifact = None
        backend_cache_key = BackendCacheKey(SceneCacheKey(scene_id="a"), "0" * 64)

    class _Cache:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.in_use: set = set()
            self._spares: list = []
            self.fail_spares_update = False
            self.maxsize = 8  # read (grow-only) by _WorkerSceneTable.__setitem__'s capacity bump

        @property
        def spares(self) -> list:
            return self._spares

        @spares.setter
        def spares(self, value: list) -> None:
            if self.fail_spares_update:
                self.fail_spares_update = False
                raise RuntimeError("spare removal boom")
            self._spares = value

        def put(self, key: str, backend) -> None:  # noqa: ANN001
            backend.cache_key = key
            self.in_use.add(backend)

        def checkin(self, backend) -> None:  # noqa: ANN001
            self.in_use.remove(backend)
            self.spares.append(backend)

        def remove_by_key(self, key: str):  # noqa: ANN201
            with self.lock:
                backend = next((b for b in self.in_use if b.cache_key == key), None)
                if backend is not None:
                    self.in_use.discard(backend)
                else:
                    backend = next((b for b in self.spares if b.cache_key == key), None)
                    self.spares = [b for b in self.spares if b.cache_key != key]  # may raise (setter injection)
                if backend is not None:
                    backend.cache_key = None
                return backend

    registry = OwnerSceneRegistry()
    registry.ensure_loaded("a", _tiny_model)
    cache = _Cache()
    table = _WorkerSceneTable(_ResidentSceneService(cache))
    push_q: queue.Queue = queue.Queue()
    ack_q: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=_scene_attach_listener,
        args=(table, lambda _sid, _path, _ipc: _Backend(), push_q, ack_q),
        daemon=True,
    )
    thread.start()
    channel = SceneAttachChannel(push_q, ack_q, registry.registry, is_worker_alive=lambda: True)
    try:
        channel.attach("a", "/art", ack_timeout_s=2.0)
        cache.fail_spares_update = True

        with pytest.raises(RuntimeError, match="still-live worker"):
            channel.detach("a", ack_timeout_s=2.0)

        assert "a" in table and channel.is_attached("a")
        assert registry.registry.attached_count("a") == 1

        channel.detach("a", ack_timeout_s=2.0)

        assert "a" not in table and not cache.in_use and not cache.spares
        assert registry.registry.attached_count("a") == 0
        assert registry.release("a") is True
    finally:
        channel.stop()
        thread.join(timeout=2.0)


def test_worker_scene_table_refuses_detach_of_in_use_backend_until_render_completes() -> None:
    # F001 (blocker): the mandatory deterministic reproduction of the eviction-vs-in-flight-render race.
    #
    # Scene A is attached on the worker (a spare in the REAL BackendCache), then a render checks the backend
    # OUT (in_use) — modelling a render that runs to completion on the worker AFTER the proxy already released
    # the coordinator lease on the client's expired deadline. The coordinator, seeing A idle, drives a detach
    # of A (exactly as eviction of A to admit B would). Because the worker backend is still in use, that
    # detach MUST be refused: the backend is NOT unlinked and the owner refcount is NOT dropped (either would
    # cudaFree the IPC-aliased tensors the in-flight render still holds — the use-after-free), and the render's
    # later checkin MUST still succeed. The refusal surfaces as the existing typed, retryable "still-live
    # worker" detach outcome; only after the render checks the backend back in (spare) does a retry succeed.
    from nurec.grpc.serve import Backend, BackendCache

    def _new_backend() -> Backend:
        return Backend(
            renderable_model=None,  # type: ignore[arg-type]
            camera_bank=None,  # type: ignore[arg-type]
            world_to_nurec=None,  # type: ignore[arg-type]
            lidar_bank=None,  # type: ignore[arg-type]
            asset_bank=None,  # type: ignore[arg-type]
            source_artifact=None,  # type: ignore[arg-type]
            backend_cache_key=BackendCacheKey(SceneCacheKey(scene_id="a"), "0" * 64),
        )

    registry = OwnerSceneRegistry()
    registry.ensure_loaded("a", _tiny_model)  # owner holds the loadable + its refcount
    cache = BackendCache(maxsize=8)
    table = _WorkerSceneTable(_ResidentSceneService(cache))
    push_q: queue.Queue = queue.Queue()
    ack_q: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=_scene_attach_listener,
        args=(table, lambda _sid, _path, _ipc: _new_backend(), push_q, ack_q),
        daemon=True,
    )
    thread.start()
    channel = SceneAttachChannel(push_q, ack_q, registry.registry, is_worker_alive=lambda: True)
    try:
        channel.attach("a", "/art", ack_timeout_s=2.0)
        assert registry.registry.attached_count("a") == 1

        # A render checks the scene's backend out on the worker: it is now in_use (in flight).
        # The table publishes under `backend_cache_key.key_hash`, so a render resolves by that key.
        resident_key = BackendCacheKey(SceneCacheKey(scene_id="a"), "0" * 64).key_hash
        in_flight = cache.checkout(resident_key)
        assert in_flight is not None and in_flight in cache.in_use

        # Eviction of A drives a detach while the render is still in flight -> refused, typed-and-retryable.
        with pytest.raises(RuntimeError, match="still-live worker"):
            channel.detach("a", ack_timeout_s=2.0)

        assert "a" in table  # scene table record retained
        assert in_flight in cache.in_use  # backend NOT removed while in use (no use-after-free)
        assert in_flight.cache_key == resident_key  # still keyed
        assert channel.is_attached("a")  # attachment retained
        assert registry.registry.attached_count("a") == 1  # owner refcount NOT dropped

        # The in-flight render finishes and checks its backend back in — must succeed (would trip the
        # `backend in self.in_use` assert inside checkin() if the detach had wrongly removed it).
        cache.checkin(in_flight)
        assert in_flight not in cache.in_use

        # Eviction of A now succeeds (the backend is a spare) and the owner releases it.
        channel.detach("a", ack_timeout_s=2.0)
        assert "a" not in table and not cache.in_use and not cache.spares
        assert registry.registry.attached_count("a") == 0
        assert registry.release("a") is True
    finally:
        channel.stop()
        thread.join(timeout=2.0)


def test_scene_attach_listener_releases_backend_and_ipc_frame_locals_after_detach() -> None:
    # F002 (high): after ATTACH(A) then DETACH(A) with NO intervening attach, the listener frame must hold no
    # strong ref to the detached backend or its owner-IPC tensors, so gc reclaims both and the owner's
    # proven-detach release can actually free the VRAM. Weakref-track both and assert they die post-detach.
    class _Sentinel:
        """A weakref-trackable stand-in (plain dicts/tuples are not weakref-trackable)."""

        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.cache_key = None  # a real backend carries this; harmless here

    # Hold our strong refs in a list so we can drop them all with `.clear()`. `build_backend` closes over the
    # list (not the sentinels), so clearing it removes every test-side reference to the sentinels.
    held = [_Sentinel("backend"), _Sentinel("ipc")]
    backend_ref = weakref.ref(held[0])
    ipc_ref = weakref.ref(held[1])

    push_q, ack_q, attached, thread = _start_scene_listener(lambda _sid, _path, _ipc: held[0])
    try:
        push_q.put((_SCENE_ATTACH, 1, "a", "/art", held[1]))
        assert ack_q.get(timeout=2.0) == (1, "ok", _SCENE_ATTACH)
        assert "a" in attached

        push_q.put((_SCENE_DETACH, 2, "a"))
        assert ack_q.get(timeout=2.0) == (2, "ok", _SCENE_DETACH)
        assert "a" not in attached  # the worker mapping is dropped

        # Drop our strong refs. The listener frame — its cleared attach-branch locals, and the current
        # `message` = the DETACH tuple (which references neither) — is then the only place that could pin them.
        held.clear()
        gc.collect()
        assert backend_ref() is None  # backend not pinned by the listener frame after detach
        assert ipc_ref() is None  # owner-IPC tensors not pinned by the listener frame after detach
    finally:
        push_q.put((_SCENE_STOP,))
        thread.join(timeout=2.0)


# ---------------------------------------------------------------------------------------------------
# SceneCoordinator — first-load + all-worker attach barrier + partial-attach rollback.
# ---------------------------------------------------------------------------------------------------


def _coordinator_with_workers(builders, scene_artifacts, *, attached_scenes=(), loader=None, max_scenes=0, pinned=()):
    """An OwnerSceneRegistry + a SceneCoordinator over N worker listeners (one per builder)."""
    osr = OwnerSceneRegistry()
    channels = []
    tables = []
    stops = []
    for builder in builders:
        push_q, ack_q, attached, thread = _start_scene_listener(builder)
        channels.append(SceneAttachChannel(push_q, ack_q, osr.registry))
        tables.append(attached)
        stops.append((push_q, thread))
    coord = SceneCoordinator(
        osr,
        channels,
        scene_artifacts,
        loader or (lambda scene_id, artifact_path: _tiny_model()),
        attached_scenes=attached_scenes,
        max_scenes=max_scenes,
        pinned=pinned,
    )
    return osr, coord, channels, tables, stops


def _stop_workers(stops) -> None:
    for push_q, thread in stops:
        push_q.put((_SCENE_STOP,))
        thread.join(timeout=2.0)


def test_scene_coordinator_attaches_on_all_workers_and_is_idempotent() -> None:
    builders = [
        (lambda scene_id, artifact_path, ipc_tensors: f"w0::{scene_id}"),
        (lambda scene_id, artifact_path, ipc_tensors: f"w1::{scene_id}"),
    ]
    loads: list[str] = []

    def loader(scene_id, artifact_path):
        loads.append(scene_id)
        return _tiny_model()

    osr, coord, channels, tables, stops = _coordinator_with_workers(builders, {"b": "/art-b"}, loader=loader)
    try:
        coord.ensure_attached("b")
        coord.ensure_attached("b")  # idempotent: no second load, no re-barrier

        assert loads == ["b"]  # owner loaded the scene exactly once
        assert tables[0] == {"b": "w0::b"} and tables[1] == {"b": "w1::b"}  # attached on every worker
        assert coord.attached_scenes() == {"b"}
        assert all(channel.attached_scenes() == {"b"} for channel in channels)
        assert osr.registry.attached_count("b") == 2  # one refcount per worker
    finally:
        _stop_workers(stops)


def test_worker_scene_table_headroom_keeps_coordinator_attached_scenes() -> None:
    """B2: the worker BackendCache must not be a second eviction authority over coordinator-attached scenes.

    A coordinated attach registers a scene via ``_WorkerSceneTable.__setitem__`` = ``BackendCache.put`` then
    ``checkin``, so the new backend is briefly *in-use* while the previously-attached scenes are spares — the
    worker momentarily holds ``max_scenes + 1`` backends. ``start_worker_pool`` now sizes the worker cache to
    ``max_scenes + 1`` so this in-flight moment cannot trip the worker's own count-LRU and evict a scene the
    coordinator still treats as attached (and which the worker cannot rebuild, as it is not in ``artifacts``).
    With the +1 headroom every coordinator-managed scene stays resident; without it (``maxsize ==
    max_scenes``) the oldest spare is silently dropped — the uncoordinated eviction the fix closes.
    """
    from nurec.grpc.serve import Backend, BackendCache

    def _key(scene_id: str) -> BackendCacheKey:
        return BackendCacheKey(SceneCacheKey(scene_id=scene_id), "0" * 64)

    def _new_backend(scene_id: str) -> Backend:
        # One distinct key per scene: the table now publishes under `backend_cache_key.key_hash`, so
        # sharing a key across scenes would collide them into a single cache entry.
        return Backend(
            renderable_model=None,  # type: ignore[arg-type]
            camera_bank=None,  # type: ignore[arg-type]
            world_to_nurec=None,  # type: ignore[arg-type]
            lidar_bank=None,  # type: ignore[arg-type]
            asset_bank=None,  # type: ignore[arg-type]
            source_artifact=None,  # type: ignore[arg-type]
            backend_cache_key=_key(scene_id),
        )

    def _resident_keys(cache: BackendCache) -> set:
        return {backend.cache_key for backend in cache.spares}

    max_scenes = 3
    scene_ids = [f"s{i}" for i in range(max_scenes + 1)]
    resident_keys = {_key(sid).key_hash for sid in scene_ids}

    # Fixed sizing (max_scenes + 1, what start_worker_pool now applies): the in-flight put->checkin of the
    # (N+1)th scene evicts nothing — every coordinator-attached scene stays resident on the worker.
    cache = BackendCache(maxsize=max_scenes + 1)
    service = _ResidentSceneService(cache)
    table = _WorkerSceneTable(service)
    for sid in scene_ids:
        table[sid] = _new_backend(sid)
    assert _resident_keys(cache) == resident_keys
    assert all(cache.checkout(_key(sid).key_hash) is not None for sid in scene_ids)
    # every resident scene is also registered, so `get_backend` resolves it instead of NOT_FOUND
    assert set(service.registered) == {_key(sid).scene_key.key_hash for sid in scene_ids}

    # Undersized cache (e.g. `max_scenes == 0` sizes the worker floor at 1 while coordinator admission is
    # unbounded): `__setitem__` grows the capacity in step with the attach stream, so the worker LRU can
    # never evict a coordinator-attached scene regardless of the initial sizing — every scene stays
    # resident and checkout-able, and the pool's loaded-scenes report stays truthful on every worker.
    undersized = BackendCache(maxsize=max_scenes)
    undersized_table = _WorkerSceneTable(_ResidentSceneService(undersized))
    for sid in scene_ids:
        undersized_table[sid] = _new_backend(sid)
    assert undersized.maxsize == len(scene_ids) + 1  # grown in step: table size + in-flight headroom
    assert _resident_keys(undersized) == resident_keys
    assert all(undersized.checkout(_key(sid).key_hash) is not None for sid in scene_ids)


def test_owner_bookkeeping_stays_bounded_across_evict_and_reregister() -> None:
    # The per-scene *locks* (the coordinator's `_scene_locks` and the owner registry's `_scene_locks`)
    # must not grow without bound as distinct on-demand scenes churn through attach→evict; they are pruned on
    # eviction and recreated on next access. (: the scene *registration* `_scene_artifacts` now SURVIVES
    # GPU eviction — it is bounded instead by the on-disk download cache via `invalidate_scene`, exercised
    # below and in the SceneCache tests.)
    builders = [lambda scene_id, artifact_path, ipc_tensors: f"w0::{scene_id}"]
    osr, coord, _channels, _tables, stops = _coordinator_with_workers(builders, {}, max_scenes=1)
    try:
        for i in range(12):
            sid = f"s{i}"
            coord.register_scene(sid, f"/art-{i}", source=f"http://ex/{i}")
            coord.ensure_attached(sid)  # evicts the LRU victim (the previous scene); prunes its lock
        assert coord.attached_scenes() == {"s11"}  # only the last scene is resident
        # Locks are bounded by the resident set, not the 12 scenes ever attached.
        assert len(coord._scene_locks) <= 2  # noqa: SLF001
        assert len(osr._scene_locks) <= 2  # noqa: SLF001 - owner registry locks pruned via forget_scene_lock
        # Registrations survive GPU eviction: every on-demand scene stays ID-only reloadable.
        assert all(f"s{i}" in coord._scene_artifacts for i in range(12))  # noqa: SLF001
        # A disk-cache eviction (invalidate_scene) drops a non-resident on-demand registration -> NOT_FOUND.
        coord.invalidate_scene("s3")
        assert "s3" not in coord._scene_artifacts  # noqa: SLF001
        assert "s3" not in coord._scene_sources  # noqa: SLF001
    finally:
        _stop_workers(stops)


def test_owner_bookkeeping_keeps_permanent_scene_artifacts() -> None:
    # a scene's artifact registration survives GPU eviction — a permanent (glob/bootstrap) scene stays
    # re-attachable with no header, AND an on-demand scene stays ID-only reloadable while its backing
    # artifact is usable. Only a disk-cache invalidation drops an on-demand registration.
    builders = [lambda scene_id, artifact_path, ipc_tensors: f"w0::{scene_id}"]
    # "perm" is registered at startup (permanent); "od" arrives on-demand.
    _osr, coord, _channels, _tables, stops = _coordinator_with_workers(builders, {"perm": "/art-perm"}, max_scenes=1)
    try:
        coord.ensure_attached("perm")
        coord.register_scene("od", "/art-od", source="http://ex/od")
        coord.ensure_attached("od")  # evicts "perm"
        assert coord.attached_scenes() == {"od"}
        assert "perm" in coord._scene_artifacts  # noqa: SLF001 - permanent path kept (re-attachable, no header)
        assert "perm" not in coord._scene_locks  # noqa: SLF001 - the per-scene lock was still pruned (recreatable)
        coord.ensure_attached("perm")  # re-attaches with no re-registration -> evicts "od"
        assert coord.attached_scenes() == {"perm"}
        assert "od" in coord._scene_artifacts  # noqa: SLF001 - on-demand registration survives GPU eviction
        # A permanent scene's registration is never dropped by a disk-cache invalidation.
        coord.invalidate_scene("perm")
        assert "perm" in coord._scene_artifacts  # noqa: SLF001
        # The evicted on-demand scene IS dropped by a disk-cache invalidation (it is not resident now).
        coord.invalidate_scene("od")
        assert "od" not in coord._scene_artifacts  # noqa: SLF001
    finally:
        _stop_workers(stops)


def test_prune_evicted_bookkeeping_respects_active_attach() -> None:
    # Gating invariant: the per-scene bookkeeping prune must NOT drop a scene's lock while an attach is
    # active for it (a waiting re-attach holds that lock — pruning it would let a stale and a fresh lock both
    # admit the scene). It prunes only once quiescent. This locks in the gate `_prune` relies on.
    builders = [lambda scene_id, artifact_path, ipc_tensors: f"w0::{scene_id}"]
    _osr, coord, _channels, _tables, stops = _coordinator_with_workers(builders, {}, max_scenes=2)
    try:
        coord.register_scene("s", "/art-s")  # on-demand registration
        coord._scene_lock("s")  # noqa: SLF001 - materialize the per-scene lock entry
        with coord._lock:  # noqa: SLF001
            coord._attach_active["s"] = 1  # noqa: SLF001 - pretend a re-attach holds the lock
            coord._prune_evicted_bookkeeping_locked("s")  # noqa: SLF001
            assert "s" in coord._scene_locks  # noqa: SLF001 - NOT pruned while an attach is active
            assert "s" in coord._scene_artifacts  # noqa: SLF001
            coord._attach_active.pop("s")  # noqa: SLF001 - now quiescent
            coord._prune_evicted_bookkeeping_locked("s")  # noqa: SLF001
            assert "s" not in coord._scene_locks  # noqa: SLF001 - lock pruned once quiescent
            # the registration survives GPU eviction (no pending disk-cache invalidation), so it stays
            # ID-only reloadable; a disk-cache invalidation is what would drop it.
            assert "s" in coord._scene_artifacts  # noqa: SLF001
    finally:
        _stop_workers(stops)


def test_eviction_prune_does_not_drop_lock_held_by_concurrent_attach() -> None:
    # Placement regression (deterministic interleaving): the gate (`_attach_active`) must be set BEFORE
    # the per-scene lock is acquired, so an eviction's prune can never pop a lock a concurrent attach holds.
    # We instrument the per-scene lock to pause a real `ensure_attached` thread the instant it holds the
    # lock, then snapshot the gate and run the prune:
    # * current placement (increment BEFORE acquire): gate == 1 at acquire → prune skips → lock kept.
    # * old buggy placement (increment AFTER acquire): gate == 0 in this window → prune would pop the lock
    # the attach thread holds (the double-attach race). Both asserts below would then fail.
    builders = [lambda scene_id, artifact_path, ipc_tensors: f"w0::{scene_id}"]
    _osr, coord, _channels, _tables, stops = _coordinator_with_workers(builders, {}, max_scenes=2)

    lock_held = threading.Event()
    release = threading.Event()
    gate_at_acquire: dict[str, int] = {}

    class _GatedLock:
        """Per-scene lock that, the first time it is entered, snapshots the prune gate and blocks until the
        test releases it — pinning the attach thread at 'lock acquired, ensure_attached not yet proceeded'."""

        def __init__(self) -> None:
            self._l = threading.Lock()
            self._fired = False

        def __enter__(self):
            self._l.acquire()
            if not self._fired:
                self._fired = True
                gate_at_acquire["count"] = coord._attach_active.get("s", 0)  # noqa: SLF001
                lock_held.set()
                release.wait(timeout=5.0)
            return self

        def __exit__(self, *exc) -> None:
            self._l.release()

    coord.register_scene("s", "/art-s")
    coord._scene_locks["s"] = _GatedLock()  # noqa: SLF001 - instrument the per-scene lock

    attach_thread = threading.Thread(target=lambda: coord.ensure_attached("s"), daemon=True)
    try:
        attach_thread.start()
        assert lock_held.wait(timeout=2.0)  # attach thread holds the per-scene lock, paused before proceeding

        # The gate was set before the lock was taken (current placement); the old placement would read 0 here.
        assert gate_at_acquire["count"] >= 1
        # ...so the eviction's gated prune must NOT drop the lock the attach thread is holding.
        with coord._lock:  # noqa: SLF001
            coord._prune_evicted_bookkeeping_locked("s")  # noqa: SLF001
            assert "s" in coord._scene_locks  # noqa: SLF001 - lock kept (old placement would have popped it)
    finally:
        release.set()
        attach_thread.join(timeout=5.0)
        _stop_workers(stops)
    assert not attach_thread.is_alive()


def test_render_proxy_close_closes_worker_channels() -> None:
    # L4: the proxy holds a worker channel per worker for its lifetime; close() must close them (fds + bg
    # gRPC threads) and be idempotent, so a long-lived embedder calling serve_proxy repeatedly doesn't leak.
    closed: list = []

    class _Chan:
        def close(self) -> None:
            closed.append(self)

    channels = [_Chan(), _Chan()]
    proxy = RenderProxy([_RecordingStub(i, [], threading.Lock()) for i in range(len(channels))], channels=channels)

    proxy.close()
    assert len(closed) == 2  # every worker channel closed
    proxy.close()  # idempotent: no second close, no error
    assert len(closed) == 2


def test_await_seq_ack_discards_future_and_stale_acks() -> None:
    # L3: each channel has at most one outstanding command, so an ack for a LATER seq than the one awaited
    # violates that invariant — it must be discarded (and surfaced), not mistaken for our ack. A stale
    # (earlier) ack from a previously timed-out command is likewise dropped. In both cases a correct ack
    # arriving after is still returned.
    from nurec.grpc.multiproc_serve import _await_seq_ack

    future_first: queue.Queue = queue.Queue()
    future_first.put((5, "ok", "future"))  # unexpected future ack (5 > 3)
    future_first.put((3, "ok", "expected"))
    assert _await_seq_ack(future_first, seq=3, timeout_s=1.0) == ("ok", "expected")

    stale_first: queue.Queue = queue.Queue()
    stale_first.put((2, "error", "stale"))  # late ack from a previously timed-out command (2 < 3)
    stale_first.put((3, "ok", "expected"))
    assert _await_seq_ack(stale_first, seq=3, timeout_s=1.0) == ("ok", "expected")


class _Pool:
    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def all_workers_alive(self) -> bool:
        return self._alive


def test_await_pool_shutdown_reports_typed_terminal_reason() -> None:
    # / : the teardown decision is a TYPED reason (not a bare event), so a degradation is never
    # mistaken for a normal shutdown. A dead worker -> WORKER_EXIT; an explicit shut_down -> REQUESTED; a
    # degraded-recorded stop -> SCENE_POOL_DEGRADED. Only REQUESTED exits 0.
    dead = _Pool(alive=False)
    assert _await_pool_shutdown(dead, PoolShutdown(), poll_timeout_s=0.01) == PoolTerminalReason.WORKER_EXIT  # type: ignore[arg-type]

    requested = PoolShutdown()
    requested.set()  # the client shut_down RPC path (default reason)
    live = _Pool(alive=True)
    assert _await_pool_shutdown(live, requested, poll_timeout_s=0.01) == PoolTerminalReason.REQUESTED  # type: ignore[arg-type]

    degraded = PoolShutdown()
    degraded.set(PoolTerminalReason.SCENE_POOL_DEGRADED)  # the scene-pool degraded callback path
    assert _await_pool_shutdown(live, degraded, poll_timeout_s=0.01) == PoolTerminalReason.SCENE_POOL_DEGRADED  # type: ignore[arg-type]

    # A bare threading.Event (no recorded reason) that is set counts as a requested stop.
    bare = threading.Event()
    bare.set()
    assert _await_pool_shutdown(_Pool(alive=True), bare, poll_timeout_s=0.01) == PoolTerminalReason.REQUESTED  # type: ignore[arg-type]


def test_pool_shutdown_records_first_reason_and_maps_exit_code() -> None:
    # First reason wins: a later requested stop cannot overwrite an earlier SCENE_POOL_DEGRADED, so a degraded
    # fleet always exits nonzero even if a shut_down races in afterward.
    shutdown = PoolShutdown()
    shutdown.set(PoolTerminalReason.SCENE_POOL_DEGRADED)
    shutdown.set(PoolTerminalReason.REQUESTED)  # loses (first wins)
    assert shutdown.reason is PoolTerminalReason.SCENE_POOL_DEGRADED
    assert shutdown.is_set()

    # Exit-code mapping: only a requested stop is clean (0); worker death / degradation are nonzero.
    assert PoolTerminalReason.REQUESTED.exit_code == 0 and PoolTerminalReason.REQUESTED.is_clean
    assert PoolTerminalReason.WORKER_EXIT.exit_code != 0 and not PoolTerminalReason.WORKER_EXIT.is_clean
    assert PoolTerminalReason.SCENE_POOL_DEGRADED.exit_code != 0 and not PoolTerminalReason.SCENE_POOL_DEGRADED.is_clean


def test_proxy_health_transition_updates_aggregate_and_concrete_service() -> None:
    # Scene-pool degradation transitions health before the proxy drains. Named-service probes must not remain
    # SERVING after the aggregate empty-name probe is NOT_SERVING, or they can keep routing to a mixed fleet.
    calls: list[tuple[str, int]] = []

    class _HealthServicer:
        def set(self, service_name: str, status: int) -> None:
            calls.append((service_name, status))

    servicer = _HealthServicer()
    _mp._set_proxy_health_status(servicer, _mp.health_pb2.HealthCheckResponse.SERVING)  # noqa: SLF001
    _mp._set_proxy_health_status(servicer, _mp.health_pb2.HealthCheckResponse.NOT_SERVING)  # noqa: SLF001

    names = ["", "nurec.grpc.protos.sensorsim.SensorsimService"]
    assert calls == [
        *((name, _mp.health_pb2.HealthCheckResponse.SERVING) for name in names),
        *((name, _mp.health_pb2.HealthCheckResponse.NOT_SERVING) for name in names),
    ]


class _ReconfigurableStringIO(io.StringIO):
    """A stdout stand-in that tolerates `sys.stdout.reconfigure(line_buffering=True)` (a plain StringIO or
    pytest's capture object may not), so the launch entrypoints can be exercised under test."""

    def reconfigure(self, *args: object, **kwargs: object) -> None:
        pass


class _StubPool:
    """Records `shutdown()` so an entrypoint test can prove the fleet is always torn down."""

    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_main_maps_terminal_reason_to_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # / : the standalone launch path returns NONZERO on worker death / scene-pool degradation (so a
    # supervisor restarts the fleet) and 0 on a requested shutdown — and always tears the pool down.
    artifact = tmp_path / "scene.usdz"
    artifact.write_text("stub")
    fake_pool = _StubPool()
    monkeypatch.setattr(_mp, "start_worker_pool", lambda *a, **k: fake_pool)
    monkeypatch.setattr(sys, "argv", ["multiproc_serve", "--artifact", str(artifact), "--workers", "1"])
    monkeypatch.setattr(sys, "stdout", _ReconfigurableStringIO())

    monkeypatch.setattr(_mp, "serve_proxy", lambda *a, **k: PoolTerminalReason.SCENE_POOL_DEGRADED)
    assert _mp.main() == 1  # degradation → nonzero
    assert fake_pool.shutdown_calls == 1  # fleet torn down

    monkeypatch.setattr(_mp, "serve_proxy", lambda *a, **k: PoolTerminalReason.WORKER_EXIT)
    assert _mp.main() == 1  # worker death → nonzero

    monkeypatch.setattr(_mp, "serve_proxy", lambda *a, **k: PoolTerminalReason.REQUESTED)
    assert _mp.main() == 0  # client shut_down → zero


def test_main_keyboard_interrupt_exits_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A SIGINT after drain (KeyboardInterrupt out of serve_proxy) is a clean stop → exit 0, pool torn down.
    artifact = tmp_path / "scene.usdz"
    artifact.write_text("stub")
    fake_pool = _StubPool()
    monkeypatch.setattr(_mp, "start_worker_pool", lambda *a, **k: fake_pool)
    monkeypatch.setattr(sys, "argv", ["multiproc_serve", "--artifact", str(artifact), "--workers", "1"])
    monkeypatch.setattr(sys, "stdout", _ReconfigurableStringIO())

    def _interrupt(*_a: object, **_k: object) -> PoolTerminalReason:
        raise KeyboardInterrupt

    monkeypatch.setattr(_mp, "serve_proxy", _interrupt)
    assert _mp.main() == 0
    assert fake_pool.shutdown_calls == 1


def test_main_artifact_help_is_file_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # standalone --artifact accepts exactly one existing .usdz file (main() rejects anything that is not
    # is_file()); no glob expansion is implemented here. Its help must therefore describe a single file and must
    # NOT claim glob support — multi-scene glob discovery belongs only to the serve-grpc drop-in (--artifact-glob).
    monkeypatch.setattr(sys, "argv", ["multiproc_serve", "--help"])
    captured = _ReconfigurableStringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    with pytest.raises(SystemExit) as exc_info:
        _mp.main()  # argparse prints --help to stdout and exits 0 before any launch work
    assert exc_info.value.code == 0

    # Collapse argparse line-wrapping, then isolate the --artifact entry (listed before --workers in `options:`).
    normalized = " ".join(captured.getvalue().split())
    options = normalized[normalized.index("options:") :]
    artifact_start = options.index("--artifact ARTIFACT")
    artifact_help = options[artifact_start : options.index("--workers", artifact_start)]

    assert ".usdz" in artifact_help
    assert "file" in artifact_help.lower()
    assert "or glob" not in artifact_help.lower()
    assert "glob" not in artifact_help.lower()  # strictly file-only: no glob claim of any kind


def test_serve_grpc_dropin_still_exposes_artifact_glob() -> None:
    # cross-mode guard: making the standalone --artifact help file-only must NOT hide the drop-in's public
    # glob/multi-scene discovery. serve-grpc keeps exposing --artifact-glob, so the file-only standalone help is a
    # truthful mode distinction rather than a silent capability regression.
    from click.testing import CliRunner

    from nurec.grpc.serve import serve_grpc

    result = CliRunner().invoke(serve_grpc, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--artifact-glob" in result.output


def test_run_drop_in_maps_terminal_reason_to_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    # / : the drop-in (serve-grpc --multiproc-workers) launch path returns the same typed exit
    # code, so the Click entrypoint can `ctx.exit(code)` and a supervisor restarts on degradation/worker death.
    from nurec.grpc.grpc_server_config import GrpcServerConfig

    fake_pool = _StubPool()
    monkeypatch.setattr(_mp, "_plan_glob_scenes", lambda matches: (Path("boot.usdz"), "boot", {}))
    monkeypatch.setattr(_mp, "mps_preflight", lambda *a, **k: None)
    monkeypatch.setattr(_mp, "_resolve_download_cache_size", lambda *a, **k: 1)
    monkeypatch.setattr(_mp, "SceneCache", lambda *a, **k: object())
    monkeypatch.setattr(_mp, "Downloader", lambda *a, **k: object())
    monkeypatch.setattr(_mp, "start_worker_pool", lambda *a, **k: fake_pool)
    monkeypatch.setattr(_mp.SensorSimService, "build_server_config_map", staticmethod(lambda cfg: {}))
    monkeypatch.setattr(sys, "stdout", _ReconfigurableStringIO())
    cfg = GrpcServerConfig()

    monkeypatch.setattr(_mp, "serve_proxy", lambda *a, **k: PoolTerminalReason.SCENE_POOL_DEGRADED)
    assert _mp.run_drop_in(cfg, num_workers=2) == 1
    assert fake_pool.shutdown_calls == 1

    monkeypatch.setattr(_mp, "serve_proxy", lambda *a, **k: PoolTerminalReason.WORKER_EXIT)
    assert _mp.run_drop_in(cfg, num_workers=2) == 1

    monkeypatch.setattr(_mp, "serve_proxy", lambda *a, **k: PoolTerminalReason.REQUESTED)
    assert _mp.run_drop_in(cfg, num_workers=2) == 0


def test_run_drop_in_validation_failure_tears_down_before_proxy_serves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail-fast startup validation owns teardown and never exposes a partially validated proxy."""
    from nurec.grpc.grpc_server_config import GrpcServerConfig

    class _FailingValidationCoordinator:
        def handle_cache_eviction(self, eviction: object) -> None:
            pass

        def ensure_attached(self, scene_id: str) -> None:
            raise SceneLoadError(f"scene {scene_id!r} failed validation")

    class _ValidationPool(_StubPool):
        def __init__(self) -> None:
            super().__init__()
            self.coordinator = _FailingValidationCoordinator()

    pool = _ValidationPool()
    serve_calls = 0

    def unexpected_serve(*args: object, **kwargs: object) -> PoolTerminalReason:
        nonlocal serve_calls
        serve_calls += 1
        return PoolTerminalReason.REQUESTED

    monkeypatch.setattr(_mp, "_plan_glob_scenes", lambda matches: (Path("boot.usdz"), "boot", {"bad": "bad.usdz"}))
    monkeypatch.setattr(_mp, "mps_preflight", lambda *a, **k: None)
    monkeypatch.setattr(_mp, "start_worker_pool", lambda *a, **k: pool)
    monkeypatch.setattr(_mp, "serve_proxy", unexpected_serve)
    monkeypatch.setattr(sys, "stdout", _ReconfigurableStringIO())
    config = GrpcServerConfig(
        test_scenes_are_valid=True,
        download_cache_dir=str(tmp_path / "cache"),
    )

    with pytest.raises(SceneLoadError, match="failed validation"):
        _mp.run_drop_in(config, num_workers=2)

    assert pool.shutdown_calls == 1
    assert serve_calls == 0  # proxy never binds/becomes SERVING


def test_serve_grpc_click_entrypoint_propagates_fleet_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    # / : the serve-grpc Click entrypoint must actually EXIT with the fleet's terminal code (via
    # ctx.exit), so run_drop_in returning nonzero (worker death / scene-pool degradation) becomes a nonzero
    # PROCESS exit a supervisor restarts on, while a requested shutdown (0) stays 0.
    from click.testing import CliRunner

    from nurec.grpc.serve import serve_grpc

    # serve_grpc imports run_drop_in lazily from the multiproc_serve module, so patch it there.
    monkeypatch.setattr(_mp, "run_drop_in", lambda *a, **k: 1)
    degraded = CliRunner().invoke(serve_grpc, ["--multiproc-workers", "2"])
    assert degraded.exit_code == 1  # ctx.exit(1) → nonzero process exit

    monkeypatch.setattr(_mp, "run_drop_in", lambda *a, **k: 0)
    requested = CliRunner().invoke(serve_grpc, ["--multiproc-workers", "2"])
    assert requested.exit_code == 0  # clean stop stays 0


class _ReapFakeProc:
    """A worker stand-in for _terminate_and_reap: `dies_on_terminate` models a well-behaved worker;
    otherwise it ignores SIGTERM (stays alive) and only SIGKILL reaps it."""

    def __init__(self, *, dies_on_terminate: bool, alive: bool = True) -> None:
        self._alive = alive
        self._dies_on_terminate = dies_on_terminate
        self.terminated = False
        self.killed = False

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminated = True
        if self._dies_on_terminate:
            self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False  # SIGKILL always reaps

    def join(self, timeout=None) -> None:  # noqa: ANN001
        pass


class _UnreapableFakeProc(_ReapFakeProc):
    """Models the exceptional case where even the kill/reap sequence cannot prove the worker is dead."""

    def __init__(self) -> None:
        super().__init__(dies_on_terminate=False)

    def kill(self) -> None:
        self.killed = True  # remains alive: shutdown must retain owner refs and surface the failure


class _JoinFailingFakeProc(_ReapFakeProc):
    """Dies when terminated but raises from every join, exercising best-effort fleet-wide reap."""

    def __init__(self) -> None:
        super().__init__(dies_on_terminate=True)
        self.join_calls = 0

    def join(self, timeout=None) -> None:  # noqa: ANN001
        self.join_calls += 1
        raise RuntimeError("join boom")


class _TerminateFailingFakeProc(_ReapFakeProc):
    """Cannot be reaped: terminate itself faults, so shutdown must conservatively retain owner refs."""

    def __init__(self) -> None:
        super().__init__(dies_on_terminate=False)

    def terminate(self) -> None:
        self.terminated = True
        raise RuntimeError("terminate boom")


class _GracefulLifecycleProc(_ReapFakeProc):
    """A worker that exits when it is asked to stop over its lifecycle channel (records join order)."""

    def __init__(self, worker_id: int, order: list) -> None:
        super().__init__(dies_on_terminate=True)
        self.worker_id = worker_id
        self._order = order

    def join(self, timeout=None) -> None:  # noqa: ANN001
        self._order.append(("join", self.worker_id))


class _StopThenDieLifecycleTx:
    """Owner lifecycle write end whose STOP makes its (graceful) worker exit; records order.

    ``send`` raises after ``close`` (like a real closed `Connection`) so a repeated shutdown that resent
    through a closed endpoint would fault loudly rather than silently pass."""

    def __init__(self, worker_id: int, proc: _GracefulLifecycleProc, order: list, *, close_failures: int = 0) -> None:
        self.worker_id = worker_id
        self._proc = proc
        self._order = order
        self._close_failures = close_failures
        self.sent: list = []
        self.closed = False
        self.close_calls = 0

    def send(self, obj: object) -> None:
        if self.closed:
            raise OSError("send on a closed lifecycle channel")
        self._order.append(("stop", self.worker_id))
        self.sent.append(obj)
        self._proc._alive = False  # noqa: SLF001 - a graceful worker exits on the stop request

    def close(self) -> None:
        self._order.append(("lifecycle_close", self.worker_id))
        self.close_calls += 1
        if self._close_failures:
            self._close_failures -= 1
            raise OSError("lifecycle close boom")
        self.closed = True


class _OrderingSceneChannel:
    """Minimal scene-attach channel stand-in that records its stop() order and no-ops post-reap release."""

    def __init__(self, index: int, order: list) -> None:
        self._index = index
        self._order = order

    def stop(self) -> None:
        self._order.append(("channel_stop", self._index))

    def release_after_reap(self) -> "tuple[list, list]":
        self._order.append(("channel_release", self._index))
        return [], []


def test_worker_pool_shutdown_stops_listeners_before_lifecycle_stop_then_joins() -> None:
    # Shutdown ordering contract: the scene listeners are stopped (while workers are still alive to drain
    # their stop sentinels) BEFORE the lifecycle STOP that ends each worker's serve loop, and every lifecycle
    # STOP precedes the bounded join. A graceful worker exits on the STOP, so no SIGTERM/SIGKILL escalation is
    # needed and the owner closes the write ends after reap.
    order: list = []
    procs = [_GracefulLifecycleProc(0, order), _GracefulLifecycleProc(1, order)]
    txs = [_StopThenDieLifecycleTx(i, procs[i], order) for i in range(2)]
    pool = WorkerPool(
        scene_id="s",
        worker_addresses=[],
        owner_backend=None,
        scene_channels=[_OrderingSceneChannel(0, order), _OrderingSceneChannel(1, order)],
        _procs=procs,
        _lifecycle_txs=list(txs),  # pool owns/mutates its own list (drops closed endpoints to None)
    )
    pool.defer_close(lambda: order.append(("deferred_close", -1)))

    pool.shutdown()

    for tx in txs:
        assert tx.sent == [WorkerLifecycleCommand.STOP]  # exactly one owner-directed stop per worker
        assert tx.closed  # owner write end closed post-reap
    kinds = [ev[0] for ev in order]
    listener_idxs = [i for i, k in enumerate(kinds) if k == "channel_stop"]
    stop_idxs = [i for i, k in enumerate(kinds) if k == "stop"]
    join_idxs = [i for i, k in enumerate(kinds) if k == "join"]
    lifecycle_close_idxs = [i for i, k in enumerate(kinds) if k == "lifecycle_close"]
    owner_release_idxs = [i for i, k in enumerate(kinds) if k == "channel_release"]
    deferred_close_idxs = [i for i, k in enumerate(kinds) if k == "deferred_close"]
    assert len(listener_idxs) == 2 and len(stop_idxs) == 2 and len(join_idxs) == 2
    assert max(listener_idxs) < min(stop_idxs)  # all listeners stopped before ANY lifecycle STOP
    assert max(stop_idxs) < min(join_idxs)  # all lifecycle STOPs before ANY bounded join
    assert max(join_idxs) < min(lifecycle_close_idxs)  # transmitters close only after every worker is reaped
    assert max(lifecycle_close_idxs) < min(owner_release_idxs)  # lifecycle endpoints precede owner refs/queues
    assert max(owner_release_idxs) < min(deferred_close_idxs)  # deferred proxy channels are the final phase
    # Graceful exit on the request: reaped without SIGTERM/SIGKILL escalation.
    assert all(not proc.is_alive() and not proc.terminated and not proc.killed for proc in procs)


def test_worker_pool_shutdown_is_idempotent_on_repeat() -> None:
    # A repeated shutdown must not resend a STOP through (nor re-close) an already-closed lifecycle endpoint:
    # the first pass drops each closed endpoint to None, and the reaped workers are provably dead.
    order: list = []
    procs = [_GracefulLifecycleProc(0, order)]
    txs = [_StopThenDieLifecycleTx(0, procs[0], order)]
    pool = WorkerPool(
        scene_id="s",
        worker_addresses=[],
        owner_backend=None,
        _procs=procs,
        _lifecycle_txs=list(txs),  # pool owns/mutates its own list (drops closed endpoints to None)
    )

    pool.shutdown()
    assert txs[0].sent == [WorkerLifecycleCommand.STOP]
    assert txs[0].closed

    pool.shutdown()  # must not raise, resend, or re-close (endpoint already dropped to None)
    assert txs[0].sent == [WorkerLifecycleCommand.STOP]  # unchanged: no resend through the closed endpoint


def test_worker_pool_shutdown_retries_failed_post_reap_close_without_resending_stop() -> None:
    # The first pass reaps the worker but its lifecycle close fails. The retained endpoint is retried on the
    # second pass; because the worker is already provably dead, STOP is not resent through that endpoint.
    order: list = []
    proc = _GracefulLifecycleProc(0, order)
    tx = _StopThenDieLifecycleTx(0, proc, order, close_failures=1)
    pool = WorkerPool(
        scene_id="s",
        worker_addresses=[],
        owner_backend=None,
        _procs=[proc],
        _lifecycle_txs=[tx],
    )

    with pytest.raises(RuntimeError, match="lifecycle close boom"):
        pool.shutdown()

    assert tx.sent == [WorkerLifecycleCommand.STOP]
    assert tx.close_calls == 1 and not tx.closed
    assert pool._lifecycle_txs == [tx]  # noqa: SLF001 - failed endpoint retained for retry

    pool.shutdown()

    assert tx.sent == [WorkerLifecycleCommand.STOP]  # no resend after the post-reap failure
    assert tx.close_calls == 2 and tx.closed
    assert pool._lifecycle_txs == [None]  # noqa: SLF001 - successful retry retires endpoint state


def test_worker_stop_reason_exit_code_mapping() -> None:
    # Only a control-plane failure is a nonzero worker exit; an owner-directed stop and stopping because the
    # owner is already gone are clean (0). The owner maps a nonzero worker death to WORKER_EXIT.
    assert WorkerStopReason.STOP.exit_code == 0
    assert WorkerStopReason.OWNER_DEATH.exit_code == 0
    assert WorkerStopReason.CONTROL_PLANE_FAILURE.exit_code == 1


def test_await_pool_shutdown_maps_worker_death_to_worker_exit() -> None:
    # The owner-side complement of a worker's nonzero control-plane exit: a worker that dies while no shutdown
    # was requested makes the owner loop return WORKER_EXIT (a nonzero fleet exit).
    class _Pool:
        def all_workers_alive(self) -> bool:
            return False  # a worker has died

    reason = _await_pool_shutdown(_Pool(), PoolShutdown(), poll_timeout_s=0.01)  # type: ignore[arg-type]
    assert reason is PoolTerminalReason.WORKER_EXIT
    assert not reason.is_clean and reason.exit_code == 1


# ---------------------------------------------------------------------------
# Real process boundary: parent-sentinel supervision + lifecycle exit codes
# ---------------------------------------------------------------------------


class _AlreadyStoppedServer:
    """A `_serve_until_stop`-compatible server stand-in: `stop()` reports an already-completed stop.

    Lets the process-boundary probes exercise the real lifecycle/sentinel wait + exit-code mapping without a
    gRPC server (gRPC and fork interact poorly; the wait/exit logic is what these tests cover)."""

    def stop(self, _grace: float) -> "threading.Event":
        done = threading.Event()
        done.set()
        return done


def _lifecycle_probe_target(lifecycle_rx: Any, inherited_lifecycle_tx: Any) -> None:
    # Child worker stand-in: derive the real parent (test process) sentinel and run the production supervision
    # wait, then exit with the stop reason's exit code — exactly as `_worker` does after readiness.
    # Fork inherits the parent's transmitter even though production spawn does not; close that test-only copy
    # so closing the real parent transmitter exercises lifecycle EOF correctly.
    inherited_lifecycle_tx.close()
    parent_sentinel, is_parent_alive = _mp._owner_parent_signals()  # noqa: SLF001
    reason = _mp._serve_until_stop(  # noqa: SLF001
        _AlreadyStoppedServer(),
        lifecycle_rx,
        shutdown_grace_s=0.1,
        parent_sentinel=parent_sentinel,
        is_parent_alive=is_parent_alive,
    )
    sys.exit(reason.exit_code)


def _spawn_lifecycle_probe() -> "tuple[Any, Any, Any]":
    # Pytest's Bazel launcher is not safely re-importable as multiprocessing `__main__`; the dedicated
    # `worker_lifecycle_spawn_test` target covers the production spawn boundary. Keep these fast helper/process
    # tests on fork so they exercise the same wait/exit logic without recursively launching pytest.
    ctx = multiprocessing.get_context("fork")
    lifecycle_rx, lifecycle_tx = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_lifecycle_probe_target, args=(lifecycle_rx, lifecycle_tx))
    proc.start()
    lifecycle_rx.close()  # only the child keeps the read end
    return proc, lifecycle_tx, ctx


def test_spawned_worker_exits_zero_on_lifecycle_stop() -> None:
    # Real process boundary: an owner-directed lifecycle STOP over the real pipe makes the worker exit 0.
    proc, lifecycle_tx, _ctx = _spawn_lifecycle_probe()
    try:
        time.sleep(0.4)  # let the child reach the supervision wait
        assert proc.is_alive()  # no traffic, no message, live parent -> still serving
        send_worker_stop(lifecycle_tx)
        proc.join(timeout=10.0)
        assert not proc.is_alive()
        assert proc.exitcode == 0  # owner-directed graceful stop is a clean exit
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)
        lifecycle_tx.close()


def test_spawned_worker_exits_nonzero_on_bad_lifecycle_command() -> None:
    # Real process boundary: a malformed lifecycle payload with a live parent is a control-plane failure, so
    # the worker exits NONZERO (which the owner would observe as WORKER_EXIT).
    proc, lifecycle_tx, _ctx = _spawn_lifecycle_probe()
    try:
        time.sleep(0.4)
        lifecycle_tx.send("not-a-lifecycle-command")  # bad payload, parent (test) still alive
        proc.join(timeout=10.0)
        assert not proc.is_alive()
        assert proc.exitcode == 1  # control-plane failure -> nonzero
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)
        lifecycle_tx.close()


def test_spawned_worker_exits_nonzero_on_lifecycle_eof() -> None:
    # The fork probe closes its test-only inherited write end at entry, matching production spawn ownership.
    # Closing the live parent's transmitter therefore produces a real EOF and a fail-closed worker exit.
    proc, lifecycle_tx, _ctx = _spawn_lifecycle_probe()
    try:
        time.sleep(0.4)
        lifecycle_tx.close()
        proc.join(timeout=10.0)
        assert not proc.is_alive()
        assert proc.exitcode == 1
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)
        with contextlib.suppress(Exception):
            lifecycle_tx.close()


def _no_sentinel_worker_target(port_q: Any, attach_req_q: Any, lifecycle_rx: Any) -> None:
    # Force supervision discovery to report "no parent sentinel", then run the real `_worker`: it must refuse
    # to start (before any attach) and exit nonzero.
    _mp._owner_parent_signals = lambda: (None, None)  # noqa: SLF001
    _mp._worker(  # noqa: SLF001
        0,
        "scene",
        "/nonexistent.usdz",
        WorkerSpawnConfig(renderer_name="gsplat", enable_editing_actors=False),
        attach_req_q,
        object(),
        port_q,
        lifecycle_rx,
        object(),
        object(),
    )


def test_worker_refuses_to_start_without_parent_sentinel() -> None:
    # HIGH: OS owner-liveness supervision is mandatory and checked FIRST. With no parent sentinel the worker
    # reports a startup error, never requests an attach (proving the check precedes attach/load/bind), and
    # exits nonzero instead of degrading to lifecycle-channel-only supervision.
    ctx = multiprocessing.get_context("fork")
    port_q: Any = ctx.Queue()
    attach_req_q: Any = ctx.Queue()
    lifecycle_rx, lifecycle_tx = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_no_sentinel_worker_target, args=(port_q, attach_req_q, lifecycle_rx))
    proc.start()
    lifecycle_rx.close()
    try:
        message = port_q.get(timeout=10.0)
        proc.join(timeout=10.0)
        assert message[0] == "error"
        assert "sentinel" in message[2] or "supervision" in message[2]
        assert proc.exitcode == 1  # refused startup exits nonzero
        assert attach_req_q.empty()  # supervision was established BEFORE any attach was requested
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)
        with contextlib.suppress(Exception):
            lifecycle_tx.close()
        for q in (port_q, attach_req_q):
            with contextlib.suppress(Exception):
                q.close()


def _sigkill_worker_target(lifecycle_rx: Any, result_tx: Any) -> None:
    # Deepest child ("worker"): its multiprocessing parent is the intermediate "owner". Block on the real
    # supervision wait; when the owner is SIGKILLed the parent sentinel becomes ready -> OWNER_DEATH.
    parent_sentinel, is_parent_alive = _mp._owner_parent_signals()  # noqa: SLF001
    reason = _mp._wait_for_worker_stop_reason(lifecycle_rx, parent_sentinel, is_parent_alive)  # noqa: SLF001
    result_tx.send(reason.name)


def _sigkill_owner_target(lifecycle_rx: Any, result_tx: Any, pid_tx: Any) -> None:
    # Intermediate "owner": spawn the worker (so the worker's parent is THIS process), hand its pid back, then
    # block until SIGKILLed by the test.
    ctx = multiprocessing.get_context("fork")
    worker = ctx.Process(target=_sigkill_worker_target, args=(lifecycle_rx, result_tx))
    worker.start()
    pid_tx.send(worker.pid)
    worker.join()  # never returns once the test SIGKILLs this owner; the worker detects the death itself


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_worker_detects_owner_sigkill_via_parent_sentinel() -> None:
    # Actual owner SIGKILL: the worker's parent-process sentinel becomes ready and the worker resolves
    # OWNER_DEATH with no traffic, message, or timeout — the authoritative OS liveness path end to end.
    ctx = multiprocessing.get_context("fork")
    lifecycle_rx, lifecycle_tx = ctx.Pipe(duplex=False)  # test keeps tx open -> only the sentinel can fire
    result_rx, result_tx = ctx.Pipe(duplex=False)
    pid_rx, pid_tx = ctx.Pipe(duplex=False)
    owner = ctx.Process(target=_sigkill_owner_target, args=(lifecycle_rx, result_tx, pid_tx))
    owner.start()
    lifecycle_rx.close()
    result_tx.close()
    pid_tx.close()
    owner_pid = owner.pid
    assert owner_pid is not None
    worker_pid = None
    try:
        assert pid_rx.poll(10.0)
        worker_pid = pid_rx.recv()
        time.sleep(0.5)  # let the worker reach its supervision wait
        assert _pid_alive(worker_pid)
        os.kill(owner_pid, signal.SIGKILL)  # owner dies -> worker's parent sentinel becomes ready
        assert result_rx.poll(10.0)
        assert result_rx.recv() == WorkerStopReason.OWNER_DEATH.name
        deadline = time.monotonic() + 10.0
        while _pid_alive(worker_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_alive(worker_pid)  # worker exited on its own after detecting owner death
    finally:
        with contextlib.suppress(Exception):
            os.kill(owner_pid, signal.SIGKILL)
        owner.join(timeout=5.0)
        if worker_pid is not None and _pid_alive(worker_pid):
            with contextlib.suppress(Exception):
                os.kill(worker_pid, signal.SIGKILL)
        for conn in (lifecycle_tx, result_rx, pid_rx):
            with contextlib.suppress(Exception):
                conn.close()


def test_spawn_worker_records_proc_and_lifecycle_tx_together_before_wiring_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # MEDIUM (aligned ownership): even if owner-side wiring fails after the process starts, on_started has
    # already recorded BOTH the process and its lifecycle endpoint, so a partial-startup rollback never
    # retains a process without its lifecycle channel.
    recorded: dict[str, Any] = {}

    def _on_started(proc: Any, lifecycle_tx: Any) -> None:
        recorded["proc"] = proc
        recorded["lifecycle_tx"] = lifecycle_tx

    # Fail the first owner-side wiring step after on_started (the scene-attach-channel construction).
    def _boom_channel(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("owner-side wiring boom")

    monkeypatch.setattr(_mp, "SceneAttachChannel", _boom_channel)

    ctx = multiprocessing.get_context("fork")
    with pytest.raises(RuntimeError, match="wiring boom"):
        _mp._spawn_worker(  # noqa: SLF001
            0,
            ctx=ctx,
            scene_id="scene",
            artifact_path=Path("/nonexistent.usdz"),
            spawn_config=WorkerSpawnConfig(
                renderer_name="gsplat",
                enable_editing_actors=False,
                metrics_output_dir=None,
                ray_chunk_size=1,
                scene_cache_size=1,
                post_process=None,
            ),
            registry=_registry_with_scene("scene"),
            on_started=_on_started,
        )

    assert "proc" in recorded and recorded.get("lifecycle_tx") is not None  # both recorded together
    recorded["proc"].join(timeout=5.0)
    assert not recorded["proc"].is_alive()  # the local rollback reaped the half-wired worker


def test_spawn_worker_creates_only_scene_and_handshake_queues() -> None:
    # `_spawn_worker` creates exactly the five scene/handshake queues (attach req/resp + port + scene
    # req/ack) and no extra queue pair, and the worker payload it builds carries only those slots (bind
    # against `_worker` so a signature drift is caught). Editing is unsupported in multiproc serving, so no
    # edit-set queues are ever wired.
    class _FakePipeEnd:
        def close(self) -> None:
            pass

    class _FakeSpawnProc:
        def __init__(self, target: Any, args: "tuple[Any, ...]") -> None:
            self.target, self.args = target, args

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return True

    class _FakeSpawnCtx:
        def __init__(self) -> None:
            self.queues: list[object] = []

        def Queue(self) -> object:  # noqa: N802 - multiprocessing context API
            q = object()
            self.queues.append(q)
            return q

        def Pipe(self, duplex: bool = True) -> "tuple[_FakePipeEnd, _FakePipeEnd]":  # noqa: N802
            return _FakePipeEnd(), _FakePipeEnd()

        def Process(self, target: Any, args: "tuple[Any, ...]") -> _FakeSpawnProc:  # noqa: N802
            return _FakeSpawnProc(target, args)

    ctx = _FakeSpawnCtx()
    handle = _mp._spawn_worker(  # noqa: SLF001
        0,
        ctx=ctx,
        scene_id="scene",
        artifact_path=Path("/nonexistent.usdz"),
        spawn_config=WorkerSpawnConfig(renderer_name="gsplat", enable_editing_actors=False, ray_chunk_size=1),
        registry=_registry_with_scene("scene"),
        on_started=None,
    )

    # attach req/resp + port + scene req/ack only — no edit-set pair was ever created.
    assert len(ctx.queues) == 5
    # The worker signature carries no edit-set slots, and the child payload binds cleanly against the NEW
    # contract: ids/queues/pipes stay positional and the static per-worker knobs collapse into one
    # WorkerSpawnConfig slot (F013), so the pickled payload is exactly 10 args (down from 16).
    params = inspect.signature(_mp._worker).parameters  # noqa: SLF001
    assert "editset_q" not in params and "editset_ack_q" not in params
    assert len(handle.proc.args) == 10  # worker_id, scene_id, artifact_path, config, 4 queues/pipe, 2 scene queues
    assert isinstance(handle.proc.args[3], WorkerSpawnConfig)  # the bundled static knobs travel as one object
    inspect.signature(_mp._worker).bind(*handle.proc.args)  # noqa: SLF001 - binds cleanly against the new contract


def test_worker_pool_shutdown_records_failed_lifecycle_stop_and_still_reaps() -> None:
    # A failed control send must not be reported clean: shutdown surfaces it loudly, while the bounded
    # terminate/reap still guarantees the worker is dead (the reap is the fallback for the failed request).
    class _FailingLifecycleTx:
        def __init__(self) -> None:
            self.closed = False

        def send(self, _obj: object) -> None:
            raise OSError("lifecycle send boom")

        def close(self) -> None:
            self.closed = True

    proc = _ReapFakeProc(dies_on_terminate=True)  # ignores the (failed) request but dies on SIGTERM
    tx = _FailingLifecycleTx()
    pool = WorkerPool(
        scene_id="s",
        worker_addresses=[],
        owner_backend=None,
        _procs=[proc],
        _lifecycle_txs=[tx],
    )

    with pytest.raises(RuntimeError, match="lifecycle stop send"):
        pool.shutdown()

    assert proc.terminated and not proc.is_alive()  # bounded terminate/reap remained the fallback
    assert tx.closed  # still closed for fd hygiene despite the send failure


def test_terminate_and_reap_escalates_to_kill_for_stubborn_worker() -> None:
    # B1: a worker that ignores SIGTERM (wedged in a CUDA kernel / gRPC I/O) must still be reaped, or its
    # CUDA context / IPC mappings / port binding linger. terminate() -> join times out -> kill().
    proc = _ReapFakeProc(dies_on_terminate=False)
    _mp._terminate_and_reap(proc, join_timeout_s=0.01)  # noqa: SLF001
    assert proc.terminated and proc.killed and not proc.is_alive()


def test_terminate_and_reap_no_kill_when_terminate_succeeds() -> None:
    # A well-behaved worker exits on SIGTERM; we must NOT escalate to SIGKILL.
    proc = _ReapFakeProc(dies_on_terminate=True)
    _mp._terminate_and_reap(proc, join_timeout_s=0.01)  # noqa: SLF001
    assert proc.terminated and not proc.killed and not proc.is_alive()


def test_terminate_and_reap_noop_for_already_dead_worker() -> None:
    proc = _ReapFakeProc(dies_on_terminate=True, alive=False)
    _mp._terminate_and_reap(proc, join_timeout_s=0.01)  # noqa: SLF001
    assert not proc.terminated and not proc.killed  # never signalled a dead proc


def test_scene_coordinator_seeded_primary_is_already_attached() -> None:
    built: list[str] = []
    builders = [lambda scene_id, artifact_path, ipc_tensors: built.append(scene_id) or "x"]  # type: ignore[func-returns-value]
    _osr, coord, _channels, _tables, stops = _coordinator_with_workers(builders, {}, attached_scenes=("primary",))
    try:
        coord.ensure_attached("primary")  # seeded -> immediate no-op, no attach pushed
        assert built == []
        assert coord.attached_scenes() == {"primary"}
        assert "primary" in coord.known_scenes()
    finally:
        _stop_workers(stops)


def test_scene_coordinator_unknown_scene_raises() -> None:
    builders = [lambda scene_id, artifact_path, ipc_tensors: "x"]
    _osr, coord, _channels, _tables, stops = _coordinator_with_workers(builders, {"b": "/art-b"})
    try:
        with pytest.raises(SceneNotKnownError, match="not known"):
            coord.ensure_attached("does-not-exist")
    finally:
        _stop_workers(stops)


def test_scene_coordinator_owner_load_failure_is_typed_and_retains_cause() -> None:
    """A registered-but-invalid artifact stays known, rolls back cleanly, and surfaces a domain error."""
    load_error = RuntimeError("corrupt model payload")

    def invalid_loader(scene_id: str, artifact_path: str) -> object:
        raise load_error

    osr, coord, channels, tables, stops = _coordinator_with_workers(
        _two_worker_builders(), {"invalid": "/art-invalid"}, loader=invalid_loader
    )
    try:
        with pytest.raises(SceneLoadError, match=r"failed to load.*corrupt model payload") as exc_info:
            coord.ensure_attached("invalid")

        assert exc_info.value.__cause__ is load_error
        assert coord.known_scenes() == {"invalid"}  # discovery is registration-based when validation is off
        assert coord.attached_scenes() == set()
        assert not osr.is_loaded("invalid")
        assert all(table == {} for table in tables)
        assert all(channel.attached_scenes() == set() for channel in channels)
        assert not coord.is_degraded()  # a clean, expected load failure does not tear the fleet down
    finally:
        _stop_workers(stops)


def test_scene_coordinator_registry_publication_failure_remains_internal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the injected loader is translated; owner-registry invariant defects remain loud/raw."""
    osr, coord, _channels, _tables, stops = _coordinator_no_workers({"scene": "/art-scene"})

    def publication_failure(scene_id: str, model: torch.nn.Module) -> None:
        raise RuntimeError("registry publication invariant failed")

    monkeypatch.setattr(osr.registry, "put_loaded", publication_failure)
    try:
        with pytest.raises(RuntimeError, match="registry publication invariant failed") as exc_info:
            coord.ensure_attached("scene")
        assert not isinstance(exc_info.value, SceneLoadError)
        assert not osr.is_loaded("scene")
        assert not coord.is_degraded()
    finally:
        _stop_workers(stops)


def test_scene_coordinator_rolls_back_partial_attach() -> None:
    # Worker 0 attaches fine; worker 1's build fails. The coordinator must detach worker 0, release the
    # owner load, and raise — so the scene is neither routed nor left half-attached.
    def ok_builder(scene_id, artifact_path, ipc_tensors):
        return f"w0::{scene_id}"

    def failing_builder(scene_id, artifact_path, ipc_tensors):
        raise RuntimeError(f"build boom {scene_id}")

    osr, coord, channels, tables, stops = _coordinator_with_workers([ok_builder, failing_builder], {"b": "/art-b"})
    try:
        with pytest.raises(SceneAttachError, match="failed to attach"):
            coord.ensure_attached("b")

        assert coord.attached_scenes() == set()  # not routable
        assert tables[0] == {} and tables[1] == {}  # worker 0 rolled back (detached), worker 1 never had it
        assert all(channel.attached_scenes() == set() for channel in channels)
        # All refcounts rolled back -> the scene became idle and the owner load was released (entry gone).
        assert not osr.is_loaded("b")
        assert "b" not in osr.registry.scene_keys()
    finally:
        _stop_workers(stops)


def test_scene_coordinator_known_scenes_includes_seed_and_artifacts() -> None:
    builders = [lambda scene_id, artifact_path, ipc_tensors: "x"]
    _osr, coord, _channels, _tables, stops = _coordinator_with_workers(
        builders, {"b": "/art-b", "c": "/art-c"}, attached_scenes=("primary",)
    )
    try:
        assert coord.known_scenes() == {"primary", "b", "c"}
    finally:
        _stop_workers(stops)


# ---------------------------------------------------------------------------------------------------
# SceneCoordinator coordinated eviction — budget + LRU victim, detach-before-release, pin
# primary, in-flight lease protection, re-attach, budget-exhausted shed.
# ---------------------------------------------------------------------------------------------------


def _two_worker_builders():
    return [
        (lambda scene_id, artifact_path, ipc_tensors: f"w0::{scene_id}"),
        (lambda scene_id, artifact_path, ipc_tensors: f"w1::{scene_id}"),
    ]


def test_eviction_evicts_idle_scene_at_budget() -> None:
    # Budget 2 (primary pinned + 1 slot). Attaching B over the cap evicts the idle A on every worker, then
    # the owner releases A's IPC tensors; B takes the slot.
    osr, coord, _channels, tables, stops = _coordinator_with_workers(
        _two_worker_builders(), {"a": "/art-a", "b": "/art-b"}, attached_scenes=("primary",), max_scenes=2
    )
    try:
        coord.ensure_attached("a")
        assert coord.attached_scenes() == {"primary", "a"}
        assert tables[0] == {"a": "w0::a"} and tables[1] == {"a": "w1::a"}

        coord.ensure_attached("b")  # over budget -> evict idle LRU non-pinned victim = a
        assert coord.attached_scenes() == {"primary", "b"}  # a evicted, b attached
        assert tables[0] == {"b": "w0::b"} and tables[1] == {"b": "w1::b"}  # a detached on every worker
        assert not osr.is_loaded("a") and "a" not in osr.registry.scene_keys()  # owner released a
        assert osr.registry.attached_count("b") == 2  # b attached on both workers
    finally:
        _stop_workers(stops)


def test_eviction_picks_least_recently_used() -> None:
    # Budget 3 (primary + 2 slots). Attach a, b; touch a (via acquire/release); attach c -> evict b (LRU).
    osr, coord, _channels, tables, stops = _coordinator_with_workers(
        _two_worker_builders(),
        {"a": "/art-a", "b": "/art-b", "c": "/art-c"},
        attached_scenes=("primary",),
        max_scenes=3,
    )
    try:
        coord.ensure_attached("a")
        coord.ensure_attached("b")
        coord.acquire("a")  # touch a -> a is now more-recently-used than b
        coord.release_scene("a")
        coord.ensure_attached("c")  # over budget -> evict LRU idle = b (a was touched after b attached)
        assert coord.attached_scenes() == {"primary", "a", "c"}  # b evicted
        assert "b" not in tables[0] and "b" not in tables[1]
        assert not osr.is_loaded("b")
    finally:
        _stop_workers(stops)


def test_eviction_never_evicts_pinned_primary() -> None:
    # Budget 1 = only the pinned primary fits; any on-demand scene has no idle victim -> shed.
    _osr, coord, _channels, tables, stops = _coordinator_with_workers(
        _two_worker_builders(), {"a": "/art-a"}, attached_scenes=("primary",), max_scenes=1
    )
    try:
        with pytest.raises(SceneBudgetError, match="pinned"):
            coord.ensure_attached("a")
        assert coord.attached_scenes() == {"primary"}  # primary intact, a not attached
        assert tables[0] == {} and tables[1] == {}
    finally:
        _stop_workers(stops)


def test_eviction_skips_in_flight_scene_then_sheds() -> None:
    # Budget 2. a attached + leased (in-flight). Attaching b can't evict a -> SceneBudgetError.
    _osr, coord, _channels, _tables, stops = _coordinator_with_workers(
        _two_worker_builders(), {"a": "/art-a", "b": "/art-b"}, attached_scenes=("primary",), max_scenes=2
    )
    try:
        coord.acquire("a")  # attach + hold a lease (in-flight)
        try:
            with pytest.raises(SceneBudgetError):
                coord.ensure_attached("b")  # a is in-flight, primary is pinned -> nothing to evict
            assert coord.attached_scenes() == {"primary", "a"}
        finally:
            coord.release_scene("a")
        # With the lease released, a is idle and can be evicted to admit b.
        coord.ensure_attached("b")
        assert coord.attached_scenes() == {"primary", "b"}
    finally:
        _stop_workers(stops)


def test_eviction_concurrent_first_attaches_respect_budget() -> None:
    # Over-admission guard: two different first-time attaches must NOT both pass the cap.
    # Budget 2 = pinned primary + 1 slot. Block A mid-load (its slot reserved) and attempt B concurrently;
    # B must be shed (primary pinned, nothing evictable) instead of producing the resident set {primary,a,b}.
    a_loading = threading.Event()
    release_a = threading.Event()

    def loader(scene_id, artifact_path):
        if scene_id == "a":
            a_loading.set()  # set AFTER _reserve_slot has claimed a's slot, before the (blocked) load finishes
            assert release_a.wait(timeout=5.0)
        return _tiny_model()

    _osr, coord, _channels, tables, stops = _coordinator_with_workers(
        _two_worker_builders(),
        {"a": "/art-a", "b": "/art-b"},
        attached_scenes=("primary",),
        max_scenes=2,
        loader=loader,
    )
    try:
        thread_a = threading.Thread(target=lambda: coord.ensure_attached("a"))
        thread_a.start()
        assert a_loading.wait(2.0)  # A has reserved its budget slot and is blocked mid-load
        with pytest.raises(SceneBudgetError):
            coord.ensure_attached("b")  # occupancy = primary(attached) + a(reserved) = 2; primary pinned -> shed
        release_a.set()
        thread_a.join(timeout=5.0)
        assert coord.attached_scenes() == {"primary", "a"}  # never {primary, a, b}
        assert "b" not in tables[0] and "b" not in tables[1]  # B never attached on any worker
    finally:
        release_a.set()
        _stop_workers(stops)


def test_eviction_reattaches_after_evict() -> None:
    # Budget 2. a -> b (evicts a) -> a (evicts b): the re-attach reloads a on the owner + every worker.
    osr, coord, _channels, tables, stops = _coordinator_with_workers(
        _two_worker_builders(), {"a": "/art-a", "b": "/art-b"}, attached_scenes=("primary",), max_scenes=2
    )
    try:
        coord.ensure_attached("a")
        coord.ensure_attached("b")  # evicts a
        coord.ensure_attached("a")  # evicts b, re-attaches a
        assert coord.attached_scenes() == {"primary", "a"}
        assert tables[0] == {"a": "w0::a"} and tables[1] == {"a": "w1::a"}
        assert not osr.is_loaded("b") and osr.is_loaded("a")
    finally:
        _stop_workers(stops)


def test_eviction_release_not_freed_degrades_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # a victim whose detach acks CLEAN but whose owner release still reports not-freed is an
    # ownership-accounting invariant failure. The pool must FAIL CLOSED — mark degraded, fire the teardown
    # callback, keep the victim quarantined (charged against the budget, owner tensors retained), keep its
    # per-scene lock (forget_scene_lock's quiescent-only precondition), and REFUSE to admit the incoming scene
    # — rather than over-admitting on top of still-resident VRAM (the old code silently proceeded).
    fired: list[str] = []
    _osr, coord, _channels, _tables, stops = _coordinator_with_workers(
        _two_worker_builders(), {"a": "/art-a", "b": "/art-b"}, attached_scenes=("primary",), max_scenes=2
    )
    coord.set_degraded_callback(lambda reason: fired.append(reason))
    try:
        coord.ensure_attached("a")
        assert "a" in coord._scene_locks  # noqa: SLF001 - per-scene lock created for a

        # Detach acks clean, but the owner reports the loaded victim did NOT free -> invariant failure.
        monkeypatch.setattr(coord._owner_registry, "release", lambda scene_id: False)  # noqa: SLF001
        with pytest.raises(ScenePoolDegradedError):
            coord.ensure_attached("b")  # evict a -> release not-freed -> degrade -> b fails closed

        assert coord.is_degraded()
        assert fired and "release returned False" in fired[0]  # teardown callback fired with the reason
        assert "a" in coord._retained  # noqa: SLF001 - victim quarantined (charged; owner tensors retained)
        assert "a" in coord._scene_locks  # noqa: SLF001 - lock kept (not pruned under a possibly-live mapping)
        assert "b" not in coord.attached_scenes()  # no over-admission onto a degraded pool
    finally:
        _stop_workers(stops)


def test_eviction_disabled_when_max_scenes_zero() -> None:
    # max_scenes=0 (default) -> unlimited: all scenes stay resident, no eviction.
    _osr, coord, _channels, tables, stops = _coordinator_with_workers(
        _two_worker_builders(),
        {"a": "/art-a", "b": "/art-b", "c": "/art-c"},
        attached_scenes=("primary",),
        max_scenes=0,
    )
    try:
        for scene in ("a", "b", "c"):
            coord.ensure_attached(scene)
        assert coord.attached_scenes() == {"primary", "a", "b", "c"}
        assert tables[0] == {"a": "w0::a", "b": "w0::b", "c": "w0::c"}
    finally:
        _stop_workers(stops)


# ---------------------------------------------------------------------------------------------------
# Path-driven pinning (pin iff no registered path, or explicitly pinned) + glob planning.
# ---------------------------------------------------------------------------------------------------


def test_pinning_registered_scenes_are_zero_pinned_and_evictable() -> None:
    # A bootstrap scene that IS registered (has a path) + no explicit pins -> zero pinned -> evictable.
    osr, coord, _channels, tables, stops = _coordinator_with_workers(
        _two_worker_builders(), {"a": "/art-a", "b": "/art-b"}, attached_scenes=("a",), max_scenes=1
    )
    try:
        assert coord.pinned_scenes() == set()  # registered bootstrap is not pinned
        coord.ensure_attached("b")  # over budget (1) -> a is evictable (registered, unpinned, idle)
        assert coord.attached_scenes() == {"b"}  # a evicted
        assert "a" not in tables[0] and not osr.is_loaded("a")
    finally:
        _stop_workers(stops)


def test_pinning_explicit_pin_is_never_evicted() -> None:
    # An explicitly-pinned scene is never a victim even though it has a path.
    _osr, coord, _channels, tables, stops = _coordinator_with_workers(
        _two_worker_builders(), {"a": "/art-a", "b": "/art-b"}, attached_scenes=("a",), max_scenes=1, pinned=("a",)
    )
    try:
        assert coord.pinned_scenes() == {"a"}
        with pytest.raises(SceneBudgetError):
            coord.ensure_attached("b")  # only resident scene 'a' is pinned -> nothing to evict
        assert coord.attached_scenes() == {"a"} and tables[0] == {}
    finally:
        _stop_workers(stops)


def test_pinning_pathless_scene_is_pinned() -> None:
    # A resident scene with NO registered path can't be reloaded, so it is never evicted (pinned).
    _osr, coord, _channels, _tables, stops = _coordinator_with_workers(
        _two_worker_builders(), {"b": "/art-b"}, attached_scenes=("primary",), max_scenes=1
    )
    try:
        assert coord.pinned_scenes() == {"primary"}  # path-less -> pinned
        with pytest.raises(SceneBudgetError):
            coord.ensure_attached("b")  # primary is path-less (unevictable), so no room
        assert coord.attached_scenes() == {"primary"}
    finally:
        _stop_workers(stops)


def test_plan_glob_scenes_single_match() -> None:
    bootstrap, scene_id, extra = _plan_glob_scenes(["/g/only.usdz"], scene_id_of=lambda p: "scene-only")
    assert bootstrap == Path("/g/only.usdz") and scene_id == "scene-only" and extra == {}


def test_plan_glob_scenes_multi_match_keys_by_embedded_scene_id() -> None:
    ids = {"/g/a.usdz": "scene-a", "/g/b.usdz": "scene-b", "/g/c.usdz": "scene-c"}
    bootstrap, scene_id, extra = _plan_glob_scenes(sorted(ids), scene_id_of=lambda p: ids[p])
    assert bootstrap == Path("/g/a.usdz") and scene_id == "scene-a"  # first sorted match is the bootstrap
    assert extra == {"scene-b": "/g/b.usdz", "scene-c": "/g/c.usdz"}  # the rest, keyed by embedded scene_id


def test_plan_glob_scenes_rejects_duplicate_scene_id() -> None:
    # Two files with the same embedded scene_id (e.g. a naive copy+rename) is rejected, like serve.
    with pytest.raises(ValueError, match="duplicate scene_id"):
        _plan_glob_scenes(["/g/a.usdz", "/g/copy.usdz"], scene_id_of=lambda p: "same-id")


def test_plan_glob_scenes_empty_returns_no_bootstrap() -> None:
    # An empty glob is a valid ZERO-startup-scene state, not an error — no bootstrap path, no bootstrap
    # scene_id, no extras. (The duplicate-scene_id guard above still raises for a non-empty glob.)
    assert _plan_glob_scenes([], scene_id_of=lambda p: "x") == (None, None, {})


# ---------------------------------------------------------------------------------------------------
# RenderProxy scene routing — ensure_attached before routing; map errors; list known scenes.
# ---------------------------------------------------------------------------------------------------


class _FakeSceneCoordinator:
    """Records `acquire`/`release_scene` (the lease the proxy now uses) and optionally raises, to
    test the proxy's scene-routing gate. `ensure_attached` is kept for direct-call tests."""

    def __init__(self, known=("s",), raise_for=None, attached=None, max_scenes: int = 0) -> None:
        self._known = set(known)
        self._attached = set(known) if attached is None else set(attached)
        self.max_scenes = max_scenes
        self._raise_for = dict(raise_for or {})  # scene_id -> exception instance to raise
        self.acquire_calls: list[str] = []
        self.release_calls: list[str] = []
        self.ensure_calls: list[str] = []

    def acquire(self, scene_id: str) -> None:
        self.acquire_calls.append(scene_id)
        exc = self._raise_for.get(scene_id)
        if exc is not None:
            raise exc

    def release_scene(self, scene_id: str) -> None:
        self.release_calls.append(scene_id)

    def ensure_attached(self, scene_id: str) -> None:
        self.ensure_calls.append(scene_id)
        exc = self._raise_for.get(scene_id)
        if exc is not None:
            raise exc

    def known_scenes(self) -> set:
        return set(self._known)

    def attached_scenes(self) -> set:
        return set(self._attached)

    def is_known(self, scene_id: str) -> bool:
        return scene_id in self._known


def _proxy_with_coordinator(coordinator, num_workers: int = 2):
    calls: list[_CallRecord] = []
    lock = threading.Lock()
    stubs = [_RecordingStub(worker_id, calls, lock) for worker_id in range(num_workers)]
    return RenderProxy(stubs, scene_coordinator=coordinator), calls


def test_render_rgb_ensures_scene_before_routing() -> None:
    coord = _FakeSceneCoordinator(known={"b"})
    proxy, calls = _proxy_with_coordinator(coord)
    response = proxy.render_rgb(sensorsim_pb2.RGBRenderRequest(scene_id="b"), _FakeContext())
    assert coord.acquire_calls == ["b"]  # attach barrier + lease ran before routing
    assert coord.release_calls == ["b"]  # lease released after the forward
    assert response[0] == "worker_response"
    assert [call.method for call in calls] == ["render_rgb"]


def test_render_rgb_unknown_scene_aborts_not_found() -> None:
    coord = _FakeSceneCoordinator(raise_for={"zzz": SceneNotKnownError("scene 'zzz' is not known")})
    proxy, calls = _proxy_with_coordinator(coord)
    with pytest.raises(_AbortError) as exc:
        proxy.render_rgb(sensorsim_pb2.RGBRenderRequest(scene_id="zzz"), _FakeContext())
    assert exc.value.code == grpc.StatusCode.NOT_FOUND
    assert calls == []  # never routed to a worker
    assert coord.release_calls == []  # a failed acquire takes no lease to release


def test_render_rgb_attach_failure_aborts_unavailable() -> None:
    coord = _FakeSceneCoordinator(raise_for={"b": SceneAttachError("scene 'b' failed to attach")})
    proxy, calls = _proxy_with_coordinator(coord)
    with pytest.raises(_AbortError) as exc:
        proxy.render_rgb(sensorsim_pb2.RGBRenderRequest(scene_id="b"), _FakeContext())
    assert exc.value.code == grpc.StatusCode.UNAVAILABLE
    assert calls == []  # half-attached scene is not routed


def test_render_rgb_degraded_pool_aborts_unavailable() -> None:
    # The degraded (fail-closed) lease outcome is the fifth `_scene_lease` mapping; it must surface at the
    # proxy boundary as UNAVAILABLE (fleet restarting), not collapse to UNKNOWN if the except-branch is lost.
    coord = _FakeSceneCoordinator(raise_for={"b": ScenePoolDegradedError("scene pool degraded (fail-closed)")})
    proxy, calls = _proxy_with_coordinator(coord)
    with pytest.raises(_AbortError) as exc:
        proxy.render_rgb(sensorsim_pb2.RGBRenderRequest(scene_id="b"), _FakeContext())
    assert exc.value.code == grpc.StatusCode.UNAVAILABLE
    assert calls == []  # a degraded pool never routes to a worker


def test_render_lidar_unknown_scene_aborts_not_found() -> None:
    # render_lidar goes through the same `_scene_lease` gate as render_rgb, so a lease rejection maps to the
    # same typed status (one code suffices to pin the shared path): an unknown scene aborts NOT_FOUND.
    coord = _FakeSceneCoordinator(raise_for={"zzz": SceneNotKnownError("scene 'zzz' is not known")})
    proxy, calls = _proxy_with_coordinator(coord)
    with pytest.raises(_AbortError) as exc:
        proxy.render_lidar(sensorsim_pb2.LidarRenderRequest(scene_id="zzz"), _FakeContext())
    assert exc.value.code == grpc.StatusCode.NOT_FOUND
    assert calls == []  # never routed to a worker
    assert coord.release_calls == []  # a failed acquire takes no lease to release


def test_validation_off_invalid_registered_scene_is_listed_then_returns_unavailable() -> None:
    """Registered discovery is truthful without eager validation, and late owner-load failure is typed.

    This uses the real coordinator/proxy boundary: an invalid but metadata-readable artifact is discoverable,
    its first scene-using RPC returns UNAVAILABLE (never UNKNOWN), and an unregistered ID remains NOT_FOUND.
    """

    def invalid_loader(scene_id: str, artifact_path: str) -> object:
        raise RuntimeError("invalid backend payload")

    _osr, coord, _channels, _tables, stops = _coordinator_with_workers(
        _two_worker_builders(), {"invalid": "/art-invalid"}, loader=invalid_loader
    )
    try:
        proxy, calls = _proxy_with_coordinator(coord)
        available = proxy.get_available_scenes(None, _FakeContext())
        assert list(available.scene_ids) == ["invalid"]

        with pytest.raises(_AbortError) as invalid_exc:
            proxy.render_rgb(sensorsim_pb2.RGBRenderRequest(scene_id="invalid"), _FakeContext())
        assert invalid_exc.value.code == grpc.StatusCode.UNAVAILABLE
        assert "invalid backend payload" in invalid_exc.value.details
        assert calls == []  # the invalid scene is never routed to a render worker

        with pytest.raises(_AbortError) as unknown_exc:
            proxy.render_rgb(sensorsim_pb2.RGBRenderRequest(scene_id="unknown"), _FakeContext())
        assert unknown_exc.value.code == grpc.StatusCode.NOT_FOUND
        assert calls == []
    finally:
        _stop_workers(stops)


def test_render_rgb_budget_full_aborts_resource_exhausted() -> None:
    coord = _FakeSceneCoordinator(raise_for={"b": SceneBudgetError("scene budget reached")})
    proxy, calls = _proxy_with_coordinator(coord)
    with pytest.raises(_AbortError) as exc:
        proxy.render_rgb(sensorsim_pb2.RGBRenderRequest(scene_id="b"), _FakeContext())
    assert exc.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED  # Budget shed
    assert calls == [] and coord.release_calls == []


def test_batch_render_rgb_ensures_scene_once() -> None:
    coord = _FakeSceneCoordinator(known={"s"})
    proxy, _calls = _proxy_with_coordinator(coord)
    proxy.batch_render_rgb(_batch_request(3), _FakeContext())
    assert coord.acquire_calls == ["s"]  # one lease for the whole batch's scene
    assert coord.release_calls == ["s"]


def test_get_available_scenes_lists_coordinator_known_scenes() -> None:
    coord = _FakeSceneCoordinator(known={"b", "a", "primary"})
    proxy, _calls = _proxy_with_coordinator(coord)
    response = proxy.get_available_scenes(None, _FakeContext())
    assert list(response.scene_ids) == ["a", "b", "primary"]  # sorted, all owner-serveable scenes


def test_get_loaded_scenes_reports_attached_scenes_from_owner_view() -> None:
    # Answered from the coordinator's fleet view (attached = resident on EVERY worker), never
    # forwarded to a single worker whose local cache can disagree with the fleet.
    coord = _FakeSceneCoordinator(known={"a", "b", "c"}, attached={"c", "a"}, max_scenes=3)
    proxy, calls = _proxy_with_coordinator(coord, num_workers=2)
    response = proxy.get_loaded_scenes(None, _FakeContext())
    assert [entry.scene_id for entry in response.scenes] == ["a", "c"]  # sorted; attached only, not all known
    assert [entry.loaded_instance_count for entry in response.scenes] == [2, 2]  # one instance per worker
    assert [entry.reusable_instance_count for entry in response.scenes] == [0, 0]  # no spare/checkout pool-side
    assert response.loaded_instance_capacity == 6  # resident-scene budget (3) x workers (2)
    assert calls == []  # never routed to a worker


def test_get_loaded_scenes_unlimited_budget_reports_zero_capacity() -> None:
    # A coordinator with max_scenes=0 (unlimited residency) reports capacity 0 rather than inventing a cap.
    coord = _FakeSceneCoordinator(known={"s"}, attached={"s"}, max_scenes=0)
    proxy, _calls = _proxy_with_coordinator(coord, num_workers=3)
    response = proxy.get_loaded_scenes(None, _FakeContext())
    assert response.loaded_instance_capacity == 0
    assert [entry.scene_id for entry in response.scenes] == ["s"]
    assert response.scenes[0].loaded_instance_count == 3


def test_get_loaded_scenes_without_coordinator_forwards_to_worker_zero() -> None:
    # Coordinator-less (single-scene) pool: every worker holds exactly the bootstrap scene, so worker 0's
    # local view is the fleet view — same fallback contract as get_available_scenes.
    proxy, calls = _make_proxy(num_workers=2)
    proxy.get_loaded_scenes("req", context=None)
    assert [(call.method, call.worker_id) for call in calls] == [("get_loaded_scenes", 0)]


def test_get_available_cameras_ensures_scene_then_forwards_to_worker_zero() -> None:
    coord = _FakeSceneCoordinator(known={"b"})
    proxy, calls = _proxy_with_coordinator(coord)
    proxy.get_available_cameras(sensorsim_pb2.AvailableCamerasRequest(scene_id="b"), _FakeContext())
    assert coord.acquire_calls == ["b"] and coord.release_calls == ["b"]  # leased across the forward
    assert [(call.method, call.worker_id) for call in calls] == [("get_available_cameras", 0)]


def test_render_rgb_without_coordinator_is_passthrough() -> None:
    proxy, calls = _make_proxy(num_workers=2)  # no coordinator -> single-scene passthrough
    response = proxy.render_rgb(sensorsim_pb2.RGBRenderRequest(scene_id="anything"), _FakeContext())
    assert response[0] == "worker_response"
    assert [call.method for call in calls] == ["render_rgb"]


# ---------------------------------------------------------------------------------------------------
# On-demand scene download/registration via x-nre-scene-url.
# SceneCoordinator.register_scene/is_known (runtime add) + OnDemandSceneRegistrar (resolve+register)
# + SceneDownloadInterceptor (proxy header dispatch, mirroring serve's interceptor).
# ---------------------------------------------------------------------------------------------------


def _coordinator_no_workers(scene_artifacts=None, *, attached_scenes=(), loader=None):
    """A real SceneCoordinator with zero workers — enough to exercise register/is_known/ensure_attached
    (the all-worker barrier is a no-op with no channels; covered with workers above)."""
    return _coordinator_with_workers([], dict(scene_artifacts or {}), attached_scenes=attached_scenes, loader=loader)


def test_scene_coordinator_register_scene_adds_and_is_idempotent() -> None:
    _osr, coord, _ch, _t, stops = _coordinator_no_workers(attached_scenes=("primary",))
    try:
        assert not coord.is_known("b")
        coord.register_scene("b", "/art-b")
        assert coord.is_known("b") and "b" in coord.known_scenes()
        coord.register_scene("b", "/art-b")  # idempotent: same path is a no-op
        with pytest.raises(ValueError, match="different artifact"):
            coord.register_scene("b", "/art-OTHER")  # conflicting path rejected (mirrors serve)
        coord.register_scene("primary", "/whatever")  # already attached -> no-op (not re-registered)
        assert coord.attached_scenes() == {"primary"}
    finally:
        _stop_workers(stops)


def test_scene_coordinator_register_then_ensure_attached_loads_once() -> None:
    loads: list[tuple[str, str]] = []

    def loader(scene_id, artifact_path):
        loads.append((scene_id, artifact_path))
        return _tiny_model()

    osr, coord, _ch, _t, stops = _coordinator_no_workers(loader=loader)
    try:
        coord.register_scene("b", "/art-b")
        coord.ensure_attached("b")  # first-loads on the owner; the (empty) worker barrier is a no-op
        coord.ensure_attached("b")  # idempotent
        assert loads == [("b", "/art-b")]  # owner loaded exactly once
        assert "b" in coord.attached_scenes() and osr.is_loaded("b")
    finally:
        _stop_workers(stops)


def test_on_demand_registrar_local_file_uri_registers(tmp_path) -> None:
    artifact = tmp_path / "scene.usdz"
    artifact.write_bytes(b"fake-artifact")
    _osr, coord, _ch, _t, stops = _coordinator_no_workers()
    try:
        registrar = OnDemandSceneRegistrar(coord)  # no cache/downloader needed for a local file://
        registrar.register_scene_from_uri("dl", f"file://{artifact}")
        assert coord.is_known("dl") and "dl" in coord.known_scenes()
    finally:
        _stop_workers(stops)


def test_on_demand_registrar_skips_already_known_scene() -> None:
    # A scene the coordinator already knows is NOT re-resolved: a bogus URL must not raise (skip path).
    _osr, coord, _ch, _t, stops = _coordinator_no_workers(attached_scenes=("primary",))
    try:
        registrar = OnDemandSceneRegistrar(coord)
        registrar.register_scene_from_uri("primary", "file:///does/not/exist.usdz")  # no resolve -> no error
        assert coord.attached_scenes() == {"primary"}
    finally:
        _stop_workers(stops)


def test_on_demand_registrar_remote_downloads_then_registers(tmp_path) -> None:
    scene_cache = SceneCache(cache_dir=tmp_path / "cache", max_size=5)

    class _FakeDownloader:
        def __init__(self) -> None:
            self.calls = 0

        def download(self, url: str, output_path) -> int:
            self.calls += 1
            Path(output_path).write_bytes(b"scene-bytes")
            return 11

    downloader = _FakeDownloader()
    _osr, coord, _ch, _t, stops = _coordinator_no_workers()
    try:
        registrar = OnDemandSceneRegistrar(coord, scene_cache=scene_cache, downloader=downloader)  # type: ignore[arg-type]
        registrar.register_scene_from_uri("remote_scene", "https://example.com/scene.usdz")
        assert coord.is_known("remote_scene") and scene_cache.has_scene("remote_scene")
        assert downloader.calls == 1
        registrar.register_scene_from_uri("remote_scene", "https://example.com/scene.usdz")  # known -> skip
        assert downloader.calls == 1  # not re-downloaded
    finally:
        _stop_workers(stops)


def test_on_demand_registrar_remote_without_cache_raises() -> None:
    _osr, coord, _ch, _t, stops = _coordinator_no_workers()
    try:
        registrar = OnDemandSceneRegistrar(coord)  # no cache/downloader -> a remote URL cannot resolve
        with pytest.raises(ValueError, match="requires a scene_cache"):
            registrar.register_scene_from_uri("r", "https://example.com/scene.usdz")
        assert not coord.is_known("r")  # nothing registered on failure
    finally:
        _stop_workers(stops)


class _RecordingRegistrar:
    """Records register_scene_from_uri calls; optionally raises, to drive the interceptor's error path."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._raises = raises

    def register_scene_from_uri(self, scene_id: str, scene_url: str) -> None:
        self.calls.append((scene_id, scene_url))
        if self._raises is not None:
            raise self._raises


def _scene_dl_details(metadata):
    details = type("_HandlerCallDetails", (), {})()
    details.invocation_metadata = list(metadata)
    details.method = "/nurec.Sensorsim/RenderRgb"
    return details


def test_scene_download_interceptor_no_header_passes_through() -> None:
    registrar = _RecordingRegistrar()
    interceptor = SceneDownloadInterceptor(registrar)  # type: ignore[arg-type]
    sentinel = object()
    result = interceptor.intercept_service(lambda details: sentinel, _scene_dl_details([]))
    assert result is sentinel
    assert registrar.calls == []


def test_scene_download_interceptor_file_uri_registers_then_continues() -> None:
    registrar = _RecordingRegistrar()
    interceptor = SceneDownloadInterceptor(registrar)  # type: ignore[arg-type]
    sentinel = object()
    result = interceptor.intercept_service(
        lambda details: sentinel,
        _scene_dl_details([("x-nre-scene-url", "file:///data/scene.usdz"), ("x-nre-scene-id", "s1")]),
    )
    assert registrar.calls == [("s1", "file:///data/scene.usdz")]
    assert result is sentinel  # continuation ran after registration


def test_scene_download_interceptor_invalid_scene_id_aborts_invalid_argument() -> None:
    registrar = _RecordingRegistrar()
    interceptor = SceneDownloadInterceptor(registrar)  # type: ignore[arg-type]
    continued: list = []
    result = interceptor.intercept_service(
        lambda details: continued.append(details),
        _scene_dl_details([("x-nre-scene-url", "file:///data/scene.usdz"), ("x-nre-scene-id", "../bad")]),
    )
    assert registrar.calls == [] and continued == []  # never registered, never routed
    with pytest.raises(_AbortError) as exc:
        result.unary_unary(None, _FakeContext())
    assert exc.value.code == grpc.StatusCode.INVALID_ARGUMENT


def test_scene_download_interceptor_registrar_failure_aborts_invalid_argument() -> None:
    registrar = _RecordingRegistrar(raises=ValueError("boom"))
    interceptor = SceneDownloadInterceptor(registrar)  # type: ignore[arg-type]
    continued: list = []
    result = interceptor.intercept_service(
        lambda details: continued.append(details),
        _scene_dl_details([("x-nre-scene-url", "https://example.com/scene.usdz"), ("x-nre-scene-id", "s1")]),
    )
    assert registrar.calls == [("s1", "https://example.com/scene.usdz")] and continued == []
    with pytest.raises(_AbortError) as exc:
        result.unary_unary(None, _FakeContext())
    assert exc.value.code == grpc.StatusCode.INVALID_ARGUMENT


def test_scene_download_interceptor_no_registrar_passes_through() -> None:
    interceptor = SceneDownloadInterceptor(None)  # single-scene pool: header present but nothing wired
    sentinel = object()
    result = interceptor.intercept_service(
        lambda details: sentinel,
        _scene_dl_details([("x-nre-scene-url", "file:///data/scene.usdz"), ("x-nre-scene-id", "s1")]),
    )
    assert result is sentinel


# ---------------------------------------------------------------------------------------------------
# Render post-processing config (egocar hoods + Harmonizer/Difix) threaded from GrpcServerConfig.
# ---------------------------------------------------------------------------------------------------


def test_post_process_config_maps_egocar_and_harmonizer_from_grpc_config() -> None:
    from nurec.grpc.grpc_server_config import GrpcServerConfig

    cfg = GrpcServerConfig(
        egocar_hood_dir="/hoods",
        enable_harmonizer=True,
        harmonizer_url="http://h/model.pt",
        harmonizer_cache="/cache/harmonizer",
        harmonizer_model_filename="model.pt",
        harmonizer_resolution=(512, 768),
    )
    pp = PostProcessConfig.from_grpc_config(cfg)
    assert pp.egocar_hoods_dir == "/hoods"
    assert pp.enable_harmonizer is True
    assert pp.harmonizer_url == "http://h/model.pt"
    assert pp.harmonizer_cache == "/cache/harmonizer"
    assert pp.harmonizer_model_filename == "model.pt"
    assert pp.harmonizer_resolution == (512, 768)


def test_post_process_config_defaults_when_unset() -> None:
    from nurec.grpc.grpc_server_config import GrpcServerConfig

    pp = PostProcessConfig.from_grpc_config(GrpcServerConfig())
    assert pp.egocar_hoods_dir is None  # no ego masks by default
    assert pp.enable_harmonizer is False  # harmonizer off by default


# ---------------------------------------------------------------------------------------------------
# Parallel worker bring-up: `_await_worker_ready` handshake + `_bring_up_pool` bounded-concurrency
# orchestration (no GPU / no real processes — fake procs + in-process queues).
# ---------------------------------------------------------------------------------------------------


class _FakeProc:
    """Stand-in for an mp worker process: a controllable `is_alive()` + a `terminate()` recorder."""

    def __init__(self, alive: bool = True) -> None:
        self._alive = alive
        self.terminated = False

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminated = True


class _RecordingChannel:
    """Stand-in for a `SceneAttachChannel`: records seeded bootstrap attachments + stop()."""

    def __init__(self) -> None:
        self.seeded: list = []
        self.stopped = False

    def seed_attachment(self, scene_id: str, attachment_id: str) -> None:
        self.seeded.append((scene_id, attachment_id))

    def stop(self) -> None:
        self.stopped = True


class _RecordingLifecycleTx:
    """Stand-in for the owner's lifecycle write end: records sent commands + close() (no real fd)."""

    def __init__(self) -> None:
        self.sent: list = []
        self.closed = False

    def send(self, obj: object) -> None:
        self.sent.append(obj)

    def close(self) -> None:
        self.closed = True


def _make_handle(worker_id: int = 0, *, alive: bool = True) -> _WorkerHandle:
    return _WorkerHandle(
        worker_id=worker_id,
        proc=_FakeProc(alive=alive),
        attach_req_q=queue.Queue(),
        attach_resp_q=queue.Queue(),
        port_q=queue.Queue(),
        channel=_RecordingChannel(),  # type: ignore[arg-type]
        lifecycle_tx=_RecordingLifecycleTx(),
    )


def _registry_with_scene(scene_id: str) -> ModelOwnerRegistry:
    registry = ModelOwnerRegistry()
    registry.put_loaded(scene_id, _tiny_model())  # minimal IPC-shareable bundle
    return registry


def test_await_worker_ready_services_attach_and_returns_address() -> None:
    registry = _registry_with_scene("scene-a")
    handle = _make_handle(worker_id=2)
    # Emulate the worker: it asks the owner to attach, then (after the response) binds a port + reports.
    handle.attach_req_q.put(AttachSceneRequest("scene-a"))
    handle.port_q.put(("ready", 2, 54321))

    address = _await_worker_ready(handle, registry, "scene-a", threading.Event(), lambda _m: None, poll_interval_s=0.01)

    assert address == "127.0.0.1:54321"
    response = handle.attach_resp_q.get_nowait()  # the worker received an ok attach response
    assert response.ok and response.attachment is not None
    # The bootstrap's startup attachment is seeded so the coordinator can later detach + release it.
    assert handle.channel.seeded == [("scene-a", response.attachment.attachment_id)]  # type: ignore[attr-defined]


def test_await_worker_ready_records_bootstrap_attachment_before_response_queue_failure() -> None:
    # The registry refcount exists as soon as handle_owner_request succeeds. Record its id on the scene channel
    # before response delivery so a synchronous queue failure cannot make post-reap cleanup lose the attachment.
    class _FailingResponseQueue:
        def put(self, _response) -> None:  # noqa: ANN001
            raise RuntimeError("attach response put boom")

    registry = _registry_with_scene("scene-a")
    handle = _make_handle()
    channel = SceneAttachChannel(queue.Queue(), queue.Queue(), registry, is_worker_alive=lambda: False)
    handle.channel = channel
    handle.attach_resp_q = _FailingResponseQueue()
    handle.attach_req_q.put(AttachSceneRequest("scene-a"))

    with pytest.raises(RuntimeError, match="attach response put boom"):
        _await_worker_ready(handle, registry, "scene-a", threading.Event(), lambda _m: None, poll_interval_s=0.01)

    assert channel.is_attached("scene-a") and registry.attached_count("scene-a") == 1
    released, failures = channel.release_after_reap()
    assert released == ["scene-a"] and failures == []
    assert registry.release_if_idle("scene-a") is True


def test_await_worker_ready_raises_with_worker_traceback_on_error_message() -> None:
    registry = _registry_with_scene("scene-a")
    handle = _make_handle()
    handle.attach_req_q.put(AttachSceneRequest("scene-a"))
    handle.port_q.put(("error", 0, "boom-traceback"))

    with pytest.raises(RuntimeError, match="boom-traceback"):
        _await_worker_ready(handle, registry, "scene-a", threading.Event(), lambda _m: None, poll_interval_s=0.01)


def test_await_worker_ready_fails_fast_when_worker_dies_before_attaching() -> None:
    # A worker that hard-crashes (no error ack) must not hang the owner for the full attach timeout: the
    # liveness poll notices the dead process + empty queue and fails fast.
    registry = _registry_with_scene("scene-a")
    handle = _make_handle(alive=False)  # never sends an attach request

    with pytest.raises(RuntimeError, match="exited during startup"):
        _await_worker_ready(
            handle,
            registry,
            "scene-a",
            threading.Event(),
            lambda _m: None,
            req_timeout_s=5.0,
            poll_interval_s=0.01,
        )


def test_await_worker_ready_bails_on_cancel_event() -> None:
    registry = _registry_with_scene("scene-a")
    handle = _make_handle()  # no messages queued
    cancel = threading.Event()
    cancel.set()  # a sibling already failed

    with pytest.raises(_BringupCancelled):
        _await_worker_ready(handle, registry, "scene-a", cancel, lambda _m: None, poll_interval_s=0.01)


def test_bring_up_pool_runs_workers_in_parallel_when_unbounded() -> None:
    num_workers = 4
    # The barrier only releases once all N bring-ups are in flight at once; a serialized orchestrator
    # would time out here (turning a regression into a failure, not a hang).
    barrier = threading.Barrier(num_workers, timeout=5.0)
    seen: list[int] = []
    lock = threading.Lock()

    def bring_up_one(worker_id: int, cancel_event: threading.Event) -> None:
        barrier.wait()
        with lock:
            seen.append(worker_id)

    _bring_up_pool(num_workers, 0, bring_up_one)  # 0 => all in parallel
    assert sorted(seen) == list(range(num_workers))


def test_bring_up_pool_caps_concurrent_bringups() -> None:
    num_workers, cap = 6, 2
    # A barrier of width `cap` proves AT LEAST `cap` run concurrently; the pool size proves AT MOST `cap`.
    # Together they pin the observed peak to exactly `cap`.
    barrier = threading.Barrier(cap, timeout=5.0)
    current = 0
    observed_max = 0
    lock = threading.Lock()

    def bring_up_one(worker_id: int, cancel_event: threading.Event) -> None:
        nonlocal current, observed_max
        with lock:
            current += 1
            observed_max = max(observed_max, current)
        barrier.wait()
        with lock:
            current -= 1

    _bring_up_pool(num_workers, cap, bring_up_one)
    assert observed_max == cap


def test_bring_up_pool_keeps_results_ordered_by_worker_id() -> None:
    # The orchestrator submits every worker_id exactly once; index-addressed writes stay ordered even
    # when workers finish in reverse order (mirrors start_worker_pool's worker_addresses[worker_id]).
    num_workers = 5
    results: list = [None] * num_workers

    def bring_up_one(worker_id: int, cancel_event: threading.Event) -> None:
        time.sleep(0.01 * (num_workers - worker_id))  # later ids finish first
        results[worker_id] = f"addr-{worker_id}"

    _bring_up_pool(num_workers, 0, bring_up_one)
    assert results == [f"addr-{i}" for i in range(num_workers)]


def test_bring_up_pool_cancels_siblings_and_raises_first_error() -> None:
    num_workers = 4
    started = threading.Event()
    saw_cancel: list = []
    lock = threading.Lock()

    def bring_up_one(worker_id: int, cancel_event: threading.Event) -> None:
        if worker_id == 0:
            started.wait(timeout=5.0)  # let the siblings reach their cancel wait first
            raise ValueError("worker 0 boom")
        if worker_id == 1:
            started.set()
        got_cancel = cancel_event.wait(timeout=5.0)  # released when worker 0's failure sets cancel
        with lock:
            saw_cancel.append((worker_id, got_cancel))
        raise _BringupCancelled(f"worker {worker_id}")  # mimic the real bail-on-cancel

    with pytest.raises(ValueError, match="worker 0 boom"):
        _bring_up_pool(num_workers, 0, bring_up_one)

    # Every sibling observed the cancel signal (so in-flight bring-ups unwind instead of blocking).
    assert {worker_id for worker_id, _ in saw_cancel} == {1, 2, 3}
    assert all(got_cancel for _, got_cancel in saw_cancel)


def _patch_failed_pool_startup(
    monkeypatch: pytest.MonkeyPatch,
    proc,
    captured: dict,
) -> None:
    """Drive start_worker_pool through a bootstrap attachment followed by a ready failure, without GPU/processes."""

    class _Artifact:
        scene_id = "scene-a"
        sha256_footprint = "0" * 64

        def __init__(self, _path) -> None:  # noqa: ANN001
            pass

    class _OwnerBackend:
        def __init__(self) -> None:
            self.renderable_model = _tiny_model()

    owner = _OwnerBackend()
    monkeypatch.setattr(_mp, "Artifact", _Artifact)
    monkeypatch.setattr(_mp.Backend, "from_artifact", staticmethod(lambda *_a, **_k: owner))
    monkeypatch.setattr(_mp.mp, "get_context", lambda _method: object())

    def _spawn(worker_id, **kwargs):  # noqa: ANN001
        lifecycle_tx = _RecordingLifecycleTx()
        # on_started records the proc AND its lifecycle endpoint together (the real signature), so the
        # partial-startup rollback retains both aligned.
        kwargs["on_started"](proc, lifecycle_tx)
        channel = SceneAttachChannel(
            queue.Queue(),
            queue.Queue(),
            kwargs["registry"],
            is_worker_alive=proc.is_alive,
        )
        captured.update(proc=proc, channel=channel, registry=kwargs["registry"], owner=owner, lifecycle_tx=lifecycle_tx)
        return _WorkerHandle(
            worker_id=worker_id,
            proc=proc,
            attach_req_q=queue.Queue(),
            attach_resp_q=queue.Queue(),
            port_q=queue.Queue(),
            channel=channel,
            lifecycle_tx=lifecycle_tx,
        )

    def _fail_after_bootstrap(handle, registry, scene_id, *_args, **_kwargs):  # noqa: ANN001
        attachment = registry.attach(scene_id)
        handle.channel.seed_attachment(scene_id, attachment.attachment_id)
        raise RuntimeError("ready boom after bootstrap attach")

    monkeypatch.setattr(_mp, "_spawn_worker", _spawn)
    monkeypatch.setattr(_mp, "_await_worker_ready", _fail_after_bootstrap)


def test_failed_pool_startup_reaps_then_releases_bootstrap_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    # A post-attach ready failure must use the normal WorkerPool shutdown gate: reap first, then detach the
    # bootstrap registry record, close the channel, and release the owner scene before propagating startup error.
    captured: dict = {}
    proc = _ReapFakeProc(dies_on_terminate=True)
    _patch_failed_pool_startup(monkeypatch, proc, captured)

    with pytest.raises(RuntimeError, match="ready boom after bootstrap attach"):
        _mp.start_worker_pool(
            Path("fake.usdz"),
            num_workers=1,
            renderer_name="gsplat",
            enable_editing_actors=False,
        )

    assert not proc.is_alive()
    assert not captured["channel"].is_attached("scene-a")
    assert captured["registry"].scene_keys() == ()


def test_failed_pool_startup_retains_partial_pool_when_worker_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    # If even SIGKILL cannot prove the partial worker dead, startup must not unwind with bare owner locals. The
    # typed fatal error owns the partial pool so the model/registry/attachment remain live for diagnosis/retry.
    captured: dict = {}
    proc = _UnreapableFakeProc()
    _patch_failed_pool_startup(monkeypatch, proc, captured)

    with pytest.raises(_mp.WorkerPoolStartupCleanupError) as caught:
        _mp.start_worker_pool(
            Path("fake.usdz"),
            num_workers=1,
            renderer_name="gsplat",
            enable_editing_actors=False,
        )

    error = caught.value
    assert isinstance(error.startup_error, RuntimeError)
    assert error.retained_pool._owner_scene_registry.loaded_scenes() == ("scene-a",)  # noqa: SLF001
    assert error.retained_pool.scene_channels[0].is_attached("scene-a")
    assert proc.is_alive()


# ---------------------------------------------------------------------------------------------------
# SceneAttachChannel attach-acknowledgement uncertainty (symmetric with detach).
# ---------------------------------------------------------------------------------------------------


def test_scene_attach_channel_attach_timeout_on_live_worker_retains_refcount() -> None:
    # an attach whose ack times out while the worker is still ALIVE may have installed the IPC mapping,
    # so the owner refcount + attachment are RETAINED (release_if_idle must not free a possibly-mapped tensor).
    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)
    push_q: queue.Queue = queue.Queue()
    ack_q: queue.Queue = queue.Queue()  # no listener -> the attach ack times out
    channel = SceneAttachChannel(push_q, ack_q, osr.registry, is_worker_alive=lambda: True)

    with pytest.raises(RuntimeError, match="still-live worker"):
        channel.attach("a", "/art", ack_timeout_s=0.2)

    assert osr.registry.attached_count("a") == 1  # RETAINED — the worker may still map it
    assert channel.is_attached("a")  # recorded so a later detach / post-reap release can drop it
    assert osr.release("a") is False  # still attached -> cannot free the IPC tensors


def test_scene_attach_channel_attach_timeout_on_dead_worker_rolls_back() -> None:
    # once the worker is provably dead its IPC mapping is gone, so an uncertain attach can safely roll
    # the owner refcount back (avoids a leak) — and the fault is still surfaced.
    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)
    push_q: queue.Queue = queue.Queue()
    ack_q: queue.Queue = queue.Queue()  # no ack
    channel = SceneAttachChannel(push_q, ack_q, osr.registry, is_worker_alive=lambda: False)

    with pytest.raises(RuntimeError, match="dead worker"):
        channel.attach("a", "/art", ack_timeout_s=0.2)

    assert osr.registry.attached_count("a") == 0  # provably dead -> safe to roll back
    assert not channel.is_attached("a")
    assert osr.release("a") is True


def test_scene_attach_channel_queue_failure_retains_attachment_until_reap() -> None:
    # Queue delivery is not a confirmation that the worker did or did not receive the attach. The registry
    # attachment must already be discoverable by the channel so post-reap cleanup can release it.
    class _FailingPutQueue:
        def put(self, _message) -> None:
            raise RuntimeError("queue put boom")

    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)
    channel = SceneAttachChannel(_FailingPutQueue(), queue.Queue(), osr.registry, is_worker_alive=lambda: True)

    with pytest.raises(RuntimeError, match=r"could not be queued.*retaining the owner refcount"):
        channel.attach("a", "/art")

    assert osr.registry.attached_count("a") == 1
    assert channel.is_attached("a")
    assert osr.release("a") is False

    released, failures = channel.release_after_reap()

    assert released == ["a"] and failures == []
    assert not channel.is_attached("a")
    assert osr.registry.attached_count("a") == 0
    assert osr.release("a") is True


def test_scene_attach_channel_worker_error_rollback_failure_retains_attachment_until_reap(monkeypatch) -> None:
    # An explicit worker error makes registry rollback safe, but rollback itself can fail. Keep the provisional
    # ownership record so a retry after worker reap can recover the otherwise-lost attachment id.
    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)

    def failing_builder(scene_id, artifact_path, ipc_tensors):
        raise RuntimeError(f"build boom {scene_id}")

    push_q, ack_q, _attached, thread = _start_scene_listener(failing_builder)
    channel = SceneAttachChannel(push_q, ack_q, osr.registry, is_worker_alive=lambda: True)
    original_detach = osr.registry.detach
    fail_rollback = True

    def detach_once(attachment_id: str) -> None:
        nonlocal fail_rollback
        if fail_rollback:
            fail_rollback = False
            raise RuntimeError("registry detach boom")
        original_detach(attachment_id)

    monkeypatch.setattr(osr.registry, "detach", detach_once)
    try:
        with pytest.raises(RuntimeError, match=r"failed on worker.*rollback also failed.*retaining"):
            channel.attach("a", "/art", ack_timeout_s=2.0)

        assert osr.registry.attached_count("a") == 1
        assert channel.is_attached("a")
        assert osr.release("a") is False
    finally:
        channel.stop()
        thread.join(timeout=2.0)
    assert not thread.is_alive()

    released, failures = channel.release_after_reap()

    assert released == ["a"] and failures == []
    assert not channel.is_attached("a")
    assert osr.registry.attached_count("a") == 0
    assert osr.release("a") is True


def test_scene_attach_channel_dead_timeout_rollback_failure_retains_attachment_until_reap(monkeypatch) -> None:
    # The same ownership invariant applies when a provably-dead worker permits rollback after a timeout.
    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)
    channel = SceneAttachChannel(queue.Queue(), queue.Queue(), osr.registry, is_worker_alive=lambda: False)
    original_detach = osr.registry.detach
    fail_rollback = True

    def detach_once(attachment_id: str) -> None:
        nonlocal fail_rollback
        if fail_rollback:
            fail_rollback = False
            raise RuntimeError("registry detach boom")
        original_detach(attachment_id)

    monkeypatch.setattr(osr.registry, "detach", detach_once)

    with pytest.raises(RuntimeError, match=r"dead worker.*rollback also failed.*retaining"):
        channel.attach("a", "/art", ack_timeout_s=0.05)

    assert osr.registry.attached_count("a") == 1
    assert channel.is_attached("a")

    released, failures = channel.release_after_reap()

    assert released == ["a"] and failures == []
    assert not channel.is_attached("a")
    assert osr.registry.attached_count("a") == 0
    assert osr.release("a") is True


def test_scene_attach_channel_release_after_reap_drops_retained_refcount() -> None:
    # post-reap, a retained attachment's refcount is force-dropped (the dead worker's mapping is gone),
    # so the owner scene becomes idle and freeable.
    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)
    attachment = osr.registry.attach("a")
    push_q: queue.Queue = queue.Queue()
    ack_q: queue.Queue = queue.Queue()
    channel = SceneAttachChannel(push_q, ack_q, osr.registry, is_worker_alive=lambda: True)
    channel.seed_attachment("a", attachment.attachment_id)

    released, failures = channel.release_after_reap()

    assert released == ["a"] and failures == []
    assert osr.registry.attached_count("a") == 0
    assert not channel.is_attached("a")
    assert osr.release("a") is True  # idle now -> freeable


# ---------------------------------------------------------------------------------------------------
# SceneCoordinator detach/release-or-quarantine and degradation (fake channels: instant, no acks).
# ---------------------------------------------------------------------------------------------------


class _FakeSceneChannel:
    """Coordinator-facing `SceneAttachChannel` stand-in with scriptable attach/detach outcomes and no real
    worker/queue, so detach uncertainty + quarantine are exercised instantly (no real ack timeouts)."""

    def __init__(self) -> None:
        self._attached: set[str] = set()
        self.attach_error: "Exception | None" = None
        self.detach_error_sequence: list = []  # per-call Exception|None; a raised error leaves it "retained"
        self.detach_calls = 0
        self.detach_started: "threading.Event | None" = None
        self.detach_block: "threading.Event | None" = None
        self.reaped = False

    def attach(self, scene_id: str, artifact_path: str, *, ack_timeout_s: float | None = None) -> None:
        if self.attach_error is not None:
            raise self.attach_error
        self._attached.add(scene_id)

    def detach(self, scene_id: str, *, ack_timeout_s: float | None = None) -> None:
        self.detach_calls += 1
        if self.detach_started is not None:
            self.detach_started.set()
        if self.detach_block is not None:
            self.detach_block.wait(timeout=5.0)
        outcome = self.detach_error_sequence.pop(0) if self.detach_error_sequence else None
        if outcome is not None:
            raise outcome  # leaves _attached as-is -> is_attached True -> "retained" (a live worker may map it)
        self._attached.discard(scene_id)

    def is_attached(self, scene_id: str) -> bool:
        return scene_id in self._attached

    def attached_scenes(self) -> set[str]:
        return set(self._attached)

    def seed_attachment(self, scene_id: str, attachment_id: str) -> None:
        self._attached.add(scene_id)

    def stop(self) -> None:
        pass

    def release_after_reap(self) -> list[str]:
        self.reaped = True
        released = list(self._attached)
        self._attached.clear()
        return released


class _FakeOwnerRegistry:
    """`OwnerSceneRegistry` stand-in for SceneCoordinator tests (no torch/GPU)."""

    def __init__(self, *, release_returns: "dict[str, bool] | None" = None) -> None:
        self._loaded: set[str] = set()
        self.release_returns = dict(release_returns or {})
        self.released: list[str] = []

    def ensure_loaded(self, scene_id: str, loader) -> object:  # noqa: ANN001
        loader()
        self._loaded.add(scene_id)
        return object()

    def is_loaded(self, scene_id: str) -> bool:
        return scene_id in self._loaded

    def loaded_scenes(self) -> tuple:
        return tuple(sorted(self._loaded))

    def release(self, scene_id: str) -> bool:
        result = self.release_returns.get(scene_id, True)
        if result:
            self._loaded.discard(scene_id)
        self.released.append(scene_id)
        return result

    def forget_scene_lock(self, scene_id: str) -> None:
        pass

    def release_all_idle(self) -> list[str]:
        freed = list(self._loaded)
        self._loaded.clear()
        return freed


def _coordinator_with_fake_channels(
    channels, scene_artifacts, *, attached_scenes=(), max_scenes=0, release_returns=None
):
    reg = _FakeOwnerRegistry(release_returns=release_returns)
    coord = SceneCoordinator(
        reg,  # type: ignore[arg-type]
        channels,
        scene_artifacts,
        lambda sid, path: object(),
        attached_scenes=attached_scenes,
        max_scenes=max_scenes,
    )
    return reg, coord


def test_scene_pool_detach_recovers_on_bounded_retry_without_degrading() -> None:
    # a lost/uncertain detach ack followed by a successful retry frees the slot normally — no degradation.
    ch = _FakeSceneChannel()
    _reg, coord = _coordinator_with_fake_channels(
        [ch], {"a": "/art-a", "b": "/art-b"}, attached_scenes=("primary",), max_scenes=2
    )
    coord.ensure_attached("a")
    ch.detach_error_sequence = [RuntimeError("timed out on a still-live worker")]  # attempt 1 uncertain, then clean

    coord.ensure_attached("b")  # evict a: attempt 1 retained -> retry -> attempt 2 clean -> release -> attach b

    assert not coord.is_degraded()
    assert coord.attached_scenes() == {"primary", "b"}
    assert ch.detach_calls == 2  # one retry
    assert "a" not in coord._retained  # noqa: SLF001


def test_scene_pool_persistent_detach_failure_quarantines_and_fails_closed() -> None:
    # a live worker whose detach never confirms after bounded retries quarantines the victim (charged +
    # tensors retained), marks the pool degraded, fires the teardown callback, and refuses to admit the
    # incoming scene (no over-admission).
    fired: list[str] = []
    ch = _FakeSceneChannel()
    _reg, coord = _coordinator_with_fake_channels(
        [ch], {"a": "/art-a", "b": "/art-b"}, attached_scenes=("primary",), max_scenes=2
    )
    coord.set_degraded_callback(lambda reason: fired.append(reason))
    coord.ensure_attached("a")
    ch.detach_error_sequence = [RuntimeError("still-live worker")] * (_SCENE_DETACH_MAX_ATTEMPTS + 2)

    with pytest.raises(ScenePoolDegradedError):
        coord.ensure_attached("b")

    assert coord.is_degraded() and fired
    assert "a" in coord._retained  # noqa: SLF001 - quarantined (charged against budget, tensors retained)
    assert ch.detach_calls == _SCENE_DETACH_MAX_ATTEMPTS  # bounded retry, no unbounded spin
    assert "b" not in coord.attached_scenes()  # no over-admission on a degraded pool
    # New work against an already-attached scene also fails closed while the fleet restarts.
    with pytest.raises(ScenePoolDegradedError):
        coord.acquire("primary")


def test_scene_pool_busy_detach_does_not_degrade_and_requester_gets_retryable() -> None:
    # F001 escalation refinement (from Wave 1a's report). A victim whose worker still renders it (in-use)
    # refuses the detach with an EXPLICIT busy outcome (SceneDetachBusyError) — a KNOWN-safe, definitely-
    # retryable state where refcounts are correct and the mapping is intentionally retained. Repeated busy
    # refusals must NOT quarantine/degrade the fleet the way an UNCERTAIN detach does: the requester gets a
    # typed, retryable SceneBudgetError (-> RESOURCE_EXHAUSTED) while the pool stays SERVING, the busy victim
    # stays attached and healthy, and once the orphaned render completes a later attach evicts it cleanly.
    # Without this refinement, three rapid busy refusals would (mis)quarantine + degrade the whole fleet for a
    # supervisor restart — a fleet outage from a single client deadline expiring mid-render under evict pressure.
    fired: list[str] = []
    ch = _FakeSceneChannel()
    _reg, coord = _coordinator_with_fake_channels(
        [ch], {"a": "/art-a", "b": "/art-b"}, attached_scenes=("primary",), max_scenes=2
    )
    coord.set_degraded_callback(lambda reason: fired.append(reason))
    coord.ensure_attached("a")  # resident = {primary, a}; budget (2) full; primary is unregistered -> not evictable

    # The worker keeps rendering A after the proxy released the coordinator lease (client deadline/cancel): every
    # detach of A is refused BUSY (in-use). Queue MORE refusals than _SCENE_DETACH_MAX_ATTEMPTS to prove busy
    # never trips the retry-then-quarantine path.
    ch.detach_error_sequence = [
        SceneDetachBusyError("scene detach 'a' refused on a still-live worker: render in flight (backend in-use)")
    ] * (_SCENE_DETACH_MAX_ATTEMPTS + 5)

    # Admitting B must evict A, but A is busy and the only evictable victim -> the requester fails retryable.
    with pytest.raises(SceneBudgetError, match="rendering"):
        coord.ensure_attached("b")

    assert not coord.is_degraded()  # a busy refusal must NEVER degrade the fleet
    assert not fired  # the teardown callback never fired -> the pool stays SERVING
    assert coord.attached_scenes() == {"primary", "a"}  # A restored to _attached, still healthy + fully attached
    assert "a" not in coord._retained  # noqa: SLF001 - NOT quarantined (busy != uncertain)
    assert "b" not in coord.attached_scenes()  # no over-admission onto the still-full budget
    assert ch.detach_calls == 1  # exactly one busy refusal per reservation round — no 3x quarantine spin

    # The orphaned render completes: the worker's next detach succeeds (backend checked back in as a spare) ->
    # eviction of A frees the slot -> B is admitted, all while the fleet stayed SERVING throughout.
    ch.detach_error_sequence = []
    coord.ensure_attached("b")
    assert coord.attached_scenes() == {"primary", "b"}
    assert not coord.is_degraded() and not fired


def test_scene_pool_partial_attach_rollback_with_uncertain_detach_degrades() -> None:
    # a partial attach (worker 1 fails) rolls back through the ONE centralized cleanup path; if rolling
    # back worker 0's attach is itself uncertain, the pool quarantines + degrades and the client sees the
    # truthful terminal status ScenePoolDegradedError (-> UNAVAILABLE), not the raw SceneAttachError.
    fired: list[str] = []
    ch0 = _FakeSceneChannel()
    ch1 = _FakeSceneChannel()
    ch1.attach_error = RuntimeError("failed on worker")  # worker 1 explicitly fails to attach
    _reg, coord = _coordinator_with_fake_channels([ch0, ch1], {"a": "/art-a"}, max_scenes=0)
    coord.set_degraded_callback(lambda reason: fired.append(reason))
    ch0.detach_error_sequence = [RuntimeError("still-live worker")] * (_SCENE_DETACH_MAX_ATTEMPTS + 1)

    with pytest.raises(ScenePoolDegradedError) as exc_info:
        coord.ensure_attached("a")

    assert isinstance(exc_info.value.__cause__, SceneAttachError)  # original failure chained for diagnosis
    assert coord.is_degraded() and fired
    assert "a" in coord._retained  # noqa: SLF001


def test_scene_pool_loader_failure_does_not_degrade() -> None:
    # A loader failure never attached anything, so cleanup is a no-op and the pool must NOT degrade (release of
    # an unloaded scene reporting False is not an invariant failure).
    ch = _FakeSceneChannel()

    def _bad_loader(scene_id, artifact_path):  # noqa: ANN001
        raise RuntimeError("load boom")

    reg = _FakeOwnerRegistry()
    coord = SceneCoordinator(reg, [ch], {"a": "/art-a"}, _bad_loader, max_scenes=0)  # type: ignore[arg-type, list-item]

    with pytest.raises(SceneLoadError, match="load boom") as exc_info:
        coord.ensure_attached("a")

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert not coord.is_degraded()
    assert "a" not in coord._retained  # noqa: SLF001


def test_scene_pool_blocked_waiter_wakes_unavailable_on_degrade() -> None:
    # a thread already blocked waiting for a scene's eviction to finish must WAKE and fail bounded
    # (ScenePoolDegradedError) when that eviction degrades — never wait forever.
    ch = _FakeSceneChannel()
    ch.detach_started = threading.Event()
    ch.detach_block = threading.Event()
    ch.detach_error_sequence = [RuntimeError("still-live worker")] * (_SCENE_DETACH_MAX_ATTEMPTS + 1)
    _reg, coord = _coordinator_with_fake_channels(
        [ch], {"a": "/art-a", "b": "/art-b"}, attached_scenes=("primary",), max_scenes=2
    )
    coord.ensure_attached("a")

    evictor_error: list = []
    waiter_error: list = []

    def _evictor() -> None:
        try:
            coord.ensure_attached("b")  # evicts a; its detach blocks, then fails -> quarantine + degrade
        except BaseException as exc:  # noqa: BLE001
            evictor_error.append(exc)

    def _waiter() -> None:
        try:
            coord.ensure_attached("a")  # a is mid-eviction -> blocks on the _evicting wait
        except BaseException as exc:  # noqa: BLE001
            waiter_error.append(exc)

    evictor = threading.Thread(target=_evictor)
    evictor.start()
    assert ch.detach_started.wait(timeout=5.0)  # a is now in _evicting (detach in progress, blocked)
    waiter = threading.Thread(target=_waiter)
    waiter.start()
    time.sleep(0.1)  # let the waiter reach the _evicting wait()
    ch.detach_block.set()  # release the detach -> it fails -> retries exhaust -> degrade -> notify_all

    evictor.join(timeout=5.0)
    waiter.join(timeout=5.0)
    assert not evictor.is_alive() and not waiter.is_alive()
    assert evictor_error and isinstance(evictor_error[0], ScenePoolDegradedError)
    assert waiter_error and isinstance(waiter_error[0], ScenePoolDegradedError)


def test_coordinator_degraded_callback_signals_scene_pool_degraded_reason() -> None:
    # the coordinator's degraded callback (wired exactly as serve_proxy wires it) records the typed
    # SCENE_POOL_DEGRADED reason on the shared PoolShutdown, which maps to a nonzero fleet exit.
    shutdown = PoolShutdown()
    ch = _FakeSceneChannel()
    _reg, coord = _coordinator_with_fake_channels(
        [ch], {"a": "/art-a", "b": "/art-b"}, attached_scenes=("primary",), max_scenes=2
    )
    coord.set_degraded_callback(lambda _reason: _signal_shutdown(shutdown, PoolTerminalReason.SCENE_POOL_DEGRADED))
    coord.ensure_attached("a")
    ch.detach_error_sequence = [RuntimeError("still-live worker")] * (_SCENE_DETACH_MAX_ATTEMPTS + 1)

    with pytest.raises(ScenePoolDegradedError):
        coord.ensure_attached("b")

    assert shutdown.reason is PoolTerminalReason.SCENE_POOL_DEGRADED
    assert shutdown.reason.exit_code != 0 and not shutdown.reason.is_clean


def test_scene_pool_detach_diagnostics_include_attempt_worker_and_exception(caplog: pytest.LogCaptureFixture) -> None:
    # retry and terminal diagnostics must identify which worker retained the scene, which attempt failed,
    # and the actual exception; bare worker indices are not enough to diagnose an uncertain fleet remotely.
    ch = _FakeSceneChannel()
    _reg, coord = _coordinator_with_fake_channels(
        [ch], {"a": "/art-a", "b": "/art-b"}, attached_scenes=("primary",), max_scenes=2
    )
    coord.ensure_attached("a")
    ch.detach_error_sequence = [RuntimeError("detach detail")] * _SCENE_DETACH_MAX_ATTEMPTS

    with caplog.at_level(logging.WARNING, logger="nurec.grpc.multiproc_serve"):
        with pytest.raises(ScenePoolDegradedError):
            coord.ensure_attached("b")

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert f"attempt 1/{_SCENE_DETACH_MAX_ATTEMPTS}" in messages
    assert f"after {_SCENE_DETACH_MAX_ATTEMPTS} attempts" in messages
    assert "worker 0: RuntimeError('detach detail')" in messages
    assert "retaining the residency budget charge until fleet reap" in messages


def test_scene_pool_degraded_and_control_plane_reasons_are_nonzero() -> None:
    assert PoolTerminalReason.SCENE_POOL_DEGRADED.exit_code != 0
    assert not PoolTerminalReason.SCENE_POOL_DEGRADED.is_clean
    assert PoolTerminalReason.CONTROL_PLANE_FAILURE.exit_code != 0
    assert not PoolTerminalReason.CONTROL_PLANE_FAILURE.is_clean


def test_await_pool_shutdown_reports_scene_pool_degraded() -> None:
    shutdown = PoolShutdown()
    shutdown.set(PoolTerminalReason.SCENE_POOL_DEGRADED)
    live = _Pool(alive=True)
    assert _await_pool_shutdown(live, shutdown, poll_timeout_s=0.01) == PoolTerminalReason.SCENE_POOL_DEGRADED  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------------
# WorkerPool.shutdown post-reap scene-reference release (retain on a surviving worker).
# ---------------------------------------------------------------------------------------------------


class _RecordingSceneChannelForPool:
    def __init__(self, *, failures=None, stop_error: "Exception | None" = None) -> None:
        self.stopped = False
        self.reaped = False
        self._failures = list(failures or [])
        self._stop_error = stop_error

    def stop(self) -> None:
        self.stopped = True
        if self._stop_error is not None:
            raise self._stop_error

    def release_after_reap(self):
        self.reaped = True
        return [], self._failures


class _RecordingOwnerRegistryForPool:
    def __init__(self, *, leaked=()) -> None:
        self.released_all = False
        self._leaked = list(leaked)

    def release_all_idle(self) -> list[str]:
        self.released_all = True
        return []

    def loaded_scenes(self) -> tuple:
        return tuple(self._leaked)  # non-empty models a post-reap refcount/accounting leak


def test_worker_pool_shutdown_releases_scene_refs_after_reap() -> None:
    # a successful shutdown releases retained scene refs (and every now-idle owner scene) ONLY after the
    # worker is proven dead.
    ch = _RecordingSceneChannelForPool()
    reg = _RecordingOwnerRegistryForPool()
    proc = _ReapFakeProc(dies_on_terminate=True)
    pool = WorkerPool(
        scene_id="s",
        worker_addresses=[],
        owner_backend=None,
        scene_channels=[ch],  # type: ignore[list-item]
        _procs=[proc],
        _owner_scene_registry=reg,
    )

    pool.shutdown()

    assert not proc.is_alive()
    assert ch.reaped and reg.released_all


def test_worker_pool_shutdown_retains_scene_refs_if_worker_survives() -> None:
    # Safety complement: a shutdown that cannot reap the worker must NOT drop scene refs (a surviving worker
    # may still map the IPC tensors), and must fail loudly.
    ch = _RecordingSceneChannelForPool()
    reg = _RecordingOwnerRegistryForPool()
    proc = _UnreapableFakeProc()
    pool = WorkerPool(
        scene_id="s",
        worker_addresses=[],
        owner_backend=None,
        scene_channels=[ch],  # type: ignore[list-item]
        _procs=[proc],
        _owner_scene_registry=reg,
    )

    with pytest.raises(RuntimeError, match="retaining scene owner references for safety"):
        pool.shutdown()

    assert not ch.reaped and not reg.released_all  # retained under a surviving worker


def test_worker_pool_shutdown_continues_reaping_after_pre_reap_failures() -> None:
    # One channel-stop/join fault must not strand later workers. Once every worker is nevertheless proven dead,
    # post-reap owner cleanup is safe and is attempted before the aggregate pre-reap failure is surfaced.
    first_channel = _RecordingSceneChannelForPool(stop_error=RuntimeError("channel stop boom"))
    second_channel = _RecordingSceneChannelForPool()
    reg = _RecordingOwnerRegistryForPool()
    join_fault = _JoinFailingFakeProc()
    stubborn = _ReapFakeProc(dies_on_terminate=False)
    pool = WorkerPool(
        scene_id="s",
        worker_addresses=[],
        owner_backend=None,
        scene_channels=[first_channel, second_channel],  # type: ignore[list-item]
        _procs=[join_fault, stubborn],
        _owner_scene_registry=reg,
    )

    with pytest.raises(RuntimeError, match="shutdown incomplete after worker reap") as caught:
        pool.shutdown()

    assert "channel stop boom" in str(caught.value) and "join boom" in str(caught.value)
    assert not join_fault.is_alive() and not stubborn.is_alive() and stubborn.killed
    assert first_channel.reaped and second_channel.reaped and reg.released_all


def test_worker_pool_shutdown_reaps_later_workers_but_retains_refs_if_one_reap_faults() -> None:
    # A failed reap attempt must not prevent later workers from being reaped, but one still-live worker makes
    # post-reap owner release unsafe for the whole shared registry.
    channel = _RecordingSceneChannelForPool()
    reg = _RecordingOwnerRegistryForPool()
    survivor = _TerminateFailingFakeProc()
    later = _ReapFakeProc(dies_on_terminate=False)
    pool = WorkerPool(
        scene_id="s",
        worker_addresses=[],
        owner_backend=None,
        scene_channels=[channel],  # type: ignore[list-item]
        _procs=[survivor, later],
        _owner_scene_registry=reg,
    )

    with pytest.raises(RuntimeError, match="retaining scene owner references for safety") as caught:
        pool.shutdown()

    assert "terminate boom" in str(caught.value)
    assert survivor.is_alive() and not later.is_alive() and later.killed
    assert not channel.reaped and not reg.released_all


def test_worker_pool_shutdown_releases_uncertain_real_scene_owner_only_after_reap() -> None:
    # connect the real registry/channel ownership path to fleet teardown. A seeded attachment models
    # the persistent state left by an applied attach whose ack was lost: its model remains alive before reap,
    # then post-reap detachment + idle release free it.
    registry = OwnerSceneRegistry()
    owner = _tiny_model()
    owner_ref = weakref.ref(owner)
    # The loader is CALLED (not stored) inside ensure_loaded, before the deliberate `del owner` below
    # clears the closure cell — ruff's F821 can't see the call timing, so it is a false positive here.
    registry.ensure_loaded("a", lambda: owner)  # noqa: F821
    attachment = registry.registry.attach("a")
    channel = SceneAttachChannel(queue.Queue(), queue.Queue(), registry.registry, is_worker_alive=lambda: True)
    channel.seed_attachment("a", attachment.attachment_id)
    proc = _ReapFakeProc(dies_on_terminate=True)
    pool = WorkerPool(
        scene_id="s",
        worker_addresses=[],
        owner_backend=None,
        scene_channels=[channel],
        _procs=[proc],
        _owner_scene_registry=registry,
    )
    del attachment
    del owner
    gc.collect()
    assert owner_ref() is not None and registry.loaded_scenes() == ("a",)

    pool.shutdown()

    gc.collect()
    assert not proc.is_alive()
    assert registry.loaded_scenes() == ()
    assert owner_ref() is None


def test_worker_pool_shutdown_retains_uncertain_real_scene_owner_on_survivor() -> None:
    # Safety complement: if reap cannot be proven, neither the attachment nor owner model may be released.
    registry = OwnerSceneRegistry()
    owner = _tiny_model()
    owner_ref = weakref.ref(owner)
    # The loader is CALLED (not stored) inside ensure_loaded, before the deliberate `del owner` below
    # clears the closure cell — ruff's F821 can't see the call timing, so it is a false positive here.
    registry.ensure_loaded("a", lambda: owner)  # noqa: F821
    attachment = registry.registry.attach("a")
    channel = SceneAttachChannel(queue.Queue(), queue.Queue(), registry.registry, is_worker_alive=lambda: True)
    channel.seed_attachment("a", attachment.attachment_id)
    proc = _UnreapableFakeProc()
    pool = WorkerPool(
        scene_id="s",
        worker_addresses=[],
        owner_backend=None,
        scene_channels=[channel],
        _procs=[proc],
        _owner_scene_registry=registry,
    )
    del attachment
    del owner

    with pytest.raises(RuntimeError, match="retaining scene owner references for safety"):
        pool.shutdown()

    gc.collect()
    assert channel.is_attached("a")
    assert registry.loaded_scenes() == ("a",)
    assert owner_ref() is not None


# ---------------------------------------------------------------------------------------------------
# serve_proxy startup bind failure and deterministic drain ordering.
# ---------------------------------------------------------------------------------------------------


class _FakeDrainEvent:
    def __init__(self, result: "bool | Exception") -> None:
        self._result = result

    def wait(self, timeout: float | None = None) -> bool:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _StopRaises:
    """Script entry distinguishing an exception from ``stop`` from one raised by its event's ``wait``."""

    def __init__(self, error: Exception) -> None:
        self.error = error


class _ScriptedStopServer:
    """gRPC-server stand-in with scriptable stop and drain-wait results."""

    def __init__(self, wait_results) -> None:
        self._wait_results = list(wait_results)
        self.stop_grace_args: list = []

    def stop(self, grace):  # noqa: ANN001
        self.stop_grace_args.append(grace)
        result = self._wait_results.pop(0) if self._wait_results else True
        if isinstance(result, _StopRaises):
            raise result.error
        return _FakeDrainEvent(result)


class _RecordingProxy:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_drain_and_close_proxy_graceful_drain_then_closes_channels() -> None:
    # The graceful drain result is CHECKED; on success the worker channels are closed and the reason
    # is preserved.
    server = _ScriptedStopServer([True])
    proxy = _RecordingProxy()
    reason, closed = _drain_and_close_proxy(server, proxy, None, PoolTerminalReason.REQUESTED)  # type: ignore[arg-type]
    assert proxy.closed and closed
    assert reason is PoolTerminalReason.REQUESTED
    assert server.stop_grace_args == [2.0]  # only the graceful stop was needed


def test_drain_and_close_proxy_forced_stop_then_closes_channels() -> None:
    # A graceful drain that times out escalates to an immediate stop; channels close only after that
    # forced stop is CONFIRMED.
    server = _ScriptedStopServer([False, True])  # graceful times out, forced stop confirms
    proxy = _RecordingProxy()
    reason, closed = _drain_and_close_proxy(server, proxy, None, PoolTerminalReason.REQUESTED)  # type: ignore[arg-type]
    assert proxy.closed and closed
    assert reason is PoolTerminalReason.REQUESTED
    assert server.stop_grace_args == [2.0, None]  # graceful then immediate(None)


@pytest.mark.parametrize(
    "graceful_failure",
    [_StopRaises(RuntimeError("graceful stop boom")), RuntimeError("graceful wait boom")],
    ids=["stop-raises", "wait-raises"],
)
def test_drain_and_close_proxy_graceful_exception_forces_stop(graceful_failure) -> None:  # noqa: ANN001
    # A stop() or wait() exception during the graceful drain is uncertainty, not an escape from teardown: the
    # immediate stop is still attempted and a confirmed forced stop permits channel closure.
    server = _ScriptedStopServer([graceful_failure, True])
    proxy = _RecordingProxy()

    reason, closed = _drain_and_close_proxy(server, proxy, None, PoolTerminalReason.REQUESTED)  # type: ignore[arg-type]

    assert closed and proxy.closed
    assert reason is PoolTerminalReason.REQUESTED
    assert server.stop_grace_args == [2.0, None]


@pytest.mark.parametrize(
    "forced_failure",
    [_StopRaises(RuntimeError("forced stop boom")), RuntimeError("forced wait boom")],
    ids=["stop-raises", "wait-raises"],
)
def test_drain_and_close_proxy_forced_exception_defers_server_and_channels(forced_failure) -> None:  # noqa: ANN001
    # A forced stop() or wait() exception must fail closed, retain both the server and proxy, and register a
    # post-reap retry which stops the server before closing its worker channels.
    server = _ScriptedStopServer([False, forced_failure, True])
    proxy = _RecordingProxy()
    deferred: list = []

    reason, closed = _drain_and_close_proxy(
        server,
        proxy,  # type: ignore[arg-type]
        None,
        PoolTerminalReason.REQUESTED,
        defer_close=deferred.append,
    )

    assert not closed and not proxy.closed
    assert reason is PoolTerminalReason.CONTROL_PLANE_FAILURE
    assert len(deferred) == 1
    deferred[0]()  # models WorkerPool.shutdown invoking the retained closer after every worker is reaped
    assert proxy.closed
    assert server.stop_grace_args == [2.0, None, None]


def test_drain_and_close_proxy_unconfirmed_stop_retains_channels_and_exits_nonzero() -> None:
    # If the stop cannot be confirmed, do NOT cut worker channels out from under live handlers, and
    # escalate a REQUESTED stop to a nonzero CONTROL_PLANE_FAILURE.
    server = _ScriptedStopServer([False, False])
    proxy = _RecordingProxy()
    reason, closed = _drain_and_close_proxy(server, proxy, None, PoolTerminalReason.REQUESTED)  # type: ignore[arg-type]
    assert not proxy.closed and not closed  # channels retained for fleet teardown (caller defers the close)
    assert reason is PoolTerminalReason.CONTROL_PLANE_FAILURE and reason.exit_code != 0


def test_drain_and_close_proxy_preserves_nonclean_root_cause_on_unconfirmed_stop() -> None:
    server = _ScriptedStopServer([False, False])
    proxy = _RecordingProxy()
    reason, closed = _drain_and_close_proxy(server, proxy, None, PoolTerminalReason.WORKER_EXIT)  # type: ignore[arg-type]
    assert not proxy.closed and not closed
    assert reason is PoolTerminalReason.WORKER_EXIT and reason.exit_code != 0  # root cause preserved, still nonzero


def test_drain_and_close_proxy_health_stop_unconfirmed_exits_nonzero() -> None:
    server = _ScriptedStopServer([True])  # proxy drains fine -> channels closed
    health = _ScriptedStopServer([False])  # dedicated health stop cannot be confirmed
    proxy = _RecordingProxy()
    reason, closed = _drain_and_close_proxy(server, proxy, health, PoolTerminalReason.REQUESTED)  # type: ignore[arg-type]
    assert proxy.closed and closed  # proxy confirmed -> channels closed; health failure still escalates
    assert reason is PoolTerminalReason.CONTROL_PLANE_FAILURE and reason.exit_code != 0
    assert health.stop_grace_args == [1.0]  # health server stop was still attempted (not skipped)


@pytest.mark.parametrize(
    "health_failure",
    [_StopRaises(RuntimeError("health stop boom")), RuntimeError("health wait boom")],
    ids=["stop-raises", "wait-raises"],
)
def test_drain_and_close_proxy_health_exception_defers_retry(health_failure) -> None:  # noqa: ANN001
    # Dedicated-health uncertainty cannot escape teardown or be forgotten at function return. Retain the
    # health server for a post-reap bounded retry while preserving the independently-confirmed proxy close.
    server = _ScriptedStopServer([True])
    health = _ScriptedStopServer([health_failure, True])
    proxy = _RecordingProxy()
    deferred: list = []

    reason, closed = _drain_and_close_proxy(
        server,
        proxy,  # type: ignore[arg-type]
        health,
        PoolTerminalReason.REQUESTED,
        defer_close=deferred.append,
    )

    assert closed and proxy.closed
    assert reason is PoolTerminalReason.CONTROL_PLANE_FAILURE
    assert len(deferred) == 1
    deferred[0]()
    assert health.stop_grace_args == [1.0, None]


class _WaitRecordingEvent:
    def __init__(self, server: "_FakeBindServer", result: bool) -> None:
        self._server = server
        self._result = result

    def wait(self, timeout: float | None = None) -> bool:
        self._server.waited = True  # prove the partial-startup teardown BOUNDED-waited before closing channels
        return self._result


class _FakeBindServer:
    def __init__(
        self,
        bind_result: int,
        *,
        start_error: "Exception | None" = None,
        stop_waits=(True,),
    ) -> None:
        self._bind_result = bind_result
        self._start_error = start_error
        self._stop_waits = list(stop_waits)
        self.started = False
        self.stopped = False
        self.waited = False
        self.stop_grace_args: list = []

    def add_insecure_port(self, addr: str) -> int:
        return self._bind_result

    def start(self) -> None:
        self.started = True
        if self._start_error is not None:
            raise self._start_error

    def stop(self, grace):  # noqa: ANN001
        self.stopped = True
        self.stop_grace_args.append(grace)
        result = self._stop_waits.pop(0) if self._stop_waits else True
        return _WaitRecordingEvent(self, result)


class _FakeServeProxyServer:
    """grpc-server stand-in for a full serve_proxy run: binds ok, and its stop(grace) drain-wait is scriptable
    so an unconfirmed-stop path can be exercised end-to-end."""

    def __init__(self, *, bind_result: int = 50051, stop_waits=(True,)) -> None:
        self._bind_result = bind_result
        self._stop_waits = list(stop_waits)
        self.started = False
        self.stop_grace_args: list = []

    def add_insecure_port(self, addr: str) -> int:
        return self._bind_result

    def start(self) -> None:
        self.started = True

    def stop(self, grace):  # noqa: ANN001
        self.stop_grace_args.append(grace)
        result = self._stop_waits.pop(0) if self._stop_waits else True
        return _FakeDrainEvent(result)


class _RecordingQueue:
    """Queue stand-in that records the post-reap endpoint teardown (`cancel_join_thread`/`close`)."""

    def __init__(self) -> None:
        self.cancel_join_called = False
        self.closed = False

    def cancel_join_thread(self) -> None:
        self.cancel_join_called = True

    def close(self) -> None:
        self.closed = True


class _FlakyRecordingQueue(_RecordingQueue):
    """Queue endpoint whose teardown operations fail a configured number of times before succeeding."""

    def __init__(self, *, cancel_failures: int = 0, close_failures: int = 0) -> None:
        super().__init__()
        self.cancel_failures = cancel_failures
        self.close_failures = close_failures
        self.cancel_calls = 0
        self.close_calls = 0

    def cancel_join_thread(self) -> None:
        self.cancel_calls += 1
        if self.cancel_failures:
            self.cancel_failures -= 1
            raise RuntimeError("cancel boom")
        super().cancel_join_thread()

    def close(self) -> None:
        self.close_calls += 1
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("close boom")
        super().close()


class _PutFailingRecordingQueue(_RecordingQueue):
    def put(self, _value) -> None:  # noqa: ANN001
        raise RuntimeError("scene stop put boom")


class _FlakyRegistry:
    """ModelOwnerRegistry stand-in whose `detach` fails for specified attachment ids (post-reap aggregation)."""

    def __init__(self, fail_ids) -> None:
        self._fail_ids = set(fail_ids)
        self.detached: list = []

    def detach(self, attachment_id: str) -> None:
        if attachment_id in self._fail_ids:
            raise RuntimeError(f"detach boom {attachment_id}")
        self.detached.append(attachment_id)


class _FakeMetricsManager:
    def __init__(self, *_a, **_k) -> None:
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakePoolForServeProxy:
    def __init__(self) -> None:
        self.worker_addresses = ["127.0.0.1:1"]
        self.owner_backend = None
        self.coordinator = None
        self.scene_id = "s"
        self._owner_scene_registry = None
        self.deferred: list = []

    def all_workers_alive(self) -> bool:
        return True

    def defer_close(self, closer) -> None:  # noqa: ANN001
        self.deferred.append(closer)


def _patch_serve_proxy_plumbing(monkeypatch: pytest.MonkeyPatch, servers, proxy) -> None:
    """Route serve_proxy's gRPC construction to fakes so bind-failure startup is unit-testable without workers."""
    server_iter = iter(servers)
    monkeypatch.setattr(_mp.grpc, "server", lambda *a, **k: next(server_iter))
    monkeypatch.setattr(_mp.RenderProxy, "from_addresses", lambda *a, **k: proxy)
    monkeypatch.setattr(_mp, "add_SensorsimServiceServicer_to_server", lambda *a, **k: None)
    monkeypatch.setattr(_mp.health_pb2_grpc, "add_HealthServicer_to_server", lambda *a, **k: None)


def test_serve_proxy_public_bind_failure_cleans_up_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # a failed public proxy bind (add_insecure_port -> 0) fails startup with NO partial resources left
    # behind — the proxy channels are closed and the server is stopped — then re-raises (nonzero launch).
    server = _FakeBindServer(0)
    proxy = _RecordingProxy()
    _patch_serve_proxy_plumbing(monkeypatch, [server], proxy)

    with pytest.raises(RuntimeError, match="failed to bind proxy"):
        serve_proxy(_FakePoolForServeProxy(), host="127.0.0.1", port=50055, shutdown_event=PoolShutdown())  # type: ignore[arg-type]

    assert not server.started  # never started
    assert server.stopped and server.waited  # stop was bounded-WAITED before closing channels
    assert proxy.closed  # partial-startup resources cleaned up


def test_serve_proxy_dedicated_health_bind_failure_cleans_up_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # with a dedicated health port, a failed HEALTH bind (after a good public bind) also fails startup and
    # cleans up both servers + the proxy channels.
    proxy_server = _FakeBindServer(50056)  # public bind ok
    health_server = _FakeBindServer(0)  # dedicated health bind fails
    proxy = _RecordingProxy()
    _patch_serve_proxy_plumbing(monkeypatch, [proxy_server, health_server], proxy)

    with pytest.raises(RuntimeError, match="failed to bind dedicated health"):
        serve_proxy(
            _FakePoolForServeProxy(),  # type: ignore[arg-type]
            host="127.0.0.1",
            port=50056,
            shutdown_event=PoolShutdown(),
            health_port=50057,
        )

    assert proxy_server.stopped and proxy_server.waited  # bounded wait before channel close
    assert health_server.stopped and health_server.waited
    assert proxy.closed


def test_serve_proxy_public_start_failure_cleans_up_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # a proxy start() failure after a successful bind is also partial startup. The server stop must be
    # confirmed before its worker channels close, and the original startup failure must propagate.
    server = _FakeBindServer(50058, start_error=RuntimeError("proxy start boom"))
    proxy = _RecordingProxy()
    _patch_serve_proxy_plumbing(monkeypatch, [server], proxy)

    with pytest.raises(RuntimeError, match="proxy start boom"):
        serve_proxy(_FakePoolForServeProxy(), host="127.0.0.1", port=50058, shutdown_event=PoolShutdown())  # type: ignore[arg-type]

    assert server.started and server.stopped and server.waited
    assert proxy.closed


def test_serve_proxy_health_start_failure_cleans_up_both_servers_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a dedicated-health start failure occurs after both binds and must stop/check both server objects,
    # close proxy channels only after the proxy stop confirms, and preserve the health startup exception.
    proxy_server = _FakeBindServer(50059)
    health_server = _FakeBindServer(50060, start_error=RuntimeError("health start boom"))
    proxy = _RecordingProxy()
    _patch_serve_proxy_plumbing(monkeypatch, [proxy_server, health_server], proxy)

    with pytest.raises(RuntimeError, match="health start boom"):
        serve_proxy(
            _FakePoolForServeProxy(),  # type: ignore[arg-type]
            host="127.0.0.1",
            port=50059,
            shutdown_event=PoolShutdown(),
            health_port=50060,
        )

    assert health_server.started
    assert proxy_server.stopped and proxy_server.waited
    assert health_server.stopped and health_server.waited
    assert proxy.closed


def test_serve_proxy_partial_startup_unconfirmed_proxy_stop_defers_channel_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a false stop wait during failed startup is NOT confirmation. Keep proxy/server ownership alive in
    # the pool, leave channels open, then retry stop and close only through the post-reap deferred callback.
    server = _FakeBindServer(0, stop_waits=(False, True))
    proxy = _RecordingProxy()
    pool = _FakePoolForServeProxy()
    _patch_serve_proxy_plumbing(monkeypatch, [server], proxy)

    with pytest.raises(RuntimeError, match="failed to bind proxy"):
        serve_proxy(pool, host="127.0.0.1", port=50061, shutdown_event=PoolShutdown())  # type: ignore[arg-type]

    assert server.waited and not proxy.closed
    assert len(pool.deferred) == 1
    pool.deferred[0]()  # WorkerPool.shutdown invokes this only after every worker is reaped.
    assert proxy.closed
    assert server.stop_grace_args == [None, None]


def test_serve_proxy_partial_startup_unconfirmed_health_stop_is_deferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The dedicated health server has no worker channels, but an unconfirmed stop still must retain its server
    # object and retry post-reap instead of silently dropping the failed cleanup.
    proxy_server = _FakeBindServer(50062)
    health_server = _FakeBindServer(0, stop_waits=(False, True))
    proxy = _RecordingProxy()
    pool = _FakePoolForServeProxy()
    _patch_serve_proxy_plumbing(monkeypatch, [proxy_server, health_server], proxy)

    with pytest.raises(RuntimeError, match="failed to bind dedicated health"):
        serve_proxy(
            pool,  # type: ignore[arg-type]
            host="127.0.0.1",
            port=50062,
            shutdown_event=PoolShutdown(),
            health_port=50063,
        )

    assert proxy.closed  # proxy stop confirmed, so only the independent health cleanup is deferred
    assert len(pool.deferred) == 1
    pool.deferred[0]()
    assert health_server.stop_grace_args == [None, None]


def test_serve_proxy_partial_startup_flushes_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # a partial-startup failure must still flush the proxy metrics manager (so a bind failure does not
    # silently drop metrics), and bounded-wait for the server stop before closing channels.
    server = _FakeBindServer(0)  # public bind fails
    proxy = _RecordingProxy()
    _patch_serve_proxy_plumbing(monkeypatch, [server], proxy)
    metrics = _FakeMetricsManager()
    monkeypatch.setattr(_mp, "ServerMetricsManager", lambda *a, **k: metrics)

    with pytest.raises(RuntimeError, match="failed to bind proxy"):
        serve_proxy(
            _FakePoolForServeProxy(),  # type: ignore[arg-type]
            host="127.0.0.1",
            port=50058,
            shutdown_event=PoolShutdown(),
            metrics_output_dir=str(tmp_path),
        )

    assert metrics.shutdown_called  # metrics flushed on the partial-startup failure
    assert server.waited and proxy.closed


def test_drain_and_close_proxy_stops_health_even_when_proxy_stop_unconfirmed() -> None:
    # an unconfirmed proxy stop must NOT early-return past the health server; it keeps proxy channels (for
    # deferred close by fleet teardown) but still stops + checks the dedicated health server.
    server = _ScriptedStopServer([False, False])  # proxy stop never confirmed
    health = _ScriptedStopServer([True])  # health stops fine
    proxy = _RecordingProxy()
    reason, closed = _drain_and_close_proxy(server, proxy, health, PoolTerminalReason.REQUESTED)  # type: ignore[arg-type]
    assert not closed and not proxy.closed  # proxy channels retained for fleet teardown
    assert health.stop_grace_args == [1.0]  # health server still stopped (not skipped by an early return)
    assert reason is PoolTerminalReason.CONTROL_PLANE_FAILURE


def test_serve_proxy_unconfirmed_stop_defers_channel_close_and_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    # End-to-end: when the proxy stop cannot be confirmed, serve_proxy hands the channel close to fleet
    # teardown (pool.defer_close) instead of closing it, and returns a nonzero control-plane reason. The
    # deferred closer performs the real close later (after WorkerPool.shutdown reaps the workers).
    server = _FakeServeProxyServer(bind_result=50051, stop_waits=[False, False])  # stop never confirms
    proxy = _RecordingProxy()
    _patch_serve_proxy_plumbing(monkeypatch, [server], proxy)
    pool = _FakePoolForServeProxy()
    shutdown = PoolShutdown()
    shutdown.set()  # requested stop so the await loop returns immediately

    reason = serve_proxy(pool, host="127.0.0.1", port=50051, shutdown_event=shutdown)  # type: ignore[arg-type]

    assert reason is PoolTerminalReason.CONTROL_PLANE_FAILURE and reason.exit_code != 0
    assert not proxy.closed  # serve_proxy did NOT close the channels
    assert len(pool.deferred) == 1  # the close was handed to fleet teardown
    pool.deferred[0]()  # WorkerPool.shutdown runs this only after reaping the workers
    assert proxy.closed


# ---------------------------------------------------------------------------------------------------
# Cleanup-exception fail-closed, real withheld-ack listener, post-reap loud-fail and queue close.
# ---------------------------------------------------------------------------------------------------


def _release_boom(_scene_id: str) -> bool:
    raise RuntimeError("registry release boom")


def test_scene_pool_eviction_cleanup_exception_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # an unexpected fault inside eviction cleanup (registry.release raises) must NOT bypass quarantine or
    # recreate over-admission: the pool degrades, the victim stays charged/retained, and the reserving scene
    # fails closed.
    ch = _FakeSceneChannel()
    reg, coord = _coordinator_with_fake_channels(
        [ch], {"a": "/art-a", "b": "/art-b"}, attached_scenes=("primary",), max_scenes=2
    )
    coord.ensure_attached("a")
    monkeypatch.setattr(reg, "release", _release_boom)  # cleanup helper's release raises

    with pytest.raises(ScenePoolDegradedError):
        coord.ensure_attached("b")  # evict a: detach clean, release raises -> caught -> degrade -> fail closed

    assert coord.is_degraded()
    assert "a" in coord._retained  # noqa: SLF001 - victim retained (charged; no over-admission)
    assert "b" not in coord.attached_scenes()


def test_scene_pool_attach_cleanup_exception_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # an unexpected fault inside the ensure_attached rollback cleanup also fails closed (degrade + retain +
    # ScenePoolDegradedError), rather than being suppressed by a default-clean assumption. A worker attach
    # failure drives the rollback; the owner `release` then raises, exercising the cleanup-fault path.
    ch = _FakeSceneChannel()
    ch.attach_error = RuntimeError("failed on worker")  # worker fails to attach -> ensure_attached rolls back

    reg, coord = _coordinator_with_fake_channels([ch], {"a": "/art-a"}, max_scenes=0)
    monkeypatch.setattr(reg, "release", _release_boom)

    with pytest.raises(ScenePoolDegradedError):
        coord.ensure_attached("a")

    assert coord.is_degraded()
    assert "a" in coord._retained  # noqa: SLF001


def test_scene_attach_channel_real_listener_withheld_ack_retains_then_detach_recovers() -> None:
    # the production listener applies the attach (worker maps the IPC tensors), while an ack
    # transport wrapper WITHHOLDS its first attach ack. The owner times out on a live worker and RETAINS the
    # refcount + attachment. Releasing that ack late makes it stale; a subsequent production-listener detach
    # acks and drops the refcount, with seq correlation preventing the stale ack from reactivating the scene.
    osr = OwnerSceneRegistry()
    osr.ensure_loaded("a", _tiny_model)
    push_q: queue.Queue = queue.Queue()
    ack_q: queue.Queue = queue.Queue()
    attached: dict = {}

    class _WithholdFirstAttachAck:
        def __init__(self) -> None:
            self.withheld: "tuple | None" = None

        def put(self, message: tuple) -> None:
            if message[2] == _SCENE_ATTACH and self.withheld is None:
                self.withheld = message
                return
            ack_q.put(message)

        def release(self) -> None:
            assert self.withheld is not None
            ack_q.put(self.withheld)
            self.withheld = None

    wrapped_ack_q = _WithholdFirstAttachAck()
    thread = threading.Thread(
        target=_scene_attach_listener,
        args=(attached, lambda scene_id, _path, _ipc: f"backend::{scene_id}", push_q, wrapped_ack_q),
        daemon=True,
    )
    thread.start()
    channel = SceneAttachChannel(push_q, ack_q, osr.registry, is_worker_alive=lambda: True)
    try:
        with pytest.raises(RuntimeError, match="still-live worker"):
            channel.attach("a", "/art", ack_timeout_s=0.3)
        assert attached == {"a": "backend::a"}  # production listener really mapped the scene
        assert osr.registry.attached_count("a") == 1  # owner refcount RETAINED (no UAF)
        assert channel.is_attached("a")

        wrapped_ack_q.release()  # release the REAL late attach ack (seq=1) before sending detach (seq=2)
        channel.detach("a", ack_timeout_s=2.0)  # stale seq=1 drained; production detach ack drops refcount

        assert attached == {}
        assert osr.registry.attached_count("a") == 0
        assert not channel.is_attached("a")
        assert osr.release("a") is True
    finally:
        push_q.put((_SCENE_STOP,))
        thread.join(timeout=2.0)


def test_release_after_reap_aggregates_failures_and_closes_queues() -> None:
    # post-reap release aggregates per-attachment detach failures, RETAINS the failed record for
    # diagnosis/retry, clears only successfully-released records, and closes both queue endpoints (the consumer
    # is dead) instead of enqueueing another stop.
    push_q = _RecordingQueue()
    ack_q = _RecordingQueue()
    channel = SceneAttachChannel(push_q, ack_q, _FlakyRegistry({"att-b"}), is_worker_alive=lambda: False)  # type: ignore[arg-type]
    channel.seed_attachment("a", "att-a")
    channel.seed_attachment("b", "att-b")  # this detach will fail

    released, failures = channel.release_after_reap()

    assert released == ["a"]
    assert [scene_id for scene_id, _exc in failures] == ["b"]
    assert channel.is_attached("b")  # failed record RETAINED for diagnosis/retry
    assert not channel.is_attached("a")
    assert push_q.cancel_join_called and push_q.closed
    assert ack_q.cancel_join_called and ack_q.closed


def test_release_after_reap_reports_and_retries_only_failed_queue_operations() -> None:
    # Queue teardown faults are resource leaks too: report them alongside registry faults, retain the handles,
    # and retry only the operations that did not already succeed.
    push_q = _FlakyRecordingQueue(close_failures=1)
    ack_q = _FlakyRecordingQueue(cancel_failures=1)
    channel = SceneAttachChannel(push_q, ack_q, _FlakyRegistry(set()), is_worker_alive=lambda: False)  # type: ignore[arg-type]

    released, failures = channel.release_after_reap()

    assert released == []
    assert [resource for resource, _exc in failures] == ["push queue close", "ack queue cancel_join_thread"]
    assert push_q.cancel_calls == 1 and push_q.close_calls == 1
    assert ack_q.cancel_calls == 1 and ack_q.close_calls == 1

    released, failures = channel.release_after_reap()

    assert released == [] and failures == []
    assert push_q.cancel_calls == 1 and push_q.close_calls == 2  # successful cancel was retired
    assert ack_q.cancel_calls == 2 and ack_q.close_calls == 1  # successful close was retired
    assert push_q.closed and ack_q.cancel_join_called

    channel.release_after_reap()
    assert push_q.cancel_calls == 1 and push_q.close_calls == 2
    assert ack_q.cancel_calls == 2 and ack_q.close_calls == 1  # fully-clean queues are not touched again


def test_worker_pool_shutdown_reports_real_scene_channel_stop_failure_but_still_reaps_and_cleans() -> None:
    # SceneAttachChannel.stop must not hide a broken stop queue. WorkerPool records the pre-reap failure, still
    # reaps the worker, then safely performs the real channel's post-reap endpoint cleanup.
    push_q = _PutFailingRecordingQueue()
    ack_q = _RecordingQueue()
    channel = SceneAttachChannel(push_q, ack_q, _FlakyRegistry(set()), is_worker_alive=lambda: False)  # type: ignore[arg-type]
    proc = _ReapFakeProc(dies_on_terminate=True)
    pool = WorkerPool(
        scene_id="s",
        worker_addresses=[],
        owner_backend=None,
        scene_channels=[channel],
        _procs=[proc],
    )

    with pytest.raises(RuntimeError, match="shutdown incomplete after worker reap") as caught:
        pool.shutdown()

    assert "scene stop put boom" in str(caught.value)
    assert not proc.is_alive()
    assert push_q.cancel_join_called and push_q.closed
    assert ack_q.cancel_join_called and ack_q.closed

    # A retry sees the worker already reaped and the channel endpoints already retired. It must not attempt
    # another stop put on the closed/broken queue, so the previously reported pre-reap fault can clear.
    pool.shutdown()


def test_worker_pool_shutdown_fails_loud_on_post_reap_detach_failure() -> None:
    # a post-reap registry-detach failure is surfaced loudly (not suppressed); cleanup is still attempted
    # for every channel/scene before raising.
    ch = _RecordingSceneChannelForPool(failures=[("a", RuntimeError("detach boom"))])
    reg = _RecordingOwnerRegistryForPool()
    proc = _ReapFakeProc(dies_on_terminate=True)
    pool = WorkerPool(
        scene_id="s",
        worker_addresses=[],
        owner_backend=None,
        scene_channels=[ch],  # type: ignore[list-item]
        _procs=[proc],
        _owner_scene_registry=reg,
    )

    with pytest.raises(RuntimeError, match="post-reap cleanup incomplete"):
        pool.shutdown()

    assert ch.reaped and reg.released_all  # cleanup attempted for all before failing loud


def test_worker_pool_shutdown_fails_loud_on_leaked_owner_scene() -> None:
    # an owner scene still loaded after release_all_idle (a refcount/accounting leak) fails shutdown loudly.
    ch = _RecordingSceneChannelForPool()
    reg = _RecordingOwnerRegistryForPool(leaked=["a"])
    proc = _ReapFakeProc(dies_on_terminate=True)
    pool = WorkerPool(
        scene_id="s",
        worker_addresses=[],
        owner_backend=None,
        scene_channels=[ch],  # type: ignore[list-item]
        _procs=[proc],
        _owner_scene_registry=reg,
    )

    with pytest.raises(RuntimeError, match="still-loaded owner scenes"):
        pool.shutdown()


def test_worker_pool_shutdown_runs_deferred_closers_after_reap() -> None:
    # a deferred proxy-channel close runs only after the workers are reaped.
    closed: list[str] = []
    proc = _ReapFakeProc(dies_on_terminate=True)
    pool = WorkerPool(scene_id="s", worker_addresses=[], owner_backend=None, _procs=[proc])
    pool.defer_close(lambda: closed.append("proxy"))

    pool.shutdown()

    assert closed == ["proxy"]  # deferred close ran (post-reap)


def test_worker_pool_shutdown_aggregates_registry_and_closer_failures_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # one post-reap registry fault or closer fault must not skip independent cleanup. Surface all faults,
    # run later closers, discard successful closers, and retain only failed closer handles for diagnosis/retry.
    ch = _RecordingSceneChannelForPool()
    reg = _RecordingOwnerRegistryForPool()
    proc = _ReapFakeProc(dies_on_terminate=True)
    pool = WorkerPool(
        scene_id="s",
        worker_addresses=[],
        owner_backend=None,
        scene_channels=[ch],  # type: ignore[list-item]
        _procs=[proc],
        _owner_scene_registry=reg,
    )
    closed: list[str] = []

    def release_boom() -> list[str]:
        raise RuntimeError("release_all_idle boom")

    def close_boom() -> None:
        raise RuntimeError("deferred close boom")

    monkeypatch.setattr(reg, "release_all_idle", release_boom)
    pool.defer_close(close_boom)
    pool.defer_close(lambda: closed.append("later closer"))

    with pytest.raises(RuntimeError, match="release_all_idle boom") as caught:
        pool.shutdown()

    assert "deferred close boom" in str(caught.value)
    assert ch.reaped and closed == ["later closer"]
    assert pool._deferred_closers == [close_boom]  # noqa: SLF001 - failed handle retained; success removed


def test_worker_pool_shutdown_retains_deferred_closers_on_survivor() -> None:
    # A deferred close must NOT run while a worker survives (it could cut a live
    # handler's forward); the strong reference is retained and the failure surfaced.
    closed: list[str] = []
    proc = _UnreapableFakeProc()
    pool = WorkerPool(scene_id="s", worker_addresses=[], owner_backend=None, _procs=[proc])
    pool.defer_close(lambda: closed.append("proxy"))

    with pytest.raises(RuntimeError, match="retaining scene owner references for safety"):
        pool.shutdown()

    assert closed == []  # retained under a surviving worker


# ===================================================================================================
# Drop-in launch, admission, scene lifecycle, and get_server_config parity tests. Click tests mock the
# blocking launch; lifecycle tests use the real SceneCoordinator over fake worker transports (no GPU /
# no real gRPC servers / no live network).
# ===================================================================================================


# --- Launch-mode and Click option contract ---


def test_serve_grpc_rejects_negative_multiproc_workers() -> None:
    from click.testing import CliRunner

    from nurec.grpc.serve import serve_grpc

    result = CliRunner().invoke(serve_grpc, ["--multiproc-workers=-1"])
    assert result.exit_code == 2  # click.UsageError
    assert "--multiproc-workers must be >= 0" in result.output


@pytest.mark.parametrize(
    "extra_flag",
    [
        ["--multiproc-max-in-flight", "4"],
        ["--multiproc-admission-timeout-s", "1.0"],
        ["--require-mps"],
        ["--multiproc-require-scene-cache"],
        ["--multiproc-bringup-concurrency", "2"],
    ],
)
def test_serve_grpc_explicit_multiproc_only_option_in_single_process_is_usage_error(extra_flag: list[str]) -> None:
    # a multiprocess-only option supplied *explicitly* with --multiproc-workers=0 is a usage error
    # (never silently ignored), so an operator who expected the pool isn't left on the single-process path.
    from click.testing import CliRunner

    from nurec.grpc.serve import serve_grpc

    result = CliRunner().invoke(serve_grpc, ["--multiproc-workers", "0", *extra_flag])
    assert result.exit_code == 2
    assert "require --multiproc-workers > 0" in result.output


@pytest.mark.parametrize(
    "off_flag",
    [
        ["--no-require-mps"],
        ["--multiproc-max-in-flight", "0"],
        ["--no-multiproc-require-scene-cache"],
        ["--multiproc-admission-timeout-s", "0"],
        ["--multiproc-bringup-concurrency", "1"],
    ],
)
def test_serve_grpc_off_form_multiproc_only_option_in_single_process_is_accepted(
    off_flag: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # F006: a semantically-inert OFF/default form of a multiproc-only option still registers as COMMANDLINE via
    # Click's parameter source, but it changes nothing — so with --multiproc-workers=0 it must be ACCEPTED, not
    # hard-failed. Drop-in wrappers that always emit every flag must not break the single-process default.
    # Prove it reaches the single-process build instead of the "require --multiproc-workers > 0" usage error.
    from click.testing import CliRunner

    from nurec.grpc import serve as _serve
    from nurec.grpc.serve import serve_grpc

    class _ReachedSingleProcess(Exception):
        pass

    def _boom(*_a: object, **_k: object) -> None:
        raise _ReachedSingleProcess

    monkeypatch.setattr(_serve.grpc, "server", _boom)

    result = CliRunner().invoke(serve_grpc, ["--multiproc-workers", "0", *off_flag])
    assert isinstance(result.exception, _ReachedSingleProcess)  # OFF form accepted -> reached single-process build
    assert "require --multiproc-workers > 0" not in result.output


def test_serve_grpc_zero_workers_routes_to_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    # --multiproc-workers=0 (the default) must route to the single-process server, NOT the pool. Prove it by
    # short-circuiting the single-process build (grpc.server) and asserting run_drop_in is never called.
    from click.testing import CliRunner

    from nurec.grpc import serve as _serve
    from nurec.grpc.serve import serve_grpc

    called = {"drop_in": False}

    def _drop_in(*_a: object, **_k: object) -> int:
        called["drop_in"] = True
        return 0

    monkeypatch.setattr(_mp, "run_drop_in", _drop_in)

    class _ReachedSingleProcess(Exception):
        pass

    def _boom(*_a: object, **_k: object) -> None:
        raise _ReachedSingleProcess

    monkeypatch.setattr(_serve.grpc, "server", _boom)

    result = CliRunner().invoke(serve_grpc, ["--multiproc-workers", "0"])
    assert isinstance(result.exception, _ReachedSingleProcess)  # reached the single-process build
    assert called["drop_in"] is False  # run_drop_in NOT called in single-process mode


def test_serve_grpc_admission_timeout_without_max_in_flight_is_usage_error() -> None:
    from click.testing import CliRunner

    from nurec.grpc.serve import serve_grpc

    result = CliRunner().invoke(serve_grpc, ["--multiproc-workers", "2", "--multiproc-admission-timeout-s", "1.0"])
    assert result.exit_code == 2
    assert "requires --multiproc-max-in-flight > 0" in result.output


def test_serve_grpc_negative_max_in_flight_is_usage_error() -> None:
    from click.testing import CliRunner

    from nurec.grpc.serve import serve_grpc

    result = CliRunner().invoke(serve_grpc, ["--multiproc-workers", "2", "--multiproc-max-in-flight=-1"])
    assert result.exit_code == 2
    assert "--multiproc-max-in-flight must be >= 0" in result.output


def test_serve_grpc_require_mps_without_mps_is_clean_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # F018: --require-mps on a multi-worker launch without MPS active must surface as ONE clean CLI Error
    # line (exit 2), not a raw RuntimeError traceback from mps_preflight deep inside run_drop_in.
    from click.testing import CliRunner

    from nurec.grpc.serve import serve_grpc

    monkeypatch.setattr(_mp, "mps_active", lambda: False)
    # run_drop_in must never be reached: the pre-validation sheds the launch first.
    monkeypatch.setattr(_mp, "run_drop_in", lambda *a, **k: pytest.fail("run_drop_in should not be reached"))

    result = CliRunner().invoke(serve_grpc, ["--multiproc-workers", "2", "--require-mps"])
    assert result.exit_code == 2  # click.UsageError, not a raw traceback (which would be exit 1)
    assert isinstance(result.exception, SystemExit)  # clean CLI exit, not an uncaught RuntimeError
    assert "MPS not detected" in result.output and "--require-mps set" in result.output


def test_serve_grpc_require_scene_cache_undersized_is_clean_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # F018: --multiproc-require-scene-cache with an undersized --download-cache-size must surface as ONE clean
    # CLI Error line (exit 2), not a raw ValueError traceback from _resolve_download_cache_size.
    from click.testing import CliRunner

    from nurec.grpc.serve import serve_grpc

    monkeypatch.setattr(_mp, "mps_active", lambda: True)  # isolate the download-cache floor from the MPS check
    monkeypatch.setattr(_mp, "run_drop_in", lambda *a, **k: pytest.fail("run_drop_in should not be reached"))

    result = CliRunner().invoke(
        serve_grpc,
        [
            "--multiproc-workers",
            "2",
            "--multiproc-require-scene-cache",
            "--cache-size",
            "5",
            "--download-cache-size",
            "1",
        ],
    )
    assert result.exit_code == 2
    assert isinstance(result.exception, SystemExit)  # clean CLI exit, not an uncaught ValueError
    assert "download_cache_size" in result.output and "--multiproc-require-scene-cache set" in result.output


@pytest.mark.parametrize(
    "extra_flag",
    [
        ["--multiproc-edit-overlays"],
        ["--multiproc-max-edit-overlays", "5"],
    ],
)
def test_serve_grpc_removed_edit_overlay_flags_are_unknown_options(extra_flag: list[str]) -> None:
    # Editing support was removed from multi-process serving, so the two edit-overlay launch flags no longer
    # exist: click rejects them as unknown options (a usage error), never silently ignoring them.
    from click.testing import CliRunner

    from nurec.grpc.serve import serve_grpc

    result = CliRunner().invoke(serve_grpc, ["--multiproc-workers", "2", *extra_flag])
    assert result.exit_code == 2  # click.UsageError
    assert "No such option" in result.output


def test_serve_grpc_forwards_all_multiproc_values_to_run_drop_in(monkeypatch: pytest.MonkeyPatch) -> None:
    # every launch-mode knob is threaded to run_drop_in exactly as supplied (no drops/renames).
    from click.testing import CliRunner

    from nurec.grpc.serve import serve_grpc

    captured: dict[str, object] = {}

    def _capture(cfg: object, **kwargs: object) -> int:
        captured["cfg"] = cfg
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(_mp, "run_drop_in", _capture)
    # Host-independent forwarding test: pretend MPS is up so `--require-mps` passes the serve_grpc
    # pre-validation on a no-MPS CI host.
    monkeypatch.setattr(_mp, "mps_active", lambda: True)

    result = CliRunner().invoke(
        serve_grpc,
        [
            "--multiproc-workers",
            "3",
            "--multiproc-max-in-flight",
            "4",
            "--multiproc-admission-timeout-s",
            "2.5",
            "--multiproc-bringup-concurrency",
            "2",
            "--require-mps",
            "--multiproc-require-scene-cache",
            # --multiproc-require-scene-cache turns the download-cache auto-bump into a hard floor, so the
            # on-disk cache must be >= the resident budget (cache_size, default 10) or the pre-validation sheds.
            "--download-cache-size",
            "10",
        ],
    )
    assert result.exit_code == 0
    assert captured["num_workers"] == 3
    assert captured["max_in_flight"] == 4
    assert captured["admission_timeout_s"] == 2.5
    assert captured["max_concurrent_bringup"] == 2
    assert captured["require_mps"] is True
    assert captured["require_scene_cache"] is True


def test_serve_grpc_single_worker_stays_multiprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    # --multiproc-workers=1 is the real owner+1-worker+proxy topology, NOT single-process (routing it to
    # single-process would silently change edit-handle/admission/discovery/config-reporting semantics).
    from click.testing import CliRunner

    from nurec.grpc.serve import serve_grpc

    captured: dict[str, object] = {}

    def _capture(cfg: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(_mp, "run_drop_in", _capture)

    result = CliRunner().invoke(serve_grpc, ["--multiproc-workers", "1"])
    assert result.exit_code == 0
    assert captured["num_workers"] == 1  # multiprocess path taken for a single worker


def _invoke_serve_grpc_capturing_drop_in(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, args: list[str]
) -> tuple[Any, dict[str, object], list[str]]:
    """Invoke `serve_grpc` with `run_drop_in` stubbed; return (result, forwarded kwargs, one-worker MPS warnings)."""
    from click.testing import CliRunner

    from nurec.grpc.serve import serve_grpc

    captured: dict[str, object] = {}

    def _capture(cfg: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(_mp, "run_drop_in", _capture)
    # These forwarding tests are host-independent: pretend MPS is up so the serve_grpc pre-validation
    # (`validate_multiproc_mps`) does not shed a `--require-mps` multi-worker launch on a no-MPS CI host.
    monkeypatch.setattr(_mp, "mps_active", lambda: True)

    with caplog.at_level(logging.WARNING, logger="nurec.grpc.serve"):
        result = CliRunner().invoke(serve_grpc, args)

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "no effect" in record.getMessage()
    ]
    return result, captured, warnings


def test_serve_grpc_one_worker_require_mps_warns_and_still_forwards_flag(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # --require-mps at one worker has no effect (MPS only helps >1 workers sharing a GPU) and warns;
    # require_mps is still forwarded to run_drop_in unchanged (the warning does not alter the parsed value).
    result, captured, warnings = _invoke_serve_grpc_capturing_drop_in(
        monkeypatch, caplog, ["--multiproc-workers", "1", "--require-mps"]
    )
    assert result.exit_code == 0
    assert captured["num_workers"] == 1  # topology preserved: one-worker multiprocess, not single-process
    assert captured["require_mps"] is True
    assert len(warnings) == 1
    assert "--require-mps" in warnings[0]


def test_serve_grpc_one_worker_no_mps_flags_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Defaults and explicit negative forms at one worker must NOT warn.
    _, _, default_warnings = _invoke_serve_grpc_capturing_drop_in(monkeypatch, caplog, ["--multiproc-workers", "1"])
    assert default_warnings == []

    caplog.clear()
    _, captured, negated_warnings = _invoke_serve_grpc_capturing_drop_in(
        monkeypatch, caplog, ["--multiproc-workers", "1", "--no-require-mps"]
    )
    assert negated_warnings == []
    assert captured["num_workers"] == 1


def test_serve_grpc_multiple_workers_mps_flag_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The warning is one-worker-specific — with >1 workers, --require-mps is meaningful, so no warning.
    result, captured, warnings = _invoke_serve_grpc_capturing_drop_in(
        monkeypatch, caplog, ["--multiproc-workers", "2", "--require-mps"]
    )
    assert result.exit_code == 0
    assert captured["num_workers"] == 2
    assert warnings == []


def test_validate_multiproc_admission_accepts_valid_and_defaults() -> None:
    # The shared contract (Click + run_drop_in both call it): defaults and valid combinations return None.
    assert _mp.validate_multiproc_admission(max_in_flight=0, admission_timeout_s=0.0) is None
    assert _mp.validate_multiproc_admission(max_in_flight=2, admission_timeout_s=1.5) is None


def test_validate_multiproc_admission_reports_first_violation() -> None:
    assert "must be >= 0" in (_mp.validate_multiproc_admission(max_in_flight=-1, admission_timeout_s=0.0) or "")
    assert "finite" in (_mp.validate_multiproc_admission(max_in_flight=1, admission_timeout_s=float("inf")) or "")
    assert "requires --multiproc-max-in-flight > 0" in (
        _mp.validate_multiproc_admission(max_in_flight=0, admission_timeout_s=1.0) or ""
    )


def test_validate_multiproc_harmonizer_accepts_default_config() -> None:
    from nurec.grpc.grpc_server_config import GrpcServerConfig

    # GrpcServerConfig now has enable_harmonizer_temporal (default False): a default config — and an
    # explicit False — must pass the launch guard.
    assert _mp.validate_multiproc_harmonizer(GrpcServerConfig()) is None
    assert _mp.validate_multiproc_harmonizer(GrpcServerConfig(enable_harmonizer_temporal=False)) is None
    # A config-like object without the field (programmatic callers) is treated as disabled, not an error.
    config_without_field = cast(Any, SimpleNamespace())
    assert _mp.validate_multiproc_harmonizer(config_without_field) is None


def test_validate_multiproc_harmonizer_rejects_temporal() -> None:
    config = cast(Any, SimpleNamespace(enable_harmonizer_temporal=True))
    error = _mp.validate_multiproc_harmonizer(config)
    assert error is not None
    assert "--enable-harmonizer-temporal" in error
    assert "--multiproc-workers" in error


def test_validate_multiproc_mps_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    # F018: the shared MPS pre-validation. Only a multi-worker launch that REQUIRES MPS on a host without it
    # is an error; not-required, single-worker, or MPS-active all pass (None).
    monkeypatch.setattr(_mp, "mps_active", lambda: False)
    error = _mp.validate_multiproc_mps(num_workers=2, require_mps=True)
    assert error is not None and "MPS not detected" in error and "--require-mps set" in error
    assert _mp.validate_multiproc_mps(num_workers=2, require_mps=False) is None  # not required -> no error
    assert _mp.validate_multiproc_mps(num_workers=1, require_mps=True) is None  # single worker doesn't need MPS
    monkeypatch.setattr(_mp, "mps_active", lambda: True)
    assert _mp.validate_multiproc_mps(num_workers=8, require_mps=True) is None  # MPS active -> no error


def test_validate_multiproc_download_cache_contract() -> None:
    from nurec.grpc.grpc_server_config import GrpcServerConfig

    # F018: only a hard-floor request (--multiproc-require-scene-cache) with an undersized on-disk cache errors.
    undersized = GrpcServerConfig(cache_size=10, download_cache_size=5)
    error = _mp.validate_multiproc_download_cache(undersized, require_scene_cache=True)
    assert error is not None and "download_cache_size" in error and "--multiproc-require-scene-cache set" in error
    # No hard floor requested -> the default auto-bump path applies, so pre-validation passes.
    assert _mp.validate_multiproc_download_cache(undersized, require_scene_cache=False) is None
    # Adequately sized on-disk cache passes even with the hard floor.
    adequate = GrpcServerConfig(cache_size=5, download_cache_size=5)
    assert _mp.validate_multiproc_download_cache(adequate, require_scene_cache=True) is None


def test_run_drop_in_refuses_temporal_harmonizer_before_spawning() -> None:
    # The programmatic launch path enforces the same contract as the Click layer: an invalid
    # config fails fast with ValueError before any worker/pool state is created.
    config = cast(Any, SimpleNamespace(enable_harmonizer_temporal=True))
    with pytest.raises(ValueError, match="enable-harmonizer-temporal"):
        _mp.run_drop_in(config, num_workers=2)


# --- Admission on lidar and exception paths ---


def test_admission_sheds_lidar_when_over_max_in_flight() -> None:
    # Admission bounds lidar renders too: while one lidar render holds the only slot, a concurrent one sheds.
    holding = threading.Event()
    release = threading.Event()
    calls: list[_CallRecord] = []
    calls_lock = threading.Lock()

    class _BlockingStub(_RecordingStub):
        def render_lidar(self, request: object, timeout: float | None = None) -> tuple:
            holding.set()
            release.wait(timeout=5.0)
            return super().render_lidar(request)

    proxy = RenderProxy([_BlockingStub(0, calls, calls_lock)], max_in_flight=1)
    result: dict[str, object] = {}

    def _first() -> None:
        result["value"] = proxy.render_lidar("r0", _FakeContext())

    holder = threading.Thread(target=_first)
    holder.start()
    assert holding.wait(timeout=2.0)

    with pytest.raises(_AbortError) as exc:
        proxy.render_lidar("r1", _FakeContext())
    assert exc.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED

    release.set()
    holder.join(timeout=2.0)
    assert result["value"][0] == "worker_response"  # type: ignore[index]


def test_admission_releases_permit_after_downstream_rpc_error() -> None:
    # The admission slot is released in a `finally`, so a worker RPC failure never leaks the permit: repeated
    # failing renders keep surfacing the *relayed* worker code (INTERNAL), never a spurious RESOURCE_EXHAUSTED.
    class _FailingStub(_RecordingStub):
        def render_rgb(self, request: object, timeout: float | None = None) -> tuple:
            raise _FakeRpcError(grpc.StatusCode.INTERNAL, "worker boom")

    proxy = RenderProxy([_FailingStub(0, [], threading.Lock())], max_in_flight=1)
    for _ in range(3):
        with pytest.raises(_AbortError) as exc:
            proxy.render_rgb("r", _FakeContext())
        assert exc.value.code == grpc.StatusCode.INTERNAL  # relayed, not a leaked-permit RESOURCE_EXHAUSTED


# --- Startup validation and on-demand scene lifetime across GPU/disk eviction ---


def test_validate_startup_scenes_attaches_all_and_counts() -> None:
    # every startup glob scene is driven through the real all-worker attach barrier; the count
    # of validated scenes is returned (surfaced later as multiproc.validated_startup_scenes).
    builders = [lambda scene_id, artifact_path, ipc_tensors: f"w0::{scene_id}"]
    _osr, coord, _ch, _t, stops = _coordinator_with_workers(builders, {"a": "/art-a", "b": "/art-b"})
    try:
        progress: list[str] = []
        count = _mp._validate_startup_scenes(coord, ["a", "b"], progress=progress.append)  # noqa: SLF001
        assert count == 2
        assert coord.attached_scenes() == {"a", "b"}
        assert len(progress) == 2  # one progress line per validated scene
    finally:
        _stop_workers(stops)


def test_validate_four_startup_scenes_with_two_slot_budget_evicts_and_reattaches() -> None:
    """Validation is sequential and budgeted while discovery retains every registered startup scene."""
    loads: list[str] = []

    def loader(scene_id: str, artifact_path: str) -> object:
        loads.append(scene_id)
        return _tiny_model()

    scene_artifacts = {scene_id: f"/art-{scene_id}" for scene_id in ("a", "b", "c", "d")}
    osr, coord, channels, tables, stops = _coordinator_with_workers(
        _two_worker_builders(), scene_artifacts, loader=loader, max_scenes=2
    )
    try:
        assert _mp._validate_startup_scenes(coord, list(scene_artifacts)) == 4  # noqa: SLF001
        assert loads == ["a", "b", "c", "d"]
        assert coord.attached_scenes() == {"c", "d"}  # bounded to the two most recently validated scenes
        assert set(osr.loaded_scenes()) == {"c", "d"}
        assert all(set(table) == {"c", "d"} for table in tables)
        assert all(channel.attached_scenes() == {"c", "d"} for channel in channels)

        proxy, _calls = _proxy_with_coordinator(coord)
        available = proxy.get_available_scenes(None, _FakeContext())
        assert list(available.scene_ids) == ["a", "b", "c", "d"]  # sorted registration, not residency

        coord.ensure_attached("a")  # permanent startup mapping survives GPU eviction; ID-only reload works
        assert loads == ["a", "b", "c", "d", "a"]
        assert coord.attached_scenes() == {"a", "d"}
        assert set(osr.loaded_scenes()) == {"a", "d"}
        assert all(set(table) == {"a", "d"} for table in tables)
    finally:
        _stop_workers(stops)


def test_validate_startup_scenes_aborts_on_first_failure() -> None:
    # A scene that fails to load aborts validation immediately (raises) — the caller tears the fleet down and
    # exits startup nonzero, so no partially validated proxy serves. Scenes after the failure are not attached.
    def loader(scene_id: str, artifact_path: str) -> object:
        if scene_id == "bad":
            raise RuntimeError("cannot load bad scene")
        return _tiny_model()

    builders = [lambda scene_id, artifact_path, ipc_tensors: f"w0::{scene_id}"]
    _osr, coord, _ch, _t, stops = _coordinator_with_workers(
        builders, {"good": "/art-good", "bad": "/art-bad"}, loader=loader
    )
    try:
        with pytest.raises(SceneLoadError, match="cannot load bad scene") as exc_info:
            _mp._validate_startup_scenes(coord, ["good", "bad"])  # noqa: SLF001
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert coord.attached_scenes() == {"good"}  # only the pre-failure scene attached (no partial serve)
    finally:
        _stop_workers(stops)


def test_validate_startup_scenes_noop_without_coordinator() -> None:
    # A single-scene pool has no coordinator; validation is a no-op returning 0.
    assert _mp._validate_startup_scenes(None, ["a", "b"]) == 0  # noqa: SLF001


def test_on_demand_scene_registration_survives_gpu_eviction_and_id_reattaches() -> None:
    # An on-demand scene stays ID-only reloadable after GPU eviction — it re-attaches by id with
    # no re-registration (no client header), because its artifact registration survives GPU eviction.
    builders = [lambda scene_id, artifact_path, ipc_tensors: f"w0::{scene_id}"]
    _osr, coord, _ch, _t, stops = _coordinator_with_workers(builders, {}, max_scenes=1)
    try:
        coord.register_scene("od", "/art-od", source="file:///art-od")
        coord.ensure_attached("od")
        assert coord.attached_scenes() == {"od"}

        coord.register_scene("other", "/art-other", source="file:///art-other")
        coord.ensure_attached("other")  # evicts "od" (max_scenes=1)
        assert coord.attached_scenes() == {"other"}
        assert "od" in coord.known_scenes()  # registration survives GPU eviction

        coord.ensure_attached("od")  # re-attaches by id, no re-registration -> evicts "other"
        assert coord.attached_scenes() == {"od"}
    finally:
        _stop_workers(stops)


def test_local_file_registrar_registration_survives_gpu_eviction_and_id_reattaches(tmp_path: Path) -> None:
    """Exercise the real file:// resolution/registration path, not only direct coordinator registration."""
    first_artifact = tmp_path / "first.usdz"
    second_artifact = tmp_path / "second.usdz"
    first_artifact.write_bytes(b"first")
    second_artifact.write_bytes(b"second")

    loads: list[tuple[str, str]] = []

    def loader(scene_id: str, artifact_path: str) -> object:
        loads.append((scene_id, artifact_path))
        return _tiny_model()

    _osr, coord, _channels, _tables, stops = _coordinator_with_workers(
        _two_worker_builders(), {}, loader=loader, max_scenes=1
    )
    try:
        registrar = OnDemandSceneRegistrar(coord)
        registrar.register_scene_from_uri("first", first_artifact.as_uri())
        coord.ensure_attached("first")

        registrar.register_scene_from_uri("second", second_artifact.as_uri())
        coord.ensure_attached("second")  # evicts first from GPU residency, not registration
        assert coord.attached_scenes() == {"second"}
        assert coord.known_scenes() == {"first", "second"}

        coord.ensure_attached("first")  # ID-only: no second registrar call/header
        assert coord.attached_scenes() == {"first"}
        assert [scene_id for scene_id, _path in loads] == ["first", "second", "first"]
        assert loads[0][1] == str(first_artifact.resolve())
        assert loads[2][1] == str(first_artifact.resolve())
    finally:
        _stop_workers(stops)


def test_coordinator_rejects_conflicting_source_on_reregister() -> None:
    # Re-registering a known scene from a *different* source URI is rejected; the same source
    # (or same artifact path) is an idempotent no-op. check_source_conflict is the header-path guard.
    _osr, coord, _ch, _t, stops = _coordinator_no_workers()
    try:
        coord.register_scene("s", "/art-s", source="http://a/s")
        coord.register_scene("s", "/art-s", source="http://a/s")  # same source + path -> no-op
        with pytest.raises(ValueError, match="different source"):
            coord.register_scene("s", "/art-s", source="http://b/s")
        coord.check_source_conflict("s", "http://a/s")  # same source -> ok
        with pytest.raises(ValueError, match="different source"):
            coord.check_source_conflict("s", "http://b/s")
    finally:
        _stop_workers(stops)


def test_on_demand_registrar_rejects_conflicting_source(tmp_path) -> None:
    # The registrar rejects a conflicting-source re-registration WITHOUT re-downloading (a scene it already
    # knows is not re-resolved); the interceptor maps the ValueError to INVALID_ARGUMENT.
    scene_cache = SceneCache(cache_dir=tmp_path / "cache", max_size=5)

    class _FakeDownloader:
        def __init__(self) -> None:
            self.calls = 0

        def download(self, url: str, output_path) -> int:  # noqa: ANN001
            self.calls += 1
            Path(output_path).write_bytes(b"scene-bytes")
            return 11

    downloader = _FakeDownloader()
    _osr, coord, _ch, _t, stops = _coordinator_no_workers()
    try:
        registrar = OnDemandSceneRegistrar(coord, scene_cache=scene_cache, downloader=downloader)  # type: ignore[arg-type]
        registrar.register_scene_from_uri("s", "https://a.example/s.usdz")
        assert downloader.calls == 1
        with pytest.raises(ValueError, match="different source"):
            registrar.register_scene_from_uri("s", "https://b.example/s.usdz")
        assert downloader.calls == 1  # known scene -> not re-downloaded on the conflicting attempt
    finally:
        _stop_workers(stops)


def test_on_demand_registrar_only_accepts_exact_startup_artifact_source(tmp_path) -> None:
    startup_artifact = tmp_path / "startup.usdz"
    other_artifact = tmp_path / "other.usdz"
    startup_artifact.write_bytes(b"startup")
    other_artifact.write_bytes(b"other")
    _osr, coord, _ch, _t, stops = _coordinator_no_workers({"s": str(startup_artifact)})
    try:
        registrar = OnDemandSceneRegistrar(coord)
        registrar.register_scene_from_uri("s", startup_artifact.as_uri())  # exact local source is idempotent
        with pytest.raises(ValueError, match="different source"):
            registrar.register_scene_from_uri("s", other_artifact.as_uri())
        with pytest.raises(ValueError, match="different source"):
            registrar.register_scene_from_uri("s", "https://example.test/s.usdz")
    finally:
        _stop_workers(stops)


def test_on_demand_registrar_local_reregister_accepts_equivalent_spelling(tmp_path) -> None:
    # F005: an on-demand LOCAL re-registration (registered_source is not None) with an equivalent-but-
    # differently-spelled URI resolves to the same artifact and must be a no-op, mirroring single-process
    # `_register_local_scene` and the startup/glob branch — NOT a raw-string source conflict. A genuinely
    # different local file still conflicts.
    scene_dir = tmp_path / "scenes"
    scene_dir.mkdir()
    artifact = scene_dir / "scene.usdz"
    artifact.write_bytes(b"fake-artifact")
    other = scene_dir / "other.usdz"
    other.write_bytes(b"other")
    _osr, coord, _ch, _t, stops = _coordinator_no_workers()
    try:
        registrar = OnDemandSceneRegistrar(coord)  # local file:// needs no cache/downloader
        # First on-demand registration records a client source (registered_source is not None).
        registrar.register_scene_from_uri("dl", artifact.as_uri())
        assert coord.is_known("dl")

        # Same file, a `..` round-trip through the (existing) scenes dir -> resolves equal -> no-op (no raise),
        # even though the raw source strings differ (which check_source_conflict would have rejected).
        equivalent = (scene_dir / ".." / "scenes" / "scene.usdz").as_uri()
        assert equivalent != artifact.as_uri()  # a genuinely different spelling
        registrar.register_scene_from_uri("dl", equivalent)  # must NOT raise
        assert coord.is_known("dl")

        # A genuinely different local file is still a conflicting re-registration.
        with pytest.raises(ValueError):
            registrar.register_scene_from_uri("dl", other.as_uri())
    finally:
        _stop_workers(stops)


def test_multiproc_startup_glob_matches_recursive_and_usdz_parity(tmp_path) -> None:
    # F004: multiproc startup discovery now goes through `Artifact.discover_from_glob` (recursive `**` matching
    # + `.usdz` suffix assertion), exactly like single-process serve — so a `**` glob serving nested scenes
    # with plain serve serves the same set here. This exercises the exact discovery expression `run_drop_in`
    # uses and contrasts it with the old non-recursive `glob.glob`, and asserts the `.usdz` fail-loud (Trap 3).
    import glob as _glob

    from nurec.artifact.artifact import Artifact

    (tmp_path / "top.usdz").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.usdz").write_bytes(b"")
    recursive_glob = str(tmp_path / "**" / "*.usdz")

    # The exact expression `run_drop_in` uses for `matches`: recursive discovery -> both files.
    matches = sorted(str(a.source) for a in Artifact.discover_from_glob(recursive_glob))
    assert matches == sorted([str(tmp_path / "top.usdz"), str(tmp_path / "sub" / "nested.usdz")])

    # The old non-recursive glob.glob silently dropped the top-level file (the regression the fix closes).
    assert str(tmp_path / "top.usdz") not in _glob.glob(recursive_glob)

    # Fail-loud (Trap 3): a non-`.usdz` glob is rejected up front rather than silently serving other files.
    with pytest.raises(AssertionError, match="usdz"):
        Artifact.discover_from_glob(str(tmp_path / "*.txt"))


def test_on_demand_registrar_concurrent_sources_never_mislabel_cached_bytes(tmp_path) -> None:
    scene_cache = SceneCache(cache_dir=tmp_path / "cache", max_size=2)

    class _SourceDownloader:
        def download(self, url: str, output_path) -> int:  # noqa: ANN001
            payload = url.encode()
            Path(output_path).write_bytes(payload)
            return len(payload)

    _osr, coord, _ch, _t, stops = _coordinator_no_workers()
    try:
        registrar = OnDemandSceneRegistrar(coord, scene_cache=scene_cache, downloader=_SourceDownloader())  # type: ignore[arg-type]
        start = threading.Barrier(3)
        outcomes: list[Exception | None] = []

        def _register(url: str) -> None:
            start.wait()
            try:
                registrar.register_scene_from_uri("s", url)
            except Exception as exc:  # noqa: BLE001 - record the losing source-conflict outcome
                outcomes.append(exc)
            else:
                outcomes.append(None)

        urls = ("https://a.example/s.usdz", "https://b.example/s.usdz")
        threads = [threading.Thread(target=_register, args=(url,)) for url in urls]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()

        assert len(outcomes) == 2
        assert sum(outcome is None for outcome in outcomes) == 1
        assert sum(isinstance(outcome, ValueError) for outcome in outcomes) == 1
        registration = coord.registration("s")
        assert registration is not None
        artifact_path, source, generation = registration
        assert generation is not None
        assert source in urls
        assert Path(artifact_path).read_text() == source
        assert scene_cache.is_current("s", generation, source)
    finally:
        _stop_workers(stops)


def test_eviction_after_resolution_before_registration_cannot_publish_stale_path(tmp_path, monkeypatch) -> None:
    scene_cache = SceneCache(cache_dir=tmp_path / "cache", max_size=1)

    class _Downloader:
        def download(self, url: str, output_path) -> int:  # noqa: ANN001
            Path(output_path).write_text(url)
            return len(url)

    _osr, coord, _ch, _t, stops = _coordinator_no_workers()
    original_register = coord.register_scene
    resolved = threading.Event()
    publish = threading.Event()

    def _block_s1_publish(scene_id: str, artifact_path: str, **kwargs) -> None:  # noqa: ANN003
        if scene_id == "s1":
            resolved.set()
            assert publish.wait(timeout=2)
        original_register(scene_id, artifact_path, **kwargs)

    monkeypatch.setattr(coord, "register_scene", _block_s1_publish)
    try:
        scene_cache.set_on_evict(coord.handle_cache_eviction)
        registrar = OnDemandSceneRegistrar(coord, scene_cache=scene_cache, downloader=_Downloader())  # type: ignore[arg-type]
        errors: list[Exception] = []

        def _register_s1() -> None:
            try:
                registrar.register_scene_from_uri("s1", "https://ex/s1.usdz")
            except Exception as exc:  # noqa: BLE001 - this raced request is expected to retry
                errors.append(exc)

        thread = threading.Thread(target=_register_s1)
        thread.start()
        assert resolved.wait(timeout=2)
        registrar.register_scene_from_uri("s2", "https://ex/s2.usdz")  # evicts s1 before it publishes
        publish.set()
        thread.join(timeout=2)
        assert not thread.is_alive()

        assert len(errors) == 1 and "evicted while registering" in str(errors[0])
        assert not coord.is_known("s1")
        with pytest.raises(SceneNotKnownError):
            coord.ensure_attached("s1")
    finally:
        publish.set()
        _stop_workers(stops)


def test_delayed_old_generation_callback_cannot_invalidate_same_source_redownload(tmp_path) -> None:
    scene_cache = SceneCache(cache_dir=tmp_path / "cache", max_size=1)

    class _Downloader:
        def download(self, url: str, output_path) -> int:  # noqa: ANN001
            Path(output_path).write_text(url)
            return len(url)

    _osr, coord, _ch, _t, stops = _coordinator_no_workers()
    callback_started = threading.Event()
    release_callback = threading.Event()
    first_generation: list[int] = []

    def _delayed_callback(eviction: SceneCacheEviction) -> None:
        if eviction.scene_id == "s" and eviction.generation == first_generation[0]:
            callback_started.set()
            assert release_callback.wait(timeout=2)
        coord.handle_cache_eviction(eviction)

    try:
        scene_cache.set_on_evict(_delayed_callback)
        registrar = OnDemandSceneRegistrar(coord, scene_cache=scene_cache, downloader=_Downloader())  # type: ignore[arg-type]
        source = "https://ex/s.usdz"
        registrar.register_scene_from_uri("s", source)
        registration = coord.registration("s")
        assert registration is not None and registration[2] is not None
        first_generation.append(registration[2])

        other_temp = tmp_path / "other.tmp"
        other_temp.write_bytes(b"other")
        eviction_thread = threading.Thread(target=lambda: scene_cache.add_scene("other", other_temp))
        eviction_thread.start()
        assert callback_started.wait(timeout=2)  # s is already absent from cache; its callback is delayed

        registrar.register_scene_from_uri("s", source)  # same valid header reconciles + redownloads
        renewed = coord.registration("s")
        assert renewed is not None and renewed[2] is not None and renewed[2] > first_generation[0]
        release_callback.set()
        eviction_thread.join(timeout=2)
        assert not eviction_thread.is_alive()

        assert coord.registration("s") == renewed  # old callback was compare-and-delete and became a no-op
        assert scene_cache.is_current("s", renewed[2], source)
    finally:
        release_callback.set()
        _stop_workers(stops)


def test_disk_cache_eviction_invalidates_remote_registration_and_redownloads_once(tmp_path) -> None:
    # Wiring the disk cache's on_evict -> coordinator.invalidate_scene (as run_drop_in does)
    # means evicting a remote scene's cached file drops its reload mapping. An ID-only request then returns
    # NOT_FOUND (never UNKNOWN from a vanished file); repeating the x-nre-scene-url header re-downloads once.
    scene_cache = SceneCache(cache_dir=tmp_path / "cache", max_size=1)  # one-entry on-disk cache

    class _FakeDownloader:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def download(self, url: str, output_path) -> int:  # noqa: ANN001
            self.calls.append(url)
            Path(output_path).write_bytes(b"scene-bytes")
            return 5

    downloader = _FakeDownloader()
    _osr, coord, _ch, _t, stops = _coordinator_no_workers()
    try:
        scene_cache.set_on_evict(coord.handle_cache_eviction)  # mirror run_drop_in's wiring
        registrar = OnDemandSceneRegistrar(coord, scene_cache=scene_cache, downloader=downloader)  # type: ignore[arg-type]

        registrar.register_scene_from_uri("s1", "https://ex/s1.usdz")
        registrar.register_scene_from_uri("s2", "https://ex/s2.usdz")  # evicts s1's disk file -> invalidates s1

        assert not coord.is_known("s1")  # registration dropped by disk-cache eviction
        assert "s1" not in coord.known_scenes()
        assert coord.is_known("s2")
        with pytest.raises(SceneNotKnownError):
            coord.ensure_attached("s1")  # ID-only access -> NOT_FOUND

        registrar.register_scene_from_uri("s1", "https://ex/s1.usdz")  # repeat header -> re-register + redownload
        assert coord.is_known("s1")
        assert downloader.calls == ["https://ex/s1.usdz", "https://ex/s2.usdz", "https://ex/s1.usdz"]  # once each
    finally:
        _stop_workers(stops)


def test_disk_cache_eviction_defers_invalidation_until_scene_leaves_residency() -> None:
    # a disk-cache eviction that names a currently-resident scene must NOT drop it out from under an
    # in-flight render — the invalidation is deferred and applied once the scene leaves residency (evicted).
    builders = [lambda scene_id, artifact_path, ipc_tensors: f"w0::{scene_id}"]
    _osr, coord, _ch, _t, stops = _coordinator_with_workers(builders, {}, max_scenes=1)
    try:
        coord.register_scene("od", "/art-od", source="http://ex/od")
        coord.ensure_attached("od")  # resident
        coord.invalidate_scene("od")  # deferred (still resident)
        assert coord.is_known("od")  # not dropped while resident

        coord.register_scene("other", "/art-other", source="http://ex/other")
        coord.ensure_attached("other")  # evicts "od" -> applies the deferred invalidation
        assert not coord.is_known("od")  # dropped once it left residency
        with pytest.raises(SceneNotKnownError):
            coord.ensure_attached("od")
    finally:
        _stop_workers(stops)


def test_disk_invalidation_before_slot_reservation_waits_for_active_attach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-reservation attach window is active too; invalidation cannot drop its reload mapping."""
    osr, coord, _ch, _t, stops = _coordinator_with_workers([], {}, max_scenes=1)
    entered_reserve = threading.Event()
    continue_reserve = threading.Event()
    original_reserve = coord._reserve_slot  # noqa: SLF001 - deterministic concurrency seam

    def blocked_reserve(scene_id: str) -> None:
        entered_reserve.set()
        assert continue_reserve.wait(timeout=2)
        original_reserve(scene_id)

    monkeypatch.setattr(coord, "_reserve_slot", blocked_reserve)
    errors: list[BaseException] = []
    try:
        coord.register_scene("s", "/art-s", source="https://ex/s.usdz", generation=1)

        def attach() -> None:
            try:
                coord.ensure_attached("s")
            except BaseException as exc:  # noqa: BLE001 - captured for the main-thread assertion
                errors.append(exc)

        thread = threading.Thread(target=attach)
        thread.start()
        assert entered_reserve.wait(timeout=2)

        coord.invalidate_scene("s", 1)
        assert coord.registration("s") is not None  # deferred while attach_active, before `_reserved`

        continue_reserve.set()
        thread.join(timeout=2)
        assert not thread.is_alive() and errors == []
        assert coord.attached_scenes() == {"s"}

        coord.register_scene("other", "/art-other")
        coord.ensure_attached("other")  # evicts s and consumes its deferred disk invalidation
        assert coord.registration("s") is None
        assert not osr.is_loaded("s")
    finally:
        continue_reserve.set()
        _stop_workers(stops)


def test_same_source_header_renews_pending_generation_while_scene_is_resident(tmp_path) -> None:
    scene_cache = SceneCache(cache_dir=tmp_path / "cache", max_size=1)

    class _Downloader:
        def download(self, url: str, output_path) -> int:  # noqa: ANN001
            Path(output_path).write_text(url)
            return len(url)

    builders = [lambda scene_id, artifact_path, ipc_tensors: f"w0::{scene_id}"]
    _osr, coord, _ch, _t, stops = _coordinator_with_workers(builders, {}, max_scenes=1)
    try:
        scene_cache.set_on_evict(coord.handle_cache_eviction)
        registrar = OnDemandSceneRegistrar(coord, scene_cache=scene_cache, downloader=_Downloader())  # type: ignore[arg-type]
        source = "https://ex/s.usdz"
        registrar.register_scene_from_uri("s", source)
        coord.ensure_attached("s")
        first = coord.registration("s")
        assert first is not None and first[2] is not None

        orphan = tmp_path / "orphan.tmp"
        orphan.write_bytes(b"orphan")
        scene_cache.add_scene("orphan", orphan)  # gen1 is gone; invalidation is pending while s is resident
        registrar.register_scene_from_uri("s", source)  # redownload and atomically renew the reload mapping
        renewed = coord.registration("s")
        assert renewed is not None and renewed[2] is not None and renewed[2] > first[2]

        coord.register_scene("other", "/art-other")
        coord.ensure_attached("other")  # GPU-evicts s; stale pending gen1 must not remove renewed gen2
        assert coord.registration("s") == renewed
        coord.ensure_attached("s")  # ID-only reload from gen2 still works
        assert coord.attached_scenes() == {"s"}
    finally:
        _stop_workers(stops)


def test_disk_eviction_during_failed_attach_is_consumed_and_header_redownloads(tmp_path) -> None:
    scene_cache = SceneCache(cache_dir=tmp_path / "cache", max_size=1)
    load_started = threading.Event()
    finish_load = threading.Event()

    class _Downloader:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def download(self, url: str, output_path) -> int:  # noqa: ANN001
            self.calls.append(url)
            Path(output_path).write_text(url)
            return len(url)

    def _failing_loader(_scene_id: str, _artifact_path: str):
        load_started.set()
        assert finish_load.wait(timeout=2)
        raise RuntimeError("bad artifact")

    downloader = _Downloader()
    _osr, coord, _ch, _t, stops = _coordinator_no_workers(loader=_failing_loader)
    try:
        scene_cache.set_on_evict(coord.handle_cache_eviction)
        registrar = OnDemandSceneRegistrar(coord, scene_cache=scene_cache, downloader=downloader)  # type: ignore[arg-type]
        source = "https://ex/s.usdz"
        registrar.register_scene_from_uri("s", source)
        attach_errors: list[Exception] = []

        def _attach() -> None:
            try:
                coord.ensure_attached("s")
            except Exception as exc:  # noqa: BLE001 - expected typed/load failure under test
                attach_errors.append(exc)

        attach_thread = threading.Thread(target=_attach)
        attach_thread.start()
        assert load_started.wait(timeout=2)

        registrar.register_scene_from_uri("other", "https://ex/other.usdz")  # s eviction defers while reserved
        assert coord.is_known("s")
        finish_load.set()
        attach_thread.join(timeout=2)
        assert not attach_thread.is_alive() and len(attach_errors) == 1
        assert not coord.is_known("s")  # pending invalidation consumed on failed-attach quiescence

        registrar.register_scene_from_uri("s", source)
        assert coord.is_known("s")
        assert downloader.calls.count(source) == 2
    finally:
        finish_load.set()
        _stop_workers(stops)


def test_cache_invalidation_callback_failure_degrades_and_surfaces(tmp_path, monkeypatch) -> None:
    scene_cache = SceneCache(cache_dir=tmp_path / "cache", max_size=1)
    _osr, coord, _ch, _t, stops = _coordinator_no_workers()

    def _broken_invalidation(_scene_id: str, _generation: int | None = None) -> None:
        raise RuntimeError("registry fault")

    monkeypatch.setattr(coord, "invalidate_scene", _broken_invalidation)
    try:
        scene_cache.set_on_evict(coord.handle_cache_eviction)
        first = tmp_path / "first.tmp"
        second = tmp_path / "second.tmp"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        scene_cache.add_scene("first", first)
        with pytest.raises(SceneCacheEvictionError, match="registry fault"):
            scene_cache.add_scene("second", second)
        assert coord.is_degraded()
    finally:
        _stop_workers(stops)


def test_disk_cache_eviction_never_invalidates_permanent_scene() -> None:
    # A glob/bootstrap (permanent) scene keeps its registration regardless of the disk cache — a disk eviction
    # that happens to name it is a no-op, so it stays re-attachable by id with no header.
    builders = [lambda scene_id, artifact_path, ipc_tensors: f"w0::{scene_id}"]
    _osr, coord, _ch, _t, stops = _coordinator_with_workers(builders, {"perm": "/art-perm"}, max_scenes=2)
    try:
        coord.invalidate_scene("perm")  # permanent -> ignored
        assert coord.is_known("perm")
    finally:
        _stop_workers(stops)


# --- Effective get_server_config map ---


def test_multiproc_server_config_map_reports_effective_values() -> None:
    # the effective config map drops the single-process-only `max_workers`, marks serving.mode, and
    # surfaces the authoritative multiproc.* values (worker count, effective renderer, caps, validation).
    from nurec.grpc.grpc_server_config import GrpcServerConfig

    cfg = GrpcServerConfig(cache_size=3, allowed_image_formats="JPEG")
    config_map = _mp._multiproc_server_config_map(  # noqa: SLF001
        cfg,
        num_workers=4,
        renderer_name="gsplat",
        max_concurrent_bringup=2,
        max_in_flight=8,
        admission_timeout_s=1.5,
        startup_scenes=5,
        validated_startup_scenes=5,
        download_cache_size=10,
        tracks_timestamp_ray_stride=16,
        mps_is_active=False,
    )
    assert "max_workers" not in config_map  # single-process worker-thread count dropped
    # Popped before GrpcServerConfig (like single-process serve), surfaced with the same unprefixed key.
    assert config_map["tracks_timestamp_ray_stride"] == "16"
    assert config_map["serving.mode"] == "multiprocess"
    assert config_map["allowed_image_formats"] == "JPEG"
    assert config_map["multiproc.workers"] == "4"
    assert config_map["multiproc.renderer"] == "gsplat"
    assert config_map["multiproc.mps_active"] == "false"
    assert config_map["multiproc.bringup_concurrency"] == "2"
    assert config_map["multiproc.max_scenes"] == "3"
    assert config_map["multiproc.startup_scenes"] == "5"
    assert config_map["multiproc.validated_startup_scenes"] == "5"
    assert config_map["multiproc.download_cache_size"] == "10"
    assert config_map["multiproc.max_in_flight"] == "8"
    assert config_map["multiproc.admission_timeout_s"] == "1.5"
    assert "multiproc.max_edit_overlays" not in config_map  # editing removed from multiproc serving
    assert "multiproc.edit_overlays" not in config_map
    assert not any(key.startswith("editing.") for key in config_map)


@pytest.mark.parametrize("num_workers", [1, 4])
def test_public_multiproc_get_server_config_reports_exact_effective_map(num_workers: int) -> None:
    """The one-worker topology and a multi-worker pool expose the same authoritative public schema."""
    from nurec.grpc.grpc_server_config import GrpcServerConfig

    config_map = _mp._multiproc_server_config_map(  # noqa: SLF001
        GrpcServerConfig(cache_size=2),
        num_workers=num_workers,
        renderer_name="gsplat",
        max_concurrent_bringup=1,
        max_in_flight=3,
        admission_timeout_s=0.25,
        startup_scenes=4,
        validated_startup_scenes=4,
        download_cache_size=10,
        mps_is_active=False,
    )
    proxy = RenderProxy([_RecordingStub(0, [], threading.Lock())], server_config_map=config_map)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_SensorsimServiceServicer_to_server(proxy, server)
    port = server.add_insecure_port("localhost:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        response = SensorsimServiceStub(channel).get_server_config(Empty(), timeout=5)
        assert dict(response.server_config) == config_map
        assert response.server_config["serving.mode"] == "multiprocess"
        assert response.server_config["multiproc.workers"] == str(num_workers)
        assert "max_workers" not in response.server_config
    finally:
        channel.close()
        server.stop(0).wait(timeout=5)


# ---------------------------------------------------------------------------------------------------
# Zero-scene multiprocess startup: an empty/absent artifact_glob is a valid on-demand deployment
# (parity with single-process serve). The bootstrap scene is OPTIONAL: workers come up scene-less, the
# proxy is SERVING, get_available_scenes is empty, and the first x-nre-scene-url registration
# loads/attaches the first scene on demand.
# ---------------------------------------------------------------------------------------------------


class _ZeroSceneFakeService:
    """A GPU-free `SensorSimService` stand-in for the zero-scene worker-startup path: records that it was
    constructed and exposes the `_metrics_manager.shutdown()` the worker's finally-block flushes."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.ctor_args = args
        self.ctor_kwargs = kwargs
        self._metrics_manager = SimpleNamespace(shutdown=lambda: None)


def test_worker_zero_scene_startup_binds_without_attach(monkeypatch: pytest.MonkeyPatch) -> None:
    # The REAL `_worker` started with scene_id=None sends NO bootstrap attach request and never waits on
    # an attach response, yet still builds its service + scene-attach listener, binds a real gRPC port, and
    # signals ("ready", ...). No GPU: the service + serve loop are stubbed; only the port bind is real.
    monkeypatch.setattr(_mp, "_owner_parent_signals", lambda: (object(), lambda: True))  # supervision present
    monkeypatch.setattr(_mp, "SensorSimService", _ZeroSceneFakeService)
    monkeypatch.setattr(_mp, "add_SensorsimServiceServicer_to_server", lambda *_a, **_k: None)

    def _stop_immediately(server: Any, *_a: object, **_k: object) -> Any:
        server.stop(None)  # tear down the real gRPC server the worker bound
        return WorkerStopReason.STOP

    monkeypatch.setattr(_mp, "_serve_until_stop", _stop_immediately)

    attach_req_q: queue.Queue = queue.Queue()
    attach_resp_q: queue.Queue = queue.Queue()
    port_q: queue.Queue = queue.Queue()
    scene_q: queue.Queue = queue.Queue()

    _mp._worker(  # noqa: SLF001
        worker_id=3,
        scene_id=None,  # zero-scene startup: no bootstrap
        artifact_path=None,
        config=WorkerSpawnConfig(renderer_name="gsplat", enable_editing_actors=False),
        attach_req_q=attach_req_q,
        attach_resp_q=attach_resp_q,
        port_q=port_q,
        lifecycle_rx=None,
        scene_q=scene_q,
        scene_ack_q=queue.Queue(),
    )

    assert attach_req_q.empty()  # never requested a bootstrap attach
    assert attach_resp_q.empty()  # and never consumed an attach response
    kind, worker_id, port = port_q.get_nowait()
    assert (kind, worker_id) == ("ready", 3) and port > 0  # bound a real port + signaled ready
    assert port_q.empty()  # exactly one "ready" message
    scene_q.put((_SCENE_STOP,))  # let the daemon scene-attach listener exit


def test_start_worker_pool_zero_scene_builds_scene_less_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    # start_worker_pool(None) builds a pool with NO bootstrap — no eager owner load, an empty attached
    # set, no registered scenes, and scene_id=None — so it is ready to attach the first scene on demand.
    monkeypatch.setattr(_mp.mp, "get_context", lambda _method: object())

    spawned_scene_ids: list = []
    awaited_scene_ids: list = []

    def _spawn(worker_id: int, **kwargs: Any) -> _WorkerHandle:
        spawned_scene_ids.append(kwargs["scene_id"])
        proc = _FakeProc(alive=True)
        kwargs["on_started"](proc, _RecordingLifecycleTx())
        channel = SceneAttachChannel(queue.Queue(), queue.Queue(), kwargs["registry"], is_worker_alive=proc.is_alive)
        return _WorkerHandle(
            worker_id=worker_id,
            proc=proc,
            attach_req_q=queue.Queue(),
            attach_resp_q=queue.Queue(),
            port_q=queue.Queue(),
            channel=channel,
            lifecycle_tx=_RecordingLifecycleTx(),
        )

    def _ready(handle: _WorkerHandle, registry: Any, scene_id: Any, cancel_event: Any, progress: Any) -> str:
        awaited_scene_ids.append(scene_id)
        return f"127.0.0.1:{5000 + handle.worker_id}"

    monkeypatch.setattr(_mp, "_spawn_worker", _spawn)
    monkeypatch.setattr(_mp, "_await_worker_ready", _ready)

    pool = _mp.start_worker_pool(
        None,  # no bootstrap artifact
        num_workers=2,
        renderer_name="gsplat",
        enable_editing_actors=False,
    )

    assert pool.scene_id is None  # zero-scene pool
    assert spawned_scene_ids == [None, None]  # every worker spawned scene-less
    assert awaited_scene_ids == [None, None]  # no bootstrap attach was serviced for any worker
    assert pool.worker_addresses == ["127.0.0.1:5000", "127.0.0.1:5001"]
    assert pool.coordinator is not None
    assert pool.coordinator.attached_scenes() == set()  # nothing attached at startup
    assert pool.coordinator.known_scenes() == set()  # no registered scenes
    assert pool._owner_scene_registry.loaded_scenes() == ()  # noqa: SLF001 - no eager owner load


def test_run_drop_in_zero_scene_empty_glob_serves_and_reports_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    # An empty/absent artifact_glob is a valid zero-startup-scene launch — run_drop_in does NOT raise,
    # reports 0 startup + 0 validated scenes in the effective config, and (even with test_scenes_are_valid)
    # exits 0 on a requested stop. Adding --multiproc-workers to an on-demand deployment no longer loops.
    from nurec.grpc.grpc_server_config import GrpcServerConfig

    fake_pool = _StubPool()
    captured: dict = {}
    monkeypatch.setattr(_mp, "mps_preflight", lambda *a, **k: None)
    monkeypatch.setattr(_mp, "_resolve_download_cache_size", lambda *a, **k: 1)
    monkeypatch.setattr(_mp, "SceneCache", lambda *a, **k: object())
    monkeypatch.setattr(_mp, "Downloader", lambda *a, **k: object())
    monkeypatch.setattr(_mp, "start_worker_pool", lambda *a, **k: fake_pool)
    monkeypatch.setattr(_mp.SensorSimService, "build_server_config_map", staticmethod(lambda cfg: {}))
    monkeypatch.setattr(sys, "stdout", _ReconfigurableStringIO())

    def _capture_serve(*_a: object, **kwargs: object) -> PoolTerminalReason:
        captured.update(kwargs)
        return PoolTerminalReason.REQUESTED

    monkeypatch.setattr(_mp, "serve_proxy", _capture_serve)

    # No artifact_glob (default None) => matches == [] => the real _plan_glob_scenes returns no bootstrap.
    cfg = GrpcServerConfig(test_scenes_are_valid=True)
    assert _mp.run_drop_in(cfg, num_workers=2) == 0  # no raise; requested stop exits 0
    config_map = captured["server_config_map"]
    assert config_map["serving.mode"] == "multiprocess"
    assert config_map["multiproc.startup_scenes"] == "0"  # zero eager scenes
    assert config_map["multiproc.validated_startup_scenes"] == "0"  # nothing to validate -> clean no-op
    assert fake_pool.shutdown_calls == 1  # fleet always torn down


def test_multiproc_server_config_map_reports_zero_startup_scenes() -> None:
    # The effective config map surfaces 0 startup + 0 validated scenes for a zero-scene launch.
    from nurec.grpc.grpc_server_config import GrpcServerConfig

    config_map = _mp._multiproc_server_config_map(  # noqa: SLF001
        GrpcServerConfig(),
        num_workers=2,
        renderer_name="gsplat",
        max_concurrent_bringup=1,
        max_in_flight=0,
        admission_timeout_s=0.0,
        startup_scenes=0,
        validated_startup_scenes=0,
        download_cache_size=10,
        mps_is_active=False,
    )
    assert config_map["serving.mode"] == "multiprocess"
    assert config_map["multiproc.startup_scenes"] == "0"
    assert config_map["multiproc.validated_startup_scenes"] == "0"


def test_zero_scene_first_registration_then_render_routes(tmp_path: Path) -> None:
    # From a zero-scene start (empty coordinator), the first x-nre-scene-url registration makes the
    # scene known; a render then drives the first owner-load + all-worker attach barrier and routes normally.
    artifact = tmp_path / "first.usdz"
    artifact.write_bytes(b"fake-artifact")
    loads: list[str] = []

    def loader(scene_id: str, artifact_path: str) -> object:
        loads.append(scene_id)
        return _tiny_model()

    osr, coord, _channels, tables, stops = _coordinator_with_workers(
        _two_worker_builders(), {}, attached_scenes=(), loader=loader
    )
    try:
        assert coord.known_scenes() == set()  # zero-scene start: nothing known

        registrar = OnDemandSceneRegistrar(coord)  # a local file:// needs no cache/downloader
        registrar.register_scene_from_uri("first", f"file://{artifact}")
        assert coord.is_known("first")  # registered, not yet attached
        assert coord.attached_scenes() == set()

        proxy, calls = _proxy_with_coordinator(coord)
        response = proxy.render_rgb(sensorsim_pb2.RGBRenderRequest(scene_id="first"), _FakeContext())

        assert response[0] == "worker_response"
        assert [call.method for call in calls] == ["render_rgb"]  # routed to a worker after the barrier
        assert loads == ["first"]  # owner loaded exactly once on the first render
        assert tables[0] == {"first": "w0::first"} and tables[1] == {"first": "w1::first"}  # attached on all
        assert osr.registry.attached_count("first") == 2  # one refcount per worker
    finally:
        _stop_workers(stops)


def test_zero_scene_concurrent_first_requests_load_once_and_attach_all() -> None:
    # Concurrent first requests for the same on-demand scene trigger exactly ONE owner load and ONE
    # all-worker attach barrier (the coordinator's per-scene single-flight), from a zero-attached start.
    loads: list[str] = []
    load_lock = threading.Lock()

    def loader(scene_id: str, artifact_path: str) -> object:
        with load_lock:
            loads.append(scene_id)
        return _tiny_model()

    osr, coord, _channels, tables, stops = _coordinator_with_workers(
        _two_worker_builders(), {"od": "/art-od"}, attached_scenes=(), loader=loader
    )
    try:
        barrier = threading.Barrier(6, timeout=5.0)
        errors: list = []

        def _attach() -> None:
            try:
                barrier.wait()  # release all six at once to contend on the first attach
                coord.ensure_attached("od")
            except Exception as exc:  # noqa: BLE001 - surface any thread failure to the assertion below
                errors.append(exc)

        workers = [threading.Thread(target=_attach) for _ in range(6)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5.0)

        assert errors == []
        assert loads == ["od"]  # loaded exactly once despite six concurrent first-attaches
        assert coord.attached_scenes() == {"od"}
        assert tables[0] == {"od": "w0::od"} and tables[1] == {"od": "w1::od"}
        assert osr.registry.attached_count("od") == 2
    finally:
        _stop_workers(stops)


def test_zero_scene_invalid_first_registration_is_typed_and_retryable(tmp_path: Path) -> None:
    # A bad first x-nre-scene-url registration fails typed (a ValueError the SceneDownloadInterceptor
    # maps to INVALID_ARGUMENT) WITHOUT degrading the scene control plane; a subsequent valid registration
    # then succeeds and attaches — the first-scene registration is retryable.
    _osr, coord, _channels, tables, stops = _coordinator_with_workers(_two_worker_builders(), {}, attached_scenes=())
    try:
        registrar = OnDemandSceneRegistrar(coord)  # no cache/downloader -> a remote URL cannot resolve

        with pytest.raises(ValueError, match="requires a scene_cache"):
            registrar.register_scene_from_uri("first", "https://example.com/scene.usdz")
        assert not coord.is_known("first")  # nothing registered on the failed attempt
        assert not coord.is_degraded()  # a client-input error must not degrade the pool

        artifact = tmp_path / "first.usdz"
        artifact.write_bytes(b"fake-artifact")
        registrar.register_scene_from_uri("first", f"file://{artifact}")  # retry with a valid local artifact
        assert coord.is_known("first")
        coord.ensure_attached("first")
        assert coord.attached_scenes() == {"first"}
        assert tables[0] == {"first": "w0::first"}
    finally:
        _stop_workers(stops)


def test_zero_scene_discovery_empty_and_unknown_render_not_found() -> None:
    # Empty-state RPC contract: get_available_scenes is [] and any scene-using render is NOT_FOUND until
    # a scene is registered (the lease's SceneNotKnownError -> NOT_FOUND), never routed to a worker.
    _osr, coord, _channels, _tables, stops = _coordinator_with_workers(_two_worker_builders(), {}, attached_scenes=())
    try:
        proxy, calls = _proxy_with_coordinator(coord)
        assert list(proxy.get_available_scenes(None, _FakeContext()).scene_ids) == []  # discovery is empty

        with pytest.raises(_AbortError) as exc:
            proxy.render_rgb(sensorsim_pb2.RGBRenderRequest(scene_id="nope"), _FakeContext())
        assert exc.value.code == grpc.StatusCode.NOT_FOUND
        assert calls == []  # never routed to a worker
    finally:
        _stop_workers(stops)


def test_worker_pool_shutdown_zero_scene_fleet_is_clean() -> None:
    # A zero-scene fleet (workers up, no scene ever attached) shuts down cleanly — every worker is
    # reaped, no owner scene is leaked, and the lifecycle + scene endpoints are released.
    osr = OwnerSceneRegistry()  # nothing loaded (zero-scene)
    channel = SceneAttachChannel(queue.Queue(), queue.Queue(), osr.registry, is_worker_alive=lambda: False)
    proc = _ReapFakeProc(dies_on_terminate=True)
    lifecycle_tx = _RecordingLifecycleTx()
    pool = WorkerPool(
        scene_id=None,
        worker_addresses=["127.0.0.1:5000"],
        owner_backend=None,
        coordinator=None,
        scene_channels=[channel],
        _procs=[proc],
        _owner_scene_registry=osr,
        _lifecycle_txs=[lifecycle_tx],
    )

    pool.shutdown()  # must not raise: an empty fleet has nothing to leak

    assert not proc.is_alive()  # reaped
    assert osr.loaded_scenes() == ()  # no leaked owner scene
    assert channel.attached_scenes() == set()  # no residual attachment
    assert lifecycle_tx.closed  # lifecycle endpoint released
